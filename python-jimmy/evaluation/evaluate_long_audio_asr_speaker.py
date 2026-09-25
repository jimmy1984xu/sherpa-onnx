#!/usr/bin/env python3
"""Evaluate long-audio ASR and speaker results without modifying pipeline output."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from asr_segment_detail_diff import (
    LabelLine as AgentSdkLabelLine,
    build_segment_detail_rows as _build_agent_sdk_segment_detail_rows,
    write_segment_detail_workbook as _write_agent_sdk_segment_detail_workbook,
)


_SEGMENT_ID_RE = re.compile(r"^(?P<file_id>.+)_(?P<start_ms>\d+)_(?P<duration_ms>\d+)$")
_TYPED_LABEL_RE = re.compile(
    r"^(?P<segment_id>\S+)\s+\((?P<speaker_id>[^()]*)\)(?:\((?P<segment_type>单人|短插话|重叠|听不清)\))?(?:\s+(?P<text>.*))?$"
)
SEGMENT_TYPE_SINGLE = "单人"
SEGMENT_TYPE_SHORT = "短插话"
SEGMENT_TYPE_OVERLAP = "重叠"
SEGMENT_TYPE_UNCLEAR = "听不清"
SEGMENT_TYPES = frozenset(
    {SEGMENT_TYPE_SINGLE, SEGMENT_TYPE_SHORT, SEGMENT_TYPE_OVERLAP, SEGMENT_TYPE_UNCLEAR}
)
BUSINESS_WER_NAME = "wer"


@dataclass(frozen=True)
class TimedSegment:
    file_id: str
    segment_id: str
    start_ms: int
    duration_ms: int
    speaker_id: str
    asr_text: str
    segment_type: str = SEGMENT_TYPE_SINGLE

    @property
    def end_ms(self) -> int:
        return self.start_ms + self.duration_ms


@dataclass(frozen=True)
class ResultRecord:
    path: Path
    audio_name: str
    file_id: str
    segments: list[TimedSegment]


@dataclass(frozen=True)
class LabelRecord:
    path: Path
    file_id: str
    segments: list[TimedSegment]


def parse_segment_id(segment_id: str) -> tuple[str, int, int]:
    match = _SEGMENT_ID_RE.fullmatch(segment_id)
    if match is None:
        raise ValueError(
            f"segment_id must end with _<start_ms>_<duration_ms>: {segment_id}"
        )
    file_id = match.group("file_id")
    start_ms = int(match.group("start_ms"))
    duration_ms = int(match.group("duration_ms"))
    if not file_id or duration_ms <= 0:
        raise ValueError(f"invalid segment_id: {segment_id}")
    return file_id, start_ms, duration_ms


def _required_string(payload: dict, field: str, context: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str):
        raise ValueError(f"{context}: missing or non-string {field}")
    return value


def _required_int(payload: dict, field: str, context: str) -> int:
    value = payload.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{context}: missing or non-integer {field}")
    return value


def _parse_result_segment(payload: object, result_path: Path, index: int) -> TimedSegment:
    context = f"{result_path} segments[{index}]"
    if not isinstance(payload, dict):
        raise ValueError(f"{context}: segment must be an object")
    segment_id = _required_string(payload, "segment_id", context)
    file_id, id_start_ms, id_duration_ms = parse_segment_id(segment_id)
    duration_ms = _required_int(payload, "duration_ms", context)
    if duration_ms <= 0:
        raise ValueError(f"{context}: duration_ms must be positive")
    if duration_ms != id_duration_ms:
        raise ValueError(
            f"{context}: duration_ms={duration_ms} disagrees with segment_id duration={id_duration_ms}"
        )
    explicit_start = payload.get("start_ms")
    if explicit_start is not None:
        if not isinstance(explicit_start, int) or isinstance(explicit_start, bool):
            raise ValueError(f"{context}: start_ms must be an integer when supplied")
        if explicit_start != id_start_ms:
            raise ValueError(
                f"{context}: start_ms={explicit_start} disagrees with segment_id start={id_start_ms}"
            )
    return TimedSegment(
        file_id=file_id,
        segment_id=segment_id,
        start_ms=id_start_ms,
        duration_ms=duration_ms,
        speaker_id=_required_string(payload, "speaker_id", context),
        asr_text=_required_string(payload, "asr_text", context),
    )


def parse_result_json(path: Path) -> ResultRecord:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: root must be an object")
    audio_name = _required_string(payload, "audio_name", str(path))
    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list):
        raise ValueError(f"{path}: missing or non-array segments")
    segments = [_parse_result_segment(item, path, index) for index, item in enumerate(raw_segments)]
    if not segments:
        raise ValueError(f"{path}: segments must not be empty")
    file_ids = {segment.file_id for segment in segments}
    if len(file_ids) != 1:
        raise ValueError(f"{path}: segments contain multiple file IDs: {sorted(file_ids)}")
    return ResultRecord(
        path=path,
        audio_name=audio_name,
        file_id=segments[0].file_id,
        segments=sorted(segments, key=lambda item: (item.start_ms, item.end_ms, item.segment_id)),
    )


def discover_result_records(results_dir: Path) -> list[ResultRecord]:
    if not results_dir.is_dir():
        raise ValueError(f"results directory does not exist: {results_dir}")
    result_paths = sorted(path for path in results_dir.rglob("result.json") if path.is_file())
    if not result_paths:
        raise ValueError(f"no result.json found under: {results_dir}")
    records = [parse_result_json(path) for path in result_paths]
    duplicate_file_ids = sorted(
        file_id
        for file_id in {record.file_id for record in records}
        if sum(record.file_id == file_id for record in records) > 1
    )
    if duplicate_file_ids:
        raise ValueError(
            "multiple result.json files represent the same audio; evaluate one run at a time: "
            + ", ".join(duplicate_file_ids)
        )
    return sorted(records, key=lambda item: (item.file_id, str(item.path)))


def _validate_typed_label(path: Path, line_number: int, speaker_id: str, segment_type: str) -> None:
    if segment_type not in SEGMENT_TYPES:
        raise ValueError(f"{path}:{line_number}: unsupported segment type: {segment_type}")
    if segment_type == SEGMENT_TYPE_OVERLAP and speaker_id != "MULTI":
        raise ValueError(f"{path}:{line_number}: 重叠 segment_type requires speaker_id MULTI")
    if segment_type == SEGMENT_TYPE_UNCLEAR and speaker_id != "UNCLEAR":
        raise ValueError(f"{path}:{line_number}: 听不清 segment_type requires speaker_id UNCLEAR")
    if segment_type in {SEGMENT_TYPE_SINGLE, SEGMENT_TYPE_SHORT} and speaker_id in {"MULTI", "UNCLEAR"}:
        raise ValueError(f"{path}:{line_number}: {segment_type} segment_type cannot use reserved speaker_id {speaker_id}")


def parse_label_file(path: Path) -> LabelRecord:
    segments: list[TimedSegment] = []
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            line = raw_line.strip()
            if not line:
                continue
            match = _TYPED_LABEL_RE.fullmatch(line)
            if match is None:
                raise ValueError(
                    f"{path}:{line_number}: expected '<segment_id> (speaker_id)(segment_type) text'"
                )
            segment_id = match.group("segment_id")
            speaker_id = match.group("speaker_id").strip()
            explicit_type = match.group("segment_type")
            segment_type = explicit_type or SEGMENT_TYPE_SINGLE
            if explicit_type is not None:
                _validate_typed_label(path, line_number, speaker_id, segment_type)
            elif speaker_id == "MULTI":
                segment_type = SEGMENT_TYPE_OVERLAP
            elif speaker_id == "UNCLEAR":
                segment_type = SEGMENT_TYPE_UNCLEAR
            file_id, start_ms, duration_ms = parse_segment_id(segment_id)
            segments.append(
                TimedSegment(
                    file_id=file_id,
                    segment_id=segment_id,
                    start_ms=start_ms,
                    duration_ms=duration_ms,
                    speaker_id=speaker_id,
                    asr_text=match.group("text") or "",
                    segment_type=segment_type,
                )
            )
    if not segments:
        raise ValueError(f"{path}: label has no usable rows")
    file_ids = {segment.file_id for segment in segments}
    if len(file_ids) != 1:
        raise ValueError(f"{path}: label contains multiple file IDs: {sorted(file_ids)}")
    return LabelRecord(
        path=path,
        file_id=segments[0].file_id,
        segments=sorted(segments, key=lambda item: (item.start_ms, item.end_ms, item.segment_id)),
    )


def _find_label_paths(labels_dir: Path, file_id: str) -> list[Path]:
    expected = f"{file_id}_label.txt"
    return sorted(path for path in labels_dir.rglob(expected) if path.is_file())


def resolve_labels(
    records: list[ResultRecord],
    labels_dir: Optional[Path],
    explicit_label: Optional[Path],
) -> dict[str, LabelRecord]:
    if explicit_label is not None:
        if len(records) != 1:
            raise ValueError("--label can only be used when exactly one result audio is discovered")
        if not explicit_label.is_file():
            raise ValueError(f"explicit label does not exist: {explicit_label}")
        label = parse_label_file(explicit_label)
        if label.file_id != records[0].file_id:
            raise ValueError(
                f"explicit label file_id={label.file_id} does not match result file_id={records[0].file_id}"
            )
        return {label.file_id: label}
    if labels_dir is None:
        return {}
    if not labels_dir.is_dir():
        raise ValueError(f"labels directory does not exist: {labels_dir}")
    labels: dict[str, LabelRecord] = {}
    for record in records:
        paths = _find_label_paths(labels_dir, record.file_id)
        if len(paths) > 1:
            joined = ", ".join(str(path) for path in paths)
            raise ValueError(f"multiple labels for {record.file_id}: {joined}")
        if len(paths) == 1:
            labels[record.file_id] = parse_label_file(paths[0])
    return labels


def _format_machine_line(segment: TimedSegment) -> str:
    prefix = f"{segment.segment_id} {segment.speaker_id}"
    return f"{prefix} {segment.asr_text}\n" if segment.asr_text else f"{prefix}\n"


def build_business_wer_inputs(
    records: list[ResultRecord], labels: dict[str, LabelRecord]
) -> tuple[list[str], list[str]]:
    """Build the single business WER input.

    ``听不清`` is a business-level silence target: its reference text is
    intentionally empty, while the full pipeline hypothesis is retained so
    any emitted text is scored as an insertion. ``单人``, ``短插话`` and
    ``重叠`` keep their reference text.
    """
    reference_lines: list[str] = []
    hypothesis_lines: list[str] = []
    for record in sorted(records, key=lambda item: item.file_id):
        label = labels.get(record.file_id)
        if label is None:
            continue
        reference_text = "".join(
            "" if segment.segment_type == SEGMENT_TYPE_UNCLEAR else segment.asr_text
            for segment in label.segments
        )
        hypothesis_text = "".join(segment.asr_text for segment in record.segments)
        reference_lines.append(f"{record.file_id} {reference_text}".rstrip())
        hypothesis_lines.append(f"{record.file_id} {hypothesis_text}".rstrip())
    return reference_lines, hypothesis_lines


def build_whole_audio_wer_inputs(
    records: list[ResultRecord], labels: dict[str, LabelRecord]
) -> tuple[list[str], list[str]]:
    """Compatibility wrapper for the v1.0 business WER input."""
    return build_business_wer_inputs(records, labels)


def _agent_sdk_file_id(references: list[TimedSegment], predictions: list[TimedSegment]) -> str:
    for segment in [*references, *predictions]:
        return segment.file_id
    raise ValueError("Agent SDK segment detail requires at least one reference or prediction segment")


def build_agent_sdk_segment_detail_rows(
    references: list[TimedSegment], predictions: list[TimedSegment], language: str
):
    """Adapt pipeline result/label segments to Agent SDK's six-column detail implementation."""
    file_id = _agent_sdk_file_id(references, predictions)
    labels = [AgentSdkLabelLine(segment.segment_id, segment.asr_text) for segment in references]
    hypotheses = [
        {
            "segmentId": segment.segment_id,
            "offsetMs": segment.start_ms,
            "durationMs": segment.duration_ms,
            "text": segment.asr_text,
        }
        for segment in predictions
    ]
    return _build_agent_sdk_segment_detail_rows(labels, hypotheses, language, file_id)


def write_agent_sdk_segment_detail_workbook(path: Path, rows: Iterable[object]) -> None:
    """Write the standard Agent SDK six-column XLSX without openpyxl."""
    _write_agent_sdk_segment_detail_workbook(path, rows)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def meeting_output_dir(output_dir: Path, file_id: str) -> Path:
    """Return the fixed artifact directory for one evaluated meeting."""
    if not file_id or file_id in {".", ".."} or Path(file_id).name != file_id:
        raise ValueError(f"unsafe meeting file_id for output directory: {file_id}")
    return output_dir / file_id


def write_meeting_asr_txt(record: ResultRecord, meeting_dir: Path) -> Path:
    """Write the public, fixed-name ASR projection for one meeting."""
    path = meeting_dir / "asr.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(_format_machine_line(segment) for segment in record.segments),
        encoding="utf-8",
    )
    return path


def _write_label_file(label: LabelRecord, path: Path) -> None:
    _write_lines(
        path,
        [
            f"{segment.segment_id} ({segment.speaker_id})({segment.segment_type}) {segment.asr_text}".rstrip()
            for segment in label.segments
        ],
    )


def run_subprocess_task(command: list[str]) -> tuple[bool, str]:
    """Run an evaluator without leaving stdout/stderr files in the output tree."""
    try:
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError as error:
        return False, str(error)
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        if not detail:
            detail = f"command exited with {completed.returncode}"
        return False, detail
    return True, ""


def summarize_wer_detail(path: Path) -> dict:
    per_file: list[dict] = []
    if not path.is_file():
        raise ValueError(f"WER detail file was not created: {path}")
    with path.open(encoding="utf-8-sig") as stream:
        for raw_line in stream:
            line = raw_line.rstrip("\n")
            if not line or line.startswith("id\twer\t"):
                continue
            columns = line.split("\t", 6)
            if len(columns) < 6:
                continue
            try:
                reference_tokens = int(columns[2])
                errors = int(columns[3])
                deletions = int(columns[4])
                insertions = int(columns[5])
                wer_percent = float(columns[1])
            except ValueError as error:
                raise ValueError(f"invalid WER detail row: {line}") from error
            per_file.append(
                {
                    "file_id": columns[0],
                    "wer_percent": wer_percent,
                    "reference_tokens": reference_tokens,
                    "errors": errors,
                    "deletions": deletions,
                    "insertions": insertions,
                    "substitutions": errors - deletions - insertions,
                }
            )
    reference_tokens = sum(item["reference_tokens"] for item in per_file)
    errors = sum(item["errors"] for item in per_file)
    deletions = sum(item["deletions"] for item in per_file)
    insertions = sum(item["insertions"] for item in per_file)
    return {
        "wer_percent": None if reference_tokens == 0 else round(100.0 * errors / reference_tokens, 4),
        "reference_tokens": reference_tokens,
        "errors": errors,
        "deletions": deletions,
        "insertions": insertions,
        "substitutions": errors - deletions - insertions,
        "files": per_file,
    }


def _read_json_if_present(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return payload if isinstance(payload, dict) else None


def _meeting_artifact_names(meeting_dir: Path) -> list[str]:
    return sorted(path.name for path in meeting_dir.iterdir() if path.is_file())


def _run_meeting_evaluation(
    record: ResultRecord,
    label: Optional[LabelRecord],
    output_dir: Path,
    language: str,
    boundary_tolerance_ms: int,
    collar_ms: int,
) -> dict:
    meeting_dir = meeting_output_dir(output_dir, record.file_id)
    asr_path = write_meeting_asr_txt(record, meeting_dir)
    outcome: dict[str, object] = {
        "file_id": record.file_id,
        "result_path": str(record.path),
        "label_path": None if label is None else str(label.path),
        "meeting_dir": meeting_dir,
        "status": "success",
        "errors": [],
        "asr_path": asr_path,
        "wer_summary": None,
        "wer_summaries": None,
        "speaker_summary": None,
    }
    if label is None:
        outcome["status"] = "skipped_no_label"
        return outcome

    with tempfile.TemporaryDirectory(prefix="long-audio-evaluation-") as temporary_directory:
        temporary_root = Path(temporary_directory)
        wer_label_lines, wer_hyp_lines = build_business_wer_inputs(
            [record], {record.file_id: label}
        )
        wer_label_path = temporary_root / "wer_label.txt"
        wer_hyp_path = temporary_root / "wer_hyp.txt"
        wer_detail_path = meeting_dir / "wer_detail.txt"
        _write_lines(wer_label_path, wer_label_lines)
        _write_lines(wer_hyp_path, wer_hyp_lines)
        wer_ok, wer_error = run_subprocess_task(
            [
                __import__("sys").executable,
                str(Path(__file__).with_name("evaluation.py")),
                "--label", str(wer_label_path),
                "--hyp", str(wer_hyp_path),
                "--language", language,
                "--detail", str(wer_detail_path),
            ]
        )
        if not wer_ok:
            outcome["errors"].append(f"WER: {wer_error}")
        else:
            try:
                wer_summary = summarize_wer_detail(wer_detail_path)
                _write_json(meeting_dir / "wer_summary.json", wer_summary)
                _write_json(meeting_dir / "wer_metrics.json", {BUSINESS_WER_NAME: wer_summary})
                outcome["wer_summary"] = wer_summary
            except (OSError, ValueError, json.JSONDecodeError) as error:
                outcome["errors"].append(f"WER: {error}")

        try:
            segment_rows = build_agent_sdk_segment_detail_rows(
                label.segments, record.segments, language
            )
            write_agent_sdk_segment_detail_workbook(
                meeting_dir / "segment_asr_detail.xlsx", segment_rows
            )
        except (OSError, ValueError) as error:
            outcome["errors"].append(f"segment ASR detail: {error}")

        speaker_results_dir = temporary_root / "speaker-results"
        speaker_labels_dir = temporary_root / "speaker-labels"
        _write_lines(
            speaker_results_dir / f"{record.file_id}_asr.txt",
            [
                _format_machine_line(segment).rstrip("\n")
                for segment in record.segments
            ],
        )
        _write_label_file(label, speaker_labels_dir / f"{record.file_id}_label.txt")
        speaker_output_dir = temporary_root / "speaker-output"
        speaker_ok, speaker_error = run_subprocess_task(
            [
                __import__("sys").executable,
                str(Path(__file__).with_name("speaker_diarization_metrics.py")),
                "--results-dir", str(speaker_results_dir),
                "--labels-dir", str(speaker_labels_dir),
                "--output-dir", str(speaker_output_dir),
                "--boundary-tolerance-ms", str(boundary_tolerance_ms),
                "--collar-ms", str(collar_ms),
            ]
        )
        if speaker_ok:
            source_summary = speaker_output_dir / "speaker_diarization_summary.json"
            source_boundary = speaker_output_dir / "speaker_diarization_boundary_details.csv"
            try:
                speaker_summary = _read_json_if_present(source_summary)
                if speaker_summary is None or not source_boundary.is_file():
                    raise ValueError("speaker evaluator did not generate its required outputs")
                parameters = speaker_summary.get("parameters")
                if isinstance(parameters, dict):
                    parameters["results_dir"] = str(record.path)
                    parameters["labels_dir"] = str(label.path)
                    parameters["output_dir"] = str(meeting_dir)
                _write_json(meeting_dir / "speaker_summary.json", speaker_summary)
                shutil.copy2(source_boundary, meeting_dir / "speaker_diarization_boundary_details.csv")
                outcome["speaker_summary"] = speaker_summary
            except (OSError, ValueError, json.JSONDecodeError) as error:
                outcome["errors"].append(f"speaker: {error}")
        else:
            outcome["errors"].append(f"speaker: {speaker_error}")

    if outcome["errors"]:
        outcome["status"] = "failed"
    return outcome


def write_report(
    path: Path,
    results_dir: Path,
    labels_dir: Optional[Path],
    outcomes: list[dict],
) -> None:
    success_count = sum(item["status"] == "success" for item in outcomes)
    skipped_count = sum(item["status"] == "skipped_no_label" for item in outcomes)
    failed_count = sum(item["status"] == "failed" for item in outcomes)
    lines = [
        "# Long-audio ASR and speaker evaluation report",
        "",
        "## Summary",
        "",
        f"- Results directory: `{results_dir}`",
        f"- Labels directory: `{labels_dir or ''}`",
        f"- Meetings: {len(outcomes)}",
        f"- Success: {success_count}",
        f"- Skipped (no label): {skipped_count}",
        f"- Failed: {failed_count}",
    ]
    for outcome in outcomes:
        meeting_dir = outcome["meeting_dir"]
        relative_dir = meeting_dir.name
        lines.extend(
            [
                "",
                f"## {outcome['file_id']}",
                "",
                f"- Status: {outcome['status']}",
                f"- Source result: `{outcome['result_path']}`",
                f"- Label: `{outcome['label_path'] or ''}`",
                f"- Output: `{relative_dir}/`",
                "- Artifacts:",
            ]
        )
        for artifact_name in _meeting_artifact_names(meeting_dir):
            lines.append(f"  - `{relative_dir}/{artifact_name}`")
        wer_summary = outcome.get("wer_summary")
        if isinstance(wer_summary, dict):
            lines.append(f"- WER: {wer_summary.get('wer_percent')}")
        speaker_summary = outcome.get("speaker_summary")
        metrics = speaker_summary.get("metrics") if isinstance(speaker_summary, dict) else None
        if isinstance(metrics, dict):
            lines.append(f"- DER: {metrics.get('der')}")
        errors = outcome.get("errors")
        if errors:
            lines.append("- Errors:")
            for error in errors:
                lines.append(f"  - {error}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True, help="pipeline run directory or recursive root")
    parser.add_argument("--labels-dir", type=Path, help="directory containing *_label.txt files")
    parser.add_argument("--label", type=Path, help="explicit label for one result audio")
    parser.add_argument("--output-dir", type=Path, required=True, help="independent evaluation output directory")
    parser.add_argument("--language", default="ZH", help="WER language, default: ZH")
    parser.add_argument("--boundary-tolerance-ms", type=int, default=500)
    parser.add_argument("--collar-ms", type=int, default=500)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.boundary_tolerance_ms < 0 or args.collar_ms < 0:
        raise ValueError("boundary tolerance and collar must not be negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = discover_result_records(args.results_dir)
    labels = resolve_labels(records, args.labels_dir, args.label)
    outcomes = [
        _run_meeting_evaluation(
            record,
            labels.get(record.file_id),
            args.output_dir,
            args.language,
            args.boundary_tolerance_ms,
            args.collar_ms,
        )
        for record in records
    ]
    report_path = args.output_dir / "evaluation_report.md"
    write_report(report_path, args.results_dir, args.labels_dir, outcomes)
    print(f"evaluated {len(records)} meeting result(s) into {args.output_dir}")
    print(f"matched {len(labels)} label file(s)")
    print(f"report: {report_path}")
    return 1 if any(outcome["status"] == "failed" for outcome in outcomes) else 0


if __name__ == "__main__":
    raise SystemExit(main())
