#!/usr/bin/env python3
"""Evaluate long-audio ASR and speaker results without modifying pipeline output."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from agent_sdk_segment_detail import (
    LabelLine as AgentSdkLabelLine,
    build_segment_detail_rows as _build_agent_sdk_segment_detail_rows,
    write_segment_detail_workbook as _write_agent_sdk_segment_detail_workbook,
)


_SEGMENT_ID_RE = re.compile(r"^(?P<file_id>.+)_(?P<start_ms>\d+)_(?P<duration_ms>\d+)$")


@dataclass(frozen=True)
class TimedSegment:
    file_id: str
    segment_id: str
    start_ms: int
    duration_ms: int
    speaker_id: str
    asr_text: str

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


def parse_label_file(path: Path) -> LabelRecord:
    segments: list[TimedSegment] = []
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            line = raw_line.strip()
            if not line:
                continue
            columns = line.split(maxsplit=2)
            if len(columns) < 2:
                raise ValueError(f"{path}:{line_number}: expected segment_id and speaker")
            segment_id, speaker_id = columns[:2]
            file_id, start_ms, duration_ms = parse_segment_id(segment_id)
            text = columns[2] if len(columns) > 2 else ""
            segments.append(
                TimedSegment(
                    file_id=file_id,
                    segment_id=segment_id,
                    start_ms=start_ms,
                    duration_ms=duration_ms,
                    speaker_id=speaker_id.strip().strip("()"),
                    asr_text=text,
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


def _iter_segments(records: Iterable[ResultRecord]) -> Iterable[TimedSegment]:
    for record in sorted(records, key=lambda item: item.file_id):
        yield from record.segments


def _format_machine_line(segment: TimedSegment) -> str:
    prefix = f"{segment.segment_id} {segment.speaker_id}"
    return f"{prefix} {segment.asr_text}\n" if segment.asr_text else f"{prefix}\n"


def write_public_result_txt(records: list[ResultRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(_format_machine_line(segment) for segment in _iter_segments(records)), encoding="utf-8")


def write_internal_asr_files(records: list[ResultRecord], inputs_dir: Path) -> dict[str, Path]:
    inputs_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for record in records:
        path = inputs_dir / f"{record.file_id}_asr.txt"
        path.write_text("".join(_format_machine_line(segment) for segment in record.segments), encoding="utf-8")
        paths[record.file_id] = path
    return paths



def build_whole_audio_wer_inputs(
    records: list[ResultRecord], labels: dict[str, LabelRecord]
) -> tuple[list[str], list[str]]:
    label_lines: list[str] = []
    hypothesis_lines: list[str] = []
    for record in sorted(records, key=lambda item: item.file_id):
        label = labels.get(record.file_id)
        if label is None:
            continue
        reference_text = "".join(segment.asr_text for segment in label.segments)
        hypothesis_text = "".join(segment.asr_text for segment in record.segments)
        label_lines.append(f"{record.file_id} {reference_text}".rstrip())
        hypothesis_lines.append(f"{record.file_id} {hypothesis_text}".rstrip())
    return label_lines, hypothesis_lines


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


def write_internal_label_files(
    labels: dict[str, LabelRecord], labels_dir: Path
) -> dict[str, Path]:
    labels_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for file_id, label in sorted(labels.items()):
        path = labels_dir / f"{file_id}_label.txt"
        lines = []
        for segment in label.segments:
            prefix = f"{segment.segment_id} {segment.speaker_id}"
            lines.append(f"{prefix} {segment.asr_text}" if segment.asr_text else prefix)
        _write_lines(path, lines)
        paths[file_id] = path
    return paths


def _task(status: str, *, error: Optional[str] = None, artifacts: Optional[list[Path]] = None,
          command: Optional[list[str]] = None, stdout_log: Optional[Path] = None,
          stderr_log: Optional[Path] = None, return_code: Optional[int] = None) -> dict:
    payload: dict[str, object] = {"status": status}
    if error:
        payload["error"] = error
    if artifacts:
        payload["artifacts"] = [str(path) for path in artifacts]
    if command:
        payload["command"] = command
    if stdout_log:
        payload["stdout_log"] = str(stdout_log)
    if stderr_log:
        payload["stderr_log"] = str(stderr_log)
    if return_code is not None:
        payload["return_code"] = return_code
    return payload


def run_subprocess_task(
    command: list[str], stdout_path: Path, stderr_path: Path
) -> dict:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import subprocess

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
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text(f"{error}\n", encoding="utf-8")
        return _task(
            "failed",
            error=str(error),
            command=command,
            stdout_log=stdout_path,
            stderr_log=stderr_path,
        )
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        return _task(
            "failed",
            error=f"command exited with {completed.returncode}",
            command=command,
            stdout_log=stdout_path,
            stderr_log=stderr_path,
            return_code=completed.returncode,
        )
    return _task(
        "success",
        command=command,
        stdout_log=stdout_path,
        stderr_log=stderr_path,
        return_code=completed.returncode,
    )


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


def write_report(
    path: Path,
    manifest: dict,
    status: dict,
    wer_summary: Optional[dict],
    speaker_summary: Optional[dict],
    workbook_paths: list[Path],
) -> None:
    lines = [
        "# Long-audio ASR and speaker evaluation report",
        "",
        "## Inputs",
        "",
        f"- Results directory: `{manifest['parameters']['results_dir']}`",
        f"- Labels directory: `{manifest['parameters'].get('labels_dir') or ''}`",
        f"- Output directory: `{manifest['parameters']['output_dir']}`",
        f"- Result audio files: {len(manifest['inputs']['result_files'])}",
        f"- Matched label files: {len(manifest['inputs']['matched_labels'])}",
        "",
        "## Task status",
        "",
        "| Task | Status | Note |",
        "| --- | --- | --- |",
    ]
    for task_name, task_payload in status["tasks"].items():
        lines.append(
            f"| {task_name} | {task_payload['status']} | {task_payload.get('error', '')} |"
        )
    lines.extend(["", "## ASR", ""])
    if wer_summary is None:
        lines.append("WER was not generated.")
    else:
        lines.extend(
            [
                f"- Whole-audio WER: {wer_summary['wer_percent']}",
                f"- Reference tokens: {wer_summary['reference_tokens']}",
                f"- Errors: {wer_summary['errors']} (S={wer_summary['substitutions']}, D={wer_summary['deletions']}, I={wer_summary['insertions']})",
            ]
        )
    lines.extend(["", "## Speaker", ""])
    speaker_metrics = None if speaker_summary is None else speaker_summary.get("metrics")
    if not isinstance(speaker_metrics, dict):
        lines.append("Speaker metrics were not generated.")
    else:
        lines.extend(
            [
                f"- DER: {speaker_metrics.get('der')}",
                f"- Speaker-change hit rate: {speaker_metrics.get('speaker_change_hit_rate')}",
                f"- Speaker-change hits: {speaker_metrics.get('speaker_change_hits')}/{speaker_metrics.get('speaker_change_count')}",
                f"- Overlap reference segments: {speaker_metrics.get('overlap_segments')}",
            ]
        )
    lines.extend(["", "## Artifacts", ""])
    for relative_path in (
        "result.txt",
        "evaluation_manifest.json",
        "evaluation_status.json",
        "asr/wer_detail.txt",
        "asr/wer_summary.json",
        "speaker/speaker_diarization_summary.json",
        "speaker/speaker_diarization_boundary_details.csv",
    ):
        lines.append(f"- `{relative_path}`")
    for workbook_path in workbook_paths:
        lines.append(f"- `{workbook_path.name}` (Agent SDK six-column time-aligned ASR detail)")
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
    evaluation_records = [record for record in records if record.file_id in labels]
    inputs_dir = args.output_dir / "inputs"
    asr_dir = args.output_dir / "asr"
    speaker_dir = args.output_dir / "speaker"
    result_txt = args.output_dir / "result.txt"
    write_public_result_txt(records, result_txt)
    asr_paths = write_internal_asr_files(evaluation_records, inputs_dir)
    label_paths = write_internal_label_files(labels, inputs_dir / "labels")

    manifest = {
        "schema_version": 1,
        "tool": "python-jimmy/evaluation",
        "parameters": {
            "results_dir": str(args.results_dir),
            "labels_dir": None if args.labels_dir is None else str(args.labels_dir),
            "label": None if args.label is None else str(args.label),
            "output_dir": str(args.output_dir),
            "language": args.language,
            "boundary_tolerance_ms": args.boundary_tolerance_ms,
            "collar_ms": args.collar_ms,
        },
        "inputs": {
            "result_files": [
                {"path": str(record.path), "audio_name": record.audio_name, "file_id": record.file_id}
                for record in records
            ],
            "matched_labels": [
                {"file_id": file_id, "path": str(label.path)}
                for file_id, label in sorted(labels.items())
            ],
        },
        "outputs": {
            "result_txt": str(result_txt),
            "internal_asr_files": {file_id: str(path) for file_id, path in asr_paths.items()},
            "internal_label_files": {file_id: str(path) for file_id, path in label_paths.items()},
        },
    }
    _write_json(args.output_dir / "evaluation_manifest.json", manifest)

    tasks: dict[str, dict] = {
        "normalization": _task(
            "success",
            artifacts=[result_txt, args.output_dir / "evaluation_manifest.json"],
        )
    }
    wer_summary: Optional[dict] = None
    speaker_summary: Optional[dict] = None

    if not evaluation_records:
        tasks["wer"] = _task("skipped", error="no matching label file")
        tasks["speaker"] = _task("skipped", error="no matching label file")
    else:
        wer_label_lines, wer_hyp_lines = build_whole_audio_wer_inputs(evaluation_records, labels)
        wer_label_path = inputs_dir / "wer_label.txt"
        wer_hyp_path = inputs_dir / "wer_hyp.txt"
        wer_detail_path = asr_dir / "wer_detail.txt"
        _write_lines(wer_label_path, wer_label_lines)
        _write_lines(wer_hyp_path, wer_hyp_lines)
        wer_command = [
            __import__("sys").executable,
            str(Path(__file__).with_name("evaluation.py")),
            "--label", str(wer_label_path),
            "--hyp", str(wer_hyp_path),
            "--language", args.language,
            "--detail", str(wer_detail_path),
        ]
        wer_task = run_subprocess_task(
            wer_command, asr_dir / "evaluation.stdout.log", asr_dir / "evaluation.stderr.log"
        )
        if wer_task["status"] == "success":
            try:
                wer_summary = summarize_wer_detail(wer_detail_path)
                wer_summary_path = asr_dir / "wer_summary.json"
                _write_json(wer_summary_path, wer_summary)
                wer_task["artifacts"] = [str(wer_detail_path), str(wer_summary_path)]
            except (OSError, ValueError, json.JSONDecodeError) as error:
                wer_task = _task(
                    "failed",
                    error=str(error),
                    command=wer_command,
                    stdout_log=asr_dir / "evaluation.stdout.log",
                    stderr_log=asr_dir / "evaluation.stderr.log",
                )
        tasks["wer"] = wer_task

        speaker_command = [
            __import__("sys").executable,
            str(Path(__file__).with_name("speaker_diarization_metrics.py")),
            "--results-dir", str(inputs_dir),
            "--labels-dir", str(inputs_dir / "labels"),
            "--output-dir", str(speaker_dir),
            "--boundary-tolerance-ms", str(args.boundary_tolerance_ms),
            "--collar-ms", str(args.collar_ms),
        ]
        speaker_task = run_subprocess_task(
            speaker_command,
            speaker_dir / "speaker_metrics.stdout.log",
            speaker_dir / "speaker_metrics.stderr.log",
        )
        speaker_summary_path = speaker_dir / "speaker_diarization_summary.json"
        if speaker_task["status"] == "success":
            try:
                speaker_summary = _read_json_if_present(speaker_summary_path)
                if speaker_summary is None:
                    raise ValueError(f"speaker summary was not created: {speaker_summary_path}")
                speaker_task["artifacts"] = [
                    str(speaker_summary_path),
                    str(speaker_dir / "speaker_diarization_per_file.json"),
                    str(speaker_dir / "speaker_diarization_boundary_details.csv"),
                ]
            except (OSError, ValueError, json.JSONDecodeError) as error:
                speaker_task = _task(
                    "failed",
                    error=str(error),
                    command=speaker_command,
                    stdout_log=speaker_dir / "speaker_metrics.stdout.log",
                    stderr_log=speaker_dir / "speaker_metrics.stderr.log",
                )
        tasks["speaker"] = speaker_task

    workbook_paths: list[Path] = []
    if not evaluation_records:
        tasks["excel"] = _task("skipped", error="no matched labels")
    else:
        try:
            for record in evaluation_records:
                rows = build_agent_sdk_segment_detail_rows(
                    labels[record.file_id].segments,
                    record.segments,
                    args.language,
                )
                workbook_path = args.output_dir / f"{record.file_id}_segment_asr_detail.xlsx"
                write_agent_sdk_segment_detail_workbook(workbook_path, rows)
                workbook_paths.append(workbook_path)
            tasks["excel"] = _task("success", artifacts=workbook_paths)
        except OSError as error:
            tasks["excel"] = _task("failed", error=str(error))

    manifest["outputs"]["asr_detail_workbooks"] = [str(path) for path in workbook_paths]
    _write_json(args.output_dir / "evaluation_manifest.json", manifest)
    status = {
        "schema_version": 1,
        "overall_status": "failed" if any(task["status"] == "failed" for task in tasks.values()) else "success",
        "tasks": tasks,
    }
    status_path = args.output_dir / "evaluation_status.json"
    _write_json(status_path, status)
    report_path = args.output_dir / "evaluation_report.md"
    write_report(report_path, manifest, status, wer_summary, speaker_summary, workbook_paths)
    print(f"normalized {len(records)} audio result(s) into {args.output_dir}")
    print(f"matched {len(labels)} label file(s)")
    print(f"status: {status_path}")
    return 1 if status["overall_status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())

