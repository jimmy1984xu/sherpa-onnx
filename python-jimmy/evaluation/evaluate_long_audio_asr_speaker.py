#!/usr/bin/env python3
"""Evaluate long-audio ASR and speaker results without modifying pipeline output."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


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


def interval_overlap_ms(
    left_start_ms: int, left_end_ms: int, right_start_ms: int, right_end_ms: int
) -> int:
    return max(0, min(left_end_ms, right_end_ms) - max(left_start_ms, right_start_ms))


def _tokenize_for_wer(text: str, language: str) -> list[str]:
    from evaluation import text_normalization

    normalized = text_normalization(text, language)
    return normalized.split() if normalized else []


def compute_text_error_counts(reference: str, hypothesis: str, language: str) -> dict[str, int | float | None]:
    reference_tokens = _tokenize_for_wer(reference, language)
    hypothesis_tokens = _tokenize_for_wer(hypothesis, language)
    rows = len(reference_tokens) + 1
    cols = len(hypothesis_tokens) + 1
    # value: total errors, substitutions, deletions, insertions
    matrix: list[list[tuple[int, int, int, int]]] = [
        [(0, 0, 0, 0) for _ in range(cols)] for _ in range(rows)
    ]
    for index in range(1, rows):
        matrix[index][0] = (index, 0, index, 0)
    for index in range(1, cols):
        matrix[0][index] = (index, 0, 0, index)
    for ref_index in range(1, rows):
        for hyp_index in range(1, cols):
            if reference_tokens[ref_index - 1] == hypothesis_tokens[hyp_index - 1]:
                matrix[ref_index][hyp_index] = matrix[ref_index - 1][hyp_index - 1]
                continue
            substitution = matrix[ref_index - 1][hyp_index - 1]
            deletion = matrix[ref_index - 1][hyp_index]
            insertion = matrix[ref_index][hyp_index - 1]
            candidates = (
                (substitution[0] + 1, substitution[1] + 1, substitution[2], substitution[3]),
                (deletion[0] + 1, deletion[1], deletion[2] + 1, deletion[3]),
                (insertion[0] + 1, insertion[1], insertion[2], insertion[3] + 1),
            )
            matrix[ref_index][hyp_index] = min(candidates)
    errors, substitutions, deletions, insertions = matrix[-1][-1]
    return {
        "reference_tokens": len(reference_tokens),
        "hypothesis_tokens": len(hypothesis_tokens),
        "errors": errors,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "segment_wer_percent": None
        if not reference_tokens
        else round(100.0 * errors / len(reference_tokens), 4),
    }


def _mapping_type(candidate_count: int, reverse_counts: list[int]) -> str:
    if candidate_count == 1 and reverse_counts == [1]:
        return "one_to_one"
    if candidate_count == 1:
        return "one_to_many"
    if all(count == 1 for count in reverse_counts):
        return "many_to_one"
    return "many_to_many"


def _detail_row_base(
    reference: Optional[TimedSegment], hypothesis_segments: list[TimedSegment]
) -> dict:
    hypothesis_text = "".join(segment.asr_text for segment in hypothesis_segments)
    overlaps = [
        interval_overlap_ms(
            reference.start_ms, reference.end_ms, segment.start_ms, segment.end_ms
        )
        for segment in hypothesis_segments
    ] if reference else []
    return {
        "file_id": reference.file_id if reference else hypothesis_segments[0].file_id,
        "reference_segment_id": "" if reference is None else reference.segment_id,
        "reference_start_ms": "" if reference is None else reference.start_ms,
        "reference_end_ms": "" if reference is None else reference.end_ms,
        "reference_duration_ms": "" if reference is None else reference.duration_ms,
        "reference_speaker_id": "" if reference is None else reference.speaker_id,
        "reference_text": "" if reference is None else reference.asr_text,
        "hypothesis_segment_ids": ";".join(segment.segment_id for segment in hypothesis_segments),
        "hypothesis_start_ms": "" if not hypothesis_segments else min(segment.start_ms for segment in hypothesis_segments),
        "hypothesis_end_ms": "" if not hypothesis_segments else max(segment.end_ms for segment in hypothesis_segments),
        "hypothesis_speaker_ids": ";".join(segment.speaker_id for segment in hypothesis_segments),
        "hypothesis_text": hypothesis_text,
        "overlap_ms": sum(overlaps),
        "reference_coverage_percent": None
        if reference is None
        else round(100.0 * min(reference.duration_ms, sum(overlaps)) / reference.duration_ms, 4),
    }


def build_segment_detail_rows(
    reference_segments: list[TimedSegment],
    hypothesis_segments: list[TimedSegment],
    language: str,
) -> list[dict]:
    references = sorted(reference_segments, key=lambda item: (item.start_ms, item.end_ms, item.segment_id))
    hypotheses = sorted(hypothesis_segments, key=lambda item: (item.start_ms, item.end_ms, item.segment_id))
    single_references = [item for item in references if item.speaker_id.lower() != "multi"]
    reverse_counts = {
        hypothesis.segment_id: sum(
            interval_overlap_ms(reference.start_ms, reference.end_ms, hypothesis.start_ms, hypothesis.end_ms) > 0
            for reference in single_references
        )
        for hypothesis in hypotheses
    }
    rows: list[dict] = []
    matched_hypothesis_ids: set[str] = set()
    for reference in references:
        matches = [
            hypothesis
            for hypothesis in hypotheses
            if interval_overlap_ms(reference.start_ms, reference.end_ms, hypothesis.start_ms, hypothesis.end_ms) > 0
        ]
        matched_hypothesis_ids.update(segment.segment_id for segment in matches)
        row = _detail_row_base(reference, matches)
        if reference.speaker_id.lower() == "multi":
            row.update(
                {
                    "match_status": "overlap_not_scored",
                    "mapping_type": "not_scored",
                    "reference_tokens": None,
                    "hypothesis_tokens": None,
                    "errors": None,
                    "substitutions": None,
                    "deletions": None,
                    "insertions": None,
                    "segment_wer_percent": None,
                }
            )
        elif not matches:
            row.update({"match_status": "unmatched_reference", "mapping_type": "none"})
            row.update(compute_text_error_counts(reference.asr_text, "", language))
        else:
            row.update(
                {
                    "match_status": "matched",
                    "mapping_type": _mapping_type(
                        len(matches), [reverse_counts[item.segment_id] for item in matches]
                    ),
                }
            )
            row.update(compute_text_error_counts(reference.asr_text, row["hypothesis_text"], language))
        rows.append(row)
    for hypothesis in hypotheses:
        if hypothesis.segment_id in matched_hypothesis_ids:
            continue
        row = _detail_row_base(None, [hypothesis])
        row.update(
            {
                "match_status": "unmatched_prediction",
                "mapping_type": "none",
                "reference_tokens": 0,
                "hypothesis_tokens": len(_tokenize_for_wer(hypothesis.asr_text, language)),
                "errors": None,
                "substitutions": None,
                "deletions": None,
                "insertions": None,
                "segment_wer_percent": None,
            }
        )
        rows.append(row)
    return rows

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
    records = discover_result_records(args.results_dir)
    labels = resolve_labels(records, args.labels_dir, args.label)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_public_result_txt(records, args.output_dir / "result.txt")
    write_internal_asr_files(records, args.output_dir / "inputs")
    print(f"normalized {len(records)} audio result(s) into {args.output_dir}")
    print(f"matched {len(labels)} label file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
