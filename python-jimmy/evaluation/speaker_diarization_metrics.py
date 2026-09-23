#!/usr/bin/env python3
"""根据长音频 / raw8ch 的三列 ASR 与标注计算说话人 DER 与跨说话人断句命中率。"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Optional


_SEGMENT_TIME_RE = re.compile(r"_(\d+)_(\d+)$")
_TYPED_SPEAKER_RE = re.compile(r"^\((?P<speaker_id>[^()]*)\)(?:\((?P<segment_type>单人|短插话|重叠|听不清)\))?$")
SEGMENT_TYPE_SINGLE = "单人"
SEGMENT_TYPE_SHORT = "短插话"
SEGMENT_TYPE_OVERLAP = "重叠"
SEGMENT_TYPE_UNCLEAR = "听不清"
DER_ELIGIBLE_TYPES = frozenset({SEGMENT_TYPE_SINGLE, SEGMENT_TYPE_SHORT})
UNKNOWN_REFERENCE_TYPES = frozenset({SEGMENT_TYPE_OVERLAP, SEGMENT_TYPE_UNCLEAR})
UNKNOWN_SPEAKER_ID = "UNKNOWN"
_PYANNOTE_INSTALL_HINT = (
    "无法导入 pyannote.core 或 pyannote.metrics。请在独立 Python 环境中安装依赖:\n"
    "pip install pyannote.core pyannote.metrics"
)
OVERLAP_SPEAKER_A = "__overlap_a__"
OVERLAP_SPEAKER_B = "__overlap_b__"
OVERLAP_LABEL = "multi"

BOUNDARY_CSV_FIELDS = [
    "文件ID",
    "切换序号",
    "参考前说话人",
    "参考后说话人",
    "区间起始毫秒",
    "区间结束毫秒",
    "扩展后起始毫秒",
    "扩展后结束毫秒",
    "匹配预测边界毫秒",
    "状态",
]


@dataclass(frozen=True)
class TimedSpeakerSegment:
    segment_id: str
    speaker_id: str
    start_ms: int
    end_ms: int
    text: str
    segment_type: str = SEGMENT_TYPE_SINGLE

    @property
    def is_der_eligible(self) -> bool:
        return self.segment_type in DER_ELIGIBLE_TYPES

    @property
    def is_unknown_reference(self) -> bool:
        return self.segment_type in UNKNOWN_REFERENCE_TYPES


@dataclass(frozen=True)
class EvaluationReport:
    files: list[dict]
    asr_count: int
    evaluated_count: int
    error_count: int
    summary: dict
    boundary_rows: list[dict]


def _silence_pyannote_import_warnings() -> None:
    import os
    import warnings

    os.environ.setdefault("MPLBACKEND", "Agg")
    warnings.filterwarnings(
        "ignore",
        message="The get_cmap function was deprecated",
        module=r"pyannote\.core\.notebook",
    )


def require_pyannote() -> Optional[str]:
    _silence_pyannote_import_warnings()
    try:
        import pyannote.core  # noqa: F401
        import pyannote.metrics  # noqa: F401
    except ImportError:
        return _PYANNOTE_INSTALL_HINT
    return None


def _validate_typed_reference(speaker_id: str, segment_type: str) -> None:
    if segment_type == SEGMENT_TYPE_OVERLAP and speaker_id != "MULTI":
        raise ValueError("重叠 segment_type requires speaker_id MULTI")
    if segment_type == SEGMENT_TYPE_UNCLEAR and speaker_id != "UNCLEAR":
        raise ValueError("听不清 segment_type requires speaker_id UNCLEAR")
    if segment_type in DER_ELIGIBLE_TYPES and speaker_id in {"MULTI", "UNCLEAR"}:
        raise ValueError(f"{segment_type} segment_type cannot use reserved speaker_id {speaker_id}")


def parse_speaker_line(line: str) -> TimedSpeakerSegment:
    value = line.strip()
    columns = value.split(maxsplit=2)
    if len(columns) < 2:
        raise ValueError(f"行格式错误，期望“片段ID 说话人ID [文本]”: {value}")
    segment_id = columns[0]
    speaker_field = columns[1]
    text = columns[2] if len(columns) > 2 else ""
    typed_match = _TYPED_SPEAKER_RE.fullmatch(speaker_field)
    if typed_match is None:
        if speaker_field.startswith("("):
            raise ValueError(
                "说话人字段格式错误，期望 '(speaker_id)' 或 '(speaker_id)(片段类型)'"
            )
        speaker_id = speaker_field.strip().strip("()")
        canonical_speaker_id = speaker_id.upper()
        segment_type = (
            SEGMENT_TYPE_OVERLAP
            if canonical_speaker_id == "MULTI"
            else SEGMENT_TYPE_UNCLEAR
            if canonical_speaker_id == "UNCLEAR"
            else SEGMENT_TYPE_SINGLE
        )
    else:
        speaker_id = typed_match.group("speaker_id").strip()
        explicit_type = typed_match.group("segment_type")
        canonical_speaker_id = speaker_id.upper()
        segment_type = explicit_type or (
            SEGMENT_TYPE_OVERLAP
            if canonical_speaker_id == "MULTI"
            else SEGMENT_TYPE_UNCLEAR
            if canonical_speaker_id == "UNCLEAR"
            else SEGMENT_TYPE_SINGLE
        )
        if explicit_type is not None:
            _validate_typed_reference(speaker_id, segment_type)
    match = _SEGMENT_TIME_RE.search(segment_id)
    if match is None:
        raise ValueError(f"片段ID缺少 _起始毫秒_时长毫秒 后缀: {segment_id}")
    start_ms = int(match.group(1))
    duration_ms = int(match.group(2))
    if duration_ms <= 0:
        raise ValueError(f"片段结束时间不大于起始时间: {segment_id}")
    return TimedSpeakerSegment(
        segment_id, speaker_id, start_ms, start_ms + duration_ms, text, segment_type
    )


def parse_speaker_file(path: Path) -> list[TimedSpeakerSegment]:
    segments: list[TimedSpeakerSegment] = []
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                segments.append(parse_speaker_line(line))
            except ValueError as error:
                raise ValueError(f"{path} 第 {line_number} 行: {error}") from error
    return segments


def is_overlap_speaker(speaker_id: str) -> bool:
    return speaker_id.strip().strip("()").lower() == OVERLAP_LABEL


def der_eligible_segments(segments: Iterable[TimedSpeakerSegment]) -> list[TimedSpeakerSegment]:
    return [segment for segment in segments if segment.is_der_eligible]


def unknown_reference_segments(segments: Iterable[TimedSpeakerSegment]) -> list[TimedSpeakerSegment]:
    return [segment for segment in segments if segment.is_unknown_reference]


def _overlaps_region(segment: TimedSpeakerSegment, regions: Iterable[TimedSpeakerSegment]) -> bool:
    return any(segment.start_ms < region.end_ms and region.start_ms < segment.end_ms for region in regions)


def _scoped_hypothesis_segments(
    hypothesis_segments: Iterable[TimedSpeakerSegment], reference_segments: Iterable[TimedSpeakerSegment]
) -> list[TimedSpeakerSegment]:
    regions = list(reference_segments)
    return [segment for segment in hypothesis_segments if _overlaps_region(segment, regions)]


def describe_segments(
    reference_segments: Iterable[TimedSpeakerSegment],
    hypothesis_segments: Iterable[TimedSpeakerSegment],
) -> dict:
    references = list(reference_segments)
    eligible = der_eligible_segments(references)
    hypotheses = _scoped_hypothesis_segments(hypothesis_segments, eligible)
    return {
        "reference_segments": len(eligible),
        "hypothesis_segments": len(hypotheses),
        "reference_speakers": len({item.speaker_id for item in eligible}),
        "hypothesis_speakers": len(
            {item.speaker_id for item in hypotheses if item.speaker_id != UNKNOWN_SPEAKER_ID}
        ),
        "overlap_segments": sum(1 for item in references if item.segment_type == SEGMENT_TYPE_OVERLAP),
        "unclear_segments": sum(1 for item in references if item.segment_type == SEGMENT_TYPE_UNCLEAR),
    }


def merge_same_speaker_turns(segments: Iterable[TimedSpeakerSegment]) -> list[TimedSpeakerSegment]:
    merged: list[TimedSpeakerSegment] = []
    for segment in sorted(segments, key=lambda item: (item.start_ms, item.end_ms, item.segment_id)):
        if merged and merged[-1].speaker_id == segment.speaker_id:
            previous = merged[-1]
            merged[-1] = TimedSpeakerSegment(
                previous.segment_id,
                previous.speaker_id,
                previous.start_ms,
                max(previous.end_ms, segment.end_ms),
                previous.text,
                previous.segment_type,
            )
        else:
            merged.append(segment)
    return merged


def speaker_change_intervals(merged: list[TimedSpeakerSegment]) -> list[tuple[int, int, str, str]]:
    intervals: list[tuple[int, int, str, str]] = []
    for left, right in zip(merged, merged[1:]):
        if left.speaker_id != right.speaker_id:
            intervals.append((left.end_ms, right.start_ms, left.speaker_id, right.speaker_id))
    return intervals


def match_speaker_change_boundaries(
    intervals: list[tuple[int, int, str, str]],
    prediction_ends_ms: list[int],
    tolerance_ms: int,
) -> tuple[Optional[int], list[dict]]:
    if not intervals:
        return None, []
    used = [False] * len(prediction_ends_ms)
    details: list[dict] = []
    hits = 0
    for start_ms, end_ms, from_speaker, to_speaker in intervals:
        expanded_start = min(start_ms, end_ms) - tolerance_ms
        expanded_end = max(start_ms, end_ms) + tolerance_ms
        match_end: Optional[int] = None
        for index, pred_end in enumerate(prediction_ends_ms):
            if used[index]:
                continue
            if expanded_start <= pred_end <= expanded_end:
                used[index] = True
                match_end = pred_end
                hits += 1
                break
        details.append(
            {
                "from_speaker": from_speaker,
                "to_speaker": to_speaker,
                "interval_start_ms": start_ms,
                "interval_end_ms": end_ms,
                "expanded_start_ms": expanded_start,
                "expanded_end_ms": expanded_end,
                "matched_prediction_end_ms": match_end,
                "status": "命中" if match_end is not None else "未命中",
            }
        )
    return hits, details


def _ratio(numerator: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return numerator / denominator


def _f1(precision: Optional[float], recall: Optional[float]) -> Optional[float]:
    if precision is None or recall is None:
        return None
    if precision + recall <= 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _merge_intervals(segments: Iterable[TimedSpeakerSegment]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for segment in sorted(segments, key=lambda item: (item.start_ms, item.end_ms)):
        if segment.end_ms <= segment.start_ms:
            continue
        if merged and segment.start_ms <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], segment.end_ms))
        else:
            merged.append((segment.start_ms, segment.end_ms))
    return merged


def _interval_duration(intervals: Iterable[tuple[int, int]]) -> int:
    return sum(end - start for start, end in intervals)


def _intersection_duration(left: Iterable[tuple[int, int]], right: Iterable[tuple[int, int]]) -> int:
    left_items = list(left)
    right_items = list(right)
    total = 0
    left_index = right_index = 0
    while left_index < len(left_items) and right_index < len(right_items):
        left_start, left_end = left_items[left_index]
        right_start, right_end = right_items[right_index]
        total += max(0, min(left_end, right_end) - max(left_start, right_start))
        if left_end <= right_end:
            left_index += 1
        else:
            right_index += 1
    return total


def compute_unknown_duration_metrics(
    reference_segments: Iterable[TimedSpeakerSegment], hypothesis_segments: Iterable[TimedSpeakerSegment]
) -> dict:
    references = list(reference_segments)
    reference_unknown = _merge_intervals(unknown_reference_segments(references))
    labelled_uem = _merge_intervals(references)
    predicted_unknown = _merge_intervals(
        segment for segment in hypothesis_segments if segment.speaker_id == UNKNOWN_SPEAKER_ID
    )
    predicted_in_uem: list[tuple[int, int]] = []
    for start_ms, end_ms in predicted_unknown:
        for uem_start, uem_end in labelled_uem:
            left, right = max(start_ms, uem_start), min(end_ms, uem_end)
            if right > left:
                predicted_in_uem.append((left, right))
    predicted_in_uem = _merge_intervals(
        TimedSpeakerSegment("unknown", UNKNOWN_SPEAKER_ID, start, end, "")
        for start, end in predicted_in_uem
    )
    reference_ms = _interval_duration(reference_unknown)
    predicted_ms = _interval_duration(predicted_in_uem)
    true_positive_ms = _intersection_duration(reference_unknown, predicted_in_uem)
    false_positive_ms = predicted_ms - true_positive_ms
    false_negative_ms = reference_ms - true_positive_ms
    precision = _ratio(true_positive_ms, predicted_ms)
    recall = _ratio(true_positive_ms, reference_ms)
    return {
        "precision": precision,
        "recall": 0.0 if recall is None and reference_ms > 0 else recall,
        "f1": 0.0 if reference_ms > 0 and predicted_ms == 0 else _f1(precision, recall),
        "reference_unknown_ms": reference_ms,
        "predicted_unknown_ms": predicted_ms,
        "true_positive_ms": true_positive_ms,
        "false_positive_ms": false_positive_ms,
        "false_negative_ms": false_negative_ms,
        "by_reference_type": {
            "overlap": _unknown_type_coverage(references, predicted_in_uem, SEGMENT_TYPE_OVERLAP),
            "unclear": _unknown_type_coverage(references, predicted_in_uem, SEGMENT_TYPE_UNCLEAR),
        },
    }


def _unknown_type_coverage(
    references: list[TimedSpeakerSegment], predicted_unknown: list[tuple[int, int]], segment_type: str
) -> dict:
    typed = _merge_intervals(segment for segment in references if segment.segment_type == segment_type)
    reference_ms = _interval_duration(typed)
    covered_ms = _intersection_duration(typed, predicted_unknown)
    return {
        "reference_unknown_ms": reference_ms,
        "covered_ms": covered_ms,
        "recall": _ratio(covered_ms, reference_ms),
    }


def _segment_alignment_metrics(
    references: list[TimedSpeakerSegment], hypotheses: list[TimedSpeakerSegment]
) -> dict:
    candidates: list[tuple[float, int, int]] = []
    for ref_index, reference in enumerate(references):
        for hyp_index, hypothesis in enumerate(hypotheses):
            overlap = max(0, min(reference.end_ms, hypothesis.end_ms) - max(reference.start_ms, hypothesis.start_ms))
            union = max(reference.end_ms, hypothesis.end_ms) - min(reference.start_ms, hypothesis.start_ms)
            iou = 0.0 if union <= 0 else overlap / union
            if iou >= 0.50:
                candidates.append((iou, ref_index, hyp_index))
    used_refs: set[int] = set()
    used_hyps: set[int] = set()
    matches = 0
    for _, ref_index, hyp_index in sorted(candidates, reverse=True):
        if ref_index not in used_refs and hyp_index not in used_hyps:
            used_refs.add(ref_index)
            used_hyps.add(hyp_index)
            matches += 1
    precision = _ratio(matches, len(hypotheses))
    recall = _ratio(matches, len(references))
    return {
        "matches": matches,
        "reference_count": len(references),
        "hypothesis_count": len(hypotheses),
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
    }


def prediction_file_id(path: Path) -> str:
    name = path.name
    if not name.endswith("_asr.txt"):
        raise ValueError(f"不是预测 ASR 文件: {path}")
    file_id = name[: -len("_asr.txt")]
    if not file_id:
        raise ValueError(f"无法从预测文件名提取文件 ID: {path}")
    return file_id


def find_label_paths(labels_dir: Path, file_id: str) -> list[Path]:
    wanted = {f"{file_id}_label.txt"}
    if file_id.lower().endswith("_raw8ch"):
        wanted.add(f"{file_id[:-len('_raw8ch')]}_label.txt")
    found: list[Path] = []
    seen: set[Path] = set()
    for path in labels_dir.rglob("*"):
        if not path.is_file() or path.name not in wanted:
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        found.append(path)
    return sorted(found)


def resolve_label_path(labels_dir: Path, file_id: str) -> Path:
    matches = find_label_paths(labels_dir, file_id)
    if not matches:
        raise ValueError(f"未找到 {file_id} 的标注文件")
    if len(matches) > 1:
        joined = ", ".join(str(path) for path in matches)
        raise ValueError(f"{file_id} 对应多个标注文件: {joined}")
    return matches[0]


def _iter_asr_files(results_dir: Path) -> list[Path]:
    return sorted(
        path for path in results_dir.rglob("*") if path.is_file() and path.name.endswith("_asr.txt")
    )


def _to_annotation(segments: list[TimedSpeakerSegment]):
    _silence_pyannote_import_warnings()
    from pyannote.core import Annotation, Segment

    annotation = Annotation()
    for index, segment in enumerate(segments):
        start = segment.start_ms / 1000.0
        end = segment.end_ms / 1000.0
        if end > start:
            annotation[Segment(start, end), index] = str(segment.speaker_id)
    return annotation


def compute_der(
    reference_segments: list[TimedSpeakerSegment],
    hypothesis_segments: list[TimedSpeakerSegment],
    collar_ms: int,
) -> tuple[Optional[float], dict]:
    from pyannote.metrics.diarization import DiarizationErrorRate

    eligible_references = der_eligible_segments(reference_segments)
    reference = _to_annotation(eligible_references)
    hypothesis = _to_annotation(hypothesis_segments)
    collar = collar_ms / 1000.0
    uem = reference.get_timeline()
    der_metric = DiarizationErrorRate(collar=collar, skip_overlap=True)
    components = der_metric.compute_components(reference, hypothesis, uem=uem)
    der_value = der_metric.compute_metric(components) if components.get("total") else None
    detail = {
        "miss": float(components.get("missed detection", 0.0)),
        "false_alarm": float(components.get("false alarm", 0.0)),
        "confusion": float(components.get("confusion", 0.0)),
        "total": float(components.get("total", 0.0)),
    }
    return (None if der_value is None else float(der_value), detail)


def _boundary_metrics(
    hits: Optional[int], reference_count: int, prediction_count: int
) -> dict:
    matched = 0 if hits is None else hits
    precision = _ratio(matched, prediction_count)
    recall = _ratio(matched, reference_count)
    return {
        "hits": matched,
        "reference_count": reference_count,
        "prediction_count": prediction_count,
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
    }


def _metrics_payload(
    der: Optional[float],
    inventory: Optional[dict] = None,
    boundary: Optional[dict] = None,
    segment_alignment: Optional[dict] = None,
    unknown: Optional[dict] = None,
) -> dict:
    payload = {
        "der": der,
        "speaker_change_hit_rate": None,
        "speaker_change_hits": 0,
        "speaker_change_count": 0,
        "speaker_change_prediction_count": 0,
        "speaker_change_precision": None,
        "speaker_change_recall": None,
        "speaker_change_f1": None,
        "segment_alignment": {
            "matches": 0,
            "reference_count": 0,
            "hypothesis_count": 0,
            "precision": None,
            "recall": None,
            "f1": None,
        },
        "unknown": {
            "precision": None,
            "recall": None,
            "f1": None,
            "reference_unknown_ms": 0,
            "predicted_unknown_ms": 0,
            "true_positive_ms": 0,
            "false_positive_ms": 0,
            "false_negative_ms": 0,
            "by_reference_type": {
                "overlap": {"reference_unknown_ms": 0, "covered_ms": 0, "recall": None},
                "unclear": {"reference_unknown_ms": 0, "covered_ms": 0, "recall": None},
            },
        },
        "reference_segments": 0,
        "hypothesis_segments": 0,
        "reference_speakers": 0,
        "hypothesis_speakers": 0,
        "overlap_segments": 0,
        "unclear_segments": 0,
    }
    if inventory:
        payload.update(inventory)
    if boundary:
        payload.update(
            {
                "speaker_change_hit_rate": boundary["recall"],
                "speaker_change_hits": boundary["hits"],
                "speaker_change_count": boundary["reference_count"],
                "speaker_change_prediction_count": boundary["prediction_count"],
                "speaker_change_precision": boundary["precision"],
                "speaker_change_recall": boundary["recall"],
                "speaker_change_f1": boundary["f1"],
            }
        )
    if segment_alignment:
        payload["segment_alignment"] = segment_alignment
    if unknown:
        payload["unknown"] = unknown
    return payload


def evaluate_file(
    asr_path: Path,
    labels_dir: Path,
    boundary_tolerance_ms: int,
    collar_ms: int,
) -> dict:
    file_id = prediction_file_id(asr_path)
    hypothesis_segments = parse_speaker_file(asr_path)
    label_path = resolve_label_path(labels_dir, file_id)
    reference_segments = parse_speaker_file(label_path)
    eligible_references = der_eligible_segments(reference_segments)
    scoped_hypotheses = _scoped_hypothesis_segments(hypothesis_segments, eligible_references)

    der, der_components = compute_der(reference_segments, hypothesis_segments, collar_ms)
    merged = merge_same_speaker_turns(eligible_references)
    intervals = speaker_change_intervals(merged)
    prediction_ends = [item.end_ms for item in scoped_hypotheses]
    hits, boundary_details = match_speaker_change_boundaries(
        intervals, prediction_ends, boundary_tolerance_ms
    )
    boundary = _boundary_metrics(hits, len(intervals), len(prediction_ends))
    inventory = describe_segments(reference_segments, hypothesis_segments)
    segment_alignment = _segment_alignment_metrics(eligible_references, scoped_hypotheses)
    unknown = compute_unknown_duration_metrics(reference_segments, hypothesis_segments)
    return {
        "file_id": file_id,
        "asr_path": str(asr_path),
        "label_path": str(label_path),
        "error": None,
        "metrics": _metrics_payload(
            der,
            inventory=inventory,
            boundary=boundary,
            segment_alignment=segment_alignment,
            unknown=unknown,
        ),
        "der_components": der_components,
        "boundary_details": boundary_details,
    }


def _aggregate(files: list[dict]) -> dict:
    evaluated = [item for item in files if item.get("error") is None]
    if not evaluated:
        return _metrics_payload(None)

    miss = sum(item["der_components"]["miss"] for item in evaluated)
    false_alarm = sum(item["der_components"]["false_alarm"] for item in evaluated)
    confusion = sum(item["der_components"]["confusion"] for item in evaluated)
    total = sum(item["der_components"]["total"] for item in evaluated)
    inventory = {
        key: sum(item["metrics"][key] for item in evaluated)
        for key in (
            "reference_segments",
            "hypothesis_segments",
            "reference_speakers",
            "hypothesis_speakers",
            "overlap_segments",
            "unclear_segments",
        )
    }
    boundary = _boundary_metrics(
        sum(item["metrics"]["speaker_change_hits"] for item in evaluated),
        sum(item["metrics"]["speaker_change_count"] for item in evaluated),
        sum(item["metrics"]["speaker_change_prediction_count"] for item in evaluated),
    )

    alignment_matches = sum(item["metrics"]["segment_alignment"]["matches"] for item in evaluated)
    alignment_references = sum(
        item["metrics"]["segment_alignment"]["reference_count"] for item in evaluated
    )
    alignment_hypotheses = sum(
        item["metrics"]["segment_alignment"]["hypothesis_count"] for item in evaluated
    )
    segment_alignment = {
        "matches": alignment_matches,
        "reference_count": alignment_references,
        "hypothesis_count": alignment_hypotheses,
        "precision": _ratio(alignment_matches, alignment_hypotheses),
        "recall": _ratio(alignment_matches, alignment_references),
    }
    segment_alignment["f1"] = _f1(
        segment_alignment["precision"], segment_alignment["recall"]
    )

    unknown_parts = [item["metrics"]["unknown"] for item in evaluated]
    unknown_tp = sum(item["true_positive_ms"] for item in unknown_parts)
    unknown_predicted = sum(item["predicted_unknown_ms"] for item in unknown_parts)
    unknown_reference = sum(item["reference_unknown_ms"] for item in unknown_parts)
    unknown = {
        "precision": _ratio(unknown_tp, unknown_predicted),
        "recall": _ratio(unknown_tp, unknown_reference),
        "reference_unknown_ms": unknown_reference,
        "predicted_unknown_ms": unknown_predicted,
        "true_positive_ms": unknown_tp,
        "false_positive_ms": sum(item["false_positive_ms"] for item in unknown_parts),
        "false_negative_ms": sum(item["false_negative_ms"] for item in unknown_parts),
        "by_reference_type": {},
    }
    unknown["recall"] = 0.0 if unknown["recall"] is None and unknown_reference > 0 else unknown["recall"]
    unknown["f1"] = 0.0 if unknown_reference > 0 and unknown_predicted == 0 else _f1(unknown["precision"], unknown["recall"])
    for name in ("overlap", "unclear"):
        reference_ms = sum(item["by_reference_type"][name]["reference_unknown_ms"] for item in unknown_parts)
        covered_ms = sum(item["by_reference_type"][name]["covered_ms"] for item in unknown_parts)
        unknown["by_reference_type"][name] = {
            "reference_unknown_ms": reference_ms,
            "covered_ms": covered_ms,
            "recall": _ratio(covered_ms, reference_ms),
        }

    return _metrics_payload(
        None if total <= 0 else (miss + false_alarm + confusion) / total,
        inventory=inventory,
        boundary=boundary,
        segment_alignment=segment_alignment,
        unknown=unknown,
    )


def evaluate_paths(
    results_dir: Path,
    labels_dir: Path,
    boundary_tolerance_ms: int = 500,
    collar_ms: int = 500,
) -> EvaluationReport:
    asr_files = _iter_asr_files(results_dir)
    files: list[dict] = []
    for asr_path in asr_files:
        try:
            files.append(evaluate_file(asr_path, labels_dir, boundary_tolerance_ms, collar_ms))
        except (OSError, ValueError) as error:
            file_id = asr_path.name[: -len("_asr.txt")] if asr_path.name.endswith("_asr.txt") else asr_path.stem
            files.append(
                {
                    "file_id": file_id,
                    "asr_path": str(asr_path),
                    "label_path": None,
                    "error": str(error),
                    "metrics": None,
                    "der_components": None,
                    "boundary_details": [],
                }
            )
    error_count = sum(1 for item in files if item.get("error"))
    evaluated_count = len(files) - error_count
    summary = {
        "parameters": {
            "results_dir": str(results_dir),
            "labels_dir": str(labels_dir),
            "boundary_tolerance_ms": boundary_tolerance_ms,
            "collar_ms": collar_ms,
            "skip_overlap": True,
        },
        "counts": {
            "asr_files": len(asr_files),
            "evaluated": evaluated_count,
            "errors": error_count,
        },
        "metrics": _aggregate(files),
    }
    boundary_rows: list[dict] = []
    for item in files:
        for index, detail in enumerate(item.get("boundary_details") or [], 1):
            boundary_rows.append(
                {
                    "文件ID": item["file_id"],
                    "切换序号": index,
                    "参考前说话人": detail["from_speaker"],
                    "参考后说话人": detail["to_speaker"],
                    "区间起始毫秒": detail["interval_start_ms"],
                    "区间结束毫秒": detail["interval_end_ms"],
                    "扩展后起始毫秒": detail["expanded_start_ms"],
                    "扩展后结束毫秒": detail["expanded_end_ms"],
                    "匹配预测边界毫秒": "" if detail["matched_prediction_end_ms"] is None else detail["matched_prediction_end_ms"],
                    "状态": detail["status"],
                }
            )
    return EvaluationReport(
        files=files,
        asr_count=len(asr_files),
        evaluated_count=evaluated_count,
        error_count=error_count,
        summary=summary,
        boundary_rows=boundary_rows,
    )


def _public_file_row(item: dict) -> dict:
    return {
        "file_id": item["file_id"],
        "asr_path": item["asr_path"],
        "label_path": item.get("label_path"),
        "error": item.get("error"),
        "metrics": item.get("metrics"),
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_metric(value: Optional[float]) -> str:
    if value is None:
        return "null"
    return f"{value:.4f}"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True, metavar="目录", help="预测结果目录，递归读取 *_asr.txt")
    parser.add_argument("--labels-dir", type=Path, required=True, metavar="目录", help="标注根目录，按文件 ID 查找 *_label.txt")
    parser.add_argument("--output-dir", type=Path, required=True, metavar="目录", help="指标输出目录")
    parser.add_argument("--boundary-tolerance-ms", type=int, default=500, metavar="500", help="跨说话人断句命中容差毫秒（默认：500）")
    parser.add_argument("--collar-ms", type=int, default=500, metavar="500", help="DER 边界 collar 毫秒（默认：500）")
    args = parser.parse_args(argv)
    missing = require_pyannote()
    if missing:
        print(missing, file=sys.stderr)
        return 1
    if args.boundary_tolerance_ms < 0 or args.collar_ms < 0:
        print("容差和 collar 不能为负数", file=sys.stderr)
        return 2
    if not args.results_dir.is_dir():
        print(f"预测结果目录不存在: {args.results_dir}", file=sys.stderr)
        return 2
    if not args.labels_dir.is_dir():
        print(f"标注目录不存在: {args.labels_dir}", file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = evaluate_paths(
        results_dir=args.results_dir,
        labels_dir=args.labels_dir,
        boundary_tolerance_ms=args.boundary_tolerance_ms,
        collar_ms=args.collar_ms,
    )
    summary = dict(report.summary)
    summary["parameters"]["output_dir"] = str(args.output_dir)
    summary_path = args.output_dir / "speaker_diarization_summary.json"
    per_file_path = args.output_dir / "speaker_diarization_per_file.json"
    boundary_path = args.output_dir / "speaker_diarization_boundary_details.csv"
    _write_json(summary_path, summary)
    _write_json(per_file_path, {"files": [_public_file_row(item) for item in report.files]})
    _write_csv(boundary_path, BOUNDARY_CSV_FIELDS, report.boundary_rows)
    metrics = summary["metrics"]
    hits = metrics.get("speaker_change_hits")
    change_count = metrics.get("speaker_change_count")
    print(f"说话人日志指标: {summary_path}")
    print(
        "DER={der} 切换命中率={hit} ({hits}/{changes}) "
        "标注片段={ref_seg} 预测片段={hyp_seg} "
        "标注说话人={ref_spk} 预测说话人={hyp_spk} 重叠片段={overlap} "
        "评估={evaluated} 错误={errors}".format(
            der=_format_metric(metrics.get("der")),
            hit=_format_metric(metrics.get("speaker_change_hit_rate")),
            hits=hits,
            changes=change_count,
            ref_seg=metrics.get("reference_segments"),
            hyp_seg=metrics.get("hypothesis_segments"),
            ref_spk=metrics.get("reference_speakers"),
            hyp_spk=metrics.get("hypothesis_speakers"),
            overlap=metrics.get("overlap_segments"),
            evaluated=report.evaluated_count,
            errors=report.error_count,
        )
    )
    return 1 if report.error_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
