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


def parse_speaker_line(line: str) -> TimedSpeakerSegment:
    value = line.strip()
    columns = value.split(maxsplit=2)
    if len(columns) < 2:
        raise ValueError(f"行格式错误，期望“片段ID 说话人ID [文本]”: {value}")
    segment_id = columns[0]
    speaker_id = columns[1].strip().strip("()")
    text = columns[2] if len(columns) > 2 else ""
    match = _SEGMENT_TIME_RE.search(segment_id)
    if match is None:
        raise ValueError(f"片段ID缺少 _起始毫秒_时长毫秒 后缀: {segment_id}")
    start_ms = int(match.group(1))
    duration_ms = int(match.group(2))
    if duration_ms <= 0:
        raise ValueError(f"片段结束时间不大于起始时间: {segment_id}")
    return TimedSpeakerSegment(segment_id, speaker_id, start_ms, start_ms + duration_ms, text)


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


def expand_overlap_for_der(segments: Iterable[TimedSpeakerSegment]) -> list[TimedSpeakerSegment]:
    expanded: list[TimedSpeakerSegment] = []
    for segment in segments:
        if not is_overlap_speaker(segment.speaker_id):
            expanded.append(segment)
            continue
        expanded.append(replace(segment, speaker_id=OVERLAP_SPEAKER_A))
        expanded.append(replace(segment, speaker_id=OVERLAP_SPEAKER_B, segment_id=f"{segment.segment_id}#b"))
    return expanded


def single_speaker_segments(segments: Iterable[TimedSpeakerSegment]) -> list[TimedSpeakerSegment]:
    return [segment for segment in segments if not is_overlap_speaker(segment.speaker_id)]


def describe_segments(
    reference_segments: Iterable[TimedSpeakerSegment],
    hypothesis_segments: Iterable[TimedSpeakerSegment],
) -> dict:
    references = list(reference_segments)
    hypotheses = list(hypothesis_segments)
    return {
        "reference_segments": len(references),
        "hypothesis_segments": len(hypotheses),
        "reference_speakers": len({item.speaker_id for item in single_speaker_segments(references)}),
        "hypothesis_speakers": len({item.speaker_id for item in single_speaker_segments(hypotheses)}),
        "overlap_segments": sum(1 for item in references if is_overlap_speaker(item.speaker_id)),
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

    reference = _to_annotation(expand_overlap_for_der(reference_segments))
    hypothesis = _to_annotation(hypothesis_segments)
    collar = collar_ms / 1000.0
    uem = reference.get_timeline().union(hypothesis.get_timeline())
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


def _metrics_payload(
    der: Optional[float],
    hits: Optional[int],
    change_count: int,
    inventory: Optional[dict] = None,
) -> dict:
    payload = {
        "der": der,
        "speaker_change_hit_rate": None if hits is None else _ratio(hits, change_count),
        "speaker_change_hits": 0 if hits is None else hits,
        "speaker_change_count": change_count,
        "reference_segments": 0,
        "hypothesis_segments": 0,
        "reference_speakers": 0,
        "hypothesis_speakers": 0,
        "overlap_segments": 0,
    }
    if inventory:
        payload.update(inventory)
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
    der, der_components = compute_der(reference_segments, hypothesis_segments, collar_ms)
    merged = merge_same_speaker_turns(single_speaker_segments(reference_segments))
    intervals = speaker_change_intervals(merged)
    hits, boundary_details = match_speaker_change_boundaries(
        intervals,
        [item.end_ms for item in hypothesis_segments],
        boundary_tolerance_ms,
    )
    inventory = describe_segments(reference_segments, hypothesis_segments)
    return {
        "file_id": file_id,
        "asr_path": str(asr_path),
        "label_path": str(label_path),
        "error": None,
        "metrics": _metrics_payload(der, hits, len(intervals), inventory),
        "der_components": der_components,
        "boundary_details": boundary_details,
    }


def _aggregate(files: list[dict]) -> dict:
    evaluated = [item for item in files if item.get("error") is None]
    if not evaluated:
        return _metrics_payload(None, None, 0)
    miss = sum(item["der_components"]["miss"] for item in evaluated)
    false_alarm = sum(item["der_components"]["false_alarm"] for item in evaluated)
    confusion = sum(item["der_components"]["confusion"] for item in evaluated)
    total = sum(item["der_components"]["total"] for item in evaluated)
    change_count = sum(item["metrics"]["speaker_change_count"] for item in evaluated)
    hits = sum(item["metrics"]["speaker_change_hits"] for item in evaluated)
    has_changes = any(item["metrics"]["speaker_change_count"] for item in evaluated)
    inventory = {
        "reference_segments": sum(item["metrics"]["reference_segments"] for item in evaluated),
        "hypothesis_segments": sum(item["metrics"]["hypothesis_segments"] for item in evaluated),
        "reference_speakers": sum(item["metrics"]["reference_speakers"] for item in evaluated),
        "hypothesis_speakers": sum(item["metrics"]["hypothesis_speakers"] for item in evaluated),
        "overlap_segments": sum(item["metrics"]["overlap_segments"] for item in evaluated),
    }
    return {
        "der": None if total <= 0 else (miss + false_alarm + confusion) / total,
        "speaker_change_hit_rate": _ratio(hits, change_count) if has_changes else None,
        "speaker_change_hits": hits,
        "speaker_change_count": change_count,
        **inventory,
    }


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
