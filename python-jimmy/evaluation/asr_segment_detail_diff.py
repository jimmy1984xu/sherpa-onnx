#!/usr/bin/env python3
"""计算 agent-sdk-test 音频结果的纯 ASR WER。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
from xml.sax.saxutils import escape


@dataclass(frozen=True)
class LabelLine:
    utterance_id: str
    text: str


@dataclass(frozen=True)
class TimedSegment:
    segment_id: str
    text: str
    offset_ms: Optional[int]
    duration_ms: Optional[int]

    @property
    def end_ms(self) -> Optional[int]:
        if self.offset_ms is None or self.duration_ms is None:
            return None
        return self.offset_ms + self.duration_ms


@dataclass(frozen=True)
class SegmentDetailRow:
    reference_id: str
    reference_text: str
    hypothesis_id: str
    hypothesis_text: str
    status: str
    diff: str


@dataclass(frozen=True)
class WerStats:
    reference_count: int
    errors: int
    deletions: int
    insertions: int
    substitutions: int

    @property
    def rate(self) -> float:
        return self.errors / self.reference_count if self.reference_count else 0.0


@dataclass(frozen=True)
class EvaluationRow:
    utterance_id: str
    stats: WerStats
    diff: str


def _read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"无法解析 {path} 第 {line_number} 行: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path} 第 {line_number} 行不是 JSON 对象")
            rows.append(row)
    if not rows:
        raise ValueError(f"结果文件为空: {path}")
    return rows


def _read_label_lines(path: Path) -> list[LabelLine]:
    result = []
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            value = line.strip()
            if not value:
                continue
            columns = value.split(maxsplit=2)
            if len(columns) == 2:
                utterance_id, text = columns
            elif len(columns) == 3:
                utterance_id, _speaker, text = columns
            else:
                raise ValueError(f"标注文件 {path} 第 {line_number} 行格式错误，期望两列或三列")
            if not text.strip():
                raise ValueError(f"标注文件 {path} 第 {line_number} 行 ASR 文本为空")
            result.append(LabelLine(utterance_id, text.strip()))
    if not result:
        raise ValueError(f"标注文件为空: {path}")
    return result


def _safe_relative_path(value: str) -> str:
    normalized = value.strip().replace("\\", "/")
    parts = [part for part in normalized.split("/") if part not in {"", "."}]
    if not parts or normalized.startswith("/") or re.match(r"^[A-Za-z]:/", normalized) or ".." in parts:
        raise ValueError(f"结果 file 不是安全相对路径: {value}")
    return "/".join(parts)


def _find_label(labels_dir: Path, audio_name: str, explicit: Optional[Path]) -> Path:
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(f"标注文件不存在: {explicit}")
        return explicit
    relative = _safe_relative_path(audio_name)
    audio_path = (labels_dir / Path(relative.replace("/", "\\"))).resolve()
    labels_root = labels_dir.resolve()
    try:
        audio_path.relative_to(labels_root)
    except ValueError as exc:
        raise ValueError(f"结果音频路径越出测试集目录: {audio_name}") from exc
    candidates = [audio_path.with_name(f"{audio_path.stem}_label.txt")]
    # raw8ch 音频通常是由原始音频追加后缀生成，兼容原始音频的 label 命名。
    if audio_path.stem.lower().endswith("_raw8ch"):
        source_stem = audio_path.stem[:-len("_raw8ch")]
        candidates.append(audio_path.with_name(f"{source_stem}_label.txt"))
    candidates.append(labels_root / "label.txt")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"未找到 {audio_name} 的同级标注文件: {candidates[0]}")


def _merged_hypothesis(row: dict) -> str:
    segments = row.get("segments") or []
    ordered = sorted(
        (segment for segment in segments if isinstance(segment, dict)),
        key=lambda segment: (int(segment.get("offsetMs") or 0), int(segment.get("durationMs") or 0)),
    )
    return " ".join(str(segment.get("text") or "").strip() for segment in ordered if str(segment.get("text") or "").strip())


def raw8ch_file_id(audio_name: str) -> str:
    """从 raw8ch 音频 basename 提取对外交付使用的文件 ID。"""
    base_name = audio_name.replace("\\", "/").rsplit("/", 1)[-1]
    stem = Path(base_name).stem
    if stem.lower().endswith("_raw8ch"):
        stem = stem[:-len("_raw8ch")]
    if not stem:
        raise ValueError(f"无法从 raw8ch 音频路径提取文件 ID: {audio_name}")
    return stem


def result_file_ids(audio_names: Iterable[str]) -> dict[str, str]:
    """为每条结果分配稳定文件 ID，重复 basename 时避免长音频结果覆盖。"""
    names = list(audio_names)
    base_ids = {audio_name: raw8ch_file_id(audio_name) for audio_name in names}
    counts: dict[str, int] = {}
    for file_id in base_ids.values():
        counts[file_id] = counts.get(file_id, 0) + 1
    return {
        audio_name: file_id if counts[file_id] == 1 else f"{file_id}__{hashlib.sha1(_safe_relative_path(audio_name).encode('utf-8')).hexdigest()[:8]}"
        for audio_name, file_id in base_ids.items()
    }


def _parse_evaluation_detail(path: Path) -> list[EvaluationRow]:
    rows = []
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            value = line.rstrip("\r\n")
            if not value or value.startswith("id\twer\t"):
                continue
            columns = value.split("\t", 6)
            if len(columns) < 6:
                continue
            try:
                reference_count = int(columns[2])
                errors = int(columns[3])
                deletions = int(columns[4])
                insertions = int(columns[5])
            except ValueError as error:
                raise ValueError(f"evaluation.py 输出格式错误: {value}") from error
            rows.append(EvaluationRow(
                columns[0],
                WerStats(
                    reference_count,
                    errors,
                    deletions,
                    insertions,
                    errors - deletions - insertions,
                ),
                columns[6] if len(columns) == 7 else "",
            ))
    return rows


def _run_evaluation(label_path: Path, hypothesis_path: Path, detail_path: Path, language: str) -> list[EvaluationRow]:
    """调用同目录移植的 evaluation.py，避免在本工具中维护第二套整体 WER。"""
    command = [
        sys.executable,
        str(Path(__file__).with_name("evaluation.py")),
        "--label", str(label_path),
        "--hyp", str(hypothesis_path),
        "--language", language.lower(),
        "--metric", "wer",
        "--detail", str(detail_path),
    ]
    result = subprocess.run(
        command,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "evaluation.py 执行失败").strip()
        raise ValueError(f"evaluation.py 执行失败（返回码 {result.returncode}）：{message}")
    rows = _parse_evaluation_detail(detail_path)
    if not rows:
        raise ValueError(f"evaluation.py 未生成有效 WER 明细: {detail_path}")
    return rows


def _sum_evaluation_rows(rows: Iterable[EvaluationRow]) -> WerStats:
    values = list(rows)
    return WerStats(
        sum(item.stats.reference_count for item in values),
        sum(item.stats.errors for item in values),
        sum(item.stats.deletions for item in values),
        sum(item.stats.insertions for item in values),
        sum(item.stats.substitutions for item in values),
    )


_SEGMENT_TIME_RE = re.compile(r"_(\d+)_(\d+)$")
_SEGMENT_ALIGN_TOLERANCE_MS = 500


def _parse_segment_time(segment_id: str) -> tuple[Optional[int], Optional[int]]:
    match = _SEGMENT_TIME_RE.search(segment_id)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def _int_or_none(value: object) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _segment_sort_key(segment: TimedSegment, index: int) -> tuple[int, int]:
    return (segment.offset_ms if segment.offset_ms is not None else sys.maxsize, index)


def _ordered_segments(segments: list[TimedSegment]) -> list[TimedSegment]:
    if not segments or not all(item.offset_ms is not None for item in segments):
        return segments
    return [item for _, item in sorted(enumerate(segments), key=lambda pair: _segment_sort_key(pair[1], pair[0]))]


def _segment_overlap(left: TimedSegment, right: TimedSegment) -> int:
    if left.offset_ms is None or left.end_ms is None or right.offset_ms is None or right.end_ms is None:
        return 0
    return max(0, min(left.end_ms, right.end_ms) - max(left.offset_ms, right.offset_ms))


def _segment_distance(left: TimedSegment, right: TimedSegment) -> int:
    if left.offset_ms is None or left.end_ms is None or right.offset_ms is None or right.end_ms is None:
        return sys.maxsize
    if _segment_overlap(left, right):
        return 0
    if left.end_ms < right.offset_ms:
        return right.offset_ms - left.end_ms
    return left.offset_ms - right.end_ms


def _segment_has_time(segment: TimedSegment) -> bool:
    return segment.offset_ms is not None and segment.end_ms is not None


def _segments_are_time_aligned(left: TimedSegment, right: TimedSegment) -> bool:
    if not _segment_has_time(left) or not _segment_has_time(right):
        return False
    return _segment_distance(left, right) <= _SEGMENT_ALIGN_TOLERANCE_MS


def _time_alignment_groups(
    references: list[TimedSegment], hypotheses: list[TimedSegment]
) -> list[tuple[list[TimedSegment], list[TimedSegment]]]:
    """按时间关系构造连通分组，支持一对多、多对一和多对多。"""
    node_count = len(references) + len(hypotheses)
    parent = list(range(node_count))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for reference_index, reference in enumerate(references):
        for hypothesis_index, hypothesis in enumerate(hypotheses):
            if _segments_are_time_aligned(reference, hypothesis):
                union(reference_index, len(references) + hypothesis_index)

    components: dict[int, tuple[list[int], list[int]]] = {}
    for reference_index in range(len(references)):
        root = find(reference_index)
        components.setdefault(root, ([], []))[0].append(reference_index)
    for hypothesis_index in range(len(hypotheses)):
        root = find(len(references) + hypothesis_index)
        components.setdefault(root, ([], []))[1].append(hypothesis_index)

    def component_start(component: tuple[list[int], list[int]]) -> tuple[int, int]:
        reference_indexes, hypothesis_indexes = component
        segments = [references[index] for index in reference_indexes]
        segments.extend(hypotheses[index] for index in hypothesis_indexes)
        starts = [segment.offset_ms for segment in segments if segment.offset_ms is not None]
        node_indexes = reference_indexes + [len(references) + index for index in hypothesis_indexes]
        return (min(starts) if starts else sys.maxsize, min(node_indexes))

    groups = []
    for reference_indexes, hypothesis_indexes in sorted(components.values(), key=component_start):
        groups.append((
            [references[index] for index in reference_indexes],
            [hypotheses[index] for index in hypothesis_indexes],
        ))
    return groups


def _fallback_alignment_groups(
    references: list[TimedSegment], hypotheses: list[TimedSegment]
) -> list[tuple[list[TimedSegment], list[TimedSegment]]]:
    """没有完整时间戳时按顺序近似分组，避免丢失原始文本。"""
    if not references:
        return [([], hypotheses)] if hypotheses else []
    if not hypotheses:
        return [([reference], []) for reference in references]
    groups: list[tuple[list[TimedSegment], list[TimedSegment]]] = [
        ([reference], []) for reference in references
    ]
    for index, hypothesis in enumerate(hypotheses):
        target = round(index * (len(references) - 1) / max(1, len(hypotheses) - 1))
        groups[target][1].append(hypothesis)
    return groups


def _segment_anchor_index(segment: TimedSegment, anchors: list[TimedSegment], previous: int, index: int, count: int) -> int:
    candidates = list(range(previous, len(anchors)))
    if not candidates:
        return len(anchors) - 1
    if segment.offset_ms is None or not all(anchor.offset_ms is not None for anchor in anchors):
        if count <= 1 or len(anchors) <= 1:
            return candidates[0]
        target = round(index * (len(anchors) - 1) / (count - 1))
        return min(candidates, key=lambda item: abs(item - target))
    return min(
        candidates,
        key=lambda item: (-_segment_overlap(segment, anchors[item]), _segment_distance(segment, anchors[item]), item),
    )


def _status_for_stats(stats: WerStats) -> str:
    if stats.errors == 0:
        return "正确"
    kinds = sum(value > 0 for value in (stats.deletions, stats.insertions, stats.substitutions))
    if kinds > 1:
        return "混合"
    if stats.deletions:
        return "漏识别"
    if stats.insertions:
        return "多识别"
    return "识别错误"


def _hypothesis_segments(audio_name: str, segments: Iterable[object]) -> list[TimedSegment]:
    file_id = raw8ch_file_id(audio_name)
    result = []
    for index, value in enumerate(segments):
        if not isinstance(value, dict):
            continue
        offset = _int_or_none(value.get("offsetMs"))
        duration = _int_or_none(value.get("durationMs"))
        segment_id = str(value.get("segmentId") or value.get("id") or "").strip()
        if not segment_id:
            segment_id = f"{file_id}_{offset if offset is not None else index}_{duration or 0}"
        result.append(TimedSegment(segment_id, str(value.get("text") or "").strip(), offset, duration))
    return _ordered_segments(result)


def _flatten_segment_text(text: str) -> str:
    """将 Excel 展示中的换行转换为 evaluation.py 单行输入。"""
    return re.sub(r"\s+", " ", text).strip()


def _evaluate_segment_groups(
    groups: Iterable[tuple[str, str, str]], language: str
) -> dict[str, EvaluationRow]:
    """批量调用 evaluation.py 计算所有含标注的局部对齐组。"""
    values = list(groups)
    if not values:
        return {}
    with tempfile.TemporaryDirectory(prefix="agent_sdk_segment_wer_") as temp:
        temp_dir = Path(temp)
        label_path = temp_dir / "label.txt"
        hypothesis_path = temp_dir / "hyp.txt"
        detail_path = temp_dir / "wer_detail.txt"
        label_path.write_text(
            "\n".join(f"{group_id} {_flatten_segment_text(reference)}" for group_id, reference, _ in values) + "\n",
            encoding="utf-8",
        )
        hypothesis_path.write_text(
            "\n".join(f"{group_id} {_flatten_segment_text(hypothesis)}".rstrip() for group_id, _, hypothesis in values) + "\n",
            encoding="utf-8",
        )
        return {row.utterance_id: row for row in _run_evaluation(label_path, hypothesis_path, detail_path, language)}


def build_segment_detail_rows(
    labels: list[LabelLine], hypothesis_segments: Iterable[object], language: str, file_id: str
) -> list[SegmentDetailRow]:
    """先按时间形成逻辑对齐组，再复用 evaluation.py 生成局部差异。"""
    references = _ordered_segments([
        TimedSegment(item.utterance_id, item.text, *_parse_segment_time(item.utterance_id)) for item in labels
    ])
    hypotheses = _ordered_segments(_hypothesis_segments(file_id, hypothesis_segments))
    all_have_time = all(_segment_has_time(item) for item in references + hypotheses)
    groups = (
        _time_alignment_groups(references, hypotheses)
        if all_have_time
        else _fallback_alignment_groups(references, hypotheses)
    )

    values = []
    evaluation_inputs = []
    for index, (left, right) in enumerate(groups, 1):
        reference_text = "\n".join(item.text for item in left)
        hypothesis_text = "\n".join(item.text for item in right)
        reference_id = "\n".join(item.segment_id for item in left)
        hypothesis_id = "\n".join(item.segment_id for item in right)
        group_id = f"segment_group_{index:04d}"
        values.append((group_id, reference_id, reference_text, hypothesis_id, hypothesis_text))
        if left:
            evaluation_inputs.append((group_id, reference_text, hypothesis_text))

    evaluations = _evaluate_segment_groups(evaluation_inputs, language)
    rows = []
    for group_id, reference_id, reference_text, hypothesis_id, hypothesis_text in values:
        if reference_id:
            evaluation = evaluations.get(group_id)
            if evaluation is None:
                raise ValueError(f"evaluation.py 缺少局部对齐组结果: {group_id}")
            # 空识别文本不需要根据 evaluation.py 的编辑统计再细分，展示层固定为漏识别。
            status = "漏识别" if not hypothesis_id else _status_for_stats(evaluation.stats)
            diff = evaluation.diff
        else:
            # evaluation.py 不接受空标注；仅识别片段由展示层直接标记为多识别。
            status = "多识别"
            diff = f"(->{hypothesis_text})"
        rows.append(SegmentDetailRow(
            reference_id, reference_text, hypothesis_id, hypothesis_text, status, diff
        ))
    return rows

def _xml_text(value: object) -> str:
    text = str(value).replace("\x00", "")
    text = "".join(char for char in text if char in "\t\n\r" or ord(char) >= 0x20)
    return escape(text)


def write_segment_detail_workbook(output_path: Path, rows: Iterable[SegmentDetailRow]) -> None:
    """用标准库写出可直接由 Excel 打开的六列表格。"""
    headers = ["标注片段ID", "标注ASR文本", "识别片段ID", "识别ASR文本", "对齐状态", "差异文本"]
    values = [headers] + [[
        row.reference_id, row.reference_text, row.hypothesis_id, row.hypothesis_text, row.status, row.diff
    ] for row in rows]
    columns = ["A", "B", "C", "D", "E", "F"]
    sheet_rows = []
    for row_number, values_in_row in enumerate(values, 1):
        cells = []
        for column, value in zip(columns, values_in_row):
            style = 1 if row_number == 1 else 2
            cells.append(f'<c r="{column}{row_number}" t="inlineStr" s="{style}"><is><t xml:space="preserve">{_xml_text(value)}</t></is></c>')
        sheet_rows.append(f'<row r="{row_number}">' + "".join(cells) + "</row>")
    sheet = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>
<cols><col min="1" max="1" width="32"/><col min="2" max="2" width="48"/><col min="3" max="3" width="32"/><col min="4" max="4" width="48"/><col min="5" max="5" width="12"/><col min="6" max="6" width="56"/></cols>
<sheetData>{''.join(sheet_rows)}</sheetData>
</worksheet>'''
    styles = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<fonts count="2"><font><sz val="11"/><name val="等线"/></font><font><b/><sz val="11"/><name val="等线"/></font></fonts>
<fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="solid"><fgColor rgb="D9EAF7"/><bgColor indexed="64"/></patternFill></fill></fills>
<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
<cellXfs count="3"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/><xf numFmtId="0" fontId="1" fillId="1" borderId="0" applyAlignment="1"><alignment horizontal="center" vertical="center" wrapText="1"/></xf><xf numFmtId="0" fontId="0" fillId="0" borderId="0" applyAlignment="1"><alignment vertical="top" wrapText="1"/></xf></cellXfs>
</styleSheet>'''
    files = {
        "[Content_Types].xml": '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/></Types>''',
        "_rels/.rels": '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>''',
        "xl/workbook.xml": '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="片段对齐" sheetId="1" r:id="rId1"/></sheets></workbook>''',
        "xl/_rels/workbook.xml.rels": '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>''',
        "xl/worksheets/sheet1.xml": sheet,
        "xl/styles.xml": styles,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content.encode("utf-8"))


def _build_inputs(rows: Iterable[dict], labels_dir: Path, explicit_label: Optional[Path]) -> tuple[list[str], list[str]]:
    entries = []
    label_cache: dict[Path, list[LabelLine]] = {}
    for index, row in enumerate(rows):
        audio_name = str(row.get("file") or row.get("relativePath") or "").strip()
        if not audio_name:
            raise ValueError(f"结果第 {index + 1} 行缺少 file 字段")
        label_path = _find_label(labels_dir, audio_name, explicit_label)
        label_lines = label_cache.setdefault(label_path, _read_label_lines(label_path))
        reference = " ".join(item.text for item in label_lines)
        hypothesis = _merged_hypothesis(row)
        entries.append((label_lines[0].utterance_id, reference, hypothesis, audio_name))

    id_counts: dict[str, int] = {}
    for utterance_id, _reference, _hypothesis, _audio_name in entries:
        id_counts[utterance_id] = id_counts.get(utterance_id, 0) + 1
    labels: list[str] = []
    hypotheses: list[str] = []
    used_ids: set[str] = set()
    for utterance_id, reference, hypothesis, audio_name in entries:
        unique_id = utterance_id
        if id_counts[utterance_id] > 1:
            stem = re.sub(r"\s+", "_", Path(audio_name).stem)
            unique_id = f"{utterance_id}_{stem}"
            suffix = 2
            while unique_id in used_ids:
                unique_id = f"{utterance_id}_{stem}_{suffix}"
                suffix += 1
        used_ids.add(unique_id)
        labels.append(f"{unique_id} {reference}")
        hypotheses.append(f"{unique_id} {hypothesis}".rstrip())
    return labels, hypotheses


def _write_per_audio_details(
    rows: Iterable[dict],
    labels_dir: Path,
    explicit_label: Optional[Path],
    output_dir: Path,
    language: str,
    write_segment_detail: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    values = list(rows)
    file_ids = result_file_ids(
        str(row.get("file") or row.get("relativePath") or "").strip() for row in values
    )
    for index, row in enumerate(values):
        audio_name = str(row.get("file") or row.get("relativePath") or "").strip()
        if not audio_name:
            raise ValueError(f"结果第 {index + 1} 行缺少 file 字段")
        file_id = file_ids[audio_name]
        detail_path = output_dir / f"{file_id}_wer_detail.txt"
        try:
            label_lines = _read_label_lines(_find_label(labels_dir, audio_name, explicit_label))
            reference = " ".join(item.text for item in label_lines)
            hypothesis = _merged_hypothesis(row)
            with tempfile.TemporaryDirectory(prefix="agent_sdk_wer_") as temp:
                temp_dir = Path(temp)
                label_path = temp_dir / "label.txt"
                hypothesis_path = temp_dir / "hyp.txt"
                label_path.write_text(f"{file_id} {reference}\n", encoding="utf-8")
                hypothesis_path.write_text(f"{file_id} {hypothesis}\n", encoding="utf-8")
                _run_evaluation(label_path, hypothesis_path, detail_path, language)
            if write_segment_detail:
                segment_rows = build_segment_detail_rows(
                    label_lines, row.get("segments") or [], language, file_id
                )
                write_segment_detail_workbook(
                    output_dir / f"{file_id}_segment_asr_detail.xlsx", segment_rows
                )
        except (OSError, ValueError) as error:
            detail_path.write_text(f"WER 未执行: {error}\n", encoding="utf-8")


def _write_segment_detail_files(
    rows: Iterable[dict], labels_dir: Path, explicit_label: Optional[Path], output_dir: Path, language: str
) -> None:
    values = list(rows)
    file_ids = result_file_ids(
        str(row.get("file") or row.get("relativePath") or "").strip() for row in values
    )
    for index, row in enumerate(values):
        audio_name = str(row.get("file") or row.get("relativePath") or "").strip()
        if not audio_name:
            raise ValueError(f"结果第 {index + 1} 行缺少 file 字段")
        file_id = file_ids[audio_name]
        label_lines = _read_label_lines(_find_label(labels_dir, audio_name, explicit_label))
        segment_rows = build_segment_detail_rows(label_lines, row.get("segments") or [], language, file_id)
        write_segment_detail_workbook(output_dir / f"{file_id}_segment_asr_detail.xlsx", segment_rows)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True, metavar="目录", help="ASR 结果目录（必填）")
    parser.add_argument("--labels-dir", type=Path, required=True, metavar="目录", help="测试集和 label 所在目录（必填）")
    parser.add_argument("--label", type=Path, metavar="文件", help="单个 label 文件（默认：按音频同级 label 文件自动查找）")
    parser.add_argument("--output-dir", type=Path, required=True, metavar="目录", help="WER 输出目录（必填）")
    parser.add_argument("--language", default="zh", metavar="zh", help="分词语言（默认：zh）")
    parser.add_argument("--per-audio-detail", action="store_true", help="为每个 raw8ch 音频生成独立 WER 明细（默认：关闭）")
    parser.add_argument("--segment-detail", action="store_true", help="为每个音频生成片段对齐 Excel（默认：关闭）")
    args = parser.parse_args(argv)
    try:
        result_file = args.results_dir / "asr_results.jsonl"
        if not result_file.is_file():
            result_file = args.results_dir / ".asr_results.jsonl"
        rows = _read_rows(result_file)
        if args.per_audio_detail:
            _write_per_audio_details(
                rows, args.labels_dir, args.label, args.output_dir, args.language, args.segment_detail
            )
            return 0
        if args.segment_detail:
            _write_segment_detail_files(rows, args.labels_dir, args.label, args.output_dir, args.language)
        labels, hypotheses = _build_inputs(rows, args.labels_dir, args.label)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        label_path = args.output_dir / "wer_label.txt"
        hyp_path = args.output_dir / "wer_hyp.txt"
        detail_path = args.output_dir / "wer_detail.txt"
        summary_path = args.output_dir / "wer_summary.txt"
        label_path.write_text("\n".join(labels) + "\n", encoding="utf-8")
        hyp_path.write_text("\n".join(hypotheses) + "\n", encoding="utf-8")
        evaluation_rows = _run_evaluation(label_path, hyp_path, detail_path, args.language)
        total = _sum_evaluation_rows(evaluation_rows)
        summary_path.write_text(
            f"WER={total.rate * 100:.2f}%\nreference_count={total.reference_count}\nerrors={total.errors}\n"
            f"deletions={total.deletions}\ninsertions={total.insertions}\nsubstitutions={total.substitutions}\n"
            f"wer_label={label_path}\nwer_hyp={hyp_path}\nwer_detail={detail_path}\n",
            encoding="utf-8",
        )
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"WER 处理失败: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
