#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge lag-aligned angle samples into ASR transcript lines.

Input angle txt:
- One angle value per line
- Each line represents 32 ms

Input ASR txt:
- Each line format:
  offsetMs durationMs speakerID top1=speaker(score) asr_text

Output:
- Preserve the ASR prefix and insert the merged angle sequence before ASR text
- Example:
  3392 1344 0 top1=NathanSun(0.356) [3 10 25] 这两天你最近。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


ANGLE_FRAME_MS = 32
ANGLE_LAG_MS = 320
ANGLE_MERGE_TOLERANCE = 5
ASR_PREFIX_RE = re.compile(r"^(\d+\s+\d+\s+\S+\s+(?:top1=\S+|-))(?:\s+(.*))?$")


def read_angle_values(angle_path: Path) -> list[str]:
    values: list[str] = []
    with angle_path.open("r", encoding="utf-8-sig") as file:
        for line_no, raw_line in enumerate(file, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            values.append(parse_angle_value(stripped, line_no))
    return values


def parse_angle_value(value: str, line_no: int) -> str:
    try:
        angle_value = float(value)
    except ValueError as exc:
        raise ValueError(f"Angle file line {line_no} is not a valid number: {value}") from exc

    if angle_value < 0 or angle_value > 360:
        raise ValueError(f"Angle file line {line_no} must be in [0, 360]: {value}")

    if angle_value.is_integer():
        return str(int(angle_value))
    return value


def compress_angle_groups(values: list[tuple[str, int]], merge_tolerance: float) -> list[tuple[str, int]]:
    compressed: list[tuple[str, int]] = []
    for value, duration_ms in values:
        if duration_ms <= 0:
            continue
        if not compressed:
            compressed.append((value, duration_ms))
            continue

        last_value, last_duration_ms = compressed[-1]
        if abs(float(value) - float(last_value)) <= merge_tolerance:
            compressed[-1] = (last_value, last_duration_ms + duration_ms)
        else:
            compressed.append((value, duration_ms))
    return compressed


def build_angle_timeline_segments(
    angle_values: list[str], merge_tolerance: float
) -> list[tuple[int, int, str]]:
    if not angle_values:
        return []

    segments: list[tuple[int, int, str]] = []
    current_value = angle_values[0]
    start_index = 0

    for index, value in enumerate(angle_values[1:], start=1):
        if abs(float(value) - float(current_value)) <= merge_tolerance:
            continue

        segments.append(
            (start_index * ANGLE_FRAME_MS, index * ANGLE_FRAME_MS, current_value)
        )
        current_value = value
        start_index = index

    segments.append(
        (start_index * ANGLE_FRAME_MS, len(angle_values) * ANGLE_FRAME_MS, current_value)
    )
    return segments


def write_angle_segments_txt(
    angle_values: list[str], output_path: Path, merge_tolerance: float
) -> None:
    segments = build_angle_timeline_segments(angle_values, merge_tolerance)
    lines = [
        f"{start_ms} {end_ms} {end_ms - start_ms} {angle_value}"
        for start_ms, end_ms, angle_value in segments
    ]
    output_path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def sample_angles_for_segment(
    angle_values: list[str], offset_ms: int, duration_ms: int, merge_tolerance: float
) -> list[tuple[str, int]]:
    if duration_ms <= 0:
        raise ValueError(f"duration_ms must be > 0, got: {duration_ms}")
    if offset_ms < 0:
        raise ValueError(f"offset_ms must be >= 0, got: {offset_ms}")

    start_ms = offset_ms + ANGLE_LAG_MS
    end_ms = start_ms + duration_ms
    selected: list[tuple[str, int]] = []

    for index, value in enumerate(angle_values):
        sample_start_ms = index * ANGLE_FRAME_MS
        sample_end_ms = sample_start_ms + ANGLE_FRAME_MS

        if sample_end_ms <= start_ms:
            continue
        if sample_start_ms >= end_ms:
            break

        overlap_start_ms = max(sample_start_ms, start_ms)
        overlap_end_ms = min(sample_end_ms, end_ms)
        overlap_duration_ms = overlap_end_ms - overlap_start_ms
        if overlap_duration_ms > 0:
            selected.append((value, overlap_duration_ms))

    return compress_angle_groups(selected, merge_tolerance)


def format_angle_groups(angle_groups: list[tuple[str, int]]) -> str:
    if not angle_groups:
        return "[]"

    parts = [f"{angle_value}:{duration_ms}" for angle_value, duration_ms in angle_groups]
    return "[" + " ".join(parts) + "]"


def merge_angle_line(
    line: str, angle_values: list[str], line_no: int, merge_tolerance: float
) -> str:
    stripped = line.rstrip("\r\n")
    newline = line[len(stripped):] or "\n"
    if not stripped:
        return line

    match = ASR_PREFIX_RE.match(stripped)
    if match is None:
        return line

    prefix = match.group(1)
    text = (match.group(2) or "").strip()
    prefix_parts = prefix.split(maxsplit=2)
    offset_ms = int(prefix_parts[0])
    duration_ms = int(prefix_parts[1])

    merged_angles = sample_angles_for_segment(
        angle_values, offset_ms, duration_ms, merge_tolerance
    )
    angle_text = format_angle_groups(merged_angles)

    if text:
        return f"{prefix} {angle_text} {text}{newline}"
    return f"{prefix} {angle_text}{newline}"


def merge_angle_into_asr(
    angle_path: Path,
    asr_path: Path,
    output_path: Path,
    merge_tolerance: float = ANGLE_MERGE_TOLERANCE,
) -> None:
    if merge_tolerance < 0:
        raise ValueError(f"merge_tolerance must be >= 0, got: {merge_tolerance}")

    angle_values = read_angle_values(angle_path)

    merged_lines: list[str] = []
    with asr_path.open("r", encoding="utf-8-sig") as input_file:
        for line_no, line in enumerate(input_file, start=1):
            merged_lines.append(
                merge_angle_line(line, angle_values, line_no, merge_tolerance)
            )

    output_path.write_text("".join(merged_lines), encoding="utf-8")
    write_angle_segments_txt(
        angle_values,
        angle_path.with_name(f"{angle_path.stem}.segments{angle_path.suffix}"),
        merge_tolerance,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge angle values into ASR text with 320ms lag alignment"
    )
    parser.add_argument("angle_txt", help="Path to angle txt file, one angle per line")
    parser.add_argument("asr_txt", help="Path to ASR txt file")
    parser.add_argument(
        "-o",
        "--output",
        help="Output ASR txt path. Default: <input>.angle.txt",
    )
    parser.add_argument(
        "--merge-tolerance",
        type=float,
        default=ANGLE_MERGE_TOLERANCE,
        help="Merge adjacent angles when their difference is within this value. Default: 5",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    angle_path = Path(args.angle_txt)
    asr_path = Path(args.asr_txt)
    output_path = (
        Path(args.output)
        if args.output
        else asr_path.with_name(f"{asr_path.stem}.angle{asr_path.suffix}")
    )

    if not angle_path.exists():
        raise FileNotFoundError(f"Angle txt file not found: {angle_path}")
    if not asr_path.exists():
        raise FileNotFoundError(f"ASR txt file not found: {asr_path}")

    merge_angle_into_asr(
        angle_path,
        asr_path,
        output_path,
        merge_tolerance=args.merge_tolerance,
    )
    print(f"Output file: {output_path}")


if __name__ == "__main__":
    main()
