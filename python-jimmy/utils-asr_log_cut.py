#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cut PCM audio by ASR logs and generate segment WAV files.

This script reads a 16 kHz, mono, int16 PCM file and an ASR log file.
It supports both legacy top1 logs and [ASR] Finished(...) logs.
If parsed segments contain top1 speaker info, segment_id uses 5 parts:
prefix_offsetMs_durationMs_top1speaker_top1score.
Otherwise segment_id uses 3 parts:
prefix_offsetMs_durationMs.

Examples:
python3 ./python-jimmy/utils-asr_log_cut.py asr.log input.pcm

python3 ./python-jimmy/utils-asr_log_cut.py input.pcm asr.log -o ./output

python3 ./python-jimmy/utils-asr_log_cut.py input.pcm asr.log --prefix room24

Notes:
- Input PCM format: 16000Hz / mono / int16
- Outputs: segments.txt, segments/*.wav, wav.scp
- segments.txt format:
  segment_id<TAB>top1_speaker<TAB>top1_score<TAB>asr_text
"""

from __future__ import annotations

import argparse
import re
import wave
from dataclasses import dataclass
from pathlib import Path


SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2
TOP1_RE = re.compile(r"\btop1=([A-Za-z0-9_.-]+)\(([^()]+)\)")
FINISHED_RE = re.compile(r"\[ASR\]\s+Finished\((\d+)/(\d+)\)\s*->\s*(.*)")
SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
MULTI_META_RE = re.compile(r"^(multi=1(?:;[^\s]*)?(?:\s+[^\u4e00-\u9fffA-Za-z0-9].*?)?)\s+(.*)$")


@dataclass(frozen=True)
class AsrSegment:
    offset_ms: int
    duration_ms: int
    name: str
    score: str
    text: str = ""

    @property
    def end_ms(self) -> int:
        return self.offset_ms + self.duration_ms

    @property
    def has_top1(self) -> bool:
        return bool(self.name and self.score)


def sanitize_filename_part(value: str) -> str:
    sanitized = SAFE_NAME_RE.sub("_", value.strip()).strip("._")
    return sanitized or "unknown"


def normalize_segment_prefix(prefix: str) -> str:
    sanitized = sanitize_filename_part(prefix).replace("_", "")
    return sanitized or "audio"


def extract_top1(stripped: str, line_no: int) -> tuple[str, str]:
    match = TOP1_RE.search(stripped)
    if match is None:
        raise ValueError(f"ASR file line {line_no} contains top1= but cannot parse speaker and score")

    name = sanitize_filename_part(match.group(1))
    score = sanitize_filename_part(match.group(2))
    return name, score


def parse_finished_log_line(stripped: str, line_no: int) -> AsrSegment | None:
    match = FINISHED_RE.search(stripped)
    if match is None:
        return None

    offset_ms = int(match.group(1))
    duration_ms = int(match.group(2))
    if offset_ms < 0:
        raise ValueError(f"ASR file line {line_no} offsetMs must be >= 0")
    if duration_ms <= 0:
        raise ValueError(f"ASR file line {line_no} durationMs must be > 0")

    if "top1=" in stripped:
        name, score = extract_top1(stripped, line_no)
    elif is_multi_metadata_line(stripped):
        name, score = "Multi", "NA"
    else:
        name, score = "unknown", "NA"

    tail = match.group(3)
    text_part = tail.split("|extraInfo:", 1)[0].strip()
    if ":" in text_part:
        _, text = text_part.split(":", 1)
    else:
        text = text_part

    return AsrSegment(
        offset_ms=offset_ms,
        duration_ms=duration_ms,
        name=name,
        score=score,
        text=text.strip(),
    )


def parse_plain_log_line(stripped: str, line_no: int) -> AsrSegment:
    parts = stripped.split(maxsplit=3)
    if len(parts) < 3:
        raise ValueError(
            f"ASR file line {line_no} format error, expected at least: offsetMs durationMs speaker/text"
        )

    offset_ms_text, duration_ms_text = parts[0], parts[1]
    try:
        offset_ms = int(offset_ms_text)
    except ValueError as exc:
        raise ValueError(f"ASR file line {line_no} offsetMs is not an integer: {offset_ms_text}") from exc

    try:
        duration_ms = int(duration_ms_text)
    except ValueError as exc:
        raise ValueError(
            f"ASR file line {line_no} durationMs is not an integer: {duration_ms_text}"
        ) from exc

    if offset_ms < 0:
        raise ValueError(f"ASR file line {line_no} offsetMs must be >= 0")
    if duration_ms <= 0:
        raise ValueError(f"ASR file line {line_no} durationMs must be > 0")

    if "top1=" in stripped:
        name, score = extract_top1(stripped, line_no)
        top1_match = TOP1_RE.search(stripped)
        assert top1_match is not None
        text = stripped[top1_match.end():].strip()
    elif is_multi_metadata_line(stripped):
        name, score = "Multi", "NA"
        text = extract_multi_line_text(stripped)
    else:
        name, score = "unknown", "NA"
        text = parts[3].strip() if len(parts) >= 4 else ""

    return AsrSegment(
        offset_ms=offset_ms,
        duration_ms=duration_ms,
        name=name,
        score=score,
        text=text,
    )


def is_multi_metadata_line(stripped: str) -> bool:
    return "multi=1" in stripped


def extract_multi_line_text(stripped: str) -> str:
    parts = stripped.split(maxsplit=3)
    if len(parts) < 4:
        return ""

    meta_and_text = parts[3]
    if "] " in meta_and_text:
        return meta_and_text.split("] ", 1)[1].strip()

    match = MULTI_META_RE.match(meta_and_text)
    if match is not None:
        return match.group(2).strip()

    meta_parts = meta_and_text.split(maxsplit=1)
    if len(meta_parts) == 1:
        return ""
    return meta_parts[1].strip()


def parse_asr_line(line: str, line_no: int) -> AsrSegment | None:
    stripped = line.strip()
    if not stripped:
        return None

    finished_segment = parse_finished_log_line(stripped, line_no)
    if finished_segment is not None:
        return finished_segment

    if re.match(r"^\d+\s+\d+\s+", stripped) is None:
        return None

    return parse_plain_log_line(stripped, line_no)


def read_asr_segments(asr_path: Path) -> list[AsrSegment]:
    segments: list[AsrSegment] = []
    with asr_path.open("r", encoding="utf-8-sig") as file:
        for line_no, raw_line in enumerate(file, start=1):
            segment = parse_asr_line(raw_line, line_no)
            if segment is not None:
                segments.append(segment)
    return segments


def resolve_input_paths(first_path: Path, second_path: Path) -> tuple[Path, Path]:
    first_suffix = first_path.suffix.lower()
    second_suffix = second_path.suffix.lower()

    if first_suffix == ".pcm" and second_suffix != ".pcm":
        return first_path, second_path
    if second_suffix == ".pcm" and first_suffix != ".pcm":
        return second_path, first_path
    if first_suffix == ".pcm" and second_suffix == ".pcm":
        raise ValueError("Two PCM files were provided. Please provide one PCM file and one ASR log file.")

    raise ValueError("Cannot identify PCM file. Please provide one .pcm file and one ASR log file.")


def ms_to_byte_pos(ms_value: int) -> int:
    frame_index = ms_value * SAMPLE_RATE // 1000
    return frame_index * CHANNELS * SAMPLE_WIDTH


def write_wav_file(output_path: Path, audio_bytes: bytes) -> None:
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(SAMPLE_WIDTH)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(audio_bytes)


def use_top1_segment_id(segments: list[AsrSegment]) -> bool:
    return any(segment.has_top1 and segment.score != "NA" for segment in segments)


def build_segment_id(prefix: str, segment: AsrSegment, with_top1: bool) -> str:
    base_id = f"{prefix}_{segment.offset_ms}_{segment.duration_ms}"
    if not with_top1:
        return base_id
    return f"{base_id}_{segment.name}_{segment.score}"


def build_segments_summary_lines(duration_ms_values: list[int]) -> list[str]:
    segment_count = len(duration_ms_values)
    if segment_count == 0:
        return [
            "# segment_count=0",
            "# duration_sec_min=NA duration_sec_max=NA duration_sec_avg=NA",
            "# duration_distribution <3s=0.00% 3-6s=0.00% 6-10s=0.00% 10-20s=0.00% 20-30s=0.00% >=30s=0.00%",
        ]

    bucket_specs = [
        ("<3s", lambda duration_ms: duration_ms < 3000),
        ("3-6s", lambda duration_ms: 3000 <= duration_ms < 6000),
        ("6-10s", lambda duration_ms: 6000 <= duration_ms < 10000),
        ("10-20s", lambda duration_ms: 10000 <= duration_ms < 20000),
        ("20-30s", lambda duration_ms: 20000 <= duration_ms < 30000),
        (">=30s", lambda duration_ms: duration_ms >= 30000),
    ]

    distribution_parts = []
    for bucket_name, predicate in bucket_specs:
        bucket_count = sum(1 for duration_ms in duration_ms_values if predicate(duration_ms))
        bucket_ratio = bucket_count * 100.0 / segment_count
        distribution_parts.append(f"{bucket_name}={bucket_ratio:.2f}%")

    min_duration_sec = min(duration_ms_values) / 1000.0
    max_duration_sec = max(duration_ms_values) / 1000.0
    avg_duration_sec = sum(duration_ms_values) / segment_count / 1000.0

    return [
        f"# segment_count={segment_count}",
        (
            "# duration_sec_min="
            f"{min_duration_sec:.3f} duration_sec_max={max_duration_sec:.3f} "
            f"duration_sec_avg={avg_duration_sec:.3f}"
        ),
        "# duration_distribution " + " ".join(distribution_parts),
    ]


def write_segments_txt(output_path: Path, entries: list[tuple[str, str, str, str, int]]) -> None:
    lines = []
    duration_ms_values = [duration_ms for _, _, _, _, duration_ms in entries]
    lines.extend(build_segments_summary_lines(duration_ms_values))
    lines.append("")

    for segment_id, speaker_name, speaker_score, asr_text, _ in entries:
        normalized_text = asr_text.replace("\t", " ").strip()
        lines.append(f"{segment_id}\t{speaker_name}\t{speaker_score}\t{normalized_text}")
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_wav_scp(output_path: Path, entries: list[tuple[str, Path]]) -> None:
    with output_path.open("w", encoding="utf-8") as file:
        for segment_id, wav_path in entries:
            file.write(f"{segment_id} {wav_path.resolve()}\n")


def cut_pcm_by_asr_log(
    asr_path: Path,
    pcm_path: Path,
    output_dir: Path | None = None,
    min_duration_ms: int = 0,
    prefix: str | None = None,
) -> int:
    if min_duration_ms < 0:
        raise ValueError("min_duration_ms must be >= 0")

    pcm_bytes = pcm_path.read_bytes()
    frame_size = CHANNELS * SAMPLE_WIDTH
    if len(pcm_bytes) % frame_size != 0:
        raise ValueError("PCM file size is not aligned for int16 mono audio")

    segments = read_asr_segments(asr_path)
    base_dir = output_dir if output_dir is not None else pcm_path.parent
    base_dir.mkdir(parents=True, exist_ok=True)
    segments_dir = base_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    segment_prefix = normalize_segment_prefix(prefix or pcm_path.stem)
    segment_id_with_top1 = use_top1_segment_id(segments)

    generated_count = 0
    wav_scp_entries: list[tuple[str, Path]] = []
    segments_txt_entries: list[tuple[str, str, str, str, int]] = []

    for segment in segments:
        if segment.duration_ms < min_duration_ms:
            continue

        start_byte = ms_to_byte_pos(segment.offset_ms)
        end_byte = ms_to_byte_pos(segment.end_ms)
        if start_byte >= len(pcm_bytes):
            print(
                f"Warning: skip segment offsetMs={segment.offset_ms} "
                f"durationMs={segment.duration_ms}, out of PCM range"
            )
            continue

        audio_slice = pcm_bytes[start_byte:min(end_byte, len(pcm_bytes))]
        if not audio_slice:
            print(
                f"Warning: skip segment offsetMs={segment.offset_ms} "
                f"durationMs={segment.duration_ms}, no audio data"
            )
            continue

        segment_id = build_segment_id(segment_prefix, segment, segment_id_with_top1)
        wav_path = segments_dir / f"{segment_id}.wav"
        write_wav_file(wav_path, audio_slice)

        wav_scp_entries.append((segment_id, wav_path))
        segments_txt_entries.append(
            (segment_id, segment.name, segment.score, segment.text, segment.duration_ms)
        )
        generated_count += 1

    write_wav_scp(base_dir / "wav.scp", wav_scp_entries)
    write_segments_txt(base_dir / "segments.txt", segments_txt_entries)
    return generated_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cut 16k int16 mono PCM by ASR log and output standard or top1 segment ids"
    )
    parser.add_argument("input1", help="ASR log path or PCM path")
    parser.add_argument("input2", help="PCM path or ASR log path")
    parser.add_argument(
        "-o",
        "--output-dir",
        help="Output base directory. Default: PCM parent directory",
    )
    parser.add_argument(
        "--min-duration-ms",
        type=int,
        default=0,
        help="Skip segments shorter than this value in ms. Default: 0",
    )
    parser.add_argument(
        "--prefix",
        help="Optional segment_id prefix. Default: PCM file stem",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pcm_path, asr_path = resolve_input_paths(Path(args.input1), Path(args.input2))
    output_dir = Path(args.output_dir) if args.output_dir else None

    if not asr_path.exists():
        raise FileNotFoundError(f"ASR log file not found: {asr_path}")
    if not pcm_path.exists():
        raise FileNotFoundError(f"PCM file not found: {pcm_path}")

    generated_count = cut_pcm_by_asr_log(
        asr_path=asr_path,
        pcm_path=pcm_path,
        output_dir=output_dir,
        min_duration_ms=args.min_duration_ms,
        prefix=args.prefix,
    )

    base_dir = output_dir if output_dir is not None else pcm_path.parent
    print(f"Output directory: {base_dir}")
    print(f"Generated wav files: {generated_count}")


if __name__ == "__main__":
    main()
