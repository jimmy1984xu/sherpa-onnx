#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cut ASR segments, extract speaker embeddings, and cluster speakers in one file.

This script reads a 16 kHz mono int16 PCM file plus an ASR log file, cuts segment
WAV files, extracts speaker embeddings with sherpa-onnx, runs fast clustering,
and exports a cluster summary plus clustered WAV folders.

Examples:
python ./python-jimmy/utils-asr_speaker_cluster_all_in_one.py input.pcm input.log ^
  --model D:/TransAI/audio_models/speaker/nemo_en_titanet_large.onnx ^
  --prefix 25

python ./python-jimmy/utils-asr_speaker_cluster_all_in_one.py input.log input.pcm ^
  --model D:/TransAI/audio_models/speaker/nemo_en_titanet_large.onnx ^
  --output-dir ./output ^
  --min-duration-ms 3000 ^
  --threshold 0.65 ^
  --num-threads 4

Notes:
- Input PCM format: 16000 Hz / mono / int16.
- Outputs: segments/, segments.txt, wav.scp, nemo-embedding.txt,
  cluster summary txt, and clustered WAV folders.
- If ASR log contains top1 speaker info, segment_id uses:
  prefix_offsetMs_durationMs_top1speaker_top1score.
"""

from __future__ import annotations

import argparse
import base64
import re
import shutil
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2
DEFAULT_NUM_CLUSTERS = -1
DEFAULT_THRESHOLD = 0.65
DEFAULT_MIN_DURATION_MS = 3000

REGISTERED_PRIMARY_RATIO = 0.70
REGISTERED_PRIMARY_AVG_THRESHOLD = 0.60
REGISTERED_PRIMARY_MAX_THRESHOLD = 0.70
REGISTERED_FALLBACK_RATIO = 0.80
REGISTERED_FALLBACK_AVG_THRESHOLD = 0.57
REGISTERED_FALLBACK_MAX_THRESHOLD = 0.68
REGISTERED_MIN_COUNT = 3
UNKNOWN_MIN_SEGMENTS = 10
UNKNOWN_MAX_SCORE_THRESHOLD = 0.55
SHORT_SEGMENT_KEEP_SCORE_THRESHOLD = 0.65

TOP1_RE = re.compile(r"\btop1=([A-Za-z0-9_.-]+)\(([^()]+)\)")
FINISHED_RE = re.compile(r"\[ASR\]\s+Finished\((\d+)/(\d+)\)\s*->\s*(.*)")
SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
MULTI_META_RE = re.compile(r"^(multi=1(?:;[^\s]*)?(?:\s+[^\u4e00-\u9fffA-Za-z0-9].*?)?)\s+(.*)$")


@dataclass(frozen=True)
class AsrSegment:
    offset_ms: int
    duration_ms: int
    speaker: str
    score: str
    text: str = ""

    @property
    def end_ms(self) -> int:
        return self.offset_ms + self.duration_ms

    @property
    def has_top1(self) -> bool:
        return bool(self.speaker and self.score and self.score != "NA")


def sanitize_filename_part(value: str) -> str:
    sanitized = SAFE_NAME_RE.sub("_", value.strip()).strip("._")
    return sanitized or "unknown"


def normalize_segment_prefix(prefix: str) -> str:
    sanitized = sanitize_filename_part(prefix).replace("_", "")
    return sanitized or "audio"


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


def extract_top1(stripped: str, line_no: int) -> tuple[str, str]:
    match = TOP1_RE.search(stripped)
    if match is None:
        raise ValueError(f"ASR file line {line_no} contains top1= but cannot parse speaker and score")
    return sanitize_filename_part(match.group(1)), sanitize_filename_part(match.group(2))


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
        speaker, score = extract_top1(stripped, line_no)
    elif is_multi_metadata_line(stripped):
        speaker, score = "Multi", "NA"
    else:
        speaker, score = "unknown", "NA"
    text_part = match.group(3).split("|extraInfo:", 1)[0].strip()
    text = text_part.split(":", 1)[1] if ":" in text_part else text_part
    return AsrSegment(offset_ms, duration_ms, speaker, score, text.strip())


def parse_plain_log_line(stripped: str, line_no: int) -> AsrSegment:
    parts = stripped.split(maxsplit=3)
    if len(parts) < 3:
        raise ValueError(f"ASR file line {line_no} format error, expected: offsetMs durationMs text")

    try:
        offset_ms = int(parts[0])
        duration_ms = int(parts[1])
    except ValueError as exc:
        raise ValueError(f"ASR file line {line_no} offsetMs/durationMs is not an integer") from exc

    if offset_ms < 0:
        raise ValueError(f"ASR file line {line_no} offsetMs must be >= 0")
    if duration_ms <= 0:
        raise ValueError(f"ASR file line {line_no} durationMs must be > 0")

    if "top1=" in stripped:
        speaker, score = extract_top1(stripped, line_no)
        top1_match = TOP1_RE.search(stripped)
        assert top1_match is not None
        text = stripped[top1_match.end():].strip()
    elif is_multi_metadata_line(stripped):
        speaker, score = "Multi", "NA"
        text = extract_multi_line_text(stripped)
    else:
        speaker, score = "unknown", "NA"
        text = parts[3].strip() if len(parts) >= 4 else ""

    return AsrSegment(offset_ms, duration_ms, speaker, score, text)


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
    if not segments:
        raise ValueError(f"No ASR segments parsed from: {asr_path}")
    return segments


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
    return any(segment.has_top1 for segment in segments)


def build_segment_id(prefix: str, segment: AsrSegment, with_top1: bool) -> str:
    base_id = f"{prefix}_{segment.offset_ms}_{segment.duration_ms}"
    if not with_top1:
        return base_id
    return f"{base_id}_{segment.speaker}_{segment.score}"


def build_segments_summary_lines(duration_ms_values: list[int]) -> list[str]:
    if not duration_ms_values:
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

    segment_count = len(duration_ms_values)
    distribution_parts = []
    for bucket_name, predicate in bucket_specs:
        bucket_count = sum(1 for duration_ms in duration_ms_values if predicate(duration_ms))
        bucket_ratio = bucket_count * 100.0 / segment_count
        distribution_parts.append(f"{bucket_name}={bucket_ratio:.2f}%")

    return [
        f"# segment_count={segment_count}",
        (
            "# duration_sec_min="
            f"{min(duration_ms_values) / 1000.0:.3f} "
            f"duration_sec_max={max(duration_ms_values) / 1000.0:.3f} "
            f"duration_sec_avg={sum(duration_ms_values) / segment_count / 1000.0:.3f}"
        ),
        "# duration_distribution " + " ".join(distribution_parts),
    ]


def write_segments_txt(output_path: Path, entries: list[tuple[str, str, str, str, int]]) -> None:
    lines = build_segments_summary_lines([item[4] for item in entries])
    lines.append("")
    for segment_id, speaker_name, speaker_score, asr_text, _ in entries:
        lines.append(f"{segment_id}\t{speaker_name}\t{speaker_score}\t{asr_text.replace(chr(9), ' ').strip()}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_wav_scp(output_path: Path, entries: list[tuple[str, Path]]) -> None:
    with output_path.open("w", encoding="utf-8") as file:
        for segment_id, wav_path in entries:
            file.write(f"{segment_id} {wav_path.resolve()}\n")


def cut_pcm_by_asr_log(
    asr_path: Path,
    pcm_path: Path,
    output_dir: Path,
    prefix: str | None,
) -> tuple[Path, list[tuple[str, Path]]]:
    print("[1/4] Cutting PCM by ASR log")
    pcm_bytes = pcm_path.read_bytes()
    frame_size = CHANNELS * SAMPLE_WIDTH
    if len(pcm_bytes) % frame_size != 0:
        raise ValueError("PCM file size is not aligned for int16 mono audio")

    segments = read_asr_segments(asr_path)
    segments_dir = output_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    segment_prefix = normalize_segment_prefix(prefix or pcm_path.stem)
    segment_id_with_top1 = use_top1_segment_id(segments)

    wav_scp_entries: list[tuple[str, Path]] = []
    segments_txt_entries: list[tuple[str, str, str, str, int]] = []
    for segment in segments:
        start_byte = ms_to_byte_pos(segment.offset_ms)
        end_byte = ms_to_byte_pos(segment.end_ms)
        if start_byte >= len(pcm_bytes):
            print(f"      Warning: skip out-of-range segment offset={segment.offset_ms} duration={segment.duration_ms}")
            continue

        audio_slice = pcm_bytes[start_byte:min(end_byte, len(pcm_bytes))]
        if not audio_slice:
            print(f"      Warning: skip empty segment offset={segment.offset_ms} duration={segment.duration_ms}")
            continue

        segment_id = build_segment_id(segment_prefix, segment, segment_id_with_top1)
        wav_path = segments_dir / f"{segment_id}.wav"
        write_wav_file(wav_path, audio_slice)
        wav_scp_entries.append((segment_id, wav_path))
        segments_txt_entries.append((segment_id, segment.speaker, segment.score, segment.text, segment.duration_ms))

    wav_scp_path = output_dir / "wav.scp"
    write_wav_scp(wav_scp_path, wav_scp_entries)
    write_segments_txt(output_dir / "segments.txt", segments_txt_entries)
    print(f"      Generated {len(wav_scp_entries)} segment WAV files")
    return wav_scp_path, wav_scp_entries


def load_wav_mono_float32(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width != 2:
        raise ValueError(f"Only int16 WAV is supported: {path}")

    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(samples), sample_rate


def resample_linear(samples: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return samples
    duration = samples.shape[0] / float(src_sr)
    dst_len = int(round(duration * dst_sr))
    if dst_len <= 0:
        return np.zeros((0,), dtype=np.float32)
    x_src = np.linspace(0.0, duration, num=samples.shape[0], endpoint=False, dtype=np.float64)
    x_dst = np.linspace(0.0, duration, num=dst_len, endpoint=False, dtype=np.float64)
    return np.interp(x_dst, x_src, samples.astype(np.float64)).astype(np.float32, copy=False)


def build_extractor(model: str, num_threads: int, provider: str, debug: bool):
    import sherpa_onnx

    cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=model,
        num_threads=num_threads,
        debug=debug,
        provider=provider,
    )
    if not cfg.validate():
        raise ValueError(f"Invalid SpeakerEmbeddingExtractorConfig: {cfg}")
    return sherpa_onnx.SpeakerEmbeddingExtractor(cfg)


def extract_embedding_for_wav(extractor, wav_path: Path, max_duration: float | None) -> np.ndarray | None:
    samples, sample_rate = load_wav_mono_float32(wav_path)
    if sample_rate != SAMPLE_RATE:
        samples = resample_linear(samples, sample_rate, SAMPLE_RATE)

    if max_duration is not None and max_duration > 0:
        max_samples = int(max_duration * SAMPLE_RATE)
        if len(samples) > max_samples:
            samples = samples[:max_samples]

    stream = extractor.create_stream()
    stream.accept_waveform(sample_rate=SAMPLE_RATE, waveform=samples)
    stream.input_finished()
    if not extractor.is_ready(stream):
        return None
    return np.asarray(extractor.compute(stream), dtype=np.float32)


def serialize_embedding(embedding: np.ndarray) -> str:
    return base64.b64encode(embedding.astype(np.float32).tobytes()).decode("ascii")


def extract_embeddings_to_file(
    wav_entries: list[tuple[str, Path]],
    model_path: Path,
    output_path: Path,
    num_threads: int,
    provider: str,
    debug: bool,
    max_audio_duration: float,
) -> list[tuple[str, str]]:
    print("[2/4] Extracting speaker embeddings")
    extractor = build_extractor(str(model_path), num_threads, provider, debug)
    print(f"      Extractor ready. embedding_dim={extractor.dim}")

    max_duration = max_audio_duration if max_audio_duration > 0 else None
    results: list[tuple[str, str]] = []
    failed_count = 0
    start_all = time.time()
    for index, (segment_id, wav_path) in enumerate(wav_entries, start=1):
        try:
            embedding = extract_embedding_for_wav(extractor, wav_path, max_duration)
        except Exception as exc:
            print(f"      Warning: failed to extract {segment_id}: {exc}")
            embedding = None

        if embedding is None or embedding.size == 0:
            failed_count += 1
            continue

        results.append((segment_id, serialize_embedding(embedding)))
        if index % max(1, len(wav_entries) // 20) == 0 or index == len(wav_entries):
            print(f"      Progress: {index}/{len(wav_entries)} ({100.0 * index / len(wav_entries):.1f}%)", end="\r")

    print()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for segment_id, embedding_text in results:
            file.write(f"{segment_id} {embedding_text}\n")

    print(f"      Successfully extracted {len(results)} embeddings")
    if failed_count:
        print(f"      Failed: {failed_count}")
    print(f"      Embedding file: {output_path}")
    print(f"      Elapsed: {time.time() - start_all:.3f}s")
    return results


def deserialize_embedding(embedding_text: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(embedding_text.encode("ascii")), dtype=np.float32)


def create_skip_counters() -> dict[str, int]:
    return {
        "multi": 0,
        "short": 0,
        "invalid_embedding": 0,
    }


def parse_offset_duration(segment_id: str) -> tuple[int, int]:
    parts = segment_id.split("_")
    if len(parts) >= 5:
        try:
            return int(parts[-4]), int(parts[-3])
        except ValueError:
            pass
    if len(parts) >= 3:
        try:
            return int(parts[-2]), int(parts[-1])
        except ValueError:
            pass
    numbers = re.findall(r"\d+", segment_id)
    if len(numbers) >= 2:
        return int(numbers[-2]), int(numbers[-1])
    return 0, 0


def parse_old_speaker_score(segment_id: str) -> tuple[Optional[str], Optional[float]]:
    parts = segment_id.split("_")
    if len(parts) < 5:
        return None, None
    try:
        int(parts[-4])
        int(parts[-3])
    except ValueError:
        return None, None
    if parts[-1] == "NA":
        return parts[-2], None
    try:
        score = float(parts[-1])
    except ValueError:
        return None, None
    return parts[-2], score


def load_segments_from_embeddings(embedding_entries: list[tuple[str, str]]) -> list[dict]:
    segments: list[dict] = []
    for segment_id, embedding_text in embedding_entries:
        offset_ms, duration_ms = parse_offset_duration(segment_id)
        speaker, _ = parse_old_speaker_score(segment_id)
        segments.append(
            {
                "id": segment_id,
                "offsetMs": offset_ms,
                "durationMs": duration_ms,
                "speaker": speaker,
                "embedding": embedding_text,
            }
        )
    if not segments:
        raise ValueError("No embeddings are available for clustering")
    return segments


def filter_segments_for_clustering(
    segments: list[dict],
    min_duration_ms: int,
    skip_counters: Optional[dict[str, int]] = None,
) -> list[dict]:
    filtered: list[dict] = []
    for segment in segments:
        if segment.get("speaker") == "Multi":
            if skip_counters is not None:
                skip_counters["multi"] += 1
            continue
        if segment.get("durationMs", 0) < min_duration_ms:
            _, old_score = parse_old_speaker_score(segment.get("id", ""))
            if old_score is None or old_score <= SHORT_SEGMENT_KEEP_SCORE_THRESHOLD:
                if skip_counters is not None:
                    skip_counters["short"] += 1
                continue
        filtered.append(segment)
    return filtered


def extract_embedding_array(segments: list[dict]) -> tuple[np.ndarray, list[int]]:
    embeddings = []
    valid_indices = []
    for index, segment in enumerate(segments):
        try:
            embeddings.append(deserialize_embedding(segment["embedding"]))
            valid_indices.append(index)
        except Exception as exc:
            print(f"      Warning: skip invalid embedding for {segment.get('id')}: {exc}")

    if not embeddings:
        raise ValueError("No valid embeddings found for clustering")
    return np.array(embeddings, dtype=np.float32), valid_indices


def perform_fast_clustering(
    segments: list[dict],
    embeddings_array: np.ndarray,
    valid_indices: list[int],
    num_clusters: int,
    threshold: float,
) -> list[tuple[str, int]]:
    import sherpa_onnx

    print("[3/4] Running fast clustering")
    if num_clusters > 0:
        print(f"      Mode: num_clusters={num_clusters}")
        config = sherpa_onnx.FastClusteringConfig(num_clusters=num_clusters)
    else:
        print(f"      Mode: threshold={threshold:g}")
        config = sherpa_onnx.FastClusteringConfig(threshold=threshold)

    clustering = sherpa_onnx.FastClustering(config)
    start_time = time.time()
    cluster_labels = clustering(embeddings_array)
    elapsed_time = time.time() - start_time

    unique_labels = sorted(set(cluster_labels))
    label_to_speaker = {label: index + 1 for index, label in enumerate(unique_labels)}
    results: list[tuple[str, int]] = []
    for valid_index, cluster_label in zip(valid_indices, cluster_labels):
        results.append((segments[valid_index]["id"], label_to_speaker[cluster_label]))

    print(f"      Clustering done: {len(unique_labels)} speakers, {len(results)} segments, {elapsed_time:.3f}s")
    return results


def format_score(value: float) -> str:
    return f"{value:.3f}"


def summarize_scores(scores: list[float]) -> str:
    if not scores:
        return "N/A"
    return (
        f"min={format_score(min(scores))},"
        f"max={format_score(max(scores))},"
        f"avg={format_score(sum(scores) / len(scores))}"
    )


def build_cluster_name_map(results: list[tuple[str, int]]) -> dict[int, str]:
    cluster_to_segments: dict[int, list[str]] = {}
    for segment_id, cluster_id in results:
        cluster_to_segments.setdefault(cluster_id, []).append(segment_id)

    name_map: dict[int, str] = {}
    unknown_index = 1
    for cluster_id in sorted(cluster_to_segments):
        segment_ids = cluster_to_segments[cluster_id]
        cluster_count = len(segment_ids)
        speaker_scores: dict[str, list[float]] = {}
        all_scores: list[float] = []

        for segment_id in segment_ids:
            old_speaker, old_score = parse_old_speaker_score(segment_id)
            if old_speaker is None or old_score is None:
                continue
            speaker_scores.setdefault(old_speaker, []).append(old_score)
            all_scores.append(old_score)

        best_name: Optional[str] = None
        best_count = -1
        best_avg = -1.0
        best_max = -1.0
        for speaker, scores in speaker_scores.items():
            speaker_count = len(scores)
            speaker_avg = sum(scores) / speaker_count
            speaker_max = max(scores)
            speaker_ratio = speaker_count / cluster_count
            is_registered = speaker_count >= REGISTERED_MIN_COUNT and (
                (
                    speaker_ratio > REGISTERED_PRIMARY_RATIO
                    and speaker_avg >= REGISTERED_PRIMARY_AVG_THRESHOLD
                    and speaker_max >= REGISTERED_PRIMARY_MAX_THRESHOLD
                )
                or (
                    speaker_ratio > REGISTERED_FALLBACK_RATIO
                    and speaker_avg >= REGISTERED_FALLBACK_AVG_THRESHOLD
                    and speaker_max >= REGISTERED_FALLBACK_MAX_THRESHOLD
                )
            )
            if is_registered and (
                speaker_count > best_count
                or (speaker_count == best_count and speaker_avg > best_avg)
                or (speaker_count == best_count and speaker_avg == best_avg and speaker_max > best_max)
            ):
                best_name = speaker
                best_count = speaker_count
                best_avg = speaker_avg
                best_max = speaker_max

        if best_name is not None:
            name_map[cluster_id] = best_name
        elif cluster_count > UNKNOWN_MIN_SEGMENTS and all_scores and max(all_scores) > UNKNOWN_MAX_SCORE_THRESHOLD:
            name_map[cluster_id] = f"S{unknown_index}"
            unknown_index += 1
        else:
            name_map[cluster_id] = "invalid"

    return name_map


def format_float_for_name(value: float) -> str:
    return f"{value:g}"


def build_output_tag(num_clusters: int, threshold: float, min_duration_ms: int) -> str:
    if num_clusters > 0:
        return f"k-{num_clusters}_{min_duration_ms}"
    return f"th-{format_float_for_name(threshold)}_{min_duration_ms}"


def build_cluster_summary_lines(
    results: list[tuple[str, int]],
    cluster_name_map: dict[int, str],
    input_segment_count: int,
    min_duration_ms: int,
    num_clusters: int,
    threshold: float,
    skipped_multi_segments: int = 0,
    skipped_short_segments: int = 0,
    skipped_invalid_embedding_segments: int = 0,
) -> list[str]:
    mode_text = f"num_clusters={num_clusters}" if num_clusters > 0 else f"threshold={format_float_for_name(threshold)}"
    cluster_to_segments: dict[int, list[str]] = {}
    for segment_id, cluster_id in results:
        cluster_to_segments.setdefault(cluster_id, []).append(segment_id)

    valid_speakers: list[str] = []
    for cluster_id in sorted(cluster_to_segments):
        cluster_name = cluster_name_map.get(cluster_id, str(cluster_id))
        if cluster_name == "invalid":
            continue
        if cluster_name not in valid_speakers:
            valid_speakers.append(cluster_name)

    lines = [
        f"# mode: {mode_text}",
        f"# min_duration_ms: {min_duration_ms}",
        f"# input_segments: {input_segment_count}",
        f"# skipped_multi_segments: {skipped_multi_segments}",
        f"# skipped_short_segments: {skipped_short_segments}",
        f"# skipped_invalid_embedding_segments: {skipped_invalid_embedding_segments}",
        f"# clustered_segments: {len(results)}",
        f"# cluster_number: {len(cluster_to_segments)}",
        f"# valid_speakers: {','.join(valid_speakers)}",
        "# rules:",
        (
            "# rule_registered_primary: "
            f"count>={REGISTERED_MIN_COUNT} and ratio>{REGISTERED_PRIMARY_RATIO:.2f} "
            f"and avg_score>={REGISTERED_PRIMARY_AVG_THRESHOLD:.2f} "
            f"and max_score>={REGISTERED_PRIMARY_MAX_THRESHOLD:.2f}"
        ),
        (
            "# rule_registered_fallback: "
            f"count>={REGISTERED_MIN_COUNT} and ratio>{REGISTERED_FALLBACK_RATIO:.2f} "
            f"and avg_score>={REGISTERED_FALLBACK_AVG_THRESHOLD:.2f} "
            f"and max_score>={REGISTERED_FALLBACK_MAX_THRESHOLD:.2f}"
        ),
        (
            "# rule_unknown: "
            f"if no registered speaker matches, assign S<n> when cluster_count>{UNKNOWN_MIN_SEGMENTS} "
            f"and cluster_max_score>{UNKNOWN_MAX_SCORE_THRESHOLD:.2f}"
        ),
        "# rule_invalid: if neither registered nor unknown rule matches, mark cluster as invalid",
        "# rule_tiebreak: prefer larger count, then larger avg_score, then larger max_score",
    ]

    for cluster_id in sorted(cluster_to_segments):
        segment_ids = cluster_to_segments[cluster_id]
        old_speaker_scores: dict[str, list[float]] = {}
        for segment_id in segment_ids:
            old_speaker, old_score = parse_old_speaker_score(segment_id)
            if old_speaker is None or old_score is None:
                continue
            old_speaker_scores.setdefault(old_speaker, []).append(old_score)

        cluster_name = cluster_name_map.get(cluster_id, str(cluster_id))
        if old_speaker_scores:
            per_speaker_parts = []
            for speaker, scores in sorted(old_speaker_scores.items(), key=lambda item: (-len(item[1]), item[0])):
                per_speaker_parts.append(f"{speaker}(count={len(scores)},{summarize_scores(scores)})")
            lines.append(f"# cluster_{cluster_id}: {cluster_name} {len(segment_ids)} " + "; ".join(per_speaker_parts))
        else:
            lines.append(f"# cluster_{cluster_id}: {cluster_name} {len(segment_ids)} N/A")

    return lines


def export_cluster_summary(
    results: list[tuple[str, int]],
    output_path: Path,
    input_segment_count: int,
    min_duration_ms: int,
    num_clusters: int,
    threshold: float,
    skipped_multi_segments: int = 0,
    skipped_short_segments: int = 0,
    skipped_invalid_embedding_segments: int = 0,
) -> None:
    cluster_name_map = build_cluster_name_map(results)
    summary_lines = build_cluster_summary_lines(
        results=results,
        cluster_name_map=cluster_name_map,
        input_segment_count=input_segment_count,
        min_duration_ms=min_duration_ms,
        num_clusters=num_clusters,
        threshold=threshold,
        skipped_multi_segments=skipped_multi_segments,
        skipped_short_segments=skipped_short_segments,
        skipped_invalid_embedding_segments=skipped_invalid_embedding_segments,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for line in summary_lines:
            file.write(f"{line}\n")
        for segment_id, cluster_id in sorted(results, key=lambda item: (item[1], item[0])):
            file.write(f"{cluster_id} {segment_id}\n")


def export_cluster_audio_files(results: list[tuple[str, int]], wav_entries: list[tuple[str, Path]], output_dir: Path) -> None:
    wav_map = {segment_id: wav_path for segment_id, wav_path in wav_entries}
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    missing = 0
    for segment_id, cluster_id in results:
        wav_path = wav_map.get(segment_id)
        if wav_path is None or not wav_path.is_file():
            missing += 1
            print(f"      Warning: missing WAV for segment: {segment_id}")
            continue
        cluster_dir = output_dir / str(cluster_id)
        cluster_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(wav_path, cluster_dir / wav_path.name)
        copied += 1
    print(f"      Copied {copied} WAV files to {output_dir}")
    if missing:
        print(f"      Missing WAV files: {missing}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run ASR cut, speaker embedding extraction, and clustering in one standalone script",
    )
    parser.add_argument("input1", help="ASR log path or PCM path")
    parser.add_argument("input2", help="PCM path or ASR log path")
    parser.add_argument("--model", required=True, help="Path to speaker embedding model (.onnx)")
    parser.add_argument("--prefix", help="Optional segment_id prefix. Default: PCM file stem")
    parser.add_argument("--output-dir", help="Output base directory. Default: PCM parent directory")
    parser.add_argument("--min-duration-ms", type=int, default=DEFAULT_MIN_DURATION_MS, help="Minimum duration for clustering input")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Fast clustering threshold")
    parser.add_argument("--num-clusters", type=int, default=DEFAULT_NUM_CLUSTERS, help="Fixed cluster count. If > 0, threshold is ignored")
    parser.add_argument("--num-threads", type=int, default=4, help="Speaker embedding extraction threads")
    parser.add_argument("--provider", default="cpu", choices=["cpu", "cuda", "coreml"], help="ONNX Runtime provider")
    parser.add_argument("--debug", action="store_true", help="Enable extractor debug logs")
    parser.add_argument("--max-audio-duration", type=float, default=10.0, help="Maximum audio duration in seconds for embedding extraction")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_duration_ms < 0:
        raise ValueError("--min-duration-ms must be >= 0")

    pcm_path, asr_path = resolve_input_paths(Path(args.input1), Path(args.input2))
    if not pcm_path.is_file():
        raise FileNotFoundError(f"PCM file not found: {pcm_path}")
    if not asr_path.is_file():
        raise FileNotFoundError(f"ASR log file not found: {asr_path}")

    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    output_dir = Path(args.output_dir) if args.output_dir else pcm_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    wav_scp_path, wav_entries = cut_pcm_by_asr_log(
        asr_path=asr_path,
        pcm_path=pcm_path,
        output_dir=output_dir,
        prefix=args.prefix,
    )
    embedding_path = output_dir / "nemo-embedding.txt"
    embedding_entries = extract_embeddings_to_file(
        wav_entries=wav_entries,
        model_path=model_path,
        output_path=embedding_path,
        num_threads=args.num_threads,
        provider=args.provider,
        debug=args.debug,
        max_audio_duration=args.max_audio_duration,
    )

    segments = load_segments_from_embeddings(embedding_entries)
    input_segment_count = len(segments)
    skip_counters = create_skip_counters()
    filtered_segments = filter_segments_for_clustering(
        segments,
        args.min_duration_ms,
        skip_counters=skip_counters,
    )
    print(f"      Clustering input segments: {len(filtered_segments)} / {len(segments)}")
    if not filtered_segments:
        raise ValueError("No segments left for clustering after duration filtering")

    embeddings_array, valid_indices = extract_embedding_array(filtered_segments)
    results = perform_fast_clustering(
        segments=filtered_segments,
        embeddings_array=embeddings_array,
        valid_indices=valid_indices,
        num_clusters=args.num_clusters,
        threshold=args.threshold,
    )

    print("[4/4] Exporting cluster results")
    tag = build_output_tag(args.num_clusters, args.threshold, args.min_duration_ms)
    cluster_summary_path = output_dir / f"cluster-{tag}.txt"
    cluster_audio_dir = output_dir / f"clustered_wavs_{tag}"
    export_cluster_summary(
        results=results,
        output_path=cluster_summary_path,
        input_segment_count=input_segment_count,
        min_duration_ms=args.min_duration_ms,
        num_clusters=args.num_clusters,
        threshold=args.threshold,
        skipped_multi_segments=skip_counters["multi"],
        skipped_short_segments=skip_counters["short"],
        skipped_invalid_embedding_segments=skip_counters["invalid_embedding"],
    )
    export_cluster_audio_files(results, wav_entries, cluster_audio_dir)

    print()
    print("All-in-one pipeline completed.")
    print(f"Output directory: {output_dir}")
    print(f"wav.scp: {wav_scp_path}")
    print(f"Embedding file: {embedding_path}")
    print(f"Cluster summary: {cluster_summary_path}")
    print(f"Cluster audio directory: {cluster_audio_dir}")


if __name__ == "__main__":
    main()
