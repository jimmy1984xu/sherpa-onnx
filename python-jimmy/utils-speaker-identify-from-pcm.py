#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Identify speakers from a long PCM file using silero VAD and speaker embeddings.

Input requirements:
- PCM format: 16 kHz / mono / int16 / little-endian
- Registered embedding file format:
    <speaker_id> <model_name>:<v1>,<v2>,...,<vn>

Example:
python ./python-jimmy/utils-speaker-identify-from-pcm.py ^
  --registered-embeddings D:/data/registered.txt ^
  --pcm D:/data/input.pcm ^
  --speaker-model D:/models/speaker.onnx ^
  --silero-vad-model D:/models/silero_vad.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np


SAMPLE_RATE = 16000
VAD_THRESHOLD = 0.5
VAD_MIN_SILENCE_DURATION = 0.7
VAD_MIN_SPEECH_DURATION = 0.0
VAD_MAX_SPEECH_DURATION = 25.0
VAD_BUFFER_SIZE_SECONDS = 3600


def require_sherpa_onnx():
    try:
        import sherpa_onnx  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Cannot import sherpa_onnx. Please run this script in an environment "
            "where the sherpa_onnx Python module is installed and importable."
        ) from exc

    return sherpa_onnx


def parse_registered_embedding_line(line: str) -> dict:
    stripped = line.strip()
    if not stripped:
        raise ValueError("embedding line is empty")

    parts = stripped.split(None, 1)
    if len(parts) != 2:
        raise ValueError(f"invalid embedding line: {line}")

    speaker_id, payload = parts
    if ":" not in payload:
        raise ValueError(f"missing model prefix: {line}")

    model_name, values_str = payload.split(":", 1)
    if not model_name.strip():
        raise ValueError(f"empty model name: {line}")

    values = [float(item) for item in values_str.split(",") if item.strip()]
    if not values:
        raise ValueError(f"empty embedding values: {line}")

    embedding = np.asarray(values, dtype=np.float32)
    return {
        "speaker_id": speaker_id,
        "model_name": model_name.strip(),
        "embedding": embedding,
    }


def load_registered_embeddings(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"registered embedding file not found: {path}")

    records = []
    seen_speakers = set()
    with open(path, "r", encoding="utf-8") as file:
        for line_num, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = parse_registered_embedding_line(stripped)
            speaker_id = record["speaker_id"]
            if speaker_id in seen_speakers:
                raise ValueError(
                    f"duplicate speaker_id in registered embeddings: {speaker_id} at line {line_num}"
                )
            seen_speakers.add(speaker_id)
            records.append(record)

    if not records:
        raise ValueError(f"no valid registered embeddings found in {path}")

    return records


def pcm_s16le_bytes_to_float32(data: bytes) -> np.ndarray:
    if len(data) % 2 != 0:
        raise ValueError("PCM byte length must be even for int16 samples")

    samples_int16 = np.frombuffer(data, dtype="<i2")
    samples = samples_int16.astype(np.float32) / 32768.0
    return np.ascontiguousarray(samples)


def read_pcm_file(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"pcm file not found: {path}")
    data = path.read_bytes()
    return pcm_s16le_bytes_to_float32(data)


def format_compact_result_line(
    offset_ms: int,
    duration_ms: int,
    speaker_id: str,
    score: float,
) -> str:
    return f"{offset_ms} {duration_ms} {speaker_id} {score:.6f}"


def format_verbose_result_line(
    offset_ms: int,
    duration_ms: int,
    speaker_id: str,
    score: float,
) -> str:
    return (
        f"offset={offset_ms}ms duration={duration_ms}ms "
        f"top1={speaker_id} score={score:.6f}"
    )


def build_extractor(
    model_path: str,
    num_threads: int,
    provider: str,
    debug: bool,
):
    sherpa_onnx = require_sherpa_onnx()
    config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=model_path,
        num_threads=num_threads,
        provider=provider,
        debug=debug,
    )
    if not config.validate():
        raise ValueError(f"invalid speaker embedding extractor config: {config}")

    return sherpa_onnx.SpeakerEmbeddingExtractor(config)


def build_manager(registered_records: list[dict]):
    sherpa_onnx = require_sherpa_onnx()
    dim = int(registered_records[0]["embedding"].shape[0])
    manager = sherpa_onnx.SpeakerEmbeddingManager(dim)

    for record in registered_records:
        speaker_id = record["speaker_id"]
        embedding = record["embedding"]
        if int(embedding.shape[0]) != dim:
            raise ValueError(
                f"embedding dimension mismatch for {speaker_id}: "
                f"{embedding.shape[0]} != {dim}"
            )
        if not manager.add(speaker_id, embedding):
            raise RuntimeError(f"failed to register speaker: {speaker_id}")

    return manager, dim


def build_silero_vad(vad_model_path: str):
    sherpa_onnx = require_sherpa_onnx()
    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = vad_model_path
    config.silero_vad.threshold = VAD_THRESHOLD
    config.silero_vad.min_silence_duration = VAD_MIN_SILENCE_DURATION
    config.silero_vad.min_speech_duration = VAD_MIN_SPEECH_DURATION
    config.silero_vad.max_speech_duration = VAD_MAX_SPEECH_DURATION
    config.sample_rate = SAMPLE_RATE

    if not config.validate():
        raise ValueError("invalid silero VAD config")

    vad = sherpa_onnx.VoiceActivityDetector(
        config,
        buffer_size_in_seconds=VAD_BUFFER_SIZE_SECONDS,
    )
    return vad, int(config.silero_vad.window_size)


def compute_embedding(extractor, samples: np.ndarray) -> Optional[np.ndarray]:
    stream = extractor.create_stream()
    stream.accept_waveform(sample_rate=SAMPLE_RATE, waveform=samples)
    stream.input_finished()

    if not extractor.is_ready(stream):
        return None

    embedding = extractor.compute(stream)
    return np.asarray(embedding, dtype=np.float32)


def find_top1_speaker(manager, embedding: np.ndarray) -> tuple[str, float]:
    best_speaker = ""
    best_score = float("-inf")

    for speaker_id in manager.all_speakers:
        score = float(manager.score(speaker_id, embedding))
        if score > best_score:
            best_speaker = speaker_id
            best_score = score

    return best_speaker, best_score


def process_vad_segment(extractor, manager, dim: int, segment) -> Optional[dict]:
    samples = np.asarray(segment.samples, dtype=np.float32)
    if samples.size == 0:
        return None

    embedding = compute_embedding(extractor, samples)
    if embedding is None or embedding.size == 0:
        return None

    if int(embedding.shape[0]) != dim:
        raise ValueError(
            f"segment embedding dimension mismatch: {embedding.shape[0]} != {dim}"
        )

    top1_speaker, top1_score = find_top1_speaker(manager, embedding)
    offset_ms = int(segment.start / SAMPLE_RATE * 1000)
    duration_ms = int(len(samples) / SAMPLE_RATE * 1000)
    return {
        "offset_ms": offset_ms,
        "duration_ms": duration_ms,
        "top1_speaker": top1_speaker,
        "top1_score": top1_score,
    }


def print_segment_result(result: dict) -> None:
    print(
        format_compact_result_line(
            result["offset_ms"],
            result["duration_ms"],
            result["top1_speaker"],
            result["top1_score"],
        )
    )
    print(
        format_verbose_result_line(
            result["offset_ms"],
            result["duration_ms"],
            result["top1_speaker"],
            result["top1_score"],
        )
    )


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Identify speaker segments from a long PCM file using silero VAD.",
    )
    parser.add_argument(
        "--registered-embeddings",
        type=str,
        required=True,
        help="registered speaker embedding file",
    )
    parser.add_argument(
        "--pcm",
        type=str,
        required=True,
        help="input PCM file (16k/mono/int16/s16le)",
    )
    parser.add_argument(
        "--speaker-model",
        type=str,
        required=True,
        help="speaker embedding model path",
    )
    parser.add_argument(
        "--silero-vad-model",
        type=str,
        required=True,
        help="silero_vad.onnx path",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="number of inference threads",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "coreml"],
        help="inference provider",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="enable sherpa-onnx debug logs",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    registered_path = Path(args.registered_embeddings)
    pcm_path = Path(args.pcm)
    speaker_model_path = Path(args.speaker_model)
    silero_vad_model_path = Path(args.silero_vad_model)

    if not speaker_model_path.is_file():
        raise FileNotFoundError(f"speaker model not found: {speaker_model_path}")
    if not silero_vad_model_path.is_file():
        raise FileNotFoundError(f"silero vad model not found: {silero_vad_model_path}")

    print("[1/4] Loading registered speaker embeddings")
    registered_records = load_registered_embeddings(registered_path)
    print(f"      registered speakers: {len(registered_records)}")

    print("[2/4] Building speaker manager and extractor")
    manager, dim = build_manager(registered_records)
    extractor = build_extractor(
        model_path=str(speaker_model_path),
        num_threads=args.num_threads,
        provider=args.provider,
        debug=args.debug,
    )
    print(f"      embedding_dim={dim}")

    print("[3/4] Loading PCM and building silero VAD")
    pcm_samples = read_pcm_file(pcm_path)
    vad, window_size = build_silero_vad(str(silero_vad_model_path))
    print(f"      pcm_samples={len(pcm_samples)} window_size={window_size}")
    print(
        "      silero_vad fixed params: "
        f"threshold={VAD_THRESHOLD}, "
        f"min_silence_duration={VAD_MIN_SILENCE_DURATION}, "
        f"max_speech_duration={VAD_MAX_SPEECH_DURATION}"
    )

    print("[4/4] Running VAD and speaker matching")
    result_count = 0
    for start in range(0, len(pcm_samples), window_size):
        vad.accept_waveform(pcm_samples[start : start + window_size])

        while not vad.empty():
            result = process_vad_segment(extractor, manager, dim, vad.front)
            vad.pop()
            if result is None:
                continue
            print_segment_result(result)
            result_count += 1

    vad.flush()
    while not vad.empty():
        result = process_vad_segment(extractor, manager, dim, vad.front)
        vad.pop()
        if result is None:
            continue
        print_segment_result(result)
        result_count += 1

    print(f"\nDone! matched segments: {result_count}")


if __name__ == "__main__":
    main()
