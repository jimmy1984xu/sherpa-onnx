#!/usr/bin/env python3
"""VAD + streaming pyannote speaker-count cuts + ASR + global speaker clustering.

Unlike ``offline-long-audio-pipeline-asr-speaker.py``, this entry point uses the
object-level ``SpeakerSegmentation`` API.  The returned local activity does not
identify global speakers: it is used only for VAD-internal cut boundaries and
for marking overlap regions before the existing embedding/clustering stage.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

PIPELINE_DIR = Path(__file__).with_name("offline-long-audio-pipeline")
sys.path.insert(0, str(PIPELINE_DIR))

from asr import transcribe_segments
from audio_io import AudioFormat, load_audio, write_segment_wav
from models import build_runtimes
from output import create_run_directory, write_metadata, write_results
from pipeline import (
    ASR_ENGINE_PARAFORMER,
    ASR_ENGINE_WHISPER,
    DEFAULT_ASR_DIR,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SEGMENTATION_DIR,
    DEFAULT_SPEAKER_DIR,
    DEFAULT_VAD_DIR,
    DEFAULT_WHISPER_URL,
)
from speaker import assign_speaker_ids_with_centroids
from vad import SpeechSegment, collect_vad_segments
from whisper_asr import BILINGUAL_MIN_TEXT_CONFIDENCE, WhisperClientConfig, parse_whisper_languages, transcribe_segments_with_whisper

SAMPLE_RATE = 16000
CONTINUE = 0
SPEAKER_COUNT_CHANGED = 1
SINGLE_SPEAKER_CHANGED = 2
INPUT_FINISHED = 4
SINGLE_SPEAKER = "single_speaker"
OVERLAPPED_SPEAKERS = "overlapped_speakers"


def _resolve_audio_format(
    audio: Path, requested: str, sample_rate: int, channels: int, sample_width: int
) -> AudioFormat:
    kind = requested if requested != "auto" else ("pcm" if audio.suffix.lower() == ".pcm" else "wav")
    return AudioFormat(kind=kind, sample_rate=sample_rate, channels=channels, sample_width=sample_width)


def _span_value(span: Mapping[str, Any] | object, name: str) -> Any:
    return span[name] if isinstance(span, Mapping) else getattr(span, name)


def _normalise_spans(spans: Sequence[Mapping[str, Any] | object]) -> list[dict[str, float | int]]:
    result = [
        {
            "start": float(_span_value(span, "start")),
            "end": float(_span_value(span, "end")),
            "speaker_count": int(_span_value(span, "speaker_count")),
            "flag": int(_span_value(span, "flag")),
        }
        for span in spans
    ]
    result.sort(key=lambda span: (float(span["start"]), float(span["end"])))
    previous_end = 0.0
    for index, span in enumerate(result):
        start = float(span["start"])
        end = float(span["end"])
        count = int(span["speaker_count"])
        if not 0 <= count <= 2 or end <= start:
            raise ValueError(f"Invalid speaker-segmentation span at index {index}")
        if index and start < previous_end - 1e-6:
            raise ValueError("Speaker-segmentation spans must not overlap")
        previous_end = max(previous_end, end)
    return result


def _count_for_interval(spans: Sequence[dict[str, float | int]], start_ms: int, end_ms: int) -> int:
    midpoint = (start_ms + end_ms) / 2000.0
    for span in spans:
        if float(span["start"]) <= midpoint < float(span["end"]):
            return int(span["speaker_count"])
    return 0


def _right_boundary_reason(span: Mapping[str, float | int] | None, current_count: int, next_count: int) -> str | None:
    if span is not None:
        flag = int(span["flag"])
        if flag & SINGLE_SPEAKER_CHANGED:
            return "speaker_change"
        if flag & SPEAKER_COUNT_CHANGED:
            return "speaker_count"
    if current_count != next_count:
        return "speaker_count"
    return None


def _drain_speaker_segmentation(segmenter: object) -> list[dict[str, float | int]]:
    spans: list[dict[str, float | int]] = []
    while not segmenter.empty():
        span = segmenter.front
        spans.append(
            {
                "start": float(span.start),
                "end": float(span.end),
                "speaker_count": int(span.speaker_count),
                "flag": int(span.flag),
            }
        )
        segmenter.pop()
    return spans


def stream_speaker_segmentation_samples(
    samples: np.ndarray,
    *,
    model: Path,
    chunk_ms: int,
    num_threads: int,
    min_duration_on: float,
    min_duration_off: float,
    change_vote_threshold: float,
    api: object | None = None,
) -> list[dict[str, float | int]]:
    """Run the new object-level API over an in-memory 16 kHz waveform.

    ``api`` is injectable only for the no-model unit test; production imports
    the built sherpa_onnx module here, rather than importing an old installed
    extension during CLI/help or timeline-only tests.
    """
    if chunk_ms <= 0 or SAMPLE_RATE * chunk_ms % 1000:
        raise ValueError("segmentation chunk_ms must contain an integral positive sample count")
    if api is None:
        import sherpa_onnx as api

    pyannote = api.OfflineSpeakerSegmentationPyannoteModelConfig(str(model))
    model_config = api.OfflineSpeakerSegmentationModelConfig(
        pyannote=pyannote, num_threads=num_threads, debug=False, provider="cpu"
    )
    config = api.SpeakerSegmentationConfig(
        model=model_config,
        min_duration_on=min_duration_on,
        min_duration_off=min_duration_off,
        change_vote_threshold=change_vote_threshold,
    )
    if not config.validate():
        raise ValueError(f"invalid SpeakerSegmentationConfig: {config}")
    segmenter = api.SpeakerSegmentation(config)
    if segmenter.sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"SpeakerSegmentation sample rate must be {SAMPLE_RATE}, got {segmenter.sample_rate}"
        )

    waveform = np.ascontiguousarray(np.asarray(samples, dtype=np.float32))
    chunk_samples = SAMPLE_RATE * chunk_ms // 1000
    spans: list[dict[str, float | int]] = []
    for start in range(0, waveform.size, chunk_samples):
        segmenter.accept_waveform(waveform[start : start + chunk_samples].tolist())
        spans.extend(_drain_speaker_segmentation(segmenter))
    segmenter.input_finished()
    spans.extend(_drain_speaker_segmentation(segmenter))

    from streaming_segmentation import normalize_spans

    return normalize_spans(spans)

def resolve_vad_segments_with_speaker_segmentation(
    vad_segments: Sequence[SpeechSegment],
    spans: Sequence[Mapping[str, Any] | object],
    waveform: np.ndarray,
    sample_rate: int,
) -> list[SpeechSegment]:
    """Split VAD speech only on stable speaker-count/change span boundaries.

    A count of zero inside a VAD speech region falls back to one speaker, so an
    uncertain segmentation frame cannot delete VAD-confirmed speech.  Local
    track indices are deliberately never read or exposed.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    timeline = _normalise_spans(spans)
    samples = np.ascontiguousarray(np.asarray(waveform, dtype=np.float32))
    output: list[SpeechSegment] = []

    for vad_segment in sorted(vad_segments, key=lambda item: (item.start_ms, item.end_ms)):
        if vad_segment.end_ms <= vad_segment.start_ms:
            continue
        relevant = [
            span for span in timeline
            if float(span["start"]) * 1000 < vad_segment.end_ms
            and vad_segment.start_ms < float(span["end"]) * 1000
        ]
        points = {vad_segment.start_ms, vad_segment.end_ms}
        for span in relevant:
            points.add(max(vad_segment.start_ms, int(round(float(span["start"]) * 1000))))
            points.add(min(vad_segment.end_ms, int(round(float(span["end"]) * 1000))))
        ordered = sorted(point for point in points if vad_segment.start_ms <= point <= vad_segment.end_ms)
        atomic: list[dict[str, Any]] = []
        for start_ms, end_ms in zip(ordered, ordered[1:]):
            if end_ms <= start_ms:
                continue
            count = _count_for_interval(relevant, start_ms, end_ms)
            # VAD is a hard speech boundary.  Segmentation silence within it is
            # intentionally treated as uncertain one-speaker speech.
            composition = OVERLAPPED_SPEAKERS if count == 2 else SINGLE_SPEAKER
            ending_span = next(
                (span for span in relevant if abs(float(span["end"]) * 1000 - end_ms) <= 1),
                None,
            )
            atomic.append(
                {
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "count": count,
                    "composition": composition,
                    "ending_span": ending_span,
                }
            )

        merged: list[dict[str, Any]] = []
        for index, item in enumerate(atomic):
            next_count = int(atomic[index + 1]["count"]) if index + 1 < len(atomic) else int(item["count"])
            reason = _right_boundary_reason(item["ending_span"], int(item["count"]), next_count)
            item["right_reason"] = reason
            if (
                merged
                and merged[-1]["end_ms"] == item["start_ms"]
                and merged[-1]["composition"] == item["composition"]
                and merged[-1]["right_reason"] is None
            ):
                merged[-1]["end_ms"] = item["end_ms"]
                merged[-1]["count"] = item["count"]
                merged[-1]["right_reason"] = reason
            else:
                merged.append(item)

        for index, item in enumerate(merged):
            start_ms = int(item["start_ms"])
            end_ms = int(item["end_ms"])
            start_sample = max(0, math.floor(start_ms * sample_rate / 1000))
            end_sample = min(samples.size, math.ceil(end_ms * sample_rate / 1000))
            if end_sample <= start_sample:
                continue
            left_reason = "vad" if index == 0 else str(merged[index - 1]["right_reason"] or "speaker_count")
            right_reason = "vad" if index == len(merged) - 1 else str(item["right_reason"] or "speaker_count")
            is_overlap = item["composition"] == OVERLAPPED_SPEAKERS
            output.append(
                SpeechSegment(
                    segment_index=len(output) + 1,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    samples=np.ascontiguousarray(samples[start_sample:end_sample]),
                    speaker_composition=str(item["composition"]),
                    overlap_regions=[(start_ms, end_ms)] if is_overlap else [],
                    cut_left=left_reason,
                    cut_right=right_reason,
                    pyannote_mask=f"speaker_count={int(item['count'])}",
                )
            )
    if any(previous.end_ms > current.start_ms for previous, current in zip(output, output[1:])):
        raise AssertionError("Resolved speaker-segmentation output overlaps")
    return output


def _composition_counts(segments: Sequence[SpeechSegment]) -> dict[str, int]:
    return {
        SINGLE_SPEAKER: sum(segment.speaker_composition == SINGLE_SPEAKER for segment in segments),
        OVERLAPPED_SPEAKERS: sum(segment.speaker_composition == OVERLAPPED_SPEAKERS for segment in segments),
    }


def _normalise_reference_text(text: str) -> str:
    text = re.sub(r"\([^)]*\)", "", text)
    return "".join(character for character in text if not character.isspace())


def _edit_distance(reference: str, hypothesis: str) -> int:
    row = list(range(len(hypothesis) + 1))
    for reference_index, reference_character in enumerate(reference, start=1):
        next_row = [reference_index]
        for hypothesis_index, hypothesis_character in enumerate(hypothesis, start=1):
            next_row.append(min(
                next_row[-1] + 1,
                row[hypothesis_index] + 1,
                row[hypothesis_index - 1] + (reference_character != hypothesis_character),
            ))
        row = next_row
    return row[-1]


def _result_summary(run_dir: Path, reference: Path | None) -> dict[str, Any]:
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    segments = result.get("segments", [])
    hypothesis = _normalise_reference_text("".join(str(segment.get("asr_text", "")) for segment in segments))
    wer: float | None = None
    if reference is not None:
        reference_text = _normalise_reference_text(reference.read_text(encoding="utf-8"))
        wer = _edit_distance(reference_text, hypothesis) / max(len(reference_text), 1)
    multi_segments = [segment for segment in segments if segment.get("speaker_composition") == OVERLAPPED_SPEAKERS]
    return {
        "wer": wer,
        "segment_count": len(segments),
        "multi_segment_count": len(multi_segments),
        "multi_break_count": sum(
            segment.get("cut_left") != "vad" or segment.get("cut_right") != "vad"
            for segment in multi_segments
        ),
        "multi_ranges": [segment.get("time_range") for segment in multi_segments],
        "wer_note": None if reference is not None else "N/A: no --reference supplied",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run VAD, streaming pyannote speaker-count cuts, ASR, and global speaker clustering.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--audio", required=True, help="Input PCM or WAV file")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--audio-format", choices=("auto", "pcm", "wav"), default="auto")
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--sample-width", type=int, default=2)
    parser.add_argument("--asr-dir", default=str(DEFAULT_ASR_DIR))
    parser.add_argument("--vad-dir", default=str(DEFAULT_VAD_DIR))
    parser.add_argument("--speaker-dir", default=str(DEFAULT_SPEAKER_DIR))
    parser.add_argument("--segmentation-model", default=str(DEFAULT_SEGMENTATION_DIR / "model.onnx"))
    parser.add_argument("--asr-num-threads", type=int, default=1)
    parser.add_argument("--speaker-num-threads", type=int, default=2)
    parser.add_argument("--segmentation-num-threads", type=int, default=4)
    parser.add_argument("--segmentation-chunk-ms", type=int, default=32)
    parser.add_argument("--min-duration-on", type=float, default=0.30)
    parser.add_argument("--min-duration-off", type=float, default=0.50)
    parser.add_argument("--change-vote-threshold", type=float, default=0.50)
    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--min-silence-duration", type=float, default=0.8)
    parser.add_argument("--min-speech-duration", type=float, default=0.25)
    parser.add_argument("--max-speech-duration", type=float, default=25.0)
    parser.add_argument("--pre-speech-pad-duration", type=float, default=0.0)
    parser.add_argument("--cluster-threshold", type=float, default=0.5)
    parser.add_argument("--num-clusters", type=int, default=2)
    parser.add_argument("--centroid-assignment-similarity-threshold", type=float, default=0.5)
    parser.add_argument("--asr-engine", choices=(ASR_ENGINE_PARAFORMER, ASR_ENGINE_WHISPER), default=ASR_ENGINE_PARAFORMER)
    parser.add_argument("--whisper-url", default=DEFAULT_WHISPER_URL)
    parser.add_argument("--whisper-languages", default="")
    parser.add_argument("--whisper-timeout-ms", type=int, default=30000)
    parser.add_argument("--min-text-confidence", type=float, default=BILINGUAL_MIN_TEXT_CONFIDENCE)
    parser.add_argument("--save-segments", action="store_true")
    parser.add_argument("--baseline-run-dir", type=Path, help="Baseline run directory containing result.json")
    parser.add_argument("--reference", type=Path, help="Optional plain-text reference for character WER")
    parser.add_argument("--run-label", default="streaming-speaker-segmentation")
    parser.add_argument("--debug", action="store_true")
    return parser


def run(args: argparse.Namespace) -> Path:
    if args.sample_rate != SAMPLE_RATE:
        raise ValueError("The pyannote segmentation model requires --sample-rate=16000")
    if args.segmentation_chunk_ms <= 0:
        raise ValueError("--segmentation-chunk-ms must be positive")
    if not Path(args.segmentation_model).is_file():
        raise FileNotFoundError(f"Segmentation model not found: {args.segmentation_model}")
    if args.reference is not None and not args.reference.is_file():
        raise FileNotFoundError(f"Reference file not found: {args.reference}")
    if args.baseline_run_dir is not None and not (args.baseline_run_dir / "result.json").is_file():
        raise FileNotFoundError(f"Baseline result.json not found: {args.baseline_run_dir}")

    from streaming_segmentation import write_comparison_report, write_span_artifacts

    audio = Path(args.audio)
    run_dir = create_run_directory(Path(args.output_root), audio.name, run_label=args.run_label)
    started_total = time.perf_counter()
    loaded = load_audio(audio, _resolve_audio_format(audio, args.audio_format, args.sample_rate, args.channels, args.sample_width))
    runtimes = build_runtimes(
        asr_dir=Path(args.asr_dir), vad_dir=Path(args.vad_dir), speaker_dir=Path(args.speaker_dir),
        segmentation_dir=Path(args.segmentation_model).parent, asr_num_threads=args.asr_num_threads,
        speaker_num_threads=args.speaker_num_threads, vad_threshold=args.vad_threshold,
        min_silence_duration=args.min_silence_duration, min_speech_duration=args.min_speech_duration,
        max_speech_duration=args.max_speech_duration, pre_speech_pad_duration=args.pre_speech_pad_duration,
        cluster_threshold=args.cluster_threshold, num_clusters=args.num_clusters, debug=args.debug,
        enable_segmentation=False, enable_local_asr=args.asr_engine == ASR_ENGINE_PARAFORMER,
    )
    vad_started = time.perf_counter()
    raw_vad_segments = collect_vad_segments(runtimes.vad, loaded.samples, runtimes.vad_window_size, loaded.sample_rate)
    vad_seconds = time.perf_counter() - vad_started
    segmentation_started = time.perf_counter()
    spans = stream_speaker_segmentation_samples(
        loaded.samples, model=Path(args.segmentation_model), chunk_ms=args.segmentation_chunk_ms,
        num_threads=args.segmentation_num_threads, min_duration_on=args.min_duration_on,
        min_duration_off=args.min_duration_off, change_vote_threshold=args.change_vote_threshold,
    )
    segmentation_seconds = time.perf_counter() - segmentation_started
    write_span_artifacts(run_dir, audio.stem, loaded.duration_ms / 1000.0, spans)
    resolution_started = time.perf_counter()
    segments = resolve_vad_segments_with_speaker_segmentation(raw_vad_segments, spans, loaded.samples, loaded.sample_rate)
    resolution_seconds = time.perf_counter() - resolution_started
    if args.save_segments:
        for segment in segments:
            write_segment_wav(run_dir / "segments" / f"{segment.segment_id}.wav", segment.samples)
    speaker_started = time.perf_counter()
    embedding_error_count, centroid_assigned, unknown_excluded = assign_speaker_ids_with_centroids(
        runtimes.extractor, segments, cluster_threshold=args.cluster_threshold, num_clusters=args.num_clusters,
        assignment_similarity_threshold=args.centroid_assignment_similarity_threshold,
    )
    speaker_seconds = time.perf_counter() - speaker_started
    asr_started = time.perf_counter()
    if args.asr_engine == ASR_ENGINE_WHISPER:
        transcribe_segments_with_whisper(segments, WhisperClientConfig(
            url=args.whisper_url, timeout_ms=args.whisper_timeout_ms,
            languages=parse_whisper_languages(args.whisper_languages), min_text_confidence=args.min_text_confidence,
        ))
    else:
        if runtimes.recognizer is None:
            raise RuntimeError("Local Paraformer recognizer is required")
        transcribe_segments(runtimes.recognizer, segments, loaded.sample_rate)
    asr_seconds = time.perf_counter() - asr_started
    total_seconds = time.perf_counter() - started_total
    timings = {
        "vad_seconds": vad_seconds, "segmentation_seconds": segmentation_seconds,
        "timeline_resolution_seconds": resolution_seconds, "speaker_seconds": speaker_seconds,
        "asr_seconds": asr_seconds, "total_seconds": total_seconds,
        "rtf": total_seconds / max(loaded.duration_ms / 1000.0, 0.001),
    }
    write_results(run_dir, audio.name, loaded.duration_ms, segments, timings)
    write_metadata(run_dir, {
        "audio": str(audio), "audio_duration_ms": loaded.duration_ms,
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "resolved_models": {role: {"model": str(files.model), "tokens": str(files.tokens) if files.tokens else None} for role, files in runtimes.resolved_files.items()},
        "segmentation_model": str(args.segmentation_model), "raw_vad_segment_count": len(raw_vad_segments),
        "final_asr_segment_count": len(segments), "speaker_composition_counts": _composition_counts(segments),
        "true_overlap_count": sum(segment.speaker_composition == OVERLAPPED_SPEAKERS for segment in segments),
        "asr_error_count": sum(segment.asr_error is not None for segment in segments),
        "embedding_error_count": embedding_error_count, "centroid_assigned_excluded_segment_count": centroid_assigned,
        "unknown_excluded_segment_count": unknown_excluded, "timings": timings,
    })
    if args.baseline_run_dir is not None:
        baseline = _result_summary(args.baseline_run_dir, args.reference)
        candidate = _result_summary(run_dir, args.reference)
        write_comparison_report(run_dir, baseline, candidate)
    return run_dir


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_dir = run(args)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Pipeline failed: {error}", file=sys.stderr)
        return 2
    print(f"Pipeline finished: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
