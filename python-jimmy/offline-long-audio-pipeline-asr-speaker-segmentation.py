#!/usr/bin/env python3
"""Baseline-compatible VAD + SpeakerSegmentation + ASR pipeline.

This is intentionally a companion to ``offline-long-audio-pipeline-asr-speaker.py``.
It delegates the complete VAD, ASR, clustering, output, metadata and timing flow to
that existing pipeline.  Its only functional substitution is the pyannote activity
provider: it uses the object-level streaming ``SpeakerSegmentation`` API and resolves
its ``[start, end, speaker_count, flag]`` spans into the baseline pipeline timeline.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

PIPELINE_DIR = Path(__file__).with_name("offline-long-audio-pipeline")
sys.path.insert(0, str(PIPELINE_DIR))

import pipeline as baseline_pipeline
from diarization import (
    TimelineResolutionStats,
    resolve_final_segments as resolve_baseline_final_segments,
)
from segmentation import SpeakerCountSpan
from models import ResolvedModelFiles
from output import write_metadata
from vad import SpeechSegment
from streaming_segmentation import write_comparison_report, write_span_artifacts

SAMPLE_RATE = 16000
CONTINUE = 0
SINGLE_SPEAKER_CHANGED = 2
INPUT_FINISHED = 4
OVERLAPPED_SPEAKERS = "overlapped_speakers"

_BASELINE_ENTRYPOINT_PATH = Path(__file__).with_name("offline-long-audio-pipeline-asr-speaker.py")
_BASELINE_ENTRYPOINT: object | None = None


def _load_baseline_entrypoint() -> object:
    """Load the unchanged baseline CLI module once for parser compatibility."""
    global _BASELINE_ENTRYPOINT
    if _BASELINE_ENTRYPOINT is None:
        spec = importlib.util.spec_from_file_location(
            "offline_long_audio_pipeline_asr_speaker_baseline", _BASELINE_ENTRYPOINT_PATH
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {_BASELINE_ENTRYPOINT_PATH}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _BASELINE_ENTRYPOINT = module
    return _BASELINE_ENTRYPOINT


def _span_value(span: Mapping[str, Any] | object, name: str) -> Any:
    return span[name] if isinstance(span, Mapping) else getattr(span, name)


def _optional_span_value(span: Mapping[str, Any] | object, name: str, default: Any) -> Any:
    if isinstance(span, Mapping):
        return span.get(name, default)
    return getattr(span, name, default)


def _normalise_spans(spans: Sequence[Mapping[str, Any] | object]) -> list[dict[str, float | int]]:
    result = [
        {
            "start": float(_span_value(span, "start")),
            "end": float(_span_value(span, "end")),
            "speaker_count": int(_span_value(span, "speaker_count")),
            "flag": int(_span_value(span, "flag")),
            "local_speaker_mask": int(_optional_span_value(span, "local_speaker_mask", 0)),
            "local_speaker_mask_confidence": float(
                _optional_span_value(span, "local_speaker_mask_confidence", 0.0)
            ),
        }
        for span in spans
    ]
    result.sort(key=lambda span: (float(span["start"]), float(span["end"])))
    previous_end = 0.0
    for index, span in enumerate(result):
        start = float(span["start"])
        end = float(span["end"])
        count = int(span["speaker_count"])
        mask = int(span["local_speaker_mask"])
        confidence = float(span["local_speaker_mask_confidence"])
        if not 0 <= count <= 2 or end <= start or not 0 <= mask <= 0b111 or not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Invalid speaker-segmentation span at index {index}")
        if index and start < previous_end - 1e-6:
            raise ValueError("Speaker-segmentation spans must not overlap")
        previous_end = max(previous_end, end)
    return result


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
                "local_speaker_mask": int(getattr(span, "local_speaker_mask", 0)),
                "local_speaker_mask_confidence": float(
                    getattr(span, "local_speaker_mask_confidence", 0.0)
                ),
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

def _adapt_spans_to_baseline_activity(
    spans: Sequence[Mapping[str, Any] | object],
) -> list[SpeakerCountSpan]:
    """Map count/change-only spans to the baseline resolver's local-track form.

    The streaming public API deliberately exposes only speaker count plus a
    change flag.  This adapter uses temporary local tracks solely to preserve
    boundary semantics for the established timeline resolver:

    * count ``0`` remains unknown activity, allowing the VAD fallback to
      absorb it rather than creating an ASR-only fragment;
    * count ``2`` is represented as an overlap containing the current and next
      temporary track, preserving exact ``overlap_regions`` metadata;
    * ``SINGLE_SPEAKER_CHANGED`` advances the temporary track after the span.
      The baseline weak-cut policy then suppresses short, unstable changes.

    These identities are implementation-local and are never surfaced to users.
    """
    one_hot_masks = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    activity: list[SpeakerCountSpan] = []
    single_track_index = 0

    for span in _normalise_spans(spans):
        start_ms = int(round(float(span["start"]) * 1000.0))
        end_ms = int(round(float(span["end"]) * 1000.0))
        speaker_count = int(span["speaker_count"])
        if end_ms <= start_ms:
            continue

        local_mask = int(span.get("local_speaker_mask", 0))
        confidence = float(span.get("local_speaker_mask_confidence", 0.0))
        if local_mask and sum((local_mask >> bit) & 1 for bit in range(3)) == speaker_count:
            speaker_mask = tuple((local_mask >> bit) & 1 for bit in range(3))
        elif speaker_count == 0:
            speaker_mask = None
        elif speaker_count == 1:
            speaker_mask = one_hot_masks[single_track_index]
        else:
            next_track_index = (single_track_index + 1) % len(one_hot_masks)
            current_mask = one_hot_masks[single_track_index]
            next_mask = one_hot_masks[next_track_index]
            speaker_mask = tuple(
                int(current_mask[index] or next_mask[index])
                for index in range(len(one_hot_masks[0]))
            )

        activity.append(
            SpeakerCountSpan(
                start_ms=start_ms,
                end_ms=end_ms,
                active_speaker_count=speaker_count,
                speaker_mask=speaker_mask,
                local_speaker_mask_confidence=confidence,
            )
        )
        if int(span["flag"]) & SINGLE_SPEAKER_CHANGED:
            single_track_index = (single_track_index + 1) % len(one_hot_masks)
    return activity


def _overlap_regions_from_count_spans(
    spans: Sequence[Mapping[str, Any] | object], start_ms: int, end_ms: int
) -> list[tuple[int, int]]:
    """Return exact count-two regions after a resolver fold/absorb operation."""
    regions = []
    for span in _normalise_spans(spans):
        if int(span["speaker_count"]) < 2:
            continue
        region_start = max(start_ms, int(round(float(span["start"]) * 1000.0)))
        region_end = min(end_ms, int(round(float(span["end"]) * 1000.0)))
        if region_end > region_start:
            regions.append((region_start, region_end))
    merged: list[tuple[int, int]] = []
    for region in regions:
        if merged and region[0] <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], region[1]))
        else:
            merged.append(region)
    return merged


def _raw_segment_mask_metadata(
    spans: Sequence[Mapping[str, Any] | object], start_ms: int, end_ms: int
) -> tuple[list[tuple[int, int]], int, float]:
    """Return clean portions plus the dominant fused local mask for one ASR turn."""
    clean: list[tuple[int, int]] = []
    weighted: dict[int, tuple[int, float]] = {}
    for span in _normalise_spans(spans):
        if int(span["speaker_count"]) != 1:
            continue
        mask = int(span["local_speaker_mask"])
        confidence = float(span["local_speaker_mask_confidence"])
        if mask not in (1, 2, 4):
            continue
        left = max(start_ms, int(round(float(span["start"]) * 1000.0)))
        right = min(end_ms, int(round(float(span["end"]) * 1000.0)))
        if right <= left:
            continue
        clean.append((left, right))
        duration = right - left
        previous_duration, previous_confidence_sum = weighted.get(mask, (0, 0.0))
        weighted[mask] = (previous_duration + duration, previous_confidence_sum + duration * confidence)
    if not weighted:
        return clean, 0, 0.0
    mask, (duration, confidence_sum) = max(weighted.items(), key=lambda item: item[1][0])
    return clean, mask, confidence_sum / duration if duration else 0.0


def resolve_vad_segments_with_speaker_segmentation(
    vad_segments: Sequence[SpeechSegment],
    spans: Sequence[Mapping[str, Any] | object],
    waveform: np.ndarray,
    sample_rate: int,
    *,
    min_duration_on: float = 0.3,
    min_duration_off: float = 0.5,
) -> list[SpeechSegment]:
    """Resolve streaming count spans with the baseline ASR-safe timeline policy."""
    activity = _adapt_spans_to_baseline_activity(spans)
    segments, _ = resolve_baseline_final_segments(
        vad_segments,
        activity,
        waveform,
        sample_rate,
        min_duration_on=min_duration_on,
        min_duration_off=min_duration_off,
    )
    # A short count-two span can be dropped by activity smoothing before the
    # baseline resolver folds it into a single-speaker ASR sentence.  Recover
    # its exact raw region here so callers retain overlap information without
    # exposing temporary local masks or emitting a short ASR request.
    enriched: list[SpeechSegment] = []
    for segment in segments:
        clean_spans, local_mask, confidence = _raw_segment_mask_metadata(
            spans, segment.start_ms, segment.end_ms
        )
        enriched.append(
            replace(
                segment,
                overlap_regions=_overlap_regions_from_count_spans(
                    spans, segment.start_ms, segment.end_ms
                ),
                clean_spans=clean_spans,
                local_speaker_mask=local_mask,
                local_speaker_mask_confidence=confidence,
            )
        )
    return enriched


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
    """Reuse every baseline CLI option, then add only streaming API options."""
    parser = _load_baseline_entrypoint().build_parser()
    parser.description = (
        "Run the unchanged VAD/ASR/clustering pipeline with the streaming "
        "SpeakerSegmentation activity provider."
    )
    group = parser.add_argument_group("streaming pyannote segmentation")
    group.add_argument(
        "--segmentation-model",
        type=Path,
        default=None,
        help="Explicit model.onnx. Default: --segmentation-dir/model.onnx.",
    )
    group.add_argument("--segmentation-num-threads", type=int, default=4)
    group.add_argument("--segmentation-chunk-ms", type=int, default=32)
    group.add_argument("--min-duration-on", type=float, default=0.30)
    group.add_argument("--min-duration-off", type=float, default=0.50)
    group.add_argument("--change-vote-threshold", type=float, default=0.50)
    group.add_argument(
        "--baseline-run-dir",
        type=Path,
        help="Optional baseline run directory containing result.json for comparison.",
    )
    group.add_argument(
        "--reference", type=Path, help="Optional plain-text reference for character WER."
    )
    return parser


def resolve_segmentation_model(args: argparse.Namespace) -> Path:
    return args.segmentation_model or (Path(args.segmentation_dir) / "model.onnx")


def build_baseline_pipeline_config(args: argparse.Namespace) -> baseline_pipeline.PipelineConfig:
    """Translate the baseline CLI namespace without changing its semantics."""
    if args.segmentation_mode != baseline_pipeline.SEGMENTATION_MODE_VAD_PYANNOTE:
        raise ValueError(
            "This script requires --segmentation-mode=vad-pyannote; "
            "use offline-long-audio-pipeline-asr-speaker.py for --segmentation-mode=vad"
        )
    audio = Path(args.audio)
    model = resolve_segmentation_model(args)
    requested = args.audio_format
    audio_format = _load_baseline_entrypoint()._resolve_audio_format(
        audio, requested, args.sample_rate, args.channels, args.sample_width
    )
    return baseline_pipeline.PipelineConfig(
        audio=audio,
        output_root=Path(args.output_root),
        audio_format=audio_format,
        asr_dir=Path(args.asr_dir),
        vad_dir=Path(args.vad_dir),
        speaker_dir=Path(args.speaker_dir),
        segmentation_dir=model.parent,
        asr_num_threads=args.asr_num_threads,
        speaker_num_threads=args.speaker_num_threads,
        vad_threshold=args.vad_threshold,
        min_silence_duration=args.min_silence_duration,
        min_speech_duration=args.min_speech_duration,
        max_speech_duration=args.max_speech_duration,
        pre_speech_pad_duration=args.pre_speech_pad_duration,
        cluster_threshold=args.cluster_threshold,
        num_clusters=args.num_clusters,
        asr_engine=args.asr_engine,
        whisper_url=args.whisper_url,
        whisper_languages=tuple(
            _load_baseline_entrypoint().parse_whisper_languages(args.whisper_languages)
        ),
        whisper_timeout_ms=args.whisper_timeout_ms,
        min_text_confidence=args.min_text_confidence,
        # Retained at their baseline values for PipelineConfig/CLI compatibility.
        # The replacement resolver below consumes the streaming API's own
        # min-duration settings, not the legacy diarization smoothing options.
        diarization_min_duration_on=args.diarization_min_duration_on,
        diarization_min_duration_off=args.diarization_min_duration_off,
        min_cluster_duration=args.min_cluster_duration,
        centroid_assignment_similarity_threshold=args.centroid_assignment_similarity_threshold,
        save_segments=args.save_segments,
        debug=args.debug,
        segmentation_mode=baseline_pipeline.SEGMENTATION_MODE_VAD_PYANNOTE,
        run_label=args.run_label,
    )


class _StreamingSpeakerSegmentationRuntime:
    """Adapter with the method consumed by the unchanged baseline pipeline."""

    def __init__(
        self,
        model: Path,
        *,
        chunk_ms: int,
        num_threads: int,
        min_duration_on: float,
        min_duration_off: float,
        change_vote_threshold: float,
    ) -> None:
        self.model = model
        self.chunk_ms = chunk_ms
        self.num_threads = num_threads
        self.min_duration_on = min_duration_on
        self.min_duration_off = min_duration_off
        self.change_vote_threshold = change_vote_threshold
        self.spans: list[dict[str, float | int]] = []

    def infer_speaker_count_spans(self, samples: np.ndarray) -> list[dict[str, float | int]]:
        self.spans = stream_speaker_segmentation_samples(
            samples,
            model=self.model,
            chunk_ms=self.chunk_ms,
            num_threads=self.num_threads,
            min_duration_on=self.min_duration_on,
            min_duration_off=self.min_duration_off,
            change_vote_threshold=self.change_vote_threshold,
        )
        return self.spans


@contextmanager
def _use_streaming_segmentation_runtime(
    runtime: _StreamingSpeakerSegmentationRuntime,
) -> Iterator[None]:
    """Temporarily inject the new provider into the unchanged baseline runner."""
    original_build_runtimes = baseline_pipeline.build_runtimes
    original_resolve_final_segments = baseline_pipeline.resolve_final_segments

    def build_runtimes_with_streaming_provider(**kwargs: Any) -> object:
        # Do not instantiate the legacy SegmentationRuntime; the returned proxy
        # supplies exactly the method run_pipeline needs at this extension point.
        kwargs["enable_segmentation"] = False
        runtimes = original_build_runtimes(**kwargs)
        resolved_files = dict(runtimes.resolved_files)
        resolved_files["segmentation"] = ResolvedModelFiles(model=runtime.model)
        return replace(runtimes, segmentation=runtime, resolved_files=resolved_files)

    def resolve_with_streaming_spans(
        vad_segments: Sequence[SpeechSegment],
        activity: Sequence[Mapping[str, Any] | object],
        waveform: np.ndarray,
        sample_rate: int,
        **_: Any,
    ) -> tuple[list[SpeechSegment], TimelineResolutionStats]:
        segments = resolve_vad_segments_with_speaker_segmentation(
            vad_segments,
            activity,
            waveform,
            sample_rate,
            min_duration_on=runtime.min_duration_on,
            min_duration_off=runtime.min_duration_off,
        )
        return segments, TimelineResolutionStats(
            true_overlap_count=sum(bool(segment.overlap_regions) for segment in segments)
        )

    baseline_pipeline.build_runtimes = build_runtimes_with_streaming_provider
    baseline_pipeline.resolve_final_segments = resolve_with_streaming_spans
    try:
        yield
    finally:
        baseline_pipeline.build_runtimes = original_build_runtimes
        baseline_pipeline.resolve_final_segments = original_resolve_final_segments


def _update_streaming_metadata(
    run_dir: Path, args: argparse.Namespace, model: Path, spans: Sequence[Mapping[str, Any] | object]
) -> None:
    path = run_dir / "run_metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["streaming_speaker_segmentation"] = {
        "model": str(model),
        "chunk_ms": args.segmentation_chunk_ms,
        "num_threads": args.segmentation_num_threads,
        "min_duration_on": args.min_duration_on,
        "min_duration_off": args.min_duration_off,
        "change_vote_threshold": args.change_vote_threshold,
        "span_count": len(spans),
    }
    write_metadata(run_dir, metadata)


def run(args: argparse.Namespace) -> baseline_pipeline.PipelineResult:
    if args.sample_rate != SAMPLE_RATE:
        raise ValueError("The pyannote segmentation model requires --sample-rate=16000")
    if args.segmentation_chunk_ms <= 0:
        raise ValueError("--segmentation-chunk-ms must be positive")
    model = resolve_segmentation_model(args)
    if not model.is_file():
        raise FileNotFoundError(f"Segmentation model not found: {model}")
    if args.reference is not None and not args.reference.is_file():
        raise FileNotFoundError(f"Reference file not found: {args.reference}")
    if args.baseline_run_dir is not None and not (args.baseline_run_dir / "result.json").is_file():
        raise FileNotFoundError(f"Baseline result.json not found: {args.baseline_run_dir}")

    runtime = _StreamingSpeakerSegmentationRuntime(
        model,
        chunk_ms=args.segmentation_chunk_ms,
        num_threads=args.segmentation_num_threads,
        min_duration_on=args.min_duration_on,
        min_duration_off=args.min_duration_off,
        change_vote_threshold=args.change_vote_threshold,
    )
    config = build_baseline_pipeline_config(args)
    with _use_streaming_segmentation_runtime(runtime):
        result = baseline_pipeline.run_pipeline(config)

    write_span_artifacts(
        result.run_dir,
        config.audio.stem,
        result.timings["total_seconds"] / result.rtf if result.rtf > 0 else 0.0,
        runtime.spans,
    )
    _update_streaming_metadata(result.run_dir, args, model, runtime.spans)
    if args.baseline_run_dir is not None:
        write_comparison_report(
            result.run_dir,
            _result_summary(args.baseline_run_dir, args.reference),
            _result_summary(result.run_dir, args.reference),
        )
    elif args.reference is not None:
        (result.run_dir / "wer.json").write_text(
            json.dumps(_result_summary(result.run_dir, args.reference), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Pipeline failed: {error}", file=sys.stderr)
        return 2
    print(f"Pipeline finished: {result.run_dir}")
    print(f"segments={result.segment_count} rtf={result.rtf:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
