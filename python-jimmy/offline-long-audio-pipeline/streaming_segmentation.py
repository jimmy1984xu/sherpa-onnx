"""Utilities and CLI implementation for streaming pyannote segmentation PCM tests.

The module deliberately imports ``sherpa_onnx`` only while executing the
streaming runner.  This keeps artifact/report tests usable with a Python
installation that does not yet contain the new extension symbols.
"""

from __future__ import annotations

import argparse
from array import array
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

CONTINUE = 0
SPEAKER_COUNT_CHANGED = 1
SINGLE_SPEAKER_CHANGED = 2
INPUT_FINISHED = 4

_VALID_FLAG_MASK = (
    SPEAKER_COUNT_CHANGED | SINGLE_SPEAKER_CHANGED | INPUT_FINISHED
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the SpeakerSegmentation streaming API over recursively "
            "discovered little-endian signed-16-bit PCM files."
        )
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--sample-width", type=int, default=2)
    parser.add_argument("--chunk-ms", type=int, default=32)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--min-duration-on", type=float, default=0.30)
    parser.add_argument("--min-duration-off", type=float, default=0.50)
    parser.add_argument("--change-vote-threshold", type=float, default=0.50)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if not args.input_dir.is_dir():
        raise ValueError(f"input-dir is not a directory: {args.input_dir}")
    if not str(args.model):
        raise ValueError("model must not be empty")
    if args.sample_rate <= 0:
        raise ValueError("sample-rate must be positive")
    if args.channels <= 0:
        raise ValueError("channels must be positive")
    if args.sample_width != 2:
        raise ValueError("sample-width must be 2 for signed int16 PCM")
    if args.chunk_ms <= 0:
        raise ValueError("chunk-ms must be positive")
    if args.num_threads <= 0:
        raise ValueError("num-threads must be positive")
    if args.min_duration_on < 0:
        raise ValueError("min-duration-on must be non-negative")
    if args.min_duration_off < 0:
        raise ValueError("min-duration-off must be non-negative")
    if not 0.0 <= args.change_vote_threshold <= 1.0:
        raise ValueError("change-vote-threshold must be in [0, 1]")
    if args.sample_rate * args.chunk_ms % 1000 != 0:
        raise ValueError(
            "sample-rate * chunk-ms must produce an integer number of samples"
        )


def _span_record(span: Mapping[str, Any] | Any) -> dict[str, Any]:
    if isinstance(span, Mapping):
        return {
            "start": span["start"],
            "end": span["end"],
            "speaker_count": span["speaker_count"],
            "flag": span["flag"],
            "local_speaker_mask": span.get("local_speaker_mask", 0),
            "local_speaker_mask_confidence": span.get("local_speaker_mask_confidence", 0.0),
        }
    return {
        "start": span.start,
        "end": span.end,
        "speaker_count": span.speaker_count,
        "flag": span.flag,
        "local_speaker_mask": getattr(span, "local_speaker_mask", 0),
        "local_speaker_mask_confidence": getattr(span, "local_speaker_mask_confidence", 0.0),
    }


def normalize_spans(
    spans: Iterable[Mapping[str, Any] | Any],
) -> list[dict[str, Any]]:
    """Validate and normalize API spans for JSON artifact output."""
    records: list[dict[str, Any]] = []
    previous_end: float | None = None

    for index, span in enumerate(spans):
        record = _span_record(span)
        try:
            start = float(record["start"])
            end = float(record["end"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"span {index} start/end must be numeric") from error

        speaker_count = record["speaker_count"]
        flag = record["flag"]
        local_speaker_mask = record.get("local_speaker_mask", 0)
        local_speaker_mask_confidence = record.get("local_speaker_mask_confidence", 0.0)
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError(f"span {index} start/end must be finite")
        if end < start:
            raise ValueError(f"span {index} has end before start")
        if previous_end is not None and start < previous_end:
            raise ValueError(f"span {index} overlaps the previous span")
        if type(speaker_count) is not int or not 0 <= speaker_count <= 2:
            raise ValueError(f"span {index} speaker_count must be an integer in [0, 2]")
        if type(flag) is not int or flag < 0 or flag & ~_VALID_FLAG_MASK:
            raise ValueError(f"span {index} has an unknown flag bit: {flag!r}")
        if type(local_speaker_mask) is not int or not 0 <= local_speaker_mask <= 0b111:
            raise ValueError(f"span {index} local_speaker_mask must use the low three bits")
        try:
            local_speaker_mask_confidence = float(local_speaker_mask_confidence)
        except (TypeError, ValueError) as error:
            raise ValueError(f"span {index} local_speaker_mask_confidence must be numeric") from error
        if not math.isfinite(local_speaker_mask_confidence) or not 0.0 <= local_speaker_mask_confidence <= 1.0:
            raise ValueError(f"span {index} local_speaker_mask_confidence must be in [0, 1]")

        records.append(
            {
                "start": start,
                "end": end,
                "speaker_count": speaker_count,
                "flag": flag,
                "local_speaker_mask": local_speaker_mask,
                "local_speaker_mask_confidence": local_speaker_mask_confidence,
            }
        )
        previous_end = end

    for index, record in enumerate(records[:-1]):
        if record["flag"] & INPUT_FINISHED:
            raise ValueError(f"span {index} has INPUT_FINISHED before the final span")
    return records


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_span_artifacts(
    output_dir: Path, stem: str, audio_seconds: float, spans: Iterable[Mapping[str, Any] | Any]
) -> dict[str, Any]:
    """Write exact API spans and their duration/count summary for one PCM file."""
    if not stem:
        raise ValueError("stem must not be empty")
    audio_seconds = float(audio_seconds)
    if not math.isfinite(audio_seconds) or audio_seconds < 0:
        raise ValueError("audio_seconds must be a non-negative finite value")

    records = normalize_spans(spans)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"{stem}.spans.jsonl"
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    durations = {count: 0.0 for count in range(3)}
    for record in records:
        durations[record["speaker_count"]] += record["end"] - record["start"]
    summary = {
        "audio_seconds": float(audio_seconds),
        "span_count": len(records),
        "count_0_seconds": durations[0],
        "count_1_seconds": durations[1],
        "count_2_seconds": durations[2],
        "single_speaker_change_count": sum(
            bool(record["flag"] & SINGLE_SPEAKER_CHANGED) for record in records
        ),
    }
    _write_json(output_dir / f"{stem}.summary.json", summary)
    return summary


def merge_display_runs(
    spans: Iterable[Mapping[str, Any] | Any],
) -> list[dict[str, Any]]:
    """Merge adjacent same-count spans only across a CONTINUE boundary."""
    records = normalize_spans(spans)
    merged: list[dict[str, Any]] = []
    for record in records:
        if (
            merged
            and merged[-1]["flag"] == CONTINUE
            and merged[-1]["end"] == record["start"]
            and merged[-1]["speaker_count"] == record["speaker_count"]
        ):
            merged[-1]["end"] = record["end"]
            merged[-1]["flag"] = record["flag"]
        else:
            merged.append(dict(record))
    return merged


def write_comparison_report(
    output_dir: Path,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Write comparable baseline/new metrics without interpreting speaker IDs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = {
        "baseline": dict(baseline),
        "speaker_segmentation": dict(candidate),
    }
    _write_json(output_dir / "comparison.json", comparison)

    report = "\n".join(
        [
            "# Speaker-segmentation comparison",
            "",
            "| Metric | Baseline | Speaker segmentation |",
            "| --- | ---: | ---: |",
            f"| WER | {baseline.get('wer', 'n/a')} | {candidate.get('wer', 'n/a')} |",
            (
                "| segment count | "
                f"{baseline.get('segment_count', 'n/a')} | "
                f"{candidate.get('segment_count', 'n/a')} |"
            ),
            (
                "| multi/overlap segments | "
                f"{baseline.get('multi_segment_count', 'n/a')} | "
                f"{candidate.get('multi_segment_count', 'n/a')} |"
            ),
            (
                "| multi/overlap breaks | "
                f"{baseline.get('multi_break_count', 'n/a')} | "
                f"{candidate.get('multi_break_count', 'n/a')} |"
            ),
            "",
        ]
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8", newline="\n")
    return comparison


def pcm_s16le_to_float32(data: bytes, channels: int) -> tuple[array, int]:
    """Decode interleaved S16LE PCM and average channels into a mono float array."""
    if channels <= 0:
        raise ValueError("channels must be positive")
    if len(data) % 2 != 0:
        raise ValueError("PCM byte length must be even for signed int16 samples")

    samples = array("h")
    samples.frombytes(data)
    if sys.byteorder != "little":
        samples.byteswap()
    if len(samples) % channels != 0:
        raise ValueError("PCM sample count is not divisible by channels")

    frames = len(samples) // channels
    waveform = array("f")
    if channels == 1:
        waveform.extend(sample / 32768.0 for sample in samples)
    else:
        for frame_start in range(0, len(samples), channels):
            waveform.append(
                sum(samples[frame_start : frame_start + channels])
                / (32768.0 * channels)
            )
    return waveform, frames


def _drain(segmenter: Any) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    while not segmenter.empty():
        spans.append(_span_record(segmenter.front))
        segmenter.pop()
    return spans


def _pcm_paths(input_dir: Path) -> list[Path]:
    """Return every recursively discovered PCM input in stable path order."""
    return sorted(path for path in input_dir.rglob("*.pcm") if path.is_file())


def _artifact_output_dir(output_dir: Path, input_dir: Path, pcm_path: Path) -> Path:
    """Mirror the PCM's relative parent so duplicate stems cannot collide."""
    relative = pcm_path.relative_to(input_dir)
    return output_dir / relative.parent


def run_streaming_segmentation(args: argparse.Namespace) -> dict[str, Any]:
    """Run direct SpeakerSegmentation streaming inference and write artifacts.

    No download is initiated here: ``args.model`` is passed unchanged to the
    locally installed sherpa_onnx extension.
    """
    import sherpa_onnx

    pyannote = sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
        str(args.model)
    )
    model = sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
        pyannote=pyannote,
        num_threads=args.num_threads,
        debug=False,
        provider="cpu",
    )
    config = sherpa_onnx.SpeakerSegmentationConfig(
        model=model,
        min_duration_on=args.min_duration_on,
        min_duration_off=args.min_duration_off,
        change_vote_threshold=args.change_vote_threshold,
    )
    if not config.validate():
        raise ValueError(f"invalid SpeakerSegmentationConfig: {config}")

    chunk_samples = args.sample_rate * args.chunk_ms // 1000
    segmenter = sherpa_onnx.SpeakerSegmentation(config)
    if segmenter.sample_rate != args.sample_rate:
        raise ValueError(
            "sample-rate does not match the model sample rate: "
            f"{args.sample_rate} != {segmenter.sample_rate}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    file_summaries: list[dict[str, Any]] = []
    for pcm_path in _pcm_paths(args.input_dir):
        waveform, frames = pcm_s16le_to_float32(
            pcm_path.read_bytes(), args.channels
        )
        spans: list[dict[str, Any]] = []
        for begin in range(0, len(waveform), chunk_samples):
            segmenter.accept_waveform(waveform[begin : begin + chunk_samples])
            spans.extend(_drain(segmenter))
        segmenter.input_finished()
        spans.extend(_drain(segmenter))
        spans = normalize_spans(spans)

        artifact_dir = _artifact_output_dir(args.output_dir, args.input_dir, pcm_path)
        summary = write_span_artifacts(
            artifact_dir,
            pcm_path.stem,
            frames / args.sample_rate,
            spans,
        )
        file_summaries.append(
            {
                "input": str(pcm_path.relative_to(args.input_dir)),
                "artifacts": str(artifact_dir.relative_to(args.output_dir) / pcm_path.stem),
                "stem": pcm_path.stem,
                **summary,
            }
        )
        segmenter.reset()

    global_summary = {
        "input_dir": str(args.input_dir),
        "model": str(args.model),
        "file_count": len(file_summaries),
        "audio_seconds": sum(item["audio_seconds"] for item in file_summaries),
        "span_count": sum(item["span_count"] for item in file_summaries),
        "files": file_summaries,
    }
    _write_json(args.output_dir / "summary.json", global_summary)
    return global_summary


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        validate_args(args)
        run_streaming_segmentation(args)
    except ValueError as error:
        print(f"validation error: {error}", file=sys.stderr)
        return 2
    except ImportError as error:
        print(
            "cannot import the required sherpa_onnx streaming API: "
            f"{error}",
            file=sys.stderr,
        )
        return 1
    except Exception as error:
        print(f"streaming segmentation failed: {error}", file=sys.stderr)
        return 1
    return 0
