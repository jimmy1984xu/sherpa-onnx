#!/usr/bin/env python3

"""Cut VAD speech segments and merge nearby short segments."""

import argparse
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf


SCRIPT_DIR = Path(__file__).resolve().parent
VAD_CUT_SCRIPT = SCRIPT_DIR / "k2-vad_cut.py"
VAD_CUT_SPEC = importlib.util.spec_from_file_location("k2_vad_cut", VAD_CUT_SCRIPT)
if VAD_CUT_SPEC is None or VAD_CUT_SPEC.loader is None:
    raise RuntimeError(f"Unable to load VAD script: {VAD_CUT_SCRIPT}")

VAD_CUT = importlib.util.module_from_spec(VAD_CUT_SPEC)
VAD_CUT_SPEC.loader.exec_module(VAD_CUT)


@dataclass(frozen=True)
class VadSegment:
    start: int
    end: int

    @property
    def duration(self) -> int:
        return self.end - self.start


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Cut VAD speech segments and merge nearby short segments",
    )
    parser.add_argument("--audio", required=True, help="Input PCM/WAV audio file")
    parser.add_argument(
        "--output-dir", required=True, help="Directory for merged WAV segments"
    )
    vad_group = parser.add_mutually_exclusive_group(required=True)
    vad_group.add_argument("--silero-vad-model", help="Path to silero_vad.onnx")
    vad_group.add_argument("--ten-vad-model", help="Path to ten-vad.onnx")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-silence-duration", type=float, default=0.5)
    parser.add_argument("--min-speech-duration", type=float, default=0.25)
    parser.add_argument("--max-speech-duration", type=float, default=20.0)
    parser.add_argument("--pre-speech-pad-duration", type=float, default=0.0)
    parser.add_argument(
        "--merge-gap-duration",
        type=float,
        default=2.0,
        help="Maximum gap between adjacent segments to consider merging, in seconds",
    )
    parser.add_argument(
        "--short-segment-duration",
        type=float,
        default=6.0,
        help="Segments shorter than this duration are considered short, in seconds",
    )
    parser.add_argument(
        "--max-merged-duration",
        type=float,
        default=30.0,
        help="Maximum duration of a merged segment, in seconds",
    )
    parser.add_argument("--wav-scp", default="", help="Optional output wav.scp path")
    parser.add_argument("--debug", action="store_true")
    return parser


def validate_merge_options(args: argparse.Namespace) -> None:
    if args.merge_gap_duration < 0:
        raise ValueError("--merge-gap-duration must be non-negative")
    if args.short_segment_duration <= 0:
        raise ValueError("--short-segment-duration must be positive")
    if args.max_merged_duration <= 0:
        raise ValueError("--max-merged-duration must be positive")


def collect_vad_segments(
    vad, wav_16k: np.ndarray, window_size: int
) -> List[VadSegment]:
    segments: List[VadSegment] = []

    def drain() -> None:
        while not vad.empty():
            segment = vad.front
            start = int(segment.start)
            end = start + len(segment.samples)
            segments.append(VadSegment(start=start, end=end))
            vad.pop()

    for offset in range(0, len(wav_16k), window_size):
        vad.accept_waveform(wav_16k[offset : offset + window_size])
        drain()

    vad.flush()
    drain()
    return segments


def merge_segments(
    segments: List[VadSegment],
    sample_rate: int,
    merge_gap_duration: float,
    short_segment_duration: float,
    max_merged_duration: float,
) -> List[List[VadSegment]]:
    if not segments:
        return []

    merge_gap_samples = int(round(merge_gap_duration * sample_rate))
    short_segment_samples = short_segment_duration * sample_rate
    max_merged_samples = int(round(max_merged_duration * sample_rate))
    groups: List[List[VadSegment]] = []
    current: List[VadSegment] = [segments[0]]

    for segment in segments[1:]:
        previous = current[-1]
        gap = segment.start - previous.end
        merged_start = current[0].start
        merged_end = segment.end
        current_duration = previous.end - merged_start
        merged_duration = merged_end - merged_start
        current_is_long = current_duration >= short_segment_samples
        next_is_long = segment.duration >= short_segment_samples

        can_merge = (
            gap <= merge_gap_samples
            and merged_duration <= max_merged_samples
            and not (current_is_long and next_is_long)
        )
        if can_merge:
            current.append(segment)
        else:
            groups.append(current)
            current = [segment]

    groups.append(current)
    return groups


def write_merged_segments(
    wav_16k: np.ndarray,
    sample_rate: int,
    groups: List[List[VadSegment]],
    output_dir: Path,
) -> List[Tuple[str, Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: List[Tuple[str, Path]] = []

    for sequence, group in enumerate(groups, start=1):
        start = max(0, group[0].start)
        end = min(len(wav_16k), group[-1].end)
        samples = np.asarray(wav_16k[start:end], dtype=np.float32)
        offset_ms = int(start / sample_rate * 1000)
        duration_ms = int(len(samples) / sample_rate * 1000)
        segment_id = VAD_CUT.build_segment_id(sequence, offset_ms, duration_ms)
        output_path = output_dir / f"{segment_id}.wav"
        sf.write(str(output_path), samples, sample_rate)
        outputs.append((segment_id, output_path))
        print(
            f"Merged segment {segment_id} "
            f"[{offset_ms / 1000:.2f}s-"
            f"{(offset_ms + duration_ms) / 1000:.2f}s] "
            f"from {len(group)} VAD segment(s)"
        )

    return outputs


def main() -> int:
    args = build_parser().parse_args()
    validate_merge_options(args)

    audio_path = Path(args.audio)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    wav, source_sample_rate = VAD_CUT.load_audio_mono_float32(str(audio_path))
    vad_sample_rate = 16000
    wav_16k = VAD_CUT.resample_linear(wav, source_sample_rate, vad_sample_rate)
    vad_type = "silero" if args.silero_vad_model else "ten"
    vad_model_path = args.silero_vad_model or args.ten_vad_model
    vad, window_size, _ = VAD_CUT.build_vad(
        vad_model_path=vad_model_path,
        vad_type=vad_type,
        sample_rate=vad_sample_rate,
        threshold=args.threshold,
        min_silence_duration=args.min_silence_duration,
        min_speech_duration=args.min_speech_duration,
        max_speech_duration=args.max_speech_duration,
        pre_speech_pad_duration=args.pre_speech_pad_duration,
        debug=args.debug,
    )

    segments = collect_vad_segments(vad, wav_16k, window_size)
    groups = merge_segments(
        segments,
        sample_rate=vad_sample_rate,
        merge_gap_duration=args.merge_gap_duration,
        short_segment_duration=args.short_segment_duration,
        max_merged_duration=args.max_merged_duration,
    )
    outputs = write_merged_segments(
        wav_16k,
        sample_rate=vad_sample_rate,
        groups=groups,
        output_dir=Path(args.output_dir),
    )

    if args.wav_scp:
        wav_scp_path = Path(args.wav_scp)
        wav_scp_path.parent.mkdir(parents=True, exist_ok=True)
        with wav_scp_path.open("w", encoding="utf-8") as wav_scp_file:
            for segment_id, output_path in outputs:
                wav_scp_file.write(f"{segment_id} {output_path}\n")

    print(
        f"Done! Collected {len(segments)} VAD segment(s), "
        f"wrote {len(outputs)} merged segment(s) to {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
