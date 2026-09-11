#!/usr/bin/env python3
"""Offline local VAD + Paraformer ASR + Titanet clustering pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PIPELINE_DIR = Path(__file__).with_name("offline-long-audio-pipeline")
sys.path.insert(0, str(PIPELINE_DIR))

from audio_io import AudioFormat
from pipeline import (
    DEFAULT_ASR_DIR,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SEGMENTATION_DIR,
    DEFAULT_SPEAKER_DIR,
    DEFAULT_VAD_DIR,
    PipelineConfig,
    run_pipeline,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run local Silero VAD, pyannote diarization, Chinese Paraformer ASR, and Titanet similarity extraction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--audio", required=True, help="Input PCM or WAV file")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--audio-format", choices=("auto", "pcm", "wav"), default="auto")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--sample-width", type=int, default=2)
    parser.add_argument("--asr-dir", default=str(DEFAULT_ASR_DIR))
    parser.add_argument("--vad-dir", default=str(DEFAULT_VAD_DIR))
    parser.add_argument("--speaker-dir", default=str(DEFAULT_SPEAKER_DIR))
    parser.add_argument("--segmentation-dir", default=str(DEFAULT_SEGMENTATION_DIR))
    parser.add_argument("--asr-num-threads", type=int, default=1)
    parser.add_argument("--speaker-num-threads", type=int, default=2)
    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--min-silence-duration", type=float, default=0.8)
    parser.add_argument("--min-speech-duration", type=float, default=0.25)
    parser.add_argument("--max-speech-duration", type=float, default=25.0)
    parser.add_argument("--pre-speech-pad-duration", type=float, default=0.0)
    parser.add_argument("--cluster-threshold", type=float, default=0.5)
    parser.add_argument("--num-clusters", type=int, default=-1)
    parser.add_argument("--min-cluster-duration", type=float, default=1.0)
    parser.add_argument("--centroid-assignment-similarity-threshold", type=float, default=0.5)
    parser.add_argument("--diarization-min-duration-on", type=float, default=0.5)
    parser.add_argument("--diarization-min-duration-off", type=float, default=0.5)
    parser.add_argument("--save-segments", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--segmentation-mode",
        choices=("vad", "vad-pyannote"),
        default="vad-pyannote",
        help="vad keeps Silero VAD regions; vad-pyannote splits VAD speech with local pyannote activity",
    )
    parser.add_argument(
        "--run-label",
        default=None,
        help="Optional experiment label included in the output directory name",
    )
    return parser


def _resolve_audio_format(audio: Path, requested: str, sample_rate: int, channels: int, sample_width: int) -> AudioFormat:
    kind = requested if requested != "auto" else ("pcm" if audio.suffix.lower() == ".pcm" else "wav")
    return AudioFormat(kind=kind, sample_rate=sample_rate, channels=channels, sample_width=sample_width)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    audio = Path(args.audio)
    try:
        result = run_pipeline(
            PipelineConfig(
                audio=audio,
                output_root=Path(args.output_root),
                audio_format=_resolve_audio_format(audio, args.audio_format, args.sample_rate, args.channels, args.sample_width),
                asr_dir=Path(args.asr_dir),
                vad_dir=Path(args.vad_dir),
                speaker_dir=Path(args.speaker_dir),
                segmentation_dir=Path(args.segmentation_dir),
                asr_num_threads=args.asr_num_threads,
                speaker_num_threads=args.speaker_num_threads,
                vad_threshold=args.vad_threshold,
                min_silence_duration=args.min_silence_duration,
                min_speech_duration=args.min_speech_duration,
                max_speech_duration=args.max_speech_duration,
                pre_speech_pad_duration=args.pre_speech_pad_duration,
                cluster_threshold=args.cluster_threshold,
                num_clusters=args.num_clusters,
                min_cluster_duration=args.min_cluster_duration,
                centroid_assignment_similarity_threshold=args.centroid_assignment_similarity_threshold,
                diarization_min_duration_on=args.diarization_min_duration_on,
                diarization_min_duration_off=args.diarization_min_duration_off,
                save_segments=args.save_segments,
                debug=args.debug,
                segmentation_mode=args.segmentation_mode,
                run_label=args.run_label,
            )
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Pipeline failed: {error}", file=sys.stderr)
        return 2
    print(f"Pipeline finished: {result.run_dir}")
    print(f"segments={result.segment_count} rtf={result.rtf:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
