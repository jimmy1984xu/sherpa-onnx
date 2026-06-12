#!/usr/bin/env python3

"""Run VAD cutting, offline ASR, ASR merging, and WER evaluation in order."""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


SCRIPT_DIR = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Run VAD cutting, Paraformer ASR, ASR merging, and WER evaluation "
            "for one audio file."
        ),
    )

    # Required by the underlying scripts.
    parser.add_argument("--audio", required=True, help="Input PCM/WAV audio file")
    vad_group = parser.add_mutually_exclusive_group(required=True)
    vad_group.add_argument("--silero-vad-model", help="Path to silero_vad.onnx")
    vad_group.add_argument("--ten-vad-model", help="Path to ten-vad.onnx")
    parser.add_argument("--model-dir", required=True, help="Paraformer model directory")
    parser.add_argument(
        "--language",
        required=True,
        help="Recognition language; the underlying ASR script currently supports zh",
    )
    parser.add_argument(
        "--label",
        default="label.txt",
        help="Reference label file used by evaluation.py",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for all generated intermediate and result files",
    )

    # Defaults mirror k2-vad_cut.py, utils-merge-asr-text.py, and evaluation.py.
    parser.add_argument("--threshold", type=float, default=0.5, help="VAD threshold")
    parser.add_argument(
        "--min-silence-duration",
        type=float,
        default=0.5,
        help="Minimum silence duration in seconds",
    )
    parser.add_argument(
        "--min-speech-duration",
        type=float,
        default=0.25,
        help="Minimum speech duration in seconds",
    )
    parser.add_argument(
        "--max-speech-duration",
        type=float,
        default=20.0,
        help="Maximum speech duration in seconds",
    )
    parser.add_argument("--utt-id", default="utt001", help="Merged output utterance ID")
    parser.add_argument("--metric", default="wer", help="Evaluation metric")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Pass --debug to the VAD cutting script",
    )
    return parser


def validate_inputs(args: argparse.Namespace) -> None:
    audio = Path(args.audio)
    if not audio.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio}")

    label = Path(args.label)
    if not label.is_file():
        raise FileNotFoundError(f"Label file not found: {label}")

    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    vad_model = Path(args.silero_vad_model or args.ten_vad_model)
    if not vad_model.is_file():
        raise FileNotFoundError(f"VAD model not found: {vad_model}")

    for script_name in (
        "k2-vad_cut.py",
        "k2-offline-asr.py",
        "utils-merge-asr-text.py",
        "evaluation.py",
    ):
        script_path = SCRIPT_DIR / script_name
        if not script_path.is_file():
            raise FileNotFoundError(f"Pipeline script not found: {script_path}")


def run_step(step_name: str, command: List[str]) -> None:
    print(f"\n[{step_name}] {' '.join(command)}")
    subprocess.run(command, check=True)


def run_pipeline(args: argparse.Namespace) -> dict:
    validate_inputs(args)

    output_dir = Path(args.output_dir)
    segments_dir = output_dir / "segments"
    asr_output = output_dir / "asr.txt"
    merged_output = output_dir / "merged.txt"
    detail_output = output_dir / "wer_detail.txt"
    output_dir.mkdir(parents=True, exist_ok=True)

    vad_command = [
        sys.executable,
        str(SCRIPT_DIR / "k2-vad_cut.py"),
        "--audio",
        str(Path(args.audio)),
        "--output-dir",
        str(segments_dir),
        "--threshold",
        str(args.threshold),
        "--min-silence-duration",
        str(args.min_silence_duration),
        "--min-speech-duration",
        str(args.min_speech_duration),
        "--max-speech-duration",
        str(args.max_speech_duration),
    ]
    vad_option = "--silero-vad-model" if args.silero_vad_model else "--ten-vad-model"
    vad_command.extend([vad_option, str(args.silero_vad_model or args.ten_vad_model)])
    if args.debug:
        vad_command.append("--debug")
    run_step("1/4 VAD", vad_command)

    run_step(
        "2/4 ASR",
        [
            sys.executable,
            str(SCRIPT_DIR / "k2-offline-asr.py"),
            "--model-dir",
            str(Path(args.model_dir)),
            "--language",
            args.language,
            "--audio-dir",
            str(segments_dir),
            "--output",
            str(asr_output),
        ],
    )

    run_step(
        "3/4 merge ASR",
        [
            sys.executable,
            str(SCRIPT_DIR / "utils-merge-asr-text.py"),
            "--input",
            str(asr_output),
            "--output",
            str(merged_output),
            "--utt-id",
            args.utt_id,
        ],
    )

    run_step(
        "4/4 WER",
        [
            sys.executable,
            str(SCRIPT_DIR / "evaluation.py"),
            "--label",
            str(Path(args.label)),
            "--hyp",
            str(merged_output),
            "--language",
            args.language,
            "--metric",
            args.metric,
            "--detail",
            str(detail_output),
        ],
    )

    return {
        "segments": segments_dir,
        "asr": asr_output,
        "merged": merged_output,
        "detail": detail_output,
    }


def main() -> int:
    args = build_parser().parse_args()
    try:
        outputs = run_pipeline(args)
    except (FileNotFoundError, OSError, subprocess.CalledProcessError, ValueError) as error:
        print(f"Pipeline failed: {error}", file=sys.stderr)
        return error.returncode if isinstance(error, subprocess.CalledProcessError) else 1

    print("\nPipeline complete. Generated files:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
