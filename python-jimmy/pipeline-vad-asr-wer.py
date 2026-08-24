#!/usr/bin/env python3

"""Run VAD cutting, offline ASR, ASR merging, and WER evaluation in order."""

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import List


SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_MERGE_GAP_DURATION = 2.0
DEFAULT_SHORT_SEGMENT_DURATION = 6.0
DEFAULT_MAX_MERGED_DURATION = 30.0


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
    parser.add_argument("--asr-model", required=True, help="Paraformer model directory")
    parser.add_argument(
        "--language",
        required=True,
        help="Recognition language; the underlying ASR script currently supports zh",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Optional reference label file used by evaluation.py; omitted to skip WER",
    )
    parser.add_argument(
        "--skip-wer",
        action="store_true",
        help="Skip WER evaluation even when --label is provided",
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
    parser.add_argument(
        "--pre-speech-pad-duration",
        type=float,
        default=0.0,
        help="Audio retained before detected speech in seconds; 0 keeps VAD defaults",
    )
    parser.add_argument(
        "--merge-gap-duration",
        type=float,
        default=None,
        help="Enable VAD segment merging and set the maximum gap in seconds",
    )
    parser.add_argument(
        "--short-segment-duration",
        type=float,
        default=None,
        help="Enable VAD segment merging and set the short-segment threshold in seconds",
    )
    parser.add_argument(
        "--max-merged-duration",
        type=float,
        default=None,
        help="Enable VAD segment merging and set the maximum merged duration in seconds",
    )
    parser.add_argument("--utt-id", default="utt001", help="Merged output utterance ID")
    parser.add_argument("--metric", default="wer", help="Evaluation metric")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Pass --debug to the VAD cutting script",
    )
    return parser


def skip_wer(args: argparse.Namespace) -> bool:
    """Return whether the optional WER stage should be omitted."""

    return args.skip_wer or args.label is None


def vad_merge_enabled(args: argparse.Namespace) -> bool:
    return any(
        value is not None
        for value in (
            args.merge_gap_duration,
            args.short_segment_duration,
            args.max_merged_duration,
        )
    )


def validate_inputs(args: argparse.Namespace) -> None:
    audio = Path(args.audio)
    if not audio.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio}")

    if not skip_wer(args):
        label = Path(args.label)
        if not label.is_file():
            raise FileNotFoundError(f"Label file not found: {label}")

    asr_model = Path(args.asr_model)
    if not asr_model.is_dir():
        raise FileNotFoundError(f"ASR model directory not found: {asr_model}")

    vad_model = Path(args.silero_vad_model or args.ten_vad_model)
    if not vad_model.is_file():
        raise FileNotFoundError(f"VAD model not found: {vad_model}")

    vad_script_name = (
        "k2-vad-cut-merge.py" if vad_merge_enabled(args) else "k2-vad_cut.py"
    )
    script_names = [
        vad_script_name,
        "k2-offline-asr.py",
        "utils-merge-asr-text.py",
    ]
    if not skip_wer(args):
        script_names.append("evaluation.py")

    for script_name in script_names:
        script_path = SCRIPT_DIR / script_name
        if not script_path.is_file():
            raise FileNotFoundError(f"Pipeline script not found: {script_path}")


def run_step(step_name: str, command: List[str]) -> None:
    print(f"\n[{step_name}] {' '.join(command)}")
    subprocess.run(command, check=True)


def print_wer_result(detail_path: Path, metric: str) -> None:
    """Print the metric values saved by evaluation.py."""

    with detail_path.open("r", encoding="utf-8", newline="") as detail_file:
        rows = list(csv.DictReader(detail_file, delimiter="\t"))

    print(f"\n{metric.upper()} result:")
    if not rows:
        print(f"  No results found in {detail_path}")
        return

    for row in rows:
        print(f"  {row.get('id', '<unknown>')}: {row.get('wer', '<unknown>')}%")


def run_pipeline(args: argparse.Namespace) -> dict:
    validate_inputs(args)
    run_wer = not skip_wer(args)

    output_dir = Path(args.output_dir)
    segments_dir = output_dir / "segments"
    asr_output = output_dir / "asr.txt"
    merged_output = output_dir / "merged.txt"
    detail_output = output_dir / "wer_detail.txt"
    total_steps = 4 if run_wer else 3
    output_dir.mkdir(parents=True, exist_ok=True)
    use_vad_merge = vad_merge_enabled(args)
    vad_script_name = "k2-vad-cut-merge.py" if use_vad_merge else "k2-vad_cut.py"

    vad_command = [
        sys.executable,
        str(SCRIPT_DIR / vad_script_name),
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
        "--pre-speech-pad-duration",
        str(args.pre_speech_pad_duration),
    ]
    if use_vad_merge:
        vad_command.extend(
            [
                "--merge-gap-duration",
                str(
                    args.merge_gap_duration
                    if args.merge_gap_duration is not None
                    else DEFAULT_MERGE_GAP_DURATION
                ),
                "--short-segment-duration",
                str(
                    args.short_segment_duration
                    if args.short_segment_duration is not None
                    else DEFAULT_SHORT_SEGMENT_DURATION
                ),
                "--max-merged-duration",
                str(
                    args.max_merged_duration
                    if args.max_merged_duration is not None
                    else DEFAULT_MAX_MERGED_DURATION
                ),
            ]
        )
    vad_option = "--silero-vad-model" if args.silero_vad_model else "--ten-vad-model"
    vad_command.extend([vad_option, str(args.silero_vad_model or args.ten_vad_model)])
    if args.debug:
        vad_command.append("--debug")
    vad_mode = "merge" if use_vad_merge else "normal"
    run_step(f"1/{total_steps} VAD ({vad_mode})", vad_command)

    run_step(
        f"2/{total_steps} ASR",
        [
            sys.executable,
            str(SCRIPT_DIR / "k2-offline-asr.py"),
            "--model-dir",
            str(Path(args.asr_model)),
            "--language",
            args.language,
            "--audio-dir",
            str(segments_dir),
            "--output",
            str(asr_output),
        ],
    )

    run_step(
        f"3/{total_steps} merge ASR",
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

    outputs = {
        "segments": segments_dir,
        "asr": asr_output,
        "merged": merged_output,
    }
    if run_wer:
        run_step(
            f"4/{total_steps} WER",
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
        print_wer_result(detail_output, args.metric)
        outputs["detail"] = detail_output

    return outputs


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
