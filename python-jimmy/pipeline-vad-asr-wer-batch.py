#!/usr/bin/env python3

"""Run the VAD/ASR/WER pipeline for a Cartesian product of VAD parameters."""

import argparse
import csv
import importlib.util
import subprocess
import sys
from copy import copy
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
SINGLE_RUN_SCRIPT = SCRIPT_DIR / "pipeline-vad-asr-wer.py"
SINGLE_RUN_SPEC = importlib.util.spec_from_file_location(
    "pipeline_vad_asr_wer", SINGLE_RUN_SCRIPT
)
if SINGLE_RUN_SPEC is None or SINGLE_RUN_SPEC.loader is None:
    raise RuntimeError(f"Unable to load single-run pipeline: {SINGLE_RUN_SCRIPT}")

SINGLE_RUN = importlib.util.module_from_spec(SINGLE_RUN_SPEC)
SINGLE_RUN_SPEC.loader.exec_module(SINGLE_RUN)


SUMMARY_FIELDS = [
    "min_silence_duration",
    "pre_speech_pad_duration",
    "merge_gap_duration",
    "short_segment_duration",
    "max_merged_duration",
    "id",
    "wer",
    "ref_words",
    "err_words",
    "del_words",
    "ins_words",
    "detail_path",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Run pipeline-vad-asr-wer.py for every combination of VAD "
            "silence and pre-speech padding durations."
        ),
    )
    parser.add_argument("--audio", required=True, help="Input PCM/WAV audio file")
    vad_group = parser.add_mutually_exclusive_group(required=True)
    vad_group.add_argument("--silero-vad-model", help="Path to silero_vad.onnx")
    vad_group.add_argument("--ten-vad-model", help="Path to ten-vad.onnx")
    parser.add_argument("--asr-model", required=True, help="Paraformer model directory")
    parser.add_argument("--language", required=True, help="Recognition language")
    parser.add_argument("--label", default=None, help="Optional reference label file")
    parser.add_argument("--skip-wer", action="store_true")
    parser.add_argument("--output-dir", required=True, help="Pipeline output directory")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--min-silence-duration",
        nargs="+",
        type=float,
        default=[0.5],
        help="One or more minimum silence durations in seconds",
    )
    parser.add_argument("--min-speech-duration", type=float, default=0.25)
    parser.add_argument("--max-speech-duration", type=float, default=20.0)
    parser.add_argument(
        "--pre-speech-pad-duration",
        nargs="+",
        type=float,
        default=[0.0],
        help="One or more VAD pre-speech padding durations in seconds",
    )
    parser.add_argument(
        "--merge-gap-duration",
        nargs="+",
        type=float,
        default=None,
        help="One or more merge gap durations in seconds; enables merging",
    )
    parser.add_argument(
        "--short-segment-duration",
        nargs="+",
        type=float,
        default=None,
        help="One or more short-segment thresholds in seconds; enables merging",
    )
    parser.add_argument(
        "--max-merged-duration",
        nargs="+",
        type=float,
        default=None,
        help="One or more maximum merged durations in seconds; enables merging",
    )
    parser.add_argument("--utt-id", default="utt001")
    parser.add_argument("--metric", default="wer")
    parser.add_argument("--debug", action="store_true")
    return parser


def format_parameter(value: Optional[float]) -> str:
    """Keep user-visible floating-point directory names stable."""
    return "" if value is None else str(value)


def combination_output_dir(
    output_dir: Path,
    min_silence_duration: float,
    pre_speech_pad_duration: float,
    merge_gap_duration: Optional[float],
    short_segment_duration: Optional[float],
    max_merged_duration: Optional[float],
) -> Path:
    name = (
        f"min-silence-{format_parameter(min_silence_duration)}"
        f"_pre-speech-pad-{format_parameter(pre_speech_pad_duration)}"
    )
    if merge_gap_duration is not None:
        name += (
            f"_merge-gap-{format_parameter(merge_gap_duration)}"
            f"_short-segment-{format_parameter(short_segment_duration)}"
            f"_max-merged-{format_parameter(max_merged_duration)}"
        )
    return output_dir / name


def read_detail_rows(
    detail_path: Path,
    min_silence_duration: float,
    pre_speech_pad_duration: float,
    merge_gap_duration: Optional[float],
    short_segment_duration: Optional[float],
    max_merged_duration: Optional[float],
) -> List[Dict[str, str]]:
    with detail_path.open("r", encoding="utf-8", newline="") as detail_file:
        detail_rows = list(csv.DictReader(detail_file, delimiter="\t"))

    rows = []
    for detail_row in detail_rows:
        row = {
            "min_silence_duration": format_parameter(min_silence_duration),
            "pre_speech_pad_duration": format_parameter(pre_speech_pad_duration),
            "merge_gap_duration": format_parameter(merge_gap_duration),
            "short_segment_duration": format_parameter(short_segment_duration),
            "max_merged_duration": format_parameter(max_merged_duration),
            "detail_path": str(detail_path),
        }
        for field in SUMMARY_FIELDS:
            if field not in row:
                row[field] = detail_row.get(field, "")
        rows.append(row)

    return rows


def write_summary(output_dir: Path, rows: List[Dict[str, str]]) -> Path:
    summary_path = output_dir / "batch_wer_summary.tsv"
    with summary_path.open("w", encoding="utf-8", newline="") as summary_file:
        writer = csv.DictWriter(
            summary_file, fieldnames=SUMMARY_FIELDS, delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(rows)
    return summary_path


def parameter_combinations(
    args: argparse.Namespace,
) -> List[Tuple[float, float, Optional[float], Optional[float], Optional[float]]]:
    merge_enabled = any(
        value is not None
        for value in (
            args.merge_gap_duration,
            args.short_segment_duration,
            args.max_merged_duration,
        )
    )
    if not merge_enabled:
        merge_values = [(None, None, None)]
    else:
        merge_values = list(
            product(
                args.merge_gap_duration or [SINGLE_RUN.DEFAULT_MERGE_GAP_DURATION],
                args.short_segment_duration
                or [SINGLE_RUN.DEFAULT_SHORT_SEGMENT_DURATION],
                args.max_merged_duration
                or [SINGLE_RUN.DEFAULT_MAX_MERGED_DURATION],
            )
        )

    return [
        (min_silence, pre_pad, merge_gap, short_segment, max_merged)
        for min_silence, pre_pad, (merge_gap, short_segment, max_merged) in product(
            args.min_silence_duration,
            args.pre_speech_pad_duration,
            merge_values,
        )
    ]


def run_batch(args: argparse.Namespace) -> Optional[Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_wer = not SINGLE_RUN.skip_wer(args)
    summary_rows: List[Dict[str, str]] = []
    combinations = parameter_combinations(args)

    for index, (
        min_silence_duration,
        pre_speech_pad_duration,
        merge_gap_duration,
        short_segment_duration,
        max_merged_duration,
    ) in enumerate(
        combinations, start=1
    ):
        run_args = copy(args)
        run_args.min_silence_duration = min_silence_duration
        run_args.pre_speech_pad_duration = pre_speech_pad_duration
        run_args.merge_gap_duration = merge_gap_duration
        run_args.short_segment_duration = short_segment_duration
        run_args.max_merged_duration = max_merged_duration
        run_args.output_dir = str(
            combination_output_dir(
                output_dir,
                min_silence_duration,
                pre_speech_pad_duration,
                merge_gap_duration,
                short_segment_duration,
                max_merged_duration,
            )
        )

        print(
            "\n"
            f"[batch {index}/{len(combinations)}] "
            f"min_silence_duration={min_silence_duration}, "
            f"pre_speech_pad_duration={pre_speech_pad_duration}, "
            f"merge_gap_duration={merge_gap_duration}, "
            f"short_segment_duration={short_segment_duration}, "
            f"max_merged_duration={max_merged_duration}"
        )
        outputs = SINGLE_RUN.run_pipeline(run_args)

        if run_wer:
            detail_path = Path(outputs["detail"])
            summary_rows.extend(
                read_detail_rows(
                    detail_path,
                    min_silence_duration,
                    pre_speech_pad_duration,
                    merge_gap_duration,
                    short_segment_duration,
                    max_merged_duration,
                )
            )

    if not run_wer:
        return None

    summary_path = write_summary(output_dir, summary_rows)
    print(f"\nBatch WER summary: {summary_path}")
    return summary_path


def main() -> int:
    args = build_parser().parse_args()
    try:
        run_batch(args)
    except (
        FileNotFoundError,
        OSError,
        ValueError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"Batch pipeline failed: {error}", file=sys.stderr)
        return (
            error.returncode
            if isinstance(error, subprocess.CalledProcessError)
            else 1
        )

    print("\nBatch pipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
