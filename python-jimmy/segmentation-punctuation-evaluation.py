"""Build and validate fair segmentation-punctuation evaluation invocations."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
STREAMING_INVARIANTS = {
    "--segmentation-chunk-ms": "32",
    "--segmentation-num-threads": "4",
    "--min-duration-on": "0.5",
    "--min-duration-off": "0.5",
    "--change-vote-threshold": "0.5",
}
_SPECIAL_VARIANT_OPTIONS = {
    "--audio",
    "--run-label",
    *STREAMING_INVARIANTS,
}
_REQUIRED_FAIR_OPTIONS = {
    "--num-clusters": "-1",
    "--cluster-threshold": "0.6",
}


@dataclass(frozen=True)
class ModelPaths:
    """Model directories supplied explicitly to both pipeline variants."""

    asr_dir: Path
    vad_dir: Path
    speaker_dir: Path
    segmentation_dir: Path


def build_common_arguments(paths: ModelPaths, output_root: Path) -> list[str]:
    """Return the arguments that must remain identical across variants."""
    return [
        "--output-root", str(output_root), "--audio-format", "pcm",
        "--sample-rate", "16000", "--channels", "1", "--sample-width", "2",
        "--asr-dir", str(paths.asr_dir), "--vad-dir", str(paths.vad_dir),
        "--speaker-dir", str(paths.speaker_dir), "--segmentation-dir", str(paths.segmentation_dir),
        "--asr-num-threads", "1", "--speaker-num-threads", "2",
        "--vad-threshold", "0.5", "--min-silence-duration", "0.8",
        "--min-speech-duration", "0.25", "--max-speech-duration", "25.0",
        "--pre-speech-pad-duration", "0.0", "--cluster-threshold", "0.6",
        "--num-clusters", "-1", "--min-cluster-duration", "1.0",
        "--centroid-assignment-similarity-threshold", "0.5",
        "--diarization-min-duration-on", "0.5", "--diarization-min-duration-off", "0.5",
        "--asr-engine", "paraformer", "--segmentation-mode", "vad-pyannote",
    ]


def build_invocation(kind: str, pcm: Path, common: list[str]) -> list[str]:
    """Build one pipeline command without executing it."""
    script = {
        "baseline": "offline-long-audio-pipeline-asr-speaker.py",
        "streaming": "offline-long-audio-pipeline-asr-speaker-segmentation.py",
    }[kind]
    invocation = [
        sys.executable,
        str(SCRIPT_DIR / script),
        "--audio",
        str(pcm),
        *common,
    ]
    if kind == "streaming":
        for option, value in STREAMING_INVARIANTS.items():
            invocation.extend([option, value])
    return invocation


def _option_map(argv: Sequence[str]) -> dict[str, str]:
    """Parse long options that each take exactly one value."""
    options: dict[str, str] = {}
    index = 0
    while index < len(argv):
        token = argv[index]
        if not token.startswith("--"):
            index += 1
            continue
        if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
            raise ValueError(f"option requires one value: {token}")
        if token in options:
            raise ValueError(f"duplicate option: {token}")
        options[token] = argv[index + 1]
        index += 2
    return options


def option_value(argv: Sequence[str], option: str) -> str:
    """Return an option value or raise a descriptive error when it is absent."""
    try:
        return _option_map(argv)[option]
    except KeyError as error:
        raise ValueError(f"missing option: {option}") from error


def common_option_map(argv: Sequence[str]) -> dict[str, str]:
    """Return the option mapping that must be equal for baseline and streaming."""
    return {
        option: value
        for option, value in _option_map(argv).items()
        if option not in _SPECIAL_VARIANT_OPTIONS
    }


def validate_variant_fairness(
    baseline_argv: Sequence[str], streaming_argv: Sequence[str]
) -> None:
    """Verify only approved variant-specific invocation differences exist."""
    baseline_options = _option_map(baseline_argv)
    streaming_options = _option_map(streaming_argv)

    for option, expected_value in STREAMING_INVARIANTS.items():
        if option in baseline_options:
            raise ValueError(f"baseline invocation must not set {option}")
        if streaming_options.get(option) != expected_value:
            raise ValueError(
                f"streaming {option} must equal {expected_value}, "
                f"got {streaming_options.get(option)!r}"
            )

    if common_option_map(baseline_argv) != common_option_map(streaming_argv):
        raise ValueError("baseline and streaming common arguments differ")

    for option, expected_value in _REQUIRED_FAIR_OPTIONS.items():
        for name, options in (
            ("baseline", baseline_options),
            ("streaming", streaming_options),
        ):
            if options.get(option) != expected_value:
                raise ValueError(
                    f"{name} {option} must equal {expected_value}, "
                    f"got {options.get(option)!r}"
                )


def _segment_times(segment: dict[str, object]) -> tuple[int, int]:
    """Parse the trailing ``_<start_ms>_<end_ms>`` fields from a segment ID."""
    segment_id = segment.get("segment_id")
    if not isinstance(segment_id, str):
        raise ValueError("segment_id must be a string ending in _<start_ms>_<end_ms>")

    parts = segment_id.rsplit("_", 2)
    if len(parts) != 3 or not parts[0] or not parts[1].isdigit() or not parts[2].isdigit():
        raise ValueError(f"malformed segment_id: {segment_id!r}")

    start_ms, end_ms = int(parts[1]), int(parts[2])
    if end_ms <= start_ms:
        raise ValueError(f"invalid segment interval: {start_ms}-{end_ms}")

    if "duration_ms" in segment:
        try:
            duration_ms = int(segment["duration_ms"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid duration_ms for {segment_id!r}") from error
        if duration_ms != end_ms - start_ms:
            raise ValueError(
                f"inconsistent duration_ms for {segment_id!r}: "
                f"expected {end_ms - start_ms}, got {duration_ms}"
            )
    return start_ms, end_ms


def write_metrics_asr_file(output_dir: Path, file_id: str, result: dict[str, object]) -> Path:
    """Convert result.json segments to the three-column diarization metric format."""
    rows: list[str] = []
    for item in result["segments"]:
        if not isinstance(item, dict):
            raise ValueError("result segment must be an object")
        start_ms, end_ms = _segment_times(item)
        speaker = str(item.get("speaker_id", "-")).strip().strip("()") or "-"
        text = str(item.get("asr_text", "")).replace("\r", " ").replace("\n", " ").strip()
        rows.append(f"{file_id}_{start_ms}_{end_ms - start_ms} {speaker} {text}".rstrip())

    output_path = output_dir / f"{file_id}_asr.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return output_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse a command-preview CLI; execution is intentionally out of scope."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("baseline", "streaming"), required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--asr-dir", type=Path, required=True)
    parser.add_argument("--vad-dir", type=Path, required=True)
    parser.add_argument("--speaker-dir", type=Path, required=True)
    parser.add_argument("--segmentation-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Print a reproducible command preview without accessing data or models."""
    args = parse_args(argv)
    paths = ModelPaths(
        asr_dir=args.asr_dir,
        vad_dir=args.vad_dir,
        speaker_dir=args.speaker_dir,
        segmentation_dir=args.segmentation_dir,
    )
    command = build_invocation(
        args.kind,
        args.audio,
        build_common_arguments(paths, args.output_root),
    )
    print(json.dumps({"argv": command}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
