#!/usr/bin/env python3

"""Recognize one audio file or a directory of WAV files with Paraformer."""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple


def resolve_paraformer_model_files(model_dir: Path) -> Tuple[Path, Path]:
    """Return the Paraformer model and tokens file from a model directory."""
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    model = next(
        (model_dir / filename for filename in ("model.int8.onnx", "model.onnx")
         if (model_dir / filename).is_file()),
        None,
    )
    if model is None:
        raise FileNotFoundError(
            f"Expected model.int8.onnx or model.onnx in {model_dir}"
        )

    tokens = model_dir / "tokens.txt"
    if not tokens.is_file():
        raise FileNotFoundError(f"Expected tokens.txt in {model_dir}")
    return model, tokens


def extract_text(result_line: str) -> str:
    """Remove the segment ID and optional confidence from an ASR result line."""
    fields = result_line.strip().split(maxsplit=2)
    if len(fields) < 2:
        raise ValueError(f"Invalid ASR result line: {result_line!r}")

    try:
        float(fields[1])
        if len(fields) == 2:
            return ""
        if len(fields) == 3:
            return fields[2]
    except ValueError:
        pass
    return " ".join(fields[1:])


def build_asr_short_command(
    asr_short: Path,
    wav_scp: Path,
    model: Path,
    tokens: Path,
    output: Path,
) -> List[str]:
    """Build the subprocess command for the existing short-audio ASR script."""
    return [
        sys.executable,
        str(asr_short),
        "--wav-scp", str(wav_scp),
        "--paraformer", str(model),
        "--tokens", str(tokens),
        "--output", str(output),
    ]


def build_asr_short_audio_dir_command(
    asr_short: Path,
    audio_dir: Path,
    model: Path,
    tokens: Path,
    output: Path,
) -> List[str]:
    """Build the subprocess command for directory-based short-audio ASR."""
    return [
        sys.executable,
        str(asr_short),
        "--audio-dir", str(audio_dir),
        "--paraformer", str(model),
        "--tokens", str(tokens),
        "--output", str(output),
    ]


def run_asr_short(command: List[str]) -> subprocess.CompletedProcess:
    """Run k2-asr-short.py while tolerating non-UTF-8 diagnostic output."""
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def recognize_one_audio(audio: Path, model_dir: Path) -> str:
    """Run k2-asr-short.py for one file and return only its recognized text."""
    if not audio.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio}")

    model, tokens = resolve_paraformer_model_files(model_dir)
    asr_short = Path(__file__).with_name("k2-asr-short.py")
    if not asr_short.is_file():
        raise FileNotFoundError(f"ASR script not found: {asr_short}")

    with tempfile.TemporaryDirectory(prefix="k2-offline-asr-") as directory:
        work_dir = Path(directory)
        wav_scp = work_dir / "input.scp"
        output = work_dir / "result.txt"
        wav_scp.write_text(f"audio {audio.resolve()}\n", encoding="utf-8")

        command = build_asr_short_command(asr_short, wav_scp, model, tokens, output)
        try:
            completed = run_asr_short(command)
        except subprocess.CalledProcessError as error:
            if error.stderr:
                print(error.stderr, end="", file=sys.stderr)
            raise RuntimeError(
                f"k2-asr-short.py failed with exit code {error.returncode}"
            ) from error

        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)
        if not output.is_file():
            raise RuntimeError("k2-asr-short.py did not create a result file")

        lines = [line for line in output.read_text(encoding="utf-8").splitlines() if line]
        if len(lines) != 1:
            raise RuntimeError(
                f"Expected one ASR result for one audio file, but got {len(lines)}"
            )
        return extract_text(lines[0])


def recognize_audio_dir(audio_dir: Path, model_dir: Path, output: Path) -> None:
    """Run k2-asr-short.py for a directory and write its results to output."""
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    model, tokens = resolve_paraformer_model_files(model_dir)
    asr_short = Path(__file__).with_name("k2-asr-short.py")
    if not asr_short.is_file():
        raise FileNotFoundError(f"ASR script not found: {asr_short}")

    command = build_asr_short_audio_dir_command(
        asr_short, audio_dir, model, tokens, output
    )
    try:
        completed = run_asr_short(command)
    except subprocess.CalledProcessError as error:
        if error.stderr:
            print(error.stderr, end="", file=sys.stderr)
        raise RuntimeError(
            f"k2-asr-short.py failed with exit code {error.returncode}"
        ) from error

    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if not output.is_file():
        raise RuntimeError("k2-asr-short.py did not create a result file")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recognize audio with an offline Paraformer Chinese model"
    )
    parser.add_argument("--model-dir", required=True, help="Paraformer model directory")
    parser.add_argument("--language", required=True, help="Recognition language; only zh is supported")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--audio", help="Audio file to recognize")
    input_group.add_argument(
        "--audio-dir", help="Directory containing first-level WAV files"
    )
    parser.add_argument("--output", help="Output file for directory recognition")
    args = parser.parse_args()
    if args.audio_dir and not args.output:
        parser.error("--output is required with --audio-dir")
    if args.audio and args.output:
        parser.error("--output is only supported with --audio-dir")
    return args


def main() -> int:
    args = parse_args()
    if args.language != "zh":
        print("Error: Only --language zh is supported", file=sys.stderr)
        return 2

    try:
        if args.audio_dir:
            recognize_audio_dir(
                Path(args.audio_dir), Path(args.model_dir), Path(args.output)
            )
            return 0
        text = recognize_one_audio(Path(args.audio), Path(args.model_dir))
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
