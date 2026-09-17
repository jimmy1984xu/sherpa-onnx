#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert audio files to ASR-ready WAV format.

This script converts .m4a and .mp3 files to 16 kHz, mono, int16 PCM WAV.
It can process one file or recursively process a directory.

Examples:
python3 ./python-jimmy/utils-audio_files_to_asr_wav.py input.m4a

python3 ./python-jimmy/utils-audio_files_to_asr_wav.py input.mp3 -o output.wav

python3 ./python-jimmy/utils-audio_files_to_asr_wav.py ./audio_dir -o ./wav_dir

Notes:
- Supported input formats: .m4a and .mp3
- Output format: 16000Hz / mono / int16 PCM WAV
- ffmpeg must be installed and available in PATH
"""

from __future__ import annotations

import argparse
import subprocess
from collections.abc import Callable
from pathlib import Path

SUPPORTED_AUDIO_SUFFIXES = {".m4a", ".mp3"}


def default_output_path(input_path: Path) -> Path:
    return input_path.with_suffix(".wav")


def build_ffmpeg_command(
    input_path: Path,
    output_path: Path,
    overwrite: bool,
    ffmpeg_bin: str = "ffmpeg",
) -> list[str]:
    command = [ffmpeg_bin]
    command.append("-y" if overwrite else "-n")
    command.extend(
        [
            "-i",
            str(input_path),
            "-vn",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-sample_fmt",
            "s16",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
    )
    return command


def convert_audio_file_to_asr_wav(
    input_path: Path,
    output_path: Path | None = None,
    overwrite: bool = True,
    ffmpeg_bin: str = "ffmpeg",
) -> Path:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix not in SUPPORTED_AUDIO_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_AUDIO_SUFFIXES))
        raise ValueError(f"Input file must use one of {supported}: {input_path}")

    target_path = output_path if output_path is not None else default_output_path(input_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    command = build_ffmpeg_command(input_path, target_path, overwrite, ffmpeg_bin)
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg was not found. Install it and make sure it is in PATH.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg conversion failed with exit code: {exc.returncode}") from exc

    return target_path


def convert_m4a_to_asr_wav(
    input_path: Path,
    output_path: Path | None = None,
    overwrite: bool = True,
    ffmpeg_bin: str = "ffmpeg",
) -> Path:
    return convert_audio_file_to_asr_wav(input_path, output_path, overwrite, ffmpeg_bin)


def find_audio_files(input_dir: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES
        ],
        key=lambda path: (
            len(path.relative_to(input_dir).parts),
            str(path.relative_to(input_dir)).lower(),
        ),
    )


def convert_input_path(
    input_path: Path,
    output_path: Path | None = None,
    overwrite: bool = True,
    ffmpeg_bin: str = "ffmpeg",
    convert_one: Callable[[Path, Path | None, bool, str], Path] = convert_audio_file_to_asr_wav,
) -> list[Path]:
    if input_path.is_file():
        return [convert_one(input_path, output_path, overwrite, ffmpeg_bin)]

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    if not input_path.is_dir():
        raise ValueError(f"Input path is not a file or directory: {input_path}")

    audio_files = find_audio_files(input_path)
    if not audio_files:
        print(f"Warning: no .m4a or .mp3 files were found in: {input_path}")
        return []

    results: list[Path] = []
    for audio_path in audio_files:
        target_path = None
        if output_path is not None:
            relative_path = audio_path.relative_to(input_path).with_suffix(".wav")
            target_path = output_path / relative_path
        results.append(convert_one(audio_path, target_path, overwrite, ffmpeg_bin))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert .m4a or .mp3 audio to 16kHz/mono/int16 PCM WAV for ASR"
    )
    parser.add_argument("input_path", help="Input audio file path, or a directory that contains audio files")
    parser.add_argument(
        "-o",
        "--output",
        help="Output WAV path for a file input, or output directory for a directory input",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not overwrite existing output files",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        default="ffmpeg",
        help="ffmpeg executable path or command name",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output) if args.output else None

    result_paths = convert_input_path(
        input_path=input_path,
        output_path=output_path,
        overwrite=not args.no_overwrite,
        ffmpeg_bin=args.ffmpeg_bin,
    )
    for result_path in result_paths:
        print(f"Written: {result_path}")
    print(f"Converted files: {len(result_paths)}")
    print("Format: 16000Hz / mono / int16 PCM WAV")


if __name__ == "__main__":
    main()
