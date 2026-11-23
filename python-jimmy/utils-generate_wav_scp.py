#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
遍历音频文件夹中的 WAV 文件并生成标准 wav.scp。

wav.scp 每行格式：
  utt_id /absolute/path/to/audio.wav
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


SAFE_UTT_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def find_wav_files(audio_dir: Path) -> list[Path]:
    return sorted(
        [path for path in audio_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".wav"],
        key=lambda path: (
            len(path.relative_to(audio_dir).parts),
            str(path.relative_to(audio_dir)).lower(),
        ),
    )


def make_utt_id(audio_dir: Path, wav_path: Path) -> str:
    relative_stem = wav_path.relative_to(audio_dir).with_suffix("")
    raw_id = "_".join(relative_stem.parts)
    utt_id = SAFE_UTT_ID_RE.sub("_", raw_id).strip("._")
    return utt_id or wav_path.stem


def build_wav_scp_entries(audio_dir: Path) -> list[tuple[str, Path]]:
    return [(make_utt_id(audio_dir, wav_path), wav_path.resolve()) for wav_path in find_wav_files(audio_dir)]


def write_wav_scp(entries: list[tuple[str, Path]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for utt_id, wav_path in entries:
            file.write(f"{utt_id} {wav_path}\n")


def generate_wav_scp(audio_dir: Path, output_path: Path | None = None) -> int:
    if not audio_dir.exists():
        raise FileNotFoundError(f"音频文件夹不存在: {audio_dir}")
    if not audio_dir.is_dir():
        raise ValueError(f"输入路径不是文件夹: {audio_dir}")

    target_path = output_path if output_path is not None else audio_dir / "wav.scp"
    entries = build_wav_scp_entries(audio_dir)
    write_wav_scp(entries, target_path)
    return len(entries)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="递归遍历 WAV 文件并生成 wav.scp")
    parser.add_argument("audio_dir", help="输入音频文件夹路径")
    parser.add_argument(
        "-o",
        "--output",
        help="输出 wav.scp 路径。默认生成在音频文件夹下",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audio_dir = Path(args.audio_dir)
    output_path = Path(args.output) if args.output else audio_dir / "wav.scp"

    count = generate_wav_scp(audio_dir, output_path)
    print(f"输入目录: {audio_dir}")
    print(f"输出文件: {output_path}")
    print(f"写入 wav 数量: {count}")


if __name__ == "__main__":
    main()
