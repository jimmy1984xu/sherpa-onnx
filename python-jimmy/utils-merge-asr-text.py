#!/usr/bin/env python3

"""Merge two-column ASR results into one punctuation-free utterance."""

import argparse
import unicodedata
from pathlib import Path


def remove_punctuation(text: str) -> str:
    """Remove punctuation characters from text without changing other content."""
    return "".join(
        character
        for character in text
        if not unicodedata.category(character).startswith("P")
    )


def read_asr_text(input_path: Path) -> str:
    """Read the text column from a two-column ASR result file."""
    if not input_path.is_file():
        raise FileNotFoundError(f"ASR result file not found: {input_path}")

    texts = []
    has_record = False
    for line in input_path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        fields = line.split(maxsplit=1)
        has_record = True
        texts.append(fields[1] if len(fields) == 2 else "")

    if not has_record:
        raise ValueError(f"No ASR records found in: {input_path}")
    return "".join(texts)


def merge_asr_text(
    input_path: Path,
    output_path: Path,
    utt_id: str = "utt001",
) -> None:
    """Write merged, punctuation-free ASR text as one two-column record."""
    if not utt_id or any(character.isspace() for character in utt_id):
        raise ValueError("utt_id must be a non-empty value without whitespace")

    merged_text = remove_punctuation(read_asr_text(input_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f"{utt_id} {merged_text}\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge two-column ASR results into one punctuation-free record"
    )
    parser.add_argument("--input", required=True, help="Two-column ASR result file")
    parser.add_argument("--output", required=True, help="Merged output file")
    parser.add_argument("--utt-id", default="utt001", help="Output utterance ID")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        merge_asr_text(Path(args.input), Path(args.output), args.utt_id)
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
