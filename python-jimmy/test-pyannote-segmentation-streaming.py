#!/usr/bin/env python3
"""Direct PCM smoke runner for the pyannote SpeakerSegmentation streaming API."""

from __future__ import annotations

from pathlib import Path
import sys

PIPELINE_DIR = Path(__file__).resolve().parent / "offline-long-audio-pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from streaming_segmentation import main, parse_args, validate_args


if __name__ == "__main__":
    raise SystemExit(main())
