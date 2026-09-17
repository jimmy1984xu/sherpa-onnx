"""Per-segment local Paraformer transcription."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from vad import SpeechSegment


MIN_ASR_SAMPLES = 2400


def transcribe_segments(
    recognizer: Any, segments: Iterable[SpeechSegment], sample_rate: int
) -> None:
    """Transcribe every segment and retain individual failures in its record."""
    if sample_rate <= 0:
        raise ValueError("ASR sample rate must be positive")

    for segment in segments:
        if segment.samples.size < MIN_ASR_SAMPLES:
            segment.asr_text = "asr_error"
            segment.asr_error = (
                "segment is shorter than the Paraformer minimum input "
                f"of {MIN_ASR_SAMPLES} samples"
            )
            continue
        try:
            stream = recognizer.create_stream()
            stream.accept_waveform(sample_rate, segment.samples)
            recognizer.decode_stream(stream)
            segment.asr_text = stream.result.text.strip()
            segment.asr_error = None
        except Exception as error:
            segment.asr_text = "asr_error"
            segment.asr_error = str(error)
