"""Per-segment local Paraformer transcription."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from vad import SpeechSegment


MIN_ASR_SAMPLES = 2400
_MISSING = object()


def _get_text_confidence(result: Any) -> float:
    """Read sentence confidence while remaining compatible with old bindings."""
    native_confidence = getattr(result, "confidence", _MISSING)
    if native_confidence is not _MISSING and native_confidence is not None:
        try:
            confidence = float(native_confidence)
        except (TypeError, ValueError):
            confidence = None
        if confidence is not None and math.isfinite(confidence):
            return confidence

    tokens = getattr(result, "tokens", None)
    ys_log_probs = getattr(result, "ys_log_probs", None)
    try:
        if not tokens or not ys_log_probs or len(tokens) != len(ys_log_probs):
            return 0.0
        log_probs = [float(value) for value in ys_log_probs]
    except (TypeError, ValueError):
        return 0.0

    if not log_probs or not all(math.isfinite(value) for value in log_probs):
        return 0.0
    return math.exp(sum(log_probs) / len(log_probs))


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
            result = stream.result
            segment.asr_text = result.text.strip()
            segment.text_confidence = _get_text_confidence(result)
            segment.asr_error = None
        except Exception as error:
            segment.asr_text = "asr_error"
            segment.text_confidence = None
            segment.asr_error = str(error)
