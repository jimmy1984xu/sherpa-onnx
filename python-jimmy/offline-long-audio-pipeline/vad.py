"""VAD segment domain model and incremental segment collection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

POST_OVERLAP_PAD_MS = 1000


@dataclass
class SpeechSegment:
    segment_index: int
    start_ms: int
    end_ms: int
    samples: np.ndarray = field(repr=False)
    asr_text: str = ""
    asr_error: str | None = None
    asr_language: str = ""
    whisper_language: str = ""
    whisper_lang_prob: float | None = None
    text_confidence: float | None = None
    asr_candidates: dict[str, dict[str, Any]] = field(default_factory=dict)
    asr_valid: int = 1
    embedding_error: str | None = None
    speaker_id: str = "unknown"
    previous_segment_similarity: float | None = None
    cluster_assignment_similarity: float | None = None
    speaker_composition: str = "unknown_activity"
    embedding: np.ndarray | None = field(default=None, repr=False)
    overlap_regions: list[tuple[int, int]] = field(default_factory=list)
    cut_left: str = "vad"
    cut_right: str = "vad"
    pyannote_mask: str = ""

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    @property
    def duration_class(self) -> str:
        return "long" if self.duration_ms >= 1000 else "short"

    def embedding_skip_regions(self, pad_ms: int = POST_OVERLAP_PAD_MS) -> list[tuple[int, int]]:
        """Overlap plus a short tail after each overlap, merged and clipped to the segment."""
        intervals = []
        for start_ms, end_ms in self.overlap_regions:
            intervals.append((start_ms, min(self.end_ms, end_ms + pad_ms)))
        intervals.sort()
        merged: list[tuple[int, int]] = []
        for start_ms, end_ms in intervals:
            if merged and start_ms <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end_ms))
            else:
                merged.append((start_ms, end_ms))
        return merged

    @property
    def exclusive_speech_duration_ms(self) -> int:
        skipped_ms = sum(end_ms - start_ms for start_ms, end_ms in self.embedding_skip_regions())
        return max(0, self.duration_ms - skipped_ms)

    @property
    def is_cluster_eligible(self) -> bool:
        return (
            self.asr_valid == 1
            and self.speaker_composition == "single_speaker"
            and self.exclusive_speech_duration_ms >= 1000
        )

    @property
    def segment_id(self) -> str:
        """Return the stable diagnostic WAV basename for this VAD segment."""
        return f"{self.segment_index:04d}_{self.start_ms}_{self.end_ms}"


def collect_vad_segments(
    vad: Any, waveform: np.ndarray, window_size: int, sample_rate: int
) -> list[SpeechSegment]:
    """Collect finalized speech regions from an incrementally-fed VAD."""
    if window_size <= 0 or sample_rate <= 0:
        raise ValueError("VAD window size and sample rate must be positive")

    segments: list[SpeechSegment] = []

    def drain() -> None:
        while not vad.empty():
            detected = vad.front
            samples = np.ascontiguousarray(
                np.asarray(detected.samples, dtype=np.float32)
            )
            start_ms = int(detected.start * 1000 / sample_rate)
            end_ms = start_ms + int(samples.size * 1000 / sample_rate)
            segments.append(
                SpeechSegment(
                    segment_index=len(segments) + 1,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    samples=samples,
                )
            )
            vad.pop()

    for start in range(0, waveform.size, window_size):
        vad.accept_waveform(waveform[start : start + window_size])
        drain()
    vad.flush()
    drain()
    return segments
