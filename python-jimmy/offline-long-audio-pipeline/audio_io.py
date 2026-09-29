"""Audio decoding and normalization helpers for the offline pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_SAMPLE_RATE = 16000


@dataclass(frozen=True)
class AudioFormat:
    kind: str
    sample_rate: int = TARGET_SAMPLE_RATE
    channels: int = 1
    sample_width: int = 2


@dataclass(frozen=True)
class LoadedAudio:
    samples: np.ndarray
    sample_rate: int

    @property
    def duration_ms(self) -> int:
        return int(self.samples.size * 1000 / self.sample_rate)


def _resample_to_target(samples: np.ndarray, source_rate: int) -> np.ndarray:
    if source_rate <= 0:
        raise ValueError("Audio sample rate must be positive")
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    if samples.size == 0:
        raise ValueError("Audio contains no samples")
    if source_rate == TARGET_SAMPLE_RATE:
        return samples

    target_size = int(round(samples.size * TARGET_SAMPLE_RATE / source_rate))
    if target_size <= 0:
        raise ValueError("Resampling produced no samples")
    source_positions = np.arange(samples.size, dtype=np.float64) / source_rate
    target_positions = np.arange(target_size, dtype=np.float64) / TARGET_SAMPLE_RATE
    return np.ascontiguousarray(
        np.interp(target_positions, source_positions, samples).astype(np.float32)
    )


def _load_pcm(path: Path, audio_format: AudioFormat) -> np.ndarray:
    if audio_format.sample_rate <= 0 or audio_format.channels <= 0:
        raise ValueError("PCM sample rate and channel count must be positive")
    if audio_format.sample_width != 2:
        raise ValueError("Only 16-bit little-endian PCM is supported")

    raw = path.read_bytes()
    frame_size = audio_format.channels * audio_format.sample_width
    if len(raw) % frame_size:
        raise ValueError("PCM file does not contain whole PCM frames")
    if not raw:
        raise ValueError("Audio contains no samples")

    frames = np.frombuffer(raw, dtype="<i2").reshape(-1, audio_format.channels)
    return np.ascontiguousarray(frames.astype(np.float32).mean(axis=1) / 32768.0)


def load_audio(path: Path, audio_format: AudioFormat) -> LoadedAudio:
    """Decode PCM/WAV and return mono 16 kHz float32 audio."""
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    kind = audio_format.kind.lower()
    if kind == "pcm":
        mono = _load_pcm(path, audio_format)
        source_rate = audio_format.sample_rate
    elif kind == "wav":
        waveform, source_rate = sf.read(str(path), always_2d=True, dtype="float32")
        if waveform.shape[0] == 0:
            raise ValueError("Audio contains no samples")
        mono = np.ascontiguousarray(waveform.mean(axis=1), dtype=np.float32)
    else:
        raise ValueError(f"Unsupported audio format: {audio_format.kind}")

    return LoadedAudio(_resample_to_target(mono, source_rate), TARGET_SAMPLE_RATE)

def write_segment_wav(path: Path, samples: np.ndarray) -> None:
    """Write one normalized diagnostic segment as 16 kHz PCM WAV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        str(path),
        np.ascontiguousarray(samples, dtype=np.float32),
        TARGET_SAMPLE_RATE,
        subtype="PCM_16",
    )
