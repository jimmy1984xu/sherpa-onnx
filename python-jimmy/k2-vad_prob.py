#!/usr/bin/env python3

"""Print VAD speech probabilities for every frame in an audio interval."""

import argparse
import math
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import onnxruntime as ort
import soundfile as sf


SAMPLE_RATE = 16000
SILERO_V5_WINDOW_SIZE = 576
SILERO_V5_WINDOW_SHIFT = 512
TEN_VAD_WINDOW_SIZE = 768
TEN_VAD_FEATURE_DIM = 41
TEN_VAD_MEL_BINS = 40
TEN_VAD_FFT_SIZE = 1024


def load_audio(path: Path) -> np.ndarray:
    """Load 16 kHz mono WAV or 16 kHz mono S16LE PCM as float32 samples."""
    if not path.is_file():
        raise FileNotFoundError(f"Audio not found: {path}")

    if path.suffix.lower() == ".wav":
        data, sample_rate = sf.read(path, always_2d=True, dtype="float32")
        if sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"WAV sample rate must be 16 kHz, but got {sample_rate} Hz"
            )
        if data.shape[1] != 1:
            raise ValueError(
                f"WAV must be mono, but got {data.shape[1]} channels"
            )
        return np.ascontiguousarray(data[:, 0])

    pcm = np.fromfile(path, dtype="<i2")
    return np.ascontiguousarray(pcm.astype(np.float32) / 32768.0)


def extract_interval(
    samples: np.ndarray, offset_ms: int, duration_ms: int
) -> np.ndarray:
    """Return the requested audio interval, rejecting invalid boundaries."""
    if offset_ms < 0:
        raise ValueError("offset-ms must be non-negative")
    if duration_ms <= 0:
        raise ValueError("duration-ms must be positive")

    start = offset_ms * SAMPLE_RATE // 1000
    end = (offset_ms + duration_ms) * SAMPLE_RATE // 1000
    if end > len(samples):
        raise ValueError("Requested interval is outside the input audio")
    return np.ascontiguousarray(samples[start:end])


def split_complete_frames(
    samples: np.ndarray, frame_size: int
) -> Tuple[List[np.ndarray], int]:
    """Split samples into complete frames and return the trailing sample count."""
    complete_sample_count = len(samples) // frame_size * frame_size
    frames = [
        samples[start : start + frame_size]
        for start in range(0, complete_sample_count, frame_size)
    ]
    return frames, len(samples) - complete_sample_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print VAD speech probabilities for an audio interval",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--audio", required=True, help="Input WAV or S16LE PCM file")
    parser.add_argument(
        "--offset-ms", "--offsetMs", dest="offset_ms", type=int, required=True,
        help="Interval start in milliseconds",
    )
    parser.add_argument(
        "--duration-ms", "--durationMs", dest="duration_ms", type=int, required=True,
        help="Interval duration in milliseconds",
    )
    parser.add_argument(
        "--silero-vad-model", default="", help="Path to a Silero VAD ONNX model"
    )
    parser.add_argument(
        "--ten-vad-model", default="", help="Path to a TEN VAD ONNX model"
    )
    return parser.parse_args()


class SileroVadV5Runner:
    """Run a Silero v5 VAD model while carrying its recurrent state."""

    def __init__(self, session: ort.InferenceSession):
        self.session = session
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.window_size = SILERO_V5_WINDOW_SIZE
        self.window_shift = SILERO_V5_WINDOW_SHIFT

    def compute(self, frame: np.ndarray) -> float:
        if frame.ndim != 1 or frame.shape[0] not in (512, SILERO_V5_WINDOW_SIZE):
            raise ValueError(
                "Silero VAD v5 requires a 512-sample standalone frame or a "
                f"{SILERO_V5_WINDOW_SIZE}-sample VAD-cut window, got {frame.shape}"
            )

        output, self.state = self.session.run(
            ["output", "stateN"],
            {
                "input": np.ascontiguousarray(frame.reshape(1, -1), dtype=np.float32),
                "state": self.state,
                "sr": np.array(SAMPLE_RATE, dtype=np.int64),
            },
        )
        return float(output[0, 0])


def parse_float_metadata(metadata: dict, key: str, expected_size: int) -> np.ndarray:
    """Read a fixed-size float vector from the TEN model metadata."""
    try:
        values = np.fromstring(metadata[key], sep=",", dtype=np.float32)
    except KeyError as error:
        raise ValueError(f"TEN VAD model metadata is missing '{key}'") from error
    if values.size != expected_size:
        raise ValueError(
            f"TEN VAD model metadata '{key}' must contain {expected_size} values, "
            f"but got {values.size}"
        )
    return values


def hz_to_slaney_mel(frequencies: np.ndarray) -> np.ndarray:
    """Convert frequencies to the Slaney/librosa mel scale."""
    frequencies = np.asarray(frequencies, dtype=np.float32)
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0
    mels = frequencies / f_sp
    log_mask = frequencies >= min_log_hz
    mels[log_mask] = (
        min_log_mel + np.log(frequencies[log_mask] / min_log_hz) / logstep
    )
    return mels.astype(np.float32)


def slaney_mel_to_hz(mels: np.ndarray) -> np.ndarray:
    """Convert Slaney/librosa mel values to Hz."""
    mels = np.asarray(mels, dtype=np.float32)
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0
    return np.where(
        mels >= min_log_mel,
        min_log_hz * np.exp(logstep * (mels - min_log_mel)),
        f_sp * mels,
    ).astype(np.float32)


def make_ten_mel_filter_bank() -> np.ndarray:
    """Match TEN's 40-bin unnormalised Slaney/librosa mel banks."""
    low_mel = hz_to_slaney_mel(np.array([0.0], dtype=np.float32))[0]
    high_mel = hz_to_slaney_mel(np.array([SAMPLE_RATE / 2], dtype=np.float32))[0]
    mel_points = np.linspace(low_mel, high_mel, TEN_VAD_MEL_BINS + 2)
    hz_points = slaney_mel_to_hz(mel_points)
    fft_frequencies = np.arange(TEN_VAD_FFT_SIZE // 2, dtype=np.float32)
    fft_frequencies *= SAMPLE_RATE / TEN_VAD_FFT_SIZE
    ramps = hz_points[:, None] - fft_frequencies[None, :]
    widths = np.diff(hz_points)
    lower = -ramps[:-2] / widths[:-1, None]
    upper = ramps[2:] / widths[1:, None]
    return np.maximum(0.0, np.minimum(lower, upper)).astype(np.float32)


class TenVadRunner:
    """Run a TEN VAD model with the same frontend and state layout as C++."""

    def __init__(self, session: ort.InferenceSession):
        self.session = session
        metadata = session.get_modelmeta().custom_metadata_map
        if metadata.get("model_type") != "ten-vad":
            raise ValueError("TEN VAD model metadata 'model_type' must be 'ten-vad'")

        self.mean = parse_float_metadata(metadata, "mean", TEN_VAD_FEATURE_DIM)
        self.inv_stddev = parse_float_metadata(
            metadata, "inv_stddev", TEN_VAD_FEATURE_DIM
        )
        self.window = parse_float_metadata(metadata, "window", TEN_VAD_WINDOW_SIZE)
        self.mel_filter_bank = make_ten_mel_filter_bank()
        self.last_sample = np.float32(0.0)
        self.last_features = np.zeros((3, TEN_VAD_FEATURE_DIM), dtype=np.float32)
        self.states = [np.zeros((1, 64), dtype=np.float32) for _ in range(4)]
        self.window_size = TEN_VAD_WINDOW_SIZE
        self.window_shift = TEN_VAD_WINDOW_SIZE

        self.input_names = [item.name for item in session.get_inputs()]
        self.output_names = [item.name for item in session.get_outputs()]
        if len(self.input_names) != 5 or len(self.output_names) != 5:
            raise ValueError("TEN VAD model must have one feature input and four states")

    def compute_features(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim != 1 or frame.shape[0] != TEN_VAD_WINDOW_SIZE:
            raise ValueError(
                f"TEN VAD requires {TEN_VAD_WINDOW_SIZE}-sample frames, got {frame.shape}"
            )

        scaled = np.ascontiguousarray(frame, dtype=np.float32) * np.float32(32768.0)
        emphasized = np.empty(TEN_VAD_WINDOW_SIZE, dtype=np.float32)
        emphasized[0] = scaled[0] - np.float32(0.97) * self.last_sample
        emphasized[1:] = scaled[1:] - np.float32(0.97) * scaled[:-1]
        self.last_sample = scaled[-1]

        fft_input = np.zeros(TEN_VAD_FFT_SIZE, dtype=np.float32)
        fft_input[:TEN_VAD_WINDOW_SIZE] = emphasized * self.window
        spectrum = np.fft.rfft(fft_input)
        power = (spectrum.real * spectrum.real + spectrum.imag * spectrum.imag).astype(
            np.float32
        )
        log_mel = np.log(self.mel_filter_bank @ power[: TEN_VAD_FFT_SIZE // 2] + 1e-10)
        features = np.empty(TEN_VAD_FEATURE_DIM, dtype=np.float32)
        features[:TEN_VAD_MEL_BINS] = log_mel - np.float32(20.79441541679836)
        features[-1] = 0.0
        features = (features - self.mean) * self.inv_stddev
        self.last_features[:-1] = self.last_features[1:]
        self.last_features[-1] = features
        return self.last_features

    def compute(self, frame: np.ndarray) -> float:
        features = self.compute_features(frame)
        inputs = {self.input_names[0]: features.reshape(1, 3, TEN_VAD_FEATURE_DIM)}
        inputs.update(zip(self.input_names[1:], self.states))
        outputs = self.session.run(self.output_names, inputs)
        self.states = [np.ascontiguousarray(state, dtype=np.float32) for state in outputs[1:]]
        return float(np.asarray(outputs[0]).reshape(-1)[0])


def iter_vad_cut_probabilities(
    samples: np.ndarray,
    offset_ms: int,
    duration_ms: int,
    runner: SileroVadV5Runner,
):
    """Yield VAD-cut-equivalent probabilities inside the requested interval."""
    extract_interval(samples, offset_ms, duration_ms)
    interval_start = offset_ms * SAMPLE_RATE // 1000
    interval_end = (offset_ms + duration_ms) * SAMPLE_RATE // 1000

    window_size = getattr(runner, "window_size", SILERO_V5_WINDOW_SIZE)
    window_shift = getattr(runner, "window_shift", SILERO_V5_WINDOW_SHIFT)
    last_window_start = min(
        ((interval_end - 1) // window_shift) * window_shift,
        len(samples) - window_size,
    )
    if last_window_start < 0:
        return

    for window_start in range(0, last_window_start + 1, window_shift):
        probability = runner.compute(samples[window_start : window_start + window_size])
        if interval_start <= window_start < interval_end:
            offset = window_start * 1000 // SAMPLE_RATE
            yield offset, probability


def build_vad(args: argparse.Namespace):
    if bool(args.silero_vad_model) == bool(args.ten_vad_model):
        raise ValueError(
            "Specify exactly one of --silero-vad-model or --ten-vad-model"
        )

    model_path = Path(args.ten_vad_model or args.silero_vad_model)
    if not model_path.is_file():
        raise FileNotFoundError(f"VAD model not found: {model_path}")

    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3
    session = ort.InferenceSession(
        str(model_path), session_options, providers=["CPUExecutionProvider"]
    )
    if args.ten_vad_model:
        return TenVadRunner(session)

    input_names = [item.name for item in session.get_inputs()]
    output_names = [item.name for item in session.get_outputs()]
    if input_names != ["input", "state", "sr"] or output_names != ["output", "stateN"]:
        raise ValueError(
            "Only Silero VAD v5 ONNX models with input/state/sr and "
            "output/stateN tensors are supported"
        )
    return SileroVadV5Runner(session)


def main() -> None:
    args = parse_args()
    samples = load_audio(Path(args.audio))
    vad = build_vad(args)

    for frame_index, (frame_offset_ms, probability) in enumerate(
        iter_vad_cut_probabilities(
            samples, args.offset_ms, args.duration_ms, vad
        )
    ):
        print(f"{frame_index} {frame_offset_ms} {probability:.6f}")


if __name__ == "__main__":
    main()
