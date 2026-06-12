# VAD Frame Probability Script Design

## Goal

Add a `python-jimmy` command-line script that prints the VAD speech
probability for every complete VAD frame in a requested audio interval.

## Scope

The new script is `python-jimmy/k2-vad_prob.py`. It accepts a WAV file or a
raw PCM file, exactly one VAD model, and an interval described by
`offsetMs` and `durationMs`.

The supported raw PCM format is 16 kHz, mono, signed 16-bit little-endian.
WAV input must also be 16 kHz mono. The script rejects any WAV with a
different sample rate or channel count; it never resamples input audio.

## Interface

The command accepts these required arguments:

- `--audio`: Input WAV or raw PCM file.
- `--offset-ms`: Non-negative start position in milliseconds.
- `--duration-ms`: Positive interval duration in milliseconds.
- Exactly one of `--silero-vad-model` or `--ten-vad-model`.

The script detects WAV input from the `.wav` suffix. All other filenames are
read as raw PCM using the fixed format above. The requested interval must be
wholly inside the audio file. Invalid input, an invalid interval, or selecting
zero or two VAD models causes a clear error and non-zero exit status.

## Processing

The script creates a new `sherpa_onnx.VoiceActivityDetector` and processes
only the requested interval. Model state starts at `offsetMs`; audio before
that position is intentionally not fed to the detector.

Each model's native window size defines a frame:

- Silero VAD: 512 samples, or 32 ms at 16 kHz.
- TEN VAD: 256 samples, or 16 ms at 16 kHz.

For every complete frame, the script calls a new Python binding named
`VoiceActivityDetector.compute(samples)`. The binding directly delegates to
the existing C++ `VoiceActivityDetector::Compute()` method and returns the
model's unthresholded speech probability. This keeps model-specific state and
inference behavior within the established sherpa-onnx implementation.

The script discards a trailing partial frame and emits one diagnostic on
standard error identifying the dropped sample count. It writes no output
files.

## Output

Standard output has no header. Each complete frame produces one
space-separated line:

```text
frame_index offset_ms probability
```

`frame_index` begins at zero, `offset_ms` is the frame start on the original
audio time axis, and `probability` is the VAD speech probability. Diagnostics
and errors go to standard error, leaving standard output machine-readable.

## Testing

Add unit tests for the script's pure helpers without requiring a model:

- raw PCM decoding;
- WAV sample-rate and channel validation;
- interval validation and extraction;
- complete-frame splitting and trailing-partial-frame handling.

Add a Python binding-level test that constructs the VAD API only when a test
model is available, or otherwise verifies the `compute` method is exposed by
the built extension. Build the Python extension and run the targeted tests to
confirm the new binding and script work together.
