#!/usr/bin/env python3

import importlib.util
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path

import numpy as np


def load_module():
    script_path = Path(__file__).resolve().parent.parent / "k2-vad_prob.py"
    spec = importlib.util.spec_from_file_location("k2_vad_prob", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules.setdefault("sherpa_onnx", types.ModuleType("sherpa_onnx"))
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestK2VadProb(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_load_raw_pcm_as_normalized_float32(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio.pcm"
            np.array([-32768, 0, 32767], dtype="<i2").tofile(path)

            samples = self.module.load_audio(path)

        np.testing.assert_allclose(samples, [-1.0, 0.0, 32767 / 32768])
        self.assertEqual(samples.dtype, np.float32)

    def test_load_wav_rejects_non_16khz_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio.wav"
            with wave.open(str(path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(8000)
                wav_file.writeframes(b"\x00\x00" * 80)

            with self.assertRaisesRegex(ValueError, "16 kHz"):
                self.module.load_audio(path)

    def test_extract_interval_requires_in_range_positive_duration(self):
        samples = np.arange(1600, dtype=np.float32)

        interval = self.module.extract_interval(samples, offset_ms=25, duration_ms=50)

        np.testing.assert_array_equal(interval, samples[400:1200])
        with self.assertRaisesRegex(ValueError, "positive"):
            self.module.extract_interval(samples, offset_ms=0, duration_ms=0)
        with self.assertRaisesRegex(ValueError, "outside"):
            self.module.extract_interval(samples, offset_ms=75, duration_ms=50)

    def test_split_complete_frames_reports_trailing_samples(self):
        frames, trailing_samples = self.module.split_complete_frames(
            np.arange(1300, dtype=np.float32), frame_size=512
        )

        self.assertEqual(len(frames), 2)
        np.testing.assert_array_equal(frames[0], np.arange(512, dtype=np.float32))
        np.testing.assert_array_equal(frames[1], np.arange(512, 1024, dtype=np.float32))
        self.assertEqual(trailing_samples, 276)

    def test_silero_v5_runner_preserves_recurrent_state(self):
        class FakeSession:
            def __init__(self):
                self.inputs = []

            def run(self, output_names, inputs):
                self.inputs.append(inputs)
                state = np.full((2, 1, 128), len(self.inputs), dtype=np.float32)
                return [np.array([[0.25]], dtype=np.float32), state]

        session = FakeSession()
        runner = self.module.SileroVadV5Runner(session)

        self.assertEqual(runner.compute(np.zeros(512, dtype=np.float32)), 0.25)
        self.assertEqual(runner.compute(np.ones(512, dtype=np.float32)), 0.25)

        self.assertEqual(session.inputs[0]["input"].shape, (1, 512))
        self.assertEqual(session.inputs[0]["sr"].dtype, np.int64)
        np.testing.assert_array_equal(
            session.inputs[0]["state"], np.zeros((2, 1, 128), dtype=np.float32)
        )
        np.testing.assert_array_equal(
            session.inputs[1]["state"], np.ones((2, 1, 128), dtype=np.float32)
        )

    def test_ten_runner_uses_768_sample_frames_and_preserves_all_states(self):
        class FakeSession:
            def __init__(self):
                self.inputs = []

            def get_modelmeta(self):
                return types.SimpleNamespace(
                    custom_metadata_map={
                        "model_type": "ten-vad",
                        "mean": ",".join(["0"] * 41),
                        "inv_stddev": ",".join(["1"] * 41),
                        "window": ",".join(["1"] * 768),
                    }
                )

            def get_inputs(self):
                return [
                    types.SimpleNamespace(name=name)
                    for name in ("input_1", "input_2", "input_3", "input_6", "input_7")
                ]

            def get_outputs(self):
                return [
                    types.SimpleNamespace(name=name)
                    for name in ("output_1", "output_2", "output_3", "output_6", "output_7")
                ]

            def run(self, output_names, inputs):
                self.inputs.append(inputs)
                state = np.full((1, 64), len(self.inputs), dtype=np.float32)
                return [np.array([[[0.75]]], dtype=np.float32)] + [state] * 4

        session = FakeSession()
        runner = self.module.TenVadRunner(session)

        self.assertEqual(runner.compute(np.zeros(768, dtype=np.float32)), 0.75)
        self.assertEqual(runner.compute(np.ones(768, dtype=np.float32)), 0.75)

        self.assertEqual(runner.window_size, 768)
        self.assertEqual(runner.window_shift, 768)
        self.assertEqual(session.inputs[0]["input_1"].shape, (1, 3, 41))
        np.testing.assert_array_equal(
            session.inputs[0]["input_2"], np.zeros((1, 64), dtype=np.float32)
        )
        np.testing.assert_array_equal(
            session.inputs[1]["input_2"], np.ones((1, 64), dtype=np.float32)
        )

    def test_vad_cut_probabilities_warm_up_from_audio_start(self):
        class FakeRunner:
            def __init__(self):
                self.frames = []

            def compute(self, frame):
                self.frames.append(frame.copy())
                return float(frame[0])

        samples = np.arange(2048, dtype=np.float32)
        runner = FakeRunner()

        probabilities = list(
            self.module.iter_vad_cut_probabilities(
                samples, offset_ms=32, duration_ms=64, runner=runner
            )
        )

        self.assertEqual([item[0] for item in probabilities], [32, 64])
        self.assertEqual([item[1] for item in probabilities], [512.0, 1024.0])
        self.assertEqual(len(runner.frames), 3)
        self.assertEqual(runner.frames[0].shape, (576,))
        np.testing.assert_array_equal(runner.frames[0], samples[:576])

    def test_ten_probabilities_warm_up_with_768_sample_shift(self):
        class FakeTenRunner:
            window_size = 768
            window_shift = 768

            def __init__(self):
                self.frames = []

            def compute(self, frame):
                self.frames.append(frame.copy())
                return float(frame[0])

        samples = np.arange(3072, dtype=np.float32)
        runner = FakeTenRunner()

        probabilities = list(
            self.module.iter_vad_cut_probabilities(
                samples, offset_ms=48, duration_ms=96, runner=runner
            )
        )

        self.assertEqual([item[0] for item in probabilities], [48, 96])
        self.assertEqual([item[1] for item in probabilities], [768.0, 1536.0])
        self.assertEqual(len(runner.frames), 3)
        self.assertEqual(runner.frames[0].shape, (768,))


if __name__ == "__main__":
    unittest.main()
