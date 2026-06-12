#!/usr/bin/env python3

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np


def load_module():
    script_path = Path(__file__).resolve().parent.parent / "k2-vad_cut.py"
    spec = importlib.util.spec_from_file_location("k2_vad_cut", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sherpa_onnx = sys.modules.setdefault("sherpa_onnx", types.SimpleNamespace())
    if not hasattr(sherpa_onnx, "VoiceActivityDetector"):
        sherpa_onnx.VoiceActivityDetector = object
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestK2VadCut(unittest.TestCase):
    def test_load_audio_mono_float32_reads_s16le_pcm_at_16khz(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio.pcm"
            np.array([-32768, 0, 32767], dtype="<i2").tofile(path)

            samples, sample_rate = module.load_audio_mono_float32(str(path))

        self.assertEqual(sample_rate, 16000)
        self.assertEqual(samples.dtype, np.float32)
        np.testing.assert_allclose(samples, [-1.0, 0.0, 32767 / 32768])


if __name__ == "__main__":
    unittest.main()
