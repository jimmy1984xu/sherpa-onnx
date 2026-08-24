#!/usr/bin/env python3

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def load_module():
    script_path = Path(__file__).resolve().parent.parent / "k2-offline-asr.py"
    spec = importlib.util.spec_from_file_location("k2_offline_asr", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestK2OfflineAsr(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_resolve_paraformer_model_files_prefers_int8_model(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            int8_model = model_dir / "model.int8.onnx"
            int8_model.touch()
            (model_dir / "model.onnx").touch()
            tokens = model_dir / "tokens.txt"
            tokens.touch()

            model, resolved_tokens = self.module.resolve_paraformer_model_files(model_dir)

        self.assertEqual(model, int8_model)
        self.assertEqual(resolved_tokens, tokens)

    def test_extract_text_removes_audio_id_and_optional_confidence(self):
        self.assertEqual(
            self.module.extract_text("audio 0.8750 你好，世界"), "你好，世界"
        )
        self.assertEqual(
            self.module.extract_text("audio hello world"), "hello world"
        )
        self.assertEqual(self.module.extract_text("audio 0.0000"), "")

    def test_build_asr_short_command_uses_current_python(self):
        command = self.module.build_asr_short_command(
            asr_short=Path("k2-asr-short.py"),
            wav_scp=Path("input.scp"),
            model=Path("model.int8.onnx"),
            tokens=Path("tokens.txt"),
            output=Path("result.txt"),
        )

        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[1], "k2-asr-short.py")
        self.assertEqual(command[2:], [
            "--wav-scp", "input.scp",
            "--paraformer", "model.int8.onnx",
            "--tokens", "tokens.txt",
            "--output", "result.txt",
        ])

    def test_build_asr_short_audio_dir_command_uses_audio_dir(self):
        command = self.module.build_asr_short_audio_dir_command(
            asr_short=Path("k2-asr-short.py"),
            audio_dir=Path("input-audio"),
            model=Path("model.int8.onnx"),
            tokens=Path("tokens.txt"),
            output=Path("result.txt"),
        )

        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[1], "k2-asr-short.py")
        self.assertEqual(command[2:], [
            "--audio-dir", "input-audio",
            "--paraformer", "model.int8.onnx",
            "--tokens", "tokens.txt",
            "--output", "result.txt",
        ])

    def test_parse_args_accepts_audio_dir_with_output(self):
        with patch.object(sys, "argv", [
            "k2-offline-asr.py",
            "--model-dir", "model",
            "--language", "zh",
            "--audio-dir", "input-audio",
            "--output", "results.txt",
        ]):
            args = self.module.parse_args()

        self.assertEqual(args.audio_dir, "input-audio")
        self.assertEqual(args.output, "results.txt")
        self.assertIsNone(args.audio)

    def test_run_asr_short_replaces_invalid_utf8_in_subprocess_logs(self):
        completed = self.module.run_asr_short([
            sys.executable,
            "-c",
            "import sys; sys.stderr.buffer.write(b'\\xc8')",
        ])

        self.assertEqual(completed.stderr, "\ufffd")


if __name__ == "__main__":
    unittest.main()
