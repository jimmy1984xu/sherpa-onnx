#!/usr/bin/env python3

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def load_module():
    script_path = Path(__file__).resolve().parent.parent / "k2-asr-short.py"
    spec = importlib.util.spec_from_file_location("k2_asr_short", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestK2AsrShort(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_read_audio_dir_returns_sorted_first_level_wav_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            audio_dir = Path(directory)
            first = audio_dir / "b name.WAV"
            first.touch()
            second = audio_dir / "a.wav"
            second.touch()
            (audio_dir / "ignored.mp3").touch()
            nested_dir = audio_dir / "nested"
            nested_dir.mkdir()
            (nested_dir / "nested.wav").touch()

            entries = self.module.read_audio_dir(audio_dir)

        self.assertEqual(entries, [
            ("a", str(second.resolve())),
            ("b_name", str(first.resolve())),
        ])

    def test_read_audio_dir_disambiguates_colliding_segment_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            audio_dir = Path(directory)
            first = audio_dir / "same    name.wav"
            first.touch()
            second = audio_dir / "same name.wav"
            second.touch()

            entries = self.module.read_audio_dir(audio_dir)

        self.assertEqual([entry[0] for entry in entries], [
            "same_name",
            "same_name_2",
        ])

    def test_read_audio_dir_rejects_directory_without_wav_files(self):
        with tempfile.TemporaryDirectory() as directory:
            audio_dir = Path(directory)
            (audio_dir / "ignored.mp3").touch()

            with self.assertRaisesRegex(ValueError, "No WAV files found"):
                self.module.read_audio_dir(audio_dir)

    def test_parse_args_accepts_audio_dir_instead_of_wav_scp(self):
        with patch.object(sys, "argv", [
            "k2-asr-short.py",
            "--audio-dir", "input-audio",
            "--output", "results.txt",
        ]):
            args = self.module.parse_args()

        self.assertEqual(args.audio_dir, "input-audio")
        self.assertIsNone(args.wav_scp)

    def test_parse_args_rejects_both_input_sources(self):
        with patch.object(sys, "argv", [
            "k2-asr-short.py",
            "--audio-dir", "input-audio",
            "--wav-scp", "input.scp",
            "--output", "results.txt",
        ]):
            with self.assertRaises(SystemExit) as error:
                self.module.parse_args()

        self.assertEqual(error.exception.code, 2)

    def test_write_results_without_confidence_uses_id_and_text_only(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.txt"

            self.module.write_results(
                output,
                [("clip", "recognized text", 0.875)],
                include_confidence=False,
            )

            contents = output.read_text(encoding="utf-8")

        self.assertEqual(contents, "clip recognized text\n")


if __name__ == "__main__":
    unittest.main()
