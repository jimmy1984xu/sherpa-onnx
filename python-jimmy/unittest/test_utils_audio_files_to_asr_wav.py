#!/usr/bin/env python3

import importlib.util
import tempfile
import unittest
from pathlib import Path


def load_module():
    script_path = Path(__file__).resolve().parent.parent / "utils-audio_files_to_asr_wav.py"
    spec = importlib.util.spec_from_file_location("utils_audio_files_to_asr_wav", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestAudioFilesToAsrWav(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_convert_input_path_accepts_mp3_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "sample.mp3"
            input_path.write_bytes(b"mp3")

            calls = []

            def fake_convert_one(src, dst, overwrite, ffmpeg_bin):
                calls.append((src, dst, overwrite, ffmpeg_bin))
                return Path(tmpdir) / "sample.wav"

            result = self.module.convert_input_path(
                input_path=input_path,
                output_path=None,
                overwrite=True,
                ffmpeg_bin="ffmpeg",
                convert_one=fake_convert_one,
            )

            self.assertEqual(result, [Path(tmpdir) / "sample.wav"])
            self.assertEqual(calls, [(input_path, None, True, "ffmpeg")])

    def test_find_audio_files_includes_m4a_and_mp3(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir)
            (input_dir / "a.m4a").write_bytes(b"m4a")
            (input_dir / "b.mp3").write_bytes(b"mp3")
            (input_dir / "c.wav").write_bytes(b"wav")

            result = self.module.find_audio_files(input_dir)

            self.assertEqual([p.name for p in result], ["a.m4a", "b.mp3"])


if __name__ == "__main__":
    unittest.main()
