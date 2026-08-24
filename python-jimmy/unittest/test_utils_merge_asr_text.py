#!/usr/bin/env python3

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


def load_module():
    script_path = Path(__file__).resolve().parent.parent / "utils-merge-asr-text.py"
    spec = importlib.util.spec_from_file_location("utils_merge_asr_text", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestUtilsMergeAsrText(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_merge_asr_text_removes_punctuation_and_writes_default_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "asr.txt"
            output_path = root / "merged.txt"
            input_path.write_text(
                "segment-1 来吧，我们聊。\n"
                "segment-2 正常聊天就行！可以，直接讲。\n"
                "segment-3\n",
                encoding="utf-8",
            )

            self.module.merge_asr_text(input_path, output_path)

            contents = output_path.read_text(encoding="utf-8")

        self.assertEqual(contents, "utt001 来吧我们聊正常聊天就行可以直接讲\n")

    def test_merge_asr_text_uses_custom_utterance_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "asr.txt"
            output_path = root / "merged.txt"
            input_path.write_text("segment hello, world!\n", encoding="utf-8")

            self.module.merge_asr_text(input_path, output_path, "answer-001")

            contents = output_path.read_text(encoding="utf-8")

        self.assertEqual(contents, "answer-001 hello world\n")

    def test_read_asr_text_rejects_empty_input_file(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "asr.txt"
            input_path.write_text("\n\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "No ASR records found"):
                self.module.read_asr_text(input_path)


if __name__ == "__main__":
    unittest.main()
