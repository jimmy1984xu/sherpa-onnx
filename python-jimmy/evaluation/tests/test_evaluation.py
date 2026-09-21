from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

EVALUATION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVALUATION_DIR))
import evaluation  # noqa: E402


class EvaluationTest(unittest.TestCase):
    def test_zh_text_normalization_and_detail_output(self) -> None:
        self.assertEqual(
            evaluation.text_normalization("\u4f60\u597d\uff0cABC", "ZH"),
            "\u4f60 \u597d ABC",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            label = root / "label.txt"
            hypothesis = root / "hyp.txt"
            detail = root / "wer_detail.txt"
            label.write_text("audio_1 \u4f60\u597d\u4e16\u754c\n", encoding="utf-8")
            hypothesis.write_text("audio_1 \u4f60\u597d\u4e16\u754c\n", encoding="utf-8")
            previous_argv = sys.argv
            try:
                sys.argv = [
                    "evaluation.py",
                    "--label", str(label),
                    "--hyp", str(hypothesis),
                    "--language", "ZH",
                    "--detail", str(detail),
                ]
                evaluation.main()
            finally:
                sys.argv = previous_argv
            detail_text = detail.read_text(encoding="utf-8")
            self.assertIn("id\twer\tref_words", detail_text)
            self.assertIn("audio_1", detail_text)


if __name__ == "__main__":
    unittest.main()
