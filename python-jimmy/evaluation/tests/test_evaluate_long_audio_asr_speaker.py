from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

EVALUATION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVALUATION_DIR))
import evaluate_long_audio_asr_speaker as runner  # noqa: E402


class ResultAndLabelParsingTest(unittest.TestCase):
    def _write_result(self, root: Path) -> Path:
        run = root / "run"
        run.mkdir()
        (run / "result.json").write_text(
            json.dumps(
                {
                    "audio_name": "23_asr_1782715267098.pcm",
                    "segments": [
                        {
                            "segment_id": "23_asr_1782715267098_1118_2308",
                            "duration_ms": 2308,
                            "speaker_id": "speaker_02",
                            "asr_text": "\u6765\u5427",
                        },
                        {
                            "segment_id": "23_asr_1782715267098_3426_6846",
                            "duration_ms": 6846,
                            "speaker_id": "speaker_00",
                            "asr_text": "\u53ef\u4ee5",
                        },
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return run

    def test_parse_result_resolve_label_and_write_public_projection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            results_dir = self._write_result(root)
            labels_dir = root / "labels"
            labels_dir.mkdir()
            (labels_dir / "23_asr_1782715267098_label.txt").write_text(
                "23_asr_1782715267098_1118_2335 (leslie) \u6765\u5427\n"
                "23_asr_1782715267098_3453_6823 (hui) \u53ef\u4ee5\n",
                encoding="utf-8",
            )
            records = runner.discover_result_records(results_dir)
            self.assertEqual(records[0].file_id, "23_asr_1782715267098")
            self.assertEqual(records[0].segments[0].start_ms, 1118)
            self.assertEqual(records[0].segments[0].end_ms, 3426)
            labels = runner.resolve_labels(records, labels_dir, explicit_label=None)
            self.assertEqual(
                labels[records[0].file_id].path.name,
                "23_asr_1782715267098_label.txt",
            )
            output = root / "out"
            output.mkdir()
            runner.write_public_result_txt(records, output / "result.txt")
            self.assertEqual(
                (output / "result.txt").read_text(encoding="utf-8"),
                "23_asr_1782715267098_1118_2308 speaker_02 \u6765\u5427\n"
                "23_asr_1782715267098_3426_6846 speaker_00 \u53ef\u4ee5\n",
            )
            runner.write_internal_asr_files(records, output / "inputs")
            self.assertTrue((output / "inputs" / "23_asr_1782715267098_asr.txt").is_file())

    def test_explicit_label_requires_a_single_result_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            records = [
                runner.ResultRecord(Path("one/result.json"), "one.pcm", "one", []),
                runner.ResultRecord(Path("two/result.json"), "two.pcm", "two", []),
            ]
            explicit = root / "one_label.txt"
            explicit.write_text("one_0_1000 (speaker) text\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "--label"):
                runner.resolve_labels(records, None, explicit)


if __name__ == "__main__":
    unittest.main()
