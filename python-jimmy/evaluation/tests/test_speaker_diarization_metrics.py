from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

EVALUATION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVALUATION_DIR))
import speaker_diarization_metrics as metrics  # noqa: E402


class SpeakerDiarizationMetricsTest(unittest.TestCase):
    def test_overlap_label_is_counted_and_skipped_for_der(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            results_dir = root / "results"
            labels_dir = root / "labels"
            results_dir.mkdir()
            labels_dir.mkdir()
            (results_dir / "audio_asr.txt").write_text(
                "audio_0_1000 speaker_00 \u7532\n"
                "audio_1000_1000 speaker_01 \u4e59\n",
                encoding="utf-8",
            )
            (labels_dir / "audio_label.txt").write_text(
                "audio_0_1000 (alice) \u7532\n"
                "audio_1000_500 (multi) \u91cd\u53e0\n"
                "audio_1500_500 (bob) \u4e59\n",
                encoding="utf-8",
            )
            report = metrics.evaluate_paths(
                results_dir=results_dir,
                labels_dir=labels_dir,
                boundary_tolerance_ms=500,
                collar_ms=0,
            )
            self.assertEqual(report.summary["metrics"]["overlap_segments"], 1)
            self.assertEqual(report.evaluated_count, 1)
            self.assertEqual(report.error_count, 0)


if __name__ == "__main__":
    unittest.main()
