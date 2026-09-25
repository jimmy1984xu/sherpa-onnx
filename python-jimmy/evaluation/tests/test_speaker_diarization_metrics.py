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
            summary_metrics = report.summary["metrics"]
            self.assertEqual(summary_metrics["overlap_segments"], 1)
            self.assertEqual(summary_metrics["reference_segments"], 2)
            self.assertEqual(summary_metrics["reference_speakers"], 2)
            self.assertEqual(summary_metrics["unknown"]["reference_unknown_ms"], 500)
            self.assertIsNone(summary_metrics["unknown"]["precision"])
            self.assertEqual(summary_metrics["unknown"]["recall"], 0.0)
            self.assertEqual(summary_metrics["unknown"]["f1"], 0.0)
            self.assertEqual(report.files[0]["der_components"]["false_alarm"], 0.0)
            self.assertEqual(report.evaluated_count, 1)
            self.assertEqual(report.error_count, 0)



class TypedSpeakerMetricContractTest(unittest.TestCase):
    def test_typed_unknown_regions_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            label = Path(temp_dir) / "audio_label.txt"
            label.write_text(
                "audio_0_1000 (alice)(单人) 甲\n"
                "audio_1000_1000 (MULTI)(重叠) 乙\n"
                "audio_2000_1000 (UNCLEAR)(听不清) 丙\n",
                encoding="utf-8",
            )
            segments = metrics.parse_speaker_file(label)
            self.assertEqual([item.segment_type for item in segments], ["单人", "重叠", "听不清"])
            self.assertEqual([item.is_unknown_reference for item in segments], [False, True, False])

    def test_invalid_parenthesized_type_fails_with_file_and_line_number(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            label = Path(temp_dir) / "invalid_label.txt"
            label.write_text("audio_0_1000 (alice)(未知类型) 甲\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"invalid_label.txt 第 1 行"):
                metrics.parse_speaker_file(label)

    def test_unclear_is_not_part_of_primary_unknown_duration_metric(self) -> None:
        references = [
            metrics.TimedSpeakerSegment("audio_0_1000", "UNCLEAR", 0, 1000, "听不清", "听不清"),
        ]
        predictions = [
            metrics.TimedSpeakerSegment("audio_0_1000", "UNKNOWN", 0, 1000, "", "单人"),
        ]
        summary = metrics.compute_unknown_duration_metrics(references, predictions)
        self.assertEqual(summary["reference_unknown_ms"], 0)
        self.assertIsNone(summary["recall"])
        self.assertEqual(summary["by_reference_type"]["unclear"]["reference_unknown_ms"], 1000)

    def test_unknown_duration_metrics_report_no_prediction_as_zero_recall(self) -> None:
        references = [
            metrics.TimedSpeakerSegment("audio_0_1000", "alice", 0, 1000, "甲", "单人"),
            metrics.TimedSpeakerSegment("audio_1000_1000", "MULTI", 1000, 2000, "乙", "重叠"),
        ]
        predictions = [
            metrics.TimedSpeakerSegment("audio_0_2000", "speaker_00", 0, 2000, "甲乙"),
        ]
        summary = metrics.compute_unknown_duration_metrics(references, predictions)
        self.assertIsNone(summary["precision"])
        self.assertEqual(summary["recall"], 0.0)
        self.assertEqual(summary["f1"], 0.0)
        self.assertEqual(summary["reference_unknown_ms"], 1000)
if __name__ == "__main__":
    unittest.main()
