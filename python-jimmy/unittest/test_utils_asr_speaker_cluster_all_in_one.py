#!/usr/bin/env python3

import importlib.util
import sys
import types
import unittest
from pathlib import Path


def load_module():
    if "sherpa_onnx" not in sys.modules:
        sys.modules["sherpa_onnx"] = types.SimpleNamespace()

    script_path = Path(__file__).resolve().parent.parent / "utils-asr_speaker_cluster_all_in_one.py"
    spec = importlib.util.spec_from_file_location("utils_asr_speaker_cluster_all_in_one", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestUtilsAsrSpeakerClusterAllInOne(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_parse_plain_log_line_maps_multi_metadata_to_multi_speaker(self):
        segment = self.module.parse_plain_log_line(
            "1112030 20422 0 multi=1;angles=[77:578 114:1088] 这是一句测试文本",
            1,
        )

        self.assertEqual(segment.offset_ms, 1112030)
        self.assertEqual(segment.duration_ms, 20422)
        self.assertEqual(segment.speaker, "Multi")
        self.assertEqual(segment.score, "NA")
        self.assertEqual(segment.text, "这是一句测试文本")

    def test_parse_finished_log_line_maps_multi_metadata_to_multi_speaker(self):
        segment = self.module.parse_finished_log_line(
            "[2026-06-24 19:27:57] [asr_1782297438824] [ASR] Finished(541736/4934) -> 0: 就办法，他这个拖到后面了吧，不你你你不是这样，你把就是依赖强的。   |extraInfo:multi=1;angles=[145:1538 100:1536 106:672 112:1188]",
            1,
        )

        assert segment is not None
        self.assertEqual(segment.offset_ms, 541736)
        self.assertEqual(segment.duration_ms, 4934)
        self.assertEqual(segment.speaker, "Multi")
        self.assertEqual(segment.score, "NA")

    def test_filter_segments_skips_multi_and_counts_reasons(self):
        segments = [
            {"id": "audio_100_1000_Multi_NA", "durationMs": 1000, "speaker": "Multi"},
            {"id": "audio_1100_2000_unknown_NA", "durationMs": 2000, "speaker": "unknown"},
            {"id": "audio_2200_4000_Alice_0.9", "durationMs": 4000, "speaker": "Alice"},
        ]
        skip_counters = self.module.create_skip_counters()

        filtered = self.module.filter_segments_for_clustering(
            segments,
            min_duration_ms=3000,
            skip_counters=skip_counters,
        )

        self.assertEqual([segment["id"] for segment in filtered], ["audio_2200_4000_Alice_0.9"])
        self.assertEqual(skip_counters["multi"], 1)
        self.assertEqual(skip_counters["short"], 1)
        self.assertEqual(skip_counters["invalid_embedding"], 0)

    def test_load_segments_from_embeddings_marks_multi_speaker_from_segment_id(self):
        segments = self.module.load_segments_from_embeddings(
            [
                ("1_1000_4000_Multi_NA", "AAAA"),
                ("1_5000_4000_BenitaChen_0.677", "BBBB"),
            ]
        )

        self.assertEqual(segments[0]["speaker"], "Multi")
        self.assertEqual(segments[1]["speaker"], "BenitaChen")

    def test_build_cluster_summary_lines_include_skipped_counts_valid_speakers_and_rules(self):
        results = [
            ("1_1021726_12134_CocoDai_0.678", 1),
            ("1_1040638_5670_KarlHe_0.563", 2),
        ]
        cluster_name_map = {
            1: "S1",
            2: "invalid",
        }

        lines = self.module.build_cluster_summary_lines(
            results=results,
            cluster_name_map=cluster_name_map,
            input_segment_count=6,
            min_duration_ms=3000,
            num_clusters= -1,
            threshold=0.65,
            skipped_multi_segments=2,
            skipped_short_segments=1,
            skipped_invalid_embedding_segments=1,
        )

        self.assertIn("# input_segments: 6", lines)
        self.assertIn("# skipped_multi_segments: 2", lines)
        self.assertIn("# skipped_short_segments: 1", lines)
        self.assertIn("# skipped_invalid_embedding_segments: 1", lines)
        self.assertIn("# clustered_segments: 2", lines)
        self.assertIn("# cluster_number: 2", lines)
        self.assertIn("# valid_speakers: S1", lines)
        self.assertIn("# rules:", lines)
        self.assertIn(
            "# rule_registered_primary: count>=3 and ratio>0.70 and avg_score>=0.60 and max_score>=0.70",
            lines,
        )
        self.assertIn(
            "# rule_registered_fallback: count>=3 and ratio>0.80 and avg_score>=0.57 and max_score>=0.68",
            lines,
        )
        self.assertIn(
            "# rule_unknown: if no registered speaker matches, assign S<n> when cluster_count>10 and cluster_max_score>0.55",
            lines,
        )
        self.assertIn(
            "# rule_invalid: if neither registered nor unknown rule matches, mark cluster as invalid",
            lines,
        )
        self.assertIn(
            "# rule_tiebreak: prefer larger count, then larger avg_score, then larger max_score",
            lines,
        )
        self.assertFalse(any(line.startswith("# reason_") for line in lines))

    def test_multi_segments_loaded_from_embeddings_are_skipped_from_clustering(self):
        segments = self.module.load_segments_from_embeddings(
            [
                ("1_1000_4000_Multi_NA", "AAAA"),
                ("1_5000_4000_BenitaChen_0.677", "BBBB"),
                ("1_9000_4000_BenitaChen_0.653", "CCCC"),
            ]
        )
        skip_counters = self.module.create_skip_counters()

        filtered = self.module.filter_segments_for_clustering(
            segments,
            min_duration_ms=3000,
            skip_counters=skip_counters,
        )

        self.assertEqual([segment["id"] for segment in filtered], ["1_5000_4000_BenitaChen_0.677", "1_9000_4000_BenitaChen_0.653"])
        self.assertEqual(skip_counters["multi"], 1)


if __name__ == "__main__":
    unittest.main()
