#!/usr/bin/env python3

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path


def load_module():
    if "sherpa_onnx" not in sys.modules:
        sys.modules["sherpa_onnx"] = types.SimpleNamespace()

    script_path = Path(__file__).resolve().parent.parent / "k2-speaker-cluster-fastclustering.py"
    spec = importlib.util.spec_from_file_location("k2_speaker_cluster_fastclustering", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestK2SpeakerClusterFastClustering(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_filter_segments_skips_multi_named_segments(self):
        segments = [
            {"id": "audio_100_1000", "durationMs": 1000, "speaker": "Multi"},
            {"id": "audio_1100_1000", "durationMs": 1000, "speaker": "JimmyXu"},
            {"id": "audio_2200_1000", "durationMs": 1000, "speaker": None},
        ]

        filtered = self.module.filter_segments_by_speaker(
            segments,
            mode="all",
            min_duration_ms=0,
        )

        self.assertEqual(
            [segment["id"] for segment in filtered],
            ["audio_1100_1000", "audio_2200_1000"],
        )

    def test_load_segments_marks_multi_from_segment_id_without_speaker_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "emb.txt"
            input_path.write_text(
                "\n".join(
                    [
                        "1_1002878_6918_Multi_NA AAAA",
                        "1_1021726_12134_CocoDai_0.678 BBBB",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            segments = self.module.load_segments(input_path)
            filtered = self.module.filter_segments_by_speaker(
                segments,
                mode="all",
                min_duration_ms=0,
            )

            self.assertEqual(segments[0]["speaker"], "Multi")
            self.assertEqual(segments[1]["speaker"], "CocoDai")
            self.assertEqual(
                [segment["id"] for segment in filtered],
                ["1_1021726_12134_CocoDai_0.678"],
            )

    def test_build_cluster_summary_lines_include_skipped_counts(self):
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
            num_clusters_arg=-1,
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


if __name__ == "__main__":
    unittest.main()
