#!/usr/bin/env python3

import importlib.util
import sys
import unittest
from pathlib import Path


def load_module():
    script_path = Path(__file__).resolve().parent.parent / "utils-asr_log_cut.py"
    spec = importlib.util.spec_from_file_location("utils_asr_log_cut", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestUtilsAsrLogCut(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_parse_plain_log_line_maps_multi_metadata_to_multi_speaker(self):
        segment = self.module.parse_plain_log_line(
            "1112030 20422 0 multi=1;angles=[77:578 114:1088] 这是一句测试文本",
            1,
        )

        self.assertEqual(segment.offset_ms, 1112030)
        self.assertEqual(segment.duration_ms, 20422)
        self.assertEqual(segment.name, "Multi")
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
        self.assertEqual(segment.name, "Multi")
        self.assertEqual(segment.score, "NA")


if __name__ == "__main__":
    unittest.main()
