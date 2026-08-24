#!/usr/bin/env python3

import importlib.util
import tempfile
import unittest
from pathlib import Path


def load_module():
    script_path = Path(__file__).resolve().parent.parent / "utils-asr_angle_merge.py"
    spec = importlib.util.spec_from_file_location("utils_asr_angle_merge", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestAsrAngleMerge(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_merge_inserts_deduplicated_angles_with_320ms_lag(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            angle_path = tmp_path / "angle.txt"
            asr_path = tmp_path / "asr.txt"
            output_path = tmp_path / "merged.txt"

            angle_path.write_text(
                "\n".join(["1"] * 10 + ["3"] * 2 + ["10"] * 2 + ["25"] * 2) + "\n",
                encoding="utf-8",
            )
            asr_path.write_text(
                "0 192 0 top1=NathanSun(0.356) 这两天你最近。\n",
                encoding="utf-8",
            )

            self.module.merge_angle_into_asr(angle_path, asr_path, output_path)

            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                "0 192 0 top1=NathanSun(0.356) [3:64 10:64 25:64] 这两天你最近。\n",
            )

    def test_merge_truncates_window_when_angle_data_is_short(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            angle_path = tmp_path / "angle.txt"
            asr_path = tmp_path / "asr.txt"
            output_path = tmp_path / "merged.txt"

            angle_path.write_text("\n".join(["5"] * 12) + "\n", encoding="utf-8")
            asr_path.write_text(
                "0 192 0 top1=NathanSun(0.356) 你好\n",
                encoding="utf-8",
            )

            self.module.merge_angle_into_asr(angle_path, asr_path, output_path)

            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                "0 192 0 top1=NathanSun(0.356) [5:64] 你好\n",
            )

    def test_merge_outputs_empty_brackets_when_no_angle_sample_overlaps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            angle_path = tmp_path / "angle.txt"
            asr_path = tmp_path / "asr.txt"
            output_path = tmp_path / "merged.txt"

            angle_path.write_text("7\n", encoding="utf-8")
            asr_path.write_text(
                "0 64 0 top1=NathanSun(0.356) hi\n",
                encoding="utf-8",
            )

            self.module.merge_angle_into_asr(angle_path, asr_path, output_path)

            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                "0 64 0 top1=NathanSun(0.356) [] hi\n",
            )

    def test_merge_combines_angles_within_five_degrees_and_accumulates_duration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            angle_path = tmp_path / "angle.txt"
            asr_path = tmp_path / "asr.txt"
            output_path = tmp_path / "merged.txt"

            angle_path.write_text(
                "\n".join(["10"] * 10 + ["74", "77", "75", "76", "91", "92"]) + "\n",
                encoding="utf-8",
            )
            asr_path.write_text(
                "0 192 0 top1=WoodWang(0.523) test\n",
                encoding="utf-8",
            )

            self.module.merge_angle_into_asr(angle_path, asr_path, output_path)

            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                "0 192 0 top1=WoodWang(0.523) [74:128 91:64] test\n",
            )

    def test_merge_tolerance_zero_keeps_different_adjacent_angles_separate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            angle_path = tmp_path / "angle.txt"
            asr_path = tmp_path / "asr.txt"
            output_path = tmp_path / "merged.txt"

            angle_path.write_text(
                "\n".join(["10"] * 10 + ["74", "77", "75", "76", "91", "92"]) + "\n",
                encoding="utf-8",
            )
            asr_path.write_text(
                "0 192 0 top1=WoodWang(0.523) test\n",
                encoding="utf-8",
            )

            self.module.merge_angle_into_asr(
                angle_path,
                asr_path,
                output_path,
                merge_tolerance=0,
            )

            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                "0 192 0 top1=WoodWang(0.523) [74:32 77:32 75:32 76:32 91:32 92:32] test\n",
            )

    def test_merge_writes_default_angle_segments_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            angle_path = tmp_path / "angle.txt"
            asr_path = tmp_path / "asr.txt"
            output_path = tmp_path / "merged.txt"

            angle_path.write_text(
                "\n".join(["1", "2", "1", "2", "10", "11"]) + "\n",
                encoding="utf-8",
            )
            asr_path.write_text(
                "0 64 0 top1=NathanSun(0.356) hi\n",
                encoding="utf-8",
            )

            self.module.merge_angle_into_asr(
                angle_path,
                asr_path,
                output_path,
                merge_tolerance=2,
            )

            segments_path = tmp_path / "angle.segments.txt"
            self.assertTrue(segments_path.exists())
            self.assertEqual(
                segments_path.read_text(encoding="utf-8"),
                "0 128 128 1\n128 192 64 10\n",
            )

    def test_merge_supports_lines_without_top1_result(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            angle_path = tmp_path / "angle.txt"
            asr_path = tmp_path / "asr.txt"
            output_path = tmp_path / "merged.txt"

            angle_path.write_text("\n".join(["1"] * 12 + ["9"] * 12) + "\n", encoding="utf-8")
            asr_path.write_text(
                "90846 386 0 - 账目。\n",
                encoding="utf-8",
            )

            self.module.merge_angle_into_asr(
                angle_path,
                asr_path,
                output_path,
                merge_tolerance=2,
            )

            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                "90846 386 0 - [] 账目。\n",
            )

    def test_merge_uses_exact_overlap_duration_instead_of_full_32ms_frames(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            angle_path = tmp_path / "angle.txt"
            asr_path = tmp_path / "asr.txt"
            output_path = tmp_path / "merged.txt"

            angle_path.write_text("\n".join(["174"] * 300) + "\n", encoding="utf-8")
            asr_path.write_text(
                "4638 1958 JimmyXu top1=JimmyXu(0.529);multi=0 现在我在一个位置说话。\n",
                encoding="utf-8",
            )

            self.module.merge_angle_into_asr(
                angle_path,
                asr_path,
                output_path,
                merge_tolerance=2,
            )

            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                "4638 1958 JimmyXu top1=JimmyXu(0.529);multi=0 [174:1958] 现在我在一个位置说话。\n",
            )


if __name__ == "__main__":
    unittest.main()
