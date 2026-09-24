import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

PIPELINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_DIR))

from output import _format_time, create_run_directory, write_metadata, write_results
from vad import SpeechSegment


class OutputTest(unittest.TestCase):
    def test_writes_compact_segment_contract_and_readable_time_range(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = create_run_directory(root, "input name.pcm", timestamp="20260909-120000")
            duplicate_dir = create_run_directory(root, "input name.pcm", timestamp="20260909-120000")
            segment = SpeechSegment(
                1,
                125850,
                129456,
                np.zeros(1, dtype=np.float32),
                asr_text="\u4f60\u597d",
                text_confidence=0.91,
                speaker_id="speaker_00",
                speaker_composition="single_speaker",
                cluster_assignment_similarity=0.875,
                cut_left="vad",
                cut_right="pyannote",
                pyannote_mask="[100,1][200,3]",
                clean_spans=[(126000, 127000), (128000, 129000)],
            )
            write_results(
                run_dir,
                "input name.pcm",
                129456,
                [segment],
                {"total_seconds": 2.5},
            )
            write_metadata(run_dir, {"audio": "input name.pcm"})
            payload = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
            transcript = (run_dir / "result.txt").read_text(encoding="utf-8")
            metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))

        self.assertEqual(run_dir.name, "20260909-120000_input_name")
        self.assertEqual(duplicate_dir.name, "20260909-120000_input_name_01")
        self.assertEqual(
            payload["segments"],
            [
                {
                    "segment_id": "0001_125850_129456",
                    "time_range": "2:05.850-2:09.456",
                    "duration_ms": 3606,
                    "duration_class": "long",
                    "speaker_composition": "single_speaker",
                    "cut_left": "vad",
                    "cut_right": "pyannote",
                    "asr_text": "\u4f60\u597d",
                    "asr_language": "",
                    "whisper_language": "",
                    "whisper_lang_prob": None,
                    "text_confidence": 0.91,
                    "asr_candidates": {},
                    "asr_valid": 1,
                    "speaker_id": "speaker_00",
                    "previous_segment_similarity": None,
                    "cluster_assignment_similarity": 0.875,
                    "pyannote_mask": "[100,1][200,3]",
                    "clean_spans": "[126000,127000] [128000,129000]",
                    "local_speaker_mask": 0,
                    "local_speaker_mask_confidence": 0.0,
                    "speaker_assignment_source": "unknown",
                }
            ],
        )
        self.assertIn("[2:05.850 - 2:09.456] speaker_00: \u4f60\u597d", transcript)
        self.assertEqual(metadata["audio"], "input name.pcm")

    def test_writes_absorbed_overlap_ranges_on_host_segment(self):
        with TemporaryDirectory() as directory:
            run_dir = create_run_directory(Path(directory), "input.pcm", timestamp="20260910-180000")
            segment = SpeechSegment(
                1,
                2000,
                4000,
                np.zeros(1, dtype=np.float32),
                speaker_id="speaker_01",
                speaker_composition="single_speaker",
                overlap_regions=[(2000, 2400)],
            )
            write_results(run_dir, "input.pcm", 4000, [segment], {"total_seconds": 1.0})
            payload = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        self.assertNotIn("overlap_regions", payload["segments"][0])

    def test_run_label_makes_experiment_directory_meaningful(self):
        with TemporaryDirectory() as directory:
            run_dir = create_run_directory(
                Path(directory),
                "102_asr_1788405384390.pcm",
                run_label="titanet_speaker_turn_bad_cases",
                timestamp="20260910-120000",
            )
        self.assertEqual(
            run_dir.name,
            "20260910-120000_titanet_speaker_turn_bad_cases_102_asr_1788405384390",
        )

    def test_formats_time_as_minutes_seconds_milliseconds_without_hour_field(self):
        self.assertEqual(_format_time(2456618), "40:56.618")
        with self.assertRaisesRegex(ValueError, "non-negative"):
            _format_time(-1)


if __name__ == "__main__":
    unittest.main()
