import sys
import unittest
from pathlib import Path

import numpy as np

PIPELINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_DIR))

from vad import SpeechSegment, collect_vad_segments


class FakeVadSegment:
    def __init__(self, start, samples):
        self.start = start
        self.samples = samples


class FakeVad:
    def __init__(self):
        self.pending = []
        self.accept_calls = 0

    def accept_waveform(self, _samples):
        self.accept_calls += 1
        if self.accept_calls == 1:
            self.pending.append(FakeVadSegment(320, np.zeros(1600, dtype=np.float32)))

    def flush(self):
        self.pending.append(FakeVadSegment(4000, np.zeros(800, dtype=np.float32)))

    def empty(self):
        return not self.pending

    @property
    def front(self):
        return self.pending[0]

    def pop(self):
        self.pending.pop(0)


class VadCollectionTest(unittest.TestCase):
    def test_collects_available_and_flushed_segments_in_timestamp_order(self):
        segments = collect_vad_segments(
            FakeVad(), np.zeros(320, dtype=np.float32), window_size=160, sample_rate=16000
        )

        self.assertEqual([segment.segment_index for segment in segments], [1, 2])
        self.assertEqual([(segment.start_ms, segment.end_ms) for segment in segments], [(20, 120), (250, 300)])
        self.assertEqual([segment.duration_ms for segment in segments], [100, 50])

    def test_duration_class_and_cluster_eligibility_are_independent_of_composition(self):
        short_single = SpeechSegment(1, 0, 999, np.zeros(15984), speaker_composition="single_speaker")
        long_overlap = SpeechSegment(2, 1000, 2500, np.zeros(24000), speaker_composition="overlapped_speakers")
        long_single = SpeechSegment(3, 2500, 3500, np.zeros(16000), speaker_composition="single_speaker")

        self.assertEqual(short_single.duration_class, "short")
        self.assertFalse(short_single.is_cluster_eligible)
        self.assertEqual(long_overlap.duration_class, "long")
        self.assertFalse(long_overlap.is_cluster_eligible)
        self.assertTrue(long_single.is_cluster_eligible)

    def test_cluster_eligibility_uses_duration_excluding_absorbed_overlap(self):
        host = SpeechSegment(
            1,
            0,
            1500,
            np.zeros(24000),
            speaker_composition="single_speaker",
            overlap_regions=[(0, 600)],
        )
        self.assertFalse(host.is_cluster_eligible)

    def test_cluster_eligibility_keeps_long_prefix_before_overlap_and_short_tail(self):
        host = SpeechSegment(
            1,
            2910,
            14720,
            np.zeros((14720 - 2910) * 16),
            speaker_composition="single_speaker",
            overlap_regions=[(12944, 13803)],
        )
        self.assertTrue(host.is_cluster_eligible)
        self.assertEqual(host.exclusive_speech_duration_ms, 10034)

    def test_invalid_asr_segments_are_not_cluster_eligible(self):
        segment = SpeechSegment(1, 0, 2000, np.zeros(32000), speaker_composition="single_speaker")
        segment.asr_valid = 0
        self.assertFalse(segment.is_cluster_eligible)


if __name__ == "__main__":
    unittest.main()
