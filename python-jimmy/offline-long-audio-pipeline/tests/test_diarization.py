import sys
import unittest
from pathlib import Path

import numpy as np

PIPELINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_DIR))

from diarization import (
    finalize_vad_only_segments,
    format_pyannote_mask_rle,
    resolve_final_segments,
    smooth_activity_spans,
)
from segmentation import SpeakerCountSpan
from vad import SpeechSegment


class ActivitySmoothingTest(unittest.TestCase):
    def test_merges_same_mask_when_gap_is_at_most_min_duration_off(self):
        activity = [
            SpeakerCountSpan(0, 400, 1, (1, 0, 0)),
            SpeakerCountSpan(800, 1200, 1, (1, 0, 0)),
        ]
        smoothed = smooth_activity_spans(
            activity, min_duration_on_ms=300, min_duration_off_ms=500
        )
        self.assertEqual(
            [(s.start_ms, s.end_ms, s.speaker_mask) for s in smoothed],
            [(0, 1200, (1, 0, 0))],
        )

    def test_does_not_merge_same_mask_when_gap_exceeds_min_duration_off(self):
        activity = [
            SpeakerCountSpan(0, 400, 1, (1, 0, 0)),
            SpeakerCountSpan(1000, 1400, 1, (1, 0, 0)),
        ]
        smoothed = smooth_activity_spans(
            activity, min_duration_on_ms=300, min_duration_off_ms=500
        )
        self.assertEqual(
            [(s.start_ms, s.end_ms) for s in smoothed],
            [(0, 400), (1000, 1400)],
        )

    def test_drops_islands_shorter_than_min_duration_on(self):
        activity = [
            SpeakerCountSpan(0, 80, 1, (0, 1, 0)),
            SpeakerCountSpan(200, 600, 1, (1, 0, 0)),
            SpeakerCountSpan(600, 1000, 1, (0, 0, 1)),
        ]
        smoothed = smooth_activity_spans(
            activity, min_duration_on_ms=300, min_duration_off_ms=500
        )
        self.assertEqual(
            [(s.start_ms, s.end_ms, s.speaker_mask) for s in smoothed],
            [(200, 600, (1, 0, 0)), (600, 1000, (0, 0, 1))],
        )

    def test_rejects_negative_smoothing_durations(self):
        with self.assertRaisesRegex(ValueError, "min_duration_on"):
            smooth_activity_spans([], min_duration_on_ms=-1, min_duration_off_ms=500)
        with self.assertRaisesRegex(ValueError, "min_duration_off"):
            smooth_activity_spans([], min_duration_on_ms=300, min_duration_off_ms=-1)


class PyannoteMaskRleTest(unittest.TestCase):
    def test_merges_identical_classes_and_fills_uncovered_as_class_zero(self):
        activity = [
            SpeakerCountSpan(100, 200, 1, (1, 0, 0), 1),
            SpeakerCountSpan(200, 400, 2, (1, 1, 0), 4),
            SpeakerCountSpan(500, 600, 1, (1, 0, 0), 1),
        ]
        self.assertEqual(
            format_pyannote_mask_rle(activity, 100, 600),
            "[100,1][200,3][100,0][100,1]",
        )

    def test_clips_to_segment_window(self):
        activity = [
            SpeakerCountSpan(0, 1000, 1, (1, 0, 0), 1),
            SpeakerCountSpan(1000, 2000, 1, (0, 1, 0), 2),
        ]
        self.assertEqual(format_pyannote_mask_rle(activity, 800, 1200), "[200,1][200,2]")


class TimelineResolutionTest(unittest.TestCase):
    def test_drops_overlap_shorter_than_min_duration_on_and_keeps_longer_overlap(self):
        waveform = np.zeros(16000 * 5, dtype=np.float32)
        vad = [SpeechSegment(1, 0, 4000, waveform[:64000])]
        activity = [
            SpeakerCountSpan(0, 1000, 1, (1, 0, 0)),
            SpeakerCountSpan(1000, 1050, 2, (1, 1, 0)),
            SpeakerCountSpan(1050, 2000, 1, (1, 0, 0)),
            SpeakerCountSpan(2000, 2400, 2, (1, 1, 0)),
            SpeakerCountSpan(2400, 4000, 1, (0, 1, 0)),
        ]
        segments, stats = resolve_final_segments(
            vad,
            activity,
            waveform,
            16000,
            min_duration_on=0.3,
            min_duration_off=0.5,
        )
        self.assertEqual(
            [(s.start_ms, s.end_ms, s.speaker_composition) for s in segments],
            [
                (0, 2000, "single_speaker"),
                (2000, 4000, "single_speaker"),
            ],
        )
        self.assertEqual(segments[1].overlap_regions, [(2000, 2400)])
        self.assertEqual(stats.true_overlap_count, 0)
        self.assertTrue(all(a.end_ms <= b.start_ms for a, b in zip(segments, segments[1:])))
        self.assertEqual(segments[0].pyannote_mask, "[1000,1][50,3][950,1]")
        self.assertEqual(segments[1].pyannote_mask, "[400,3][1600,2]")

    def test_folds_short_overlap_into_following_segment_and_keeps_range(self):
        waveform = np.zeros(16000 * 4, dtype=np.float32)
        vad = [SpeechSegment(1, 0, 4000, waveform[:64000])]
        activity = [
            SpeakerCountSpan(0, 2000, 1, (1, 0, 0)),
            SpeakerCountSpan(2000, 2400, 2, (1, 1, 0)),
            SpeakerCountSpan(2400, 4000, 1, (0, 1, 0)),
        ]
        segments, stats = resolve_final_segments(
            vad, activity, waveform, 16000, min_duration_on=0.3, min_duration_off=0.5
        )
        self.assertEqual(
            [(s.start_ms, s.end_ms, s.speaker_composition) for s in segments],
            [(0, 2000, "single_speaker"), (2000, 4000, "single_speaker")],
        )
        self.assertEqual(segments[0].overlap_regions, [])
        self.assertEqual(segments[1].overlap_regions, [(2000, 2400)])
        self.assertEqual(stats.true_overlap_count, 0)

    def test_folds_long_overlap_into_following_segment_and_keeps_range(self):
        waveform = np.zeros(16000 * 5, dtype=np.float32)
        vad = [SpeechSegment(1, 0, 5000, waveform[:80000])]
        activity = [
            SpeakerCountSpan(0, 2000, 1, (1, 0, 0)),
            SpeakerCountSpan(2000, 3200, 2, (1, 1, 0)),
            SpeakerCountSpan(3200, 5000, 1, (0, 1, 0)),
        ]
        segments, stats = resolve_final_segments(vad, activity, waveform, 16000)
        self.assertEqual(
            [(s.start_ms, s.end_ms, s.speaker_composition) for s in segments],
            [(0, 2000, "single_speaker"), (2000, 5000, "single_speaker")],
        )
        self.assertEqual(segments[0].overlap_regions, [])
        self.assertEqual(segments[1].overlap_regions, [(2000, 3200)])
        self.assertEqual(stats.true_overlap_count, 0)

    def test_folds_trailing_short_overlap_into_previous_segment(self):
        waveform = np.zeros(16000 * 3, dtype=np.float32)
        vad = [SpeechSegment(1, 0, 2400, waveform[:38400])]
        activity = [
            SpeakerCountSpan(0, 2000, 1, (1, 0, 0)),
            SpeakerCountSpan(2000, 2400, 2, (1, 1, 0)),
        ]
        segments, _ = resolve_final_segments(
            vad, activity, waveform, 16000, min_duration_on=0.3, min_duration_off=0.5
        )
        self.assertEqual(
            [(s.start_ms, s.end_ms, s.speaker_composition) for s in segments],
            [(0, 2400, "single_speaker")],
        )
        self.assertEqual(segments[0].overlap_regions, [(2000, 2400)])

    def test_one_millisecond_flickers_do_not_become_final_segments(self):
        waveform = np.zeros(16000, dtype=np.float32)
        vad = [SpeechSegment(1, 0, 1000, waveform[:16000])]
        activity = [
            SpeakerCountSpan(0, 400, 1, (1, 0, 0)),
            SpeakerCountSpan(400, 401, 0, (0, 0, 0)),
            SpeakerCountSpan(401, 485, 1, (0, 1, 0)),
            SpeakerCountSpan(485, 486, 0, (0, 0, 0)),
            SpeakerCountSpan(486, 1000, 1, (1, 0, 0)),
        ]
        segments, _ = resolve_final_segments(
            vad, activity, waveform, 16000, min_duration_on=0.3, min_duration_off=0.5
        )
        durations = [segment.end_ms - segment.start_ms for segment in segments]
        self.assertNotIn(1, durations)
        self.assertNotIn(84, durations)

    def test_absorbs_one_millisecond_holes_between_smoothed_activity_spans(self):
        waveform = np.zeros(16000, dtype=np.float32)
        vad = [SpeechSegment(1, 0, 1000, waveform[:16000])]
        activity = [
            SpeakerCountSpan(0, 500, 1, (1, 0, 0)),
            SpeakerCountSpan(501, 1000, 1, (0, 1, 0)),
        ]
        segments, _ = resolve_final_segments(
            vad, activity, waveform, 16000, min_duration_on=0.3, min_duration_off=0.5
        )
        durations = [segment.end_ms - segment.start_ms for segment in segments]
        self.assertNotIn(1, durations)
        self.assertEqual(
            [(s.start_ms, s.end_ms) for s in segments],
            [(0, 1000)],
        )

    def test_folds_unknown_activity_into_following_segment(self):
        waveform = np.zeros(16000 * 4, dtype=np.float32)
        vad = [SpeechSegment(1, 0, 4000, waveform[:64000])]
        activity = [
            SpeakerCountSpan(0, 2000, 1, (1, 0, 0)),
            SpeakerCountSpan(2000, 2400, 0, (0, 0, 0)),
            SpeakerCountSpan(2400, 4000, 1, (0, 1, 0)),
        ]
        segments, _ = resolve_final_segments(
            vad, activity, waveform, 16000, min_duration_on=0.3, min_duration_off=0.5
        )
        self.assertEqual(
            [(s.start_ms, s.end_ms, s.speaker_composition) for s in segments],
            [(0, 2000, "single_speaker"), (2000, 4000, "single_speaker")],
        )
        self.assertTrue(all(s.speaker_composition != "unknown_activity" for s in segments))

    def test_folds_trailing_unknown_activity_into_previous_segment(self):
        waveform = np.zeros(16000 * 3, dtype=np.float32)
        vad = [SpeechSegment(1, 0, 2400, waveform[:38400])]
        activity = [
            SpeakerCountSpan(0, 2000, 1, (1, 0, 0)),
            SpeakerCountSpan(2000, 2400, 0, (0, 0, 0)),
        ]
        segments, _ = resolve_final_segments(vad, activity, waveform, 16000)
        self.assertEqual(
            [(s.start_ms, s.end_ms, s.speaker_composition) for s in segments],
            [(0, 2400, "single_speaker")],
        )

    def test_merges_short_single_island_into_longer_neighbor(self):
        waveform = np.zeros(16000 * 3, dtype=np.float32)
        vad = [SpeechSegment(1, 0, 3000, waveform[:48000])]
        activity = [
            SpeakerCountSpan(0, 800, 1, (1, 0, 0)),
            SpeakerCountSpan(800, 3000, 1, (0, 1, 0)),
        ]
        segments, _ = resolve_final_segments(vad, activity, waveform, 16000)
        self.assertEqual(
            [(s.start_ms, s.end_ms, s.speaker_composition) for s in segments],
            [(0, 3000, "single_speaker")],
        )

    def test_short_single_island_prefers_following_segment(self):
        waveform = np.zeros(16000 * 5, dtype=np.float32)
        vad = [SpeechSegment(1, 0, 4800, waveform[:76800])]
        activity = [
            SpeakerCountSpan(0, 2000, 1, (1, 0, 0)),
            SpeakerCountSpan(2000, 2800, 1, (0, 1, 0)),
            SpeakerCountSpan(2800, 4800, 1, (0, 0, 1)),
        ]
        segments, _ = resolve_final_segments(vad, activity, waveform, 16000)
        self.assertEqual(
            [(s.start_ms, s.end_ms) for s in segments],
            [(0, 2000), (2000, 4800)],
        )

    def test_trailing_short_single_merges_into_previous_when_no_follower(self):
        waveform = np.zeros(16000 * 3, dtype=np.float32)
        vad = [SpeechSegment(1, 0, 2800, waveform[:44800])]
        activity = [
            SpeakerCountSpan(0, 2000, 1, (1, 0, 0)),
            SpeakerCountSpan(2000, 2800, 1, (0, 1, 0)),
        ]
        segments, _ = resolve_final_segments(vad, activity, waveform, 16000)
        self.assertEqual(
            [(s.start_ms, s.end_ms) for s in segments],
            [(0, 2800)],
        )

    def test_does_not_fold_unknown_longer_than_two_seconds(self):
        waveform = np.zeros(16000 * 8, dtype=np.float32)
        vad = [SpeechSegment(1, 0, 7000, waveform[:112000])]
        activity = [
            SpeakerCountSpan(0, 2000, 1, (1, 0, 0)),
            SpeakerCountSpan(2000, 4500, 0, (0, 0, 0)),
            SpeakerCountSpan(4500, 7000, 1, (0, 1, 0)),
        ]
        segments, _ = resolve_final_segments(vad, activity, waveform, 16000)
        self.assertEqual(
            [(s.start_ms, s.end_ms, s.speaker_composition) for s in segments],
            [
                (0, 2000, "single_speaker"),
                (2000, 4500, "unknown_activity"),
                (4500, 7000, "single_speaker"),
            ],
        )

    def test_keeps_vad_hard_bounds_and_does_not_merge_across_vad(self):
        waveform = np.zeros(16000 * 4, dtype=np.float32)
        vad = [
            SpeechSegment(1, 0, 1500, waveform[:24000]),
            SpeechSegment(2, 2500, 4000, waveform[40000:64000]),
        ]
        activity = [
            SpeakerCountSpan(0, 800, 1, (1, 0, 0)),
            SpeakerCountSpan(800, 1500, 1, (0, 1, 0)),
            SpeakerCountSpan(2500, 4000, 1, (0, 1, 0)),
        ]
        segments, _ = resolve_final_segments(vad, activity, waveform, 16000)
        self.assertEqual(segments[0].start_ms, 0)
        self.assertEqual(segments[0].end_ms, 1500)
        self.assertEqual(segments[-1].start_ms, 2500)
        self.assertEqual(segments[-1].end_ms, 4000)
        self.assertTrue(all(s.end_ms <= 1500 or s.start_ms >= 2500 for s in segments))

    def test_marks_left_and_right_cuts_as_vad_or_pyannote(self):
        waveform = np.zeros(16000 * 4, dtype=np.float32)
        vad = [
            SpeechSegment(1, 0, 3000, waveform[:48000]),
            SpeechSegment(2, 3500, 4000, waveform[56000:64000]),
        ]
        activity = [
            SpeakerCountSpan(0, 1000, 1, (1, 0, 0)),
            SpeakerCountSpan(1000, 2000, 1, (0, 1, 0)),
            SpeakerCountSpan(2000, 3000, 1, (1, 0, 0)),
            SpeakerCountSpan(3500, 4000, 1, (0, 1, 0)),
        ]
        segments, _ = resolve_final_segments(vad, activity, waveform, 16000)
        self.assertEqual(
            [(s.start_ms, s.end_ms, s.cut_left, s.cut_right) for s in segments],
            [
                (0, 1000, "vad", "pyannote"),
                (1000, 2000, "pyannote", "pyannote"),
                (2000, 3000, "pyannote", "vad"),
                (3500, 4000, "vad", "vad"),
            ],
        )

    def test_does_not_merge_adjacent_single_speaker_masks(self):
        waveform = np.zeros(16000 * 2, dtype=np.float32)
        vad = [SpeechSegment(1, 0, 2000, waveform[:32000])]
        activity = [
            SpeakerCountSpan(0, 1000, 1, (1, 0, 0)),
            SpeakerCountSpan(1000, 2000, 1, (0, 1, 0)),
        ]

        segments, _ = resolve_final_segments(vad, activity, waveform, 16000)

        self.assertEqual(
            [(s.start_ms, s.end_ms, s.speaker_composition) for s in segments],
            [(0, 1000, "single_speaker"), (1000, 2000, "single_speaker")],
        )

    def test_does_not_fill_unknown_hole_between_different_single_speaker_masks(self):
        waveform = np.zeros(16000 * 2, dtype=np.float32)
        vad = [SpeechSegment(1, 0, 2000, waveform[:32000])]
        activity = [
            SpeakerCountSpan(0, 800, 1, (1, 0, 0)),
            SpeakerCountSpan(800, 1200, 0, (0, 0, 0)),
            SpeakerCountSpan(1200, 2000, 1, (0, 1, 0)),
        ]

        segments, _ = resolve_final_segments(vad, activity, waveform, 16000)

        self.assertEqual(
            [(s.start_ms, s.end_ms, s.speaker_composition) for s in segments],
            [(0, 2000, "single_speaker")],
        )

    def test_emits_unknown_activity_and_never_merges_across_vad_silence(self):
        waveform = np.zeros(16000 * 3, dtype=np.float32)
        vad = [
            SpeechSegment(1, 0, 500, waveform[:8000]),
            SpeechSegment(2, 1000, 1500, waveform[16000:24000]),
        ]
        activity = [
            SpeakerCountSpan(0, 500, 0),
            SpeakerCountSpan(1000, 1500, 1),
        ]

        segments, stats = resolve_final_segments(vad, activity, waveform, 16000)

        self.assertEqual(
            [(s.segment_index, s.start_ms, s.end_ms, s.speaker_composition) for s in segments],
            [(1, 0, 500, "unknown_activity"), (2, 1000, 1500, "single_speaker")],
        )
        self.assertEqual(stats.true_overlap_count, 0)

    def test_uses_vad_speech_as_single_speaker_fallback_for_pyannote_activity_holes(self):
        waveform = np.zeros(16000 * 2, dtype=np.float32)
        vad = [SpeechSegment(1, 0, 1500, waveform[:24000])]
        activity = [
            SpeakerCountSpan(0, 500, 1),
            SpeakerCountSpan(500, 1000, 0),
            SpeakerCountSpan(1000, 1500, 1),
        ]

        segments, stats = resolve_final_segments(vad, activity, waveform, 16000)

        self.assertEqual(
            [(s.start_ms, s.end_ms, s.speaker_composition) for s in segments],
            [(0, 1500, "single_speaker")],
        )
        self.assertEqual(stats.true_overlap_count, 0)

    def test_rejects_invalid_resolution_arguments(self):
        with self.assertRaisesRegex(ValueError, "sample rate"):
            resolve_final_segments([], [], np.zeros(1, dtype=np.float32), 0)
        with self.assertRaisesRegex(ValueError, "min_duration_on"):
            resolve_final_segments([], [], np.zeros(1, dtype=np.float32), 16000, min_duration_on=-0.1)

    def test_trailing_short_exclusive_after_folded_overlap_merges_into_previous(self):
        waveform = np.zeros(16000 * 12, dtype=np.float32)
        vad = [SpeechSegment(1, 0, 11780, waveform[: 11780 * 16])]
        activity = [
            SpeakerCountSpan(0, 10000, 1, (1, 0, 0)),
            SpeakerCountSpan(10000, 10860, 2, (1, 1, 0)),
            SpeakerCountSpan(10860, 11780, 1, (0, 1, 0)),
        ]
        segments, stats = resolve_final_segments(vad, activity, waveform, 16000)
        self.assertEqual(
            [(s.start_ms, s.end_ms, s.speaker_composition) for s in segments],
            [(0, 11780, "single_speaker")],
        )
        self.assertEqual(segments[0].overlap_regions, [(10000, 10860)])
        self.assertEqual(stats.true_overlap_count, 0)

    def test_same_mask_singles_merge_after_overlap_is_folded_between_them(self):
        waveform = np.zeros(16000 * 5, dtype=np.float32)
        vad = [SpeechSegment(1, 0, 3938, waveform[: 3938 * 16])]
        activity = [
            SpeakerCountSpan(0, 578, 1, (1, 0, 0)),
            SpeakerCountSpan(578, 1962, 2, (1, 1, 0)),
            SpeakerCountSpan(1962, 3938, 1, (1, 0, 0)),
        ]
        segments, stats = resolve_final_segments(vad, activity, waveform, 16000)
        self.assertEqual(
            [(s.start_ms, s.end_ms, s.speaker_composition) for s in segments],
            [(0, 3938, "single_speaker")],
        )
        self.assertEqual(segments[0].overlap_regions, [(578, 1962)])
        self.assertEqual(stats.true_overlap_count, 0)

    def test_keeps_same_mask_cut_when_both_exclusive_sides_are_long(self):
        waveform = np.zeros(16000 * 7, dtype=np.float32)
        vad = [SpeechSegment(1, 0, 6000, waveform[:96000])]
        activity = [
            SpeakerCountSpan(0, 2000, 1, (0, 0, 1), 3),
            SpeakerCountSpan(2000, 2500, 2, (0, 1, 1), 6),
            SpeakerCountSpan(2500, 6000, 1, (0, 0, 1), 3),
        ]
        segments, stats = resolve_final_segments(vad, activity, waveform, 16000)
        self.assertEqual(
            [(s.start_ms, s.end_ms, s.speaker_composition) for s in segments],
            [(0, 2000, "single_speaker"), (2000, 6000, "single_speaker")],
        )
        self.assertEqual(segments[0].overlap_regions, [])
        self.assertEqual(segments[1].overlap_regions, [(2000, 2500)])
        self.assertEqual(stats.true_overlap_count, 0)

    def test_merges_same_mask_cut_when_one_exclusive_side_is_under_two_seconds(self):
        waveform = np.zeros(16000 * 5, dtype=np.float32)
        vad = [SpeechSegment(1, 0, 4200, waveform[:67200])]
        activity = [
            SpeakerCountSpan(0, 1200, 1, (0, 0, 1), 3),
            SpeakerCountSpan(1200, 1700, 2, (0, 1, 1), 6),
            SpeakerCountSpan(1700, 4200, 1, (0, 0, 1), 3),
        ]
        segments, stats = resolve_final_segments(vad, activity, waveform, 16000)
        self.assertEqual(
            [(s.start_ms, s.end_ms, s.speaker_composition) for s in segments],
            [(0, 4200, "single_speaker")],
        )
        self.assertEqual(segments[0].overlap_regions, [(1200, 1700)])
        self.assertEqual(stats.true_overlap_count, 0)

    def test_records_uninterrupted_clean_spans_without_crossing_overlap(self):
        waveform = np.zeros(16000 * 7, dtype=np.float32)
        vad = [SpeechSegment(1, 0, 7000, waveform[:112000])]
        activity = [
            SpeakerCountSpan(0, 3500, 1, (1, 0, 0)),
            SpeakerCountSpan(3500, 4000, 2, (1, 1, 0)),
            SpeakerCountSpan(4000, 7000, 1, (1, 0, 0)),
        ]

        segments, _ = resolve_final_segments(vad, activity, waveform, 16000)

        self.assertEqual(
            [(segment.start_ms, segment.end_ms) for segment in segments],
            [(0, 3500), (3500, 7000)],
        )
        self.assertEqual(segments[0].clean_spans, [(0, 3500)])
        self.assertEqual(segments[1].overlap_regions, [(3500, 4000)])
        self.assertEqual(segments[1].clean_spans, [(4000, 7000)])

    def test_vad_only_keeps_speech_regions_as_single_speaker_segments(self):
        waveform = np.zeros(16000, dtype=np.float32)
        vad = [
            SpeechSegment(9, 100, 400, waveform[1600:6400]),
            SpeechSegment(8, 500, 900, waveform[8000:14400]),
        ]

        segments, stats = finalize_vad_only_segments(vad)

        self.assertEqual(
            [(s.segment_index, s.start_ms, s.end_ms, s.speaker_composition) for s in segments],
            [(1, 100, 400, "single_speaker"), (2, 500, 900, "single_speaker")],
        )
        self.assertEqual(
            [(s.cut_left, s.cut_right) for s in segments],
            [("vad", "vad"), ("vad", "vad")],
        )
        self.assertEqual(stats.true_overlap_count, 0)


if __name__ == "__main__":
    unittest.main()
