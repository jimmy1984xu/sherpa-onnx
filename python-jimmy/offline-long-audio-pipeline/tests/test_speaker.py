import sys
import unittest
from pathlib import Path

import numpy as np

PIPELINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_DIR))

from speaker import (
    _default_clusterer_factory,
    assign_speaker_ids,
    assign_speaker_ids_with_centroids,
    populate_previous_segment_similarities,
    samples_for_embedding,
    stable_speaker_ids,
)
from vad import SpeechSegment


class FakeSpeakerStream:
    def accept_waveform(self, sample_rate, waveform):
        self.sample_rate = sample_rate
        self.waveform = waveform

    def input_finished(self):
        self.finished = True


class FakeExtractor:
    def create_stream(self):
        return FakeSpeakerStream()

    def is_ready(self, stream):
        return True

    def compute(self, stream):
        return [0.1, 0.2, 0.3]


class SpeakerTest(unittest.TestCase):
    def test_orders_cluster_names_by_first_matching_segment(self):
        self.assertEqual(
            stable_speaker_ids([7, 3, 7, 9]),
            ["speaker_00", "speaker_01", "speaker_00", "speaker_02"],
        )

    def test_extracts_short_segments_and_sets_adjacent_similarity(self):
        segments = [
            SpeechSegment(1, 0, 1000, np.zeros(16000, dtype=np.float32)),
            SpeechSegment(2, 1000, 1200, np.zeros(3200, dtype=np.float32)),
        ]
        received = []

        def clusterer_factory(_threshold, _num_clusters):
            def cluster(embeddings):
                received.append(embeddings)
                return [8, 8]
            return cluster

        assign_speaker_ids(
            FakeExtractor(),
            segments,
            cluster_threshold=0.5,
            num_clusters=-1,
            clusterer_factory=clusterer_factory,
        )

        self.assertEqual(segments[0].speaker_id, "speaker_00")
        self.assertEqual(segments[1].speaker_id, "speaker_00")
        self.assertIsNone(segments[0].previous_segment_similarity)
        self.assertEqual(segments[1].previous_segment_similarity, 1.0)
        self.assertEqual(received[0].shape, (2, 3))


class CleanSpanAndLocalMaskAssignmentTest(unittest.TestCase):
    @staticmethod
    def _segment(index, start_ms, end_ms, *, mask, confidence, clean_spans=()):
        return SpeechSegment(
            index,
            start_ms,
            end_ms,
            np.zeros((end_ms - start_ms) * 16, dtype=np.float32),
            speaker_composition="single_speaker",
            local_speaker_mask=mask,
            local_speaker_mask_confidence=confidence,
            clean_spans=list(clean_spans),
        )

    def test_only_one_continuous_clean_span_of_at_least_three_seconds_clusters(self):
        split_clean = self._segment(
            1, 0, 5000, mask=1, confidence=0.99, clean_spans=((0, 2000), (3000, 5000))
        )
        exact_clean = self._segment(
            2, 5000, 8000, mask=2, confidence=0.99, clean_spans=((5000, 8000),)
        )
        clusterer = RecordingClusterer([7])

        errors, assigned, unknown, skipped = assign_speaker_ids_with_centroids(
            SequenceExtractor([[1.0, 0.0], [0.0, 1.0]]),
            [split_clean, exact_clean],
            cluster_threshold=0.5,
            num_clusters=-1,
            assignment_similarity_threshold=0.5,
            clusterer_factory=ControlledSpeakerAssignmentTest._factory(clusterer),
        )

        self.assertEqual((errors, assigned, unknown, skipped), (0, 0, 1, 0))
        self.assertEqual(clusterer.received.shape, (1, 2))
        self.assertEqual(exact_clean.speaker_id, "speaker_00")
        self.assertEqual(exact_clean.speaker_assignment_source, "clean_cluster")
        self.assertEqual(split_clean.speaker_id, "UNKNOWN")

    def test_short_turn_inherits_a_clean_cluster_through_same_fused_mask(self):
        clean = self._segment(
            1, 0, 5000, mask=1, confidence=0.99, clean_spans=((0, 5000),)
        )
        short = self._segment(2, 8000, 9000, mask=1, confidence=0.95)
        clusterer = RecordingClusterer([4])

        errors, assigned, unknown, skipped = assign_speaker_ids_with_centroids(
            # Deliberately make the short embedding unlike the clean centroid:
            # mask continuity, not global embedding, must decide this turn.
            SequenceExtractor([[1.0, 0.0], [0.0, 1.0]]),
            [clean, short],
            cluster_threshold=0.5,
            num_clusters=-1,
            assignment_similarity_threshold=0.95,
            clusterer_factory=ControlledSpeakerAssignmentTest._factory(clusterer),
        )

        self.assertEqual((errors, assigned, unknown, skipped), (0, 1, 0, 0))
        self.assertEqual([clean.speaker_id, short.speaker_id], ["speaker_00", "speaker_00"])
        self.assertEqual(clean.speaker_assignment_source, "clean_cluster")
        self.assertEqual(short.speaker_assignment_source, "local_mask_inherit")
        self.assertIsNone(short.cluster_assignment_similarity)


if __name__ == "__main__":
    unittest.main()

class NativeClusteringTest(unittest.TestCase):
    def test_default_factory_clusters_float32_embedding_matrix(self):
        cluster = _default_clusterer_factory(0.5, -1)
        labels = list(cluster(np.asarray([[1.0, 0.0], [0.99, 0.1]], dtype=np.float32)))

        self.assertEqual(len(labels), 2)
        self.assertTrue(all(isinstance(label, int) for label in labels))



class SequenceExtractor(FakeExtractor):
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)

    def compute(self, stream):
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class DiarizationSimilarityTest(unittest.TestCase):
    def test_populates_similarity_without_overwriting_diarization_ids(self):
        segments = [
            SpeechSegment(1, 0, 1000, np.zeros(16000, dtype=np.float32), speaker_id="speaker_00"),
            SpeechSegment(2, 1000, 2000, np.zeros(16000, dtype=np.float32), speaker_id="speaker_01"),
        ]

        error_count = populate_previous_segment_similarities(
            SequenceExtractor([[1.0, 0.0], [0.0, 1.0]]), segments
        )

        self.assertEqual(error_count, 0)
        self.assertEqual([item.speaker_id for item in segments], ["speaker_00", "speaker_01"])
        self.assertIsNone(segments[0].previous_segment_similarity)
        self.assertEqual(segments[1].previous_segment_similarity, 0.0)

    def test_embedding_failure_leaves_existing_ids_and_neighbor_similarity_undefined(self):
        segments = [
            SpeechSegment(1, 0, 1000, np.zeros(16000, dtype=np.float32), speaker_id="speaker_00"),
            SpeechSegment(2, 1000, 2000, np.zeros(16000, dtype=np.float32), speaker_id="speaker_01"),
        ]

        error_count = populate_previous_segment_similarities(
            SequenceExtractor([[1.0, 0.0], RuntimeError("bad embedding")]), segments
        )

        self.assertEqual(error_count, 1)
        self.assertEqual([item.speaker_id for item in segments], ["speaker_00", "speaker_01"])
        self.assertIsNone(segments[0].previous_segment_similarity)
        self.assertIsNone(segments[1].previous_segment_similarity)


class RecordingClusterer:
    def __init__(self, labels):
        self.labels = labels
        self.received = None

    def __call__(self, embeddings):
        self.received = embeddings.copy()
        return self.labels


class ControlledSpeakerAssignmentTest(unittest.TestCase):
    @staticmethod
    def _segment(index, start_ms, end_ms, composition):
        return SpeechSegment(
            index,
            start_ms,
            end_ms,
            np.zeros((end_ms - start_ms) * 16, dtype=np.float32),
            speaker_composition=composition,
            clean_spans=[(start_ms, end_ms)] if composition == "single_speaker" else [],
        )

    @staticmethod
    def _factory(clusterer):
        def factory(_threshold, _num_clusters):
            return clusterer
        return factory

    def test_clusters_only_long_single_speaker_segments_and_backfills_excluded(self):
        eligible_a = self._segment(1, 0, 3000, "single_speaker")
        eligible_b = self._segment(2, 3000, 6000, "single_speaker")
        short = self._segment(3, 6000, 6500, "single_speaker")
        overlap = self._segment(4, 6500, 8000, "overlapped_speakers")
        clusterer = RecordingClusterer([4, 4])

        errors, assigned_excluded, unknown_excluded, skipped = assign_speaker_ids_with_centroids(
            SequenceExtractor([[1.0, 0.0], [0.8, 0.6], [0.9, 0.1], [0.0, 1.0]]),
            [eligible_a, eligible_b, short, overlap],
            cluster_threshold=0.5,
            num_clusters=-1,
            assignment_similarity_threshold=0.5,
            clusterer_factory=self._factory(clusterer),
        )

        self.assertEqual(errors, 0)
        self.assertEqual((assigned_excluded, unknown_excluded), (1, 1))
        self.assertEqual(clusterer.received.shape, (2, 2))
        self.assertTrue(np.allclose(clusterer.received, [[1.0, 0.0], [0.8, 0.6]]))
        self.assertEqual(
            [eligible_a.speaker_id, eligible_b.speaker_id], ["speaker_00", "speaker_00"]
        )
        self.assertEqual(short.speaker_id, "speaker_00")
        self.assertGreaterEqual(short.cluster_assignment_similarity, 0.5)
        self.assertEqual(overlap.speaker_id, "UNKNOWN")
        self.assertLess(overlap.cluster_assignment_similarity, 0.5)
        self.assertIsNone(eligible_a.previous_segment_similarity)
        self.assertGreater(eligible_b.previous_segment_similarity, 0.0)
        self.assertGreater(short.previous_segment_similarity, 0.0)
        self.assertGreater(overlap.previous_segment_similarity, 0.0)

    def test_no_eligible_embeddings_marks_all_excluded_unknown_without_clustering(self):
        short = self._segment(1, 0, 500, "single_speaker")
        overlap = self._segment(2, 500, 2000, "overlapped_speakers")

        errors, assigned_excluded, unknown_excluded, skipped = assign_speaker_ids_with_centroids(
            SequenceExtractor([[1.0, 0.0], [0.0, 1.0]]),
            [short, overlap],
            cluster_threshold=0.5,
            num_clusters=-1,
            assignment_similarity_threshold=0.5,
            clusterer_factory=lambda *_args: self.fail("clusterer must not be created"),
        )

        self.assertEqual((errors, assigned_excluded, unknown_excluded, skipped), (0, 0, 2, 0))
        self.assertEqual([short.speaker_id, overlap.speaker_id], ["UNKNOWN", "UNKNOWN"])
        self.assertIsNone(short.cluster_assignment_similarity)
        self.assertIsNone(overlap.cluster_assignment_similarity)

    def test_embedding_failure_does_not_stop_later_eligible_clustering(self):
        failed = self._segment(1, 0, 3000, "single_speaker")
        good = self._segment(2, 3000, 6000, "single_speaker")
        clusterer = RecordingClusterer([9])

        errors, assigned_excluded, unknown_excluded, skipped = assign_speaker_ids_with_centroids(
            SequenceExtractor([RuntimeError("bad embedding"), [1.0, 0.0]]),
            [failed, good],
            cluster_threshold=0.5,
            num_clusters=-1,
            assignment_similarity_threshold=0.5,
            clusterer_factory=self._factory(clusterer),
        )

        self.assertEqual((errors, assigned_excluded, unknown_excluded, skipped), (1, 0, 0, 0))
        self.assertEqual(failed.speaker_id, "UNKNOWN")
        self.assertIsNone(failed.embedding)
        self.assertEqual(good.speaker_id, "speaker_00")
        self.assertEqual(clusterer.received.shape, (1, 2))

    def test_explicit_cluster_count_cannot_exceed_successful_eligible_embeddings(self):
        eligible = self._segment(1, 0, 3000, "single_speaker")
        failed = self._segment(2, 3000, 6000, "single_speaker")

        with self.assertRaisesRegex(
            ValueError, "num_clusters=2 exceeds successful eligible embeddings=1"
        ):
            assign_speaker_ids_with_centroids(
                SequenceExtractor([[1.0, 0.0], RuntimeError("bad embedding")]),
                [eligible, failed],
                cluster_threshold=0.5,
                num_clusters=2,
                assignment_similarity_threshold=0.5,
            )


    def test_invalid_asr_segments_skip_embedding_and_keep_dash_speaker(self):
        invalid = self._segment(1, 0, 3000, "single_speaker")
        invalid.asr_valid = 0
        good = self._segment(2, 3000, 6000, "single_speaker")
        clusterer = RecordingClusterer([1])

        errors, assigned_excluded, unknown_excluded, skipped = assign_speaker_ids_with_centroids(
            SequenceExtractor([[1.0, 0.0]]),
            [invalid, good],
            cluster_threshold=0.5,
            num_clusters=-1,
            assignment_similarity_threshold=0.5,
            clusterer_factory=self._factory(clusterer),
        )

        self.assertEqual((errors, assigned_excluded, unknown_excluded, skipped), (0, 0, 0, 0))
        self.assertEqual(invalid.speaker_id, "-")
        self.assertIsNone(invalid.embedding)
        self.assertEqual(good.speaker_id, "speaker_00")
        self.assertEqual(clusterer.received.shape, (1, 2))


class EmbeddingOverlapCropTest(unittest.TestCase):
    def test_embedding_skips_overlap_and_a_short_tail_after_it(self):
        samples = np.arange(32000, dtype=np.float32)
        segment = SpeechSegment(
            1,
            0,
            2000,
            samples,
            speaker_composition="single_speaker",
            overlap_regions=[(0, 400)],
        )
        cropped = samples_for_embedding(segment)
        self.assertEqual(cropped.size, 9600)
        np.testing.assert_array_equal(cropped, samples[22400:])

    def test_embedding_skips_whole_segment_when_only_overlap_and_short_tail_remain(self):
        samples = np.arange(1776 * 16, dtype=np.float32)
        segment = SpeechSegment(
            1,
            12944,
            14720,
            samples,
            speaker_composition="single_speaker",
            overlap_regions=[(12944, 13803)],
        )
        cropped = samples_for_embedding(segment)
        self.assertEqual(cropped.size, 0)

    def test_embedding_skips_absorbed_overlap_prefix(self):
        samples = np.arange(32000, dtype=np.float32)
        segment = SpeechSegment(
            1,
            0,
            2000,
            samples,
            speaker_composition="single_speaker",
            overlap_regions=[(0, 400)],
        )
        received = []

        class CaptureExtractor(FakeExtractor):
            def compute(self, stream):
                received.append(np.ascontiguousarray(stream.waveform, dtype=np.float32))
                return [1.0, 0.0]

        clusterer = RecordingClusterer([3])
        assign_speaker_ids_with_centroids(
            CaptureExtractor(),
            [segment],
            cluster_threshold=0.5,
            num_clusters=-1,
            assignment_similarity_threshold=0.5,
            clusterer_factory=lambda *_args: clusterer,
        )

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].size, 9600)
        np.testing.assert_array_equal(received[0], samples[22400:])
        self.assertEqual(segment.speaker_id, "UNKNOWN")


class NoUsableEmbeddingAudioTest(unittest.TestCase):
    class ExtractorMustNotRun:
        def __init__(self):
            self.create_stream_calls = 0

        def create_stream(self):
            self.create_stream_calls += 1
            raise AssertionError("extractor must not run when no usable embedding audio exists")

    def test_full_overlap_is_expected_skip_without_extractor_call(self):
        segment = SpeechSegment(
            1,
            0,
            1000,
            np.zeros(16000, dtype=np.float32),
            speaker_composition="overlapped_speakers",
            overlap_regions=[(0, 1000)],
        )
        extractor = self.ExtractorMustNotRun()

        errors, assigned, unknown, skipped = assign_speaker_ids_with_centroids(
            extractor,
            [segment],
            cluster_threshold=0.5,
            num_clusters=-1,
            assignment_similarity_threshold=0.5,
        )

        self.assertEqual((errors, assigned, unknown, skipped), (0, 0, 1, 1))
        self.assertEqual(extractor.create_stream_calls, 0)
        self.assertEqual(segment.speaker_id, "UNKNOWN")
        self.assertEqual(segment.speaker_assignment_source, "no_usable_embedding_audio")
        self.assertIsNone(segment.embedding_error)

    def test_overlap_padding_can_skip_a_single_speaker_turn_without_extractor_call(self):
        segment = SpeechSegment(
            1,
            0,
            2000,
            np.zeros(32000, dtype=np.float32),
            speaker_composition="single_speaker",
            overlap_regions=[(0, 1000)],
        )
        extractor = self.ExtractorMustNotRun()

        errors, assigned, unknown, skipped = assign_speaker_ids_with_centroids(
            extractor,
            [segment],
            cluster_threshold=0.5,
            num_clusters=-1,
            assignment_similarity_threshold=0.5,
        )

        self.assertEqual((errors, assigned, unknown, skipped), (0, 0, 1, 1))
        self.assertEqual(extractor.create_stream_calls, 0)
        self.assertEqual(segment.speaker_id, "UNKNOWN")
        self.assertEqual(segment.speaker_assignment_source, "no_usable_embedding_audio")
        self.assertIsNone(segment.embedding_error)

    def test_tiny_overlap_remainder_is_expected_skip_without_extractor_call(self):
        segment = SpeechSegment(
            1,
            229406,
            231648,
            np.zeros(35872, dtype=np.float32),
            speaker_composition="single_speaker",
            overlap_regions=[(229416, 230445), (231238, 231289)],
        )
        extractor = self.ExtractorMustNotRun()

        errors, assigned, unknown, skipped = assign_speaker_ids_with_centroids(
            extractor,
            [segment],
            cluster_threshold=0.5,
            num_clusters=-1,
            assignment_similarity_threshold=0.5,
        )

        self.assertEqual((errors, assigned, unknown, skipped), (0, 0, 1, 1))
        self.assertEqual(extractor.create_stream_calls, 0)
        self.assertEqual(segment.speaker_id, "UNKNOWN")
        self.assertEqual(segment.speaker_assignment_source, "no_usable_embedding_audio")
        self.assertIsNone(segment.embedding_error)

    def test_nonempty_extractor_failure_remains_embedding_error(self):
        segment = SpeechSegment(
            1,
            0,
            1000,
            np.zeros(16000, dtype=np.float32),
            speaker_composition="single_speaker",
        )

        errors, assigned, unknown, skipped = assign_speaker_ids_with_centroids(
            SequenceExtractor([RuntimeError("Titanet compute failed")]),
            [segment],
            cluster_threshold=0.5,
            num_clusters=-1,
            assignment_similarity_threshold=0.5,
        )

        self.assertEqual((errors, assigned, unknown, skipped), (1, 0, 1, 0))
        self.assertEqual(segment.speaker_id, "UNKNOWN")
        self.assertEqual(segment.embedding_error, "Titanet compute failed")

    def test_no_usable_audio_is_not_overridden_by_local_mask_inheritance(self):
        clean = SpeechSegment(
            1,
            0,
            3000,
            np.zeros(48000, dtype=np.float32),
            speaker_composition="single_speaker",
            clean_spans=[(0, 3000)],
            local_speaker_mask=1,
            local_speaker_mask_confidence=0.99,
        )
        skipped_segment = SpeechSegment(
            2,
            3000,
            4000,
            np.zeros(16000, dtype=np.float32),
            speaker_composition="overlapped_speakers",
            overlap_regions=[(3000, 4000)],
            local_speaker_mask=1,
            local_speaker_mask_confidence=0.99,
        )
        clusterer = RecordingClusterer([1])

        errors, assigned, unknown, skipped = assign_speaker_ids_with_centroids(
            SequenceExtractor([[1.0, 0.0]]),
            [clean, skipped_segment],
            cluster_threshold=0.5,
            num_clusters=-1,
            assignment_similarity_threshold=0.5,
            clusterer_factory=ControlledSpeakerAssignmentTest._factory(clusterer),
        )

        self.assertEqual((errors, assigned, unknown, skipped), (0, 0, 1, 1))
        self.assertEqual(skipped_segment.speaker_id, "UNKNOWN")
        self.assertEqual(skipped_segment.speaker_assignment_source, "no_usable_embedding_audio")
