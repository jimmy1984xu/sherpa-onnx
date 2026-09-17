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
        )

    @staticmethod
    def _factory(clusterer):
        def factory(_threshold, _num_clusters):
            return clusterer
        return factory

    def test_clusters_only_long_single_speaker_segments_and_backfills_excluded(self):
        eligible_a = self._segment(1, 0, 1500, "single_speaker")
        eligible_b = self._segment(2, 1500, 3000, "single_speaker")
        short = self._segment(3, 3000, 3500, "single_speaker")
        overlap = self._segment(4, 3500, 5000, "overlapped_speakers")
        clusterer = RecordingClusterer([4, 4])

        errors, assigned_excluded, unknown_excluded = assign_speaker_ids_with_centroids(
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
        self.assertEqual(overlap.speaker_id, "unknown")
        self.assertLess(overlap.cluster_assignment_similarity, 0.5)
        self.assertIsNone(eligible_a.previous_segment_similarity)
        self.assertGreater(eligible_b.previous_segment_similarity, 0.0)
        self.assertGreater(short.previous_segment_similarity, 0.0)
        self.assertGreater(overlap.previous_segment_similarity, 0.0)

    def test_no_eligible_embeddings_marks_all_excluded_unknown_without_clustering(self):
        short = self._segment(1, 0, 500, "single_speaker")
        overlap = self._segment(2, 500, 2000, "overlapped_speakers")

        errors, assigned_excluded, unknown_excluded = assign_speaker_ids_with_centroids(
            SequenceExtractor([[1.0, 0.0], [0.0, 1.0]]),
            [short, overlap],
            cluster_threshold=0.5,
            num_clusters=-1,
            assignment_similarity_threshold=0.5,
            clusterer_factory=lambda *_args: self.fail("clusterer must not be created"),
        )

        self.assertEqual((errors, assigned_excluded, unknown_excluded), (0, 0, 2))
        self.assertEqual([short.speaker_id, overlap.speaker_id], ["unknown", "unknown"])
        self.assertIsNone(short.cluster_assignment_similarity)
        self.assertIsNone(overlap.cluster_assignment_similarity)

    def test_embedding_failure_does_not_stop_later_eligible_clustering(self):
        failed = self._segment(1, 0, 1500, "single_speaker")
        good = self._segment(2, 1500, 3000, "single_speaker")
        clusterer = RecordingClusterer([9])

        errors, assigned_excluded, unknown_excluded = assign_speaker_ids_with_centroids(
            SequenceExtractor([RuntimeError("bad embedding"), [1.0, 0.0]]),
            [failed, good],
            cluster_threshold=0.5,
            num_clusters=-1,
            assignment_similarity_threshold=0.5,
            clusterer_factory=self._factory(clusterer),
        )

        self.assertEqual((errors, assigned_excluded, unknown_excluded), (1, 0, 0))
        self.assertEqual(failed.speaker_id, "unknown")
        self.assertIsNone(failed.embedding)
        self.assertEqual(good.speaker_id, "speaker_00")
        self.assertEqual(clusterer.received.shape, (1, 2))

    def test_explicit_cluster_count_cannot_exceed_successful_eligible_embeddings(self):
        eligible = self._segment(1, 0, 1500, "single_speaker")
        failed = self._segment(2, 1500, 3000, "single_speaker")

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
        invalid = self._segment(1, 0, 1500, "single_speaker")
        invalid.asr_valid = 0
        good = self._segment(2, 1500, 3000, "single_speaker")
        clusterer = RecordingClusterer([1])

        errors, assigned_excluded, unknown_excluded = assign_speaker_ids_with_centroids(
            SequenceExtractor([[1.0, 0.0]]),
            [invalid, good],
            cluster_threshold=0.5,
            num_clusters=-1,
            assignment_similarity_threshold=0.5,
            clusterer_factory=self._factory(clusterer),
        )

        self.assertEqual((errors, assigned_excluded, unknown_excluded), (0, 0, 0))
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
        self.assertEqual(segment.speaker_id, "unknown")
