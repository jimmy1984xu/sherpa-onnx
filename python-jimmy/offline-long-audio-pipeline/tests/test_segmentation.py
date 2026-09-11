import sys
import unittest
from pathlib import Path

import numpy as np

PIPELINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_DIR))

from segmentation import SegmentationRuntime


class FakeMeta:
    def __init__(self, metadata):
        self.custom_metadata_map = metadata


class FakeTensor:
    def __init__(self, name):
        self.name = name


class FakeSession:
    def __init__(self, metadata=None):
        self.metadata = metadata or {
            "sample_rate": "16000",
            "window_size": "160000",
            "receptive_field_shift": "270",
            "num_speakers": "3",
            "powerset_max_classes": "2",
            "num_classes": "7",
        }

    def get_modelmeta(self):
        return FakeMeta(self.metadata)

    def get_inputs(self):
        return [FakeTensor("x")]

    def get_outputs(self):
        return [FakeTensor("y")]

    def run(self, _outputs, inputs):
        batch = inputs["x"].shape[0]
        logits = np.zeros((batch, 3, 7), dtype=np.float32)
        logits[:, 0, 1] = 1.0
        logits[:, 1, 4] = 1.0
        logits[:, 2, 0] = 1.0
        return [logits]


class SegmentationRuntimeTest(unittest.TestCase):
    def test_decodes_powerset_logits_to_speaker_counts(self):
        runtime = SegmentationRuntime(FakeSession())
        logits = FakeSession().run([], {"x": np.zeros((1, 1, 160000), np.float32)})[0]
        np.testing.assert_array_equal(
            runtime.decode_speaker_counts(logits),
            np.array([[1, 2, 0]], dtype=np.int8),
        )

    def test_decodes_powerset_logits_to_local_speaker_masks(self):
        runtime = SegmentationRuntime(FakeSession())
        logits = np.full((1, 3, 7), -1.0, dtype=np.float32)
        logits[0, 0, 1] = 3.0  # A
        logits[0, 1, 2] = 3.0  # B
        logits[0, 2, 4] = 3.0  # A+B

        np.testing.assert_array_equal(
            runtime.decode_speaker_masks(logits),
            np.array([[[1, 0, 0], [0, 1, 0], [1, 1, 0]]], dtype=np.int8),
        )

    def test_decodes_powerset_logits_to_class_indices(self):
        runtime = SegmentationRuntime(FakeSession())
        logits = np.full((1, 3, 7), -1.0, dtype=np.float32)
        logits[0, 0, 1] = 3.0
        logits[0, 1, 2] = 3.0
        logits[0, 2, 6] = 3.0

        np.testing.assert_array_equal(
            runtime.decode_speaker_classes(logits),
            np.array([[1, 2, 6]], dtype=np.int64),
        )

    def test_remaps_all_powerset_probabilities_when_tracks_are_permuted(self):
        runtime = SegmentationRuntime(FakeSession())
        probabilities = np.zeros((1, 7), dtype=np.float32)
        probabilities[0, 1] = 1.0  # local track A
        probabilities[0, 6] = 0.5  # local tracks B+C

        remapped = runtime._remap_class_probabilities(
            probabilities,
            (1, 0, 2),  # local A -> global B, local B -> global A
        )

        self.assertAlmostEqual(float(remapped[0, 2]), 1.0)
        self.assertAlmostEqual(float(remapped[0, 5]), 0.5)  # A+C -> global B+C

    def test_rejects_missing_required_metadata(self):
        metadata = dict(FakeSession().metadata)
        metadata.pop("window_size")
        with self.assertRaisesRegex(ValueError, "window_size"):
            SegmentationRuntime(FakeSession(metadata))


if __name__ == "__main__":
    unittest.main()

class AllSingleSpeakerSession(FakeSession):
    def run(self, _outputs, inputs):
        batch = inputs["x"].shape[0]
        logits = np.zeros((batch, 589, 7), dtype=np.float32)
        logits[:, :, 1] = 1.0
        return [logits]


class SegmentationInferenceTest(unittest.TestCase):
    def test_infers_clipped_activity_span_for_padded_short_audio(self):
        from segmentation import SpeakerCountSpan

        runtime = SegmentationRuntime(AllSingleSpeakerSession())
        spans = runtime.infer_speaker_count_spans(np.zeros(8000, dtype=np.float32))

        self.assertEqual(spans, [SpeakerCountSpan(0, 500, 1, (1, 0, 0), 1)])
