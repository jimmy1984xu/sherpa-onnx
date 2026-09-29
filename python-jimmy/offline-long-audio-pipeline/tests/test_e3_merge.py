import unittest
from pathlib import Path
import sys

import numpy as np

PIPELINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_DIR))

from e3_merge import merge_initial_clusters, robust_center


class E3MergeTest(unittest.TestCase):
    def test_trimmed_center_preserves_all_members(self):
        embeddings = np.asarray(
            [[1.0, 0.0], [0.99, 0.1], [0.98, 0.2], [0.0, 1.0]],
            dtype=np.float32,
        )
        center = robust_center(embeddings, [0, 1, 2], trim_ratio=0.10)
        self.assertEqual(center.shape, (2,))
        self.assertAlmostEqual(float(np.linalg.norm(center)), 1.0, places=6)

    def test_merges_only_complete_initial_clusters_and_reports_operation(self):
        embeddings = np.asarray(
            [
                [1.0, 0.0],
                [0.99, 0.1],
                [0.98, 0.2],
                [0.97, 0.24],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
        labels, diagnostics = merge_initial_clusters(
            embeddings,
            [0, 0, 1, 1, 2],
            similarity_threshold=0.75,
        )
        self.assertEqual(len(labels), len(embeddings))
        self.assertEqual(len(set(labels)), 2)
        self.assertEqual(labels[0], labels[2])
        self.assertEqual(labels[1], labels[3])
        self.assertNotEqual(labels[0], labels[4])
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(diagnostics[0]["left_member_count"], 2)
        self.assertEqual(diagnostics[0]["right_member_count"], 2)
        self.assertGreaterEqual(diagnostics[0]["similarity"], 0.75)

    def test_does_not_merge_below_threshold(self):
        embeddings = np.asarray(
            [[1.0, 0.0], [0.0, 1.0]], dtype=np.float32
        )
        labels, diagnostics = merge_initial_clusters(
            embeddings, [0, 1], similarity_threshold=0.75
        )
        self.assertEqual(len(set(labels)), 2)
        self.assertEqual(diagnostics, [])

    def test_merge_is_invariant_to_initial_cluster_order(self):
        embeddings = np.asarray(
            [[1.0, 0.0], [0.99, 0.1], [0.98, 0.2], [0.97, 0.24]],
            dtype=np.float32,
        )
        forward, _ = merge_initial_clusters(embeddings, [0, 0, 1, 1])
        reversed_labels, _ = merge_initial_clusters(embeddings, [1, 1, 0, 0])
        self.assertEqual(forward, reversed_labels)

    def test_rejects_mismatched_inputs(self):
        with self.assertRaises(ValueError):
            merge_initial_clusters(np.ones((2, 2), dtype=np.float32), [0])


if __name__ == "__main__":
    unittest.main()
