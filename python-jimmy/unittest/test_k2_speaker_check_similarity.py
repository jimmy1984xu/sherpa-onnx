#!/usr/bin/env python3

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


def load_module():
    script_path = Path(__file__).resolve().parent.parent / "k2-speaker-check-similarity.py"
    spec = importlib.util.spec_from_file_location("k2_speaker_check_similarity", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestK2SpeakerCheckSimilarity(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_deserialize_embedding_accepts_prefixed_float_list(self):
        emb = self.module.deserialize_embedding("NEMO_EN_TITANET_LARGE:0.1,0.2,0.3")

        self.assertEqual(emb.dtype.name, "float32")
        self.assertTrue(self.module.np.allclose(emb, self.module.np.array([0.1, 0.2, 0.3], dtype=self.module.np.float32)))

    def test_read_embeddings_accepts_name_and_prefixed_float_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "registered.txt"
            path.write_text(
                "\n".join(
                    [
                        "AbbyLi NEMO_EN_TITANET_LARGE:1.0,0.0",
                        "BobLi 0.0,1.0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            embeddings, durations = self.module.read_embeddings(path)

            self.assertEqual(sorted(embeddings.keys()), ["AbbyLi", "BobLi"])
            self.assertTrue(self.module.np.allclose(embeddings["AbbyLi"], self.module.np.array([1.0, 0.0], dtype=self.module.np.float32)))
            self.assertTrue(self.module.np.allclose(embeddings["BobLi"], self.module.np.array([0.0, 1.0], dtype=self.module.np.float32)))
            self.assertEqual(durations["AbbyLi"], 0)
            self.assertEqual(durations["BobLi"], 0)

    def test_compute_pairwise_similarity_is_sorted_descending(self):
        normed = {
            "AbbyLi": self.module.normalize_embeddings({"AbbyLi": self.module.np.array([1.0, 0.0], dtype=self.module.np.float32)})["AbbyLi"],
            "BobLi": self.module.normalize_embeddings({"BobLi": self.module.np.array([0.9, 0.1], dtype=self.module.np.float32)})["BobLi"],
            "CindyLi": self.module.normalize_embeddings({"CindyLi": self.module.np.array([0.0, 1.0], dtype=self.module.np.float32)})["CindyLi"],
        }

        results = self.module.compute_pairwise_similarity(normed)

        self.assertEqual(results[0][0:2], ("AbbyLi", "BobLi"))
        self.assertGreaterEqual(results[0][2], results[1][2])
        self.assertGreaterEqual(results[1][2], results[2][2])


if __name__ == "__main__":
    unittest.main()
