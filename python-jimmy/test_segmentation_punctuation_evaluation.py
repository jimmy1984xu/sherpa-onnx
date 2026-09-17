import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("segmentation-punctuation-evaluation.py")


def load_module():
    if not MODULE_PATH.is_file():
        raise ImportError(f"Missing evaluation module: {MODULE_PATH}")
    spec = importlib.util.spec_from_file_location("segmentation_punctuation_evaluation", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load evaluation module: {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = load_module()


def model_paths():
    return module.ModelPaths(
        asr_dir=Path("models/asr"),
        vad_dir=Path("models/vad"),
        speaker_dir=Path("models/speaker"),
        segmentation_dir=Path("models/segmentation"),
    )


class InvocationTest(unittest.TestCase):
    def setUp(self):
        self.common = module.build_common_arguments(model_paths(), output_root=Path("out"))
        self.baseline = module.build_invocation("baseline", Path("a.pcm"), self.common)
        self.streaming = module.build_invocation("streaming", Path("a.pcm"), self.common)

    def test_variants_share_all_common_arguments_and_only_streaming_adds_chunk(self):
        self.assertEqual(module.common_option_map(self.baseline), module.common_option_map(self.streaming))
        self.assertNotIn("--segmentation-chunk-ms", self.baseline)
        self.assertEqual(
            self.streaming[self.streaming.index("--segmentation-chunk-ms") + 1], "32"
        )
        self.assertEqual(module.option_value(self.baseline, "--num-clusters"), "-1")
        self.assertEqual(module.option_value(self.streaming, "--cluster-threshold"), "0.6")

    def test_validate_variant_fairness_accepts_standard_variants(self):
        self.assertIsNone(module.validate_variant_fairness(self.baseline, self.streaming))

    def test_validate_variant_fairness_rejects_baseline_chunk_option(self):
        with self.assertRaisesRegex(ValueError, "baseline invocation"):
            module.validate_variant_fairness(
                [*self.baseline, "--segmentation-chunk-ms", "32"], self.streaming
            )

    def test_validate_variant_fairness_rejects_a_changed_common_option(self):
        changed_streaming = list(self.streaming)
        threshold_index = changed_streaming.index("--cluster-threshold") + 1
        changed_streaming[threshold_index] = "0.7"
        with self.assertRaisesRegex(ValueError, "common arguments differ"):
            module.validate_variant_fairness(self.baseline, changed_streaming)


class ResultAdapterTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.idir = self.temp_dir.name

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_result_json_becomes_metrics_compatible_three_column_asr_file(self):
        payload = {
            "segments": [
                {
                    "start_ms": 1000,
                    "end_ms": 2500,
                    "speaker_id": "(spk_1)",
                    "asr_text": "你好",
                },
                {
                    "start_ms": 2600,
                    "end_ms": 3000,
                    "speaker_id": "spk_0",
                    "asr_text": "世界",
                },
            ]
        }
        output = module.write_metrics_asr_file(Path(self.idir), "sample", payload)
        self.assertEqual(
            output.read_text(encoding="utf-8").splitlines(),
            ["sample_1000_1500 spk_1 你好", "sample_2600_400 spk_0 世界"],
        )

    def test_result_json_rejects_nonpositive_segment_duration(self):
        payload = {"segments": [{"start_ms": 1200, "end_ms": 1200}]}
        with self.assertRaisesRegex(ValueError, "invalid segment interval: 1200-1200"):
            module.write_metrics_asr_file(Path(self.idir), "sample", payload)


class CliTest(unittest.TestCase):
    def test_main_prints_a_command_preview_without_executing_pipeline(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = module.main([
                "--kind", "streaming",
                "--audio", "sample.pcm",
                "--output-root", "out",
                "--asr-dir", "models/asr",
                "--vad-dir", "models/vad",
                "--speaker-dir", "models/speaker",
                "--segmentation-dir", "models/segmentation",
            ])
        command = stdout.getvalue().strip()
        self.assertEqual(exit_code, 0)
        self.assertIn("offline-long-audio-pipeline-asr-speaker-segmentation.py", command)
        self.assertIn("--segmentation-chunk-ms 32", command)


if __name__ == "__main__":
    unittest.main()
