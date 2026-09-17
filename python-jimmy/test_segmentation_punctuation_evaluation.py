import contextlib
import importlib.util
import io
import json
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

    def test_variants_share_common_arguments_and_streaming_sets_active_flags(self):
        self.assertEqual(module.common_option_map(self.baseline), module.common_option_map(self.streaming))
        self.assertEqual(
            module.STREAMING_INVARIANTS,
            {
                "--segmentation-chunk-ms": "32",
                "--segmentation-num-threads": "4",
                "--min-duration-on": "0.5",
                "--min-duration-off": "0.5",
                "--change-vote-threshold": "0.5",
            },
        )
        for option, expected_value in module.STREAMING_INVARIANTS.items():
            self.assertNotIn(option, self.baseline)
            self.assertEqual(module.option_value(self.streaming, option), expected_value)
        self.assertEqual(module.option_value(self.baseline, "--num-clusters"), "-1")
        self.assertEqual(module.option_value(self.streaming, "--cluster-threshold"), "0.6")

    def test_validate_variant_fairness_accepts_standard_variants(self):
        self.assertIsNone(module.validate_variant_fairness(self.baseline, self.streaming))

    def test_validate_variant_fairness_rejects_streaming_invariant_absence_or_mismatch(self):
        for option, expected_value in module.STREAMING_INVARIANTS.items():
            with self.subTest(option=option, condition="missing"):
                missing = list(self.streaming)
                option_index = missing.index(option)
                del missing[option_index : option_index + 2]
                with self.assertRaisesRegex(ValueError, option):
                    module.validate_variant_fairness(self.baseline, missing)
            with self.subTest(option=option, condition="mismatched"):
                mismatched = list(self.streaming)
                mismatched[mismatched.index(option) + 1] = f"{expected_value}-wrong"
                with self.assertRaisesRegex(ValueError, option):
                    module.validate_variant_fairness(self.baseline, mismatched)

    def test_validate_variant_fairness_rejects_baseline_streaming_option(self):
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

    def test_actual_result_json_schema_becomes_metrics_compatible_three_column_asr_file(self):
        payload = {
            "segments": [
                {
                    "segment_id": "speaker_turn_0001_1000_2500",
                    "duration_ms": 1500,
                    "speaker_id": "(spk_1)",
                    "asr_text": "你好",
                },
                {
                    "segment_id": "0002_2600_3000",
                    "duration_ms": 400,
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

    def test_result_json_rejects_missing_or_malformed_segment_id(self):
        invalid_segments = (
            {},
            {"segment_id": None},
            {"segment_id": "speaker_turn_1000_end"},
            {"segment_id": "only_one_component"},
        )
        for segment in invalid_segments:
            payload = {"segments": [segment]}
            with self.subTest(segment=segment):
                with self.assertRaisesRegex(ValueError, "segment_id"):
                    module.write_metrics_asr_file(Path(self.idir), "sample", payload)

    def test_result_json_rejects_inconsistent_duration_ms(self):
        payload = {
            "segments": [
                {
                    "segment_id": "0001_1000_2500",
                    "duration_ms": 1499,
                }
            ]
        }
        with self.assertRaisesRegex(ValueError, "duration_ms"):
            module.write_metrics_asr_file(Path(self.idir), "sample", payload)

    def test_result_json_rejects_nonpositive_segment_duration(self):
        payload = {"segments": [{"segment_id": "0001_1200_1200"}]}
        with self.assertRaisesRegex(ValueError, "invalid segment interval: 1200-1200"):
            module.write_metrics_asr_file(Path(self.idir), "sample", payload)


class CliTest(unittest.TestCase):
    def test_main_prints_json_argv_that_round_trips_paths_with_spaces_and_unicode(self):
        audio = Path("测试 音频/sample file.pcm")
        output_root = Path("输出 目录")
        paths = module.ModelPaths(
            asr_dir=Path("模型 目录/asr 模型"),
            vad_dir=Path("模型 目录/vad"),
            speaker_dir=Path("模型 目录/speaker"),
            segmentation_dir=Path("模型 目录/segmentation"),
        )
        expected_argv = module.build_invocation(
            "streaming", audio, module.build_common_arguments(paths, output_root)
        )
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = module.main([
                "--kind", "streaming",
                "--audio", str(audio),
                "--output-root", str(output_root),
                "--asr-dir", str(paths.asr_dir),
                "--vad-dir", str(paths.vad_dir),
                "--speaker-dir", str(paths.speaker_dir),
                "--segmentation-dir", str(paths.segmentation_dir),
            ])
        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(stdout.getvalue()), {"argv": expected_argv})


if __name__ == "__main__":
    unittest.main()
