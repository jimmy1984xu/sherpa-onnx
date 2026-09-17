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

    @staticmethod
    def _without_option(argv, option):
        updated = list(argv)
        option_index = updated.index(option)
        del updated[option_index : option_index + 2]
        return updated

    def test_validate_variant_fairness_rejects_wrong_pipeline_entry_script(self):
        for variant, baseline, streaming in (
            ("baseline", [self.baseline[0], "wrong-baseline.py", *self.baseline[2:]], self.streaming),
            ("streaming", self.baseline, [self.streaming[0], "wrong-streaming.py", *self.streaming[2:]]),
        ):
            with self.subTest(variant=variant):
                with self.assertRaisesRegex(ValueError, variant):
                    module.validate_variant_fairness(baseline, streaming)

    def test_validate_variant_fairness_requires_every_common_option_in_both_variants(self):
        for option in module.common_option_map(self.baseline):
            with self.subTest(option=option):
                baseline = self._without_option(self.baseline, option)
                streaming = self._without_option(self.streaming, option)
                with self.assertRaisesRegex(ValueError, option):
                    module.validate_variant_fairness(baseline, streaming)
    def test_validate_variant_fairness_allows_different_run_labels(self):
        baseline = [*self.baseline, "--run-label", "baseline"]
        streaming = [*self.streaming, "--run-label", "streaming"]
        self.assertIsNone(module.validate_variant_fairness(baseline, streaming))
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

    def test_validate_variant_fairness_rejects_mismatched_audio(self):
        different_audio = list(self.streaming)
        different_audio[different_audio.index("--audio") + 1] = "second.pcm"
        with self.assertRaisesRegex(ValueError, "audio"):
            module.validate_variant_fairness(self.baseline, different_audio)
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


class TaskTwoTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    @staticmethod
    def _model_inventory(prefix: str, digest: str):
        return {
            role: {"directory": f"{prefix}/{role}", "sha256": digest}
            for role in module.MODEL_ROLES
        }

    def _valid_manifest(self):
        output_root = self.root / "evaluation"
        inputs = []
        commands = {"baseline": {}, "streaming": {}}
        remote_models = self._model_inventory("/models", "c" * 64)
        # Task 4 invokes the commands remotely, so their directories must
        # bind to the audited remote inventory, not the Windows inventory.
        # Keep the actual argv values as POSIX strings: Path("/models/asr")
        # serializes as a Windows path when these tests run on Windows.
        paths = module.ModelPaths(
            asr_dir=Path("C:/placeholder/asr"),
            vad_dir=Path("C:/placeholder/vad"),
            speaker_dir=Path("C:/placeholder/speaker"),
            segmentation_dir=Path("C:/placeholder/segmentation"),
        )
        for test_case in module.TEST_CASES:
            frozen_pcm = output_root / "01_input" / "pcm" / test_case.pcm.name
            frozen_label = output_root / "01_input" / "labels" / test_case.label.name
            inputs.append({
                "file_id": test_case.file_id,
                "pcm": {
                    "source_path": str(test_case.pcm),
                    "path": str(frozen_pcm),
                    "sha256": "a" * 64,
                },
                "label": {
                    "source_path": str(test_case.label),
                    "path": str(frozen_label),
                    "sha256": "b" * 64,
                },
            })
            common = module.build_common_arguments(paths, output_root)
            for option, role in (
                ("--asr-dir", "asr"),
                ("--vad-dir", "vad"),
                ("--speaker-dir", "speaker"),
                ("--segmentation-dir", "segmentation"),
            ):
                self._replace_option(common, option, remote_models[role]["directory"])
            commands["baseline"][test_case.file_id] = module.build_invocation(
                "baseline", frozen_pcm, common
            )
            commands["streaming"][test_case.file_id] = module.build_invocation(
                "streaming", frozen_pcm, common
            )
        return module.create_manifest(
            self.root / "manifest.json",
            inputs=inputs,
            commands=commands,
            local_models=self._model_inventory("C:/models", "c" * 64),
            remote_models=remote_models,
            windows_repo={"commit": "1" * 40, "status_porcelain": ""},
            remote_repo={"commit": "1" * 40, "status_porcelain": ""},
            created_at="2026-09-17T00:00:00Z",
            status="prepared",
        )

    def _materialize_frozen_inputs(self, manifest):
        for index, test_case in enumerate(manifest["test_cases"]):
            for kind, payload in (("pcm", f"pcm-{index}".encode()), ("label", f"label-{index}".encode())):
                frozen = Path(test_case[kind]["path"])
                frozen.parent.mkdir(parents=True, exist_ok=True)
                frozen.write_bytes(payload)
                test_case[kind]["sha256"] = module.sha256_file(frozen)

    @staticmethod
    def _replace_option(argv, option, value):
        argv[argv.index(option) + 1] = value

    def test_manifest_commands_bind_each_model_dir_to_remote_inventory(self):
        for option, role in (
            ("--asr-dir", "asr"),
            ("--vad-dir", "vad"),
            ("--speaker-dir", "speaker"),
            ("--segmentation-dir", "segmentation"),
        ):
            with self.subTest(option=option):
                manifest = self._valid_manifest()
                file_id = module.TEST_CASES[0].file_id
                for kind in ("baseline", "streaming"):
                    self._replace_option(
                        manifest["commands"][kind][file_id], option, f"/unverified/{role}"
                    )
                with self.assertRaisesRegex(ValueError, f"remote {role} model directory"):
                    module.validate_manifest(manifest)

    def test_manifest_requires_pcm_and_label_at_frozen_paths_under_command_output_root(self):
        manifest = self._valid_manifest()
        file_id = module.TEST_CASES[0].file_id
        record = manifest["test_cases"][0]
        record["pcm"]["path"] = str(self.root / "unfrozen.pcm")
        for kind in ("baseline", "streaming"):
            self._replace_option(manifest["commands"][kind][file_id], "--audio", record["pcm"]["path"])
        with self.assertRaisesRegex(ValueError, "PCM frozen path"):
            module.validate_manifest(manifest)

        manifest = self._valid_manifest()
        manifest["test_cases"][0]["label"]["path"] = str(self.root / "unfrozen_label.txt")
        with self.assertRaisesRegex(ValueError, "label frozen path"):
            module.validate_manifest(manifest)

    def test_manifest_requires_one_consistent_absolute_output_root(self):
        manifest = self._valid_manifest()
        test_case = module.TEST_CASES[0]
        record = manifest["test_cases"][0]
        changed_root = self.root / "other-evaluation"
        frozen_pcm = changed_root / "01_input" / "pcm" / test_case.pcm.name
        frozen_label = changed_root / "01_input" / "labels" / test_case.label.name
        record["pcm"]["path"] = str(frozen_pcm)
        record["label"]["path"] = str(frozen_label)
        for kind in ("baseline", "streaming"):
            command = manifest["commands"][kind][test_case.file_id]
            self._replace_option(command, "--output-root", str(changed_root))
            self._replace_option(command, "--audio", str(frozen_pcm))

        with self.assertRaisesRegex(ValueError, "single consistent absolute --output-root"):
            module.validate_manifest(manifest)

    def test_frozen_file_verification_rejects_missing_and_hash_mismatches_without_unc_access(self):
        manifest = self._valid_manifest()
        self._materialize_frozen_inputs(manifest)
        self.assertIsNone(module.validate_manifest(manifest, verify_frozen_files=True))

        missing = self._valid_manifest()
        self._materialize_frozen_inputs(missing)
        Path(missing["test_cases"][0]["label"]["path"]).unlink()
        with self.assertRaisesRegex(FileNotFoundError, "frozen label file"):
            module.validate_manifest(missing, verify_frozen_files=True)

        mismatched = self._valid_manifest()
        self._materialize_frozen_inputs(mismatched)
        Path(mismatched["test_cases"][0]["pcm"]["path"]).write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "frozen PCM SHA-256"):
            module.validate_manifest(mismatched, verify_frozen_files=True)

    def test_validate_manifest_cli_verifies_frozen_file_hashes(self):
        manifest = self._valid_manifest()
        self._materialize_frozen_inputs(manifest)
        manifest_path = self.root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        self.assertEqual(module.main(["validate-manifest", "--manifest", str(manifest_path)]), 0)
        Path(manifest["test_cases"][0]["pcm"]["path"]).write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "frozen PCM SHA-256"):
            module.main(["validate-manifest", "--manifest", str(manifest_path)])

    def test_task_two_fixed_common_and_metric_argument_constants(self):
        self.assertEqual(
            module.COMMON_ARGUMENTS,
            (
                "--audio-format", "pcm", "--sample-rate", "16000", "--channels", "1",
                "--sample-width", "2", "--asr-engine", "paraformer", "--num-clusters", "-1",
                "--cluster-threshold", "0.6", "--segmentation-mode", "vad-pyannote",
            ),
        )
        self.assertEqual(
            module.METRIC_ARGUMENTS,
            ("--collar-ms", "500", "--boundary-tolerance-ms", "500"),
        )

    def test_fixed_test_cases_are_exact_user_confirmed_unc_paths(self):
        self.assertEqual(
            module.TEST_CASES,
            (
                module.TestCase(
                    file_id="23_asr_1782715267098",
                    pcm=Path(r"\\10.88.0.243\xainas\ProMax\测试集\音频测试集\中文\内部会议\双人安静咨询\promax\23_asr_1782715267098.pcm"),
                    label=Path(r"\\10.88.0.243\xainas\ProMax\测试集\音频测试集\中文\内部会议\双人安静咨询\promax\23_asr_1782715267098_label.txt"),
                ),
                module.TestCase(
                    file_id="asr_1788402212076",
                    pcm=Path(r"\\10.88.0.243\xainas\ProMax\测试集\音频测试集\中文\内部会议\信息流投放实习生面试\asr_1788402212076.pcm"),
                    label=Path(r"\\10.88.0.243\xainas\ProMax\测试集\音频测试集\中文\内部会议\信息流投放实习生面试\asr_1788402212076_label.txt"),
                ),
            ),
        )

    def test_prepare_inputs_hashes_and_replaces_changed_frozen_files(self):
        source_dir = self.root / "source"
        source_dir.mkdir()
        pcm = source_dir / "sample.pcm"
        label = source_dir / "sample_label.txt"
        pcm.write_bytes(b"first pcm")
        label.write_text("sample_0_1 (alice) 你好\n", encoding="utf-8")
        cases = (module.TestCase("sample", pcm, label),)

        first = module.prepare_inputs(self.root / "output", cases=cases)
        frozen_pcm = self.root / "output" / "01_input" / "pcm" / pcm.name
        self.assertEqual(frozen_pcm.read_bytes(), b"first pcm")
        self.assertEqual(first[0]["pcm"]["sha256"], module.sha256_file(pcm))

        pcm.write_bytes(b"second pcm")
        second = module.prepare_inputs(self.root / "output", cases=cases)
        self.assertEqual(frozen_pcm.read_bytes(), b"second pcm")
        self.assertEqual(second[0]["pcm"]["sha256"], module.sha256_file(pcm))
        self.assertFalse(list(frozen_pcm.parent.glob("*.tmp")))

    def test_parse_three_column_file_normalizes_speaker_and_rejects_missing_columns(self):
        records = self.root / "records.txt"
        records.write_text("sample_0_1000 (alice) 你好 世界\n", encoding="utf-8")
        self.assertEqual(
            module.parse_three_column_file(records),
            [module.TimedText("sample_0_1000", "alice", "你好 世界")],
        )
        records.write_text("missing-speaker\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "expected id speaker"):
            module.parse_three_column_file(records)

    def test_label_and_predictions_generate_zh_wer_counts(self):
        label = self.root / "sample_label.txt"
        label.write_text("sample_0_1000 (alice) 你好，世界！\n", encoding="utf-8")
        predicted = self.root / "sample_asr.txt"
        predicted.write_text("sample_0_1000 spk_0 你好 世界\n", encoding="utf-8")
        summary = module.compute_wer(label, predicted, language="zh")
        self.assertEqual(summary["reference_tokens"], 4)
        self.assertEqual(summary["errors"], 0)
        self.assertEqual(summary["insertions"], 0)
        self.assertEqual(summary["deletions"], 0)
        self.assertEqual(summary["substitutions"], 0)
        self.assertEqual(summary["wer"], 0.0)

    def test_wer_sorts_by_start_time_and_writes_requested_artifacts(self):
        label = self.root / "sample_label.txt"
        label.write_text(
            "sample_1000_1000 (alice) 世界\nsample_0_1000 (alice) 你好\n",
            encoding="utf-8",
        )
        predicted = self.root / "sample_asr.txt"
        predicted.write_text(
            "sample_1000_1000 spk_0 世界\nsample_0_1000 spk_0 你好\n",
            encoding="utf-8",
        )
        summary = module.compute_wer(
            label,
            predicted,
            language="zh",
            output_root=self.root / "output",
            scheme="baseline",
            file_id="sample",
        )
        metrics = self.root / "output" / "baseline" / "sample" / "metrics"
        self.assertEqual(summary["reference_text"], "你 好 世 界")
        self.assertTrue((metrics / "wer.json").is_file())
        self.assertIn("reference_tokens", (metrics / "wer_detail.md").read_text(encoding="utf-8"))
        self.assertEqual(json.loads((metrics / "wer.json").read_text(encoding="utf-8"))["errors"], 0)

    def test_wer_reports_insertions_deletions_and_substitutions(self):
        label = self.root / "sample_label.txt"
        label.write_text("sample_0_1000 alice 甲乙丙\n", encoding="utf-8")
        predicted = self.root / "sample_asr.txt"
        predicted.write_text("sample_0_1000 spk_0 甲丁\n", encoding="utf-8")
        summary = module.compute_wer(label, predicted, language="zh")
        self.assertEqual(summary["errors"], 2)
        self.assertEqual(summary["insertions"], 0)
        self.assertEqual(summary["deletions"], 1)
        self.assertEqual(summary["substitutions"], 1)
        self.assertEqual(summary["reference_tokens"], 3)

    def test_manifest_validates_fixed_cases_commands_models_and_metrics(self):
        manifest = self._valid_manifest()
        self.assertIsNone(module.validate_manifest(manifest))
        saved = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(saved, manifest)

    def test_manifest_rejects_nonabsolute_model_directory(self):
        manifest = self._valid_manifest()
        manifest["models"]["local"]["asr"]["directory"] = "relative/models/asr"
        with self.assertRaisesRegex(ValueError, "absolute"):
            module.validate_manifest(manifest)

    def test_manifest_rejects_same_role_model_hash_mismatch(self):
        manifest = self._valid_manifest()
        manifest["models"]["remote"]["vad"]["sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "VAD.*SHA-256"):
            module.validate_manifest(manifest)

    def test_manifest_rejects_nonfixed_case_ids_and_audio_or_metric_changes(self):
        manifest = self._valid_manifest()
        manifest["test_cases"][0]["file_id"] = "unexpected"
        with self.assertRaisesRegex(ValueError, "file IDs"):
            module.validate_manifest(manifest)

        manifest = self._valid_manifest()
        manifest["commands"]["streaming"][module.TEST_CASES[0].file_id][3] = "wrong.pcm"
        with self.assertRaisesRegex(ValueError, "audio"):
            module.validate_manifest(manifest)

        manifest = self._valid_manifest()
        manifest["metrics_parameters"]["collar_ms"] = 100
        with self.assertRaisesRegex(ValueError, "collar_ms"):
            module.validate_manifest(manifest)

    def test_task_two_cli_subcommands_prepare_validate_normalize_and_compute_wer(self):
        self.assertEqual(
            module.parse_args(["prepare", "--output-root", str(self.root / "output")]).command,
            "prepare",
        )
        self.assertEqual(
            module.parse_args(["validate-manifest", "--manifest", str(self.root / "manifest.json")]).command,
            "validate-manifest",
        )
        self.assertEqual(
            module.parse_args([
                "normalize-results", "--result-json", str(self.root / "result.json"),
                "--file-id", "sample", "--output-dir", str(self.root),
            ]).command,
            "normalize-results",
        )
        self.assertEqual(
            module.parse_args([
                "compute-wer", "--label", str(self.root / "label"),
                "--predicted", str(self.root / "prediction"), "--scheme", "baseline",
                "--file-id", "sample", "--output-root", str(self.root),
            ]).command,
            "compute-wer",
        )
        self.assertEqual(module.parse_args(["summarize"]).command, "summarize")


if __name__ == "__main__":
    unittest.main()
