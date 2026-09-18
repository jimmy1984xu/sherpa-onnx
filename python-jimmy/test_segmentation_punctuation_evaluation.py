import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from unittest import mock
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

    def test_build_invocation_preserves_raw_posix_path_strings(self):
        paths = module.ModelPaths(
            asr_dir="/opt/models/asr",
            vad_dir="/opt/models/vad",
            speaker_dir="/opt/models/speaker",
            segmentation_dir="/opt/models/segmentation",
        )
        common = module.build_common_arguments(paths, "/opt/evaluation")
        invocation = module.build_invocation("baseline", "/opt/input/audio.pcm", common)

        self.assertEqual(module.option_value(invocation, "--output-root"), "/opt/evaluation")
        self.assertEqual(module.option_value(invocation, "--audio"), "/opt/input/audio.pcm")
        self.assertEqual(module.option_value(invocation, "--asr-dir"), "/opt/models/asr")
        self.assertEqual(module.option_value(invocation, "--vad-dir"), "/opt/models/vad")
        self.assertEqual(module.option_value(invocation, "--speaker-dir"), "/opt/models/speaker")
        self.assertEqual(module.option_value(invocation, "--segmentation-dir"), "/opt/models/segmentation")

    def test_validate_variant_fairness_requires_explicit_remote_script_root(self):
        remote_root = "/srv/sherpa-onnx/python-jimmy"
        baseline = [
            self.baseline[0],
            f"{remote_root}/offline-long-audio-pipeline-asr-speaker.py",
            *self.baseline[2:],
        ]
        streaming = [
            self.streaming[0],
            f"{remote_root}/offline-long-audio-pipeline-asr-speaker-segmentation.py",
            *self.streaming[2:],
        ]
        with self.assertRaisesRegex(ValueError, "entry script"):
            module.validate_variant_fairness(baseline, streaming)
        self.assertIsNone(
            module.validate_variant_fairness(
                baseline, streaming, expected_script_root=remote_root
            )
        )

    def test_validate_variant_fairness_rejects_untrusted_same_basename_paths(self):
        trusted_root = "/srv/sherpa-onnx/python-jimmy"
        untrusted_root = "/untrusted/python-jimmy"
        baseline = [
            self.baseline[0],
            f"{untrusted_root}/offline-long-audio-pipeline-asr-speaker.py",
            *self.baseline[2:],
        ]
        streaming = [
            self.streaming[0],
            f"{untrusted_root}/offline-long-audio-pipeline-asr-speaker-segmentation.py",
            *self.streaming[2:],
        ]
        with self.assertRaisesRegex(ValueError, "entry script"):
            module.validate_variant_fairness(
                baseline, streaming, expected_script_root=trusted_root
            )

    def test_validate_variant_fairness_rejects_wrong_remote_entrypoint_basenames(self):
        trusted_root = "/srv/sherpa-onnx/python-jimmy"
        baseline = [self.baseline[0], f"{trusted_root}/wrong-baseline.py", *self.baseline[2:]]
        streaming = [
            self.streaming[0],
            f"{trusted_root}/wrong-streaming.py",
            *self.streaming[2:],
        ]
        with self.assertRaisesRegex(ValueError, "entry script"):
            module.validate_variant_fairness(
                baseline, streaming, expected_script_root=trusted_root
            )

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
        local_models = self._model_inventory("C:/models", "c" * 64)
        remote_models = self._model_inventory("/models", "c" * 64)
        # The manifest stores Windows/local canonical commands. Task 4 will
        # render and record remote argv separately from this template.
        paths = module.ModelPaths(
            asr_dir=local_models["asr"]["directory"],
            vad_dir=local_models["vad"]["directory"],
            speaker_dir=local_models["speaker"]["directory"],
            segmentation_dir=local_models["segmentation"]["directory"],
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
            local_models=local_models,
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

    def test_manifest_commands_bind_each_model_dir_to_local_inventory(self):
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
                with self.assertRaisesRegex(ValueError, f"local {role} model directory"):
                    module.validate_manifest(manifest)

    def test_preflight_pending_allows_all_remote_hashes_to_be_pending(self):
        manifest = self._valid_manifest()
        manifest["status"] = "preflight-pending"
        for role in module.MODEL_ROLES:
            manifest["models"]["remote"][role]["sha256"] = "pending"
        self.assertIsNone(module.validate_manifest(manifest))

    def test_manifest_rejects_pending_remote_hashes_outside_preflight_pending(self):
        manifest = self._valid_manifest()
        for role in module.MODEL_ROLES:
            manifest["models"]["remote"][role]["sha256"] = "pending"
        with self.assertRaisesRegex(ValueError, "pending.*preflight-pending"):
            module.validate_manifest(manifest)

    def test_preflight_pending_rejects_a_real_remote_hash_that_mismatches_local(self):
        manifest = self._valid_manifest()
        manifest["status"] = "preflight-pending"
        for role in module.MODEL_ROLES:
            manifest["models"]["remote"][role]["sha256"] = "pending"
        manifest["models"]["remote"]["asr"]["sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "ASR.*SHA-256"):
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

    def test_build_model_inventory_hashes_all_string_model_directories(self):
        directories = {}
        for role in module.MODEL_ROLES:
            directory = self.root / "models" / role
            directory.mkdir(parents=True)
            (directory / "model.bin").write_bytes(f"{role}-model".encode())
            directories[role] = directory

        inventory = module.build_model_inventory(
            module.ModelPaths(
                asr_dir=str(directories["asr"]),
                vad_dir=str(directories["vad"]),
                speaker_dir=str(directories["speaker"]),
                segmentation_dir=str(directories["segmentation"]),
            )
        )

        for role, directory in directories.items():
            self.assertEqual(inventory[role]["directory"], str(directory))
            self.assertEqual(inventory[role]["sha256"], module.sha256_path(directory))

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

class TaskThreeTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    @staticmethod
    def _write_boundary_csv(path, rows):
        import csv
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "文件ID", "切换序号", "参考前说话人", "参考后说话人",
            "区间起始毫秒", "区间结束毫秒", "扩展后起始毫秒", "扩展后结束毫秒",
            "匹配预测边界毫秒", "状态",
        ]
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _metrics_summary(
        *, der, hit_rate, evaluated=2, errors=0, speaker_change_count=2, speaker_change_hits=None,
    ):
        if speaker_change_hits is None:
            speaker_change_hits = speaker_change_count
        return {
            "counts": {"asr_files": evaluated + errors, "evaluated": evaluated, "errors": errors},
            "metrics": {
                "der": der,
                "speaker_change_hit_rate": hit_rate,
                "speaker_change_hits": speaker_change_hits,
                "speaker_change_count": speaker_change_count,
            },
        }

    def _write_fake_metrics_source(self, *, exit_code=0):
        source = self.root / "speaker_diarization_metrics.py"
        source.write_text(
            "import json, sys\n"
            "from pathlib import Path\n"
            "out = Path(sys.argv[sys.argv.index('--output-dir') + 1])\n"
            "out.mkdir(parents=True, exist_ok=True)\n"
            "(out / 'received_args.json').write_text(json.dumps(sys.argv[1:]), encoding='utf-8')\n"
            "(out / 'speaker_diarization_summary.json').write_text(json.dumps({"
            "'counts': {'asr_files': 2, 'evaluated': 2, 'errors': 0}, "
            "'metrics': {'der': 0.2, 'speaker_change_hit_rate': 0.7, "
            "'speaker_change_hits': 7, 'speaker_change_count': 10}}), encoding='utf-8')\n"
            f"print('fake metrics exit {exit_code}')\n"
            f"raise SystemExit({exit_code})\n",
            encoding="utf-8",
        )
        return source

    def test_winner_prefers_boundary_hit_rate_then_lower_der(self):
        winner = module.select_winner({
            "baseline": {"hit_rate": 0.80, "der": 0.10},
            "streaming": {"hit_rate": 0.80, "der": 0.08},
        })
        self.assertEqual(winner, "streaming")
        self.assertEqual(
            module.select_winner({
                "baseline": {"hit_rate": 0.81, "der": 0.10},
                "streaming": {"hit_rate": 0.80, "der": 0.01},
            }),
            "baseline",
        )
        self.assertIsNone(module.select_winner({
            "baseline": {"hit_rate": 0.80, "der": 0.10},
            "streaming": {"hit_rate": 0.805, "der": 0.10},
        }))

    def test_boundary_difference_keeps_text_and_both_matched_times(self):
        row = module.make_boundary_difference(
            reference_boundary_ms=5000,
            baseline_match_ms=None,
            streaming_match_ms=5100,
            tolerance_ms=500,
            before_text="甲",
            after_text="乙",
        )
        self.assertEqual(row["classification"], "streaming_only_hit")
        self.assertEqual(row["after_text"], "乙")
        self.assertEqual(row["reference_boundary_ms"], 5000)
        self.assertIsNone(row["baseline_match_ms"])
        self.assertEqual(row["streaming_match_ms"], 5100)

    def test_der_summary_and_per_file_values_must_be_finite_and_nonnegative(self):
        cases = {
            "negative summary": lambda results: results["baseline"]["summary"]["metrics"].update({"der": -0.1}),
            "negative per-file": lambda results: results["baseline"]["per_file"]["files"][0]["metrics"].update({"der": -0.1}),
            "non-finite per-file": lambda results: results["baseline"]["per_file"]["files"][0]["metrics"].update({"der": float("nan")}),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name):
                labels, _, _, scheme_results = self._write_complete_boundary_evidence()
                mutate(scheme_results)
                comparison = module.write_comparison(self.root, scheme_results, labels)
                self.assertFalse(comparison["ranking_eligible"])
                self.assertIsNone(comparison["winner"])
                self.assertIn("DER must be a finite numeric value and be nonnegative", comparison["ranking_reason"])

    def test_der_components_must_be_nonnegative_and_match_der_values(self):
        cases = {
            "negative aggregate component": lambda results: results["baseline"]["der_components"].update({"miss": -0.1}),
            "negative per-file component": lambda results: results["baseline"]["der_components_by_file"][module.TEST_CASES[0].file_id].update({"confusion": -0.1}),
            "aggregate formula mismatch": lambda results: results["baseline"]["der_components"].update({"false_alarm": 2.0}),
            "per-file formula mismatch": lambda results: results["baseline"]["der_components_by_file"][module.TEST_CASES[0].file_id].update({"false_alarm": 0.75}),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name):
                labels, _, _, scheme_results = self._write_complete_boundary_evidence()
                scheme_results["baseline"]["summary"] = self._metrics_summary(
                    der=0.1, hit_rate=0.0, speaker_change_count=2, speaker_change_hits=0,
                )
                scheme_results["baseline"]["per_file"] = self._completed_per_file_metrics(
                    first_der=0.1, second_der=0.1,
                )
                scheme_results["baseline"]["der_components"] = {
                    "miss": 1.0, "false_alarm": 1.0, "confusion": 0.0, "total": 20.0,
                }
                scheme_results["baseline"]["der_components_by_file"] = {
                    case.file_id: {"miss": 0.5, "false_alarm": 0.5, "confusion": 0.0, "total": 10.0}
                    for case in module.TEST_CASES
                }
                mutate(scheme_results)
                comparison = module.write_comparison(self.root, scheme_results, labels)
                self.assertFalse(comparison["ranking_eligible"])
                self.assertIsNone(comparison["winner"])
                self.assertIn("DER component", comparison["ranking_reason"])

    def test_install_metrics_tool_copy_failure_leaves_no_receipt_or_partial_destination(self):
        source = self._write_fake_metrics_source()
        tools_dir = self.root / "tools"
        with mock.patch.object(module.shutil, "copyfile", side_effect=OSError("copy denied")):
            with self.assertRaisesRegex(OSError, "copy denied"):
                module.install_metrics_tool(source, tools_dir)
        self.assertFalse((tools_dir / module.METRICS_TOOL_FILENAME).exists())
        self.assertFalse((tools_dir / "speaker_diarization_metrics.sha256.json").exists())
        self.assertEqual(list(tools_dir.glob(".*.tmp")), [])

    def test_summarize_missing_metrics_source_records_shared_absolute_install_error_log(self):
        manifest_builder = TaskTwoTest()
        manifest_builder.root = self.root
        manifest = manifest_builder._valid_manifest()
        manifest_builder._materialize_frozen_inputs(manifest)
        module._write_json(self.root / "manifest.json", manifest)
        missing_source = self.root / module.METRICS_TOOL_FILENAME

        comparison = module.summarize_evaluation(self.root, metrics_source=missing_source)

        install_log = (self.root / "tools" / "metrics_tool_install.error.log").resolve()
        self.assertFalse(comparison["ranking_eligible"])
        self.assertTrue((self.root / "05_comparison" / "comparison.json").is_file())
        self.assertTrue((self.root / "05_comparison" / "boundary_differences.csv").is_file())
        self.assertTrue((self.root / "报告.md").is_file())
        self.assertTrue(install_log.is_file())
        log_text = install_log.read_text(encoding="utf-8")
        self.assertIn("FileNotFoundError", log_text)
        self.assertIn("metrics source is missing", log_text)
        for scheme in module.SCHEMES:
            result = comparison["schemes"][scheme]
            self.assertEqual(result["status"], "metrics_failed")
            self.assertIn("metrics source is missing", result["error"])
            self.assertEqual(result["install_error_log"], str(install_log))
        failure_text = "\n".join(comparison["failures"])
        self.assertIn("metrics source is missing", failure_text)
        self.assertIn(str(install_log), failure_text)
        report_text = (self.root / "报告.md").read_text(encoding="utf-8")
        self.assertIn("metrics source is missing", report_text)
        self.assertIn(str(install_log), report_text)

    def test_install_metrics_tool_records_hash_before_copy_and_refuses_wrong_name(self):
        source = self._write_fake_metrics_source()
        tools_dir = self.root / "tools"
        installed = module.install_metrics_tool(source, tools_dir)
        self.assertEqual(installed, tools_dir / "speaker_diarization_metrics.py")
        self.assertEqual(installed.read_text(encoding="utf-8"), source.read_text(encoding="utf-8"))
        receipt = json.loads((tools_dir / "speaker_diarization_metrics.sha256.json").read_text(encoding="utf-8"))
        self.assertEqual(receipt["sha256"], module.sha256_file(source))
        wrong = self.root / "wrong.py"
        wrong.write_text("pass\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "speaker_diarization_metrics.py"):
            module.install_metrics_tool(wrong, tools_dir)

    def test_run_speaker_metrics_executes_copied_tool_with_fixed_arguments_and_logs(self):
        source = self._write_fake_metrics_source()
        tools_dir = self.root / "tools"
        module.install_metrics_tool(source, tools_dir)
        normalized = self.root / "normalized"
        labels = self.root / "labels"
        normalized.mkdir()
        labels.mkdir()
        metrics_dir = self.root / "metrics"
        result = module.run_speaker_metrics("baseline", tools_dir, normalized, labels, metrics_dir)
        self.assertEqual(result["status"], "metrics_completed")
        self.assertEqual(result["summary"]["metrics"]["der"], 0.2)
        self.assertTrue((metrics_dir / "speaker_metrics.stdout.log").is_file())
        self.assertTrue((metrics_dir / "speaker_metrics.stderr.log").is_file())
        received = json.loads((metrics_dir / "received_args.json").read_text(encoding="utf-8"))
        self.assertEqual(received[received.index("--boundary-tolerance-ms") + 1], "500")
        self.assertEqual(received[received.index("--collar-ms") + 1], "500")

    def test_run_speaker_metrics_failure_preserves_logs_and_is_not_rankable(self):
        source = self._write_fake_metrics_source(exit_code=3)
        tools_dir = self.root / "tools"
        module.install_metrics_tool(source, tools_dir)
        normalized = self.root / "normalized"
        labels = self.root / "labels"
        normalized.mkdir()
        labels.mkdir()
        metrics_dir = self.root / "metrics"
        result = module.run_speaker_metrics("streaming", tools_dir, normalized, labels, metrics_dir)
        self.assertEqual(result["status"], "metrics_failed")
        self.assertEqual(result["returncode"], 3)
        self.assertIn("fake metrics exit 3", (metrics_dir / "speaker_metrics.stdout.log").read_text(encoding="utf-8"))
        self.assertIsNone(module.select_winner({
            "baseline": {"hit_rate": 0.8, "der": 0.1},
            "streaming": {"hit_rate": None, "der": None},
        }))

    def test_comparison_merges_boundaries_and_chinese_report_uses_valid_ranking(self):
        labels = self.root / "01_input" / "labels"
        first_case, second_case = module.TEST_CASES
        labels.mkdir(parents=True)
        for case in (first_case, second_case):
            (labels / case.label.name).write_text(
                f"{case.file_id}_0_5000 A 甲\n{case.file_id}_5000_5000 B 乙\n",
                encoding="utf-8",
            )
        metrics_root = self.root / "03_metrics"
        baseline_csv = metrics_root / "baseline" / "speaker_diarization_boundary_details.csv"
        streaming_csv = metrics_root / "streaming" / "speaker_diarization_boundary_details.csv"
        rows = []
        for case in (first_case, second_case):
            rows.append({
                "文件ID": case.file_id, "切换序号": 1, "参考前说话人": "A", "参考后说话人": "B",
                "区间起始毫秒": 5000, "区间结束毫秒": 5000,
                "扩展后起始毫秒": 4500, "扩展后结束毫秒": 5500,
                "匹配预测边界毫秒": "", "状态": "未命中",
            })
        self._write_boundary_csv(baseline_csv, rows)
        streaming_rows = [dict(row) for row in rows]
        streaming_rows[0]["匹配预测边界毫秒"] = 5100
        streaming_rows[0]["状态"] = "命中"
        self._write_boundary_csv(streaming_csv, streaming_rows)
        scheme_results = {
            "baseline": {
                "status": "metrics_completed", "summary": self._metrics_summary(der=0.10, hit_rate=0.0, speaker_change_hits=0),
                "per_file": self._completed_per_file_metrics(),
                "metrics_dir": str(baseline_csv.parent), "boundary_details_path": str(baseline_csv),
                "der_components": {"miss": 1.0, "false_alarm": 1.0, "confusion": 0.0, "total": 20.0},
            },
            "streaming": {
                "status": "metrics_completed", "summary": self._metrics_summary(der=0.08, hit_rate=0.5, speaker_change_hits=1),
                "per_file": self._completed_per_file_metrics(),
                "metrics_dir": str(streaming_csv.parent), "boundary_details_path": str(streaming_csv),
                "der_components": {"miss": 1.0, "false_alarm": 0.5, "confusion": 0.1, "total": 20.0},
            },
        }
        for scheme in ("baseline", "streaming"):
            for case in (first_case, second_case):
                wer_dir = self.root / scheme / case.file_id / "metrics"
                wer_dir.mkdir(parents=True, exist_ok=True)
                (wer_dir / "wer.json").write_text(json.dumps({"wer": 0.0}), encoding="utf-8")
        runtime = self.root / "remote-artifacts" / "raw" / "baseline" / first_case.file_id / "run_metadata.json"
        runtime.parent.mkdir(parents=True)
        runtime.write_text(json.dumps({"argv": ["python3", "baseline-real.py"], "rtf": 0.5}), encoding="utf-8")
        comparison = module.write_comparison(self.root, scheme_results, labels)
        boundary_rows = comparison["boundary_differences"]
        self.assertEqual(boundary_rows[0]["classification"], "streaming_only_hit")
        self.assertEqual(boundary_rows[0]["before_text"], "甲")
        self.assertEqual(comparison["winner"], "streaming")
        self.assertTrue((self.root / "05_comparison" / "comparison.json").is_file())
        self.assertTrue((self.root / "05_comparison" / "boundary_differences.csv").is_file())
        manifest = {
            "commands": {"baseline": {first_case.file_id: ["python", "baseline.py"]}, "streaming": {first_case.file_id: ["python", "streaming.py"]}},
            "models": {"local": {"asr": {"directory": "C:/models/asr", "sha256": "a" * 64}}},
        }
        report = module.write_report(self.root, manifest, comparison)
        report_text = report.read_text(encoding="utf-8")
        self.assertIn("本次数据集排名", report_text)
        self.assertIn("streaming", report_text)
        self.assertIn("C:/models/asr", report_text)
        self.assertIn("baseline-real.py", report_text)
        self.assertIn("false_alarm", report_text)

    def _write_completed_wer_artifacts(self):
        for scheme in module.SCHEMES:
            for case in module.TEST_CASES:
                wer_dir = self.root / scheme / case.file_id / "metrics"
                wer_dir.mkdir(parents=True, exist_ok=True)
                (wer_dir / "wer.json").write_text(json.dumps({"wer": 0.0}), encoding="utf-8")

    def _write_empty_boundary_details(self, scheme):
        path = self.root / "03_metrics" / scheme / "speaker_diarization_boundary_details.csv"
        self._write_boundary_csv(path, [])
        return path

    @staticmethod
    def _completed_per_file_metrics(*, first_der=0.11, first_hit_rate=0.21, second_der=0.12, second_hit_rate=0.22, first_count=1, second_count=1):
        first_case, second_case = module.TEST_CASES
        return {
            "files": [
                {"file_id": first_case.file_id, "error": None, "metrics": {"der": first_der, "speaker_change_hit_rate": first_hit_rate, "speaker_change_count": first_count}},
                {"file_id": second_case.file_id, "error": None, "metrics": {"der": second_der, "speaker_change_hit_rate": second_hit_rate, "speaker_change_count": second_count}},
            ]
        }

    def _write_complete_boundary_evidence(self):
        labels = self.root / "01_input" / "labels"
        labels.mkdir(parents=True, exist_ok=True)
        rows = []
        for case in module.TEST_CASES:
            (labels / case.label.name).write_text(
                f"{case.file_id}_0_5000 A 甲\n{case.file_id}_5000_5000 B 乙\n",
                encoding="utf-8",
            )
            rows.append({
                "文件ID": case.file_id, "切换序号": 1, "参考前说话人": "A", "参考后说话人": "B",
                "区间起始毫秒": 5000, "区间结束毫秒": 5000,
                "扩展后起始毫秒": 4500, "扩展后结束毫秒": 5500,
                "匹配预测边界毫秒": "", "状态": "未命中",
            })
        paths = {}
        for scheme in module.SCHEMES:
            path = self.root / "03_metrics" / scheme / "speaker_diarization_boundary_details.csv"
            self._write_boundary_csv(path, rows)
            paths[scheme] = path
        self._write_completed_wer_artifacts()
        scheme_results = {
            scheme: {
                "status": "metrics_completed",
                "summary": self._metrics_summary(
                    der=0.1, hit_rate=0.0, speaker_change_count=len(rows), speaker_change_hits=0,
                ),
                "per_file": self._completed_per_file_metrics(),
                "boundary_details_path": str(paths[scheme]),
            }
            for scheme in module.SCHEMES
        }
        return labels, paths, rows, scheme_results

    def _assert_boundary_evidence_failure(self, comparison):
        self.assertFalse(comparison["ranking_eligible"])
        self.assertFalse(comparison["boundary_differences_available"])
        self.assertEqual(comparison["boundary_differences"], [])
        self.assertTrue(comparison["boundary_differences_error"])
        self.assertIsNone(comparison["winner"])

    def test_invalid_wer_values_are_invalid_and_prevent_ranking(self):
        first_case = module.TEST_CASES[0]
        invalid_payloads = {
            "missing": {},
            "non_numeric": {"wer": "0.1"},
            "nan": {"wer": float("nan")},
            "infinity": {"wer": float("inf")},
            "bool": {"wer": True},
        }
        for name, payload in invalid_payloads.items():
            with self.subTest(name=name):
                labels, _, _, scheme_results = self._write_complete_boundary_evidence()
                wer_path = self.root / "baseline" / first_case.file_id / "metrics" / "wer.json"
                wer_path.write_text(json.dumps(payload), encoding="utf-8")
                comparison = module.write_comparison(self.root, scheme_results, labels)
                wer = comparison["schemes"]["baseline"]["wer_by_file"][first_case.file_id]
                self.assertEqual(wer["status"], "invalid")
                self.assertFalse(comparison["ranking_eligible"])
                self.assertIn("WER is unavailable", comparison["ranking_reason"])
                report = module.write_report(self.root, {"commands": {}, "models": {}}, comparison)
                report_text = report.read_text(encoding="utf-8")
                self.assertIn("未形成有效总体排名", report_text)
                self.assertNotIn("all fixed audio cases have speaker metrics and WER", report_text)

    def test_malformed_boundary_csv_is_recorded_without_crashing_summarize(self):
        malformed = ("missing column", "truncated row", "invalid integer", "duplicate reference boundary")
        for kind in malformed:
            with self.subTest(kind=kind):
                labels, paths, rows, scheme_results = self._write_complete_boundary_evidence()
                if kind == "missing column":
                    paths["streaming"].write_text("文件ID,状态\nexample,命中\n", encoding="utf-8-sig")
                elif kind == "truncated row":
                    paths["streaming"].write_text(
                        "文件ID,切换序号,参考前说话人,参考后说话人,区间起始毫秒,区间结束毫秒,扩展后起始毫秒,扩展后结束毫秒,匹配预测边界毫秒,状态\n"
                        f"{module.TEST_CASES[0].file_id},1,A,B,5000\n",
                        encoding="utf-8-sig",
                    )
                elif kind == "invalid integer":
                    bad_rows = [dict(row) for row in rows]
                    bad_rows[0]["区间起始毫秒"] = "not-an-integer"
                    self._write_boundary_csv(paths["streaming"], bad_rows)
                else:
                    self._write_boundary_csv(paths["streaming"], rows + [dict(rows[0])])
                comparison = module.write_comparison(self.root, scheme_results, labels)
                self._assert_boundary_evidence_failure(comparison)
                self.assertIn("boundary detail CSV is invalid", comparison["boundary_differences_error"])
                self.assertFalse(comparison["schemes"]["streaming"]["boundary_details_validation"]["available"])
                comparison_path = self.root / "05_comparison" / "comparison.json"
                self.assertTrue(comparison_path.is_file())
                persisted = json.loads(comparison_path.read_text(encoding="utf-8"))
                self.assertFalse(persisted["schemes"]["streaming"]["boundary_details_validation"]["available"])
                report = module.write_report(self.root, {"commands": {}, "models": {}}, comparison)
                self.assertTrue(report.is_file())
                report_text = report.read_text(encoding="utf-8")
                self.assertIn("边界差异（不可用）", report_text)
                self.assertIn("未形成有效总体排名", report_text)

    def test_incomplete_or_inconsistent_boundary_evidence_prevents_ranking(self):
        cases = ("empty", "missing row", "unknown file", "different keys", "hit without match", "miss with match", "count mismatch")
        for kind in cases:
            with self.subTest(kind=kind):
                labels, paths, rows, scheme_results = self._write_complete_boundary_evidence()
                if kind == "empty":
                    self._write_boundary_csv(paths["baseline"], [])
                elif kind == "missing row":
                    self._write_boundary_csv(paths["baseline"], rows[:1])
                elif kind == "unknown file":
                    bad_rows = [dict(row) for row in rows]
                    bad_rows[0]["文件ID"] = "unknown-file"
                    self._write_boundary_csv(paths["baseline"], bad_rows)
                elif kind == "different keys":
                    bad_rows = [dict(row) for row in rows]
                    bad_rows[0]["区间起始毫秒"] = 4999
                    bad_rows[0]["区间结束毫秒"] = 4999
                    self._write_boundary_csv(paths["streaming"], bad_rows)
                elif kind == "hit without match":
                    bad_rows = [dict(row) for row in rows]
                    bad_rows[0]["状态"] = "命中"
                    self._write_boundary_csv(paths["baseline"], bad_rows)
                elif kind == "miss with match":
                    bad_rows = [dict(row) for row in rows]
                    bad_rows[0]["匹配预测边界毫秒"] = 5000
                    self._write_boundary_csv(paths["baseline"], bad_rows)
                else:
                    scheme_results["baseline"]["summary"] = self._metrics_summary(der=0.1, hit_rate=0.8, speaker_change_count=3)
                comparison = module.write_comparison(self.root, scheme_results, labels)
                self._assert_boundary_evidence_failure(comparison)
                self.assertIn("evidence is incomplete", comparison["boundary_differences_error"])

    def test_report_uses_per_file_der_and_hit_rate_and_keeps_aggregate_separate(self):
        first_case, second_case = module.TEST_CASES
        per_file = self._completed_per_file_metrics()
        comparison = {
            "ranking_eligible": False,
            "ranking_reason": "test-only",
            "boundary_differences_available": True,
            "boundary_differences": [],
            "failures": [],
            "schemes": {
                "baseline": {
                    "summary": self._metrics_summary(der=0.90, hit_rate=0.80),
                    "per_file": per_file,
                    "wer_by_file": {
                        first_case.file_id: {"wer": 0.01},
                        second_case.file_id: {"wer": 0.02},
                    },
                    "der_components_by_file": {
                        first_case.file_id: {"miss": 1.0, "false_alarm": 2.0, "confusion": 3.0},
                        second_case.file_id: {"miss": 4.0, "false_alarm": 5.0, "confusion": 6.0},
                    },
                },
                "streaming": {},
            },
        }
        report = module.write_report(self.root, {"commands": {}, "models": {}}, comparison)
        text = report.read_text(encoding="utf-8")
        self.assertIn("## 方案汇总指标", text)
        self.assertIn("| baseline | 0.9000 | 0.8000 |", text)
        self.assertIn(f"| baseline | {first_case.file_id} | 0.0100 | 0.1100 |", text)
        self.assertIn(f"| baseline | {second_case.file_id} | 0.0200 | 0.1200 |", text)
        self.assertNotIn(f"| baseline | {first_case.file_id} | 0.0100 | 0.9000 |", text)

    def test_non_finite_summary_metrics_prevent_ranking_and_no_difference_conclusion(self):
        self._write_completed_wer_artifacts()
        boundary_paths = {scheme: self._write_empty_boundary_details(scheme) for scheme in module.SCHEMES}
        scheme_results = {
            scheme: {
                "status": "metrics_completed",
                "summary": self._metrics_summary(der=float("nan"), hit_rate=0.8),
                "per_file": self._completed_per_file_metrics(),
                "boundary_details_path": str(boundary_paths[scheme]),
            }
            for scheme in module.SCHEMES
        }
        comparison = module.write_comparison(self.root, scheme_results, self.root / "01_input" / "labels")
        self.assertFalse(comparison["ranking_eligible"])
        self.assertIsNone(comparison["winner"])
        self.assertIn("DER must be a finite numeric value", comparison["ranking_reason"])
        report = module.write_report(self.root, {"commands": {}, "models": {}}, comparison)
        report_text = report.read_text(encoding="utf-8")
        self.assertIn("未形成有效总体排名", report_text)
        self.assertNotIn("无明显差异", report_text)

    def test_missing_boundary_details_is_a_failure_and_prevents_ranking(self):
        self._write_completed_wer_artifacts()
        baseline_path = self._write_empty_boundary_details("baseline")
        missing_streaming_path = self.root / "03_metrics" / "streaming" / "speaker_diarization_boundary_details.csv"
        scheme_results = {
            "baseline": {
                "status": "metrics_completed",
                "summary": self._metrics_summary(der=0.1, hit_rate=0.8),
                "per_file": self._completed_per_file_metrics(),
                "boundary_details_path": str(baseline_path),
            },
            "streaming": {
                "status": "metrics_completed",
                "summary": self._metrics_summary(der=0.1, hit_rate=0.8),
                "per_file": self._completed_per_file_metrics(),
                "boundary_details_path": str(missing_streaming_path),
            },
        }
        comparison = module.write_comparison(self.root, scheme_results, self.root / "01_input" / "labels")
        self.assertFalse(comparison["ranking_eligible"])
        self.assertFalse(comparison["boundary_differences_available"])
        self.assertTrue(any("boundary detail CSV is missing" in item for item in comparison["failures"]))
        report = module.write_report(self.root, {"commands": {}, "models": {}}, comparison)
        report_text = report.read_text(encoding="utf-8")
        self.assertIn("边界差异（不可用）", report_text)
        self.assertIn("未形成有效总体排名", report_text)

    def test_comparison_refuses_ranking_when_metrics_evaluate_wrong_file_ids(self):
        labels = self.root / "01_input" / "labels"
        labels.mkdir(parents=True)
        for scheme in module.SCHEMES:
            for case in module.TEST_CASES:
                wer_dir = self.root / scheme / case.file_id / "metrics"
                wer_dir.mkdir(parents=True, exist_ok=True)
                (wer_dir / "wer.json").write_text(json.dumps({"wer": 0.0}), encoding="utf-8")
        wrong_per_file = {
            "files": [
                {"file_id": "unexpected_a", "error": None},
                {"file_id": "unexpected_b", "error": None},
            ]
        }
        boundary_paths = {scheme: self._write_empty_boundary_details(scheme) for scheme in module.SCHEMES}
        comparison = module.write_comparison(
            self.root,
            {
                scheme: {
                    "status": "metrics_completed",
                    "summary": self._metrics_summary(der=0.1, hit_rate=0.8),
                    "per_file": wrong_per_file,
                    "boundary_details_path": str(boundary_paths[scheme]),
                }
                for scheme in module.SCHEMES
            },
            labels,
        )
        self.assertFalse(comparison["ranking_eligible"])
        self.assertIsNone(comparison["winner"])
        self.assertIn("fixed audio file IDs", comparison["ranking_reason"])

    def test_report_does_not_rank_when_any_case_is_not_successfully_evaluated(self):
        comparison = {
            "winner": "streaming", "ranking_eligible": False, "ranking_reason": "缺少完整指标",
            "schemes": {}, "boundary_differences": [], "failures": ["streaming metrics_failed"],
        }
        report = module.write_report(self.root, {"commands": {}, "models": {}}, comparison)
        text = report.read_text(encoding="utf-8")
        self.assertIn("未形成有效总体排名", text)
        self.assertNotIn("本次数据集排名：streaming", text)

    def test_copied_metrics_tool_exposes_per_file_der_components_after_cli_run(self):
        source = self.root / "speaker_diarization_metrics.py"
        source.write_text(
            "import json, sys\n"
            "from dataclasses import dataclass\n"
            "from pathlib import Path\n"
            "@dataclass\nclass Report:\n    files: list\n"
            "def evaluate_paths(**kwargs):\n"
            "    return Report([{'error': None, 'file_id': 'sample', 'der_components': {'miss': 1.0, 'false_alarm': 2.0, 'confusion': 3.0, 'total': 10.0}, 'metrics': {'der': 0.6}}])\n"
            "if __name__ == '__main__':\n"
            "    out = Path(sys.argv[sys.argv.index('--output-dir') + 1]); out.mkdir(parents=True, exist_ok=True)\n"
            "    (out / 'speaker_diarization_summary.json').write_text(json.dumps({'counts': {'asr_files': 1, 'evaluated': 1, 'errors': 0}, 'metrics': {'der': 0.6, 'speaker_change_hit_rate': 0.5, 'speaker_change_hits': 1, 'speaker_change_count': 2}}), encoding='utf-8')\n",
            encoding="utf-8",
        )
        tools_dir = self.root / "tools"
        module.install_metrics_tool(source, tools_dir)
        normalized = self.root / "normalized"
        labels = self.root / "labels"
        normalized.mkdir()
        labels.mkdir()
        result = module.run_speaker_metrics("baseline", tools_dir, normalized, labels, self.root / "metrics")
        self.assertEqual(result["der_components"], {"miss": 1.0, "false_alarm": 2.0, "confusion": 3.0, "total": 10.0})
        self.assertEqual(result["der_components_by_file"]["sample"]["confusion"], 3.0)

    def test_boundary_text_uses_last_text_of_merged_same_speaker_turn(self):
        case = module.TEST_CASES[0]
        labels = self.root / "labels"
        labels.mkdir()
        (labels / case.label.name).write_text(
            f"{case.file_id}_0_1000 A 第一段\n{case.file_id}_1000_1000 A 第二段\n{case.file_id}_2000_1000 B 后续\n",
            encoding="utf-8",
        )
        row = {
            "文件ID": case.file_id, "切换序号": 1, "参考前说话人": "A", "参考后说话人": "B",
            "区间起始毫秒": 2000, "区间结束毫秒": 2000,
            "扩展后起始毫秒": 1500, "扩展后结束毫秒": 2500,
            "匹配预测边界毫秒": "2000", "状态": "命中",
        }
        baseline = self.root / "baseline.csv"
        streaming = self.root / "streaming.csv"
        self._write_boundary_csv(baseline, [row])
        self._write_boundary_csv(streaming, [row])
        differences = module.merge_boundary_differences(baseline, streaming, labels)
        self.assertEqual(differences[0]["before_text"], "第二段")
        self.assertEqual(differences[0]["after_text"], "后续")

    def test_summarize_cli_accepts_task_three_artifact_locations(self):
        args = module.parse_args([
            "summarize", "--output-root", str(self.root),
            "--manifest", str(self.root / "manifest.json"),
            "--metrics-source", str(self.root / "speaker_diarization_metrics.py"),
            "--normalized-results-root", str(self.root / "normalized"),
            "--labels-dir", str(self.root / "labels"),
        ])
        self.assertEqual(args.command, "summarize")
        self.assertEqual(args.output_root, self.root)


    def test_boundary_evidence_enforces_expansion_hit_count_and_rate(self):
        cases = {
            "expanded interval": ("expanded interval", lambda rows, metrics: rows[0].update({"扩展后起始毫秒": 4501})),
            "matched prediction": (
                "matched prediction boundary",
                lambda rows, metrics: (
                    rows[0].update({"匹配预测边界毫秒": 5501, "状态": "命中"}),
                    metrics.update({"speaker_change_hits": 1, "speaker_change_hit_rate": 0.5}),
                ),
            ),
            "hit count": (
                "speaker_change_hits",
                lambda rows, metrics: rows[0].update({"匹配预测边界毫秒": 5000, "状态": "命中"}),
            ),
            "hit rate": (
                "speaker_change_hit_rate",
                lambda rows, metrics: (
                    rows[0].update({"匹配预测边界毫秒": 5000, "状态": "命中"}),
                    metrics.update({"speaker_change_hits": 1, "speaker_change_hit_rate": 0.25}),
                ),
            ),
        }
        for name, (expected_reason, mutate) in cases.items():
            with self.subTest(name=name):
                labels, paths, rows, scheme_results = self._write_complete_boundary_evidence()
                bad_rows = [dict(row) for row in rows]
                metrics = scheme_results["baseline"]["summary"]["metrics"]
                mutate(bad_rows, metrics)
                self._write_boundary_csv(paths["baseline"], bad_rows)
                comparison = module.write_comparison(self.root, scheme_results, labels)
                self._assert_boundary_evidence_failure(comparison)
                self.assertIn(expected_reason, comparison["boundary_differences_error"])

    def test_wer_requires_nonnegative_consistent_detail_metrics(self):
        first_case = module.TEST_CASES[0]
        invalid_payloads = {
            "negative": {"wer": -0.1},
            "partial detail": {"wer": 0.0, "reference_tokens": 0},
            "component sum": {
                "wer": 0.3, "reference_tokens": 10, "errors": 3,
                "insertions": 1, "deletions": 1, "substitutions": 2,
            },
            "wer formula": {
                "wer": 0.2, "reference_tokens": 10, "errors": 3,
                "insertions": 1, "deletions": 1, "substitutions": 1,
            },
        }
        for name, payload in invalid_payloads.items():
            with self.subTest(name=name):
                labels, _, _, scheme_results = self._write_complete_boundary_evidence()
                wer_path = self.root / "baseline" / first_case.file_id / "metrics" / "wer.json"
                wer_path.write_text(json.dumps(payload), encoding="utf-8")
                comparison = module.write_comparison(self.root, scheme_results, labels)
                wer = comparison["schemes"]["baseline"]["wer_by_file"][first_case.file_id]
                self.assertEqual(wer["status"], "invalid")
                self.assertFalse(comparison["ranking_eligible"])
                self.assertIn("WER is unavailable", comparison["ranking_reason"])

    def test_reference_boundary_rebuild_matches_copied_metrics_sort_order(self):
        labels = self.root / "01_input" / "labels"
        labels.mkdir(parents=True)
        rows = []
        for case in module.TEST_CASES:
            # Intentionally unordered, with identical starts, different ends and overlap.
            (labels / case.label.name).write_text(
                f"{case.file_id}_500_700 A 后续\n"
                f"{case.file_id}_0_1000 A 长片段\n"
                f"{case.file_id}_0_900 B 短片段\n",
                encoding="utf-8",
            )
            rows.append({
                "文件ID": case.file_id, "切换序号": 1, "参考前说话人": "B", "参考后说话人": "A",
                "区间起始毫秒": 900, "区间结束毫秒": 0,
                "扩展后起始毫秒": -500, "扩展后结束毫秒": 1400,
                "匹配预测边界毫秒": "", "状态": "未命中",
            })
        expected_keys = {(case.file_id, 900, 0) for case in module.TEST_CASES}
        self.assertEqual(set(module._reference_boundary_texts(labels)), expected_keys)
        paths = {}
        for scheme in module.SCHEMES:
            path = self.root / "03_metrics" / scheme / "speaker_diarization_boundary_details.csv"
            self._write_boundary_csv(path, rows)
            paths[scheme] = path
        self._write_completed_wer_artifacts()
        scheme_results = {
            scheme: {
                "status": "metrics_completed",
                "summary": self._metrics_summary(der=0.1, hit_rate=0.0, speaker_change_hits=0),
                "per_file": self._completed_per_file_metrics(),
                "boundary_details_path": str(paths[scheme]),
            }
            for scheme in module.SCHEMES
        }
        comparison = module.write_comparison(self.root, scheme_results, labels)
        self.assertTrue(comparison["boundary_differences_available"])
        self.assertTrue(comparison["ranking_eligible"])

    def test_metrics_process_oserror_logs_failure_and_summarize_writes_artifacts(self):
        source = self._write_fake_metrics_source()
        tools_dir = self.root / "tools"
        module.install_metrics_tool(source, tools_dir)
        normalized = self.root / "normalized"
        labels = self.root / "labels"
        normalized.mkdir()
        labels.mkdir()
        with mock.patch.object(module.subprocess, "run", side_effect=OSError("spawn denied")):
            result = module.run_speaker_metrics("baseline", tools_dir, normalized, labels, self.root / "metrics")
        self.assertEqual(result["status"], "metrics_failed")
        self.assertIn("spawn denied", (self.root / "metrics" / "speaker_metrics.exception.log").read_text(encoding="utf-8"))
        self.assertTrue((self.root / "metrics" / "speaker_metrics.stdout.log").is_file())
        self.assertTrue((self.root / "metrics" / "speaker_metrics.stderr.log").is_file())

        manifest_builder = TaskTwoTest()
        manifest_builder.root = self.root
        manifest = manifest_builder._valid_manifest()
        manifest_builder._materialize_frozen_inputs(manifest)
        module._write_json(self.root / "manifest.json", manifest)
        with mock.patch.object(module.subprocess, "run", side_effect=OSError("spawn denied")):
            comparison = module.summarize_evaluation(self.root, metrics_source=source)
        self.assertFalse(comparison["ranking_eligible"])
        self.assertTrue((self.root / "05_comparison" / "comparison.json").is_file())
        self.assertTrue((self.root / "05_comparison" / "boundary_differences.csv").is_file())
        report = self.root / "报告.md"
        self.assertTrue(report.is_file())
        self.assertIn("未形成有效总体排名", report.read_text(encoding="utf-8"))

    def test_atomic_artifact_writer_preserves_existing_file_when_replace_fails(self):
        target = self.root / "05_comparison" / "comparison.json"
        target.parent.mkdir(parents=True)
        target.write_text("old artifact\n", encoding="utf-8")
        with mock.patch.object(module.os, "replace", side_effect=OSError("replace denied")):
            with self.assertRaisesRegex(OSError, "replace denied"):
                module._atomic_write_text(target, "new artifact\n")
        self.assertEqual(target.read_text(encoding="utf-8"), "old artifact\n")
        self.assertEqual(list(target.parent.glob(f".{target.name}.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
