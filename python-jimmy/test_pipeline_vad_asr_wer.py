import importlib.util
import pathlib
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch


SCRIPT = pathlib.Path(__file__).with_name("pipeline-vad-asr-wer.py")
SPEC = importlib.util.spec_from_file_location("pipeline_vad_asr_wer", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PipelineArgumentTest(unittest.TestCase):
    def test_asr_model_replaces_model_dir_and_label_is_optional(self):
        args = MODULE.build_parser().parse_args(
            [
                "--audio",
                "audio.wav",
                "--silero-vad-model",
                "silero.onnx",
                "--asr-model",
                "asr-model",
                "--language",
                "zh",
                "--output-dir",
                "out",
            ]
        )

        self.assertEqual(args.asr_model, "asr-model")
        self.assertIsNone(args.label)
        self.assertTrue(MODULE.skip_wer(args))

    def test_skip_wer_does_not_require_label(self):
        args = MODULE.build_parser().parse_args(
            [
                "--audio",
                "audio.wav",
                "--ten-vad-model",
                "ten.onnx",
                "--asr-model",
                "asr-model",
                "--language",
                "zh",
                "--output-dir",
                "out",
                "--skip-wer",
            ]
        )

        self.assertTrue(MODULE.skip_wer(args))

    def test_label_enables_wer_unless_explicitly_skipped(self):
        args = MODULE.build_parser().parse_args(
            [
                "--audio",
                "audio.wav",
                "--silero-vad-model",
                "silero.onnx",
                "--asr-model",
                "asr-model",
                "--language",
                "zh",
                "--label",
                "label.txt",
                "--output-dir",
                "out",
            ]
        )
        self.assertFalse(MODULE.skip_wer(args))
        args.skip_wer = True
        self.assertTrue(MODULE.skip_wer(args))

    def test_pipeline_omits_wer_without_label(self):
        with TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            args = MODULE.build_parser().parse_args(
                [
                    "--audio",
                    str(root / "audio.wav"),
                    "--silero-vad-model",
                    str(root / "silero.onnx"),
                    "--asr-model",
                    str(root / "asr-model"),
                    "--language",
                    "zh",
                    "--output-dir",
                    str(root / "out"),
                ]
            )
            commands = []
            with patch.object(MODULE, "validate_inputs"), patch.object(
                MODULE, "run_step", side_effect=lambda name, command: commands.append(command)
            ):
                outputs = MODULE.run_pipeline(args)

        self.assertEqual(len(commands), 3)
        self.assertNotIn("evaluation.py", " ".join(" ".join(command) for command in commands))
        asr_command = commands[1]
        self.assertIn("--model-dir", asr_command)
        self.assertIn(str(root / "asr-model"), asr_command)
        self.assertNotIn("detail", outputs)

    def test_vad_merge_is_disabled_without_merge_arguments(self):
        with TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            args = MODULE.build_parser().parse_args(
                [
                    "--audio",
                    str(root / "audio.wav"),
                    "--silero-vad-model",
                    str(root / "silero.onnx"),
                    "--asr-model",
                    str(root / "asr-model"),
                    "--language",
                    "zh",
                    "--output-dir",
                    str(root / "out"),
                ]
            )
            commands = []
            with patch.object(MODULE, "validate_inputs"), patch.object(
                MODULE, "run_step", side_effect=lambda name, command: commands.append(command)
            ):
                MODULE.run_pipeline(args)

        self.assertIn("k2-vad_cut.py", commands[0])
        self.assertNotIn("k2-vad-cut-merge.py", commands[0])

    def test_any_merge_argument_enables_merge_with_defaults(self):
        with TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            args = MODULE.build_parser().parse_args(
                [
                    "--audio",
                    str(root / "audio.wav"),
                    "--silero-vad-model",
                    str(root / "silero.onnx"),
                    "--asr-model",
                    str(root / "asr-model"),
                    "--language",
                    "zh",
                    "--output-dir",
                    str(root / "out"),
                    "--merge-gap-duration",
                    "0.0",
                ]
            )
            commands = []
            with patch.object(MODULE, "validate_inputs"), patch.object(
                MODULE, "run_step", side_effect=lambda name, command: commands.append(command)
            ):
                MODULE.run_pipeline(args)

        vad_command = commands[0]
        self.assertIn("k2-vad-cut-merge.py", vad_command)
        self.assertIn("--merge-gap-duration", vad_command)
        self.assertIn("0.0", vad_command)
        self.assertIn("--short-segment-duration", vad_command)
        self.assertIn("6.0", vad_command)
        self.assertIn("--max-merged-duration", vad_command)
        self.assertIn("30.0", vad_command)


if __name__ == "__main__":
    unittest.main()
