import csv
import importlib.util
import pathlib
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch


SCRIPT = pathlib.Path(__file__).with_name("pipeline-vad-asr-wer-batch.py")
SPEC = importlib.util.spec_from_file_location("pipeline_vad_asr_wer_batch", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def parse_args(root: pathlib.Path, extra_args: list[str]):
    return MODULE.build_parser().parse_args(
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
        + extra_args
    )


class PipelineBatchTest(unittest.TestCase):
    def test_expands_parameter_cartesian_product(self):
        with TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            args = parse_args(
                root,
                [
                    "--skip-wer",
                    "--min-silence-duration",
                    "1.0",
                    "0.9",
                    "--pre-speech-pad-duration",
                    "0.0",
                    "0.1",
                ],
            )
            calls = []

            def run_pipeline(run_args):
                calls.append(
                    (
                        run_args.min_silence_duration,
                        run_args.pre_speech_pad_duration,
                        pathlib.Path(run_args.output_dir),
                    )
                )
                return {}

            with patch.object(MODULE.SINGLE_RUN, "run_pipeline", run_pipeline):
                MODULE.run_batch(args)

        self.assertEqual(
            [(silence, pad) for silence, pad, _ in calls],
            [(1.0, 0.0), (1.0, 0.1), (0.9, 0.0), (0.9, 0.1)],
        )
        self.assertEqual(len({path.name for _, _, path in calls}), 4)

    def test_stops_after_first_failed_combination(self):
        with TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            args = parse_args(
                root,
                [
                    "--skip-wer",
                    "--min-silence-duration",
                    "1.0",
                    "0.9",
                    "--pre-speech-pad-duration",
                    "0.0",
                    "0.1",
                ],
            )
            call_count = 0

            def run_pipeline(_):
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    raise ValueError("forced failure")
                return {}

            with patch.object(MODULE.SINGLE_RUN, "run_pipeline", run_pipeline):
                with self.assertRaisesRegex(ValueError, "forced failure"):
                    MODULE.run_batch(args)

        self.assertEqual(call_count, 2)

    def test_expands_merge_parameter_values_and_fills_defaults(self):
        with TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            args = parse_args(
                root,
                [
                    "--skip-wer",
                    "--merge-gap-duration=2.0",
                    "1.0",
                    "--short-segment-duration=8.0",
                    "6.0",
                    "--max-merged-duration=30.0",
                ],
            )
            calls = []

            def run_pipeline(run_args):
                calls.append(
                    (
                        run_args.merge_gap_duration,
                        run_args.short_segment_duration,
                        run_args.max_merged_duration,
                    )
                )
                return {}

            with patch.object(MODULE.SINGLE_RUN, "run_pipeline", run_pipeline):
                MODULE.run_batch(args)

        self.assertEqual(
            calls,
            [
                (2.0, 8.0, 30.0),
                (2.0, 6.0, 30.0),
                (1.0, 8.0, 30.0),
                (1.0, 6.0, 30.0),
            ],
        )

    def test_writes_wer_summary(self):
        with TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            args = parse_args(
                root,
                [
                    "--label",
                    str(root / "label.txt"),
                    "--min-silence-duration",
                    "1.0",
                    "--pre-speech-pad-duration",
                    "0.2",
                ],
            )

            def run_pipeline(run_args):
                detail_path = pathlib.Path(run_args.output_dir) / "wer_detail.txt"
                detail_path.parent.mkdir(parents=True, exist_ok=True)
                detail_path.write_text(
                    "id\twer\tref_words\terr_words\tdel_words\tins_words\n"
                    "utt001\t12.50\t8\t1\t0\t1\n",
                    encoding="utf-8",
                )
                return {"detail": detail_path}

            with patch.object(MODULE.SINGLE_RUN, "run_pipeline", run_pipeline):
                summary_path = MODULE.run_batch(args)

            with summary_path.open("r", encoding="utf-8", newline="") as summary_file:
                rows = list(csv.DictReader(summary_file, delimiter="\t"))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["min_silence_duration"], "1.0")
        self.assertEqual(rows[0]["pre_speech_pad_duration"], "0.2")
        self.assertEqual(rows[0]["wer"], "12.50")

    def test_expands_merge_parameter_values(self):
        with TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            args = parse_args(
                root,
                [
                    "--skip-wer",
                    "--merge-gap-duration",
                    "2.0",
                    "3.0",
                    "--short-segment-duration",
                    "10.0",
                    "11.0",
                    "--max-merged-duration",
                    "25.0",
                ],
            )
            calls = []

            def run_pipeline(run_args):
                calls.append(
                    (
                        run_args.merge_gap_duration,
                        run_args.short_segment_duration,
                        run_args.max_merged_duration,
                        pathlib.Path(run_args.output_dir),
                    )
                )
                return {}

            with patch.object(MODULE.SINGLE_RUN, "run_pipeline", run_pipeline):
                MODULE.run_batch(args)

        self.assertEqual(
            [(gap, short, maximum) for gap, short, maximum, _ in calls],
            [
                (2.0, 10.0, 25.0),
                (2.0, 11.0, 25.0),
                (3.0, 10.0, 25.0),
                (3.0, 11.0, 25.0),
            ],
        )
        self.assertEqual(len({call[3].name for call in calls}), 4)


if __name__ == "__main__":
    unittest.main()
