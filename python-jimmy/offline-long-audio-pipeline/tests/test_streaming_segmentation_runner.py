import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


class SpeakerSegmentationApiTest(unittest.TestCase):
    def test_public_symbols_and_span_shape(self):
        import sherpa_onnx

        for name in (
            "SpeakerSegmentationConfig",
            "SpeakerSegmentation",
            "SpeakerSegmentationSpan",
            "CONTINUE",
            "SPEAKER_COUNT_CHANGED",
            "SINGLE_SPEAKER_CHANGED",
            "INPUT_FINISHED",
        ):
            self.assertTrue(hasattr(sherpa_onnx, name), name)


PIPELINE_DIR = Path(__file__).resolve().parents[1]


def _load_module(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_streaming_segmentation_helper():
    return _load_module(
        PIPELINE_DIR / "streaming_segmentation.py", "streaming_segmentation"
    )


def _load_direct_runner():
    return _load_module(
        PIPELINE_DIR.parent / "test-pyannote-segmentation-streaming.py",
        "test_pyannote_segmentation_streaming",
    )


class StreamingSegmentationReportTest(unittest.TestCase):
    def test_writes_span_summary_and_comparison_report(self):
        helper = _load_streaming_segmentation_helper()
        spans = [
            {"start": 0.0, "end": 1.0, "speaker_count": 1, "flag": 0},
            {"start": 1.0, "end": 2.0, "speaker_count": 1, "flag": 2},
            {"start": 2.0, "end": 3.0, "speaker_count": 2, "flag": 4},
        ]
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            summary = helper.write_span_artifacts(output_dir, "sample", 3.0, spans)
            jsonl = output_dir / "sample.spans.jsonl"
            summary_path = output_dir / "sample.summary.json"
            self.assertTrue(jsonl.is_file())
            self.assertTrue(summary_path.is_file())
            self.assertEqual(len(jsonl.read_text(encoding="utf-8").splitlines()), 3)
            self.assertEqual(
                set(json.loads(jsonl.read_text(encoding="utf-8").splitlines()[0])),
                {"start", "end", "speaker_count", "flag"},
            )
            written_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(
                set(written_summary),
                {
                    "audio_seconds",
                    "span_count",
                    "count_0_seconds",
                    "count_1_seconds",
                    "count_2_seconds",
                    "single_speaker_change_count",
                },
            )
            self.assertEqual(written_summary, summary)
            self.assertEqual(summary["audio_seconds"], 3.0)
            self.assertEqual(summary["span_count"], 3)
            self.assertEqual(summary["count_0_seconds"], 0.0)
            self.assertEqual(summary["count_1_seconds"], 2.0)
            self.assertEqual(summary["count_2_seconds"], 1.0)
            self.assertEqual(summary["single_speaker_change_count"], 1)

            runs = helper.merge_display_runs(spans)
            self.assertEqual(
                runs,
                [
                    {"start": 0.0, "end": 2.0, "speaker_count": 1, "flag": 2},
                    {"start": 2.0, "end": 3.0, "speaker_count": 2, "flag": 4},
                ],
            )

            comparison = helper.write_comparison_report(
                output_dir,
                baseline={"wer": 0.25, "segment_count": 7, "multi_segment_count": 2, "multi_break_count": 1},
                candidate={"wer": 0.20, "segment_count": 6, "multi_segment_count": 1, "multi_break_count": 0},
            )
            self.assertEqual(comparison["baseline"]["wer"], 0.25)
            self.assertEqual(comparison["speaker_segmentation"]["segment_count"], 6)
            self.assertTrue((output_dir / "comparison.json").is_file())
            report = (output_dir / "report.md").read_text(encoding="utf-8")
            self.assertIn("WER", report)
            self.assertIn("multi/overlap", report)

    def test_allows_terminal_input_finished_combined_with_a_boundary(self):
        helper = _load_streaming_segmentation_helper()
        spans = [
            {"start": 0.0, "end": 1.0, "speaker_count": 1, "flag": 0},
            {"start": 1.0, "end": 2.0, "speaker_count": 2, "flag": 5},
        ]
        self.assertEqual(helper.normalize_spans(spans), spans)

    def test_counts_terminal_combined_single_speaker_change(self):
        helper = _load_streaming_segmentation_helper()
        spans = [
            {"start": 0.0, "end": 1.0, "speaker_count": 1, "flag": 6},
        ]
        with tempfile.TemporaryDirectory() as directory:
            summary = helper.write_span_artifacts(Path(directory), "sample", 1.0, spans)
        self.assertEqual(summary["single_speaker_change_count"], 1)

    def test_recursive_same_stem_files_keep_distinct_output_directories(self):
        helper = _load_streaming_segmentation_helper()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "input"
            output = Path(directory) / "output"
            first = root / "a" / "session.pcm"
            second = root / "b" / "session.pcm"
            first.parent.mkdir(parents=True)
            second.parent.mkdir(parents=True)
            first.write_bytes(b"\x00\x00")
            second.write_bytes(b"\x00\x00")
            paths = helper._pcm_paths(root)
            self.assertEqual(paths, [first, second])
            self.assertEqual(helper._artifact_output_dir(output, root, first), output / "a")
            self.assertEqual(helper._artifact_output_dir(output, root, second), output / "b")

    def test_rejects_invalid_span_streams(self):
        helper = _load_streaming_segmentation_helper()
        invalid_span_streams = (
            [
                {"start": 0.0, "end": 1.0, "speaker_count": 1, "flag": 0},
                {"start": 0.5, "end": 2.0, "speaker_count": 1, "flag": 4},
            ],
            [{"start": 1.0, "end": 0.0, "speaker_count": 1, "flag": 4}],
            [{"start": 0.0, "end": 1.0, "speaker_count": 3, "flag": 4}],
            [
                {"start": 0.0, "end": 1.0, "speaker_count": 1, "flag": 4},
                {"start": 1.0, "end": 2.0, "speaker_count": 1, "flag": 0},
            ],
        )
        for spans in invalid_span_streams:
            with self.subTest(spans=spans):
                with self.assertRaises(ValueError):
                    helper.normalize_spans(spans)


class DirectStreamingSegmentationRunnerTest(unittest.TestCase):
    def test_cli_defaults_and_validation_do_not_import_extension(self):
        runner = _load_direct_runner()
        args = runner.parse_args(
            [
                "--input-dir", "input",
                "--output-dir", "output",
                "--model", "model.onnx",
            ]
        )
        self.assertEqual(args.sample_rate, 16000)
        self.assertEqual(args.channels, 1)
        self.assertEqual(args.sample_width, 2)
        self.assertEqual(args.chunk_ms, 32)
        self.assertEqual(args.num_threads, 4)
        self.assertEqual(args.min_duration_on, 0.30)
        self.assertEqual(args.min_duration_off, 0.50)
        self.assertEqual(args.change_vote_threshold, 0.50)

        args.input_dir = Path(tempfile.gettempdir())
        args.chunk_ms = 0
        with self.assertRaisesRegex(ValueError, "chunk-ms"):
            runner.validate_args(args)


if __name__ == "__main__":
    unittest.main()
