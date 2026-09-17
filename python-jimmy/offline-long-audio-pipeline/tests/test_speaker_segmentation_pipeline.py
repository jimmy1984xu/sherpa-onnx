import importlib.util
from pathlib import Path
import unittest
import tempfile

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER_PATH = REPO_ROOT / "python-jimmy" / "offline-long-audio-pipeline-asr-speaker-segmentation.py"
PIPELINE_DIR = REPO_ROOT / "python-jimmy" / "offline-long-audio-pipeline"


def _load_runner():
    import sys

    sys.path.insert(0, str(PIPELINE_DIR))
    spec = importlib.util.spec_from_file_location(
        "offline_long_audio_pipeline_asr_speaker_segmentation", RUNNER_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SpeakerSegmentationPipelineTimelineTest(unittest.TestCase):
    @staticmethod
    def _vad_segment(end_ms: int) -> list[object]:
        runner = _load_runner()
        return [
            runner.SpeechSegment(
                segment_index=1,
                start_ms=0,
                end_ms=end_ms,
                samples=np.zeros(end_ms * 16, dtype=np.float32),
            )
        ]

    def test_folds_overlap_into_following_single_speaker_segment(self):
        runner = _load_runner()
        spans = [
            {"start": 0.0, "end": 2.0, "speaker_count": 1, "flag": 2},
            {"start": 2.0, "end": 3.0, "speaker_count": 2, "flag": 1},
            {"start": 3.0, "end": 5.0, "speaker_count": 1, "flag": 4},
        ]

        segments = runner.resolve_vad_segments_with_speaker_segmentation(
            self._vad_segment(5000), spans, np.zeros(80_000, dtype=np.float32), 16000
        )

        self.assertEqual(
            [(segment.start_ms, segment.end_ms, segment.speaker_composition) for segment in segments],
            [(0, 2000, "single_speaker"), (2000, 5000, "single_speaker")],
        )
        self.assertEqual(segments[1].overlap_regions, [(2000, 3000)])

    def test_absorbs_count_zero_tail_inside_vad_speech(self):
        runner = _load_runner()
        spans = [
            {"start": 0.0, "end": 0.9, "speaker_count": 1, "flag": 1},
            {"start": 0.9, "end": 1.0, "speaker_count": 0, "flag": 4},
        ]

        segments = runner.resolve_vad_segments_with_speaker_segmentation(
            self._vad_segment(1000), spans, np.zeros(16_000, dtype=np.float32), 16000
        )

        self.assertEqual([(segment.start_ms, segment.end_ms) for segment in segments], [(0, 1000)])

    def test_folds_short_overlap_island_into_single_asr_segment(self):
        runner = _load_runner()
        spans = [
            {"start": 0.0, "end": 1.2, "speaker_count": 1, "flag": 1},
            {"start": 1.2, "end": 1.28, "speaker_count": 2, "flag": 1},
            {"start": 1.28, "end": 3.0, "speaker_count": 1, "flag": 4},
        ]

        segments = runner.resolve_vad_segments_with_speaker_segmentation(
            self._vad_segment(3000), spans, np.zeros(48_000, dtype=np.float32), 16000
        )

        self.assertEqual([(segment.start_ms, segment.end_ms) for segment in segments], [(0, 3000)])
        self.assertEqual(segments[0].overlap_regions, [(1200, 1280)])

    def test_suppresses_weak_single_speaker_change_but_keeps_strong_change(self):
        runner = _load_runner()
        weak_spans = [
            {"start": 0.0, "end": 0.5, "speaker_count": 1, "flag": 2},
            {"start": 0.5, "end": 2.0, "speaker_count": 1, "flag": 4},
        ]
        strong_spans = [
            {"start": 0.0, "end": 1.1, "speaker_count": 1, "flag": 2},
            {"start": 1.1, "end": 2.2, "speaker_count": 1, "flag": 4},
        ]

        weak = runner.resolve_vad_segments_with_speaker_segmentation(
            self._vad_segment(2000), weak_spans, np.zeros(32_000, dtype=np.float32), 16000
        )
        strong = runner.resolve_vad_segments_with_speaker_segmentation(
            self._vad_segment(2200), strong_spans, np.zeros(35_200, dtype=np.float32), 16000
        )

        self.assertEqual([(segment.start_ms, segment.end_ms) for segment in weak], [(0, 2000)])
        self.assertEqual([(segment.start_ms, segment.end_ms) for segment in strong], [(0, 1100), (1100, 2200)])

    def test_summarizes_overlapped_segments_for_comparison_report(self):
        runner = _load_runner()
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "result.json").write_text(
                '{"segments": [{"asr_text": "测试", '
                '"speaker_composition": "overlapped_speakers", '
                '"cut_left": "pyannote", "cut_right": "vad", '
                '"time_range": "00:00.000-00:01.000"}]}',
                encoding="utf-8",
            )
            reference = run_dir / "reference.txt"
            reference.write_text("测试", encoding="utf-8")

            summary = runner._result_summary(run_dir, reference)

        self.assertEqual(summary["wer"], 0.0)
        self.assertEqual(summary["multi_segment_count"], 1)
        self.assertEqual(summary["multi_break_count"], 1)



class SpeakerSegmentationPipelineCompatibilityTest(unittest.TestCase):
    def test_parser_preserves_baseline_pipeline_options_and_defaults(self):
        runner = _load_runner()
        baseline_path = REPO_ROOT / "python-jimmy" / "offline-long-audio-pipeline-asr-speaker.py"
        spec = importlib.util.spec_from_file_location("offline_long_audio_pipeline_asr_speaker", baseline_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {baseline_path}")
        baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline)

        baseline_args = baseline.build_parser().parse_args(["--audio", "input.pcm"])
        candidate_args = runner.build_parser().parse_args(["--audio", "input.pcm"])

        for name in (
            "output_root", "audio_format", "sample_rate", "channels", "sample_width",
            "asr_dir", "vad_dir", "speaker_dir", "asr_num_threads",
            "speaker_num_threads", "vad_threshold", "min_silence_duration",
            "min_speech_duration", "max_speech_duration", "pre_speech_pad_duration",
            "cluster_threshold", "num_clusters", "asr_engine", "whisper_url",
            "whisper_languages", "whisper_timeout_ms", "min_text_confidence",
            "min_cluster_duration", "centroid_assignment_similarity_threshold",
            "save_segments", "debug", "run_label",
        ):
            self.assertEqual(getattr(candidate_args, name), getattr(baseline_args, name), name)

    def test_builds_baseline_pipeline_config_with_original_diarization_disabled(self):
        runner = _load_runner()
        args = runner.build_parser().parse_args([
            "--audio", "input.pcm",
            "--segmentation-model", "model.onnx",
            "--min-duration-on", "0.3",
            "--min-duration-off", "0.5",
            "--change-vote-threshold", "0.5",
        ])

        config = runner.build_baseline_pipeline_config(args)

        self.assertEqual(config.audio, Path("input.pcm"))
        self.assertEqual(config.segmentation_dir, Path("model.onnx").parent)
        self.assertEqual(config.segmentation_mode, "vad-pyannote")
        self.assertEqual(config.diarization_min_duration_on, 0.5)
        self.assertEqual(config.diarization_min_duration_off, 0.5)

    def test_rejects_baseline_vad_only_mode(self):
        runner = _load_runner()
        args = runner.build_parser().parse_args([
            "--audio", "input.pcm", "--segmentation-mode", "vad"
        ])

        with self.assertRaisesRegex(ValueError, "requires --segmentation-mode=vad-pyannote"):
            runner.build_baseline_pipeline_config(args)


if __name__ == "__main__":
    unittest.main()

class StreamingApiAdapterTest(unittest.TestCase):
    def test_streams_fixed_chunks_and_drains_after_input_finished(self):
        from types import SimpleNamespace

        runner = _load_runner()
        accepted: list[list[float]] = []

        class FakeSegmenter:
            sample_rate = 16000

            def __init__(self, _config):
                self._queue = []

            def accept_waveform(self, chunk):
                accepted.append(list(chunk))

            def input_finished(self):
                self._queue.append(
                    SimpleNamespace(start=0.0, end=0.0625, speaker_count=1, flag=4)
                )

            def empty(self):
                return not self._queue

            @property
            def front(self):
                return self._queue[0]

            def pop(self):
                self._queue.pop(0)

        class FakeConfig:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

            def validate(self):
                return True

        fake_api = SimpleNamespace(
            OfflineSpeakerSegmentationPyannoteModelConfig=FakeConfig,
            OfflineSpeakerSegmentationModelConfig=FakeConfig,
            SpeakerSegmentationConfig=FakeConfig,
            SpeakerSegmentation=FakeSegmenter,
        )
        spans = runner.stream_speaker_segmentation_samples(
            np.zeros(1000, dtype=np.float32),
            model=Path("model.onnx"),
            chunk_ms=32,
            num_threads=4,
            min_duration_on=0.30,
            min_duration_off=0.50,
            change_vote_threshold=0.50,
            api=fake_api,
        )
        self.assertEqual([len(chunk) for chunk in accepted], [512, 488])
        self.assertEqual(
            spans,
            [{"start": 0.0, "end": 0.0625, "speaker_count": 1, "flag": 4}],
        )
