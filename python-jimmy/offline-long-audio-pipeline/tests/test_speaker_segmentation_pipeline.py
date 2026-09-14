import importlib.util
from pathlib import Path
import unittest

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
    def test_splits_vad_on_count_and_single_speaker_change_boundaries(self):
        runner = _load_runner()
        vad_segments = [
            runner.SpeechSegment(
                segment_index=1,
                start_ms=0,
                end_ms=5000,
                samples=np.zeros(80_000, dtype=np.float32),
            )
        ]
        spans = [
            {"start": 0.0, "end": 2.0, "speaker_count": 1, "flag": 2},
            {"start": 2.0, "end": 3.0, "speaker_count": 2, "flag": 1},
            {"start": 3.0, "end": 5.0, "speaker_count": 1, "flag": 4},
        ]

        segments = runner.resolve_vad_segments_with_speaker_segmentation(
            vad_segments, spans, np.zeros(80_000, dtype=np.float32), 16000
        )

        self.assertEqual(
            [(segment.start_ms, segment.end_ms, segment.speaker_composition) for segment in segments],
            [
                (0, 2000, "single_speaker"),
                (2000, 3000, "overlapped_speakers"),
                (3000, 5000, "single_speaker"),
            ],
        )
        self.assertEqual(
            [segment.cut_right for segment in segments],
            ["speaker_change", "speaker_count", "vad"],
        )
        self.assertEqual(segments[1].cut_left, "speaker_change")
        self.assertEqual(segments[1].overlap_regions, [(2000, 3000)])
        self.assertEqual(segments[2].cut_left, "speaker_count")


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
