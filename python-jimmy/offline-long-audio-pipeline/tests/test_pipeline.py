import copy
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

PIPELINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_DIR))

from audio_io import AudioFormat, LoadedAudio
from pipeline import PipelineConfig, run_pipeline
from vad import SpeechSegment


class PipelineTest(unittest.TestCase):
    def test_pipeline_resolves_nonoverlapping_segments_then_clusters_and_records_metadata(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "test"
            run_dir.mkdir(parents=True)
            config = PipelineConfig(
                audio=root / "input.pcm",
                output_root=root,
                audio_format=AudioFormat("pcm", 16000, 1, 2),
            )
            waveform = np.zeros(16000, dtype=np.float32)
            raw_vad_segments = [
                SpeechSegment(1, 0, 300, waveform[:4800]),
                SpeechSegment(2, 300, 600, waveform[4800:9600]),
                SpeechSegment(3, 600, 1000, waveform[9600:]),
            ]
            final_segments = [
                SpeechSegment(
                    1, 0, 1000, waveform[:8000],
                    speaker_id="speaker_00", speaker_composition="single_speaker",
                ),
                SpeechSegment(
                    2, 1000, 1500, waveform[8000:],
                    speaker_id="unknown", speaker_composition="overlapped_speakers",
                ),
            ]
            activity = [object()]
            segmentation = SimpleNamespace(
                infer_speaker_count_spans=Mock(return_value=activity)
            )
            runtimes = SimpleNamespace(
                vad=object(),
                vad_window_size=512,
                recognizer=object(),
                extractor=object(),
                segmentation=segmentation,
                resolved_files={},
            )
            resolution_stats = SimpleNamespace(true_overlap_count=1)
            captured_metadata = {}
            with patch("pipeline.load_audio", return_value=LoadedAudio(waveform, 16000)), \
                 patch("pipeline.build_runtimes", return_value=runtimes) as build_runtimes, \
                 patch("pipeline.collect_vad_segments", return_value=raw_vad_segments), \
                 patch("pipeline.resolve_final_segments", return_value=(final_segments, resolution_stats)) as resolve, \
                 patch("pipeline.assign_speaker_ids_with_centroids", return_value=(1, 0, 1)) as assign, \
                 patch("pipeline.transcribe_segments") as transcribe, \
                 patch("pipeline.create_run_directory", return_value=run_dir), \
                 patch("pipeline.write_results") as write_results, \
                 patch("pipeline.write_metadata", side_effect=lambda _run_dir, metadata: captured_metadata.update(copy.deepcopy(metadata))) as write_metadata:
                result = run_pipeline(config)
                log_content = (run_dir / "run.log").read_text(encoding="utf-8")

        self.assertEqual(result.run_dir, run_dir)
        self.assertEqual(result.segment_count, 2)
        segmentation.infer_speaker_count_spans.assert_called_once_with(waveform)
        resolve.assert_called_once_with(
            raw_vad_segments,
            activity,
            waveform,
            16000,
            min_duration_on=0.5,
            min_duration_off=0.5,
        )
        assign.assert_called_once_with(
            runtimes.extractor,
            final_segments,
            cluster_threshold=0.5,
            num_clusters=2,
            assignment_similarity_threshold=0.5,
        )
        transcribe.assert_called_once_with(runtimes.recognizer, final_segments, 16000)
        self.assertIs(write_results.call_args.args[3], final_segments)
        write_metadata.assert_called_once()
        self.assertTrue(build_runtimes.call_args.kwargs["enable_segmentation"])
        self.assertNotIn("diarization_min_duration_on", build_runtimes.call_args.kwargs)
        self.assertNotIn("diarization_min_duration_off", build_runtimes.call_args.kwargs)
        self.assertEqual(captured_metadata["raw_vad_segment_count"], 3)
        self.assertEqual(captured_metadata["final_asr_segment_count"], 2)
        self.assertEqual(captured_metadata["speaker_composition_counts"], {
            "single_speaker": 1, "overlapped_speakers": 1, "unknown_activity": 0,
        })
        self.assertEqual(captured_metadata["duration_class_counts"], {"long": 1, "short": 1})
        self.assertEqual(captured_metadata["cluster_eligible_segment_count"], 1)
        self.assertEqual(captured_metadata["excluded_segment_count"], 1)
        self.assertEqual(captured_metadata["final_clustered_speaker_count"], 1)
        self.assertEqual(captured_metadata["centroid_assigned_excluded_segment_count"], 0)
        self.assertEqual(captured_metadata["unknown_excluded_segment_count"], 1)
        self.assertEqual(captured_metadata["true_overlap_count"], 1)
        self.assertNotIn("tolerated_overlap_count", captured_metadata)
        self.assertEqual(captured_metadata["embedding_error_count"], 1)
        self.assertIn("output_seconds", captured_metadata["timings"])
        for stage in ("audio_io", "model_load", "vad", "segmentation", "timeline_resolution", "speaker", "asr", "output"):
            self.assertIn(f"stage={stage} event=complete", log_content)

    def test_rejects_non_fixed_cluster_duration_before_loading_models(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = PipelineConfig(
                audio=root / "input.pcm",
                output_root=root,
                min_cluster_duration=1.5,
            )
            with patch("pipeline.build_runtimes") as build_runtimes:
                with self.assertRaisesRegex(ValueError, "min_cluster_duration is fixed at 1.0"):
                    run_pipeline(config)
        build_runtimes.assert_not_called()

    def test_rejects_negative_diarization_min_durations(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = PipelineConfig(audio=root / "input.pcm", diarization_min_duration_on=-0.1)
            with patch("pipeline.build_runtimes") as build_runtimes:
                with self.assertRaisesRegex(ValueError, "min_duration_on"):
                    run_pipeline(config)
        build_runtimes.assert_not_called()

    def test_rejects_min_text_confidence_outside_unit_interval(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = PipelineConfig(audio=root / "input.pcm", min_text_confidence=1.2)
            with patch("pipeline.build_runtimes") as build_runtimes:
                with self.assertRaisesRegex(ValueError, "min_text_confidence must be in"):
                    run_pipeline(config)
        build_runtimes.assert_not_called()

    def test_vad_only_mode_skips_pyannote_and_keeps_vad_segments(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "test"
            run_dir.mkdir(parents=True)
            config = PipelineConfig(
                audio=root / "input.pcm",
                output_root=root,
                audio_format=AudioFormat("pcm", 16000, 1, 2),
                segmentation_mode="vad",
                run_label="vad_only",
            )
            waveform = np.zeros(16000, dtype=np.float32)
            raw_vad_segments = [
                SpeechSegment(1, 0, 400, waveform[:6400]),
                SpeechSegment(2, 500, 1000, waveform[8000:]),
            ]
            vad_only_segments = [
                SpeechSegment(1, 0, 400, waveform[:6400], speaker_composition="single_speaker"),
                SpeechSegment(2, 500, 1000, waveform[8000:], speaker_composition="single_speaker"),
            ]
            runtimes = SimpleNamespace(
                vad=object(),
                vad_window_size=512,
                recognizer=object(),
                extractor=object(),
                segmentation=None,
                resolved_files={},
            )
            captured_metadata = {}
            with patch("pipeline.load_audio", return_value=LoadedAudio(waveform, 16000)), \
                 patch("pipeline.build_runtimes", return_value=runtimes) as build_runtimes, \
                 patch("pipeline.collect_vad_segments", return_value=raw_vad_segments), \
                 patch("pipeline.finalize_vad_only_segments", return_value=(vad_only_segments, SimpleNamespace(true_overlap_count=0))) as finalize, \
                 patch("pipeline.resolve_final_segments") as resolve, \
                 patch("pipeline.assign_speaker_ids_with_centroids", return_value=(0, 0, 0)), \
                 patch("pipeline.transcribe_segments"), \
                 patch("pipeline.create_run_directory", return_value=run_dir), \
                 patch("pipeline.write_results"), \
                 patch("pipeline.write_metadata", side_effect=lambda _run_dir, metadata: captured_metadata.update(copy.deepcopy(metadata))):
                result = run_pipeline(config)
                log_content = (run_dir / "run.log").read_text(encoding="utf-8")

        self.assertEqual(result.segment_count, 2)
        self.assertFalse(build_runtimes.call_args.kwargs["enable_segmentation"])
        finalize.assert_called_once_with(raw_vad_segments)
        resolve.assert_not_called()
        self.assertIn("stage=segmentation event=skipped mode=vad", log_content)
        self.assertEqual(captured_metadata["raw_vad_segment_count"], 2)
        self.assertEqual(captured_metadata["final_asr_segment_count"], 2)

    def test_whisper_engine_transcribes_before_clustering(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "test"
            run_dir.mkdir(parents=True)
            config = PipelineConfig(
                audio=root / "input.wav",
                output_root=root,
                audio_format=AudioFormat("wav", 16000, 1, 2),
                asr_engine="whisper",
                whisper_languages=("en", "hi"),
                num_clusters=2,
            )
            waveform = np.zeros(16000, dtype=np.float32)
            final_segments = [
                SpeechSegment(1, 0, 1000, waveform[:8000], speaker_composition="single_speaker"),
            ]
            order = []
            runtimes = SimpleNamespace(
                vad=object(),
                vad_window_size=512,
                recognizer=None,
                extractor=object(),
                segmentation=SimpleNamespace(infer_speaker_count_spans=Mock(return_value=[])),
                resolved_files={},
            )
            with patch("pipeline.load_audio", return_value=LoadedAudio(waveform, 16000)), \
                 patch("pipeline.build_runtimes", return_value=runtimes) as build_runtimes, \
                 patch("pipeline.collect_vad_segments", return_value=final_segments), \
                 patch("pipeline.resolve_final_segments", return_value=(final_segments, SimpleNamespace(true_overlap_count=0))), \
                 patch("pipeline.transcribe_segments_with_whisper", side_effect=lambda *_args, **_kwargs: order.append("asr")) as whisper, \
                 patch("pipeline.assign_speaker_ids_with_centroids", side_effect=lambda *_args, **_kwargs: order.append("speaker") or (0, 0, 0)), \
                 patch("pipeline.transcribe_segments") as paraformer, \
                 patch("pipeline.create_run_directory", return_value=run_dir), \
                 patch("pipeline.write_results"), \
                 patch("pipeline.write_metadata"):
                run_pipeline(config)

        self.assertEqual(order, ["asr", "speaker"])
        self.assertFalse(build_runtimes.call_args.kwargs["enable_local_asr"])
        paraformer.assert_not_called()
        whisper.assert_called_once()
        self.assertEqual(whisper.call_args.args[1].min_text_confidence, 0.30)


if __name__ == "__main__":
    unittest.main()
