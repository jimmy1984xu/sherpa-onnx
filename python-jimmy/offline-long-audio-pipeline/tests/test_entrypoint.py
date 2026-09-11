import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "offline-long-audio-pipeline-asr-speaker.py"
SPEC = importlib.util.spec_from_file_location("offline_long_audio_pipeline_entry", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class EntryPointTest(unittest.TestCase):
    def test_pcm_defaults_and_local_model_defaults_are_exposed(self):
        args = MODULE.build_parser().parse_args(["--audio", "input.pcm"])

        self.assertEqual(args.audio_format, "auto")
        self.assertEqual(args.sample_rate, 16000)
        self.assertEqual(args.channels, 1)
        self.assertEqual(args.sample_width, 2)
        self.assertIn("D:\\TransAI\\audio_model", args.asr_dir)

    def test_defaults_keep_brief_pauses_in_the_same_vad_segment(self):
        args = MODULE.build_parser().parse_args(["--audio", "input.pcm"])

        self.assertEqual(args.min_silence_duration, 0.8)
        self.assertEqual(args.max_speech_duration, 25.0)

    def test_allows_explicit_pipeline_tuning_without_embedding_duration_threshold(self):
        args = MODULE.build_parser().parse_args(
            ["--audio", "input.pcm", "--save-segments", "--cluster-threshold", "0.4"]
        )

        self.assertTrue(args.save_segments)
        self.assertEqual(args.cluster_threshold, 0.4)
        self.assertFalse(hasattr(args, "min_embedding_duration_ms"))
        self.assertEqual(args.min_cluster_duration, 1.0)
        self.assertEqual(args.centroid_assignment_similarity_threshold, 0.5)
        self.assertFalse(hasattr(args, "overlap_tolerance_ms"))

    def test_allows_controlled_cluster_assignment_overrides(self):
        args = MODULE.build_parser().parse_args([
            "--audio", "input.pcm",
            "--min-cluster-duration", "1.0",
            "--centroid-assignment-similarity-threshold", "0.6",
        ])

        self.assertEqual(args.min_cluster_duration, 1.0)
        self.assertEqual(args.centroid_assignment_similarity_threshold, 0.6)

    def test_does_not_expose_overlap_tolerance_ms(self):
        with self.assertRaises(SystemExit):
            MODULE.build_parser().parse_args(
                ["--audio", "input.pcm", "--overlap-tolerance-ms", "50"]
            )

    def test_does_not_expose_a_minimum_embedding_duration_option(self):
        with self.assertRaises(SystemExit):
            MODULE.build_parser().parse_args(
                ["--audio", "input.pcm", "--min-embedding-duration-ms", "500"]
            )


if __name__ == "__main__":
    unittest.main()

class DiarizationEntryPointTest(unittest.TestCase):
    def test_exposes_local_segmentation_and_confirmed_smoothing_defaults(self):
        args = MODULE.build_parser().parse_args(["--audio", "input.pcm"])

        self.assertIn("speaker_segmentation", args.segmentation_dir)
        self.assertEqual(args.diarization_min_duration_on, 0.5)
        self.assertEqual(args.diarization_min_duration_off, 0.5)

    def test_allows_diarization_model_and_smoothing_overrides(self):
        args = MODULE.build_parser().parse_args([
            "--audio", "input.pcm",
            "--segmentation-dir", r"D:\models\segmentation",
            "--diarization-min-duration-on", "0.4",
            "--diarization-min-duration-off", "0.6",
        ])

        self.assertEqual(args.segmentation_dir, r"D:\models\segmentation")
        self.assertEqual(args.diarization_min_duration_on, 0.4)
        self.assertEqual(args.diarization_min_duration_off, 0.6)

    def test_does_not_expose_unsafe_segmentation_window_tuning(self):
        with self.assertRaises(SystemExit):
            MODULE.build_parser().parse_args(
                ["--audio", "input.pcm", "--segmentation-window", "8"]
            )

    def test_exposes_vad_only_and_vad_pyannote_segmentation_modes(self):
        default_args = MODULE.build_parser().parse_args(["--audio", "input.pcm"])
        vad_args = MODULE.build_parser().parse_args(
            ["--audio", "input.pcm", "--segmentation-mode", "vad", "--run-label", "vad_only"]
        )

        self.assertEqual(default_args.segmentation_mode, "vad-pyannote")
        self.assertIsNone(default_args.run_label)
        self.assertEqual(vad_args.segmentation_mode, "vad")
        self.assertEqual(vad_args.run_label, "vad_only")
