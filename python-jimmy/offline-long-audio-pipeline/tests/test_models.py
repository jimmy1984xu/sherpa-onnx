import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

PIPELINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_DIR))

from models import configure_pre_speech_pad, resolve_model_files


class ModelResolutionTest(unittest.TestCase):
    def test_resolves_expected_local_model_filenames(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.int8.onnx").touch()
            (root / "tokens.txt").touch()

            files = resolve_model_files(root, "asr")

        self.assertEqual(files.model.name, "model.int8.onnx")
        self.assertEqual(files.tokens.name, "tokens.txt")

    def test_resolves_vad_and_speaker_models(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "silero_vad.onnx").touch()
            self.assertEqual(resolve_model_files(root, "vad").model.name, "silero_vad.onnx")
            (root / "nemo_en_titanet_large.onnx").touch()
            self.assertEqual(
                resolve_model_files(root, "speaker").model.name,
                "nemo_en_titanet_large.onnx",
            )

    def test_missing_speaker_model_names_role_and_filename(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(
                FileNotFoundError, "speaker.*nemo_en_titanet_large.onnx"
            ):
                resolve_model_files(root, "speaker")


if __name__ == "__main__":
    unittest.main()

class VadCompatibilityTest(unittest.TestCase):
    def test_zero_pre_speech_pad_is_compatible_with_old_runtime(self):
        class OldVadConfig:
            pass

        configure_pre_speech_pad(OldVadConfig(), 0.0)

    def test_nonzero_pre_speech_pad_fails_clearly_on_old_runtime(self):
        class OldVadConfig:
            pass

        with self.assertRaisesRegex(ValueError, "does not support pre-speech padding"):
            configure_pre_speech_pad(OldVadConfig(), 0.1)

    def test_sets_pre_speech_pad_when_runtime_supports_it(self):
        class NewVadConfig:
            pre_speech_pad_duration = 0.0

        config = NewVadConfig()
        configure_pre_speech_pad(config, 0.2)
        self.assertEqual(config.pre_speech_pad_duration, 0.2)


class DiarizationModelRuntimeTest(unittest.TestCase):
    def test_resolves_segmentation_model(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.onnx").touch()

            files = resolve_model_files(root, "segmentation")

        self.assertEqual(files.model.name, "model.onnx")
        self.assertIsNone(files.tokens)

    def test_missing_segmentation_model_names_role_and_filename(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "segmentation.*model.onnx"):
                resolve_model_files(Path(directory), "segmentation")

    def test_build_runtimes_creates_local_segmentation_runtime(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        import models

        class ValidConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def validate(self):
                return True

        class FakeVadConfig(ValidConfig):
            def __init__(self):
                super().__init__()
                self.silero_vad = SimpleNamespace(window_size=512)

        class FakeSpeakerConfig(ValidConfig):
            pass

        fake_sherpa = SimpleNamespace(
            OfflineRecognizer=SimpleNamespace(from_paraformer=lambda **_kwargs: object()),
            VadModelConfig=FakeVadConfig,
            VoiceActivityDetector=lambda *_args, **_kwargs: object(),
            SpeakerEmbeddingExtractorConfig=FakeSpeakerConfig,
            SpeakerEmbeddingExtractor=lambda _config: object(),
        )
        segmentation_runtime = object()

        with TemporaryDirectory() as directory:
            root = Path(directory)
            asr_dir = root / "asr"
            vad_dir = root / "vad"
            speaker_dir = root / "speaker"
            segmentation_dir = root / "segmentation"
            for model_dir in (asr_dir, vad_dir, speaker_dir, segmentation_dir):
                model_dir.mkdir()
            (asr_dir / "model.int8.onnx").touch()
            (asr_dir / "tokens.txt").touch()
            (vad_dir / "silero_vad.onnx").touch()
            (speaker_dir / "nemo_en_titanet_large.onnx").touch()
            (segmentation_dir / "model.onnx").touch()

            with (
                patch.object(models, "sherpa_onnx", fake_sherpa),
                patch.object(
                    models,
                    "create_segmentation_runtime",
                    return_value=segmentation_runtime,
                ) as create_segmentation_runtime,
            ):
                runtimes = models.build_runtimes(
                    asr_dir=asr_dir,
                    vad_dir=vad_dir,
                    speaker_dir=speaker_dir,
                    segmentation_dir=segmentation_dir,
                    asr_num_threads=1,
                    speaker_num_threads=2,
                    vad_threshold=0.5,
                    min_silence_duration=0.8,
                    min_speech_duration=0.25,
                    max_speech_duration=25.0,
                    pre_speech_pad_duration=0.0,
                    cluster_threshold=0.5,
                    num_clusters=-1,
                    debug=False,
                )

        create_segmentation_runtime.assert_called_once_with(
            segmentation_dir / "model.onnx", num_threads=2
        )
        self.assertIs(runtimes.segmentation, segmentation_runtime)
        self.assertFalse(hasattr(runtimes, "diarization"))
        self.assertEqual(
            set(runtimes.resolved_files), {"asr", "vad", "speaker", "segmentation"}
        )

    def test_build_runtimes_can_skip_segmentation_for_vad_only(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        import models

        class ValidConfig:
            def validate(self):
                return True

        class FakeVadConfig(ValidConfig):
            def __init__(self):
                self.silero_vad = SimpleNamespace(window_size=512)

        fake_sherpa = SimpleNamespace(
            OfflineRecognizer=SimpleNamespace(from_paraformer=lambda **_kwargs: object()),
            VadModelConfig=FakeVadConfig,
            VoiceActivityDetector=lambda *_args, **_kwargs: object(),
            SpeakerEmbeddingExtractorConfig=ValidConfig,
            SpeakerEmbeddingExtractor=lambda _config: object(),
        )

        with TemporaryDirectory() as directory:
            root = Path(directory)
            asr_dir = root / "asr"
            vad_dir = root / "vad"
            speaker_dir = root / "speaker"
            for model_dir in (asr_dir, vad_dir, speaker_dir):
                model_dir.mkdir()
            (asr_dir / "model.int8.onnx").touch()
            (asr_dir / "tokens.txt").touch()
            (vad_dir / "silero_vad.onnx").touch()
            (speaker_dir / "nemo_en_titanet_large.onnx").touch()

            with (
                patch.object(models, "sherpa_onnx", fake_sherpa),
                patch.object(models, "create_segmentation_runtime") as create_segmentation_runtime,
            ):
                runtimes = models.build_runtimes(
                    asr_dir=asr_dir,
                    vad_dir=vad_dir,
                    speaker_dir=speaker_dir,
                    segmentation_dir=root / "missing-segmentation",
                    asr_num_threads=1,
                    speaker_num_threads=2,
                    vad_threshold=0.5,
                    min_silence_duration=0.8,
                    min_speech_duration=0.25,
                    max_speech_duration=25.0,
                    pre_speech_pad_duration=0.0,
                    cluster_threshold=0.5,
                    num_clusters=-1,
                    debug=False,
                    enable_segmentation=False,
                )

        create_segmentation_runtime.assert_not_called()
        self.assertIsNone(runtimes.segmentation)
        self.assertEqual(set(runtimes.resolved_files), {"asr", "vad", "speaker"})
