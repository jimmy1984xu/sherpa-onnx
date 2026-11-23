"""Local sherpa-onnx model discovery and runtime construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import sherpa_onnx

from segmentation import SegmentationRuntime, create_segmentation_runtime


@dataclass(frozen=True)
class ResolvedModelFiles:
    model: Path
    tokens: Path | None = None


@dataclass(frozen=True)
class PipelineRuntimes:
    recognizer: object
    vad: object
    vad_window_size: int
    extractor: object
    segmentation: SegmentationRuntime | None
    resolved_files: dict[str, ResolvedModelFiles]


_MODEL_FILES: dict[str, tuple[str, ...]] = {
    "asr": ("model.int8.onnx", "tokens.txt"),
    "vad": ("silero_vad.onnx",),
    "speaker": ("nemo_en_titanet_large.onnx",),
    "segmentation": ("model.onnx",),
}


def resolve_model_files(root: Path, role: str) -> ResolvedModelFiles:
    """Resolve one local model role to its mandatory files."""
    if role not in _MODEL_FILES:
        raise ValueError(f"Unsupported model role: {role}")
    if not root.is_dir():
        raise FileNotFoundError(f"{role} model directory not found: {root}")

    found: list[Path] = []
    for filename in _MODEL_FILES[role]:
        candidate = root / filename
        if not candidate.is_file():
            raise FileNotFoundError(f"Missing {role} model file {filename} under {root}")
        found.append(candidate)

    return ResolvedModelFiles(model=found[0], tokens=found[1] if len(found) > 1 else None)


def configure_pre_speech_pad(vad_config: object, duration: float) -> None:
    """Configure optional VAD pre-speech padding across sherpa bindings."""
    if duration < 0:
        raise ValueError("Pre-speech padding must be non-negative")
    if hasattr(vad_config, "pre_speech_pad_duration"):
        vad_config.pre_speech_pad_duration = duration
    elif duration != 0:
        raise ValueError(
            "This sherpa-onnx runtime does not support pre-speech padding; use 0.0"
        )


def build_runtimes(
    *,
    asr_dir: Path,
    vad_dir: Path,
    speaker_dir: Path,
    segmentation_dir: Path,
    asr_num_threads: int,
    speaker_num_threads: int,
    vad_threshold: float,
    min_silence_duration: float,
    min_speech_duration: float,
    max_speech_duration: float,
    pre_speech_pad_duration: float,
    cluster_threshold: float,
    num_clusters: int,
    debug: bool,
    enable_segmentation: bool = True,
    enable_local_asr: bool = True,
) -> PipelineRuntimes:
    """Validate local assets then build the three local runtime objects."""
    asr_files = resolve_model_files(asr_dir, "asr") if enable_local_asr else None
    vad_files = resolve_model_files(vad_dir, "vad")
    speaker_files = resolve_model_files(speaker_dir, "speaker")
    segmentation_files = (
        resolve_model_files(segmentation_dir, "segmentation") if enable_segmentation else None
    )

    recognizer = None
    if asr_files is not None:
        recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
            paraformer=str(asr_files.model),
            tokens=str(asr_files.tokens),
            num_threads=asr_num_threads,
            sample_rate=16000,
            provider="cpu",
            debug=debug,
        )

    vad_config = sherpa_onnx.VadModelConfig()
    vad_config.sample_rate = 16000
    vad_config.silero_vad.model = str(vad_files.model)
    vad_config.silero_vad.threshold = vad_threshold
    vad_config.silero_vad.min_silence_duration = min_silence_duration
    vad_config.silero_vad.min_speech_duration = min_speech_duration
    vad_config.silero_vad.max_speech_duration = max_speech_duration
    configure_pre_speech_pad(vad_config, pre_speech_pad_duration)
    vad_config.debug = debug
    if not vad_config.validate():
        raise ValueError("Invalid VAD configuration")
    vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=3600)

    extractor_config = sherpa_onnx.SpeakerEmbeddingExtractorConfig()
    extractor_config.model = str(speaker_files.model)
    extractor_config.num_threads = speaker_num_threads
    extractor_config.provider = "cpu"
    extractor_config.debug = debug
    if not extractor_config.validate():
        raise ValueError("Invalid speaker embedding configuration")
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(extractor_config)

    segmentation = (
        create_segmentation_runtime(segmentation_files.model, num_threads=speaker_num_threads)
        if segmentation_files is not None
        else None
    )
    resolved_files = {
        "vad": vad_files,
        "speaker": speaker_files,
    }
    if asr_files is not None:
        resolved_files["asr"] = asr_files
    if segmentation_files is not None:
        resolved_files["segmentation"] = segmentation_files

    return PipelineRuntimes(
        recognizer=recognizer,
        vad=vad,
        vad_window_size=int(vad_config.silero_vad.window_size),
        extractor=extractor,
        segmentation=segmentation,
        resolved_files=resolved_files,
    )

