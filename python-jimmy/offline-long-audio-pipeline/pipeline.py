"""End-to-end local long-audio ASR and speaker-clustering orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import logging
from pathlib import Path
import time
from typing import Any

from asr import transcribe_segments
from audio_io import AudioFormat, load_audio, write_segment_wav
from diarization import finalize_vad_only_segments, resolve_final_segments
from models import PipelineRuntimes, build_runtimes
from output import create_run_directory, write_metadata, write_results
from speaker import assign_speaker_ids_with_centroids
from vad import collect_vad_segments
from whisper_asr import WhisperClientConfig, transcribe_segments_with_whisper

SEGMENTATION_MODE_VAD = "vad"
SEGMENTATION_MODE_VAD_PYANNOTE = "vad-pyannote"
SEGMENTATION_MODES = (SEGMENTATION_MODE_VAD, SEGMENTATION_MODE_VAD_PYANNOTE)

DEFAULT_ASR_DIR = Path(r"D:\TransAI\audio_model\full\audio_models\asr\paraformer-zh")
DEFAULT_VAD_DIR = Path(r"D:\TransAI\audio_model\full\audio_models\vad\silero_vad")
DEFAULT_SPEAKER_DIR = Path(r"D:\TransAI\audio_model\full\audio_models\speaker\nemo_en_titanet_large")
DEFAULT_SEGMENTATION_DIR = Path(
    r"D:\TransAI\audio_model\full\audio_models\speaker_segmentation\sherpa-onnx-pyannote-segmentation-3-0"
)
DEFAULT_OUTPUT_ROOT = Path(r"C:\Users\admin\Downloads\python语音Pipeline优化")
DEFAULT_WHISPER_URL = "http://159.135.196.85:31401/api/v1/unify-asr/whisper"
ASR_ENGINE_PARAFORMER = "paraformer"
ASR_ENGINE_WHISPER = "whisper"
ASR_ENGINES = (ASR_ENGINE_PARAFORMER, ASR_ENGINE_WHISPER)


@dataclass(frozen=True)
class PipelineConfig:
    audio: Path
    output_root: Path = DEFAULT_OUTPUT_ROOT
    audio_format: AudioFormat = field(default_factory=lambda: AudioFormat("pcm"))
    asr_dir: Path = DEFAULT_ASR_DIR
    vad_dir: Path = DEFAULT_VAD_DIR
    speaker_dir: Path = DEFAULT_SPEAKER_DIR
    segmentation_dir: Path = DEFAULT_SEGMENTATION_DIR
    asr_num_threads: int = 1
    speaker_num_threads: int = 2
    vad_threshold: float = 0.5
    min_silence_duration: float = 0.8
    min_speech_duration: float = 0.25
    max_speech_duration: float = 25.0
    pre_speech_pad_duration: float = 0.0
    cluster_threshold: float = 0.5
    num_clusters: int = 2
    asr_engine: str = ASR_ENGINE_PARAFORMER
    whisper_url: str = DEFAULT_WHISPER_URL
    whisper_languages: tuple[str, ...] = ()
    whisper_timeout_ms: int = 30000
    diarization_min_duration_on: float = 0.5
    diarization_min_duration_off: float = 0.5
    min_cluster_duration: float = 1.0
    centroid_assignment_similarity_threshold: float = 0.5
    save_segments: bool = False
    debug: bool = False
    segmentation_mode: str = SEGMENTATION_MODE_VAD_PYANNOTE
    run_label: str | None = None


@dataclass(frozen=True)
class PipelineResult:
    run_dir: Path
    segment_count: int
    rtf: float
    timings: dict[str, float]


def _configure_run_logger(run_dir: Path) -> tuple[logging.Logger, logging.Handler]:
    logger = logging.getLogger("offline_long_audio_pipeline")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger, handler


def _resolved_paths(runtimes: PipelineRuntimes) -> dict[str, dict[str, str | None]]:
    return {
        role: {"model": str(files.model), "tokens": str(files.tokens) if files.tokens else None}
        for role, files in runtimes.resolved_files.items()
    }


def _validate_config(config: PipelineConfig) -> None:
    if config.min_cluster_duration != 1.0:
        raise ValueError("min_cluster_duration is fixed at 1.0 seconds for this pipeline")
    if not -1.0 <= config.centroid_assignment_similarity_threshold <= 1.0:
        raise ValueError("centroid assignment similarity threshold must be in [-1.0, 1.0]")
    if config.diarization_min_duration_on < 0:
        raise ValueError("min_duration_on must be non-negative")
    if config.diarization_min_duration_off < 0:
        raise ValueError("min_duration_off must be non-negative")
    if config.segmentation_mode not in SEGMENTATION_MODES:
        raise ValueError(
            f"segmentation_mode must be one of {', '.join(SEGMENTATION_MODES)}"
        )
    if config.asr_engine not in ASR_ENGINES:
        raise ValueError(f"asr_engine must be one of {', '.join(ASR_ENGINES)}")
    if config.asr_engine == ASR_ENGINE_WHISPER and not config.whisper_url.strip():
        raise ValueError("whisper_url is required when asr_engine is whisper")
    if config.num_clusters == 0 or config.num_clusters < -1:
        raise ValueError("num_clusters must be -1 or a positive integer")
    if config.whisper_timeout_ms <= 0:
        raise ValueError("whisper_timeout_ms must be positive")


def _composition_counts(segments: list[Any]) -> dict[str, int]:
    labels = ("single_speaker", "overlapped_speakers", "unknown_activity")
    return {label: sum(segment.speaker_composition == label for segment in segments) for label in labels}


def _duration_class_counts(segments: list[Any]) -> dict[str, int]:
    return {label: sum(segment.duration_class == label for segment in segments) for label in ("long", "short")}


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """Execute local VAD, activity resolution, controlled clustering, and ASR."""
    _validate_config(config)
    run_dir = create_run_directory(
        config.output_root,
        config.audio.name,
        run_label=config.run_label or config.segmentation_mode,
    )
    logger, handler = _configure_run_logger(run_dir)
    timings: dict[str, float] = {}
    total_started = time.perf_counter()
    try:
        logger.info("stage=audio_io event=start")
        started = time.perf_counter()
        loaded = load_audio(config.audio, config.audio_format)
        timings["audio_io_seconds"] = time.perf_counter() - started
        logger.info(
            "stage=audio_io event=complete seconds=%.6f duration_ms=%s sample_rate=%s",
            timings["audio_io_seconds"],
            loaded.duration_ms,
            loaded.sample_rate,
        )

        logger.info("stage=model_load event=start")
        started = time.perf_counter()
        runtimes = build_runtimes(
            asr_dir=config.asr_dir,
            vad_dir=config.vad_dir,
            speaker_dir=config.speaker_dir,
            segmentation_dir=config.segmentation_dir,
            asr_num_threads=config.asr_num_threads,
            speaker_num_threads=config.speaker_num_threads,
            vad_threshold=config.vad_threshold,
            min_silence_duration=config.min_silence_duration,
            min_speech_duration=config.min_speech_duration,
            max_speech_duration=config.max_speech_duration,
            pre_speech_pad_duration=config.pre_speech_pad_duration,
            cluster_threshold=config.cluster_threshold,
            num_clusters=config.num_clusters,
            debug=config.debug,
            enable_segmentation=config.segmentation_mode == SEGMENTATION_MODE_VAD_PYANNOTE,
            enable_local_asr=config.asr_engine == ASR_ENGINE_PARAFORMER,
        )
        timings["model_load_seconds"] = time.perf_counter() - started
        logger.info("stage=model_load event=complete seconds=%.6f", timings["model_load_seconds"])

        logger.info("stage=vad event=start")
        started = time.perf_counter()
        raw_vad_segments = collect_vad_segments(
            runtimes.vad, loaded.samples, runtimes.vad_window_size, loaded.sample_rate
        )
        timings["vad_seconds"] = time.perf_counter() - started
        logger.info("stage=vad event=complete seconds=%.6f raw_segments=%s", timings["vad_seconds"], len(raw_vad_segments))

        logger.info("stage=segmentation event=start mode=%s", config.segmentation_mode)
        started = time.perf_counter()
        if config.segmentation_mode == SEGMENTATION_MODE_VAD:
            activity = []
            timings["segmentation_seconds"] = 0.0
            logger.info("stage=segmentation event=skipped mode=vad")
        else:
            if runtimes.segmentation is None:
                raise RuntimeError("pyannote segmentation runtime is required for vad-pyannote mode")
            activity = runtimes.segmentation.infer_speaker_count_spans(loaded.samples)
            timings["segmentation_seconds"] = time.perf_counter() - started
            logger.info(
                "stage=segmentation event=complete seconds=%.6f activity_spans=%s",
                timings["segmentation_seconds"],
                len(activity),
            )

        logger.info("stage=timeline_resolution event=start")
        started = time.perf_counter()
        if config.segmentation_mode == SEGMENTATION_MODE_VAD:
            segments, resolution_stats = finalize_vad_only_segments(raw_vad_segments)
        else:
            segments, resolution_stats = resolve_final_segments(
                raw_vad_segments,
                activity,
                loaded.samples,
                loaded.sample_rate,
                min_duration_on=config.diarization_min_duration_on,
                min_duration_off=config.diarization_min_duration_off,
            )
        timings["timeline_resolution_seconds"] = time.perf_counter() - started
        logger.info(
            "stage=timeline_resolution event=complete seconds=%.6f final_segments=%s true_overlap=%s",
            timings["timeline_resolution_seconds"],
            len(segments),
            resolution_stats.true_overlap_count,
        )

        if config.save_segments:
            for segment in segments:
                write_segment_wav(run_dir / "segments" / f"{segment.segment_id}.wav", segment.samples)

        def run_speaker() -> tuple[int, int, int]:
            logger.info("stage=speaker event=start segments=%s", len(segments))
            started_local = time.perf_counter()
            counts = assign_speaker_ids_with_centroids(
                runtimes.extractor,
                segments,
                cluster_threshold=config.cluster_threshold,
                num_clusters=config.num_clusters,
                assignment_similarity_threshold=config.centroid_assignment_similarity_threshold,
            )
            timings["speaker_seconds"] = time.perf_counter() - started_local
            logger.info(
                "stage=speaker event=complete seconds=%.6f segments=%s errors=%s centroid_assigned=%s unknown_excluded=%s",
                timings["speaker_seconds"], len(segments), counts[0], counts[1], counts[2],
            )
            return counts

        def run_asr() -> int:
            logger.info("stage=asr event=start engine=%s segments=%s", config.asr_engine, len(segments))
            started_local = time.perf_counter()
            if config.asr_engine == ASR_ENGINE_WHISPER:
                transcribe_segments_with_whisper(
                    segments,
                    WhisperClientConfig(
                        url=config.whisper_url,
                        timeout_ms=config.whisper_timeout_ms,
                        languages=config.whisper_languages,
                    ),
                )
            else:
                if runtimes.recognizer is None:
                    raise RuntimeError("local Paraformer recognizer is required")
                transcribe_segments(runtimes.recognizer, segments, loaded.sample_rate)
            timings["asr_seconds"] = time.perf_counter() - started_local
            error_count = sum(segment.asr_error is not None for segment in segments)
            logger.info(
                "stage=asr event=complete seconds=%.6f segments=%s errors=%s",
                timings["asr_seconds"], len(segments), error_count,
            )
            return error_count

        if config.asr_engine == ASR_ENGINE_WHISPER:
            asr_error_count = run_asr()
            embedding_error_count, centroid_assigned, unknown_excluded = run_speaker()
        else:
            embedding_error_count, centroid_assigned, unknown_excluded = run_speaker()
            asr_error_count = run_asr()

        timings["total_seconds"] = time.perf_counter() - total_started
        audio_seconds = max(loaded.duration_ms / 1000.0, 0.001)
        timings["rtf"] = timings["total_seconds"] / audio_seconds

        logger.info("stage=output event=start segments=%s", len(segments))
        started = time.perf_counter()
        write_results(run_dir, config.audio.name, loaded.duration_ms, segments, timings)
        timings["output_seconds"] = time.perf_counter() - started
        eligible_count = sum(segment.is_cluster_eligible for segment in segments)
        clustered_speaker_ids = {
            segment.speaker_id
            for segment in segments
            if segment.is_cluster_eligible and segment.speaker_id not in {"unknown", "-"}
        }
        write_metadata(
            run_dir,
            {
                "audio": str(config.audio),
                "audio_duration_ms": loaded.duration_ms,
                "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
                "resolved_models": _resolved_paths(runtimes),
                "segment_count": len(segments),
                "raw_vad_segment_count": len(raw_vad_segments),
                "final_asr_segment_count": len(segments),
                "speaker_composition_counts": _composition_counts(segments),
                "duration_class_counts": _duration_class_counts(segments),
                "cluster_eligible_segment_count": eligible_count,
                "excluded_segment_count": len(segments) - eligible_count,
                "final_clustered_speaker_count": len(clustered_speaker_ids),
                "centroid_assigned_excluded_segment_count": centroid_assigned,
                "unknown_excluded_segment_count": unknown_excluded,
                "true_overlap_count": resolution_stats.true_overlap_count,
                "asr_error_count": asr_error_count,
                "embedding_error_count": embedding_error_count,
                "timings": timings,
            },
        )
        logger.info("stage=output event=complete seconds=%.6f segments=%s", timings["output_seconds"], len(segments))
        return PipelineResult(run_dir, len(segments), timings["rtf"], timings)
    finally:
        logger.info("pipeline finished")
        logger.removeHandler(handler)
        handler.close()
