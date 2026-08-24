#!/usr/bin/env python3

"""使用 Silero VAD、Whisper 和 FastClustering 处理单个 WAV 文件。

本脚本是独立脚本，不导入或执行 python-jimmy 中的其他脚本。

双语规则（--whisper-languages en,hi）：
1. 每个 VAD 片段先不指定语言调用 Whisper，得到原始 language、langProb、
   ASR 文本和 textConfidence。
2. 如果原始 language 属于指定双语范围，则自动结果作为该语种候选，只再指定
   另一种语言调用 Whisper。
3. 如果原始 language 不属于指定双语范围，则分别指定两种语言调用 Whisper。
4. 原始 language 属于双语范围且 langProb > 0.70 时，选择自动结果；否则选择
   两路候选中 textConfidence 更高的结果。
5. JSON 的 whisperLanguage、whisperLangProb 始终记录首次自动调用的原始结果；
   asrLanguage 表示最终 asrText 对应的候选语种。

片段无效规则（仅双语模式）：
1. 两路候选的 textConfidence 都低于 0.50，或两路均失败时，valid=0。
2. 其余情况 valid=1。valid=0 的片段不会提取声纹、不参与聚类，speakerId=-。

主要参数：
- --audio：输入 WAV；非 16 kHz 音频会在内存中重采样为 16 kHz。
- --vad-model：Silero VAD ONNX 模型路径。
- --speaker-model：声纹 ONNX 模型路径；线程数固定为 2，CPU 推理，最多取 10 秒。
- --whisper-url：Whisper 服务地址；--whisper-timeout-ms 默认 30000ms，失败后
  等待 10 秒并重试一次。
- --whisper-languages：不设置为自动模式；一个语言为强制该语言；两个语言启用
  上述双语规则。
- --output：输出文件夹；不设置时，JSON 和 TXT 输出到输入音频同目录，文件名固定
  为“输入音频文件名.json”和“输入音频文件名.txt”。
- --vad-threshold、--min-silence-duration、--min-speech-duration、
  --max-speech-duration：Silero VAD 切分参数。
- --cluster-threshold、--num-clusters：FastClustering 聚类参数。

Example:
  python python-jimmy/k2-wav-vad-whisper-speaker-cluster.py ^
    --audio input.wav ^
    --vad-model D:/models/silero_vad.onnx ^
    --speaker-model D:/models/speaker_embedding.onnx ^
    --whisper-url http://host/api/whisper
"""

import argparse
import io
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import requests
import sherpa_onnx
import soundfile as sf


SAMPLE_RATE = 16000
SPEAKER_NUM_THREADS = 2
SPEAKER_PROVIDER = "cpu"
SPEAKER_MAX_SAMPLES = 10000 * SAMPLE_RATE // 1000
# A segment is invalid when it matches any rule below. Keep these constants
# together so the policy can be adjusted without changing the pipeline.
INVALID_SHORT_DURATION_MS = 600
INVALID_SHORT_TEXT_CONFIDENCE = 0.60
INVALID_UNKNOWN_LANGUAGE_TEXT_CONFIDENCE = 0.70
BILINGUAL_MIN_TEXT_CONFIDENCE = 0.50
SUMMARY_OUTPUT_HEADER = "segmentId speakerId asrLanguage asrText\n"


@dataclass
class Segment:
    offset_ms: int
    duration_ms: int
    samples: np.ndarray
    segment_id: str = ""
    speaker_id: str = "unk"
    language: str = "unk"
    lang_prob: Optional[float] = None
    lang_prob_invalid: bool = False
    asr_language: str = "unk"
    text_confidence: Optional[float] = None
    asr_text: str = ""
    bilingual_candidate_confidences: Optional[Tuple[Optional[float], Optional[float]]] = None
    asr_candidates: Dict[str, Tuple[str, Optional[float]]] = field(default_factory=dict)
    valid: int = 1


@dataclass
class WhisperResult:
    language: str
    asr_text: str
    lang_prob: Optional[float]
    text_confidence: Optional[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run Silero VAD, Whisper ASR, speaker embeddings, and FastClustering on one 16 kHz WAV.",
    )
    parser.add_argument("--audio", required=True, help="Input 16 kHz WAV file")
    parser.add_argument("--vad-model", required=True, help="Path to silero_vad.onnx")
    parser.add_argument("--speaker-model", required=True, help="Path to speaker embedding ONNX model")
    parser.add_argument("--whisper-url", required=True, help="Whisper base URL or /transcribe endpoint")
    parser.add_argument("--output", default="", help="Output directory for the TXT and JSON results")
    parser.add_argument("--segments-dir", default="", help="Optional directory for VAD segment WAV files")
    parser.add_argument("--vad-threshold", type=float, default=0.5, help="Silero VAD threshold")
    parser.add_argument("--min-silence-duration", type=float, default=1.0, help="Silero minimum silence duration in seconds")
    parser.add_argument("--min-speech-duration", type=float, default=0.25, help="Silero minimum speech duration in seconds")
    parser.add_argument("--max-speech-duration", type=float, default=25.0, help="Silero maximum speech duration in seconds")
    parser.add_argument(
        "--whisper-languages",
        default="",
        help="Optional comma-separated language list: one language or two languages",
    )
    parser.add_argument("--whisper-timeout-ms", type=int, default=30000, help="Whisper request timeout in milliseconds")
    parser.add_argument("--cluster-threshold", type=float, default=0.5, help="FastClustering threshold")
    parser.add_argument("--num-clusters", type=int, default=-1, help="Fixed cluster count; positive values override --cluster-threshold")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> Tuple[Path, Path, Path, Path]:
    audio_path = Path(args.audio)
    vad_model_path = Path(args.vad_model)
    speaker_model_path = Path(args.speaker_model)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Input WAV not found: {audio_path}")
    if audio_path.suffix.lower() != ".wav":
        raise ValueError(f"Input must be a WAV file: {audio_path}")
    if not vad_model_path.is_file():
        raise FileNotFoundError(f"Silero VAD model not found: {vad_model_path}")
    if not speaker_model_path.is_file():
        raise FileNotFoundError(f"Speaker model not found: {speaker_model_path}")
    if not args.whisper_url.strip():
        raise ValueError("--whisper-url must not be empty")
    languages = [item.strip().lower() for item in args.whisper_languages.split(",") if item.strip()]
    if len(languages) > 2:
        raise ValueError("--whisper-languages accepts at most two language codes")
    if len(set(languages)) != len(languages):
        raise ValueError("--whisper-languages must not contain duplicate language codes")
    for language in languages:
        if not language.isalnum() or len(language) < 2:
            raise ValueError(f"Invalid language code in --whisper-languages: {language}")
    args.whisper_languages = tuple(languages)
    if not 0.0 <= args.vad_threshold <= 1.0:
        raise ValueError("--vad-threshold must be in [0, 1]")
    if args.min_silence_duration < 0.0:
        raise ValueError("--min-silence-duration must be >= 0")
    if args.min_speech_duration < 0.0:
        raise ValueError("--min-speech-duration must be >= 0")
    if args.max_speech_duration <= 0.0:
        raise ValueError("--max-speech-duration must be > 0")
    if args.whisper_timeout_ms <= 0:
        raise ValueError("--whisper-timeout-ms must be > 0")
    if args.num_clusters == 0 or args.num_clusters < -1:
        raise ValueError("--num-clusters must be -1 or a positive integer")
    if not 0.0 <= args.cluster_threshold <= 1.0:
        raise ValueError("--cluster-threshold must be in [0, 1]")

    output_dir = Path(args.output) if args.output else audio_path.parent
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"--output must be a directory: {output_dir}")
    return audio_path, vad_model_path, speaker_model_path, output_dir


def resample_linear(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample mono float32 audio without an external media tool."""
    if source_rate == target_rate:
        return np.ascontiguousarray(samples, dtype=np.float32)
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError(f"Invalid sample rate conversion: {source_rate} -> {target_rate}")
    target_length = int(round(samples.size * target_rate / source_rate))
    if target_length <= 0:
        return np.zeros(0, dtype=np.float32)
    source_positions = np.arange(samples.size, dtype=np.float64) / source_rate
    target_positions = np.arange(target_length, dtype=np.float64) / target_rate
    return np.interp(
        target_positions,
        source_positions,
        samples.astype(np.float64, copy=False),
    ).astype(np.float32, copy=False)


def load_wav_16k_mono(audio_path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(str(audio_path), always_2d=True, dtype="float32")
    if audio.shape[0] == 0:
        raise ValueError(f"Input WAV is empty: {audio_path}")
    mono = np.ascontiguousarray(audio.mean(axis=1), dtype=np.float32)
    if sample_rate != SAMPLE_RATE:
        print(f"      resampling {sample_rate} Hz -> {SAMPLE_RATE} Hz in memory")
        mono = resample_linear(mono, sample_rate, SAMPLE_RATE)
    if mono.size == 0:
        raise ValueError(f"Input WAV has no samples after resampling: {audio_path}")
    return mono


def build_vad(args: argparse.Namespace) -> Tuple[object, int]:
    config = sherpa_onnx.VadModelConfig()
    config.sample_rate = SAMPLE_RATE
    config.silero_vad.model = args.vad_model
    config.silero_vad.threshold = args.vad_threshold
    config.silero_vad.min_silence_duration = args.min_silence_duration
    config.silero_vad.min_speech_duration = args.min_speech_duration
    config.silero_vad.max_speech_duration = args.max_speech_duration
    if not config.validate():
        raise ValueError("Invalid Silero VAD configuration")
    vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=3600)
    return vad, int(config.silero_vad.window_size)


def save_segment(path: Path, samples: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(segment_to_wav_bytes(samples))


def append_available_segments(vad: object, segments: List[Segment], audio_stem: str, segments_dir: Optional[Path]) -> None:
    while not vad.empty():
        vad_segment = vad.front
        samples = np.ascontiguousarray(np.asarray(vad_segment.samples, dtype=np.float32))
        offset_ms = int(vad_segment.start * 1000 / SAMPLE_RATE)
        duration_ms = int(samples.size * 1000 / SAMPLE_RATE)
        segment_id = f"{len(segments) + 1}_{offset_ms}_{duration_ms}"
        segment = Segment(
            offset_ms=offset_ms,
            duration_ms=duration_ms,
            samples=samples,
            segment_id=segment_id,
        )
        segments.append(segment)
        if segments_dir is not None:
            save_segment(segments_dir / f"{segment_id}.wav", samples)
        vad.pop()


def cut_speech_segments(waveform: np.ndarray, args: argparse.Namespace, audio_stem: str) -> List[Segment]:
    vad, window_size = build_vad(args)
    segments_dir = Path(args.segments_dir) if args.segments_dir else None
    segments: List[Segment] = []
    for start in range(0, waveform.size, window_size):
        vad.accept_waveform(waveform[start : start + window_size])
        append_available_segments(vad, segments, audio_stem, segments_dir)
    vad.flush()
    append_available_segments(vad, segments, audio_stem, segments_dir)
    for index, segment in enumerate(segments, start=1):
        segment.segment_id = f"{index}_{segment.offset_ms}_{segment.duration_ms}"
    return segments


def segment_to_wav_bytes(samples: np.ndarray) -> bytes:
    output = io.BytesIO()
    sf.write(output, samples, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return output.getvalue()


def normalize_whisper_url(url: str) -> str:
    normalized = url.strip().rstrip("/")
    return normalized if normalized.endswith("/transcribe") else f"{normalized}/transcribe"


def parse_whisper_response(
    response: requests.Response,
) -> WhisperResult:
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Whisper response must be a JSON object")
    language = payload.get("language", payload.get("language_code", "unk"))
    text = payload.get("text", "")
    lang_prob = parse_confidence(payload.get("lang_prob"))
    text_confidence = parse_confidence(payload.get("confidence"))
    return WhisperResult(
        language=str(language or "unk"),
        asr_text=str(text or ""),
        lang_prob=lang_prob,
        text_confidence=text_confidence,
    )


def parse_confidence(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    return confidence if np.isfinite(confidence) else None


def normalize_language_code(language: str) -> str:
    return language.strip().lower().replace("_", "-").split("-", 1)[0]


def request_whisper(
    segment: Segment,
    args: argparse.Namespace,
    language_hint: Optional[str],
) -> Optional[WhisperResult]:
    endpoint = normalize_whisper_url(args.whisper_url)
    params: Dict[str, str] = {
        "encode": "false",
        "task": "transcribe",
        "vad_filter": "false",
        "word_timestamps": "false",
        "output": "json",
    }
    if language_hint:
        params["language"] = language_hint

    wav_bytes = segment_to_wav_bytes(segment.samples)
    for attempt in range(2):
        try:
            response = requests.post(
                endpoint,
                params=params,
                files={"audio_file": ("segment.wav", wav_bytes, "audio/wav")},
                timeout=args.whisper_timeout_ms / 1000.0,
            )
            response.raise_for_status()
            return parse_whisper_response(response)
        except Exception as error:
            if attempt == 0:
                route = language_hint or "auto"
                print(
                    f"  Whisper failed for offset={segment.offset_ms}ms route={route}; "
                    f"retrying in 10s: {error}"
                )
                time.sleep(10)
            else:
                route = language_hint or "auto"
                print(
                    f"  Whisper failed for offset={segment.offset_ms}ms route={route} "
                    f"after retry: {error}"
                )

    return None


def transcribe_segment(segment: Segment, args: argparse.Namespace) -> None:
    languages = args.whisper_languages
    segment.lang_prob_invalid = False
    segment.asr_language = "unk"
    segment.bilingual_candidate_confidences = None
    segment.asr_candidates = {}

    if len(languages) == 0:
        result = request_whisper(segment, args, language_hint=None)
        if result is None:
            segment.language = "unk"
            segment.lang_prob = None
            segment.text_confidence = None
            segment.asr_text = ""
            return
        segment.language = result.language
        segment.lang_prob = result.lang_prob
        segment.asr_language = normalize_language_code(result.language)
        segment.text_confidence = result.text_confidence
        segment.asr_text = result.asr_text
        segment.asr_candidates[segment.asr_language] = (
            result.asr_text,
            result.text_confidence,
        )
        return

    if len(languages) == 1:
        result = request_whisper(segment, args, language_hint=languages[0])
        if result is None:
            segment.language = "unk"
            segment.lang_prob = None
            segment.text_confidence = None
            segment.asr_text = ""
            return
        segment.language = languages[0]
        segment.lang_prob = result.lang_prob
        segment.asr_language = languages[0]
        segment.text_confidence = result.text_confidence
        segment.asr_text = result.asr_text
        segment.asr_candidates[languages[0]] = (result.asr_text, result.text_confidence)
        return

    auto_result = request_whisper(segment, args, language_hint=None)
    auto_language = normalize_language_code(auto_result.language) if auto_result else ""
    language_set = set(languages)
    # In bilingual mode, language/langProb always report the raw automatic
    # Whisper result, even when the selected transcript uses the other route.
    segment.language = auto_result.language if auto_result is not None else "unk"
    segment.lang_prob = auto_result.lang_prob if auto_result is not None else None

    candidates: Dict[str, Optional[WhisperResult]] = {}
    if auto_result is not None and auto_language in language_set:
        candidates[auto_language] = auto_result
        other_language = next(language for language in languages if language != auto_language)
        candidates[other_language] = request_whisper(segment, args, language_hint=other_language)
    else:
        candidates = {
            language: request_whisper(segment, args, language_hint=language)
            for language in languages
        }

    segment.bilingual_candidate_confidences = tuple(
        candidates[language].text_confidence if candidates.get(language) is not None else None
        for language in languages
    )
    segment.asr_candidates = {
        language: (
            candidates[language].asr_text,
            candidates[language].text_confidence,
        )
        if candidates.get(language) is not None
        else ("", None)
        for language in languages
    }
    if (
        auto_result is not None
        and auto_language in language_set
        and auto_result.lang_prob is not None
        and auto_result.lang_prob > 0.70
    ):
        selected_language = auto_language
        selected_result = candidates[auto_language]
    else:
        available_results = [
            (language, result) for language, result in candidates.items() if result is not None
        ]
        if not available_results:
            segment.asr_language = "unk"
            segment.text_confidence = None
            segment.asr_text = ""
            return
        selected_language, selected_result = max(
            available_results,
            key=lambda item: item[1].text_confidence
            if item[1].text_confidence is not None
            else float("-inf"),
        )

    if selected_result is None:
        segment.asr_language = "unk"
        segment.text_confidence = None
        segment.asr_text = ""
        return
    segment.asr_language = selected_language
    segment.text_confidence = selected_result.text_confidence
    segment.asr_text = selected_result.asr_text


def build_speaker_extractor(model_path: Path) -> object:
    config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=str(model_path),
        num_threads=SPEAKER_NUM_THREADS,
        debug=False,
        provider=SPEAKER_PROVIDER,
    )
    if not config.validate():
        raise ValueError("Invalid speaker embedding configuration")
    return sherpa_onnx.SpeakerEmbeddingExtractor(config)


def extract_embedding(extractor: object, samples: np.ndarray) -> Optional[np.ndarray]:
    stream = extractor.create_stream()
    stream.accept_waveform(
        sample_rate=SAMPLE_RATE,
        waveform=np.ascontiguousarray(samples[:SPEAKER_MAX_SAMPLES]),
    )
    stream.input_finished()
    if not extractor.is_ready(stream):
        return None
    embedding = np.asarray(extractor.compute(stream), dtype=np.float32)
    return embedding if embedding.size else None


def assign_speaker_ids(segments: Sequence[Segment], speaker_model_path: Path, args: argparse.Namespace) -> None:
    valid_indices = [index for index, segment in enumerate(segments) if segment.valid == 1]
    for segment in segments:
        if segment.valid == 0:
            segment.speaker_id = "-"
    if not valid_indices:
        print("  No valid segments available; all speaker IDs are -")
        return

    extractor = build_speaker_extractor(speaker_model_path)
    embeddings: List[np.ndarray] = []
    segment_indices: List[int] = []
    print(
        f"  Skipping {len(segments) - len(valid_indices)} invalid segments for speaker extraction"
    )
    for index in valid_indices:
        segment = segments[index]
        try:
            embedding = extract_embedding(extractor, segment.samples)
        except Exception as error:
            print(f"  Speaker embedding failed for offset={segment.offset_ms}ms: {error}")
            embedding = None
        if embedding is not None:
            embeddings.append(embedding)
            segment_indices.append(index)

    if not embeddings:
        print("  No speaker embeddings available; all speaker IDs are unk")
        return

    embedding_matrix = np.ascontiguousarray(np.stack(embeddings), dtype=np.float32)
    if args.num_clusters > len(embeddings):
        raise ValueError(
            f"--num-clusters={args.num_clusters} exceeds the number of "
            f"successful speaker embeddings ({len(embeddings)})"
        )
    if args.num_clusters > 0:
        clustering_config = sherpa_onnx.FastClusteringConfig(num_clusters=args.num_clusters)
    else:
        clustering_config = sherpa_onnx.FastClusteringConfig(threshold=args.cluster_threshold)
    if not clustering_config.validate():
        raise ValueError("Invalid FastClustering configuration")
    labels = list(sherpa_onnx.FastClustering(clustering_config)(embedding_matrix))
    if len(labels) != len(segment_indices):
        raise RuntimeError("FastClustering returned an unexpected number of labels")

    label_to_speaker = {label: f"S{index}" for index, label in enumerate(sorted(set(labels)), start=1)}
    for segment_index, label in zip(segment_indices, labels):
        segments[segment_index].speaker_id = label_to_speaker[label]


def output_text_field(value: str) -> str:
    return " ".join(value.replace("\t", " ").splitlines())


def format_confidence(value: Optional[float]) -> str:
    return "unk" if value is None else f"{value:.3f}"


def format_lang_prob(segment: Segment) -> str:
    return "-" if segment.lang_prob_invalid else format_confidence(segment.lang_prob)


def update_segment_validity(segment: Segment) -> None:
    """Apply hard-coded invalid rules; all remaining segments are valid."""
    if segment.bilingual_candidate_confidences is not None:
        segment.valid = int(
            any(
                confidence is not None and confidence >= BILINGUAL_MIN_TEXT_CONFIDENCE
                for confidence in segment.bilingual_candidate_confidences
            )
        )
        return

    confidence = segment.text_confidence
    if segment.language == "unk" or not segment.asr_text.strip() or confidence is None:
        segment.valid = 0
        return
    if (
        segment.duration_ms < INVALID_SHORT_DURATION_MS
        and confidence < INVALID_SHORT_TEXT_CONFIDENCE
    ):
        segment.valid = 0
        return
    if (
        segment.lang_prob_invalid
        and confidence < INVALID_UNKNOWN_LANGUAGE_TEXT_CONFIDENCE
    ):
        segment.valid = 0
        return
    segment.valid = 1


def write_detailed_results(
    output_path: Path,
    audio_path: Path,
    segments: Sequence[Segment],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "audioFile": audio_path.name,
        "segments": [
            {
                "segmentId": segment.segment_id,
                "offsetMs": segment.offset_ms,
                "durationMs": segment.duration_ms,
                "whisperLanguage": segment.language,
                "whisperLangProb": segment.lang_prob,
                "asrCandidates": {
                    language: {
                        "text": text,
                        "textConfidence": confidence,
                    }
                    for language, (text, confidence) in segment.asr_candidates.items()
                },
                "asrLanguage": segment.asr_language,
                "textConfidence": segment.text_confidence,
                "asrText": segment.asr_text,
                "speakerId": segment.speaker_id,
                "valid": segment.valid,
            }
            for segment in segments
        ],
    }
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(payload, output, ensure_ascii=False, indent=2)
        output.write("\n")


def write_summary_results(output_path: Path, segments: Sequence[Segment]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        output.write(SUMMARY_OUTPUT_HEADER)
        for segment in segments:
            if segment.valid == 1:
                output.write(
                    f"{output_text_field(segment.segment_id)}"
                    f" {output_text_field(segment.speaker_id)}"
                    f" {output_text_field(segment.asr_language)}"
                    f" {output_text_field(segment.asr_text)}\n"
                )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    args = parse_args()
    audio_path, _, speaker_model_path, output_dir = validate_args(args)
    summary_path = output_dir / f"{audio_path.stem}.txt"
    detailed_path = output_dir / f"{audio_path.stem}.json"

    print("[1/4] Loading 16 kHz WAV")
    waveform = load_wav_16k_mono(audio_path)
    print(f"      samples={waveform.size}, duration={waveform.size / SAMPLE_RATE:.2f}s")

    print("[2/4] Running Silero VAD")
    segments = cut_speech_segments(waveform, args, audio_path.stem)
    print(f"      segments={len(segments)}")

    print("[3/4] Calling Whisper for each segment")
    for index, segment in enumerate(segments, start=1):
        transcribe_segment(segment, args)
        update_segment_validity(segment)
        candidate_confidences = ""
        if segment.bilingual_candidate_confidences is not None:
            candidate_confidences = " ".join(
                f"{language}TextConfidence={format_confidence(confidence)}"
                for language, confidence in zip(
                    args.whisper_languages,
                    segment.bilingual_candidate_confidences,
                )
            )
        print(
            f"      {index}/{len(segments)} segmentId={segment.segment_id} "
            f"offset={segment.offset_ms}ms "
            f"duration={segment.duration_ms}ms language={segment.language} "
            f"langProb={format_lang_prob(segment)} "
            f"asrLanguage={segment.asr_language} "
            f"textConfidence={format_confidence(segment.text_confidence)} "
            f"valid={segment.valid} {candidate_confidences} "
            f"asrText={output_text_field(segment.asr_text)}"
        )

    print("[4/4] Extracting speaker embeddings and clustering")
    assign_speaker_ids(segments, speaker_model_path, args)
    write_detailed_results(detailed_path, audio_path, segments)
    write_summary_results(summary_path, segments)
    print(f"Done. Wrote {len(segments)} segments to {detailed_path}")
    print(f"Done. Wrote valid segment summary to {summary_path}")


if __name__ == "__main__":
    main()
