"""HTTP Whisper transcription with optional bilingual selection."""

from __future__ import annotations

from dataclasses import dataclass
import io
import time
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import requests
import soundfile as sf

from vad import SpeechSegment


SAMPLE_RATE = 16000
BILINGUAL_LANG_PROB_KEEP = 0.70
BILINGUAL_MIN_TEXT_CONFIDENCE = 0.50
WHISPER_RETRY_SLEEP_SECONDS = 10


@dataclass(frozen=True)
class WhisperResult:
    language: str
    asr_text: str
    lang_prob: float | None
    text_confidence: float | None


@dataclass(frozen=True)
class WhisperClientConfig:
    url: str
    timeout_ms: int = 30000
    languages: tuple[str, ...] = ()


def parse_whisper_languages(raw: str) -> tuple[str, ...]:
    languages = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if len(languages) > 2:
        raise ValueError("whisper languages accepts at most two language codes")
    if len(set(languages)) != len(languages):
        raise ValueError("whisper languages must not contain duplicate language codes")
    for language in languages:
        if not language.isalnum() or len(language) < 2:
            raise ValueError(f"Invalid language code: {language}")
    return tuple(languages)


def normalize_whisper_url(url: str) -> str:
    normalized = url.strip().rstrip("/")
    return normalized if normalized.endswith("/transcribe") else f"{normalized}/transcribe"


def parse_confidence(value: object) -> float | None:
    if value is None:
        return None
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    return confidence if np.isfinite(confidence) else None


def normalize_language_code(language: str) -> str:
    return language.strip().lower().replace("_", "-").split("-", 1)[0]


def parse_whisper_payload(payload: object) -> WhisperResult:
    if not isinstance(payload, dict):
        raise ValueError("Whisper response must be a JSON object")
    language = payload.get("language", payload.get("language_code", "unk"))
    return WhisperResult(
        language=str(language or "unk"),
        asr_text=str(payload.get("text") or ""),
        lang_prob=parse_confidence(payload.get("lang_prob")),
        text_confidence=parse_confidence(payload.get("confidence")),
    )


def segment_to_wav_bytes(samples: np.ndarray) -> bytes:
    output = io.BytesIO()
    sf.write(output, samples, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return output.getvalue()


def request_whisper(
    samples: np.ndarray,
    config: WhisperClientConfig,
    language_hint: str | None,
    *,
    poster: Any = requests.post,
    sleeper: Any = time.sleep,
) -> WhisperResult | None:
    endpoint = normalize_whisper_url(config.url)
    params: dict[str, str] = {
        "encode": "false",
        "task": "transcribe",
        "vad_filter": "false",
        "word_timestamps": "false",
        "output": "json",
    }
    if language_hint:
        params["language"] = language_hint
    wav_bytes = segment_to_wav_bytes(samples)
    for attempt in range(2):
        try:
            response = poster(
                endpoint,
                params=params,
                files={"audio_file": ("segment.wav", wav_bytes, "audio/wav")},
                timeout=config.timeout_ms / 1000.0,
            )
            response.raise_for_status()
            return parse_whisper_payload(response.json())
        except Exception:
            if attempt == 0:
                sleeper(WHISPER_RETRY_SLEEP_SECONDS)
    return None


def _apply_result(segment: SpeechSegment, result: WhisperResult | None, asr_language: str) -> None:
    if result is None:
        segment.asr_text = ""
        segment.asr_error = "whisper request failed"
        segment.asr_language = "unk"
        segment.text_confidence = None
        return
    segment.asr_text = result.asr_text
    segment.asr_error = None
    segment.asr_language = asr_language
    segment.text_confidence = result.text_confidence


def _mark_bilingual_validity(segment: SpeechSegment, languages: Sequence[str]) -> None:
    confidences = [
        segment.asr_candidates.get(language, {}).get("text_confidence")
        for language in languages
    ]
    segment.asr_valid = int(
        any(
            confidence is not None and confidence >= BILINGUAL_MIN_TEXT_CONFIDENCE
            for confidence in confidences
        )
    )
    if segment.asr_valid == 0 and not segment.asr_error:
        segment.asr_error = "bilingual text confidence below 0.50"


def transcribe_segment_with_whisper(
    segment: SpeechSegment,
    config: WhisperClientConfig,
    *,
    poster: Any = requests.post,
    sleeper: Any = time.sleep,
) -> None:
    languages = config.languages
    segment.asr_candidates = {}
    segment.asr_valid = 1

    def call(hint: str | None) -> WhisperResult | None:
        return request_whisper(
            segment.samples, config, hint, poster=poster, sleeper=sleeper
        )

    if len(languages) == 0:
        result = call(None)
        segment.whisper_language = result.language if result else "unk"
        segment.whisper_lang_prob = result.lang_prob if result else None
        asr_language = normalize_language_code(result.language) if result else "unk"
        _apply_result(segment, result, asr_language)
        if result is not None:
            segment.asr_candidates[asr_language] = {
                "text": result.asr_text,
                "text_confidence": result.text_confidence,
            }
        return

    if len(languages) == 1:
        result = call(languages[0])
        segment.whisper_language = languages[0]
        segment.whisper_lang_prob = result.lang_prob if result else None
        _apply_result(segment, result, languages[0])
        if result is not None:
            segment.asr_candidates[languages[0]] = {
                "text": result.asr_text,
                "text_confidence": result.text_confidence,
            }
        return

    auto_result = call(None)
    auto_language = normalize_language_code(auto_result.language) if auto_result else ""
    language_set = set(languages)
    segment.whisper_language = auto_result.language if auto_result is not None else "unk"
    segment.whisper_lang_prob = auto_result.lang_prob if auto_result is not None else None

    candidates: dict[str, WhisperResult | None] = {}
    if auto_result is not None and auto_language in language_set:
        candidates[auto_language] = auto_result
        other = next(language for language in languages if language != auto_language)
        candidates[other] = call(other)
    else:
        candidates = {language: call(language) for language in languages}

    segment.asr_candidates = {
        language: {
            "text": candidates[language].asr_text if candidates.get(language) else "",
            "text_confidence": candidates[language].text_confidence
            if candidates.get(language)
            else None,
        }
        for language in languages
    }

    if (
        auto_result is not None
        and auto_language in language_set
        and auto_result.lang_prob is not None
        and auto_result.lang_prob > BILINGUAL_LANG_PROB_KEEP
    ):
        selected_language = auto_language
        selected = candidates[auto_language]
    else:
        available = [
            (language, result) for language, result in candidates.items() if result is not None
        ]
        if not available:
            _apply_result(segment, None, "unk")
            _mark_bilingual_validity(segment, languages)
            return
        selected_language, selected = max(
            available,
            key=lambda item: item[1].text_confidence
            if item[1].text_confidence is not None
            else float("-inf"),
        )
    _apply_result(segment, selected, selected_language)
    _mark_bilingual_validity(segment, languages)


def transcribe_segments_with_whisper(
    segments: Iterable[SpeechSegment],
    config: WhisperClientConfig,
    *,
    poster: Any = requests.post,
    sleeper: Any = time.sleep,
) -> None:
    for segment in segments:
        transcribe_segment_with_whisper(segment, config, poster=poster, sleeper=sleeper)
