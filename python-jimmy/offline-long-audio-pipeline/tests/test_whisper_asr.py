import sys
import unittest
from pathlib import Path

import numpy as np

PIPELINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_DIR))

from vad import SpeechSegment
from whisper_asr import WhisperClientConfig, parse_whisper_languages, transcribe_segment_with_whisper


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class WhisperAsrTest(unittest.TestCase):
    def test_parse_whisper_languages_accepts_bilingual_pair(self):
        self.assertEqual(parse_whisper_languages("en,hi"), ("en", "hi"))

    def test_bilingual_keeps_auto_result_when_lang_prob_is_high(self):
        segment = SpeechSegment(1, 0, 1000, np.zeros(16000, dtype=np.float32))
        calls = []

        def poster(url, params=None, files=None, timeout=None):
            calls.append(params.get("language") if params else None)
            if params and params.get("language") == "en":
                return _FakeResponse(
                    {"language": "en", "text": "hello", "lang_prob": 0.4, "confidence": 0.9}
                )
            return _FakeResponse(
                {"language": "hi", "text": "namaste", "lang_prob": 0.91, "confidence": 0.8}
            )

        transcribe_segment_with_whisper(
            segment,
            WhisperClientConfig(url="http://example/whisper", languages=("en", "hi")),
            poster=poster,
            sleeper=lambda _seconds: None,
        )

        self.assertEqual(calls, [None, "en"])
        self.assertEqual(segment.asr_language, "hi")
        self.assertEqual(segment.asr_text, "namaste")
        self.assertEqual(segment.whisper_language, "hi")
        self.assertEqual(segment.asr_valid, 1)

    def test_bilingual_picks_higher_text_confidence_when_auto_lang_prob_is_low(self):
        segment = SpeechSegment(1, 0, 1000, np.zeros(16000, dtype=np.float32))

        def poster(url, params=None, files=None, timeout=None):
            if not params or "language" not in params:
                return _FakeResponse(
                    {"language": "fr", "text": "bonjour", "lang_prob": 0.4, "confidence": 0.2}
                )
            if params["language"] == "en":
                return _FakeResponse(
                    {"language": "en", "text": "okay thank you", "lang_prob": 0.3, "confidence": 0.63}
                )
            return _FakeResponse(
                {"language": "hi", "text": "theek hai", "lang_prob": 0.3, "confidence": 0.41}
            )

        transcribe_segment_with_whisper(
            segment,
            WhisperClientConfig(url="http://example/whisper", languages=("en", "hi")),
            poster=poster,
            sleeper=lambda _seconds: None,
        )

        self.assertEqual(segment.asr_language, "en")
        self.assertEqual(segment.asr_text, "okay thank you")
        self.assertEqual(segment.whisper_language, "fr")
        self.assertEqual(segment.asr_valid, 1)

    def test_bilingual_marks_invalid_when_both_confidences_are_low(self):
        segment = SpeechSegment(1, 0, 1000, np.zeros(16000, dtype=np.float32))

        def poster(url, params=None, files=None, timeout=None):
            language = (params or {}).get("language", "auto")
            return _FakeResponse(
                {"language": language, "text": "x", "lang_prob": 0.2, "confidence": 0.2}
            )

        transcribe_segment_with_whisper(
            segment,
            WhisperClientConfig(url="http://example/whisper", languages=("en", "hi")),
            poster=poster,
            sleeper=lambda _seconds: None,
        )

        self.assertEqual(segment.asr_valid, 0)
        self.assertFalse(segment.is_cluster_eligible)


if __name__ == "__main__":
    unittest.main()
