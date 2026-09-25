import sys
import unittest
from pathlib import Path

import numpy as np

PIPELINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_DIR))

from asr import transcribe_segments
from vad import SpeechSegment


class FakeStream:
    def __init__(self):
        self.accepted = None
        self.result = None

    def accept_waveform(self, sample_rate, samples):
        self.accepted = (sample_rate, samples.copy())


class SuccessfulRecognizer:
    def __init__(self):
        self.stream = FakeStream()

    def create_stream(self):
        return self.stream

    def decode_stream(self, stream):
        stream.result = type(
            "Result",
            (),
            {"text": "  你好，世界  ", "confidence": 0.91},
        )()


class LegacyRecognizer:
    def __init__(self):
        self.stream = FakeStream()

    def create_stream(self):
        return self.stream

    def decode_stream(self, stream):
        stream.result = type(
            "Result",
            (),
            {
                "text": "  旧官方包结果  ",
                "tokens": ["旧", "官方", "包", "结果"],
                "ys_log_probs": [
                    np.log(0.8),
                    np.log(0.9),
                    np.log(0.7),
                    np.log(0.6),
                ],
            },
        )()


class NoConfidenceRecognizer:
    def __init__(self):
        self.stream = FakeStream()

    def create_stream(self):
        return self.stream

    def decode_stream(self, stream):
        stream.result = type("Result", (), {"text": "  无置信度字段  "})()


class FailingRecognizer:
    def create_stream(self):
        raise RuntimeError("decode unavailable")


class AsrTest(unittest.TestCase):
    def test_transcribes_segment_with_local_recognizer_stream(self):
        segment = SpeechSegment(1, 0, 1000, np.zeros(16000, dtype=np.float32))
        recognizer = SuccessfulRecognizer()

        transcribe_segments(recognizer, [segment], 16000)

        self.assertEqual(segment.asr_text, "你好，世界")
        self.assertAlmostEqual(segment.text_confidence, 0.91)
        self.assertIsNone(segment.asr_error)
        self.assertEqual(recognizer.stream.accepted[0], 16000)

    def test_computes_confidence_from_legacy_result_fields(self):
        segment = SpeechSegment(1, 0, 1000, np.zeros(16000, dtype=np.float32))

        transcribe_segments(LegacyRecognizer(), [segment], 16000)

        self.assertAlmostEqual(
            segment.text_confidence, (0.8 * 0.9 * 0.7 * 0.6) ** 0.25
        )
        self.assertIsNone(segment.asr_error)

    def test_uses_zero_confidence_when_legacy_result_has_no_confidence_fields(self):
        segment = SpeechSegment(1, 0, 1000, np.zeros(16000, dtype=np.float32))

        transcribe_segments(NoConfidenceRecognizer(), [segment], 16000)

        self.assertEqual(segment.text_confidence, 0.0)
        self.assertIsNone(segment.asr_error)

    def test_marks_model_unsupported_tiny_segment_as_asr_error_without_decode(self):
        segment = SpeechSegment(1, 0, 149, np.zeros(2399, dtype=np.float32))

        transcribe_segments(FailingRecognizer(), [segment], 16000)

        self.assertEqual(segment.asr_text, "asr_error")
        self.assertIn("2400", segment.asr_error)

    def test_attaches_failure_to_only_the_affected_segment(self):
        segment = SpeechSegment(1, 0, 1000, np.zeros(16000, dtype=np.float32))

        transcribe_segments(FailingRecognizer(), [segment], 16000)

        self.assertEqual(segment.asr_text, "asr_error")
        self.assertEqual(segment.asr_error, "decode unavailable")


if __name__ == "__main__":
    unittest.main()
