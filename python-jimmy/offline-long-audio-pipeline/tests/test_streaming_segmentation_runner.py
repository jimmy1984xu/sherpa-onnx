import unittest


class SpeakerSegmentationApiTest(unittest.TestCase):
    def test_public_symbols_and_span_shape(self):
        import sherpa_onnx

        for name in (
            "SpeakerSegmentationConfig",
            "SpeakerSegmentation",
            "SpeakerSegmentationSpan",
            "CONTINUE",
            "SPEAKER_COUNT_CHANGED",
            "SINGLE_SPEAKER_CHANGED",
            "INPUT_FINISHED",
        ):
            self.assertTrue(hasattr(sherpa_onnx, name), name)


if __name__ == "__main__":
    unittest.main()