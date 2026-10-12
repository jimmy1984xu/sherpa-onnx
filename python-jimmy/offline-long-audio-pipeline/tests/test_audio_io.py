import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

PIPELINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_DIR))

from audio_io import AudioFormat, load_audio, write_segment_wav


class AudioIoTest(unittest.TestCase):
    def test_load_s16le_pcm_as_float32_mono_16khz(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "sample.pcm"
            path.write_bytes(b"\x00\x80\x00\x00\xff\x7f")
            audio = load_audio(path, AudioFormat("pcm", 16000, 1, 2))

        self.assertEqual(audio.sample_rate, 16000)
        self.assertEqual(audio.samples.dtype.name, "float32")
        self.assertEqual(audio.samples.tolist(), [-1.0, 0.0, 32767 / 32768])
        self.assertEqual(audio.duration_ms, 0)

    def test_pcm_is_mixed_down_and_resampled_to_16khz(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "stereo-8k.pcm"
            path.write_bytes(b"\x00\x80\xff\x7f\x00\x00\x00\x00")
            audio = load_audio(path, AudioFormat("pcm", 8000, 2, 2))

        self.assertEqual(audio.sample_rate, 16000)
        self.assertEqual(audio.samples.size, 4)
        self.assertAlmostEqual(float(audio.samples[0]), -1 / 65536, places=6)
        self.assertAlmostEqual(float(audio.samples[-1]), 0.0, places=6)

    def test_rejects_pcm_byte_count_not_divisible_by_frame_size(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "broken.pcm"
            path.write_bytes(b"\x00\x00\x00")
            with self.assertRaisesRegex(ValueError, "whole PCM frames"):
                load_audio(path, AudioFormat("pcm", 16000, 2, 2))


if __name__ == "__main__":
    unittest.main()

class SegmentWavOutputTest(unittest.TestCase):
    def test_write_segment_wav_creates_16khz_pcm_file(self):
        import soundfile as sf

        with TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "segment.wav"
            write_segment_wav(path, __import__("numpy").array([0.0, 0.5], dtype="float32"))
            samples, sample_rate = sf.read(path, dtype="float32")

        self.assertEqual(sample_rate, 16000)
        self.assertAlmostEqual(float(samples[1]), 0.5, places=4)

