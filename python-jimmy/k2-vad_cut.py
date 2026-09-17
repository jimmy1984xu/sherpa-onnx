#!/usr/bin/env python3

"""
Split a long audio file into speech segments with sherpa-onnx VAD.

This script supports Silero VAD and Ten VAD. It saves each detected segment
as a WAV file and can also write a wav.scp file for downstream scripts.

Examples:
python3 ./python-jimmy/k2_vad_cut.py \
  --audio /path/to/long_audio.wav \
  --silero-vad-model /path/to/silero_vad.onnx \
  --output-dir ./segments \
  --wav-scp ./segments/wav.scp

python3 ./python-jimmy/k2_vad_cut.py \
  --audio /path/to/long_audio.wav \
  --ten-vad-model /path/to/ten_vad.onnx \
  --output-dir ./segments \
  --threshold 0.6 \
  --min-silence-duration 0.3 \
  --min-speech-duration 0.2 \
  --max-speech-duration 30.0

Notes:
- Specify exactly one VAD model
- Raw PCM input must be 16 kHz, mono, and S16LE
- Output WAV files are written at 16 kHz
- wav.scp format: <segment_id> <absolute_path>
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
import sherpa_onnx


def load_audio_mono_float32(path: str) -> Tuple[np.ndarray, int]:
    if Path(path).suffix.lower() == ".pcm":
        pcm = np.fromfile(path, dtype="<i2")
        samples = np.ascontiguousarray(pcm.astype(np.float32) / 32768.0)
        return samples, 16000

    data, sr = sf.read(path, always_2d=True, dtype="float32")
    mono = data[:, 0]
    return np.ascontiguousarray(mono), sr


def resample_linear(samples: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return samples
    duration = samples.shape[0] / float(src_sr)
    dst_len = int(round(duration * dst_sr))
    if dst_len <= 0:
        return np.zeros((0,), dtype=np.float32)
    x_src = np.linspace(0.0, duration, num=samples.shape[0], endpoint=False, dtype=np.float64)
    x_dst = np.linspace(0.0, duration, num=dst_len, endpoint=False, dtype=np.float64)
    y = np.interp(x_dst, x_src, samples.astype(np.float64))
    return y.astype(np.float32, copy=False)


def build_segment_id(sequence: int, offset_ms: int, duration_ms: int) -> str:
    """Build a segment ID containing its sequence and timing fields."""
    return f"{sequence:06d}_{offset_ms}_{duration_ms}"


def build_vad(
    vad_model_path: str,
    vad_type: str,
    sample_rate: int,
    threshold: float,
    min_silence_duration: float,
    min_speech_duration: float,
    max_speech_duration: float,
    pre_speech_pad_duration: float,
    debug: bool
) -> Tuple[sherpa_onnx.VoiceActivityDetector, int, dict]:
    """
    Build VAD detector and return it along with window size and config info.
    
    Args:
        vad_model_path: Path to VAD model file
        vad_type: "silero" or "ten"
        sample_rate: Audio sample rate
        threshold: VAD threshold (default: 0.5)
        min_silence_duration: Min silence duration in seconds (default: 0.5)
        min_speech_duration: Min speech duration in seconds (default: 0.25)
        max_speech_duration: Max speech duration in seconds (default: 20.0)
        pre_speech_pad_duration: Audio retained before detected speech in seconds
        debug: Enable debug logs
    
    Returns:
        Tuple of (VAD detector, window_size in samples, config_dict)
    """
    vad_cfg = sherpa_onnx.VadModelConfig()

    if vad_type == "silero":
        vad_cfg.silero_vad.model = vad_model_path
        vad_cfg.silero_vad.threshold = threshold
        vad_cfg.silero_vad.min_silence_duration = min_silence_duration
        vad_cfg.silero_vad.min_speech_duration = min_speech_duration
        vad_cfg.silero_vad.max_speech_duration = max_speech_duration
        config_info = {
            "vad_type": "silero",
            "model_path": vad_model_path,
            "threshold": vad_cfg.silero_vad.threshold,
            "min_silence_duration": vad_cfg.silero_vad.min_silence_duration,
            "min_speech_duration": vad_cfg.silero_vad.min_speech_duration,
            "window_size": vad_cfg.silero_vad.window_size,
            "max_speech_duration": vad_cfg.silero_vad.max_speech_duration,
        }
        window_size = vad_cfg.silero_vad.window_size
    elif vad_type == "ten":
        vad_cfg.ten_vad.model = vad_model_path
        vad_cfg.ten_vad.threshold = threshold
        vad_cfg.ten_vad.min_silence_duration = min_silence_duration
        vad_cfg.ten_vad.min_speech_duration = min_speech_duration
        vad_cfg.ten_vad.max_speech_duration = max_speech_duration
        config_info = {
            "vad_type": "ten",
            "model_path": vad_model_path,
            "threshold": vad_cfg.ten_vad.threshold,
            "min_silence_duration": vad_cfg.ten_vad.min_silence_duration,
            "min_speech_duration": vad_cfg.ten_vad.min_speech_duration,
            "window_size": vad_cfg.ten_vad.window_size,
            "max_speech_duration": vad_cfg.ten_vad.max_speech_duration,
        }
        window_size = vad_cfg.ten_vad.window_size
    else:
        raise ValueError(f"Unknown VAD type: {vad_type}")

    vad_cfg.sample_rate = sample_rate
    vad_cfg.pre_speech_pad_duration = pre_speech_pad_duration
    config_info["sample_rate"] = sample_rate
    config_info["pre_speech_pad_duration"] = pre_speech_pad_duration
    config_info["debug"] = debug
    
    if not vad_cfg.validate():
        raise ValueError("Invalid VAD config")

    vad = sherpa_onnx.VoiceActivityDetector(vad_cfg, buffer_size_in_seconds=3600)

    return vad, window_size, config_info


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Cut long audio into speech segments using sherpa-onnx VAD",
    )

    parser.add_argument("--audio", type=str, required=True, help="Path to a long audio file")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save segment WAV files")
    parser.add_argument("--wav-scp", type=str, default="", help="Path to output wav.scp file (optional)")
    parser.add_argument("--silero-vad-model", type=str, default="", help="Path to silero_vad.onnx")
    parser.add_argument("--ten-vad-model", type=str, default="", help="Path to ten-vad.onnx")
    parser.add_argument("--threshold", type=float, default=0.5, help="VAD threshold (default: 0.5)")
    parser.add_argument("--min-silence-duration", type=float, default=0.5, help="Min silence duration in seconds (default: 0.5)")
    parser.add_argument("--min-speech-duration", type=float, default=0.25, help="Min speech duration in seconds (default: 0.25)")
    parser.add_argument("--max-speech-duration", type=float, default=20.0, help="Max speech duration in seconds (default: 20.0)")
    parser.add_argument("--pre-speech-pad-duration", type=float, default=0.0, help="Audio retained before detected speech in seconds (default: 0.0)")
    parser.add_argument("--debug", action="store_true", help="Enable sherpa-onnx debug logs")

    return parser.parse_args()


def main():
    args = parse_args()

    if bool(args.silero_vad_model) == bool(args.ten_vad_model):
        raise ValueError("Please specify exactly one of --silero-vad-model or --ten-vad-model")

    audio_path = Path(args.audio)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    wav_output_dir = Path(args.output_dir)
    wav_output_dir.mkdir(parents=True, exist_ok=True)

    # wav.scp is optional
    wav_scp_path = Path(args.wav_scp) if args.wav_scp else None
    if wav_scp_path:
        wav_scp_path.parent.mkdir(parents=True, exist_ok=True)

    wav, sr = load_audio_mono_float32(str(audio_path))

    vad_sr = 16000
    wav_16k = resample_linear(wav, sr, vad_sr)

    if args.silero_vad_model:
        vad_model_path = args.silero_vad_model
        vad_type = "silero"
    else:
        vad_model_path = args.ten_vad_model
        vad_type = "ten"

    vad, window_size, vad_config = build_vad(
        vad_model_path=vad_model_path,
        vad_type=vad_type,
        sample_rate=vad_sr,
        threshold=args.threshold,
        min_silence_duration=args.min_silence_duration,
        min_speech_duration=args.min_speech_duration,
        max_speech_duration=args.max_speech_duration,
        pre_speech_pad_duration=args.pre_speech_pad_duration,
        debug=args.debug
    )

    window_n = window_size

    audio_base_name = audio_path.stem
    segments_info: List[Tuple[str, Path]] = []
    segment_durations: List[float] = []  # Store durations in seconds

    for i in range(0, len(wav_16k), window_n):
        vad.accept_waveform(wav_16k[i : i + window_n])

        while not vad.empty():
            seg = vad.front
            samples = np.asarray(seg.samples, dtype=np.float32)
            
            # No minimum duration limit - process all VAD segments
            offset_samples = seg.start
            offset_ms = int(offset_samples / vad_sr * 1000)
            duration_ms = int(len(samples) / vad_sr * 1000)
            duration_seconds = duration_ms / 1000.0

            sequence = len(segments_info) + 1
            wav_filename = f"{build_segment_id(sequence, offset_ms, duration_ms)}.wav"
            wav_filepath = wav_output_dir / wav_filename
            segment_id = wav_filepath.stem

            sf.write(str(wav_filepath), samples, vad_sr)

            segments_info.append((segment_id, wav_filepath))
            segment_durations.append(duration_seconds)

            print(
                f"Segment {segment_id} [{offset_ms/1000:.2f}s-{(offset_ms + duration_ms)/1000:.2f}s] "
                f"duration={duration_ms/1000:.2f}s saved to {wav_filepath}"
            )

            vad.pop()

    # Flush buffered speech at end of input before draining the final segment.
    vad.flush()

    # Final drain
    while not vad.empty():
        seg = vad.front
        samples = np.asarray(seg.samples, dtype=np.float32)
        
        # No minimum duration limit - process all VAD segments
        offset_samples = seg.start
        offset_ms = int(offset_samples / vad_sr * 1000)
        duration_ms = int(len(samples) / vad_sr * 1000)
        duration_seconds = duration_ms / 1000.0

        sequence = len(segments_info) + 1
        wav_filename = f"{build_segment_id(sequence, offset_ms, duration_ms)}.wav"
        wav_filepath = wav_output_dir / wav_filename
        segment_id = wav_filepath.stem

        sf.write(str(wav_filepath), samples, vad_sr)

        segments_info.append((segment_id, wav_filepath))
        segment_durations.append(duration_seconds)

        print(
            f"Segment {segment_id} [{offset_ms/1000:.2f}s-{(offset_ms + duration_ms)/1000:.2f}s] "
            f"duration={duration_ms/1000:.2f}s saved to {wav_filepath}"
        )
        vad.pop()

    # Write wav.scp if specified
    if wav_scp_path:
        with open(wav_scp_path, "w", encoding="utf-8") as f:
            for segment_id, path in segments_info:
                f.write(f"{segment_id} {path}\n")
        print(f"\nDone! Saved {len(segments_info)} segments to {wav_output_dir}")
        print(f"wav.scp written to {wav_scp_path}")
    else:
        print(f"\nDone! Saved {len(segments_info)} segments to {wav_output_dir}")
    
    # Print summary information
    print(f"\n{'='*60}")
    print(f"Summary Information")
    print(f"{'='*60}")
    
    # 1. VAD parameter information
    print(f"\n[1] VAD Parameters:")
    print(f"    VAD Type: {vad_config['vad_type'].upper()}")
    print(f"    Model Path: {vad_config['model_path']}")
    print(f"    Sample Rate: {vad_config['sample_rate']} Hz")
    print(f"    Threshold: {vad_config['threshold']}")
    print(f"    Min Silence Duration: {vad_config['min_silence_duration']}s")
    print(f"    Min Speech Duration: {vad_config['min_speech_duration']}s")
    print(f"    Window Size: {vad_config['window_size']} samples ({vad_config['window_size']/vad_config['sample_rate']:.3f}s)")
    print(f"    Max Speech Duration: {vad_config['max_speech_duration']}s")
    print(f"    Pre-Speech Pad Duration: {vad_config['pre_speech_pad_duration']}s")
    print(f"    Debug: {vad_config['debug']}")
    
    # 2. Segment duration distribution
    if segment_durations:
        print(f"\n[2] Segment Duration Distribution:")
        print(f"    Total Segments: {len(segment_durations)}")
        print(f"    Min Duration: {min(segment_durations):.3f}s")
        print(f"    Max Duration: {max(segment_durations):.3f}s")
        print(f"    Average Duration: {sum(segment_durations)/len(segment_durations):.3f}s")
        print(f"    Total Duration: {sum(segment_durations):.2f}s")
        
        # Duration distribution by ranges
        duration_ranges = {
            "< 0.5s": 0,
            "0.5-1s": 0,
            "1-3s": 0,
            "3-5s": 0,
            "5-10s": 0,
            "10-20s": 0,
            "20-30s": 0,
            ">= 30s": 0
        }
        for dur in segment_durations:
            if dur < 0.5:
                duration_ranges["< 0.5s"] += 1
            elif dur < 1.0:
                duration_ranges["0.5-1s"] += 1
            elif dur < 3.0:
                duration_ranges["1-3s"] += 1
            elif dur < 5.0:
                duration_ranges["3-5s"] += 1
            elif dur < 10.0:
                duration_ranges["5-10s"] += 1
            elif dur < 20.0:
                duration_ranges["10-20s"] += 1
            elif dur < 30.0:
                duration_ranges["20-30s"] += 1
            else:
                duration_ranges[">= 30s"] += 1
        
        print(f"\n    Duration Range Distribution:")
        for range_name, count in duration_ranges.items():
            percentage = 100.0 * count / len(segment_durations) if segment_durations else 0.0
            print(f"      {range_name:8s}: {count:4d} segments ({percentage:5.1f}%)")
    else:
        print(f"\n[2] Segment Duration Distribution:")
        print(f"    No segments found!")
    
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()

