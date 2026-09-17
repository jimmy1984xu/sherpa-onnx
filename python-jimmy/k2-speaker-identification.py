#!/usr/bin/env python3

"""
Extract speaker embeddings for audio files from wav.scp or a WAV directory.

This script loads audio entries from a wav.scp file or a WAV directory, runs
a speaker embedding model, and writes one serialized embedding per segment.

Examples:
python3 ./python-jimmy/k2-speaker-identification.py \
  --wav-scp /path/to/wav.scp \
  --model /path/to/speaker_embedding.onnx \
  --output /path/to/output/embedding.txt

python3 ./python-jimmy/k2-speaker-identification.py \
  --wav-scp /path/to/wav.scp \
  --model /path/to/speaker_embedding.onnx \
  --output /path/to/output/embedding.txt \
  --max-audio-duration 15

python3 ./python-jimmy/k2-speaker-identification.py \
  --audio-dir /path/to/wav_segments \
  --model /path/to/speaker_embedding.onnx \
  --output /path/to/output/embedding.txt

Notes:
- wav.scp format: <segment_id> <absolute_path>
- audio-dir mode: recursively scans *.wav files and uses file stem as segment_id
- Output format: <segment_id> <base64_encoded_embedding>
- Audio is resampled to 16 kHz when needed
"""

import argparse
import base64
import time
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import soundfile as sf
import sherpa_onnx


def load_audio_mono_float32(path: str) -> Tuple[np.ndarray, int]:
    """Load audio file as mono float32."""
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    mono = data[:, 0]
    return np.ascontiguousarray(mono), sr


def resample_linear(samples: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Resample audio using linear interpolation."""
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


def parse_duration_from_segment_id(segment_id: str) -> Optional[float]:
    """
    从 segment_id 中解析 duration_ms，并转换为秒。
    
    格式: <prefix>_<offsetMs>_<durationMs>
    例如: meeting_zh_925190_3750 -> duration=3750ms = 3.75s
    
    Returns:
        duration in seconds, or None if parsing fails
    """
    parts = segment_id.split("_")
    if len(parts) >= 2:
        try:
            # 尝试解析最后两个部分为数字
            duration_ms = int(parts[-1])
            return duration_ms / 1000.0  # 转换为秒
        except ValueError:
            pass
    
    # 备用方案：正则提取数字，取最后一个
    import re
    numbers = re.findall(r"\d+", segment_id)
    if len(numbers) >= 1:
        try:
            duration_ms = int(numbers[-1])
            return duration_ms / 1000.0
        except ValueError:
            pass
    
    return None


def read_wav_scp(wav_scp_path: Path) -> List[Tuple[str, str]]:
    """
    Read wav.scp file and return list of (segment_id, audio_path) tuples.
    
    Format: <segment_id> <absolute_path>
    """
    segments = []
    with open(wav_scp_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            segment_id, audio_path = parts
            segments.append((segment_id, audio_path))
    return segments


def read_wav_dir(audio_dir: Path) -> List[Tuple[str, str]]:
    """
    Recursively scan a directory and return (segment_id, audio_path) tuples.

    Only *.wav files are included. segment_id uses file stem.
    """
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    segments = []
    for wav_path in sorted(audio_dir.rglob("*.wav")):
        segments.append((wav_path.stem, str(wav_path.resolve())))

    return segments


def build_extractor(model: str, num_threads: int, provider: str, debug: bool) -> sherpa_onnx.SpeakerEmbeddingExtractor:
    """Build speaker embedding extractor."""
    cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=model,
        num_threads=num_threads,
        debug=debug,
        provider=provider,
    )
    if not cfg.validate():
        raise ValueError(f"Invalid SpeakerEmbeddingExtractorConfig: {cfg}")
    return sherpa_onnx.SpeakerEmbeddingExtractor(cfg)


def extract_embedding_for_audio(
    extractor: sherpa_onnx.SpeakerEmbeddingExtractor,
    audio_path: str,
    target_sample_rate: int = 16000,
    max_duration: float = None,
) -> np.ndarray:
    """
    Extract speaker embedding from an audio file.
    
    Args:
        extractor: Speaker embedding extractor
        audio_path: Path to audio file
        target_sample_rate: Target sample rate (default: 16000)
        max_duration: Maximum audio duration in seconds. If audio is longer, only the first N seconds will be used.
    
    Returns:
        Embedding vector as numpy array, or None if extraction failed
    """
    # Load audio
    samples, sr = load_audio_mono_float32(audio_path)
    
    # Resample to target sample rate if needed
    if sr != target_sample_rate:
        samples = resample_linear(samples, sr, target_sample_rate)
    
    # Truncate audio if it exceeds max_duration
    if max_duration is not None and max_duration > 0:
        max_samples = int(max_duration * target_sample_rate)
        if len(samples) > max_samples:
            samples = samples[:max_samples]
    
    # Extract embedding
    stream = extractor.create_stream()
    stream.accept_waveform(sample_rate=target_sample_rate, waveform=samples)
    stream.input_finished()
    
    if not extractor.is_ready(stream):
        return None
    
    emb = extractor.compute(stream)
    return np.asarray(emb, dtype=np.float32)


def serialize_embedding(emb: np.ndarray) -> str:
    """Serialize embedding vector to base64 string."""
    emb_bytes = emb.astype(np.float32).tobytes()
    return base64.b64encode(emb_bytes).decode('ascii')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Extract speaker embeddings from audio files in wav.scp or a WAV directory"
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--wav-scp",
        type=str,
        help="Path to wav.scp file (format: segment_id path/to/audio.wav)"
    )
    input_group.add_argument(
        "--audio-dir",
        type=str,
        help="Path to a WAV directory. Recursively scans *.wav files and uses file stem as segment_id"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to speaker embedding model (.onnx)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path (full path) for embedding file"
    )
    
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Number of threads for inference"
    )
    
    parser.add_argument(
        "--provider",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "coreml"],
        help="Inference provider"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logs"
    )
    
    parser.add_argument(
        "--max-audio-duration",
        type=float,
        default=10.0,
        help="Maximum audio duration in seconds for embedding extraction. "
             "If audio is longer, only the first N seconds will be used. "
             "Default: 10.0 seconds. Set to 0 or negative to disable truncation."
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # 1) Read input audio list
    if args.wav_scp:
        wav_scp_path = Path(args.wav_scp)
        if not wav_scp_path.is_file():
            raise FileNotFoundError(f"wav.scp file not found: {wav_scp_path}")
        print(f"[1/4] Reading wav.scp: {wav_scp_path}")
        segments = read_wav_scp(wav_scp_path)
    else:
        audio_dir = Path(args.audio_dir)
        if not audio_dir.is_dir():
            raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
        print(f"[1/4] Reading WAV directory: {audio_dir}")
        segments = read_wav_dir(audio_dir)

    print(f"      Found {len(segments)} audio files")
    if not segments:
        raise ValueError("No audio files found!")
    
    # 2) Build speaker embedding extractor
    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"[2/4] Building Speaker Embedding Extractor: {model_path}")
    extractor = build_extractor(
        str(model_path),
        args.num_threads,
        args.provider,
        args.debug
    )
    print(f"      Extractor ready. embedding_dim={extractor.dim}")
    if args.max_audio_duration > 0:
        print(f"      Max audio duration: {args.max_audio_duration}s (audio will be truncated if longer)")
    else:
        print(f"      Max audio duration: unlimited")
    
    # 3) Extract embeddings
    print(f"[3/4] Extracting embeddings from {len(segments)} audio files...")
    results = []
    failed_count = 0
    
    # 用于统计耗时和RTF
    total_extraction_time = 0.0  # 总耗时（秒）
    segment_times = []  # 每个片段的耗时列表
    segment_rtfs = []  # 每个片段的RTF列表
    total_audio_duration = 0.0  # 总音频时长（秒）
    
    for idx, (segment_id, audio_path) in enumerate(segments, 1):
        audio_file = Path(audio_path)
        if not audio_file.is_file():
            print(f"      Warning: Audio file not found: {audio_path}, skipping...")
            failed_count += 1
            continue
        
        # 记录开始时间
        start_time = time.time()
        
        emb = extract_embedding_for_audio(
            extractor, 
            str(audio_file),
            max_duration=args.max_audio_duration if args.max_audio_duration > 0 else None
        )
        
        # 记录结束时间并计算耗时
        elapsed_time = time.time() - start_time
        total_extraction_time += elapsed_time
        segment_times.append(elapsed_time)
        
        if emb is None or emb.size == 0:
            print(f"      Warning: Failed to extract embedding from {audio_path}, skipping...")
            failed_count += 1
            continue
        
        # 解析音频时长并计算RTF
        audio_duration = parse_duration_from_segment_id(segment_id)
        if audio_duration is not None:
            # 如果设置了 max_audio_duration，使用实际处理的时长（取较小值）
            if args.max_audio_duration > 0:
                actual_processed_duration = min(audio_duration, args.max_audio_duration)
            else:
                actual_processed_duration = audio_duration
            total_audio_duration += actual_processed_duration
            # RTF = 处理时间 / 音频时长
            rtf = elapsed_time / actual_processed_duration if actual_processed_duration > 0 else 0.0
            segment_rtfs.append(rtf)
        else:
            # 如果无法从 segment_id 解析时长，使用实际音频文件时长
            try:
                samples, sr = load_audio_mono_float32(str(audio_file))
                actual_duration = len(samples) / float(sr)
                if args.max_audio_duration > 0:
                    actual_duration = min(actual_duration, args.max_audio_duration)
                total_audio_duration += actual_duration
                rtf = elapsed_time / actual_duration if actual_duration > 0 else 0.0
                segment_rtfs.append(rtf)
            except Exception:
                # 如果无法获取实际时长，跳过RTF统计
                pass
        
        emb_str = serialize_embedding(emb)
        results.append((segment_id, emb_str))
        
        if idx % max(1, len(segments) // 20) == 0 or idx == len(segments):
            pct = 100.0 * idx / len(segments)
            print(f"      Progress: {idx}/{len(segments)} ({pct:.1f}%)", end="\r")
    
    print()
    print(f"      Successfully extracted {len(results)} embeddings")
    if failed_count > 0:
        print(f"      Failed: {failed_count} files")
    
    # 4) Prepare output file path
    output_path = Path(args.output)
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 5) Write results
    print(f"[4/4] Writing results to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for segment_id, emb_str in results:
            f.write(f"{segment_id} {emb_str}\n")
    
    print(f"      Successfully wrote {len(results)} embeddings to {output_path}")
    
    # 6) Print statistics
    print(f"\n=== Summary ===")
    print(f"Total audio files: {len(segments)}")
    print(f"Successfully extracted: {len(results)}")
    print(f"Failed: {failed_count}")
    print(f"Output file: {output_path}")
    
    # 耗时统计
    if segment_times:
        avg_time_per_segment = total_extraction_time / len(segment_times)
        print(f"\n=== Performance Statistics ===")
        print(f"Total extraction time: {total_extraction_time:.3f} seconds")
        print(f"Average time per segment: {avg_time_per_segment:.3f} seconds")
        
        # RTF统计
        if segment_rtfs and total_audio_duration > 0:
            # RTF = 耗时 / 音频时长
            avg_rtf = total_extraction_time / total_audio_duration if total_audio_duration > 0 else 0.0
            min_rtf = min(segment_rtfs) if segment_rtfs else 0.0
            max_rtf = max(segment_rtfs) if segment_rtfs else 0.0
            avg_segment_rtf = sum(segment_rtfs) / len(segment_rtfs) if segment_rtfs else 0.0
            
            print(f"\nTotal audio duration: {total_audio_duration:.3f} seconds")
            print(f"Average RTF (total_time / total_duration): {avg_rtf:.3f}")
            print(f"Per-segment RTF statistics:")
            print(f"  Min RTF: {min_rtf:.3f}")
            print(f"  Max RTF: {max_rtf:.3f}")
            print(f"  Average RTF: {avg_segment_rtf:.3f}")
        else:
            print(f"\nNote: Unable to calculate RTF statistics (duration parsing may have failed)")


if __name__ == "__main__":
    main()

