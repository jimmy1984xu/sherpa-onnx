#!/usr/bin/env python3

"""
Split a long audio file into speaker activity segments.

This script uses only a speaker segmentation model. It does not run speaker
embedding extraction or clustering, so speaker IDs are temporary indices from
the segmentation result.

Examples:
python3 ./python-jimmy/k2_speaker_segmentation_cut.py \
  --audio /path/to/long_audio.wav \
  --segmentation-model /path/to/segmentation.onnx \
  --output-dir ./segments \
  --wav-scp ./segments/wav.scp

python3 ./python-jimmy/k2_speaker_segmentation_cut.py \
  --audio /path/to/long_audio.wav \
  --segmentation-model /path/to/segmentation.onnx \
  --output-dir ./segments \
  --min-duration-on 0.3 \
  --min-duration-off 0.5

Notes:
- Output WAV files are written at 16 kHz
- wav.scp format: <segment_id> <absolute_path>
- Speaker IDs are not stable identities
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import onnxruntime as ort
import soundfile as sf
from numpy.lib.stride_tricks import as_strided


def load_audio_mono_float32(path: str) -> Tuple[np.ndarray, int]:
    """Load audio file as mono float32 array."""
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


class OnnxSegmentationModel:
    """Wrapper for ONNX segmentation model."""
    
    def __init__(self, filename: str, num_threads: int = 1):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = num_threads
        session_opts.intra_op_num_threads = num_threads
        
        self.model = ort.InferenceSession(
            filename,
            sess_options=session_opts,
            providers=["CPUExecutionProvider"],
        )
        
        meta = self.model.get_modelmeta().custom_metadata_map
        
        self.window_size = int(meta["window_size"])
        self.sample_rate = int(meta["sample_rate"])
        self.window_shift = int(0.1 * self.window_size)
        self.receptive_field_size = int(meta["receptive_field_size"])
        self.receptive_field_shift = int(meta["receptive_field_shift"])
        self.num_speakers = int(meta["num_speakers"])
        self.powerset_max_classes = int(meta["powerset_max_classes"])
        self.num_classes = int(meta["num_classes"])
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
          x: (N, num_samples)
        Returns:
          A tensor of shape (N, num_frames, num_classes)
        """
        x = np.expand_dims(x, axis=1)
        (y,) = self.model.run(
            [self.model.get_outputs()[0].name],
            {self.model.get_inputs()[0].name: x}
        )
        return y


def get_powerset_mapping(num_classes: int, num_speakers: int, powerset_max_classes: int) -> np.ndarray:
    """Convert powerset classes to multi-label format."""
    mapping = np.zeros((num_classes, num_speakers), dtype=np.int32)
    
    k = 1
    for i in range(1, powerset_max_classes + 1):
        if i == 1:
            for j in range(0, num_speakers):
                mapping[k, j] = 1
                k += 1
        elif i == 2:
            for j in range(0, num_speakers):
                for m in range(j + 1, num_speakers):
                    mapping[k, j] = 1
                    mapping[k, m] = 1
                    k += 1
        else:
            raise RuntimeError(f"Unsupported powerset_max_classes: {i}")
    
    return mapping


def to_multi_label(y: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    """
    Convert powerset classes to multi-label format.
    
    Args:
      y: (num_chunks, num_frames, num_classes)
    Returns:
      A tensor of shape (num_chunks, num_frames, num_speakers)
    """
    y = np.argmax(y, axis=-1)
    labels = mapping[y.reshape(-1)].reshape(y.shape[0], y.shape[1], -1)
    return labels


def extract_segments_from_labels(
    labels: np.ndarray,
    seg_m: OnnxSegmentationModel,
    audio: np.ndarray,
    min_duration_on: float,
    min_duration_off: float,
) -> List[Tuple[float, float, int]]:
    """
    Extract speaker segments from labels.
    
    Args:
      labels: (num_chunks, num_frames, num_speakers) multi-label format
      seg_m: Segmentation model
      audio: Audio samples
      min_duration_on: Minimum segment duration in seconds.
                      Segments shorter than this will be filtered out.
      min_duration_off: Minimum gap between segments to merge in seconds.
                       If two segments from the same speaker are separated by a gap <= this value,
                       they will be merged into one segment.
    
    Returns:
      List of (start_time, end_time, speaker_index) tuples
    """
    # Compute speaker count per frame
    num_frames = int(
        (seg_m.window_size + (labels.shape[0] - 1) * seg_m.window_shift)
        / seg_m.receptive_field_shift
    ) + 1
    
    # Aggregate labels across chunks
    count = np.zeros((num_frames, labels.shape[2]), dtype=np.float32)
    weight = np.zeros((num_frames,), dtype=np.float32)
    
    for i in range(labels.shape[0]):
        start_frame = int(i * seg_m.window_shift / seg_m.receptive_field_shift + 0.5)
        end_frame = start_frame + labels.shape[1]
        end_frame = min(end_frame, num_frames)
        
        if start_frame < num_frames:
            seq_len = end_frame - start_frame
            count[start_frame:end_frame] += labels[i, :seq_len, :].astype(np.float32)
            weight[start_frame:end_frame] += 1.0
    
    # Normalize
    weight = np.maximum(weight, 1e-12)
    count = count / weight[:, np.newaxis]
    
    # Convert to binary (threshold = 0.5)
    binary_labels = (count > 0.5).astype(np.int32)
    
    # Extract segments for each speaker
    segments = []
    scale = seg_m.receptive_field_shift / seg_m.sample_rate
    scale_offset = 0.5 * seg_m.receptive_field_size / seg_m.sample_rate
    
    for speaker_idx in range(binary_labels.shape[1]):
        speaker_labels = binary_labels[:, speaker_idx]
        
        # Find active regions
        is_active = False
        start_frame = -1
        
        for frame_idx in range(len(speaker_labels)):
            if speaker_labels[frame_idx] == 1:
                if not is_active:
                    is_active = True
                    start_frame = frame_idx
            else:
                if is_active:
                    # Segment ended
                    end_frame = frame_idx
                    start_time = start_frame * scale + scale_offset
                    end_time = end_frame * scale + scale_offset
                    
                    if end_time - start_time >= min_duration_on:
                        segments.append((start_time, end_time, speaker_idx))
                    
                    is_active = False
        
        # Handle last segment
        if is_active:
            end_frame = len(speaker_labels)
            start_time = start_frame * scale + scale_offset
            end_time = end_frame * scale + scale_offset
            
            if end_time - start_time >= min_duration_on:
                segments.append((start_time, end_time, speaker_idx))
    
    # Merge segments with small gaps
    segments = sorted(segments, key=lambda x: (x[2], x[0]))  # Sort by speaker, then start time
    
    merged_segments = []
    for start_time, end_time, speaker_idx in segments:
        if merged_segments:
            last_start, last_end, last_speaker = merged_segments[-1]
            if (last_speaker == speaker_idx and 
                start_time - last_end <= min_duration_off):
                # Merge with previous segment
                merged_segments[-1] = (last_start, end_time, speaker_idx)
                continue
        
        merged_segments.append((start_time, end_time, speaker_idx))
    
    return merged_segments


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Cut long audio into speaker segments using only segmentation model",
    )

    parser.add_argument("--audio", type=str, required=True, help="Path to a long audio file")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save segment WAV files")
    parser.add_argument("--wav-scp", type=str, default="", help="Path to output wav.scp file (optional)")
    parser.add_argument("--segmentation-model", type=str, required=True, help="Path to segmentation model (e.g., model.onnx from pyannote-segmentation)")
    parser.add_argument("--min-duration-on", type=float, default=0.2, help="Minimum segment duration in seconds (default: 0.2)")
    parser.add_argument("--min-duration-off", type=float, default=0.5, help="Minimum gap between segments to merge in seconds (default: 0.5)")
    parser.add_argument("--num-threads", type=int, default=1, help="Number of threads for inference (default: 1)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for model inference (default: 32)")

    return parser.parse_args()


def main():
    args = parse_args()

    audio_path = Path(args.audio)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    wav_output_dir = Path(args.output_dir)
    wav_output_dir.mkdir(parents=True, exist_ok=True)

    # wav.scp is optional
    wav_scp_path = Path(args.wav_scp) if args.wav_scp else None
    if wav_scp_path:
        wav_scp_path.parent.mkdir(parents=True, exist_ok=True)

    # Load segmentation model
    print("Loading segmentation model...")
    seg_m = OnnxSegmentationModel(args.segmentation_model, num_threads=args.num_threads)
    print(f"Model sample rate: {seg_m.sample_rate} Hz")
    print(f"Window size: {seg_m.window_size} samples ({seg_m.window_size/seg_m.sample_rate:.2f}s)")
    print(f"Number of speakers (max): {seg_m.num_speakers}")

    # Load and resample audio
    print("Loading audio...")
    wav, sr = load_audio_mono_float32(str(audio_path))
    
    if sr != seg_m.sample_rate:
        print(f"Resampling audio from {sr}Hz to {seg_m.sample_rate}Hz...")
        wav = resample_linear(wav, sr, seg_m.sample_rate)
        sr = seg_m.sample_rate
    
    # Process audio with segmentation model
    print("Processing audio with segmentation model...")
    
    # Prepare chunks
    num_chunks = (len(wav) - seg_m.window_size) // seg_m.window_shift + 1
    has_last_chunk = ((len(wav) - seg_m.window_size) % seg_m.window_shift) > 0
    
    # Create strided view for efficient chunking
    if num_chunks > 0:
        samples = as_strided(
            wav,
            shape=(num_chunks, seg_m.window_size),
            strides=(seg_m.window_shift * wav.strides[0], wav.strides[0]),
        )
    else:
        samples = np.zeros((1, seg_m.window_size), dtype=wav.dtype)
        samples[0, :len(wav)] = wav
    
    # Run model inference
    output = []
    for i in range(0, len(samples), args.batch_size):
        batch = samples[i:i + args.batch_size]
        y = seg_m(batch)
        output.append(y)
    
    # Handle last chunk if needed
    if has_last_chunk and num_chunks > 0:
        last_chunk = wav[num_chunks * seg_m.window_shift:]
        pad_size = seg_m.window_size - len(last_chunk)
        last_chunk = np.pad(last_chunk, (0, pad_size))
        last_chunk = np.expand_dims(last_chunk, axis=0)
        y = seg_m(last_chunk)
        output.append(y)
    
    # Concatenate outputs
    y = np.vstack(output)  # (num_chunks, num_frames, num_classes)
    
    # Convert to multi-label format
    print("Converting to multi-label format...")
    mapping = get_powerset_mapping(
        num_classes=seg_m.num_classes,
        num_speakers=seg_m.num_speakers,
        powerset_max_classes=seg_m.powerset_max_classes,
    )
    labels = to_multi_label(y, mapping)  # (num_chunks, num_frames, num_speakers)
    
    # Extract segments
    print("Extracting segments...")
    segments = extract_segments_from_labels(
        labels, seg_m, wav, args.min_duration_on, args.min_duration_off
    )
    
    if not segments:
        print("No speaker segments found in the audio!")
        return
    
    # Extract segments
    audio_base_name = audio_path.stem
    segments_info: List[Tuple[str, Path, int]] = []  # (segment_id, filepath, speaker_idx)
    segment_durations: List[float] = []
    speaker_stats = {}  # Track statistics per speaker
    
    print(f"\nFound {len(segments)} segments")
    print("Saving segments...")
    
    for start_time, end_time, speaker_idx in segments:
        duration = end_time - start_time
        
        # Convert time to sample indices
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        end_sample = min(end_sample, len(wav))
        segment_samples = wav[start_sample:end_sample]
        
        # Generate filename: <audio_name>_<start_ms>_<duration_ms>.wav
        start_ms = int(start_time * 1000)
        duration_ms = int(duration * 1000)
        wav_filename = f"{audio_base_name}_{start_ms}_{duration_ms}.wav"
        wav_filepath = wav_output_dir / wav_filename
        segment_id = wav_filename.replace(".wav", "")
        
        # Save segment
        sf.write(str(wav_filepath), segment_samples, sr)
        
        segments_info.append((segment_id, wav_filepath, speaker_idx))
        segment_durations.append(duration)
        
        # Update speaker statistics
        if speaker_idx not in speaker_stats:
            speaker_stats[speaker_idx] = {
                "count": 0,
                "total_duration": 0.0,
            }
        speaker_stats[speaker_idx]["count"] += 1
        speaker_stats[speaker_idx]["total_duration"] += duration
        
        # Print log with segment ID and speaker ID
        print(
            f"Segment {segment_id} [{start_time:.3f}s-{end_time:.3f}s] "
            f"speaker_{speaker_idx:02d} duration={duration:.3f}s saved to {wav_filepath}"
        )
        print(f"  -> Segment ID: {segment_id}, Speaker ID: {speaker_idx:02d}")
    
    # Write wav.scp if specified
    if wav_scp_path:
        with open(wav_scp_path, "w", encoding="utf-8") as f:
            for segment_id, path, speaker_idx in segments_info:
                f.write(f"{segment_id} {path}\n")
        print(f"\nDone! Saved {len(segments_info)} segments to {wav_output_dir}")
        print(f"wav.scp written to {wav_scp_path}")
    else:
        print(f"\nDone! Saved {len(segments_info)} segments to {wav_output_dir}")
    
    # Print segment ID to speaker ID mapping
    print(f"\n{'='*60}")
    print(f"Segment ID to Speaker ID Mapping")
    print(f"{'='*60}")
    for segment_id, path, speaker_idx in segments_info:
        print(f"  {segment_id} -> speaker_{speaker_idx:02d}")
    
    # Print summary information
    print(f"\n{'='*60}")
    print(f"Summary Information")
    print(f"{'='*60}")
    
    # 1. Configuration information
    print(f"\n[1] Segmentation Configuration:")
    print(f"    Segmentation Model: {args.segmentation_model}")
    print(f"    Sample Rate: {sr} Hz")
    print(f"    Min Duration On: {args.min_duration_on}s")
    print(f"    Min Duration Off: {args.min_duration_off}s")
    print(f"    Num Threads: {args.num_threads}")
    print(f"    Batch Size: {args.batch_size}")
    print(f"    Note: Speaker indices are temporary (from segmentation model), not unique speaker identities.")
    
    # 2. Overall segment statistics
    if segment_durations:
        print(f"\n[2] Overall Segment Statistics:")
        print(f"    Total Segments: {len(segment_durations)}")
        print(f"    Detected Speaker Indices: {len(speaker_stats)}")
        print(f"    Min Duration: {min(segment_durations):.3f}s")
        print(f"    Max Duration: {max(segment_durations):.3f}s")
        print(f"    Average Duration: {sum(segment_durations)/len(segment_durations):.3f}s")
        print(f"    Total Duration: {sum(segment_durations):.2f}s")
    
    # 3. Per-speaker statistics
    if speaker_stats:
        print(f"\n[3] Per-Speaker Index Statistics:")
        for speaker_idx in sorted(speaker_stats.keys()):
            stats = speaker_stats[speaker_idx]
            avg_duration = stats["total_duration"] / stats["count"]
            percentage = 100.0 * stats["total_duration"] / sum(segment_durations) if segment_durations else 0.0
            print(f"    Speaker Index {speaker_idx:02d}:")
            print(f"      Segments: {stats['count']}")
            print(f"      Total Duration: {stats['total_duration']:.2f}s ({percentage:.1f}%)")
            print(f"      Average Duration: {avg_duration:.3f}s")
    
    # 4. Duration distribution
    if segment_durations:
        print(f"\n[4] Segment Duration Distribution:")
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
        
        print(f"    Duration Range Distribution:")
        for range_name, count in duration_ranges.items():
            percentage = 100.0 * count / len(segment_durations) if segment_durations else 0.0
            print(f"      {range_name:8s}: {count:4d} segments ({percentage:5.1f}%)")
    
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
