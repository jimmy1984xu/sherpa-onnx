#!/usr/bin/env python3
# Copyright (c)  2024  Xiaomi Corporation

"""
Cut audio segments from wav.scp using speaker segmentation model only.

This script reads a wav.scp file, performs speaker segmentation on each audio segment,
and cuts the audio based on segmentation results. If segmentation finds only one segment,
the original segment is used. If multiple segments are found, they are used for cutting.

Usage example:

python3 ./python-api-examples/k2_diarization_cut.py \
  --wav-scp /path/to/wav.scp \
  --segmentation-model /path/to/segmentation.onnx \
  --output-dir /path/to/output \
  --min-duration-on 0.3 \
  --min-duration-off 0.5

Notes:
- Input wav.scp format: <segment_id> <absolute_path>
- Output wav.scp format: <segment_id>_<offset_ms>_<duration_ms> <absolute_path>
- If segmentation finds only 1 segment, the original segment is used (no cutting)
- If segmentation finds >1 segments, audio is cut based on segmentation results
- Only segmentation model is used (no embedding extraction or clustering needed)
- All cut segments are saved to the output directory
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import onnxruntime as ort
import soundfile as sf
from numpy.lib.stride_tricks import as_strided


def read_wav_scp(wav_scp_path: Path) -> List[Tuple[str, str]]:
    """
    Read wav.scp file and return list of (segment_id, audio_path) tuples.
    
    Format: <segment_id> <absolute_path>
    """
    segments = []
    with open(wav_scp_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                print(f"Warning: Line {line_num} has invalid format, skipping: {line[:50]}...")
                continue
            segment_id, audio_path = parts
            segments.append((segment_id, audio_path))
    return segments


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


def cut_audio_segment(
    audio: np.ndarray,
    sample_rate: int,
    start_time: float,
    end_time: float,
) -> np.ndarray:
    """
    Cut audio segment from start_time to end_time.
    
    Args:
      audio: Audio samples
      sample_rate: Sample rate
      start_time: Start time in seconds
      end_time: End time in seconds
    
    Returns:
      Cut audio segment
    """
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    end_sample = min(end_sample, len(audio))
    start_sample = max(0, start_sample)
    
    if start_sample >= end_sample:
        return np.array([], dtype=np.float32)
    
    return audio[start_sample:end_sample]


def process_audio_segment(
    segment_id: str,
    audio_path: str,
    seg_m: OnnxSegmentationModel,
    output_dir: Path,
    min_duration_on: float,
    min_duration_off: float,
    batch_size: int = 32,
) -> List[Tuple[str, Path]]:
    """
    Process a single audio segment with segmentation and cut if needed.
    
    Args:
      segment_id: Original segment ID
      audio_path: Path to audio file
      seg_m: Segmentation model
      output_dir: Output directory for cut segments
      min_duration_on: Minimum segment duration in seconds
      min_duration_off: Minimum gap between segments to merge in seconds
      batch_size: Batch size for model inference
    
    Returns:
      List of (new_segment_id, output_path) tuples
    """
    # Load audio
    audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    audio = audio[:, 0]  # only use the first channel
    
    # Resample to match segmentation model sample rate
    if sample_rate != seg_m.sample_rate:
        audio = resample_linear(audio, sample_rate, seg_m.sample_rate)
        sample_rate = seg_m.sample_rate
    
    # Process audio with segmentation model
    # Prepare chunks
    num_chunks = (len(audio) - seg_m.window_size) // seg_m.window_shift + 1
    has_last_chunk = ((len(audio) - seg_m.window_size) % seg_m.window_shift) > 0
    
    # Create strided view for efficient chunking
    if num_chunks > 0:
        samples = as_strided(
            audio,
            shape=(num_chunks, seg_m.window_size),
            strides=(seg_m.window_shift * audio.strides[0], audio.strides[0]),
        )
    else:
        samples = np.zeros((1, seg_m.window_size), dtype=audio.dtype)
        samples[0, :len(audio)] = audio
    
    # Run model inference
    output = []
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        y = seg_m(batch)
        output.append(y)
    
    # Handle last chunk if needed
    if has_last_chunk and num_chunks > 0:
        last_chunk = audio[num_chunks * seg_m.window_shift:]
        pad_size = seg_m.window_size - len(last_chunk)
        last_chunk = np.pad(last_chunk, (0, pad_size))
        last_chunk = np.expand_dims(last_chunk, axis=0)
        y = seg_m(last_chunk)
        output.append(y)
    
    # Concatenate outputs
    y = np.vstack(output)  # (num_chunks, num_frames, num_classes)
    
    # Convert to multi-label format
    mapping = get_powerset_mapping(
        num_classes=seg_m.num_classes,
        num_speakers=seg_m.num_speakers,
        powerset_max_classes=seg_m.powerset_max_classes,
    )
    labels = to_multi_label(y, mapping)  # (num_chunks, num_frames, num_speakers)
    
    # Extract segments
    segments = extract_segments_from_labels(
        labels, seg_m, audio, min_duration_on, min_duration_off
    )
    
    output_segments = []
    
    # If only 1 segment found, use original segment
    if len(segments) <= 1:
        # Use original segment (no cutting)
        original_filename = f"{segment_id}.wav"
        original_path = output_dir / original_filename
        sf.write(str(original_path), audio, sample_rate)
        output_segments.append((segment_id, original_path))
        print(f"  {segment_id}: Using original segment (segmentation found {len(segments)} segment(s))")
    else:
        # Cut based on segmentation results
        print(f"  {segment_id}: Cutting into {len(segments)} segments based on segmentation")
        for idx, (start_time, end_time, speaker_idx) in enumerate(segments):
            # Cut audio segment
            segment_audio = cut_audio_segment(audio, sample_rate, start_time, end_time)
            
            if len(segment_audio) == 0:
                print(f"    Warning: Segment {idx} is empty, skipping")
                continue
            
            # Generate new segment ID: original_id_offsetms_durationms
            offset_ms = int(start_time * 1000)
            duration_ms = int((end_time - start_time) * 1000)
            new_segment_id = f"{segment_id}_{offset_ms}_{duration_ms}"
            
            # Save cut segment
            output_filename = f"{new_segment_id}.wav"
            output_path = output_dir / output_filename
            sf.write(str(output_path), segment_audio, sample_rate)
            
            output_segments.append((new_segment_id, output_path))
            print(f"    Segment {idx+1}/{len(segments)}: {new_segment_id} "
                  f"[{start_time:.3f}s-{end_time:.3f}s] speaker_{speaker_idx:02d}")
    
    return output_segments


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Cut audio segments from wav.scp using speaker segmentation model only",
    )
    
    parser.add_argument(
        "--wav-scp",
        type=str,
        required=True,
        help="Path to input wav.scp file (format: segment_id absolute_path)",
    )
    parser.add_argument(
        "--segmentation-model",
        type=str,
        required=True,
        help="Path to speaker segmentation model (e.g., model.onnx from pyannote-segmentation)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for cut audio files and diarization_wav.scp",
    )
    parser.add_argument(
        "--min-duration-on",
        type=float,
        default=0.3,
        help="Minimum segment duration in seconds. "
             "Segments shorter than this will be filtered out.",
    )
    parser.add_argument(
        "--min-duration-off",
        type=float,
        default=0.5,
        help="Minimum gap between segments to merge in seconds. "
             "If two segments from the same speaker are separated by a gap <= this value, "
             "they will be merged into one segment.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Number of threads for model inference",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for model inference",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logs",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Validate input wav.scp
    wav_scp_path = Path(args.wav_scp)
    if not wav_scp_path.is_file():
        raise FileNotFoundError(f"wav.scp file not found: {wav_scp_path}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read wav.scp
    print(f"[1/4] Reading wav.scp: {wav_scp_path}")
    segments = read_wav_scp(wav_scp_path)
    if not segments:
        raise ValueError("No segments found in wav.scp file!")
    print(f"  Found {len(segments)} audio segments")
    
    # Initialize segmentation model
    print(f"\n[2/4] Initializing speaker segmentation model...")
    print(f"  Segmentation model: {args.segmentation_model}")
    print(f"  Num threads: {args.num_threads}")
    print(f"  Batch size: {args.batch_size}")
    
    seg_m = OnnxSegmentationModel(args.segmentation_model, num_threads=args.num_threads)
    print(f"  Model sample rate: {seg_m.sample_rate} Hz")
    print(f"  Window size: {seg_m.window_size} samples ({seg_m.window_size/seg_m.sample_rate:.2f}s)")
    print(f"  Number of speakers (max): {seg_m.num_speakers}")
    print(f"  Min duration on: {args.min_duration_on}s")
    print(f"  Min duration off: {args.min_duration_off}s")
    
    # Process each segment
    print(f"\n[3/4] Processing {len(segments)} audio segments...")
    all_output_segments = []
    
    for idx, (segment_id, audio_path) in enumerate(segments, 1):
        audio_path_obj = Path(audio_path)
        if not audio_path_obj.is_file():
            print(f"  [{idx}/{len(segments)}] {segment_id}: Audio file not found: {audio_path}, skipping")
            continue
        
        print(f"  [{idx}/{len(segments)}] Processing: {segment_id}")
        try:
            output_segments = process_audio_segment(
                segment_id=segment_id,
                audio_path=audio_path,
                seg_m=seg_m,
                output_dir=output_dir,
                min_duration_on=args.min_duration_on,
                min_duration_off=args.min_duration_off,
                batch_size=args.batch_size,
            )
            all_output_segments.extend(output_segments)
        except Exception as e:
            print(f"    Error processing {segment_id}: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            continue
    
    # Write output wav.scp
    print(f"\n[4/4] Writing output wav.scp...")
    output_wav_scp_path = output_dir / "diarization_wav.scp"
    with open(output_wav_scp_path, "w", encoding="utf-8") as f:
        for segment_id, path in all_output_segments:
            f.write(f"{segment_id} {path.absolute()}\n")
    
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"  Input segments: {len(segments)}")
    print(f"  Output segments: {len(all_output_segments)}")
    print(f"  Output directory: {output_dir}")
    print(f"  Output wav.scp: {output_wav_scp_path}")
    print(f"{'='*60}")
    print(f"\nDone!")


if __name__ == "__main__":
    main()
