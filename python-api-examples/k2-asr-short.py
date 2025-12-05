#!/usr/bin/env python3

"""
ASR recognition for short audio segments.

This script reads a wav.scp file (format: segment_id path/to/audio.wav),
performs ASR recognition for each audio file using a specified ASR model,
and outputs the results to a text file with two columns: id and text.

Usage example:

# Using Paraformer:
python3 ./python-api-examples/k2-asr-short.py \
  --wav-scp /path/to/wav.scp \
  --paraformer /path/to/paraformer.onnx \
  --tokens /path/to/tokens.txt \
  --output /path/to/output.txt \
  --max-segments 10  # Optional: limit to 10 segments for debugging

# Using Transducer:
python3 ./python-api-examples/k2-asr-short.py \
  --wav-scp /path/to/wav.scp \
  --encoder /path/to/encoder.onnx \
  --decoder /path/to/decoder.onnx \
  --joiner /path/to/joiner.onnx \
  --tokens /path/to/tokens.txt \
  --output /path/to/output.txt

# Using NeMo Transducer (e.g., parakeet models):
python3 ./python-api-examples/k2-asr-short.py \
  --wav-scp /path/to/wav.scp \
  --encoder /path/to/encoder.onnx \
  --decoder /path/to/decoder.onnx \
  --joiner /path/to/joiner.onnx \
  --tokens /path/to/tokens.txt \
  --model-type nemo_transducer \
  --output /path/to/output.txt

Output:
- A text file at the specified output path
- Format (Paraformer/NeMo Transducer with confidence): <segment_id> <confidence> <asr_text>
- Format (other models): <segment_id> <asr_text>
- Each line contains: segment_id, confidence (0-1 range, Paraformer/NeMo Transducer only), and recognized text

Notes:
- The wav.scp file should contain lines in the format: <segment_id> <absolute_path>
- Audio files will be automatically resampled to 16kHz if needed
- Supports multiple ASR model types: transducer, paraformer, wenet-ctc, whisper, sense-voice
- Text confidence is available for Paraformer and NeMo Transducer models (based on log probabilities)
- Confidence scores range from 0.0 to 1.0, with higher values indicating higher confidence
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
import sherpa_onnx


def register_non_streaming_asr_model_args(parser):
    parser.add_argument(
        "--tokens",
        type=str,
        help="Path to tokens.txt",
    )

    parser.add_argument(
        "--encoder",
        default="",
        type=str,
        help="Path to the transducer encoder model",
    )

    parser.add_argument(
        "--decoder",
        default="",
        type=str,
        help="Path to the transducer decoder model",
    )

    parser.add_argument(
        "--joiner",
        default="",
        type=str,
        help="Path to the transducer joiner model",
    )

    parser.add_argument(
        "--paraformer",
        default="",
        type=str,
        help="Path to the model.onnx from Paraformer",
    )

    parser.add_argument(
        "--wenet-ctc",
        default="",
        type=str,
        help="Path to the CTC model.onnx from WeNet",
    )

    parser.add_argument(
        "--whisper-encoder",
        default="",
        type=str,
        help="Path to whisper encoder model",
    )

    parser.add_argument(
        "--whisper-decoder",
        default="",
        type=str,
        help="Path to whisper decoder model",
    )

    parser.add_argument(
        "--whisper-language",
        default="",
        type=str,
        help="""It specifies the spoken language in the input file.
        Example values: en, fr, de, zh, jp.
        Available languages for multilingual models can be found at
        https://github.com/openai/whisper/blob/main/whisper/tokenizer.py#L10
        If not specified, we infer the language from the input audio file.
        """,
    )

    parser.add_argument(
        "--whisper-task",
        default="transcribe",
        choices=["transcribe", "translate"],
        type=str,
        help="""For multilingual models, if you specify translate, the output
        will be in English.
        """,
    )

    parser.add_argument(
        "--whisper-tail-paddings",
        default=-1,
        type=int,
        help="""Number of tail padding frames.
        We have removed the 30-second constraint from whisper, so you need to
        choose the amount of tail padding frames by yourself.
        Use -1 to use a default value for tail padding.
        """,
    )

    parser.add_argument(
        "--decoding-method",
        type=str,
        default="greedy_search",
        help="""Valid values are greedy_search and modified_beam_search.
        modified_beam_search is valid only for transducer models.
        """,
    )

    parser.add_argument(
        "--model-type",
        type=str,
        default="",
        choices=["", "transducer", "nemo_transducer"],
        help="""Model type for transducer models.
        Use 'nemo_transducer' for NeMo Transducer models (e.g., parakeet models).
        Use 'transducer' for regular transducer models from icefall.
        If not specified, the code will try to auto-detect from the model.
        """,
    )

    parser.add_argument(
        "--feature-dim",
        type=int,
        default=80,
        help="Feature dimension. Must match the one expected by the model",
    )

    parser.add_argument(
        "--sense-voice",
        default="",
        type=str,
        help="Path to sense voice model",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="ASR recognition for short audio segments"
    )

    register_non_streaming_asr_model_args(parser)

    parser.add_argument(
        "--wav-scp",
        type=str,
        required=True,
        help="Path to wav.scp file (format: segment_id absolute_path)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path"
    )

    parser.add_argument(
        "--max-segments",
        type=int,
        default=None,
        help="Maximum number of segments to process (for debugging). If not specified, process all segments."
    )

    parser.add_argument("--num-threads", type=int, default=1, help="Threads for NN inference")
    parser.add_argument("--provider", type=str, default="cpu", choices=["cpu", "cuda", "coreml"], help="Inference provider")
    parser.add_argument("--debug", action="store_true", help="Enable debug logs")

    # Optional punctuation arguments
    parser.add_argument(
        "--punct-model",
        type=str,
        default="",
        help="Path to Offline punctuation ct-transformer model.onnx",
    )

    return parser.parse_args()


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


def assert_file_exists(filename: str):
    assert Path(filename).is_file(), (
        f"{filename} does not exist!\n"
        "Please refer to "
        "https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html to download it"
    )


def create_recognizer(args) -> sherpa_onnx.OfflineRecognizer:
    """Create ASR recognizer based on specified model type."""
    if args.encoder:
        assert len(args.paraformer) == 0, args.paraformer
        assert len(args.wenet_ctc) == 0, args.wenet_ctc
        assert len(args.whisper_encoder) == 0, args.whisper_encoder
        assert len(args.whisper_decoder) == 0, args.whisper_decoder

        assert_file_exists(args.encoder)
        assert_file_exists(args.decoder)
        assert_file_exists(args.joiner)

        # Build kwargs for from_transducer
        transducer_kwargs = {
            "encoder": args.encoder,
            "decoder": args.decoder,
            "joiner": args.joiner,
            "tokens": args.tokens,
            "num_threads": args.num_threads,
            "sample_rate": 16000,
            "feature_dim": args.feature_dim,
            "decoding_method": args.decoding_method,
            "debug": args.debug,
        }
        
        # Add model_type if specified
        if args.model_type:
            transducer_kwargs["model_type"] = args.model_type
        
        recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(**transducer_kwargs)
    elif args.paraformer:
        assert len(args.wenet_ctc) == 0, args.wenet_ctc
        assert len(args.whisper_encoder) == 0, args.whisper_encoder
        assert len(args.whisper_decoder) == 0, args.whisper_decoder

        assert_file_exists(args.paraformer)

        recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
            paraformer=args.paraformer,
            tokens=args.tokens,
            num_threads=args.num_threads,
            sample_rate=16000,
            feature_dim=args.feature_dim,
            decoding_method=args.decoding_method,
            debug=args.debug,
        )
    elif args.wenet_ctc:
        assert len(args.whisper_encoder) == 0, args.whisper_encoder
        assert len(args.whisper_decoder) == 0, args.whisper_decoder

        assert_file_exists(args.wenet_ctc)

        recognizer = sherpa_onnx.OfflineRecognizer.from_wenet_ctc(
            model=args.wenet_ctc,
            tokens=args.tokens,
            num_threads=args.num_threads,
            sample_rate=16000,
            feature_dim=args.feature_dim,
            decoding_method=args.decoding_method,
            debug=args.debug,
        )
    elif args.whisper_encoder:
        assert_file_exists(args.whisper_encoder)
        assert_file_exists(args.whisper_decoder)

        recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=args.whisper_encoder,
            decoder=args.whisper_decoder,
            tokens=args.tokens,
            num_threads=args.num_threads,
            decoding_method=args.decoding_method,
            debug=args.debug,
            language=args.whisper_language,
            task=args.whisper_task,
            tail_paddings=args.whisper_tail_paddings,
        )
    elif args.sense_voice:
        assert_file_exists(args.sense_voice)
        recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=args.sense_voice,
            tokens=args.tokens,
            num_threads=args.num_threads,
            use_itn=True,
            debug=args.debug,
        )
    else:
        raise ValueError("Please specify at least one ASR model")

    return recognizer


def create_punctuation_if_needed(args):
    """Create optional punctuation model, returns None if not configured."""
    if not args.punct_model:
        return None

    assert_file_exists(args.punct_model)

    config = sherpa_onnx.OfflinePunctuationConfig(
        model=sherpa_onnx.OfflinePunctuationModelConfig(
            ct_transformer=args.punct_model,
            num_threads=1,
            debug=False,
            provider="cpu",
        )
    )
    return sherpa_onnx.OfflinePunctuation(config)


def recognize_audio(
    recognizer: sherpa_onnx.OfflineRecognizer,
    audio_path: str,
    target_sample_rate: int = 16000,
) -> Tuple[str, float]:
    """
    Perform ASR recognition on an audio file.
    
    Returns:
        Tuple of (recognized text string, average confidence score)
        Confidence is in range [0, 1], or -1.0 if not available
    """
    # Load and resample audio
    wav, sr = load_audio_mono_float32(audio_path)
    if sr != target_sample_rate:
        wav = resample_linear(wav, sr, target_sample_rate)
    
    # Create stream and recognize
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate=target_sample_rate, waveform=wav)
    recognizer.decode_stream(stream)
    result = stream.result
    text = result.text
    
    # Get confidence directly from result (Paraformer models)
    # Confidence is already calculated in C++ code as average of token probabilities
    confidence = getattr(result, 'confidence', -1.0)
    
    # If text is empty, set confidence to 0.0 (empty text means no speech detected)
    if not text.strip():
        confidence = 0.0
    elif confidence < 0.0:
        # Model does not support confidence
        confidence = -1.0
    
    return text, confidence


def get_model_name(args) -> str:
    """Get model name for output filename."""
    if args.paraformer:
        return Path(args.paraformer).stem
    elif args.encoder:
        return Path(args.encoder).stem
    elif args.wenet_ctc:
        return Path(args.wenet_ctc).stem
    elif args.whisper_encoder:
        return Path(args.whisper_encoder).stem
    elif args.sense_voice:
        return Path(args.sense_voice).stem
    else:
        return "asr"


def main():
    args = parse_args()

    # 1) Read wav.scp file
    wav_scp_path = Path(args.wav_scp)
    if not wav_scp_path.is_file():
        raise FileNotFoundError(f"wav.scp file not found: {wav_scp_path}")
    
    print(f"[1/4] Reading wav.scp: {wav_scp_path}")
    segments = read_wav_scp(wav_scp_path)
    print(f"      Found {len(segments)} audio files")
    
    if not segments:
        raise ValueError("No audio files found in wav.scp file!")
    
    # Apply max-segments limit for debugging
    if args.max_segments is not None and args.max_segments > 0:
        original_count = len(segments)
        segments = segments[:args.max_segments]
        print(f"      Limited to {len(segments)} segments (from {original_count}) for debugging")

    # 2) Build ASR recognizer
    print(f"[2/4] Building ASR Recognizer...")
    recognizer = create_recognizer(args)
    print(f"      ASR Recognizer ready")

    punctuation = create_punctuation_if_needed(args)
    if punctuation:
        print("      Punctuation model ready")
    else:
        print("      Punctuation disabled (default)")

    # 3) Process each audio file
    print(f"[3/4] Processing audio files...")
    results: List[Tuple[str, str, float]] = []  # (segment_id, text, confidence)
    has_confidence = False
    
    for idx, (segment_id, audio_path) in enumerate(segments, 1):
        audio_file = Path(audio_path)
        if not audio_file.is_file():
            print(f"      Warning: Audio file not found: {audio_path}, skipping segment {segment_id}")
            continue
        
        try:
            text, confidence = recognize_audio(recognizer, str(audio_path), target_sample_rate=16000)
            if punctuation:
                text = punctuation.add_punctuation(text)
            
            if confidence >= 0.0:
                has_confidence = True
            
            results.append((segment_id, text, confidence))
            
            # Print progress
            if idx % max(1, len(segments) // 20) == 0 or idx == len(segments):
                pct = 100.0 * idx / len(segments)
                print(f"      Progress: {idx}/{len(segments)} ({pct:.1f}%)", end="\r")
        except Exception as e:
            print(f"      Error processing {segment_id} ({audio_path}): {e}")
            continue
    
    print()  # New line after progress

    # 4) Write output file
    print(f"[4/4] Writing output file...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for segment_id, text, confidence in results:
            if has_confidence:
                # Format: segment_id confidence text
                # Confidence: 0.0 for empty text, -1.0 if model doesn't support confidence, otherwise [0.0, 1.0]
                if confidence >= 0.0:
                    f.write(f"{segment_id} {confidence:.4f} {text}\n")
                else:
                    # Model doesn't support confidence
                    f.write(f"{segment_id} -1.0000 {text}\n")
            else:
                # Format: segment_id text (no confidence available)
                f.write(f"{segment_id} {text}\n")
    
    print(f"      Successfully exported {len(results)} results to {output_path}")
    if has_confidence:
        avg_confidence = np.mean([conf for _, _, conf in results if conf >= 0.0])
        print(f"      Average confidence: {avg_confidence:.4f}")
    print(f"\n=== Summary ===")
    print(f"Total processed: {len(results)}/{len(segments)}")
    if has_confidence:
        print(f"Confidence available: Yes (Paraformer or NeMo Transducer model)")
    else:
        print(f"Confidence available: No (model does not support confidence)")


if __name__ == "__main__":
    main()

