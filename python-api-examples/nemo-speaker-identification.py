#!/usr/bin/env python3

"""
Extract speaker embeddings from audio files listed in wav.scp using NeMo TitaNet model.

This script reads a wav.scp file (format: segment_id path/to/audio.wav),
extracts speaker embeddings for each audio file using NeMo TitaNet speaker embedding model,
and outputs the results to a text file with two columns: id and serialized embedding.

Usage example:

python3 ./python-api-examples/nemo-speaker-identification.py \
  --wav-scp /path/to/wav.scp \
  --model-name titanet_large \
  --output /path/to/output/embedding.txt

Output:
- A text file with the specified output path
- Format: <segment_id> <base64_encoded_embedding>
- Each line contains: segment_id and the serialized embedding value

Notes:
- The wav.scp file should contain lines in the format: <segment_id> <absolute_path>
- Audio files will be automatically resampled by NeMo model if needed
- Embeddings are serialized as base64-encoded float32 arrays
- Supported model names: 'titanet_large', 'titanet_small', etc.
"""

import argparse
import base64
import time
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np

try:
    import nemo.collections.asr as nemo_asr
    import torch
except ImportError:
    raise ImportError(
        "NeMo is not installed. Please install it with: pip install nemo_toolkit[all]"
    )


class SpeakerVerifier:
    def __init__(self, model_name: str = 'titanet_large', device: str = 'cpu'):
        """
        初始化说话人验证器
        
        Args:
            model_name: NeMo TitaNet模型名称，默认使用'titanet_large'
            device: 运行设备，可选 'cpu' 或 'cuda'，默认为 'cpu'
        """
        if device not in ['cpu', 'cuda']:
            raise ValueError("device must be either 'cpu' or 'cuda'")
            
        start_time = time.time()
        self.speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
            model_name=model_name
        )
        
        if device == 'cuda':
            self.speaker_model = self.speaker_model.to("cuda:0")
        else:
            self.speaker_model = self.speaker_model.to("cpu")
            
        self.model_load_time = time.time() - start_time
    
    def get_embedding(self, audio_file: Union[str, Path]) -> np.ndarray:
        """
        从音频文件中提取说话人嵌入向量
        
        Args:
            audio_file: 音频文件的路径
            
        Returns:
            说话人嵌入向量 (numpy array)
        """
        # 确保输入路径为字符串类型
        audio_file = str(audio_file)
        
        # 获取说话人嵌入向量
        embedding = self.speaker_model.get_embedding(audio_file)
        
        # 转换为 numpy array
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().numpy()
        
        # 确保是一维数组
        if embedding.ndim > 1:
            embedding = embedding.flatten()
        
        return embedding.astype(np.float32)


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


def serialize_embedding(emb: np.ndarray) -> str:
    """Serialize embedding vector to base64 string."""
    emb_bytes = emb.astype(np.float32).tobytes()
    return base64.b64encode(emb_bytes).decode('ascii')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Extract speaker embeddings from audio files in wav.scp using NeMo TitaNet"
    )
    
    parser.add_argument(
        "--wav-scp",
        type=str,
        required=True,
        help="Path to wav.scp file (format: segment_id path/to/audio.wav)"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="titanet_large",
        help="NeMo TitaNet model name (e.g., 'titanet_large', 'titanet_small')"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path (full path) for embedding file"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run inference on (cpu or cuda)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logs"
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # 1) Read wav.scp file
    wav_scp_path = Path(args.wav_scp)
    if not wav_scp_path.is_file():
        raise FileNotFoundError(f"wav.scp file not found: {wav_scp_path}")
    
    print(f"[1/4] Reading wav.scp: {wav_scp_path}")
    segments = read_wav_scp(wav_scp_path)
    print(f"      Found {len(segments)} audio files")
    
    # 2) Build speaker embedding extractor
    print(f"[2/4] Loading NeMo TitaNet model: {args.model_name}")
    print(f"      Device: {args.device}")
    try:
        verifier = SpeakerVerifier(model_name=args.model_name, device=args.device)
        print(f"      Model loaded in {verifier.model_load_time:.2f} seconds")
        # 获取嵌入向量维度（通过处理一个示例文件或直接获取）
        print(f"      Model ready")
    except Exception as e:
        raise RuntimeError(f"Failed to load NeMo model '{args.model_name}': {e}")
    
    # 3) Extract embeddings
    print(f"[3/4] Extracting embeddings from {len(segments)} audio files...")
    results = []
    failed_count = 0
    
    for idx, (segment_id, audio_path) in enumerate(segments, 1):
        audio_file = Path(audio_path)
        if not audio_file.is_file():
            print(f"      Warning: Audio file not found: {audio_path}, skipping...")
            failed_count += 1
            continue
        
        try:
            emb = verifier.get_embedding(str(audio_file))
            if emb is None or emb.size == 0:
                print(f"      Warning: Failed to extract embedding from {audio_path}, skipping...")
                failed_count += 1
                continue
            
            emb_str = serialize_embedding(emb)
            results.append((segment_id, emb_str))
        except Exception as e:
            print(f"      Warning: Error processing {audio_path}: {e}, skipping...")
            failed_count += 1
            continue
        
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
    print(f"\n=== Summary ===")
    print(f"Total audio files: {len(segments)}")
    print(f"Successfully extracted: {len(results)}")
    print(f"Failed: {failed_count}")
    print(f"Output file: {output_path}")


if __name__ == "__main__":
    main()

