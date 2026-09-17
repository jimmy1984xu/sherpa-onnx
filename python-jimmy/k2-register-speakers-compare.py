#!/usr/bin/env python3

"""
Compare input audio embeddings with registered speakers.

This script loads known speaker embeddings, reads input embeddings from a text
file, a WAV directory, or a wav.scp file, and writes the best speaker match
for each input item.

Examples:
python3 ./python-jimmy/k2-register-speakers-compare.py \
  --known-speakers known_speakers.txt \
  --input-embedding-file audio_embeddings.txt \
  --output speaker_compare.txt

python3 ./python-jimmy/k2-register-speakers-compare.py \
  --known-speakers known_speakers.txt \
  --input-wav-scp wav.scp \
  --model /path/to/speaker_embedding.onnx \
  --output speaker_compare.txt

python3 ./python-jimmy/k2-register-speakers-compare.py \
  --known-speakers known_speakers.txt \
  --input-wav-dir ./wav_segments \
  --model /path/to/speaker_embedding.onnx \
  --output speaker_compare.txt

Notes:
- Known speaker format: <speaker_id> <serialized_embedding>
- Output format: <audio_id> <speaker_id> <similarity>
- Embeddings may be plain values, prefixed values, or base64 float32 bytes
"""

import argparse
import base64
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import sherpa_onnx
import soundfile as sf


def split_model_prefix(embedding_str: str) -> Tuple[Optional[str], str]:
    """
    Split an optional model prefix from a serialized embedding.

    Example:
      NEMO_EN_TITANET_LARGE:0.1,0.2 -> ("NEMO_EN_TITANET_LARGE", "0.1,0.2")
      0.1,0.2                     -> (None, "0.1,0.2")
    """
    if ":" not in embedding_str:
        return None, embedding_str

    prefix, value = embedding_str.split(":", 1)
    if "," in prefix or not value:
        return None, embedding_str

    return prefix, value


def deserialize_embedding(embedding_str: str) -> np.ndarray:
    """Deserialize an embedding string into a float32 numpy array."""
    _, payload = split_model_prefix(embedding_str.strip())

    if "," in payload:
        values = [float(x) for x in payload.split(",") if x.strip()]
        return np.asarray(values, dtype=np.float32)

    try:
        emb_bytes = base64.b64decode(payload.encode("ascii"), validate=True)
        return np.frombuffer(emb_bytes, dtype=np.float32)
    except Exception as e:
        raise ValueError(f"无法解析声纹向量: {embedding_str[:80]}...") from e


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

    x_src = np.linspace(
        0.0, duration, num=samples.shape[0], endpoint=False, dtype=np.float64
    )
    x_dst = np.linspace(0.0, duration, num=dst_len, endpoint=False, dtype=np.float64)
    y = np.interp(x_dst, x_src, samples.astype(np.float64))
    return y.astype(np.float32, copy=False)


def read_wav_scp(wav_scp_path: Path) -> List[Tuple[str, str]]:
    """Read wav.scp file and return list of (audio_id, audio_path)."""
    segments = []
    with open(wav_scp_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(None, 1)
            if len(parts) != 2:
                continue

            audio_id, audio_path = parts
            segments.append((audio_id, audio_path))

    return segments


def build_extractor(
    model: str, num_threads: int, provider: str, debug: bool
) -> sherpa_onnx.SpeakerEmbeddingExtractor:
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
    max_duration: Optional[float] = None,
) -> Optional[np.ndarray]:
    """Extract a speaker embedding from one audio file."""
    samples, sr = load_audio_mono_float32(audio_path)
    if sr != target_sample_rate:
        samples = resample_linear(samples, sr, target_sample_rate)

    if max_duration is not None and max_duration > 0:
        max_samples = int(max_duration * target_sample_rate)
        if len(samples) > max_samples:
            samples = samples[:max_samples]

    stream = extractor.create_stream()
    stream.accept_waveform(sample_rate=target_sample_rate, waveform=samples)
    stream.input_finished()

    if not extractor.is_ready(stream):
        return None

    emb = extractor.compute(stream)
    return np.asarray(emb, dtype=np.float32)


def read_embedding_file(
    path: Path, id_name: str, forbid_duplicate_ids: bool = False
) -> List[Tuple[str, np.ndarray]]:
    """
    Read an embedding text file.

    Each non-empty line must contain two columns:
      <id> <serialized_embedding>
    """
    if not path.is_file():
        raise FileNotFoundError(f"{id_name} 文件不存在: {path}")

    ans = []
    seen_ids = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split(None, 1)
            if len(parts) != 2:
                print(f"  警告: {path} 第 {line_num} 行格式无效，跳过: {line[:50]}...")
                continue

            item_id, emb_str = parts
            if forbid_duplicate_ids:
                prev_line = seen_ids.get(item_id)
                if prev_line is not None:
                    raise ValueError(
                        f"{path} 中存在重复的ID: {item_id}. "
                        f"首次出现于第 {prev_line} 行，重复出现于第 {line_num} 行"
                    )
                seen_ids[item_id] = line_num

            try:
                emb = deserialize_embedding(emb_str)
            except Exception as e:
                print(f"  警告: {path} 第 {line_num} 行声纹解析失败，跳过: {e}")
                continue

            if emb.size == 0:
                print(f"  警告: {path} 第 {line_num} 行声纹为空，跳过")
                continue

            ans.append((item_id, emb))

    if not ans:
        raise ValueError(f"{path} 中没有有效声纹")

    return ans


def build_manager(
    known_embeddings: List[Tuple[str, np.ndarray]],
) -> Tuple[sherpa_onnx.SpeakerEmbeddingManager, int]:
    """Register known speaker embeddings into SpeakerEmbeddingManager."""
    dim = int(known_embeddings[0][1].shape[0])
    manager = sherpa_onnx.SpeakerEmbeddingManager(dim=dim)

    registered = 0
    for speaker_id, embedding in known_embeddings:
        if embedding.shape[0] != dim:
            print(
                f"  警告: 说话人 {speaker_id} 声纹维度 {embedding.shape[0]} "
                f"!= {dim}，跳过"
            )
            continue

        if not manager.add(speaker_id, embedding):
            raise RuntimeError(f"注册说话人失败: {speaker_id}")
        registered += 1

    if registered == 0:
        raise ValueError("没有成功注册任何说话人")

    return manager, dim


def find_top1_match(
    manager: sherpa_onnx.SpeakerEmbeddingManager,
    embedding: np.ndarray,
) -> Tuple[str, float]:
    """Find the speaker with the highest SpeakerEmbeddingManager score."""
    best_speaker = ""
    best_score = float("-inf")

    for speaker_id in manager.all_speakers:
        score = float(manager.score(speaker_id, embedding))
        if score > best_score:
            best_speaker = speaker_id
            best_score = score

    return best_speaker, best_score


def compare_embeddings(
    manager: sherpa_onnx.SpeakerEmbeddingManager,
    dim: int,
    audio_embeddings: List[Tuple[str, np.ndarray]],
) -> List[Tuple[str, str, float]]:
    """Compare every audio embedding and return top1 results."""
    results = []
    skipped = 0

    for audio_id, embedding in audio_embeddings:
        if embedding.shape[0] != dim:
            skipped += 1
            print(
                f"  警告: 音频 {audio_id} 声纹维度 {embedding.shape[0]} "
                f"!= {dim}，跳过"
            )
            continue

        speaker_id, score = find_top1_match(manager, embedding)
        results.append((audio_id, speaker_id, score))

    if skipped:
        print(f"  跳过 {skipped} 条维度不匹配的音频声纹")

    return results


def load_embeddings_from_wav_dir(
    wav_dir: Path,
    extractor: sherpa_onnx.SpeakerEmbeddingExtractor,
    max_audio_duration: Optional[float],
) -> List[Tuple[str, np.ndarray]]:
    """Extract embeddings from all wav files under a directory recursively."""
    if not wav_dir.is_dir():
        raise FileNotFoundError(f"音频目录不存在: {wav_dir}")

    wav_files = sorted(wav_dir.rglob("*.wav"))
    if not wav_files:
        raise ValueError(f"{wav_dir} 下没有找到 wav 文件")

    ans = []
    for wav_path in wav_files:
        audio_id = wav_path.stem
        emb = extract_embedding_for_audio(
            extractor, str(wav_path), max_duration=max_audio_duration
        )
        if emb is None or emb.size == 0:
            print(f"  警告: 提取音频声纹失败，跳过: {wav_path}")
            continue
        ans.append((audio_id, emb))

    if not ans:
        raise ValueError(f"{wav_dir} 下没有成功提取任何音频声纹")

    return ans


def load_embeddings_from_wav_scp(
    wav_scp_path: Path,
    extractor: sherpa_onnx.SpeakerEmbeddingExtractor,
    max_audio_duration: Optional[float],
) -> List[Tuple[str, np.ndarray]]:
    """Extract embeddings from a wav.scp file."""
    if not wav_scp_path.is_file():
        raise FileNotFoundError(f"wav.scp 文件不存在: {wav_scp_path}")

    segments = read_wav_scp(wav_scp_path)
    if not segments:
        raise ValueError(f"{wav_scp_path} 中没有有效的 wav.scp 记录")

    ans = []
    for audio_id, audio_path in segments:
        wav_path = Path(audio_path)
        if not wav_path.is_file():
            print(f"  警告: 音频文件不存在，跳过: {wav_path}")
            continue

        emb = extract_embedding_for_audio(
            extractor, str(wav_path), max_duration=max_audio_duration
        )
        if emb is None or emb.size == 0:
            print(f"  警告: 提取音频声纹失败，跳过: {wav_path}")
            continue
        ans.append((audio_id, emb))

    if not ans:
        raise ValueError(f"{wav_scp_path} 中没有成功提取任何音频声纹")

    return ans


def write_results(results: List[Tuple[str, str, float]], output_path: Path) -> None:
    """Write comparison summary to a text file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for audio_id, speaker_id, score in results:
            f.write(f"{audio_id} {speaker_id} {score:.6f}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Register known speaker embeddings and compare audio embeddings",
    )

    parser.add_argument(
        "--known-speakers",
        type=str,
        required=True,
        help="已知说话人 embedding 文件，格式: speaker_id serialized_embedding",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input-embedding-file",
        type=str,
        help="待比较音频 embedding 文件，格式: audio_id serialized_embedding",
    )
    group.add_argument(
        "--input-wav-dir",
        type=str,
        help="待比较音频目录。会递归读取其中的 *.wav 文件，音频ID使用文件名 stem",
    )
    group.add_argument(
        "--input-wav-scp",
        type=str,
        help="待比较 wav.scp 文件，格式: audio_id wav_path",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出汇总 txt 文件路径，格式: audio_id speaker_id similarity",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="音频输入模式使用的 speaker embedding 模型路径",
    )

    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="音频输入模式的推理线程数",
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "coreml"],
        help="音频输入模式的推理 provider",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用 debug 日志",
    )

    parser.add_argument(
        "--max-audio-duration",
        type=float,
        default=10.0,
        help="音频输入模式下，单个音频最大参与声纹提取的时长（秒）",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    known_path = Path(args.known_speakers)
    output_path = Path(args.output)

    print(f"[1/4] 加载已知说话人声纹: {known_path}")
    known_embeddings = read_embedding_file(
        known_path, "已知说话人 embedding", forbid_duplicate_ids=True
    )
    print(f"      已加载 {len(known_embeddings)} 条已知说话人声纹")

    print("[2/4] 注册已知说话人到 SpeakerEmbeddingManager")
    manager, dim = build_manager(known_embeddings)
    print(f"      已注册 {manager.num_speakers} 个说话人，声纹维度={dim}")

    print("[3/4] 加载待比较输入")
    audio_embeddings: List[Tuple[str, np.ndarray]]
    if args.input_embedding_file:
        input_path = Path(args.input_embedding_file)
        print(f"      输入模式: embedding 文件 {input_path}")
        audio_embeddings = read_embedding_file(input_path, "音频 embedding")
    else:
        if not args.model:
            raise ValueError("使用音频输入模式时必须提供 --model")

        print(f"      构建 SpeakerEmbeddingExtractor: {args.model}")
        extractor = build_extractor(
            args.model, args.num_threads, args.provider, args.debug
        )
        max_audio_duration = (
            args.max_audio_duration if args.max_audio_duration > 0 else None
        )

        if args.input_wav_dir:
            input_path = Path(args.input_wav_dir)
            print(f"      输入模式: wav 目录 {input_path}")
            audio_embeddings = load_embeddings_from_wav_dir(
                input_path, extractor, max_audio_duration
            )
        else:
            input_path = Path(args.input_wav_scp)
            print(f"      输入模式: wav.scp 文件 {input_path}")
            audio_embeddings = load_embeddings_from_wav_scp(
                input_path, extractor, max_audio_duration
            )

    print(f"      已加载 {len(audio_embeddings)} 条音频声纹")

    print("[4/4] 逐条比较并写出结果")
    results = compare_embeddings(manager, dim, audio_embeddings)
    write_results(results, output_path)

    print(f"\n完成！成功输出 {len(results)} 条结果到 {output_path}")


if __name__ == "__main__":
    main()
