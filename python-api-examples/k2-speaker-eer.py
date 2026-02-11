#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
根据 embedding 文件和说话人标注文件计算声纹模型的 EER（Equal Error Rate）。

输入：
  1) embedding 文件：每行两列（空格分隔），第一列为音频片段 ID，第二列为声纹向量的 base64 序列化值。
  2) speaker 文件：每行两列（空格分隔），第一列为音频片段 ID，第二列为说话人名字；speaker 名为 multi 表示未确定片段，不参与 EER 计算。

流程：
  - 过滤掉 speaker 为 multi 的片段；
  - 对剩余片段两两计算声纹相似度，得到相同说话人得分数组与不同说话人得分数组；
  - 调用 bob.measure.eer 得到 EER 及最优阈值。

依赖：pip install bob.measure numpy

使用示例：
  python3 ./python-api-examples/k2-speaker-eer.py \\
    --embedding /path/to/embedding.txt \\
    --speaker-file /path/to/speaker.txt
"""

import argparse
import base64
from pathlib import Path
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np

try:
    import bob.measure
except ImportError:
    raise ImportError(
        "请安装 bob.measure: pip install bob.measure"
    )


def read_embeddings(path: Path) -> Dict[str, np.ndarray]:
    """
    读取 embedding 文件。
    格式：每行 <segment_id> <base64_encoded_embedding>，空格分隔。
    返回：{ segment_id: embedding_vector (float32 ndarray) }
    """
    if not path.is_file():
        raise FileNotFoundError(f"Embedding file not found: {path}")

    embeddings: Dict[str, np.ndarray] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                print(f"Warning: Line {line_num} format invalid, skip: {line[:50]}...")
                continue
            seg_id, emb_str = parts
            try:
                emb_bytes = base64.b64decode(emb_str.encode("ascii"))
                emb = np.frombuffer(emb_bytes, dtype=np.float32)
                embeddings[seg_id] = emb
            except Exception as e:
                print(f"Warning: Line {line_num} deserialize failed: {e}")
                continue

    if not embeddings:
        raise ValueError("No valid embeddings found.")
    return embeddings


def load_speaker_labels(speaker_file: Path) -> Dict[str, str]:
    """
    从说话人标注文件加载 segment_id -> speaker 映射。
    格式：每行 <segment_id> <speaker_label>，空格分隔。
    """
    if not speaker_file.is_file():
        raise FileNotFoundError(f"Speaker file not found: {speaker_file}")

    speaker_map: Dict[str, str] = {}
    with open(speaker_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                print(f"Warning: Line {line_num} format invalid, skip: {line[:50]}...")
                continue
            segment_id, speaker_label = parts
            speaker_map[segment_id] = speaker_label

    return speaker_map


def normalize_embeddings(embeddings: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """将向量归一化为单位向量（用于余弦相似度）。"""
    normed: Dict[str, np.ndarray] = {}
    for k, v in embeddings.items():
        norm = np.linalg.norm(v)
        if norm == 0.0:
            normed[k] = v
        else:
            normed[k] = v / norm
    return normed


def collect_eer_scores(
    normed: Dict[str, np.ndarray],
    speaker_map: Dict[str, str],
    valid_ids: List[str],
) -> Tuple[List[float], List[float]]:
    """
    对 valid_ids 中的片段两两计算余弦相似度，按相同/不同说话人分为两组得分。
    - target_scores (positive): 同一说话人片段对的相似度
    - non_target_scores (negative): 不同说话人片段对的相似度
    """
    target_scores: List[float] = []
    non_target_scores: List[float] = []

    for (id1, id2) in combinations(valid_ids, 2):
        v1 = normed[id1]
        v2 = normed[id2]
        sim = float(np.dot(v1, v2))
        if speaker_map[id1] == speaker_map[id2]:
            target_scores.append(sim)
        else:
            non_target_scores.append(sim)

    return target_scores, non_target_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Compute EER of speaker embedding model from embedding file and speaker labels.",
    )
    parser.add_argument(
        "--embedding",
        type=str,
        required=True,
        help="Path to embedding file (format: segment_id base64_embedding per line)",
    )
    parser.add_argument(
        "--speaker-file",
        type=str,
        required=True,
        help="Path to speaker file (format: segment_id speaker_name per line; 'multi' excluded)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    emb_path = Path(args.embedding)
    speaker_path = Path(args.speaker_file)

    print("[1/4] Loading embeddings...")
    embeddings = read_embeddings(emb_path)
    print(f"      Loaded {len(embeddings)} embeddings.")

    print("[2/4] Loading speaker labels...")
    speaker_map = load_speaker_labels(speaker_path)
    print(f"      Loaded {len(speaker_map)} speaker labels.")

    # 仅保留有 embedding 且 speaker 不为 multi 的片段
    valid_ids = [
        seg_id
        for seg_id in embeddings
        if speaker_map.get(seg_id) is not None and speaker_map[seg_id] != "multi"
    ]
    if len(valid_ids) < 2:
        raise ValueError(
            f"Need at least 2 segments with non-multi speaker for EER; got {len(valid_ids)}."
        )
    print(f"      Segments with non-multi speaker (used for EER): {len(valid_ids)}.")

    print("[3/4] Normalizing embeddings and computing pairwise scores...")
    normed = normalize_embeddings({k: embeddings[k] for k in valid_ids})
    target_scores, non_target_scores = collect_eer_scores(normed, speaker_map, valid_ids)

    if not target_scores or not non_target_scores:
        raise ValueError(
            "Need both same-speaker and different-speaker pairs; "
            f"target_scores={len(target_scores)}, non_target_scores={len(non_target_scores)}."
        )
    print(f"      Same-speaker pairs: {len(target_scores)}, different-speaker pairs: {len(non_target_scores)}.")

    print("[4/4] Computing EER...")
    tar = np.asarray(target_scores, dtype=np.float64)
    non = np.asarray(non_target_scores, dtype=np.float64)
    eer, threshold = bob.measure.eer(tar, non)
    print(f"\nEER = {eer:.4f}")
    print(f"Threshold = {threshold:.4f}")


if __name__ == "__main__":
    main()
