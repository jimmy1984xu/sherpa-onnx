#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
读取 embedding 文件，计算两两声纹余弦相似度并输出结果。

输入格式：
    <segment_id> <base64_encoded_embedding>

可选说话人标注文件:
    <segment_id> <speaker_label>

输出格式：
    <id1> <id2> <similarity> [same_speaker]
    相似度保留 3 位小数，空格分隔；若两端都有说话人标识，则追加 same_speaker(1/0)。

使用示例：
python3 ./python-api-examples/k2-speaker-check-similarity.py \
  --embedding /path/to/embedding.txt \
  --output /path/to/output/similarity.txt \
  --filter all \
  --min-duration-ms 0

# 使用说话人标注文件，仅统计非 multi 的说话人
python3 ./python-api-examples/k2-speaker-check-similarity.py \
  --embedding /path/to/embedding.txt \
  --speaker-file /path/to/speaker.txt \
  --output /path/to/output/similarity.txt \
  --filter no_multi \
  --min-duration-ms 2000

# 仅统计指定说话人（将 --filter 设为说话人名）
python3 ./python-api-examples/k2-speaker-check-similarity.py \
  --embedding /path/to/embedding.txt \
  --speaker-file /path/to/speaker.txt \
  --output /path/to/output/similarity.txt \
  --filter steve \
  --min-duration-ms 2000
"""

import argparse
import base64
from pathlib import Path
from itertools import combinations
from typing import Dict, Tuple, List

import numpy as np


def parse_offset_duration_from_id(seg_id: str) -> Tuple[int, int]:
    """
    从 segment_id 中解析 offset_ms 和 duration_ms。
    
    格式: <audio_base_name>_<offset_ms>_<duration_ms>
    例如: meeting_zh_925190_3750 -> offset=925190, duration=3750
    找不到则返回 (0, 0)。
    """
    parts = seg_id.split("_")
    if len(parts) >= 2:
        try:
            # 尝试解析最后两个部分为数字
            duration_ms = int(parts[-1])
            offset_ms = int(parts[-2])
            return offset_ms, duration_ms
        except ValueError:
            pass
    
    # 备用方案：正则提取数字，取最后两个
    import re
    numbers = re.findall(r"\d+", seg_id)
    if len(numbers) >= 2:
        return int(numbers[-2]), int(numbers[-1])
    return 0, 0


def load_speaker_labels(speaker_file: Path) -> Dict[str, str]:
    """
    从说话人标注文件加载 segment_id 到 speaker 的映射。
    
    格式: <segment_id> <speaker_label>
    
    返回:
        Dict[segment_id, speaker_label]
    """
    if not speaker_file.is_file():
        raise FileNotFoundError(f"说话人标注文件未找到: {speaker_file}")
    
    print(f"加载说话人标注文件: {speaker_file}")
    speaker_map = {}
    
    with open(speaker_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(None, 1)
            if len(parts) != 2:
                print(f"  警告: 第 {line_num} 行格式无效，跳过: {line[:50]}...")
                continue
            
            segment_id, speaker_label = parts
            speaker_map[segment_id] = speaker_label
    
    print(f"  已加载 {len(speaker_map)} 个说话人标注")
    return speaker_map


def read_embeddings(path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """读取 embedding 文件，返回 {id: 向量} 以及 {id: duration_ms}。"""
    if not path.is_file():
        raise FileNotFoundError(f"Embedding file not found: {path}")

    embeddings: Dict[str, np.ndarray] = {}
    durations: Dict[str, int] = {}
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
                _, duration_ms = parse_offset_duration_from_id(seg_id)
                durations[seg_id] = duration_ms
            except Exception as e:
                print(f"Warning: Line {line_num} deserialize failed: {e}")
                continue

    if not embeddings:
        raise ValueError("No valid embeddings found.")
    return embeddings, durations




def filter_embeddings_by_speaker(
    embeddings: Dict[str, np.ndarray],
    durations: Dict[str, int],
    speaker_map: Dict[str, str],
    filter_value: str,
    min_duration_ms: int = 0,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int], Dict[str, str]]:
    """
    根据说话人过滤策略筛选 embedding：
      - "all": 不过滤
      - "no_multi": 仅保留 speaker != 'multi' 且有标识的
      - 其他任意字符串: 仅保留 speaker == filter_value
    同时过滤 duration >= min_duration_ms
    返回过滤后的 embeddings、durations、speaker_map。
    """
    filtered_emb = {}
    filtered_dur = {}
    filtered_spk = {}

    for seg_id, emb in embeddings.items():
        if durations.get(seg_id, 0) < min_duration_ms:
            continue
        spk = speaker_map.get(seg_id)

        if filter_value == "all":
            pass
        elif filter_value == "no_multi":
            if spk is None or spk == "multi":
                continue
        else:
            # 其他任意字符串视为指定说话人名字
            if spk is None or spk != filter_value:
                continue

        filtered_emb[seg_id] = emb
        filtered_dur[seg_id] = durations.get(seg_id, 0)
        filtered_spk[seg_id] = spk

    return filtered_emb, filtered_dur, filtered_spk


def normalize_embeddings(embeddings: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """归一化为单位向量，避免除零。"""
    normed: Dict[str, np.ndarray] = {}
    for k, v in embeddings.items():
        norm = np.linalg.norm(v)
        if norm == 0.0:
            normed[k] = v
        else:
            normed[k] = v / norm
    return normed


def print_similarity_stats(
    results: List[Tuple[str, str, float]],
    speaker_map: Dict[str, str],
    duration_map: Dict[str, int],
    bin_step: float = 0.1,
    max_examples: int = 10,
) -> None:
    """
    打印按相似度分桶的统计信息。
    仅对同时拥有说话人标识的 pair 统计 same/diff。
    """
    # 构造分桶边界 [-1.0, 0.0, 0.1, ..., 1.0]
    import math

    edges = [-1.0] + [round(i * bin_step, 1) for i in range(0, int(math.ceil(1.0 / bin_step)) + 1)]
    bins = []
    for i in range(len(edges) - 1):
        bins.append(
            {
                "lo": edges[i],
                "hi": edges[i + 1],
                "same": 0,
                "diff": 0,
                "diff_examples": [],
                "total": 0,
            }
        )

    def find_bin(sim: float):
        for b in bins:
            if (sim >= b["lo"]) and (sim <= b["hi"] if b["hi"] == edges[-1] else sim < b["hi"]):
                return b
        return None

    total_pairs = 0
    for id1, id2, sim in results:
        spk1 = speaker_map.get(id1)
        spk2 = speaker_map.get(id2)
        # 仅统计两者都带说话人标识的情况
        if spk1 is None or spk2 is None:
            continue
        b = find_bin(sim)
        if b is None:
            continue
        same = spk1 == spk2
        if same:
            b["same"] += 1
        else:
            b["diff"] += 1
            if len(b["diff_examples"]) < max_examples:
                b["diff_examples"].append((id1, id2, sim))
        b["total"] += 1
        total_pairs += 1

    print("\n[Similarity Stats] (仅统计带说话人标识的 pair)")
    print("bin(lo,hi)  total  same  diff  same%  diff%  diff_examples<=10")
    for b in bins:
        if b["total"] == 0:
            continue
        same_pct = 100.0 * b["same"] / b["total"]
        diff_pct = 100.0 * b["diff"] / b["total"]
        examples_str = "; ".join(
            [f"{a}|{c}|{d:.3f}" for a, c, d in b["diff_examples"]]
        )
        print(
            f"{b['lo']:.1f}-{b['hi']:.1f}  {b['total']:>5}  {b['same']:>4}  {b['diff']:>4}  "
            f"{same_pct:5.1f}%  {diff_pct:5.1f}%  {examples_str}"
        )
    print(f"总计统计 pair（带说话人标识）: {total_pairs}")


def print_speaker_cross_similarity_stats(
    results: List[Tuple[str, str, float]],
    speaker_map: Dict[str, str],
) -> None:
    """
    统计每个说话人标识与其他说话人标识的相似度统计（最高、最低、平均值）。
    仅统计不同说话人之间的相似度。
    """
    from collections import defaultdict
    
    # 收集每个说话人标识与其他不同说话人标识的相似度
    speaker_similarities: Dict[str, List[float]] = defaultdict(list)
    
    # 收集不同说话人之间的 pair，用于打印最高相似度的记录
    diff_speaker_pairs: List[Tuple[str, str, str, str, float]] = []
    
    for id1, id2, sim in results:
        spk1 = speaker_map.get(id1)
        spk2 = speaker_map.get(id2)
        
        # 仅统计两者都带说话人标识且不同说话人的情况
        if spk1 is None or spk2 is None:
            continue
        if spk1 == spk2:
            continue
        
        # 双向记录：spk1 与 spk2 的相似度
        speaker_similarities[spk1].append(sim)
        speaker_similarities[spk2].append(sim)
        
        # 记录不同说话人的 pair
        diff_speaker_pairs.append((spk1, spk2, id1, id2, sim))
    
    if not speaker_similarities:
        print("\n[Speaker Cross-Similarity Stats] 没有找到不同说话人之间的 pair")
        return
    
    print("\n[Speaker Cross-Similarity Stats] (每个说话人标识与其他不同说话人的相似度)")
    print("speaker  count  max     min     avg")
    print("-" * 50)
    
    # 按说话人标识排序
    for speaker in sorted(speaker_similarities.keys()):
        sims = speaker_similarities[speaker]
        if not sims:
            continue
        max_sim = max(sims)
        min_sim = min(sims)
        avg_sim = sum(sims) / len(sims)
        print(f"{speaker:8} {len(sims):>5}  {max_sim:6.3f}  {min_sim:6.3f}  {avg_sim:6.3f}")
    
    # 打印不同说话人之间相似度最高的前10条记录
    if diff_speaker_pairs:
        # 按相似度从高到低排序
        diff_speaker_pairs.sort(key=lambda x: x[4], reverse=True)
        top_n = min(10, len(diff_speaker_pairs))
        
        print(f"\n[Top {top_n} Highest Similarity Pairs] (不同说话人之间)")
        print("speaker1  speaker2  similarity  id1  id2")
        print("-" * 80)
        for spk1, spk2, id1, id2, sim in diff_speaker_pairs[:top_n]:
            # 截断过长的 id
            id1_short = id1[:30] + "..." if len(id1) > 30 else id1
            id2_short = id2[:30] + "..." if len(id2) > 30 else id2
            print(f"{spk1:6}{spk2:6} {sim:1.3f}  {id1_short}  {id2_short}")


def compute_pairwise_similarity(normed: Dict[str, np.ndarray]) -> List[Tuple[str, str, float]]:
    """两两计算余弦相似度（向量已归一化）。"""
    items = list(normed.items())
    results: List[Tuple[str, str, float]] = []
    for (id1, v1), (id2, v2) in combinations(items, 2):
        sim = float(np.dot(v1, v2))
        results.append((id1, id2, sim))
    # 按相似度从高到低排序
    results.sort(key=lambda x: x[2], reverse=True)
    return results


def write_results(
    results: List[Tuple[str, str, float]],
    path: Path,
    duration_map: Dict[str, int],
    speaker_map: Dict[str, str],
) -> None:
    """写出结果文件。

    若两个 segment_id 均有说话人标识，则附加一列 same_speaker:
    - 1 表示同一说话人
    - 0 表示不同说话人
    否则保持三列输出。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for id1, id2, sim in results:
            spk1 = speaker_map.get(id1)
            spk2 = speaker_map.get(id2)
            if spk1 is not None and spk2 is not None:
                same = 1 if spk1 == spk2 else 0
                f.write(f"{id1} {id2} {sim:.3f} {same}\n")
            else:
                f.write(f"{id1} {id2} {sim:.3f}\n")
    print(f"Done. Wrote {len(results)} pairs to {path}")

    # 统计并打印分布信息（全量）
    print_similarity_stats(results, speaker_map, duration_map)
    
    # 统计每个说话人标识与其他说话人标识的相似度
    print_speaker_cross_similarity_stats(results, speaker_map)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Compute pairwise cosine similarity of speaker embeddings",
    )
    parser.add_argument("--embedding", type=str, required=True, help="Embedding file path")
    parser.add_argument(
        "--speaker-file",
        type=str,
        default=None,
        help="Optional: Speaker label file path (format: segment_id speaker_label). "
             "Required if using --filter other than 'all'",
    )
    parser.add_argument("--output", type=str, required=True, help="Output file path for similarities")
    parser.add_argument(
        "--filter",
        type=str,
        default="all",
        help="Speaker filter: 'all' (no filter); 'no_multi' (exclude speaker=='multi'); "
             "any other string means only that speaker. Requires --speaker-file",
    )
    parser.add_argument(
        "--min-duration-ms",
        type=int,
        default=0,
        help="Minimum duration (ms); filter out segments shorter than this (0 disables)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.min_duration_ms < 0:
        raise ValueError("--min-duration-ms must be non-negative.")

    if args.filter != "all" and args.speaker_file is None:
        raise ValueError("使用说话人过滤（--filter）时必须提供 --speaker-file")

    emb_path = Path(args.embedding)
    out_path = Path(args.output)

    # 加载说话人标注（如果提供）
    speaker_map_full = {}
    if args.speaker_file:
        speaker_file = Path(args.speaker_file)
        speaker_map_full = load_speaker_labels(speaker_file)

    print(f"[1/3] Loading embeddings: {emb_path}")
    embeddings, durations = read_embeddings(emb_path)

    # 过滤 embedding
    embeddings, durations, speaker_map = filter_embeddings_by_speaker(
        embeddings, durations, speaker_map_full, args.filter, args.min_duration_ms
    )
    print(f"      Loaded {len(embeddings)} embeddings after filter={args.filter}, min_duration_ms={args.min_duration_ms}")

    print(f"[2/3] Normalizing embeddings...")
    normed = normalize_embeddings(embeddings)

    print(f"[3/3] Computing pairwise similarities...")
    results = compute_pairwise_similarity(normed)

    write_results(results, out_path, durations, speaker_map)


if __name__ == "__main__":
    main()


