#!/usr/bin/env python3

"""
基于K2 Fast Clustering算法的说话人聚类。

本脚本读取声纹文件并使用K2的Fast Clustering算法进行说话人聚类，
为每个片段分配说话人ID。

输入格式: <segment_id> <base64_encoded_embedding>
segment_id 格式: <audio_base_name>_<offset_ms>_<duration_ms>

可选说话人标注文件: <segment_id> <speaker_label>

输出格式: 每行包含:
  <segment_id> <speaker_id>

使用示例:

# 固定聚类数模式，全部片段
python3 ./python-api-examples/k2-speaker-cluster-fastclustering.py \
  --input embeddings.txt \
  --output ./output/cluster-fastclustering-k-3.txt \
  --num-clusters 3 \
  --filter all

# 阈值模式，仅单说话人（过滤 multi），使用说话人标注文件
python3 ./python-api-examples/k2-speaker-cluster-fastclustering.py \
  --input embeddings.txt \
  --speaker-file speaker.txt \
  --output ./output/cluster-fastclustering-th-0.5.txt \
  --threshold 0.5 \
  --filter no_multi

# 阈值模式，仅指定说话人（将 --filter 设为说话人名）
python3 ./python-api-examples/k2-speaker-cluster-fastclustering.py \
  --input embeddings.txt \
  --speaker-file speaker.txt \
  --output ./output/cluster-fastclustering-th-0.5.txt \
  --threshold 0.5 \
  --filter steve \
  --min-duration-ms 1000

测试时限制读取行数:

python3 ./python-api-examples/k2-speaker-cluster-fastclustering.py \
  --input embeddings.txt \
  --output ./output/cluster-fastclustering-th-0.5.txt \
  --threshold 0.5 \
  --filter all \
  --max-lines 100

处理逻辑:

Fast Clustering是K2提供的快速聚类算法，支持两种模式：

1. **固定聚类数模式**（--num-clusters > 0）:
   - 指定说话人数量，算法会将所有片段聚类到指定数量的说话人
   - 适用于已知说话人数量的场景

2. **阈值模式**（--num-clusters <= 0，使用--threshold）:
   - 根据相似度阈值自动确定说话人数量
   - 相似度低于阈值的片段会被分配到不同的说话人
   - 适用于未知说话人数量的场景

核心特点:

- **快速高效**: Fast Clustering算法针对大规模数据优化，处理速度快
- **灵活配置**: 支持固定聚类数和阈值两种模式
- **自动归一化**: 算法内部会自动归一化声纹向量
- **批量处理**: 一次性处理所有片段，适合离线场景
"""

import argparse
import base64
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import sherpa_onnx


# 常量
DEFAULT_NUM_CLUSTERS = -1  # -1表示使用阈值模式
DEFAULT_THRESHOLD = 0.5


def deserialize_embedding(emb_str: str) -> np.ndarray:
    """从base64字符串反序列化声纹向量为numpy数组。"""
    try:
        emb_bytes = base64.b64decode(emb_str.encode('ascii'))
        return np.frombuffer(emb_bytes, dtype=np.float32)
    except Exception as e:
        raise ValueError(f"反序列化声纹失败: {e}") from e


def parse_offset_duration(segment_id: str) -> Tuple[int, int]:
    """
    解析 segment_id 中的 offset_ms 和 duration_ms。

    格式: <audio_base_name>_<offset_ms>_<duration_ms>
    例如:
    - meeting_zh_925190_3750  -> offset=925190, duration=3750
    - audio_123_456           -> offset=123,   duration=456
    如果未能解析，返回 (0, 0)。
    """
    parts = segment_id.split("_")
    if len(parts) >= 2:
        try:
            # 尝试解析最后两个部分为数字
            duration_ms = int(parts[-1])
            offset_ms = int(parts[-2])
            return offset_ms, duration_ms
        except ValueError:
            pass
    
    # 备用方案：提取所有数字，取最后两个
    import re
    numbers = re.findall(r"\d+", segment_id)
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


def load_segments(input_path: Path, speaker_map: Optional[Dict[str, str]] = None, max_lines: int = None) -> List[Dict]:
    """
    从txt文件加载片段。
    
    格式: <segment_id> <base64_encoded_embedding>
    segment_id 格式: <audio_base_name>_<offset_ms>_<duration_ms>
    
    参数:
        input_path: 输入文件路径
        speaker_map: 可选的 segment_id -> speaker_label 映射
        max_lines: 最大读取行数（可选，None表示全部读取）
    """
    if not input_path.is_file():
        raise FileNotFoundError(f"输入文件未找到: {input_path}")
    
    print(f"从 {input_path} 加载片段...")
    if max_lines is not None:
        print(f"  限制读取行数: {max_lines}")
    segments = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            # 如果设置了最大行数限制，检查是否已达到
            if max_lines is not None and line_num > max_lines:
                break
            
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(None, 1)
            if len(parts) != 2:
                print(f"  警告: 第 {line_num} 行格式无效，跳过: {line[:50]}...")
                continue
            
            segment_id, emb_str = parts
            
            # 从segment_id解析offset_ms和duration_ms
            offset_ms, duration_ms = parse_offset_duration(segment_id)
            if offset_ms == 0 and duration_ms == 0:
                print(f"  警告: 无法从segment_id '{segment_id}'解析offset/duration，使用0")

            # 从speaker_map获取说话人标识（如果提供）
            speaker = speaker_map.get(segment_id) if speaker_map else None
            
            segment_data = {
                "id": segment_id,
                "offsetMs": offset_ms,
                "durationMs": duration_ms,
                "speaker": speaker,
                "embedding": emb_str,
            }
            segments.append(segment_data)
    
    if not segments:
        raise ValueError("输入文件中未找到片段！")
    
    print(f"  已加载 {len(segments)} 个片段")
    return segments


def extract_embeddings(segments: List[Dict]) -> Tuple[np.ndarray, List[int]]:
    """
    从片段中提取声纹向量。
    
    返回:
        (embeddings_array, valid_indices) 元组
        - embeddings_array: 声纹数组 (num_valid x embedding_dim)
        - valid_indices: 具有有效声纹的片段索引列表
    """
    print("提取声纹向量...")
    embeddings = []
    valid_indices = []
    
    for idx, seg in enumerate(segments):
        if "embedding" not in seg:
            print(f"  警告: 片段 {idx} 缺少声纹，跳过")
            continue
        
        try:
            emb = deserialize_embedding(seg["embedding"])
            embeddings.append(emb)
            valid_indices.append(idx)
        except Exception as e:
            print(f"  警告: 片段 {idx} 反序列化声纹失败: {e}")
            continue
    
    if not embeddings:
        raise ValueError("未找到有效的声纹！")
    
    embeddings_array = np.array(embeddings, dtype=np.float32)
    print(f"  已提取 {len(embeddings)} 个声纹，维度={embeddings_array.shape[1]}")
    
    return embeddings_array, valid_indices

def perform_fast_clustering(
    segments: List[Dict],
    embeddings_array: np.ndarray,
    valid_indices: List[int],
    num_clusters: int,
    threshold: float,
    verbose: bool = False,
) -> List[Tuple[str, int]]:
    """
    使用K2 Fast Clustering算法进行说话人聚类。
    
    返回:
        (segment_id, speaker_id) 元组列表
    """
    print(f"使用Fast Clustering算法进行聚类...")
    
    # 配置聚类参数
    if num_clusters > 0:
        print(f"  模式: 固定聚类数，num_clusters={num_clusters}")
        clustering_config = sherpa_onnx.FastClusteringConfig(num_clusters=num_clusters)
    else:
        print(f"  模式: 阈值模式，threshold={threshold}")
        clustering_config = sherpa_onnx.FastClusteringConfig(threshold=threshold)
    
    clustering = sherpa_onnx.FastClustering(clustering_config)
    
    # Fast Clustering期望特征为行主序格式 (num_segments x embedding_dim)
    # 数组会被原地修改（归一化）
    start_time = time.time()
    cluster_labels_list = clustering(embeddings_array)
    elapsed_time = time.time() - start_time
    
    num_clusters_found = len(set(cluster_labels_list))
    print(f"  聚类完成: 找到 {num_clusters_found} 个说话人，耗时 {elapsed_time:.3f}秒，处理片段数 {len(cluster_labels_list)}")
    
    # 将cluster_label重映射为连续的speaker_id（从1开始）
    unique_labels = sorted(set(cluster_labels_list))
    label_to_speaker = {label: idx + 1 for idx, label in enumerate(unique_labels)}
    
    # 生成最终结果
    final_results = []
    for valid_idx, cluster_label in zip(valid_indices, cluster_labels_list):
        segment_id = segments[valid_idx].get("id", "")
        speaker_id = label_to_speaker[cluster_label]
        final_results.append((segment_id, speaker_id))
    
    # 打印统计信息
    print(f"\n  === 聚类统计 ===")
    print(f"  总片段数: {len(final_results)}")
    print(f"  总说话人数: {num_clusters_found}")
    
    if verbose:
        print(f"\n  === 说话人分布 ===")
        speaker_counts: Dict[int, int] = {}
        for _, speaker_id in final_results:
            speaker_counts[speaker_id] = speaker_counts.get(speaker_id, 0) + 1
        
        for speaker_id in sorted(speaker_counts.keys()):
            count = speaker_counts[speaker_id]
            percentage = 100.0 * count / len(final_results) if final_results else 0.0
            print(f"    说话人 {speaker_id}: {count} 个片段 ({percentage:.1f}%)")

        # 每个聚类ID中，原始segment_id自带的说话人标识统计
        print(f"\n  === 聚类ID内标注说话人标识统计 ===")
        cluster_speaker_labels: Dict[int, Dict[str, int]] = {}
        for idx, (_, speaker_id) in enumerate(final_results):
            seg = segments[valid_indices[idx]]
            spk_label = seg.get("speaker")
            if spk_label is None:
                continue
            if speaker_id not in cluster_speaker_labels:
                cluster_speaker_labels[speaker_id] = {}
            cluster_speaker_labels[speaker_id][spk_label] = (
                cluster_speaker_labels[speaker_id].get(spk_label, 0) + 1
            )

        for speaker_id in sorted(cluster_speaker_labels.keys()):
            label_stats = cluster_speaker_labels[speaker_id]
            label_parts = [f"{label}:{cnt}" for label, cnt in sorted(label_stats.items(), key=lambda x: (-x[1], x[0]))]
            label_str = ", ".join(label_parts)
            print(f"    说话人 {speaker_id}: {label_str}")
    
    return final_results


def filter_segments_by_speaker(
    segments: List[Dict],
    mode: str,
    min_duration_ms: int = 0,
) -> List[Dict]:
    """
    根据说话人标识过滤片段。
      - all: 不过滤
      - non_multi: 仅保留 speaker != 'multi' 且存在标识的片段
      - 其他字符串: 仅保留该说话人的片段
    额外过滤 durationMs >= min_duration_ms（两侧都要满足）。
    """
    if mode == "all" and min_duration_ms <= 0:
        return segments

    filtered = []
    for seg in segments:
        if seg.get("durationMs", 0) < min_duration_ms:
            continue
        spk = seg.get("speaker")
        if mode == "no_multi":
            # 仅过滤掉明确标为 multi 的记录，未标注说话人也保留
            if spk == "multi":
                continue
        elif mode != "all":
            # mode 即为指定说话人
            if spk is None or spk != mode:
                continue
        filtered.append(seg)
    return filtered


def export_results(results: List[Tuple[str, int]], output_path: Path) -> None:
    """
    导出结果到文本文件。
    
    格式: <segment_id> <speaker_id>
    """
    print(f"导出结果到 {output_path}...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for segment_id, speaker_id in results:
            f.write(f"{segment_id} {speaker_id}\n")
    
    print(f"  成功导出 {len(results)} 条结果")




def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="基于K2 Fast Clustering算法的说话人聚类"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入txt文件路径（格式: segment_id base64_embedding）"
    )
    
    parser.add_argument(
        "--speaker-file",
        type=str,
        default=None,
        help="可选：说话人标注文件路径（格式: segment_id speaker_label）。"
             "如果不提供，则无法使用说话人过滤功能"
    )
    
    parser.add_argument(
        "--filter",
        type=str,
        default="all",
        help="说话人过滤：all(不过滤); no_multi(过滤multi，仅单说话人); "
             "其他任意字符串表示仅保留该说话人。需要配合 --speaker-file 使用"
    )
    
    parser.add_argument(
        "--min-duration-ms",
        type=int,
        default=0,
        help="最小保留时长（毫秒），过滤掉时长小于该值的片段，默认0不过滤"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出文件路径（完整路径）"
    )
    
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=DEFAULT_NUM_CLUSTERS,
        help="说话人数量（固定聚类数模式）。如果 > 0，使用固定聚类数模式；"
             "如果 <= 0，使用阈值模式（需要指定--threshold）"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="相似度阈值（阈值模式）。仅在--num-clusters <= 0时使用。"
             "值越小，产生的说话人数量越多。"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="启用详细输出，显示详细的处理信息"
    )
    
    parser.add_argument(
        "--max-lines",
        type=int,
        default=None,
        help="测试参数：限制从输入文件中读取的最大行数（可选，默认全部读取）"
    )
    
    return parser.parse_args()


def main():
    """主函数。"""
    args = parse_args()
    
    if args.min_duration_ms < 0:
        raise ValueError("--min-duration-ms 不能为负数")
    
    # 验证参数
    if args.num_clusters > 0 and args.threshold != DEFAULT_THRESHOLD:
        print(f"  警告: 使用固定聚类数模式时，--threshold参数将被忽略")
    
    if args.filter != "all" and args.speaker_file is None:
        raise ValueError("使用说话人过滤（--filter）时必须提供 --speaker-file")
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # 确保输出文件的父目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 0) 加载说话人标注（如果提供）
    speaker_map = None
    if args.speaker_file:
        speaker_file = Path(args.speaker_file)
        speaker_map = load_speaker_labels(speaker_file)
    
    # 1) 加载片段
    segments = load_segments(input_path, speaker_map, args.max_lines)
    
    # 1.1) 过滤片段（按说话人标识）
    segments = filter_segments_by_speaker(
        segments,
        args.filter,
        min_duration_ms=args.min_duration_ms,
    )
    print(f"  过滤后片段数: {len(segments)} (filter={args.filter}, min_duration_ms={args.min_duration_ms})")
    if not segments:
        raise ValueError("过滤后没有可用的片段，无法聚类。")
    
    # 2) 提取声纹
    embeddings_array, valid_indices = extract_embeddings(segments)
    
    # 3) 使用Fast Clustering进行聚类
    results = perform_fast_clustering(
        segments,
        embeddings_array,
        valid_indices,
        args.num_clusters,
        args.threshold,
        args.verbose,
    )
    
    # 4) 导出结果
    export_results(results, output_path)
    
    print(f"\n完成！结果已保存到 {output_path}")


if __name__ == "__main__":
    main()

