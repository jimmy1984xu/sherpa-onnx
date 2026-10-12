#!/usr/bin/env python3

"""
Cluster speaker embeddings with K2 fast clustering.

This script reads embeddings, clusters them by speaker, and writes the cluster
assignment for each segment. It supports both a fixed cluster count and a
threshold-based mode.

Examples:
python3 ./python-jimmy/k2-speaker-cluster-fastclustering.py \
  --input embeddings.txt \
  --num-clusters 3 \
  --filter all

python3 ./python-jimmy/k2-speaker-cluster-fastclustering.py \
  --input embeddings.txt \
  --speaker-file speaker.txt \
  --threshold 0.5 \
  --filter no_multi

python3 ./python-jimmy/k2-speaker-cluster-fastclustering.py \
  --input embeddings.txt \
  --num-clusters 3 \
  --wav-scp wav.scp

Notes:
- Input format: <segment_id> <base64_encoded_embedding>
- Output format: <speaker_id> <segment_id>
- Segment IDs are expected to include offset and duration information
- Default output names are generated from clustering mode and min-duration-ms
"""

import argparse
import base64
import re
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import sherpa_onnx


# 常量
DEFAULT_NUM_CLUSTERS = -1  # -1表示使用阈值模式
DEFAULT_THRESHOLD = 0.5
REGISTERED_PRIMARY_RATIO = 0.70
REGISTERED_PRIMARY_AVG_THRESHOLD = 0.60
REGISTERED_PRIMARY_MAX_THRESHOLD = 0.70
REGISTERED_FALLBACK_RATIO = 0.80
REGISTERED_FALLBACK_AVG_THRESHOLD = 0.57
REGISTERED_FALLBACK_MAX_THRESHOLD = 0.68
REGISTERED_MIN_COUNT = 3
UNKNOWN_MIN_SEGMENTS = 10
UNKNOWN_MAX_SCORE_THRESHOLD = 0.55
SHORT_SEGMENT_KEEP_SCORE_THRESHOLD = 0.65


def create_skip_counters() -> Dict[str, int]:
    return {
        "multi": 0,
        "short": 0,
        "invalid_embedding": 0,
    }


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
    if len(parts) >= 5:
        try:
            # 扩展格式: prefix_offset_duration_top1speaker_top1score
            duration_ms = int(parts[-3])
            offset_ms = int(parts[-4])
            return offset_ms, duration_ms
        except ValueError:
            pass

    if len(parts) >= 3:
        try:
            # 标准格式: prefix_offset_duration
            duration_ms = int(parts[-1])
            offset_ms = int(parts[-2])
            return offset_ms, duration_ms
        except ValueError:
            pass
    
    # 备用方案：提取所有数字，取最后两个
    numbers = re.findall(r"\d+", segment_id)
    if len(numbers) >= 2:
        return int(numbers[-2]), int(numbers[-1])
    
    return 0, 0


def parse_old_speaker_score(segment_id: str) -> Tuple[Optional[str], Optional[float]]:
    """
    从扩展 segment_id 中解析旧 speaker 和旧 score。

    扩展格式:
      prefix_offset_duration_oldSpeaker_oldScore
    """
    parts = segment_id.split("_")
    if len(parts) < 5:
        return None, None

    try:
        int(parts[-4])
        int(parts[-3])
        score = float(parts[-1])
    except ValueError:
        return None, None

    old_speaker = parts[-2]
    return old_speaker, score


def parse_old_speaker_name(segment_id: str) -> Optional[str]:
    parts = segment_id.split("_")
    if len(parts) < 5:
        return None

    try:
        int(parts[-4])
        int(parts[-3])
    except ValueError:
        return None

    return parts[-2]


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

            # 从speaker_map获取说话人标识（如果提供）；否则从segment_id恢复旧speaker
            if speaker_map:
                speaker = speaker_map.get(segment_id)
            else:
                speaker = parse_old_speaker_name(segment_id)
            
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


def extract_embeddings_with_stats(
    segments: List[Dict],
    skip_counters: Optional[Dict[str, int]] = None,
) -> Tuple[np.ndarray, List[int]]:
    try:
        return extract_embeddings(segments)
    except ValueError:
        if skip_counters is not None and segments:
            skip_counters["invalid_embedding"] += len(segments)
        raise

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
    skip_counters: Optional[Dict[str, int]] = None,
) -> List[Dict]:
    """
    根据说话人标识过滤片段。
      - all: 不过滤
      - non_multi: 仅保留 speaker != 'multi' 且存在标识的片段
      - 其他字符串: 仅保留该说话人的片段
    额外过滤 durationMs >= min_duration_ms。
    短片段如果 segment_id 中的 top1 score > 0.65，也保留用于聚类。
    """
    filtered = []
    for seg in segments:
        if seg.get("speaker") == "Multi":
            if skip_counters is not None:
                skip_counters["multi"] += 1
            continue
        if seg.get("durationMs", 0) < min_duration_ms:
            _, old_score = parse_old_speaker_score(seg.get("id", ""))
            if old_score is None or old_score <= SHORT_SEGMENT_KEEP_SCORE_THRESHOLD:
                if skip_counters is not None:
                    skip_counters["short"] += 1
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
    
    格式: <speaker_id> <segment_id>
    """
    print(f"导出结果到 {output_path}...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for segment_id, speaker_id in sorted(results, key=lambda x: (x[1], x[0])):
            f.write(f"{speaker_id} {segment_id}\n")
    
    print(f"  成功导出 {len(results)} 条结果")


def format_score(value: float) -> str:
    return f"{value:.3f}"


def summarize_scores(scores: List[float]) -> str:
    if not scores:
        return "N/A"
    return (
        f"min={format_score(min(scores))},"
        f"max={format_score(max(scores))},"
        f"avg={format_score(sum(scores) / len(scores))}"
    )


def build_cluster_name_map(results: List[Tuple[str, int]]) -> Dict[int, str]:
    cluster_to_segments: Dict[int, List[str]] = {}
    for segment_id, cluster_id in results:
        cluster_to_segments.setdefault(cluster_id, []).append(segment_id)

    if not cluster_to_segments:
        return {}

    name_map: Dict[int, str] = {}
    unknown_index = 1
    for cluster_id in sorted(cluster_to_segments):
        segment_ids = cluster_to_segments[cluster_id]
        cluster_count = len(segment_ids)
        speaker_scores: Dict[str, List[float]] = {}
        all_scores: List[float] = []

        for segment_id in segment_ids:
            old_speaker, old_score = parse_old_speaker_score(segment_id)
            if old_speaker is None or old_score is None:
                continue
            speaker_scores.setdefault(old_speaker, []).append(old_score)
            all_scores.append(old_score)

        best_registered_name: Optional[str] = None
        best_registered_count = -1
        best_registered_avg = -1.0
        best_registered_max = -1.0
        for speaker, scores in speaker_scores.items():
            if not scores:
                continue

            speaker_count = len(scores)
            speaker_avg = sum(scores) / speaker_count
            speaker_max = max(scores)
            cluster_ratio = speaker_count / cluster_count
            is_registered = speaker_count >= REGISTERED_MIN_COUNT and (
                (
                    cluster_ratio > REGISTERED_PRIMARY_RATIO
                    and speaker_avg >= REGISTERED_PRIMARY_AVG_THRESHOLD
                    and speaker_max >= REGISTERED_PRIMARY_MAX_THRESHOLD
                )
                or (
                    cluster_ratio > REGISTERED_FALLBACK_RATIO
                    and speaker_avg >= REGISTERED_FALLBACK_AVG_THRESHOLD
                    and speaker_max >= REGISTERED_FALLBACK_MAX_THRESHOLD
                )
            )
            if is_registered:
                if (
                    speaker_count > best_registered_count
                    or (
                        speaker_count == best_registered_count
                        and speaker_avg > best_registered_avg
                    )
                    or (
                        speaker_count == best_registered_count
                        and speaker_avg == best_registered_avg
                        and speaker_max > best_registered_max
                    )
                ):
                    best_registered_name = speaker
                    best_registered_count = speaker_count
                    best_registered_avg = speaker_avg
                    best_registered_max = speaker_max

        if best_registered_name is not None:
            name_map[cluster_id] = best_registered_name
        elif (
            cluster_count > UNKNOWN_MIN_SEGMENTS
            and all_scores
            and max(all_scores) > UNKNOWN_MAX_SCORE_THRESHOLD
        ):
            name_map[cluster_id] = f"S{unknown_index}"
            unknown_index += 1
        else:
            name_map[cluster_id] = "invalid"

    return name_map


def build_cluster_summary_lines(
    results: List[Tuple[str, int]],
    cluster_name_map: Dict[int, str],
    input_segment_count: int,
    min_duration_ms: int,
    num_clusters_arg: int,
    threshold: float,
    skipped_multi_segments: int = 0,
    skipped_short_segments: int = 0,
    skipped_invalid_embedding_segments: int = 0,
) -> List[str]:
    if num_clusters_arg > 0:
        mode_text = f"num_clusters={num_clusters_arg}"
    else:
        mode_text = f"threshold={format_float_for_name(threshold)}"

    cluster_to_segments: Dict[int, List[str]] = {}
    for segment_id, speaker_id in results:
        cluster_to_segments.setdefault(speaker_id, []).append(segment_id)

    valid_speakers: List[str] = []
    for cluster_id in sorted(cluster_to_segments):
        cluster_name = cluster_name_map.get(cluster_id, str(cluster_id))
        if cluster_name == "invalid":
            continue
        if cluster_name not in valid_speakers:
            valid_speakers.append(cluster_name)

    lines = [
        f"# mode: {mode_text}",
        f"# min_duration_ms: {min_duration_ms}",
        f"# input_segments: {input_segment_count}",
        f"# skipped_multi_segments: {skipped_multi_segments}",
        f"# skipped_short_segments: {skipped_short_segments}",
        f"# skipped_invalid_embedding_segments: {skipped_invalid_embedding_segments}",
        f"# clustered_segments: {len(results)}",
        f"# cluster_number: {len(cluster_to_segments)}",
        f"# valid_speakers: {','.join(valid_speakers)}",
    ]

    for cluster_id in sorted(cluster_to_segments):
        segment_ids = cluster_to_segments[cluster_id]
        cluster_name = cluster_name_map.get(cluster_id, str(cluster_id))
        old_speaker_counts: Dict[str, int] = {}
        old_speaker_scores: Dict[str, List[float]] = {}

        for segment_id in segment_ids:
            old_speaker, old_score = parse_old_speaker_score(segment_id)
            if old_speaker is None or old_score is None:
                continue

            old_speaker_counts[old_speaker] = old_speaker_counts.get(old_speaker, 0) + 1
            old_speaker_scores.setdefault(old_speaker, []).append(old_score)

        if old_speaker_counts:
            per_speaker_parts = []
            for speaker, scores in sorted(
                old_speaker_scores.items(),
                key=lambda item: (-len(item[1]), item[0]),
            ):
                per_speaker_parts.append(
                    f"{speaker}(count={len(scores)},{summarize_scores(scores)})"
                )
            lines.append(
                f"# cluster_{cluster_id}: {cluster_name} {len(segment_ids)} "
                + "; ".join(per_speaker_parts)
            )
        else:
            lines.append(f"# cluster_{cluster_id}: {cluster_name} {len(segment_ids)} N/A")

    return lines


def export_results_with_summary(
    results: List[Tuple[str, int]],
    output_path: Path,
    input_segment_count: int,
    min_duration_ms: int,
    num_clusters_arg: int,
    threshold: float,
    skipped_multi_segments: int = 0,
    skipped_short_segments: int = 0,
    skipped_invalid_embedding_segments: int = 0,
) -> None:
    print(f"导出结果到 {output_path}...")
    cluster_name_map = build_cluster_name_map(results)
    summary_lines = build_cluster_summary_lines(
        results=results,
        cluster_name_map=cluster_name_map,
        input_segment_count=input_segment_count,
        min_duration_ms=min_duration_ms,
        num_clusters_arg=num_clusters_arg,
        threshold=threshold,
        skipped_multi_segments=skipped_multi_segments,
        skipped_short_segments=skipped_short_segments,
        skipped_invalid_embedding_segments=skipped_invalid_embedding_segments,
    )

    with open(output_path, 'w', encoding='utf-8') as f:
        for line in summary_lines:
            f.write(f"{line}\n")
        for segment_id, speaker_id in sorted(results, key=lambda x: (x[1], x[0])):
            f.write(f"{speaker_id} {segment_id}\n")

    print(f"  成功导出 {len(results)} 条结果")




def load_wav_scp(wav_scp: Path) -> Dict[str, Path]:
    """
    加载 wav.scp 文件。

    格式: <audio_id> <wav_path>
    """
    if not wav_scp.is_file():
        raise FileNotFoundError(f"wav.scp 文件未找到: {wav_scp}")

    print(f"加载 wav.scp: {wav_scp}")
    ans = {}
    with open(wav_scp, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split(None, 1)
            if len(parts) != 2:
                print(f"  警告: wav.scp 第 {line_num} 行格式无效，跳过: {line[:50]}...")
                continue

            audio_id, wav_path = parts
            ans[audio_id] = Path(wav_path)

    print(f"  已加载 {len(ans)} 条音频路径")
    return ans


def export_cluster_audio_files(
    results: List[Tuple[str, int]],
    wav_scp: Path,
    output_dir: Path,
) -> None:
    """
    按聚类ID导出音频文件。

    输出目录结构:
      <output_dir>/<speaker_id>/<original_audio_filename>
    """
    wav_map = load_wav_scp(wav_scp)

    print(f"按聚类ID导出音频文件到 {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing = 0
    for audio_id, speaker_id in results:
        wav_path = wav_map.get(audio_id)
        if wav_path is None:
            missing += 1
            print(f"  警告: wav.scp 中未找到音频ID: {audio_id}")
            continue

        if not wav_path.is_file():
            missing += 1
            print(f"  警告: 音频文件不存在: {wav_path}")
            continue

        cluster_dir = output_dir / str(speaker_id)
        cluster_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(wav_path, cluster_dir / wav_path.name)
        copied += 1

    print(f"  成功复制 {copied} 个音频文件")
    if missing:
        print(f"  未找到 {missing} 个音频文件")


def format_float_for_name(value: float) -> str:
    return f"{value:g}"


def build_output_tag(num_clusters: int, threshold: float, min_duration_ms: int) -> str:
    if num_clusters > 0:
        return f"k-{num_clusters}_{min_duration_ms}"
    threshold_text = format_float_for_name(threshold)
    return f"th-{threshold_text}_{min_duration_ms}"


def resolve_default_output_paths(
    input_path: Path,
    output: Optional[str],
    wav_scp: Optional[str],
    cluster_audio_output_dir: Optional[str],
    num_clusters: int,
    threshold: float,
    min_duration_ms: int,
) -> Tuple[Path, Optional[Path]]:
    tag = build_output_tag(num_clusters, threshold, min_duration_ms)
    base_dir = input_path.parent

    if output:
        output_path = Path(output)
    else:
        output_path = base_dir / f"cluster-{tag}.txt"

    if wav_scp is None:
        audio_output_dir = None
    elif cluster_audio_output_dir:
        audio_output_dir = Path(cluster_audio_output_dir)
    else:
        audio_output_dir = base_dir / f"clustered_wavs_{tag}"

    return output_path, audio_output_dir


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
        default=None,
        help="输出文件路径（完整路径）"
    )

    parser.add_argument(
        "--wav-scp",
        type=str,
        default=None,
        help="可选：wav.scp 文件路径（格式: audio_id wav_path）。"
             "提供后会自动按聚类ID复制音频文件"
    )

    parser.add_argument(
        "--cluster-audio-output-dir",
        type=str,
        default=None,
        help="可选：按聚类ID划分后的音频输出目录。默认自动生成"
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
    output_path, cluster_audio_output_dir = resolve_default_output_paths(
        input_path=input_path,
        output=args.output,
        wav_scp=args.wav_scp,
        cluster_audio_output_dir=args.cluster_audio_output_dir,
        num_clusters=args.num_clusters,
        threshold=args.threshold,
        min_duration_ms=args.min_duration_ms,
    )
    
    # 确保输出文件的父目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 0) 加载说话人标注（如果提供）
    speaker_map = None
    if args.speaker_file:
        speaker_file = Path(args.speaker_file)
        speaker_map = load_speaker_labels(speaker_file)
    
    # 1) 加载片段
    segments = load_segments(input_path, speaker_map, args.max_lines)
    input_segment_count = len(segments)
    skip_counters = create_skip_counters()
    
    # 1.1) 过滤片段（按说话人标识）
    segments = filter_segments_by_speaker(
        segments,
        args.filter,
        min_duration_ms=args.min_duration_ms,
        skip_counters=skip_counters,
    )
    print(f"  过滤后片段数: {len(segments)} (filter={args.filter}, min_duration_ms={args.min_duration_ms})")
    if not segments:
        raise ValueError("过滤后没有可用的片段，无法聚类。")
    
    # 2) 提取声纹
    embeddings_array, valid_indices = extract_embeddings_with_stats(
        segments,
        skip_counters=skip_counters,
    )
    
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
    export_results_with_summary(
        results,
        output_path,
        input_segment_count=input_segment_count,
        min_duration_ms=args.min_duration_ms,
        num_clusters_arg=args.num_clusters,
        threshold=args.threshold,
        skipped_multi_segments=skip_counters["multi"],
        skipped_short_segments=skip_counters["short"],
        skipped_invalid_embedding_segments=skip_counters["invalid_embedding"],
    )

    # 5) 可选：按聚类ID导出音频文件
    if args.wav_scp:
        export_cluster_audio_files(
            results,
            Path(args.wav_scp),
            cluster_audio_output_dir,
        )
    
    print(f"\n完成！结果已保存到 {output_path}")


if __name__ == "__main__":
    main()

