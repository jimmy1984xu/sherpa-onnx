#!/usr/bin/env python3

"""
基于Fast Clustering和实时处理的混合说话人聚类算法。

本脚本结合Fast Clustering的准确性和实时处理的实时性，实现平衡的说话人聚类。

输入格式: <segment_id> <base64_encoded_embedding>
segment_id 格式: <audio_base_name>_<offset_ms>_<duration_ms>

输出格式: 每行包含:
  <segment_id> <speaker_id>

使用示例:
python3 ./python-api-examples/k2-speaker-cluster-realtime-fastclustering.py \
  --input embeddings.txt \
  --output-dir ./output \
  --threshold 0.75 \
  --merge-threshold 0.8 \
  --cluster-interval 50 \
  --fastclustering-threshold 0.8 \
  --min-duration-seconds 3.0 \
  --max-embeddings-per-speaker 10 \
  --max-lines 100

处理逻辑:

1. **实时处理**: 每个片段逐个处理，与所有已存在的说话人进行声纹比较
   - 如果相似度 >= threshold: 分配到对应说话人ID
     * 如果相似度 >= merge-threshold 且时长 >= min-duration-seconds 且该说话人未冻结且声纹列表未满:
       将当前声纹添加到该说话人的声纹列表中，并刷新SpeakerEmbeddingManager中的均值声纹
     * 如果声纹列表已满（达到max-embeddings-per-speaker）: 标记为frozen状态，不再记录声纹列表
   - 如果相似度 < threshold:
     * 如果与上个片段的相似度 >= threshold: 设置为上个说话人ID（原因: 沿用(相似)）
     * 否则，如果片段时长 < min-duration-seconds: 设置为上个说话人ID（原因: 沿用(短句)）
     * 否则: 创建新的说话人ID（基于manager当前说话人数 + 1，确保与定期聚类保持一致）

2. **定期批量聚类**: 每隔N个片段（cluster-interval）调用Fast Clustering进行批量聚类
   - 对所有已处理的片段（历史+新增）执行Fast Clustering，重新刷新所有片段的说话人ID
   - 根据聚类结果刷新每个说话人的声纹列表:
     * 如果声纹数量 > max-embeddings-per-speaker: 标记为frozen状态，清空声纹列表，不刷新manager（降低内存开销）
     * 否则: 更新声纹列表，刷新SpeakerEmbeddingManager中的均值声纹
   - 实时阶段继续使用最新的均值声纹提高匹配精度

3. **声纹管理**: 
   - 维护每个说话人的声纹列表（最多max-embeddings-per-speaker个）
   - 使用merge-threshold控制加入声纹列表的相似度要求（通常 >= threshold，确保只有高相似度的声纹才会被加入）
   - 超过max-embeddings-per-speaker后，说话人标记为frozen状态，不再记录声纹列表，降低内存开销
   - 使用SpeakerEmbeddingManager管理每个说话人的均值声纹，用于实时匹配

核心特点:

- **实时性和准确性平衡**: 
  - 实时处理保证低延迟
  - 定期批量聚类保证准确性，纠正可能的错误分配

- **自适应更新**: 
  - Fast Clustering会重新分配说话人ID，纠正实时处理中的错误
  - 均值声纹会根据聚类结果自动更新，提高后续匹配准确性

- **灵活配置**: 
  - cluster-interval: 控制批量聚类的频率（值越大，实时性越好，但准确性可能降低）
  - threshold: 控制实时匹配的相似度阈值
  - fastclustering-threshold: 控制Fast Clustering的阈值（仅在阈值模式使用）
"""

import argparse
import base64
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import sherpa_onnx


# Constants
FIRST_SPEAKER_ID = 1
INVALID_SPEAKER_ID = -1
DEFAULT_THRESHOLD = 0.7
DEFAULT_CLUSTER_INTERVAL = 50
DEFAULT_FASTCLUSTERING_THRESHOLD = 0.5
DEFAULT_MAX_EMBEDDINGS_PER_SPEAKER = 10
DEFAULT_MERGE_THRESHOLD = 0.8
DEFAULT_MIN_DURATION_SECONDS = 3.0


def deserialize_embedding(emb_str: str) -> np.ndarray:
    """从base64字符串反序列化声纹向量为numpy数组。"""
    try:
        emb_bytes = base64.b64decode(emb_str.encode('ascii'))
        return np.frombuffer(emb_bytes, dtype=np.float32)
    except Exception as e:
        raise ValueError(f"反序列化声纹失败: {e}") from e


def normalize_embedding(emb: np.ndarray) -> np.ndarray:
    """将声纹向量归一化为单位长度。"""
    norm = np.linalg.norm(emb)
    if norm == 0.0:
        return emb
    return emb / norm


def extract_offset_duration(segment_id: str) -> str:
    """
    从segment_id中提取offset_duration部分。
    
    格式: audio_base_name_offsetms_durationms
    返回: offsetms_durationms 或原始segment_id（如果无法解析）
    """
    if not segment_id:
        return "未知"
    
    try:
        # 按下划线分割并获取最后两部分
        id_parts = segment_id.rsplit('_', 2)
        if len(id_parts) >= 3:
            # 返回最后两部分，用下划线连接
            return f"{id_parts[-2]}_{id_parts[-1]}"
        else:
            # 如果无法解析，返回原始segment_id
            return segment_id
    except Exception:
        # 如果解析失败，返回原始segment_id
        return segment_id


class HybridClusteringState:
    """
    管理混合聚类算法的状态。
    
    属性:
        manager: SpeakerEmbeddingManager实例，用于声纹操作
        threshold: 实时匹配的相似度阈值
        cluster_interval: 批量聚类的间隔（每隔N个片段）
        fastclustering_threshold: Fast Clustering的阈值
        min_duration_seconds: 创建新说话人所需的最小时长
        max_embeddings_per_speaker: 每个说话人最多声纹数量
        merge_threshold: 加入到声纹列表的相似度阈值（>=threshold）
        all_segments: 已处理片段列表（历史+新增），用于批量聚类
        results: 最终结果列表
        speaker_embeddings_map: 每个说话人的声纹列表
        speaker_frozen: 已超过max_embeddings_per_speaker个声纹的说话人ID集合（不再记录声纹列表，降低内存开销）
    """
    
    def __init__(
        self,
        embedding_dim: int,
        threshold: float,
        cluster_interval: int,
        fastclustering_threshold: float,
        min_duration_seconds: float,
        max_embeddings_per_speaker: int,
        merge_threshold: float,
    ):
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim必须为正数，得到 {embedding_dim}")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold必须在0.0和1.0之间，得到 {threshold}")
        if cluster_interval <= 0:
            raise ValueError(f"cluster_interval必须为正数，得到 {cluster_interval}")
        if not 0.0 <= fastclustering_threshold <= 1.0:
            raise ValueError(f"fastclustering_threshold必须在0.0和1.0之间，得到 {fastclustering_threshold}")
        if min_duration_seconds < 0.0:
            raise ValueError(f"min_duration_seconds不能为负数，得到 {min_duration_seconds}")
        if max_embeddings_per_speaker <= 0:
            raise ValueError(f"max_embeddings_per_speaker必须为正数，得到 {max_embeddings_per_speaker}")
        if not 0.0 <= merge_threshold <= 1.0:
            raise ValueError(f"merge_threshold必须在0.0和1.0之间，得到 {merge_threshold}")
        if merge_threshold < threshold:
            raise ValueError(f"merge_threshold ({merge_threshold}) 应该 >= threshold ({threshold})")
        
        self.manager = sherpa_onnx.SpeakerEmbeddingManager(dim=embedding_dim)
        self.threshold = threshold
        self.cluster_interval = cluster_interval
        self.fastclustering_threshold = fastclustering_threshold
        self.min_duration_seconds = min_duration_seconds
        self.max_embeddings_per_speaker = max_embeddings_per_speaker
        self.merge_threshold = merge_threshold
        
        # 所有已处理的片段：存储 (segment_id, embedding, current_speaker_id) 元组
        # 包括历史片段和新增片段，用于完整聚类
        self.all_segments: List[Tuple[str, np.ndarray, int]] = []
        
        # 最终结果：存储 (segment_id, speaker_id) 元组
        self.results: List[Tuple[str, int]] = []
        self.prev_speaker_id = INVALID_SPEAKER_ID
        
        # 每个说话人的声纹列表（最多15个）
        self.speaker_embeddings_map: Dict[int, List[np.ndarray]] = {}
        
        # 已超过15个声纹的说话人ID集合（不再记录声纹列表，降低内存开销）
        self.speaker_frozen: set[int] = set()


def allocate_new_speaker_id(state: HybridClusteringState) -> int:
    """
    根据当前manager中的说话人数量生成新的说话人ID，确保不与现有ID冲突。
    """
    return len(state.manager.all_speakers) + 1


def perform_fast_clustering_on_all_segments(
    all_segments: List[Tuple[str, np.ndarray, int]],
    fastclustering_threshold: float,
) -> List[Tuple[str, int, int]]:
    """
    对所有片段执行Fast Clustering。
    
    参数:
        all_segments: 所有已处理的片段，包含 (segment_id, embedding, current_speaker_id) 元组
        fastclustering_threshold: Fast Clustering的阈值
    
    返回:
        List[Tuple[segment_id, new_speaker_id, old_speaker_id]]
    """
    if not all_segments:
        return []
    
    # 提取所有声纹向量
    embeddings_list = [emb for _, emb, _ in all_segments]
    embeddings_array = np.array(embeddings_list, dtype=np.float32)
    
    # 配置Fast Clustering（使用阈值模式）
    clustering_config = sherpa_onnx.FastClusteringConfig(threshold=fastclustering_threshold)
    clustering = sherpa_onnx.FastClustering(clustering_config)
    
    # 执行聚类
    start_time = time.time()
    cluster_labels_list = clustering(embeddings_array)
    elapsed_time = time.time() - start_time
    
    # 将cluster_label重映射为连续的speaker_id（从1开始）
    unique_labels = sorted(set(cluster_labels_list))
    label_to_speaker = {label: idx + 1 for idx, label in enumerate(unique_labels)}
    
    num_clusters_found = len(unique_labels)
    print(f"  Fast Clustering完成: 找到 {num_clusters_found} 个说话人，耗时 {elapsed_time:.3f}秒，处理片段数 {len(cluster_labels_list)}")
    
    # 生成结果：每个片段的新说话人ID
    clustering_results = []
    for (segment_id, _, old_speaker_id), cluster_label in zip(all_segments, cluster_labels_list):
        new_speaker_id = label_to_speaker[cluster_label]
        clustering_results.append((segment_id, new_speaker_id, old_speaker_id))
    
    return clustering_results


def update_speaker_embeddings_from_clustering(
    state: HybridClusteringState,
    clustering_results: List[Tuple[str, int, int]],
    all_segments: List[Tuple[str, np.ndarray, int]],
) -> None:
    """
    根据Fast Clustering的结果更新说话人的均值声纹。
    
    参数:
        state: HybridClusteringState对象
        clustering_results: Fast Clustering的结果，包含 (segment_id, new_speaker_id, old_speaker_id)
        all_segments: 所有已处理的片段，包含 (segment_id, embedding, current_speaker_id) 元组
    """
    # 先清理所有状态
    # 1. 清空manager中的所有说话人
    all_speakers = list(state.manager.all_speakers)
    for speaker_name in all_speakers:
        state.manager.remove(speaker_name)
    
    # 2. 清空speaker_frozen和speaker_embeddings_map
    state.speaker_frozen.clear()
    state.speaker_embeddings_map.clear()
    
    # 根据聚类结果重新设置状态
    # 创建映射：segment_id -> embedding
    segment_to_embedding = {seg_id: emb for seg_id, emb, _ in all_segments}
    
    # 按说话人ID分组（基于聚类结果）
    speaker_embeddings_map: Dict[int, List[np.ndarray]] = {}
    for segment_id, new_speaker_id, _ in clustering_results:
        if segment_id in segment_to_embedding:
            emb = segment_to_embedding[segment_id]
            if new_speaker_id not in speaker_embeddings_map:
                speaker_embeddings_map[new_speaker_id] = []
            speaker_embeddings_map[new_speaker_id].append(emb)
    
    # 根据聚类结果重新设置每个说话人的状态
    for speaker_id, embeddings_list in speaker_embeddings_map.items():
        # 添加到manager（使用完整列表，manager会自动计算均值）
        state.manager.add(str(speaker_id), embeddings_list)
        # 如果声纹数量超过max_embeddings_per_speaker，标记为frozen状态，不添加到manager
        if len(embeddings_list) > state.max_embeddings_per_speaker:
            state.speaker_frozen.add(speaker_id)
            # 不记录声纹列表，降低内存开销
        else:
            # 更新声纹列表
            state.speaker_embeddings_map[speaker_id] = embeddings_list
    
    print(f"  已更新 {len(speaker_embeddings_map)} 个说话人的均值声纹")


def identify_speaker_realtime(
    embedding: np.ndarray,
    segment_id: str,
    duration_ms: float,
    state: HybridClusteringState,
    verbose: bool = False,
) -> int:
    """
    实时识别片段的说话人ID。
    
    参数:
        embedding: 归一化后的声纹向量
        segment_id: 片段ID
        duration_ms: 片段时长（毫秒）
        state: HybridClusteringState对象
        verbose: 是否打印详细信息
    
    返回:
        speaker_id: 说话人ID（>=1）
    """
    all_speakers = state.manager.all_speakers
    speaker_similarities: Dict[int, float] = {}
    best_speaker_id = INVALID_SPEAKER_ID
    best_similarity = -1.0
    
    # 计算与所有说话人的相似度
    if all_speakers:
        for speaker_name in all_speakers:
            speaker_id = int(speaker_name)
            sim = state.manager.score(speaker_name, embedding)
            speaker_similarities[speaker_id] = sim
            if sim > best_similarity:
                best_similarity = sim
                best_speaker_id = speaker_id
    
    # 判断是否达到阈值
    is_match = (best_similarity >= state.threshold and best_speaker_id >= FIRST_SPEAKER_ID)
    
    # 计算与上一个片段的相似度（如果有）
    last_similarity = -1.0
    if state.all_segments:
        # 获取所有片段中最后一个片段的声纹
        _, last_embedding, _ = state.all_segments[-1]
        last_similarity = float(np.dot(embedding, last_embedding))
    
    # 计算是否为长片段
    duration_seconds = duration_ms / 1000.0
    is_long_segment = (
        duration_seconds >= state.min_duration_seconds
        if state.min_duration_seconds > 0
        else True
    )
    
    has_prev_speaker = state.prev_speaker_id >= FIRST_SPEAKER_ID
    
    # 确定最终分配的说话人ID
    embedding_action = "无"
    if is_match:
        final_speaker_id = best_speaker_id
        reason = "匹配"
        
        # 如果匹配到说话人，且相似度>=merge_threshold，且时长>=min_duration_seconds
        # 且该说话人未冻结，且声纹列表未满，则添加到声纹列表并刷新manager
        can_merge = (best_similarity >= state.merge_threshold and is_long_segment and final_speaker_id not in state.speaker_frozen)
        if can_merge:
            if final_speaker_id not in state.speaker_embeddings_map:
                state.speaker_embeddings_map[final_speaker_id] = []
            
            emb_list = state.speaker_embeddings_map[final_speaker_id]
            if len(emb_list) < state.max_embeddings_per_speaker:
                emb_list.append(embedding)
                # 刷新manager中的声纹（使用完整列表，manager会自动计算均值）
                state.manager.remove(str(final_speaker_id))
                state.manager.add(str(final_speaker_id), emb_list)
                embedding_action = "更新列表"
            # 如果声纹列表已满，标记为frozen状态，不再记录声纹列表
            elif len(emb_list) >= state.max_embeddings_per_speaker:
                state.speaker_frozen.add(final_speaker_id)
                state.speaker_embeddings_map[final_speaker_id] = []  # 清空列表，只保留状态标识，降低内存开销
                embedding_action = "已冻结"
    else:
        # 匹配不到最佳说话人时，先检查与上个片段的相似度
        if (
            has_prev_speaker
            and last_similarity >= 0
            and last_similarity >= state.threshold
        ):
            # 与上个片段的相似度 >= threshold，设置为上个说话人ID
            final_speaker_id = state.prev_speaker_id
            reason = "沿用(相似)"
        elif has_prev_speaker and not is_long_segment:
            # 时长 < min-duration-seconds，设置为上个说话人ID
            final_speaker_id = state.prev_speaker_id
            reason = "沿用(短句)"
        else:
            # 时长 >= min-duration-seconds，创建新说话人
            final_speaker_id = allocate_new_speaker_id(state)
            reason = "创建"
            # 初始化新说话人的声纹列表
            state.speaker_embeddings_map[final_speaker_id] = [embedding]
            # 将新说话人添加到manager
            state.manager.add(str(final_speaker_id), embedding)
            embedding_action = "创建"
    
    # 打印关键日志
    sim_str = ", ".join([f"s{sid}={sim:.3f}" for sid, sim in sorted(speaker_similarities.items())])
    if not speaker_similarities:
        sim_str = "无说话人"
    segment_display = extract_offset_duration(segment_id)
    segment_display_fixed = segment_display.ljust(15)
    last_str = f"last={last_similarity:.3f}" if last_similarity >= 0 else "last=N/A"
    print(f"  {segment_display_fixed} 分配:s{final_speaker_id} 原因:{reason} [{last_str} {sim_str}] 声纹操作:{embedding_action}")
    
    state.prev_speaker_id = final_speaker_id
    return final_speaker_id


def process_segments_hybrid(
    segments: List[Dict],
    state: HybridClusteringState,
    verbose: bool = False,
) -> List[Tuple[str, int]]:
    """
    使用混合算法处理所有片段。
    
    参数:
        segments: 片段列表
        state: HybridClusteringState对象
        verbose: 是否打印详细信息
    
    返回:
        (segment_id, speaker_id) 元组列表
    """
    print(f"开始混合聚类处理（cluster_interval={state.cluster_interval}）...")
    
    for idx, seg in enumerate(segments):
        segment_id = seg.get("id", "")
        
        # 反序列化声纹
        try:
            embedding = deserialize_embedding(seg["embedding"])
            embedding = normalize_embedding(embedding)
        except Exception as e:
            print(f"  警告: 片段 {idx} 处理失败: {e}，跳过")
            continue
        
        duration_ms = float(seg.get("durationMs", 0.0))
        
        # 实时识别说话人ID
        speaker_id = identify_speaker_realtime(
            embedding,
            segment_id,
            duration_ms,
            state,
            verbose,
        )
        
        # 添加到所有片段列表（历史+新增）和results
        state.all_segments.append((segment_id, embedding, speaker_id))
        state.results.append((segment_id, speaker_id))
        
        # 检查是否需要执行批量聚类（每隔N个片段）
        if len(state.all_segments) % state.cluster_interval == 0:
            print(f"\n  达到聚类间隔（已处理 {len(state.all_segments)} 个片段），执行Fast Clustering...")
            
            # 对所有已处理的片段（历史+新增）执行Fast Clustering
            clustering_results = perform_fast_clustering_on_all_segments(
                state.all_segments,
                state.fastclustering_threshold,
            )
            
            # 更新说话人的均值声纹
            update_speaker_embeddings_from_clustering(state, clustering_results, state.all_segments)
            
            # 根据聚类结果更新所有片段的说话人ID（包括results和all_segments）
            # 创建映射：segment_id -> new_speaker_id
            segment_to_new_speaker = {seg_id: new_sid for seg_id, new_sid, _ in clustering_results}
            
            # 更新results中所有片段的说话人ID
            updated_count = 0
            for i, (seg_id, _) in enumerate(state.results):
                if seg_id in segment_to_new_speaker:
                    new_sid = segment_to_new_speaker[seg_id]
                    old_sid = state.results[i][1]
                    state.results[i] = (seg_id, new_sid)
                    if verbose and new_sid != old_sid:
                        print(f"    片段 {extract_offset_duration(seg_id)}: s{old_sid} -> s{new_sid}")
                        updated_count += 1
            
            # 更新all_segments中的说话人ID
            for i, (seg_id, emb, _) in enumerate(state.all_segments):
                if seg_id in segment_to_new_speaker:
                    new_sid = segment_to_new_speaker[seg_id]
                    state.all_segments[i] = (seg_id, emb, new_sid)
            
            # 更新最后一个片段的说话人ID
            if state.all_segments:
                _, _, last_speaker_id = state.all_segments[-1]
                state.prev_speaker_id = last_speaker_id
            
            if verbose:
                print(f"  已更新 {updated_count} 个片段的说话人ID")
            print(f"  继续处理...\n")
    
    # 处理剩余的片段（如果还有，且数量足够）
    if len(state.all_segments) % state.cluster_interval != 0:
        remaining_count = len(state.all_segments) % state.cluster_interval
        if remaining_count >= 5:  # 至少5个片段才执行聚类
            print(f"\n  处理剩余片段（{remaining_count}个），执行Fast Clustering...")
            
            clustering_results = perform_fast_clustering_on_all_segments(
                state.all_segments,
                state.fastclustering_threshold,
            )
            update_speaker_embeddings_from_clustering(state, clustering_results, state.all_segments)
            
            # 更新所有片段的说话人ID
            segment_to_new_speaker = {seg_id: new_sid for seg_id, new_sid, _ in clustering_results}
            
            # 更新results中所有片段的说话人ID
            for i, (seg_id, _) in enumerate(state.results):
                if seg_id in segment_to_new_speaker:
                    state.results[i] = (seg_id, segment_to_new_speaker[seg_id])
            
            # 更新all_segments中的说话人ID
            for i, (seg_id, emb, _) in enumerate(state.all_segments):
                if seg_id in segment_to_new_speaker:
                    state.all_segments[i] = (seg_id, emb, segment_to_new_speaker[seg_id])
            
            # 更新最后一个片段的说话人ID
            if state.all_segments:
                _, _, last_speaker_id = state.all_segments[-1]
                state.prev_speaker_id = last_speaker_id
    
    return state.results


def load_segments(input_path: Path, max_lines: int = None) -> List[Dict]:
    """
    从txt文件加载片段。
    
    格式: <segment_id> <base64_encoded_embedding>
    segment_id 格式: <audio_base_name>_<offset_ms>_<duration_ms>
    
    参数:
        input_path: 输入文件路径
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
                print(f"  达到最大读取行数 {max_lines}，停止读取。")
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
            # 格式: audio_base_name_offsetms_durationms
            try:
                # 按下划线分割并获取最后两部分
                id_parts = segment_id.rsplit('_', 2)
                if len(id_parts) >= 3:
                    offset_ms = int(id_parts[-2])
                    duration_ms = int(id_parts[-1])
                else:
                    # 备用方案：尝试提取数字
                    import re
                    numbers = re.findall(r'\d+', segment_id)
                    if len(numbers) >= 2:
                        offset_ms = int(numbers[-2])
                        duration_ms = int(numbers[-1])
                    else:
                        print(f"  警告: 无法从segment_id '{segment_id}'解析offset/duration，使用0")
                        offset_ms = 0
                        duration_ms = 0
            except (ValueError, IndexError) as e:
                print(f"  警告: 从segment_id '{segment_id}'解析offset/duration失败: {e}，使用0")
                offset_ms = 0
                duration_ms = 0
            
            segment_data = {
                "id": segment_id,
                "offsetMs": offset_ms,
                "durationMs": duration_ms,
                "embedding": emb_str,
            }
            segments.append(segment_data)
    
    if not segments:
        raise ValueError("输入文件中未找到片段！")
    
    print(f"  已加载 {len(segments)} 个片段")
    return segments


def extract_embeddings(segments: List[Dict]) -> Tuple[np.ndarray, List[int]]:
    """
    从片段中提取声纹向量（用于确定embedding维度）。
    
    返回:
        (embeddings_array, valid_indices) 元组
        - embeddings_array: 声纹数组 (num_valid x embedding_dim)
        - valid_indices: 具有有效声纹的片段索引列表
    """
    embeddings = []
    valid_indices = []
    
    for idx, seg in enumerate(segments):
        if "embedding" not in seg:
            continue
        
        try:
            emb = deserialize_embedding(seg["embedding"])
            embeddings.append(emb)
            valid_indices.append(idx)
        except Exception:
            continue
    
    if not embeddings:
        raise ValueError("未找到有效的声纹！")
    
    embeddings_array = np.array(embeddings, dtype=np.float32)
    return embeddings_array, valid_indices


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


def generate_output_filename(
    threshold: float,
    cluster_interval: int,
    fastclustering_threshold: float,
) -> str:
    """
    根据参数生成输出文件名。
    
    格式: cluster-realtime-fastclustering-th-{threshold}-interval-{interval}-fc-th-{fc_threshold}.txt
    """
    threshold_str = f"{threshold:.2f}".replace(".", "p")
    fc_threshold_str = f"{fastclustering_threshold:.2f}".replace(".", "p")
    filename = f"cluster-realtime-fastclustering-th-{threshold_str}-interval-{cluster_interval}-fc-th-{fc_threshold_str}.txt"
    return filename


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="基于Fast Clustering和实时处理的混合说话人聚类算法"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入txt文件路径（格式: segment_id base64_embedding）"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="输出目录路径。输出文件名会根据参数自动生成"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="实时匹配的相似度阈值（0.0-1.0）。值越大，匹配要求越严格。"
    )
    
    parser.add_argument(
        "--cluster-interval",
        type=int,
        default=DEFAULT_CLUSTER_INTERVAL,
        help="批量聚类的间隔（每隔N个片段执行一次Fast Clustering）。"
             "值越大，实时性越好，但准确性可能降低。"
    )
    
    parser.add_argument(
        "--fastclustering-threshold",
        type=float,
        default=DEFAULT_FASTCLUSTERING_THRESHOLD,
        help="Fast Clustering的相似度阈值（0.0-1.0）。"
             "值越小，产生的说话人数量越多。"
    )
    
    parser.add_argument(
        "--min-duration-seconds",
        type=float,
        default=DEFAULT_MIN_DURATION_SECONDS,
        help="创建新说话人所需的最小时长（单位: 秒）。"
             "当片段未匹配到任何说话人且时长小于该值时，沿用上一个说话人。"
    )
    
    parser.add_argument(
        "--max-embeddings-per-speaker",
        type=int,
        default=DEFAULT_MAX_EMBEDDINGS_PER_SPEAKER,
        help="每个说话人最多声纹数量（默认: 10）。"
             "超过该数量后，说话人将被标记为frozen状态，不再记录声纹列表以降低内存开销。"
    )
    
    parser.add_argument(
        "--merge-threshold",
        type=float,
        default=DEFAULT_MERGE_THRESHOLD,
        help="加入到声纹列表的相似度阈值（0.0-1.0，默认: 0.8）。"
             "只有当相似度 >= merge_threshold 且时长 >= min-duration-seconds 时，才会将声纹加入到说话人的声纹列表中。"
             "该值应该 >= threshold。"
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
    
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 根据参数自动生成输出文件名
    output_filename = generate_output_filename(
        args.threshold,
        args.cluster_interval,
        args.fastclustering_threshold,
    )
    output_path = output_dir / output_filename
    
    # 1) 加载片段
    segments = load_segments(input_path, args.max_lines)
    
    # 2) 提取声纹以确定维度
    embeddings_array, valid_indices = extract_embeddings(segments)
    embedding_dim = embeddings_array.shape[1]
    print(f"声纹维度: {embedding_dim}")
    
    # 3) 初始化状态
    state = HybridClusteringState(
        embedding_dim=embedding_dim,
        threshold=args.threshold,
        cluster_interval=args.cluster_interval,
        fastclustering_threshold=args.fastclustering_threshold,
        min_duration_seconds=args.min_duration_seconds,
        max_embeddings_per_speaker=args.max_embeddings_per_speaker,
        merge_threshold=args.merge_threshold,
    )
    
    # 4) 使用混合算法处理片段
    results = process_segments_hybrid(segments, state, args.verbose)
    
    # 5) 导出结果
    export_results(results, output_path)
    
    # 打印统计信息
    num_speakers = len(set(sid for _, sid in results))
    print(f"\n  === 处理统计 ===")
    print(f"  总片段数: {len(results)}")
    print(f"  总说话人数: {num_speakers}")
    
    if args.verbose:
        print(f"\n  === 说话人分布 ===")
        speaker_counts: Dict[int, int] = {}
        for _, speaker_id in results:
            speaker_counts[speaker_id] = speaker_counts.get(speaker_id, 0) + 1
        
        for speaker_id in sorted(speaker_counts.keys()):
            count = speaker_counts[speaker_id]
            percentage = 100.0 * count / len(results) if results else 0.0
            print(f"    说话人 {speaker_id}: {count} 个片段 ({percentage:.1f}%)")
    
    print(f"\n完成！结果已保存到 {output_path}")


if __name__ == "__main__":
    main()

