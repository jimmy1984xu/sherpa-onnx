#!/usr/bin/env python3

"""
基于声纹的实时说话人聚类。

本脚本读取声纹文件并执行实时说话人聚类，根据声纹相似度为每个片段分配说话人ID。

输入格式: <segment_id> <base64_encoded_embedding>
segment_id 格式: <audio_base_name>_<offset_ms>_<duration_ms>

输出格式: 每行包含:
  <segment_id> <speaker_id>

使用示例:

python3 ./python-api-examples/k2-speaker-cluster-realtime.py \
  --input embeddings.txt \
  --output-dir ./output \
  --threshold 0.7 \
  --merge-threshold 0.8 \
  --min-duration-seconds 3.0 \
  --max-embeddings-per-speaker 10

测试时限制读取行数:

python3 ./python-api-examples/k2-speaker-cluster-realtime.py \
  --input embeddings.txt \
  --output-dir ./output \
  --threshold 0.7 \
  --merge-threshold 0.8 \
  --min-duration-seconds 3.0 \
  --max-embeddings-per-speaker 10 \
  --max-lines 100

处理逻辑:

1. **首个片段**: 总是分配说话人ID=1，并保存其声纹向量和时长作为该说话人的初始声纹。

2. **后续片段**:
   - 根据片段时长判断是长句还是短句（通过min-duration-seconds阈值区分）
   - 计算与所有已存在说话人的声纹相似度，找出最佳匹配
   - 使用双阈值机制进行判断:
     * **匹配阈值 (threshold)**: 用于判断是否为同一个说话人（相似度 >= threshold）
     * **合并阈值 (merge_threshold)**: 用于判断是否可以合并到声纹列表（相似度 >= merge_threshold，要求更高）
   
   - **如果找到匹配的说话人** (相似度 >= threshold):
     * **长句且相似度 >= merge_threshold**: 将当前声纹向量加入到该说话人的声纹列表
       （不超过max-embeddings-per-speaker限制），并更新manager中的声纹（内部会自动求均值）
     * **长句但相似度 < merge_threshold**: 
       - 如果说话人只有1个声纹，且当前片段时长 > 旧声纹片段时长，则替换旧声纹
       - 否则只分配说话人ID，不更新声纹列表
     * **短句**: 直接返回匹配的说话人ID，不更新声纹列表
   
   - **如果未找到匹配的说话人** (相似度 < threshold):
     * **长句**: 创建新的说话人ID，保存当前声纹向量和时长，并更新到manager
     * **短句**: 沿用上一个片段的说话人ID，避免频繁改变说话人ID

核心特点:

- **双阈值机制**: 
  - `threshold`: 匹配阈值，判断是否为同一个说话人
  - `merge_threshold`: 合并阈值（>= threshold），判断是否可以合并到声纹列表
  - 这种设计允许在匹配到说话人但相似度不够高时，只分配ID而不更新声纹，提高稳定性

- **声纹管理**: 
  - 为每个说话人维护一个声纹向量列表和对应的时长列表
  - 最多保存max-embeddings-per-speaker条声纹，超过后不再更新
  - 当说话人只有1个声纹时，如果新片段时长更长，可以替换旧声纹以提升质量

- **长句/短句区分策略**: 
  - 长句（>= min-duration-seconds）用于更新声纹和创建新说话人
  - 短句采用保守策略，减少说话人ID的频繁变化

- **自动声纹更新**: 
  - 使用SpeakerEmbeddingManager的add(name, List<embedding>)接口
  - 传入该说话人的所有声纹向量，manager内部会自动计算均值并更新

- **日志输出**: 
  - 每个片段输出一条关键日志，包含:
    * 与所有说话人的相似度
    * 与上一个片段的相似度 (last)
    * 最终分配的说话人ID
    * 分配原因（首个片段/匹配/创建/沿用）
    * 是否合并到声纹列表

- **容错处理**: 异常情况下返回最后一个有效的说话人ID，确保系统稳定性
"""

import argparse
import base64
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import sherpa_onnx


# Constants
FIRST_SPEAKER_ID = 1
INVALID_SPEAKER_ID = -1
DEFAULT_THRESHOLD = 0.8
DEFAULT_MIN_DURATION_SECONDS = 3.0
DEFAULT_MAX_EMBEDDINGS = 10


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


class SpeakerClusteringState:
    """
    管理实时说话人聚类算法的状态。
    
    属性:
        manager: SpeakerEmbeddingManager实例，用于声纹操作
        speaker_embeddings_map: 字典，映射说话人ID到声纹列表
        next_speaker_id: 下一个可用的说话人ID（从2开始，因为1已被第一个片段使用）
        prev_speaker_id: 上一个片段的说话人ID
        threshold: 同一个说话人相似度阈值（用于判断是否为同一个说话人）
        merge_threshold: 同一个说话人合并声纹的相似度阈值（用于判断是否可以合并到声纹列表）
        max_embeddings: 每个说话人的最大声纹数量
        short_segments_no_match: 未匹配到任何说话人的短句数量
    """
    def __init__(self, embedding_dim: int, threshold: float, merge_threshold: float, max_embeddings: int):
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim必须为正数，得到 {embedding_dim}")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold必须在0.0和1.0之间，得到 {threshold}")
        if not 0.0 <= merge_threshold <= 1.0:
            raise ValueError(f"merge_threshold必须在0.0和1.0之间，得到 {merge_threshold}")
        if merge_threshold < threshold:
            raise ValueError(f"merge_threshold ({merge_threshold}) 应该 >= threshold ({threshold})")
        if max_embeddings <= 0:
            raise ValueError(f"max_embeddings必须为正数，得到 {max_embeddings}")
            
        self.manager = sherpa_onnx.SpeakerEmbeddingManager(dim=embedding_dim)
        self.speaker_embeddings_map: Dict[int, List[np.ndarray]] = {}
        self.speaker_embeddings_durations_map: Dict[int, List[float]] = {}  # 保存每个声纹对应的片段时长
        self.next_speaker_id = 2  # 下一个说话人ID从2开始（1已被第一个片段使用）
        self.prev_speaker_id = INVALID_SPEAKER_ID
        self.prev_embedding: np.ndarray = None  # 保存上一个片段的声纹向量
        self.threshold = threshold
        self.merge_threshold = merge_threshold
        self.max_embeddings = max_embeddings
        self.short_segments_no_match = 0


def identify_speaker(
    embedding: np.ndarray,
    is_long: bool,
    is_first: bool,
    state: SpeakerClusteringState,
    verbose: bool = False,
    segment_id: str = "",
    duration_ms: float = 0.0,
) -> int:
    """
    识别片段的说话人ID（实时聚类的核心函数）。
    
    规则:
    1. 首个片段: 总是返回说话人ID=1，并保存声纹
    2. 后续片段:
       - 与所有已存在的说话人进行声纹比较
       - 如果找到匹配（相似度 >= threshold）:
         * 长句且相似度 >= merge_threshold: 更新该说话人的声纹列表
         * 其他情况: 返回匹配的说话人ID，不更新声纹
       - 如果未找到匹配:
         * 长句: 创建新的说话人ID
         * 短句: 沿用上一个片段的说话人ID
    
    参数:
        embedding: 归一化后的声纹向量
        is_long: 是否为长句（True=长句，False=短句）
        is_first: 是否为第一个片段
        state: SpeakerClusteringState对象，管理算法状态
        verbose: 是否打印详细信息
    
    返回:
        speaker_id: 说话人ID（>=1）
    """
    if is_first:
        # 第一句处理：总是分配说话人ID 1
        # 创建说话人1的声纹记录
        state.speaker_embeddings_map[FIRST_SPEAKER_ID] = [embedding]
        state.speaker_embeddings_durations_map[FIRST_SPEAKER_ID] = [duration_ms]  # 保存第一个片段的时长
        # 添加到manager
        if not state.manager.add(str(FIRST_SPEAKER_ID), embedding):
            raise RuntimeError("添加第一个说话人到manager失败")
        state.prev_speaker_id = FIRST_SPEAKER_ID
        state.prev_embedding = embedding  # 保存第一个片段的声纹
        
        # 打印关键日志（第一句，在处理完成后）
        segment_display = extract_offset_duration(segment_id)
        segment_display_fixed = segment_display.ljust(15)  # 固定15个字符宽度，左对齐
        print(f"  {segment_display_fixed} 分配:s{FIRST_SPEAKER_ID} 原因:首个片段 [last=N/A 无说话人] 声纹操作:无")
        
        return FIRST_SPEAKER_ID
    
    # 后续句子处理：与所有说话人进行比较
    all_speakers = state.manager.all_speakers
    speaker_similarities: Dict[int, float] = {}  # 存储每个说话人的相似度
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
    
    # 判断是否达到阈值（用于判断是否为同一个说话人）
    is_match = (best_similarity >= state.threshold and best_speaker_id >= FIRST_SPEAKER_ID)
    
    # 判断是否可以合并声纹（需要更高的阈值）
    can_merge = (is_long and best_similarity >= state.merge_threshold and best_speaker_id >= FIRST_SPEAKER_ID)
    
    # 计算与上一个片段的相似度
    last_similarity = -1.0
    if state.prev_embedding is not None:
        # 计算余弦相似度（两个向量都已归一化）
        last_similarity = float(np.dot(embedding, state.prev_embedding))
    
    # 先确定最终分配的说话人ID和分配原因
    if is_match:
        final_speaker_id = best_speaker_id
        reason = "匹配"
    elif is_long:
        final_speaker_id = state.next_speaker_id
        reason = "创建"
    else:
        final_speaker_id = state.prev_speaker_id if state.prev_speaker_id >= FIRST_SPEAKER_ID else FIRST_SPEAKER_ID
        reason = "沿用"
    
    # 初始化声纹操作类型（将在处理过程中更新）
    embedding_action = "无"
    
    # 根据匹配情况处理
    if is_match:
        # 相似度达到阈值，分配到该说话人
        if can_merge:
            # 长句且相似度达到合并阈值：更新该说话人的声纹列表
            if best_speaker_id in state.speaker_embeddings_map:
                emb_list = state.speaker_embeddings_map[best_speaker_id]
                dur_list = state.speaker_embeddings_durations_map.get(best_speaker_id, [])
                if len(emb_list) < state.max_embeddings:
                    emb_list.append(embedding)
                    dur_list.append(duration_ms)
                    state.speaker_embeddings_durations_map[best_speaker_id] = dur_list
                    # 更新manager中的声纹（使用list，manager会自动求均值）
                    state.manager.remove(str(best_speaker_id))
                    state.manager.add(str(best_speaker_id), emb_list)
                    embedding_action = "更新列表"
                    if verbose:
                        print(f"    -> 更新说话人 {best_speaker_id} 的声纹 "
                              f"(现在 {len(emb_list)}/{state.max_embeddings} 个声纹)")
                else:
                    if verbose:
                        print(f"    -> 说话人 {best_speaker_id} 的声纹不更新 "
                              f"(已达到最大 {state.max_embeddings} 个声纹)")
            else:
                # 理论上不应该发生，但为了安全起见
                state.speaker_embeddings_map[best_speaker_id] = [embedding]
                state.speaker_embeddings_durations_map[best_speaker_id] = [duration_ms]
                state.manager.add(str(best_speaker_id), embedding)
        else:
            # 未达到合并阈值，但达到匹配阈值
            # 如果说话人的声纹只有1个，且当前片段时长大于旧声纹片段时长，则替换
            if best_speaker_id in state.speaker_embeddings_map:
                emb_list = state.speaker_embeddings_map[best_speaker_id]
                dur_list = state.speaker_embeddings_durations_map.get(best_speaker_id, [])
                
                if len(emb_list) == 1 and len(dur_list) == 1:
                    old_duration = dur_list[0]
                    if duration_ms > old_duration:
                        # 替换旧的声纹
                        emb_list[0] = embedding
                        dur_list[0] = duration_ms
                        state.speaker_embeddings_durations_map[best_speaker_id] = dur_list
                        # 更新manager中的声纹
                        state.manager.remove(str(best_speaker_id))
                        state.manager.add(str(best_speaker_id), embedding)
                        embedding_action = "替换声纹"
                        if verbose:
                            print(f"    -> 替换说话人 {best_speaker_id} 的声纹 "
                                  f"(旧时长: {old_duration:.0f}ms, 新时长: {duration_ms:.0f}ms)")
        
        state.prev_speaker_id = best_speaker_id
        state.prev_embedding = embedding  # 保存当前片段的声纹
        
        # 打印关键日志（在处理完成后）
        sim_str = ", ".join([f"s{sid}={sim:.3f}" for sid, sim in sorted(speaker_similarities.items())])
        if not speaker_similarities:
            sim_str = "无说话人"
        segment_display = extract_offset_duration(segment_id)
        segment_display_fixed = segment_display.ljust(15)  # 固定15个字符宽度，左对齐
        last_str = f"last={last_similarity:.3f}" if last_similarity >= 0 else "last=N/A"
        print(f"  {segment_display_fixed} 分配:s{final_speaker_id} 原因:{reason} [{last_str} {sim_str}] 声纹操作:{embedding_action}")
        
        return best_speaker_id
    else:
        # 相似度未达到阈值
        if is_long:
            # 长句：创建新的说话人ID
            speaker_id = state.next_speaker_id
            # 添加到外部map和manager
            state.speaker_embeddings_map[speaker_id] = [embedding]
            state.speaker_embeddings_durations_map[speaker_id] = [duration_ms]
            state.manager.add(str(speaker_id), embedding)
            state.prev_speaker_id = speaker_id
            state.prev_embedding = embedding  # 保存当前片段的声纹
            state.next_speaker_id += 1
            
            # 打印关键日志（在处理完成后）
            sim_str = ", ".join([f"s{sid}={sim:.3f}" for sid, sim in sorted(speaker_similarities.items())])
            if not speaker_similarities:
                sim_str = "无说话人"
            segment_display = extract_offset_duration(segment_id)
            segment_display_fixed = segment_display.ljust(15)  # 固定15个字符宽度，左对齐
            last_str = f"last={last_similarity:.3f}" if last_similarity >= 0 else "last=N/A"
            print(f"  {segment_display_fixed} 分配:s{final_speaker_id} 原因:{reason} [{last_str} {sim_str}] 声纹操作:{embedding_action}")
            
            return speaker_id
        else:
            # 短句：未找到匹配，沿用上一个片段的说话人ID
            state.short_segments_no_match += 1
            
            if state.prev_speaker_id >= FIRST_SPEAKER_ID:
                # 上一句是合法说话人（>=1）
                state.prev_embedding = embedding  # 保存当前片段的声纹
                
                # 打印关键日志（在处理完成后）
                sim_str = ", ".join([f"s{sid}={sim:.3f}" for sid, sim in sorted(speaker_similarities.items())])
                if not speaker_similarities:
                    sim_str = "无说话人"
                segment_display = extract_offset_duration(segment_id)
                segment_display_fixed = segment_display.ljust(15)  # 固定15个字符宽度，左对齐
                last_str = f"last={last_similarity:.3f}" if last_similarity >= 0 else "last=N/A"
                print(f"  {segment_display_fixed} 分配:s{final_speaker_id} 原因:{reason} [{last_str} {sim_str}] 声纹操作:{embedding_action}")
                
                return state.prev_speaker_id
            else:
                # 上一句是-1或0（不应该发生，因为第一句总是1），但为了安全起见
                state.prev_speaker_id = FIRST_SPEAKER_ID
                state.prev_embedding = embedding  # 保存当前片段的声纹
                
                # 打印关键日志（在处理完成后）
                sim_str = ", ".join([f"s{sid}={sim:.3f}" for sid, sim in sorted(speaker_similarities.items())])
                if not speaker_similarities:
                    sim_str = "无说话人"
                segment_display = extract_offset_duration(segment_id)
                segment_display_fixed = segment_display.ljust(15)  # 固定15个字符宽度，左对齐
                last_str = f"last={last_similarity:.3f}" if last_similarity >= 0 else "last=N/A"
                print(f"  {segment_display_fixed} 分配:s{final_speaker_id} 原因:{reason} [{last_str} {sim_str}] 声纹操作:{embedding_action}")
                
                return FIRST_SPEAKER_ID


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


def process_segments_realtime(
    segments: List[Dict],
    embeddings_array: np.ndarray,
    valid_indices: List[int],
    min_duration_seconds: float,
    threshold: float,
    merge_threshold: float,
    max_embeddings: int,
    verbose: bool = False,
) -> List[Tuple[str, int]]:
    """
    实时处理片段并分配说话人ID。
    
    返回:
        (segment_id, speaker_id) 元组列表
    """
    print(f"使用实时聚类处理片段...")
    print(f"  参数: threshold={threshold}, merge_threshold={merge_threshold}, "
          f"min_duration={min_duration_seconds}s, max_embeddings={max_embeddings}")
    
    # 按时间（offsetMs）排序片段
    valid_indices_sorted = sorted(valid_indices, key=lambda idx: segments[idx].get("offsetMs", 0))
    
    # 创建从排序索引到声纹数组索引的映射
    idx_to_emb_idx = {idx: valid_indices.index(idx) for idx in valid_indices_sorted}
    
    # 归一化声纹并准备片段元数据
    min_duration_ms = min_duration_seconds * 1000
    all_segments = []  # (segment_idx, embedding, duration_ms, is_long)
    
    for sorted_idx in valid_indices_sorted:
        emb_idx = idx_to_emb_idx[sorted_idx]
        emb = normalize_embedding(embeddings_array[emb_idx])
        duration_ms = segments[sorted_idx].get("durationMs", 0)
        is_long = duration_ms >= min_duration_ms
        all_segments.append((sorted_idx, emb, duration_ms, is_long))
    
    # 统计
    long_count = sum(1 for _, _, _, is_long in all_segments if is_long)
    short_count = sum(1 for _, _, _, is_long in all_segments if not is_long)
    
    print(f"  长句 (>= {min_duration_seconds}s): {long_count}")
    print(f"  短句 (< {min_duration_seconds}s): {short_count}")
    
    if verbose:
        print(f"\n  === 按时间顺序处理片段 ===\n")
    
    # 获取声纹维度
    if len(valid_indices) == 0:
        return []
    first_emb_idx = valid_indices.index(valid_indices_sorted[0])
    embedding_dim = embeddings_array[first_emb_idx].shape[0]
    
    # 创建聚类状态对象
    state = SpeakerClusteringState(embedding_dim, threshold, merge_threshold, max_embeddings)
    
    # 按时间顺序处理片段
    results = []
    seg_counter = 0
    
    for valid_idx, emb, duration_ms, is_long in all_segments:
        seg_counter += 1
        offset_ms = segments[valid_idx].get("offsetMs", 0)
        segment_id = segments[valid_idx].get("id", "")
        seg_type = "长句" if is_long else "短句"
        
        if verbose:
            print(f"  [{seg_type} {seg_counter}/{len(all_segments)}] "
                  f"offset={offset_ms:.0f}ms, duration={duration_ms:.0f}ms, "
                  f"id='{segment_id}'")
        
        # 判断是否为第一个片段
        is_first_segment = (seg_counter == 1)
        
        # 调用核心函数识别说话人
        try:
            speaker_id = identify_speaker(
                embedding=emb,
                is_long=is_long,
                is_first=is_first_segment,
                state=state,
                verbose=verbose,
                segment_id=segment_id,  # 传递片段ID用于日志
                duration_ms=duration_ms,  # 传递片段时长用于日志
            )
            results.append((segment_id, speaker_id))
            
            if verbose:
                print(f"    -> 结果: speaker_id={speaker_id}\n")
        except Exception as e:
            # 错误处理：使用最后一个有效的说话人ID
            print(f"  处理片段 {segment_id} 时出错: {e}")
            if state.prev_speaker_id >= FIRST_SPEAKER_ID:
                speaker_id = state.prev_speaker_id
                print(f"    -> 使用上一个说话人ID: {speaker_id}")
            else:
                speaker_id = FIRST_SPEAKER_ID
                print(f"    -> 回退到说话人ID: {speaker_id}")
            results.append((segment_id, speaker_id))
    
    # 打印统计信息
    print(f"\n  === 处理统计 ===")
    print(f"  总处理片段数: {len(results)}")
    print(f"  总说话人数: {len(state.speaker_embeddings_map)}")
    if state.short_segments_no_match > 0:
        print(f"  未匹配的短句数: {state.short_segments_no_match}")
    
    if verbose:
        print(f"\n  === 说话人统计 ===")
        for speaker_id in sorted(state.speaker_embeddings_map.keys()):
            emb_count = len(state.speaker_embeddings_map[speaker_id])
            seg_count = sum(1 for _, sid in results if sid == speaker_id)
            print(f"    说话人 {speaker_id}: {seg_count} 个片段, "
                  f"{emb_count}/{max_embeddings} 个声纹")
    
    return results


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
        description="基于声纹的实时说话人聚类"
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
        help="输出目录路径。输出文件名会根据参数自动生成，格式: cluster-realtime-th-{threshold}-min-{min_duration}-max-embeddings-{max_embeddings}.txt"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="同一个说话人相似度阈值（0.0-1.0）。大于这个值，认为是同一个说话人，可以分配相同说话人ID。"
    )
    
    parser.add_argument(
        "--merge-threshold",
        type=float,
        default=None,
        help="同一个说话人合并声纹的相似度阈值（0.0-1.0）。这个值要求更高，大于这个值才能合并到说话人的声纹列表中。"
             "如果未指定，默认使用threshold的值。"
    )
    
    parser.add_argument(
        "--min-duration-seconds",
        type=float,
        default=DEFAULT_MIN_DURATION_SECONDS,
        help="被视为长句的最小片段时长（秒）。长句可以更新声纹和创建新说话人。"
    )
    
    parser.add_argument(
        "--max-embeddings-per-speaker",
        type=int,
        default=DEFAULT_MAX_EMBEDDINGS,
        help="每个说话人用于更新均值声纹的最大声纹数量。达到此限制后，不再添加更多声纹。"
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


def generate_output_filename(
    threshold: float,
    merge_threshold: float,
    min_duration_seconds: float,
    max_embeddings: int,
) -> str:
    """
    根据参数生成输出文件名。
    
    格式: cluster-realtime-th-{threshold}-merge-{merge_threshold}-min-{min_duration}-max-embeddings-{max_embeddings}.txt
    
    参数:
        threshold: 同一个说话人相似度阈值
        merge_threshold: 合并声纹的相似度阈值
        min_duration_seconds: 最小时长（秒）
        max_embeddings: 最大声纹数量
    
    返回:
        输出文件名
    """
    # 将浮点数中的点替换为p，避免文件名问题
    threshold_str = f"{threshold:.2f}".replace(".", "p")
    merge_threshold_str = f"{merge_threshold:.2f}".replace(".", "p")
    min_duration_str = f"{min_duration_seconds:.1f}".replace(".", "p")
    
    filename = f"cluster-realtime-th-{threshold_str}-merge-{merge_threshold_str}-min-{min_duration_str}-max-embeddings-{max_embeddings}.txt"
    return filename


def main():
    """主函数。"""
    args = parse_args()
    
    # 验证参数
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError(f"threshold必须在0.0和1.0之间，得到 {args.threshold}")
    
    # 设置merge_threshold，如果未指定则使用threshold
    merge_threshold = args.merge_threshold if args.merge_threshold is not None else args.threshold
    if not 0.0 <= merge_threshold <= 1.0:
        raise ValueError(f"merge_threshold必须在0.0和1.0之间，得到 {merge_threshold}")
    if merge_threshold < args.threshold:
        raise ValueError(f"merge_threshold ({merge_threshold}) 应该 >= threshold ({args.threshold})")
    
    if args.min_duration_seconds <= 0:
        raise ValueError(f"min-duration-seconds必须为正数，得到 {args.min_duration_seconds}")
    if args.max_embeddings_per_speaker <= 0:
        raise ValueError(f"max-embeddings-per-speaker必须为正数，"
                        f"得到 {args.max_embeddings_per_speaker}")
    
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 根据参数自动生成输出文件名
    output_filename = generate_output_filename(
        args.threshold,
        merge_threshold,
        args.min_duration_seconds,
        args.max_embeddings_per_speaker,
    )
    output_path = output_dir / output_filename
    
    # 1) 加载片段
    segments = load_segments(input_path, args.max_lines)
    
    # 2) 提取声纹
    embeddings_array, valid_indices = extract_embeddings(segments)
    
    # 3) 使用实时聚类处理片段
    results = process_segments_realtime(
        segments,
        embeddings_array,
        valid_indices,
        args.min_duration_seconds,
        args.threshold,
        merge_threshold,
        args.max_embeddings_per_speaker,
        args.verbose,
    )
    
    # 4) 导出结果
    export_results(results, output_path)
    
    print(f"\n完成！结果已保存到 {output_path}")


if __name__ == "__main__":
    main()

