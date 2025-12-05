#!/usr/bin/env python3

"""
合并ASR识别结果和说话人聚类结果。

本脚本读取ASR结果文件和说话人聚类结果文件，根据片段ID进行合并，
输出包含片段ID、说话人ID和ASR文本的结果文件。

输入格式:
- ASR结果文件: <segment_id> <asr_text>
- 说话人聚类结果文件: <segment_id> <speaker_id>

输出格式: 每行包含:
  <segment_id> <speaker_id> <asr_text>

使用示例:

python3 ./python-api-examples/k2-merge-asr-speaker.py \
  --asr-result ./output/paraformer_asr.txt \
  --speaker-result ./output/cluster-realtime-th-0p80-min-3p0-max-embeddings-10.txt \
  --output-dir ./output

输出文件会自动命名为: merged-asr-speaker-{asr_basename}-{speaker_basename}.txt

注意事项:
- 两个输入文件必须包含相同的segment_id集合（或子集）
- 如果某个segment_id在一个文件中存在但在另一个文件中不存在，该片段会被跳过
- 输出文件按segment_id排序
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import re


def load_asr_results(asr_path: Path) -> Dict[str, str]:
    """
    加载ASR结果文件。
    
    格式: <segment_id> <asr_text>
    
    返回:
        字典，映射 segment_id 到 asr_text
    """
    if not asr_path.is_file():
        raise FileNotFoundError(f"ASR结果文件未找到: {asr_path}")
    
    print(f"从 {asr_path} 加载ASR结果...")
    asr_dict = {}
    
    with open(asr_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            # 分割第一列（segment_id）和剩余部分（asr_text）
            # 使用split(None, 1)确保文本中的空格被保留
            parts = line.split(None, 1)
            if len(parts) < 2:
                print(f"  警告: 第 {line_num} 行格式无效（缺少ASR文本），跳过: {line[:50]}...")
                continue
            
            segment_id, asr_text = parts
            asr_dict[segment_id] = asr_text
    
    print(f"  已加载 {len(asr_dict)} 条ASR结果")
    return asr_dict


def load_speaker_results(speaker_path: Path) -> Dict[str, int]:
    """
    加载说话人聚类结果文件。
    
    格式: <segment_id> <speaker_id>
    
    返回:
        字典，映射 segment_id 到 speaker_id
    """
    if not speaker_path.is_file():
        raise FileNotFoundError(f"说话人聚类结果文件未找到: {speaker_path}")
    
    print(f"从 {speaker_path} 加载说话人聚类结果...")
    speaker_dict = {}
    
    with open(speaker_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) != 2:
                print(f"  警告: 第 {line_num} 行格式无效，跳过: {line[:50]}...")
                continue
            
            segment_id, speaker_id_str = parts
            try:
                speaker_id = int(speaker_id_str)
                speaker_dict[segment_id] = speaker_id
            except ValueError:
                print(f"  警告: 第 {line_num} 行说话人ID无效，跳过: {line[:50]}...")
                continue
    
    print(f"  已加载 {len(speaker_dict)} 条说话人聚类结果")
    return speaker_dict


def merge_results(
    asr_dict: Dict[str, str],
    speaker_dict: Dict[str, int],
) -> List[Tuple[str, int, str]]:
    """
    合并ASR结果和说话人聚类结果。
    
    返回:
        (segment_id, speaker_id, asr_text) 元组列表，按offset_ms数值排序（从小到大）
    """
    print(f"合并ASR结果和说话人聚类结果...")
    
    # 找到两个文件中都存在的segment_id
    common_segment_ids = set(asr_dict.keys()) & set(speaker_dict.keys())
    
    if not common_segment_ids:
        raise ValueError("未找到共同的片段ID！请检查输入文件是否正确。")
    
    # 统计信息
    asr_only = set(asr_dict.keys()) - set(speaker_dict.keys())
    speaker_only = set(speaker_dict.keys()) - set(asr_dict.keys())
    
    if asr_only:
        print(f"  警告: {len(asr_only)} 个片段只在ASR结果中存在，将被跳过")
        if len(asr_only) <= 10:
            for seg_id in sorted(asr_only):
                print(f"    - {seg_id}")
        else:
            for seg_id in sorted(list(asr_only)[:10]):
                print(f"    - {seg_id}")
            print(f"    ... 还有 {len(asr_only) - 10} 个")
    
    if speaker_only:
        print(f"  警告: {len(speaker_only)} 个片段只在说话人聚类结果中存在，将被跳过")
        if len(speaker_only) <= 10:
            for seg_id in sorted(speaker_only):
                print(f"    - {seg_id}")
        else:
            for seg_id in sorted(list(speaker_only)[:10]):
                print(f"    - {seg_id}")
            print(f"    ... 还有 {len(speaker_only) - 10} 个")
    
    # 合并结果，按offset_ms数值排序（从小到大）
    merged_results = []
    
    # 创建 (offset_ms, segment_id) 元组列表用于排序
    segment_id_with_offset = []
    for segment_id in common_segment_ids:
        offset_ms, _ = parse_segment_id(segment_id)
        segment_id_with_offset.append((offset_ms, segment_id))
    
    # 按offset_ms排序
    segment_id_with_offset.sort(key=lambda x: x[0])
    
    # 生成最终结果
    for offset_ms, segment_id in segment_id_with_offset:
        speaker_id = speaker_dict[segment_id]
        asr_text = asr_dict[segment_id]
        merged_results.append((segment_id, speaker_id, asr_text))
    
    print(f"  成功合并 {len(merged_results)} 条结果（按offset从小到大排序）")
    return merged_results


def parse_segment_id(segment_id: str) -> Tuple[int, int]:
    """
    从segment_id解析offset_ms和duration_ms。
    
    格式: <audio_base_name>_<offset_ms>_<duration_ms>
    
    返回:
        (offset_ms, duration_ms) 元组
    """
    try:
        # 按下划线分割并获取最后两部分
        id_parts = segment_id.rsplit('_', 2)
        if len(id_parts) >= 3:
            offset_ms = int(id_parts[-2])
            duration_ms = int(id_parts[-1])
            return offset_ms, duration_ms
        else:
            # 备用方案：尝试提取数字
            numbers = re.findall(r'\d+', segment_id)
            if len(numbers) >= 2:
                offset_ms = int(numbers[-2])
                duration_ms = int(numbers[-1])
                return offset_ms, duration_ms
    except (ValueError, IndexError):
        pass
    
    return 0, 0


def export_results(
    results: List[Tuple[str, int, str]],
    output_path: Path,
) -> None:
    """
    导出合并结果到文本文件。
    
    格式: <segment_id> <speaker_id> <asr_text>
    """
    print(f"导出结果到 {output_path}...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for segment_id, speaker_id, asr_text in results:
            f.write(f"{segment_id} {speaker_id} {asr_text}\n")
    
    print(f"  成功导出 {len(results)} 条结果")
    
    # 打印统计信息
    print(f"\n  === 合并统计 ===")
    print(f"  总片段数: {len(results)}")
    
    # 统计每个说话人的片段数
    speaker_counts: Dict[int, int] = {}
    for _, speaker_id, _ in results:
        speaker_counts[speaker_id] = speaker_counts.get(speaker_id, 0) + 1
    
    print(f"  说话人数量: {len(speaker_counts)}")
    print(f"\n  说话人分布:")
    for speaker_id in sorted(speaker_counts.keys()):
        count = speaker_counts[speaker_id]
        percentage = 100.0 * count / len(results) if results else 0.0
        print(f"    说话人 {speaker_id}: {count} 个片段 ({percentage:.1f}%)")


def generate_output_filename(
    asr_path: Path,
    speaker_path: Path,
) -> str:
    """
    根据输入文件名生成输出文件名。
    
    格式: merged-asr-speaker-{asr_basename}-{speaker_basename}.txt
    
    参数:
        asr_path: ASR结果文件路径
        speaker_path: 说话人聚类结果文件路径
    
    返回:
        输出文件名
    """
    asr_basename = asr_path.stem
    speaker_basename = speaker_path.stem
    
    # 清理文件名，移除可能的问题字符
    asr_basename = asr_basename.replace(" ", "_")
    speaker_basename = speaker_basename.replace(" ", "_")
    
    filename = f"merged-asr-speaker-{asr_basename}-{speaker_basename}.txt"
    return filename


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="合并ASR识别结果和说话人聚类结果"
    )
    
    parser.add_argument(
        "--asr-result",
        type=str,
        required=True,
        help="ASR结果文件路径（格式: segment_id asr_text）"
    )
    
    parser.add_argument(
        "--speaker-result",
        type=str,
        required=True,
        help="说话人聚类结果文件路径（格式: segment_id speaker_id）"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="输出目录路径。输出文件名会根据输入文件名自动生成，"
             "格式: merged-asr-speaker-{asr_basename}-{speaker_basename}.txt"
    )
    
    return parser.parse_args()


def main():
    """主函数。"""
    args = parse_args()
    
    asr_path = Path(args.asr_result)
    speaker_path = Path(args.speaker_result)
    output_dir = Path(args.output_dir)
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1) 加载ASR结果
    asr_dict = load_asr_results(asr_path)
    
    # 2) 加载说话人聚类结果
    speaker_dict = load_speaker_results(speaker_path)
    
    # 3) 合并结果
    merged_results = merge_results(asr_dict, speaker_dict)
    
    # 4) 生成输出文件名并导出
    output_filename = generate_output_filename(asr_path, speaker_path)
    output_path = output_dir / output_filename
    
    export_results(merged_results, output_path)
    
    print(f"\n完成！结果已保存到 {output_path}")


if __name__ == "__main__":
    main()

