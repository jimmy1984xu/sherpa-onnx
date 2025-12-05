#!/usr/bin/env python3

"""
计算ASR识别结果的WER（词错误率）。

该脚本读取参考文本文件和ASR识别结果文件，计算WER（Word Error Rate）和其他相关指标。

输入格式支持（与 k2-asr-short.py 输出格式兼容）：
- 带置信度：<segment_id> <confidence> <asr_text>
- 不带置信度：<segment_id> <asr_text>

参考文本格式：
- <segment_id> <reference_text>

输出包含：
- WER（词错误率）
- SER（句子错误率）
- 替换/插入/删除错误统计
- 详细的对齐信息（可选）

使用示例：

python3 ./python-api-examples/k2-wer.py \
  --reference /path/to/reference.txt \
  --hypothesis /path/to/asr_result.txt \
  --output /path/to/wer_result.txt \
  --detailed  # 可选：输出详细对齐信息
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from collections import defaultdict

try:
    import jiwer
except ImportError:
    print("错误: 需要安装 jiwer 库用于WER计算")
    print("请运行: pip install jiwer")
    sys.exit(1)


def parse_text_line(line: str) -> Tuple[str, str]:
    """
    解析文本文件的一行。
    
    支持格式：
    1. segment_id confidence text
    2. segment_id text
    
    返回:
        (segment_id, text)
    """
    line = line.strip()
    if not line:
        return None, None
    
    parts = line.split(None, 2)  # 最多分割成3部分
    
    if len(parts) == 2:
        # 可能是: segment_id confidence 或 segment_id text
        segment_id, part2 = parts
        # 尝试将第二部分解析为浮点数
        try:
            confidence = float(part2)
            # 如果成功解析为浮点数，且值在合理范围内，认为是置信度（没有文本）
            if -1.0 <= confidence <= 1.0:
                return segment_id, ""
            else:
                # 超出范围，认为是文本
                return segment_id, part2
        except ValueError:
            # 无法解析为浮点数，认为是文本
            return segment_id, part2
    elif len(parts) == 3:
        # 格式: segment_id confidence text 或 segment_id text_with_spaces
        segment_id, part2, text = parts
        try:
            confidence = float(part2)
            # 如果成功解析为浮点数，且值在合理范围内，认为是置信度
            if -1.0 <= confidence <= 1.0:
                return segment_id, text
            else:
                # 超出范围，可能是文本的一部分
                return segment_id, f"{part2} {text}"
        except ValueError:
            # 无法解析为浮点数，认为是文本的一部分
            return segment_id, f"{part2} {text}"
    else:
        # 格式错误
        return None, None


def load_text_file(file_path: Path) -> Dict[str, str]:
    """
    加载文本文件。
    
    返回:
        字典，映射 segment_id 到 text
    """
    if not file_path.is_file():
        raise FileNotFoundError(f"文件未找到: {file_path}")
    
    print(f"正在加载文件: {file_path}")
    results = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            segment_id, text = parse_text_line(line)
            
            if segment_id is None:
                if line.strip():  # 非空行但格式错误
                    print(f"  警告: 第 {line_num} 行格式无效，跳过: {line[:50]}...")
                continue
            
            results[segment_id] = text if text else ""
    
    print(f"  已加载 {len(results)} 条记录")
    return results


def calculate_wer(reference: str, hypothesis: str) -> Tuple[float, int, int, int, int]:
    """
    计算WER（词错误率）。
    
    返回:
        (wer, substitutions, deletions, insertions, num_words)
        wer: 词错误率（0-1之间）
        substitutions: 替换错误数
        deletions: 删除错误数
        insertions: 插入错误数
        num_words: 参考文本中的词数
    """
    # 使用jiwer库计算WER
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
    ])
    
    # 应用文本规范化
    ref_normalized = transformation(reference)
    hyp_normalized = transformation(hypothesis)
    
    # 计算WER
    measures = jiwer.compute_measures(ref_normalized, hyp_normalized)
    
    substitutions = measures['substitutions']
    deletions = measures['deletions']
    insertions = measures['insertions']
    hits = measures['hits']
    num_words = substitutions + deletions + hits
    
    if num_words == 0:
        # 参考文本为空
        if len(hyp_normalized.strip()) == 0:
            wer = 0.0  # 两者都为空，WER为0
        else:
            wer = 1.0  # 参考为空但识别结果不为空，WER为1（100%错误）
    else:
        wer = (substitutions + deletions + insertions) / num_words
    
    return wer, substitutions, deletions, insertions, num_words


def calculate_ser(reference: str, hypothesis: str) -> bool:
    """
    计算SER（句子错误率）。
    
    返回:
        True 如果句子有错误，False 如果完全正确
    """
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
    ])
    
    ref_normalized = transformation(reference)
    hyp_normalized = transformation(hypothesis)
    
    return ref_normalized != hyp_normalized


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="计算ASR识别结果的WER（词错误率）"
    )
    
    parser.add_argument(
        "--reference",
        type=str,
        required=True,
        help="参考文本文件路径（ground truth）"
    )
    
    parser.add_argument(
        "--hypothesis",
        type=str,
        required=True,
        help="ASR识别结果文件路径（hypothesis）"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="输出结果文件路径（可选，如果不指定则只打印到控制台）"
    )
    
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="输出详细的对齐信息"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("WER（词错误率）计算工具")
    print("=" * 60)
    
    ref_path = Path(args.reference)
    hyp_path = Path(args.hypothesis)
    
    # 加载参考文本和识别结果
    print(f"\n[1/3] 加载文件...")
    reference_dict = load_text_file(ref_path)
    hypothesis_dict = load_text_file(hyp_path)
    
    # 检查segment_id是否一致
    ref_ids = set(reference_dict.keys())
    hyp_ids = set(hypothesis_dict.keys())
    
    if ref_ids != hyp_ids:
        missing_in_hyp = ref_ids - hyp_ids
        missing_in_ref = hyp_ids - ref_ids
        if missing_in_hyp:
            print(f"  警告: 识别结果中缺少以下segment_id: {sorted(missing_in_hyp)[:10]}...")
        if missing_in_ref:
            print(f"  警告: 参考文本中缺少以下segment_id: {sorted(missing_in_ref)[:10]}...")
    
    # 只计算两者都有的segment_id
    common_ids = sorted(ref_ids & hyp_ids)
    if not common_ids:
        raise ValueError("参考文本和识别结果中没有共同的segment_id！")
    
    print(f"  共同segment_id数量: {len(common_ids)}")
    
    # 计算WER
    print(f"\n[2/3] 计算WER...")
    total_wer = 0.0
    total_substitutions = 0
    total_deletions = 0
    total_insertions = 0
    total_words = 0
    sentence_errors = 0
    total_sentences = len(common_ids)
    
    detailed_results = []
    
    for segment_id in common_ids:
        ref_text = reference_dict[segment_id]
        hyp_text = hypothesis_dict[segment_id]
        
        wer, subs, dels, ins, num_words = calculate_wer(ref_text, hyp_text)
        has_error = calculate_ser(ref_text, hyp_text)
        
        total_wer += wer * num_words  # 加权平均
        total_substitutions += subs
        total_deletions += dels
        total_insertions += ins
        total_words += num_words
        
        if has_error:
            sentence_errors += 1
        
        if args.detailed:
            detailed_results.append({
                'segment_id': segment_id,
                'reference': ref_text,
                'hypothesis': hyp_text,
                'wer': wer,
                'substitutions': subs,
                'deletions': dels,
                'insertions': ins,
                'num_words': num_words,
                'has_error': has_error
            })
    
    # 计算总体统计
    print(f"\n[3/3] 生成统计报告...")
    
    if total_words > 0:
        overall_wer = total_wer / total_words
    else:
        overall_wer = 0.0
    
    ser = sentence_errors / total_sentences if total_sentences > 0 else 0.0
    
    # 打印统计信息
    print(f"\n{'=' * 60}")
    print("WER统计结果")
    print(f"{'=' * 60}")
    print(f"总句子数: {total_sentences}")
    print(f"总词数: {total_words}")
    print(f"\n总体WER: {overall_wer:.4f} ({overall_wer * 100:.2f}%)")
    print(f"总体SER: {ser:.4f} ({ser * 100:.2f}%)")
    print(f"\n错误统计:")
    print(f"  替换错误 (S): {total_substitutions}")
    print(f"  删除错误 (D): {total_deletions}")
    print(f"  插入错误 (I): {total_insertions}")
    print(f"  总错误数: {total_substitutions + total_deletions + total_insertions}")
    print(f"\n错误率分布:")
    if total_words > 0:
        print(f"  替换率: {total_substitutions / total_words:.4f} ({total_substitutions / total_words * 100:.2f}%)")
        print(f"  删除率: {total_deletions / total_words:.4f} ({total_deletions / total_words * 100:.2f}%)")
        print(f"  插入率: {total_insertions / total_words:.4f} ({total_insertions / total_words * 100:.2f}%)")
    
    # 输出详细结果到文件
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"WER统计结果\n")
            f.write(f"{'=' * 60}\n")
            f.write(f"参考文件: {ref_path}\n")
            f.write(f"识别结果文件: {hyp_path}\n")
            f.write(f"\n总体统计:\n")
            f.write(f"  总句子数: {total_sentences}\n")
            f.write(f"  总词数: {total_words}\n")
            f.write(f"  总体WER: {overall_wer:.4f} ({overall_wer * 100:.2f}%)\n")
            f.write(f"  总体SER: {ser:.4f} ({ser * 100:.2f}%)\n")
            f.write(f"\n错误统计:\n")
            f.write(f"  替换错误 (S): {total_substitutions}\n")
            f.write(f"  删除错误 (D): {total_deletions}\n")
            f.write(f"  插入错误 (I): {total_insertions}\n")
            f.write(f"  总错误数: {total_substitutions + total_deletions + total_insertions}\n")
            
            if args.detailed:
                f.write(f"\n详细结果:\n")
                f.write(f"{'=' * 60}\n")
                for result in detailed_results:
                    f.write(f"\nSegment ID: {result['segment_id']}\n")
                    f.write(f"  Reference: {result['reference']}\n")
                    f.write(f"  Hypothesis: {result['hypothesis']}\n")
                    f.write(f"  WER: {result['wer']:.4f}\n")
                    f.write(f"  错误: S={result['substitutions']}, D={result['deletions']}, I={result['insertions']}\n")
                    f.write(f"  词数: {result['num_words']}\n")
                    f.write(f"  有错误: {'是' if result['has_error'] else '否'}\n")
        
        print(f"\n详细结果已保存到: {output_path}")
    
    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()

