#!/usr/bin/env python3

"""
整合说话人聚类和结果合并的完整流程。

本脚本整合了两个步骤：
1. 说话人聚类：使用 k2-speaker-cluster-realtime.py 或 k2-speaker-cluster-fastclustering.py
   对声纹进行聚类，得到每个片段的说话人ID
2. 结果合并：使用 k2-merge-asr-speaker.py 将ASR结果和说话人聚类结果合并

使用示例:

# 使用实时聚类算法
python3 ./python-api-examples/k2-speaker-cluster-merge.py \
  --asr-result ./output/paraformer_asr.txt \
  --embedding-result ./output/speaker_embedding.txt \
  --clustering-method realtime \
  --threshold 0.8 \
  --min-duration-seconds 3.0 \
  --max-embeddings-per-speaker 10 \
  --output-dir ./output

# 使用实时聚类算法（自定义合并阈值）
python3 ./python-api-examples/k2-speaker-cluster-merge.py \
  --asr-result ./output/paraformer_asr.txt \
  --embedding-result ./output/speaker_embedding.txt \
  --clustering-method realtime \
  --threshold 0.8 \
  --merge-threshold 0.85 \
  --min-duration-seconds 3.0 \
  --max-embeddings-per-speaker 10 \
  --output-dir ./output

# 使用Fast Clustering算法（固定聚类数）
python3 ./python-api-examples/k2-speaker-cluster-merge.py \
  --asr-result ./output/paraformer_asr.txt \
  --embedding-result ./output/speaker_embedding.txt \
  --clustering-method fastclustering \
  --num-clusters 3 \
  --output-dir ./output

# 使用Fast Clustering算法（阈值模式）
python3 ./python-api-examples/k2-speaker-cluster-merge.py \
  --asr-result ./output/paraformer_asr.txt \
  --embedding-result ./output/speaker_embedding.txt \
  --clustering-method fastclustering \
  --threshold 0.5 \
  --output-dir ./output

输出文件:
- 说话人聚类结果文件（根据聚类方法和参数自动命名）
- 合并结果文件（格式: segment_id speaker_id asr_text）

注意事项:
- 所有中间文件会保存在指定的输出目录中
- 如果某个步骤失败，脚本会停止并报告错误
"""

import argparse
import subprocess
import sys
from pathlib import Path


def get_script_dir() -> Path:
    """获取脚本所在目录。"""
    return Path(__file__).parent


def run_command(cmd: list, description: str) -> bool:
    """
    运行命令并检查返回码。
    
    参数:
        cmd: 命令列表
        description: 命令描述
    
    返回:
        成功返回True，失败返回False
    """
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"执行命令: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"\n错误: {description} 失败，返回码: {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n错误: 找不到脚本文件，请检查路径是否正确")
        return False


def build_realtime_clustering_cmd(
    args,
    script_dir: Path,
    output_dir: Path,
) -> list:
    """构建实时聚类命令。"""
    cmd = [
        sys.executable,
        str(script_dir / "k2-speaker-cluster-realtime.py"),
        "--input", args.embedding_result,
        "--output-dir", str(output_dir),
        "--threshold", str(args.threshold),
        "--min-duration-seconds", str(args.min_duration_seconds),
        "--max-embeddings-per-speaker", str(args.max_embeddings_per_speaker),
    ]
    
    # 添加merge-threshold参数（如果指定）
    if args.merge_threshold is not None:
        cmd.extend(["--merge-threshold", str(args.merge_threshold)])
    
    if args.verbose:
        cmd.append("--verbose")
    
    return cmd


def build_fastclustering_cmd(
    args,
    script_dir: Path,
    output_dir: Path,
) -> list:
    """构建Fast Clustering命令。"""
    cmd = [
        sys.executable,
        str(script_dir / "k2-speaker-cluster-fastclustering.py"),
        "--input", args.embedding_result,
        "--output-dir", str(output_dir),
    ]
    
    if args.num_clusters > 0:
        cmd.extend(["--num-clusters", str(args.num_clusters)])
    else:
        cmd.extend(["--threshold", str(args.threshold)])
    
    if args.verbose:
        cmd.append("--verbose")
    
    return cmd


def build_merge_cmd(
    args,
    script_dir: Path,
    output_dir: Path,
    speaker_result_path: Path,
) -> list:
    """构建合并命令。"""
    cmd = [
        sys.executable,
        str(script_dir / "k2-merge-asr-speaker.py"),
        "--asr-result", args.asr_result,
        "--speaker-result", str(speaker_result_path),
        "--output-dir", str(output_dir),
    ]
    
    return cmd


def predict_speaker_result_filename(
    clustering_method: str,
    args,
) -> str:
    """
    根据聚类方法和参数预测输出文件名。
    
    注意：这只是预测，实际文件名由聚类脚本生成。
    """
    if clustering_method == "realtime":
        threshold_str = f"{args.threshold:.2f}".replace(".", "p")
        # 确定merge_threshold的值（如果未指定则使用threshold）
        merge_threshold = args.merge_threshold if args.merge_threshold is not None else args.threshold
        merge_threshold_str = f"{merge_threshold:.2f}".replace(".", "p")
        min_duration_str = f"{args.min_duration_seconds:.1f}".replace(".", "p")
        filename = f"cluster-realtime-th-{threshold_str}-merge-{merge_threshold_str}-min-{min_duration_str}-max-embeddings-{args.max_embeddings_per_speaker}.txt"
    else:  # fastclustering
        if args.num_clusters > 0:
            filename = f"cluster-fastclustering-k-{args.num_clusters}.txt"
        else:
            threshold_str = f"{args.threshold:.2f}".replace(".", "p")
            filename = f"cluster-fastclustering-th-{threshold_str}.txt"
    
    return filename


def find_speaker_result_file(output_dir: Path, predicted_filename: str) -> Path:
    """
    在输出目录中查找说话人聚类结果文件。
    
    如果找不到预测的文件名，则查找所有以cluster-开头的txt文件。
    """
    predicted_path = output_dir / predicted_filename
    if predicted_path.is_file():
        return predicted_path
    
    # 查找所有聚类结果文件
    cluster_files = list(output_dir.glob("cluster-*.txt"))
    if cluster_files:
        # 返回最新的文件
        return max(cluster_files, key=lambda p: p.stat().st_mtime)
    
    raise FileNotFoundError(
        f"未找到说话人聚类结果文件。预期文件: {predicted_filename}"
    )


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="整合说话人聚类和结果合并的完整流程"
    )
    
    # 输入文件
    parser.add_argument(
        "--asr-result",
        type=str,
        required=True,
        help="ASR结果文件路径（格式: segment_id asr_text）"
    )
    
    parser.add_argument(
        "--embedding-result",
        type=str,
        required=True,
        help="声纹识别结果文件路径（格式: segment_id base64_embedding）"
    )
    
    # 聚类方法
    parser.add_argument(
        "--clustering-method",
        type=str,
        required=True,
        choices=["realtime", "fastclustering"],
        help="聚类方法: realtime（实时聚类）或 fastclustering（Fast Clustering）"
    )
    
    # 实时聚类参数
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="同一个说话人相似度阈值（0.0-1.0）。用于realtime和fastclustering阈值模式。"
    )
    
    parser.add_argument(
        "--merge-threshold",
        type=float,
        default=None,
        help="同一个说话人合并声纹的相似度阈值（0.0-1.0）。这个值要求更高，大于这个值才能合并到说话人的声纹列表中。"
             "如果未指定，默认使用threshold的值。仅用于realtime方法。"
    )
    
    parser.add_argument(
        "--min-duration-seconds",
        type=float,
        default=3.0,
        help="被视为长句的最小片段时长（秒）。仅用于realtime方法。"
    )
    
    parser.add_argument(
        "--max-embeddings-per-speaker",
        type=int,
        default=10,
        help="每个说话人用于更新均值声纹的最大声纹数量。仅用于realtime方法。"
    )
    
    # Fast Clustering参数
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=-1,
        help="说话人数量（固定聚类数模式）。仅用于fastclustering方法。"
             "如果 > 0，使用固定聚类数模式；如果 <= 0，使用阈值模式。"
    )
    
    # 输出参数
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="输出目录路径"
    )
    
    # 其他参数
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="启用详细输出"
    )
    
    return parser.parse_args()


def validate_args(args) -> bool:
    """验证参数。"""
    # 检查ASR结果文件
    asr_path = Path(args.asr_result)
    if not asr_path.is_file():
        print(f"错误: ASR结果文件不存在: {asr_path}")
        return False
    
    # 检查声纹结果文件
    embedding_path = Path(args.embedding_result)
    if not embedding_path.is_file():
        print(f"错误: 声纹结果文件不存在: {embedding_path}")
        return False
    
    # 验证阈值
    if not 0.0 <= args.threshold <= 1.0:
        print(f"错误: threshold必须在0.0和1.0之间，得到 {args.threshold}")
        return False
    
    # 验证merge_threshold（如果指定）
    if args.merge_threshold is not None:
        if not 0.0 <= args.merge_threshold <= 1.0:
            print(f"错误: merge_threshold必须在0.0和1.0之间，得到 {args.merge_threshold}")
            return False
        if args.merge_threshold < args.threshold:
            print(f"错误: merge_threshold ({args.merge_threshold}) 应该 >= threshold ({args.threshold})")
            return False
    
    # 验证实时聚类参数
    if args.clustering_method == "realtime":
        if args.min_duration_seconds <= 0:
            print(f"错误: min-duration-seconds必须为正数，得到 {args.min_duration_seconds}")
            return False
        if args.max_embeddings_per_speaker <= 0:
            print(f"错误: max-embeddings-per-speaker必须为正数，得到 {args.max_embeddings_per_speaker}")
            return False
    
    return True


def main():
    """主函数。"""
    args = parse_args()
    
    # 验证参数
    if not validate_args(args):
        sys.exit(1)
    
    script_dir = get_script_dir()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"整合流程: 说话人聚类 -> 结果合并")
    print(f"{'='*60}")
    print(f"ASR结果文件: {args.asr_result}")
    print(f"声纹结果文件: {args.embedding_result}")
    print(f"聚类方法: {args.clustering_method}")
    print(f"输出目录: {output_dir}")
    print()
    
    # 步骤1: 说话人聚类
    if args.clustering_method == "realtime":
        clustering_cmd = build_realtime_clustering_cmd(args, script_dir, output_dir)
        clustering_desc = "[1/2] 实时说话人聚类"
    else:  # fastclustering
        clustering_cmd = build_fastclustering_cmd(args, script_dir, output_dir)
        clustering_desc = "[1/2] Fast Clustering说话人聚类"
    
    if not run_command(clustering_cmd, clustering_desc):
        print("\n错误: 说话人聚类失败，流程终止")
        sys.exit(1)
    
    # 查找说话人聚类结果文件
    predicted_filename = predict_speaker_result_filename(args.clustering_method, args)
    try:
        speaker_result_path = find_speaker_result_file(output_dir, predicted_filename)
        print(f"\n找到说话人聚类结果文件: {speaker_result_path}")
    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        sys.exit(1)
    
    # 步骤2: 结果合并
    merge_cmd = build_merge_cmd(args, script_dir, output_dir, speaker_result_path)
    if not run_command(merge_cmd, "[2/2] 合并ASR和说话人聚类结果"):
        print("\n错误: 结果合并失败，流程终止")
        sys.exit(1)
    
    # 查找合并结果文件
    merge_files = list(output_dir.glob("merged-asr-speaker-*.txt"))
    if merge_files:
        merge_result_path = merge_files[0]
        print(f"\n找到合并结果文件: {merge_result_path}")
    
    # 完成
    print(f"\n{'='*60}")
    print(f"流程完成！")
    print(f"{'='*60}")
    print(f"输出目录: {output_dir}")
    print(f"生成的文件:")
    print(f"  - {speaker_result_path.name}: 说话人聚类结果")
    if merge_files:
        print(f"  - {merge_result_path.name}: 合并结果（segment_id speaker_id asr_text）")
    print()


if __name__ == "__main__":
    main()

