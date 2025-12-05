#!/usr/bin/env python3

"""
整合VAD切割、ASR识别和声纹识别的完整流程。

本脚本整合了三个步骤：
1. VAD切割：使用 k2_vad_cut.py 将长音频切割成短片段
2. ASR识别：使用 k2-asr-short.py 对每个片段进行语音识别
3. 声纹识别：使用 k2-speaker-identification.py 提取每个片段的声纹向量

使用示例:

# 使用 Paraformer 进行ASR识别
python3 ./python-api-examples/k2-vad-cut-asr-speaker-identification.py \
  --audio /path/to/long_audio.wav \
  --silero-vad-model /path/to/silero_vad.onnx \
  --paraformer /path/to/paraformer.onnx \
  --tokens /path/to/tokens.txt \
  --speaker-model /path/to/speaker_embedding.onnx \
  --output-dir ./output

# 使用 Transducer 进行ASR识别
python3 ./python-api-examples/k2-vad-cut-asr-speaker-identification.py \
  --audio /path/to/long_audio.wav \
  --silero-vad-model /path/to/silero_vad.onnx \
  --encoder /path/to/encoder.onnx \
  --decoder /path/to/decoder.onnx \
  --joiner /path/to/joiner.onnx \
  --tokens /path/to/tokens.txt \
  --speaker-model /path/to/speaker_embedding.onnx \
  --output-dir ./output

# 使用标点模型（可选）
python3 ./python-api-examples/k2-vad-cut-asr-speaker-identification.py \
  --audio /path/to/long_audio.wav \
  --silero-vad-model /path/to/silero_vad.onnx \
  --paraformer /path/to/paraformer.onnx \
  --tokens /path/to/tokens.txt \
  --punct-model /path/to/punctuation_model.onnx \
  --speaker-model /path/to/speaker_embedding.onnx \
  --output-dir ./output

输出文件:
- wav.scp: 片段ID和音频文件路径的映射
- {model_name}_asr.txt: ASR识别结果（格式: segment_id asr_text）
- {model_name}_embedding.txt: 声纹识别结果（格式: segment_id base64_embedding）

注意事项:
- 所有中间文件会保存在指定的输出目录中
- 如果某个步骤失败，脚本会停止并报告错误
- 标点模型（--punct-model）是可选的，如果提供，ASR识别结果会自动添加标点符号
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


def register_non_streaming_asr_model_args(parser):
    """注册非流式ASR模型参数（从k2-asr-short.py复制）。"""
    parser.add_argument(
        "--tokens",
        type=str,
        help="Path to tokens.txt",
    )

    parser.add_argument(
        "--encoder",
        default="",
        type=str,
        help="Path to the transducer encoder model",
    )

    parser.add_argument(
        "--decoder",
        default="",
        type=str,
        help="Path to the transducer decoder model",
    )

    parser.add_argument(
        "--joiner",
        default="",
        type=str,
        help="Path to the transducer joiner model",
    )

    parser.add_argument(
        "--paraformer",
        default="",
        type=str,
        help="Path to the model.onnx from Paraformer",
    )

    parser.add_argument(
        "--wenet-ctc",
        default="",
        type=str,
        help="Path to the CTC model.onnx from WeNet",
    )

    parser.add_argument(
        "--whisper-encoder",
        default="",
        type=str,
        help="Path to whisper encoder model",
    )

    parser.add_argument(
        "--whisper-decoder",
        default="",
        type=str,
        help="Path to whisper decoder model",
    )

    parser.add_argument(
        "--whisper-language",
        default="",
        type=str,
        help="Whisper language code (e.g., en, zh, jp)",
    )

    parser.add_argument(
        "--whisper-task",
        default="transcribe",
        choices=["transcribe", "translate"],
        type=str,
        help="Whisper task: transcribe or translate",
    )

    parser.add_argument(
        "--whisper-tail-paddings",
        default=-1,
        type=int,
        help="Number of tail padding frames for Whisper",
    )

    parser.add_argument(
        "--decoding-method",
        type=str,
        default="greedy_search",
        help="Decoding method: greedy_search or modified_beam_search",
    )

    parser.add_argument(
        "--feature-dim",
        type=int,
        default=80,
        help="Feature dimension",
    )

    parser.add_argument(
        "--sense-voice",
        default="",
        type=str,
        help="Path to sense voice model",
    )

    parser.add_argument(
        "--punct-model",
        type=str,
        default="",
        help="Path to Offline punctuation ct-transformer model.onnx (optional)",
    )


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="整合VAD切割、ASR识别和声纹识别的完整流程"
    )
    
    # VAD参数
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="长音频文件路径"
    )
    
    parser.add_argument(
        "--silero-vad-model",
        type=str,
        default="",
        help="Silero VAD模型路径"
    )
    
    parser.add_argument(
        "--ten-vad-model",
        type=str,
        default="",
        help="Ten VAD模型路径"
    )
    
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.5,
        help="VAD阈值（默认: 0.5）"
    )
    
    parser.add_argument(
        "--min-silence-duration",
        type=float,
        default=0.5,
        help="最小静音时长（秒，默认: 0.5）"
    )
    
    parser.add_argument(
        "--min-speech-duration",
        type=float,
        default=0.25,
        help="最小语音时长（秒，默认: 0.25）"
    )
    
    parser.add_argument(
        "--max-speech-duration",
        type=float,
        default=20.0,
        help="最大语音时长（秒，默认: 20.0）"
    )
    
    # ASR模型参数（通过register_non_streaming_asr_model_args注册）
    register_non_streaming_asr_model_args(parser)
    
    # 声纹模型参数
    parser.add_argument(
        "--speaker-model",
        type=str,
        required=True,
        help="说话人声纹模型路径（.onnx）"
    )
    
    # 输出参数
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="输出目录路径"
    )
    
    # 通用参数
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="推理线程数"
    )
    
    parser.add_argument(
        "--provider",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "coreml"],
        help="推理提供者"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试日志"
    )
    
    return parser.parse_args()


def build_vad_cut_cmd(args, script_dir: Path, output_dir: Path) -> list:
    """构建VAD切割命令。"""
    cmd = [
        sys.executable,
        str(script_dir / "k2_vad_cut.py"),
        "--audio", args.audio,
        "--output-dir", str(output_dir),
        "--wav-scp", str(output_dir / "wav.scp"),
        "--threshold", str(args.vad_threshold),
        "--min-silence-duration", str(args.min_silence_duration),
        "--min-speech-duration", str(args.min_speech_duration),
        "--max-speech-duration", str(args.max_speech_duration),
    ]
    
    if args.silero_vad_model:
        cmd.extend(["--silero-vad-model", args.silero_vad_model])
    elif args.ten_vad_model:
        cmd.extend(["--ten-vad-model", args.ten_vad_model])
    
    if args.debug:
        cmd.append("--debug")
    
    return cmd


def build_asr_cmd(args, script_dir: Path, output_dir: Path) -> list:
    """构建ASR识别命令。"""
    cmd = [
        sys.executable,
        str(script_dir / "k2-asr-short.py"),
        "--wav-scp", str(output_dir / "wav.scp"),
        "--output-dir", str(output_dir),
        "--num-threads", str(args.num_threads),
        "--provider", args.provider,
    ]
    
    # 添加ASR模型参数
    if args.encoder:
        cmd.extend(["--encoder", args.encoder])
    if args.decoder:
        cmd.extend(["--decoder", args.decoder])
    if args.joiner:
        cmd.extend(["--joiner", args.joiner])
    if args.paraformer:
        cmd.extend(["--paraformer", args.paraformer])
    if args.wenet_ctc:
        cmd.extend(["--wenet-ctc", args.wenet_ctc])
    if args.whisper_encoder:
        cmd.extend(["--whisper-encoder", args.whisper_encoder])
    if args.whisper_decoder:
        cmd.extend(["--whisper-decoder", args.whisper_decoder])
    if args.whisper_language:
        cmd.extend(["--whisper-language", args.whisper_language])
    if args.whisper_task:
        cmd.extend(["--whisper-task", args.whisper_task])
    if args.whisper_tail_paddings != -1:
        cmd.extend(["--whisper-tail-paddings", str(args.whisper_tail_paddings)])
    if args.decoding_method:
        cmd.extend(["--decoding-method", args.decoding_method])
    if args.feature_dim:
        cmd.extend(["--feature-dim", str(args.feature_dim)])
    if args.sense_voice:
        cmd.extend(["--sense-voice", args.sense_voice])
    if args.tokens:
        cmd.extend(["--tokens", args.tokens])
    
    # 添加标点模型参数（可选）
    if args.punct_model:
        cmd.extend(["--punct-model", args.punct_model])
    
    if args.debug:
        cmd.append("--debug")
    
    return cmd


def build_speaker_identification_cmd(args, script_dir: Path, output_dir: Path) -> list:
    """构建声纹识别命令。"""
    cmd = [
        sys.executable,
        str(script_dir / "k2-speaker-identification.py"),
        "--wav-scp", str(output_dir / "wav.scp"),
        "--model", args.speaker_model,
        "--output-dir", str(output_dir),
        "--num-threads", str(args.num_threads),
        "--provider", args.provider,
    ]
    
    if args.debug:
        cmd.append("--debug")
    
    return cmd


def validate_args(args) -> bool:
    """验证参数。"""
    # 检查VAD模型
    if not args.silero_vad_model and not args.ten_vad_model:
        print("错误: 必须指定 --silero-vad-model 或 --ten-vad-model 之一")
        return False
    
    if args.silero_vad_model and args.ten_vad_model:
        print("错误: 不能同时指定 --silero-vad-model 和 --ten-vad-model")
        return False
    
    # 检查ASR模型
    has_asr_model = (
        args.encoder or args.paraformer or args.wenet_ctc or
        args.whisper_encoder or args.sense_voice
    )
    if not has_asr_model:
        print("错误: 必须指定至少一个ASR模型（--encoder/--paraformer/--wenet-ctc/--whisper-encoder/--sense-voice）")
        return False
    
    # 检查音频文件
    audio_path = Path(args.audio)
    if not audio_path.is_file():
        print(f"错误: 音频文件不存在: {audio_path}")
        return False
    
    # 检查声纹模型
    speaker_model_path = Path(args.speaker_model)
    if not speaker_model_path.is_file():
        print(f"错误: 声纹模型文件不存在: {speaker_model_path}")
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
    print(f"整合流程: VAD切割 -> ASR识别 -> 声纹识别")
    print(f"{'='*60}")
    print(f"输入音频: {args.audio}")
    print(f"输出目录: {output_dir}")
    print()
    
    # 步骤1: VAD切割
    vad_cmd = build_vad_cut_cmd(args, script_dir, output_dir)
    if not run_command(vad_cmd, "[1/3] VAD切割"):
        print("\n错误: VAD切割失败，流程终止")
        sys.exit(1)
    
    # 检查wav.scp是否生成
    wav_scp_path = output_dir / "wav.scp"
    if not wav_scp_path.is_file():
        print(f"\n错误: wav.scp文件未生成: {wav_scp_path}")
        sys.exit(1)
    
    # 步骤2: ASR识别
    asr_cmd = build_asr_cmd(args, script_dir, output_dir)
    if not run_command(asr_cmd, "[2/3] ASR识别"):
        print("\n错误: ASR识别失败，流程终止")
        sys.exit(1)
    
    # 步骤3: 声纹识别
    speaker_cmd = build_speaker_identification_cmd(args, script_dir, output_dir)
    if not run_command(speaker_cmd, "[3/3] 声纹识别"):
        print("\n错误: 声纹识别失败，流程终止")
        sys.exit(1)
    
    # 完成
    print(f"\n{'='*60}")
    print(f"流程完成！")
    print(f"{'='*60}")
    print(f"输出目录: {output_dir}")
    print(f"生成的文件:")
    print(f"  - wav.scp: 片段ID和音频文件路径映射")
    
    # 查找ASR结果文件
    asr_files = list(output_dir.glob("*_asr.txt"))
    if asr_files:
        print(f"  - {asr_files[0].name}: ASR识别结果")
    
    # 查找声纹结果文件
    embedding_files = list(output_dir.glob("*_embedding.txt"))
    if embedding_files:
        print(f"  - {embedding_files[0].name}: 声纹识别结果")
    
    print()


if __name__ == "__main__":
    main()

