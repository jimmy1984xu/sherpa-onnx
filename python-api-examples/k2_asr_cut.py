from __future__ import annotations

import argparse
import wave
from pathlib import Path


def parse_asr_line(line: str, line_no: int) -> tuple[int, int, str, str]:
    parts = line.strip().split(maxsplit=3)
    if len(parts) != 4:
        raise ValueError(
            f"ASR 文件第 {line_no} 行格式错误，应为 4 列: offsetMs durationMs speakerID text"
        )

    offset_ms_text, duration_ms_text, speaker_id, text = parts

    try:
        offset_ms = int(offset_ms_text)
    except ValueError as exc:
        raise ValueError(f"ASR 文件第 {line_no} 行 offsetMs 不是整数: {offset_ms_text}") from exc

    try:
        duration_ms = int(duration_ms_text)
    except ValueError as exc:
        raise ValueError(f"ASR 文件第 {line_no} 行 durationMs 不是整数: {duration_ms_text}") from exc

    if offset_ms < 0:
        raise ValueError(f"ASR 文件第 {line_no} 行 offsetMs 必须大于等于 0")
    if duration_ms <= 0:
        raise ValueError(f"ASR 文件第 {line_no} 行 durationMs 必须大于 0")

    return offset_ms, duration_ms, speaker_id, text


def read_asr_segments(asr_path: Path) -> list[tuple[int, int, str, str]]:
    segments: list[tuple[int, int, str, str]] = []
    with asr_path.open("r", encoding="utf-8-sig") as file:
        for line_no, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            segments.append(parse_asr_line(line, line_no))

    if not segments:
        raise ValueError("ASR 文件没有有效分段数据")

    return segments


def ms_to_byte_pos(ms_value: int, sample_rate: int, channels: int, sample_width: int) -> int:
    frame_index = ms_value * sample_rate // 1000
    return frame_index * channels * sample_width


def write_wav_file(
    output_path: Path,
    audio_bytes: bytes,
    channels: int,
    sample_width: int,
    sample_rate: int,
) -> None:
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_bytes)


def write_wav_scp(entries: list[tuple[str, Path]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as file:
        for audio_id, audio_path in entries:
            file.write(f"{audio_id} {audio_path.resolve()}\n")


def split_pcm_to_wav(
    pcm_path: Path,
    asr_path: Path,
    output_dir: Path,
    sample_rate: int,
    channels: int,
    sample_width: int,
) -> tuple[int, Path]:
    pcm_bytes = pcm_path.read_bytes()
    frame_size = channels * sample_width
    if frame_size <= 0:
        raise ValueError("frame_size 必须大于 0")

    if len(pcm_bytes) % frame_size != 0:
        raise ValueError("PCM 文件字节数不是 frame_size 的整数倍，请检查声道数和采样位宽")

    segments = read_asr_segments(asr_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_scp_entries: list[tuple[str, Path]] = []

    generated_count = 0
    for offset_ms, duration_ms, _speaker_id, _text in segments:
        start_byte = ms_to_byte_pos(offset_ms, sample_rate, channels, sample_width)
        end_byte = ms_to_byte_pos(offset_ms + duration_ms, sample_rate, channels, sample_width)

        if start_byte >= len(pcm_bytes):
            print(
                f"警告: 分段 offsetMs={offset_ms} durationMs={duration_ms} 超出 PCM 文件范围，已跳过"
            )
            continue

        clipped_end_byte = min(end_byte, len(pcm_bytes))
        audio_slice = pcm_bytes[start_byte:clipped_end_byte]
        if not audio_slice:
            print(
                f"警告: 分段 offsetMs={offset_ms} durationMs={duration_ms} 未截取到有效音频，已跳过"
            )
            continue

        output_name = f"{pcm_path.stem}_{offset_ms}_{duration_ms}.wav"
        output_path = output_dir / output_name
        write_wav_file(output_path, audio_slice, channels, sample_width, sample_rate)
        wav_scp_entries.append((output_path.stem, output_path))
        generated_count += 1

    wav_scp_path = output_dir / "wav.scp"
    write_wav_scp(wav_scp_entries, wav_scp_path)
    return generated_count, wav_scp_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="根据 ASR 文件切分 PCM 音频并输出为 WAV 文件"
    )
    parser.add_argument("pcm_file", help="输入 PCM 音频文件路径")
    parser.add_argument(
        "asr_file",
        help="ASR 文件路径，每行格式为: offsetMs durationMs speakerID text",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        help="输出目录。默认在输入音频文件同级目录下创建 wav 子目录",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="采样率，默认 16000",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        help="声道数，默认 1",
    )
    parser.add_argument(
        "--sample-width",
        type=int,
        default=2,
        help="单采样字节数，默认 2，表示 16bit PCM",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pcm_path = Path(args.pcm_file)
    asr_path = Path(args.asr_file)
    output_dir = Path(args.output_dir) if args.output_dir else pcm_path.parent / "wav"

    if not pcm_path.exists():
        raise FileNotFoundError(f"PCM 文件不存在: {pcm_path}")
    if not asr_path.exists():
        raise FileNotFoundError(f"ASR 文件不存在: {asr_path}")
    if args.sample_rate <= 0:
        raise ValueError("--sample-rate 必须大于 0")
    if args.channels <= 0:
        raise ValueError("--channels 必须大于 0")
    if args.sample_width <= 0:
        raise ValueError("--sample-width 必须大于 0")

    generated_count, wav_scp_path = split_pcm_to_wav(
        pcm_path=pcm_path,
        asr_path=asr_path,
        output_dir=output_dir,
        sample_rate=args.sample_rate,
        channels=args.channels,
        sample_width=args.sample_width,
    )

    print(f"输出目录: {output_dir}")
    print(f"生成 wav 数量: {generated_count}")
    print(f"已生成 wav.scp: {wav_scp_path}")


if __name__ == "__main__":
    main()
