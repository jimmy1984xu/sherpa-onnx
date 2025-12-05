# 离线 ASR 使用文档

## 概述

`sherpa-onnx` 提供了强大的离线语音识别（ASR）功能，支持多种模型架构，包括 Paraformer、Transducer、Whisper、CTC 等。本文档重点介绍 Paraformer 模型的使用方法。

关于文本置信度功能的详细说明，请参考 [ASR 文本置信度实现文档](OFFLINE_ASR_CONFIDENCE.md)。

## 支持的模型类型

`sherpa-onnx` 支持多种离线 ASR 模型架构，涵盖非自回归、自回归和 CTC 等多种类型。以下是完整的模型列表：

### 1. Paraformer
- **特点**: 非自回归模型，推理速度快，适合中文识别
- **模型文件**: 单个 `model.onnx` 文件
- **模型类型**: `paraformer`
- **适用场景**: 中文离线识别、批量处理
- **特殊功能**: ✅ 支持文本置信度（confidence）
- **下载地址**: https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-paraformer/index.html

### 2. Transducer (Zipformer)
- **特点**: 自回归模型，支持流式和离线识别，基于 Zipformer 架构
- **模型文件**: `encoder.onnx`, `decoder.onnx`, `joiner.onnx`
- **模型类型**: `transducer`
- **适用场景**: 中英文混合、多语言识别、高准确率场景
- **下载地址**: https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/index.html

### 3. NeMo Transducer
- **特点**: NVIDIA NeMo 框架训练的 Transducer 模型
- **模型文件**: `encoder.onnx`, `decoder.onnx`, `joiner.onnx`
- **模型类型**: `nemo_transducer`
- **适用场景**: NeMo 生态模型使用、多语言识别
- **下载地址**: https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/index.html

### 4. Whisper
- **特点**: OpenAI 的多语言模型，支持语音转录和翻译
- **模型文件**: `encoder.onnx`, `decoder.onnx`
- **模型类型**: `whisper`
- **适用场景**: 多语言识别、翻译任务、通用语音识别
- **特殊功能**: 支持语言检测、转录/翻译模式切换
- **下载地址**: https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-whisper/index.html

### 5. SenseVoice
- **特点**: 阿里达摩院的多语言语音识别模型，支持中文、英文、日文、韩文、粤语等
- **模型文件**: 单个 `model.onnx` 文件
- **模型类型**: `sense_voice`
- **适用场景**: 多语言识别、中文方言识别、自动语言检测
- **特殊功能**: 支持逆文本归一化（ITN）、自动语言识别
- **支持语言**: 中文（zh）、英文（en）、日文（ja）、韩文（ko）、粤语（yue）、自动检测（auto）
- **下载地址**: https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-ctc/index.html

### 6. Moonshine
- **特点**: 基于 Wav2Vec 的模型，使用神经网络预处理器
- **模型文件**: `preprocessor.onnx`, `encoder.onnx`, `uncached_decoder.onnx`, `cached_decoder.onnx`
- **模型类型**: `moonshine`
- **适用场景**: 原始音频输入、资源受限环境
- **特殊功能**: 支持缓存解码器以提升性能

### 7. FireRed ASR
- **特点**: 自回归编码器-解码器架构模型
- **模型文件**: `encoder.onnx`, `decoder.onnx`
- **模型类型**: `fire_red_asr`
- **适用场景**: 特定领域语音识别

### 8. Canary
- **特点**: 多语言语音识别和翻译模型
- **模型文件**: `encoder.onnx`, `decoder.onnx`
- **模型类型**: `canary`
- **适用场景**: 多语言识别、语音翻译
- **特殊功能**: 支持源语言和目标语言设置、标点符号控制（PNC）

### 9. CTC 模型系列

CTC（Connectionist Temporal Classification）模型使用连接时序分类，推理速度快，适合资源受限环境。

#### 9.1 NeMo CTC
- **特点**: NVIDIA NeMo 框架训练的 CTC 模型
- **模型文件**: 单个 `model.onnx` 文件
- **模型类型**: `nemo_ctc`
- **适用场景**: NeMo 生态模型使用、快速识别
- **下载地址**: https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-ctc/index.html

#### 9.2 TDNN
- **特点**: Time Delay Neural Network 架构的 CTC 模型
- **模型文件**: 单个 `model.onnx` 文件
- **模型类型**: `tdnn`
- **适用场景**: 传统声学模型、快速识别

#### 9.3 Zipformer CTC
- **特点**: 基于 Zipformer 架构的 CTC 模型
- **模型文件**: 单个 `model.onnx` 文件
- **模型类型**: `zipformer2_ctc`
- **适用场景**: 高准确率 CTC 识别、中英文混合
- **下载地址**: https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-ctc/index.html

#### 9.4 Wenet CTC
- **特点**: WeNet 框架训练的 CTC 模型
- **模型文件**: 单个 `model.onnx` 文件
- **模型类型**: `wenet_ctc`
- **适用场景**: 中文识别、WeNet 生态模型使用
- **下载地址**: https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-ctc/index.html

#### 9.5 TeleSpeech CTC
- **特点**: TeleSpeech 框架的 CTC 模型
- **模型文件**: 单个 `model.onnx` 文件
- **模型类型**: `telespeech_ctc`
- **适用场景**: 特定领域识别

#### 9.6 Omnilingual ASR CTC
- **特点**: 多语言 CTC 模型
- **模型文件**: 单个 `model.onnx` 文件
- **模型类型**: `omnilingual`
- **适用场景**: 多语言识别、资源受限环境

#### 9.7 Dolphin
- **特点**: Dolphin 架构的 CTC 模型
- **模型文件**: 单个 `model.onnx` 文件
- **模型类型**: `dolphin`
- **适用场景**: 快速识别、特定应用场景

### 模型选择建议

| 模型类型 | 推理速度 | 准确率 | 多语言支持 | 资源占用 | 推荐场景 |
|---------|---------|--------|-----------|---------|---------|
| Paraformer | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 中文为主 | 低 | 中文识别、批量处理 |
| Transducer | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 中 | 高准确率、多语言 |
| NeMo Transducer | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 中 | NeMo 生态、多语言 |
| Whisper | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 高 | 多语言、翻译 |
| SenseVoice | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 中 | 多语言、中文方言 |
| Moonshine | ⭐⭐⭐ | ⭐⭐⭐ | 视模型而定 | 中 | 原始音频输入 |
| FireRed ASR | ⭐⭐⭐ | ⭐⭐⭐⭐ | 视模型而定 | 中 | 特定领域识别 |
| Canary | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 中 | 多语言、翻译 |
| CTC 系列 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 视模型而定 | 低 | 快速识别、资源受限 |

**注意事项**:
- Paraformer 和 NeMo Transducer 模型支持文本置信度功能，可用于质量评估
- Transducer 和 NeMo Transducer 需要三个模型文件，占用内存较大
- Whisper 模型支持语言检测和翻译功能
- SenseVoice 支持自动语言检测和逆文本归一化
- CTC 模型推理速度快，但准确率通常低于 Transducer 和 Paraformer

## Paraformer 模型使用

### Python API

#### 基本使用

```python
import sherpa_onnx

# 创建识别器
recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
    paraformer="path/to/model.onnx",
    tokens="path/to/tokens.txt",
    num_threads=4,
    sample_rate=16000,
    feature_dim=80,
    decoding_method="greedy_search",
    debug=False,
    provider="cpu",  # 或 "cuda", "coreml"
)

# 读取音频文件
wave_data = sherpa_onnx.read_wave("audio.wav", expected_sample_rate=16000)

# 创建流并输入音频
stream = recognizer.create_stream()
stream.accept_waveform(
    sample_rate=wave_data.sample_rate,
    waveform=wave_data.samples,
)

# 执行识别
recognizer.decode(stream)

# 获取结果
result = recognizer.get_result(stream)
print(f"识别文本: {result.text}")
print(f"Token 列表: {result.tokens}")
print(f"时间戳: {result.timestamps}")
```

#### 带逆文本归一化 (ITN)

```python
recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
    paraformer="path/to/model.onnx",
    tokens="path/to/tokens.txt",
    rule_fsts="path/to/rule.fst",  # ITN 规则文件
    # 或多个规则文件，用逗号分隔
    # rule_fsts="rule1.fst,rule2.fst",
)
```

#### 批量处理

```python
# 创建多个流
streams = [recognizer.create_stream() for _ in range(batch_size)]

# 为每个流输入音频
for i, audio_file in enumerate(audio_files):
    wave_data = sherpa_onnx.read_wave(audio_file, expected_sample_rate=16000)
    streams[i].accept_waveform(
        sample_rate=wave_data.sample_rate,
        waveform=wave_data.samples,
    )

# 批量解码
stream_ptrs = [s.stream for s in streams]
recognizer.decode_streams(stream_ptrs)

# 获取所有结果
for i, stream in enumerate(streams):
    result = recognizer.get_result(stream)
    print(f"文件 {i}: {result.text}")
```

### Kotlin/Java API (Android)

```kotlin
import com.k2fsa.sherpa.onnx.*

// 从 Assets 加载模型
val config = OfflineRecognizerConfig(
    modelConfig = OfflineModelConfig(
        paraformer = OfflineParaformerModelConfig(
            model = "model.int8.onnx"
        ),
        tokens = "tokens.txt",
        modelType = "paraformer",
        numThreads = 4,
        provider = "cpu"
    )
)

val recognizer = OfflineRecognizer(assetManager, config)

// 读取音频
val samples = readAudioFile("audio.wav")  // FloatArray

// 创建流并识别
val stream = recognizer.createStream()
stream.acceptWaveform(sampleRate = 16000, waveform = samples)
recognizer.decode(stream)

// 获取结果
val result = recognizer.getResult(stream)
println("识别文本: ${result.text}")
println("Token 列表: ${result.tokens.joinToString()}")
println("时间戳: ${result.timestamps.joinToString()}")

// 释放资源
stream.release()
recognizer.release()
```

### C++ API

```cpp
#include "sherpa-onnx/csrc/offline-recognizer.h"

sherpa_onnx::OfflineRecognizerConfig config;
config.model_config.paraformer.model = "model.onnx";
config.model_config.tokens = "tokens.txt";
config.model_config.model_type = "paraformer";
config.model_config.num_threads = 4;

sherpa_onnx::OfflineRecognizer recognizer(config);

// 读取音频
auto wave = sherpa_onnx::ReadWave("audio.wav", 16000);

// 创建流并识别
auto stream = recognizer.CreateStream();
stream->AcceptWaveform(16000, wave.samples.data(), wave.samples.size());
recognizer.DecodeStream(stream.get());

// 获取结果
auto result = stream->GetResult();
std::cout << "识别文本: " << result.text << std::endl;
```

## 模型下载

### Paraformer 中文模型

请访问以下链接下载预训练模型：

- **官方模型列表**: https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-paraformer/index.html

常用模型示例：

```bash
# 下载中文 Paraformer 模型
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2
tar xvf sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2

# 模型文件结构
# sherpa-onnx-paraformer-zh-2023-09-14/
#   ├── model.int8.onnx      # 模型文件
#   ├── tokens.txt           # Token 词汇表
#   └── README.md            # 模型说明
```

## 配置参数说明

### Paraformer 配置

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `paraformer` | str | 模型文件路径 | 必填 |
| `tokens` | str | Token 词汇表路径 | 必填 |
| `num_threads` | int | 推理线程数 | 1 |
| `sample_rate` | int | 音频采样率 | 16000 |
| `feature_dim` | int | 特征维度 | 80 |
| `decoding_method` | str | 解码方法 | "greedy_search" |
| `debug` | bool | 是否显示调试信息 | False |
| `provider` | str | 执行提供者 (cpu/cuda/coreml) | "cpu" |
| `rule_fsts` | str | ITN 规则文件路径（逗号分隔） | "" |
| `rule_fars` | str | ITN 规则归档路径（逗号分隔） | "" |

## 文本置信度

Paraformer 和 NeMo Transducer 模型支持文本置信度功能，可以通过 `OfflineRecognitionResult.confidence` 字段获取识别结果的可信度评估。

详细说明请参考：[ASR 文本置信度实现文档](OFFLINE_ASR_CONFIDENCE.md)

## 性能优化

### 1. 批量处理

```python
# 批量处理多个音频文件
streams = [recognizer.create_stream() for _ in range(batch_size)]

# 输入所有音频
for i, audio_file in enumerate(audio_files):
    wave = sherpa_onnx.read_wave(audio_file, 16000)
    streams[i].accept_waveform(16000, wave.samples)

# 批量解码（比逐个解码快）
stream_ptrs = [s.stream for s in streams]
recognizer.decode_streams(stream_ptrs)
```

### 2. 使用 GPU

```python
recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
    paraformer="model.onnx",
    tokens="tokens.txt",
    provider="cuda",  # 使用 GPU
)
```

### 3. 量化模型

使用 INT8 量化模型可以显著减少内存占用和加速推理：

```python
# 使用量化模型（通常文件名包含 int8）
recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
    paraformer="model.int8.onnx",  # INT8 量化模型
    tokens="tokens.txt",
)
```

## 常见问题

### 1. 识别结果为空

**可能原因**:
- 音频采样率不匹配
- 音频太短或没有语音
- 模型文件路径错误

**解决方案**:
```python
# 检查音频信息
wave = sherpa_onnx.read_wave("audio.wav", 16000)
print(f"采样率: {wave.sample_rate}, 长度: {len(wave.samples)}")

# 确保采样率正确
if wave.sample_rate != 16000:
    # 需要重采样
    pass
```

### 2. 识别准确率低

**可能原因**:
- 音频质量差（噪声、低采样率）
- 模型不匹配（语言、领域）
- 音频格式问题

**解决方案**:
- 使用音频预处理（降噪、归一化）
- 选择匹配的模型
- 检查音频格式（推荐 WAV，16kHz，16-bit PCM）

### 3. 内存占用高

**解决方案**:
- 使用 INT8 量化模型
- 减少批量大小
- 使用 CPU 而非 GPU（如果 GPU 内存不足）

## 示例代码

### 完整示例：带置信度的识别

```python
import sherpa_onnx
import numpy as np

# 创建识别器
recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
    paraformer="model.int8.onnx",
    tokens="tokens.txt",
    num_threads=4,
)

# 读取音频
wave = sherpa_onnx.read_wave("audio.wav", expected_sample_rate=16000)

# 识别
stream = recognizer.create_stream()
stream.accept_waveform(wave.sample_rate, wave.samples)
recognizer.decode(stream)

# 获取结果
result = recognizer.get_result(stream)
print(f"识别文本: {result.text}")

# 获取置信度（Paraformer 模型支持）
if result.confidence >= 0.0:
    print(f"置信度: {result.confidence:.4f}")
    
    # 判断置信度等级
    if result.confidence >= 0.8:
        print("高置信度识别结果")
    elif result.confidence >= 0.5:
        print("中等置信度识别结果")
    else:
        print("低置信度识别结果，建议人工复核")
else:
    print("注意: 当前模型不支持置信度（仅 Paraformer 模型支持）")
```

## 相关资源

- **模型下载**: https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
- **Python API 文档**: `python-api-examples/`
- **C++ API 文档**: `sherpa-onnx/csrc/offline-recognizer.h`
- **Kotlin API 文档**: `sherpa-onnx/kotlin-api/OfflineRecognizer.kt`

## 总结

1. **Paraformer 模型**是中文离线识别的优秀选择，速度快、准确率高
2. **文本置信度功能**：Paraformer 和 NeMo Transducer 模型支持 `confidence` 字段，详细说明请参考 [ASR 文本置信度实现文档](OFFLINE_ASR_CONFIDENCE.md)
3. **批量处理**和**量化模型**可以显著提升性能

