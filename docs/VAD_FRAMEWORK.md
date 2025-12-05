# VAD (Voice Activity Detection) 框架说明文档

## 目录

- [概述](#概述)
- [VAD框架架构](#vad框架架构)
- [支持的VAD模型](#支持的vad模型)
- [配置参数](#配置参数)
- [核心API](#核心api)
- [使用示例](#使用示例)
- [模型对比](#模型对比)
- [最佳实践](#最佳实践)
- [应用场景](#应用场景)

## 概述

VAD（Voice Activity Detection，语音活动检测）是 sherpa-onnx 项目中用于检测音频中语音片段的核心组件。该框架支持多种VAD模型，能够实时或离线地检测音频中的语音活动，为ASR、说话人识别等下游任务提供语音片段分割。

### 主要功能

- ✅ **实时语音检测**：支持流式音频输入，实时检测语音活动
- ✅ **离线语音检测**：支持完整音频文件的语音片段检测
- ✅ **多模型支持**：支持 Silero VAD 和 Ten VAD 两种模型
- ✅ **自动分段**：自动将连续语音分割为独立的语音片段
- ✅ **可配置参数**：支持阈值、最小/最大语音时长等参数调整
- ✅ **跨平台支持**：提供 C++、Python、Java、Kotlin、Go 等多种语言接口

### 核心特性

- **滑动窗口检测**：使用滑动窗口机制，逐帧检测语音活动
- **状态管理**：自动管理语音开始、持续、结束状态
- **缓冲区管理**：使用循环缓冲区存储音频数据，支持长音频处理
- **动态阈值调整**：当检测到超长语音时，自动提高阈值以强制分段

## VAD框架架构

### 整体架构

```
VoiceActivityDetector (公共接口)
    └── Impl (内部实现)
        ├── VadModel (抽象基类)
        │   ├── SileroVadModel (Silero VAD实现)
        │   └── TenVadModel (Ten VAD实现)
        ├── CircularBuffer (循环缓冲区)
        └── SpeechSegment (语音片段队列)
```

### 设计模式

- **策略模式**：通过 `VadModel` 抽象基类，支持不同的VAD模型实现
- **工厂模式**：`VadModel::Create()` 根据配置自动创建对应的模型实例
- **PIMPL模式**：隐藏实现细节，提供稳定的ABI

### 工作流程

```
音频输入
  │
  ├─> AcceptWaveform() 接收音频数据
  │     │
  │     ├─> 存储到循环缓冲区
  │     │
  │     ├─> 滑动窗口处理
  │     │     └─> 每个窗口调用 IsSpeech() 检测
  │     │
  │     ├─> 状态管理
  │     │     ├─> 语音开始：记录起始位置
  │     │     ├─> 语音持续：更新缓冲区
  │     │     └─> 语音结束：生成语音片段
  │     │
  │     └─> 将完整片段加入队列
  │
  └─> Front() / Pop() 获取语音片段
```

### 核心数据结构

```cpp
struct SpeechSegment {
  int32_t start;        // 起始位置（样本数）
  std::vector<float> samples;  // 音频样本
};

class VoiceActivityDetector {
  // 接收音频数据
  void AcceptWaveform(const float *samples, int32_t n);
  
  // 计算当前窗口的语音概率
  float Compute(const float *samples, int32_t n);
  
  // 检查是否有语音片段
  bool Empty() const;
  
  // 获取并移除第一个语音片段
  void Pop();
  
  // 获取第一个语音片段（不移除）
  const SpeechSegment &Front() const;
  
  // 重置状态
  void Reset();
  
  // 刷新缓冲区，处理最后一个片段
  void Flush();
};
```

## 支持的VAD模型

### 1. Silero VAD

**Silero VAD** 是由 Silero 团队开发的高性能VAD模型，基于深度学习，在多种场景下表现优秀。

#### 特点

- **高准确率**：在多种语言和场景下表现稳定
- **低延迟**：支持实时处理
- **多版本支持**：支持 V4 和 V5 版本
- **窗口大小**：512 样本（16kHz采样率下约32ms）

#### 模型下载

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
```

#### 配置参数

```cpp
struct SileroVadModelConfig {
  std::string model;                    // 模型文件路径
  float threshold = 0.5;                // 语音检测阈值 [0, 1]
  float min_silence_duration = 0.5;      // 最小静音时长（秒）
  float min_speech_duration = 0.25;      // 最小语音时长（秒）
  int32_t window_size = 512;            // 窗口大小（样本数）
  float max_speech_duration = 20;        // 最大语音时长（秒）
};
```

#### 窗口大小说明

- **V4版本**：`window_size = window_shift = 512`
- **V5版本**：`window_size = window_shift + 64`（16kHz）或 `window_shift + 32`（8kHz）
- **默认值**：512 样本（16kHz采样率下约32ms）

### 2. Ten VAD

**Ten VAD** 是另一个高性能VAD模型，在某些场景下可能比 Silero VAD 表现更好。

#### 特点

- **轻量级**：模型相对较小
- **快速推理**：推理速度较快
- **窗口大小**：256 或 160 样本（16kHz采样率下约16ms或10ms）

#### 模型下载

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/ten-vad.onnx
```

#### 配置参数

```cpp
struct TenVadModelConfig {
  std::string model;                     // 模型文件路径
  float threshold = 0.5;                // 语音检测阈值 [0, 1]
  float min_silence_duration = 0.5;     // 最小静音时长（秒）
  float min_speech_duration = 0.25;      // 最小语音时长（秒）
  int32_t window_size = 256;            // 窗口大小（样本数）：160 或 256
  float max_speech_duration = 20;       // 最大语音时长（秒）
};
```

#### 窗口大小说明

- **支持值**：160 或 256 样本
- **默认值**：256 样本（16kHz采样率下约16ms）
- **窗口移位**：256 或 128 样本

## 配置参数

### VadModelConfig

```cpp
struct VadModelConfig {
  SileroVadModelConfig silero_vad;      // Silero VAD 配置
  TenVadModelConfig ten_vad;            // Ten VAD 配置
  
  int32_t sample_rate = 16000;          // 采样率（Hz）
  int32_t num_threads = 1;               // 线程数
  std::string provider = "cpu";          // 执行提供者：cpu, cuda, coreml, rknn
  bool debug = false;                   // 是否显示调试信息
};
```

### 参数说明

#### 通用参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `sample_rate` | int32_t | 16000 | 音频采样率，通常为16000Hz |
| `num_threads` | int32_t | 1 | 推理线程数，多线程可加速处理 |
| `provider` | string | "cpu" | 执行提供者：cpu, cuda, coreml, rknn |
| `debug` | bool | false | 是否显示模型加载和推理的调试信息 |

#### Silero VAD 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | string | "" | 模型文件路径（必需） |
| `threshold` | float | 0.5 | 语音检测阈值 [0, 1]，越大越严格 |
| `min_silence_duration` | float | 0.5 | 最小静音时长（秒），用于判断语音结束 |
| `min_speech_duration` | float | 0.25 | 最小语音时长（秒），短于此长度的片段会被过滤 |
| `window_size` | int32_t | 512 | 窗口大小（样本数），必须为512 |
| `max_speech_duration` | float | 20 | 最大语音时长（秒），超过此值会强制分段 |

#### Ten VAD 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | string | "" | 模型文件路径（必需） |
| `threshold` | float | 0.5 | 语音检测阈值 [0, 1]，越大越严格 |
| `min_silence_duration` | float | 0.5 | 最小静音时长（秒），用于判断语音结束 |
| `min_speech_duration` | float | 0.25 | 最小语音时长（秒），短于此长度的片段会被过滤 |
| `window_size` | int32_t | 256 | 窗口大小（样本数），支持160或256 |
| `max_speech_duration` | float | 20 | 最大语音时长（秒），超过此值会强制分段 |

### 参数选择建议

#### threshold（阈值）

- **严格场景**（如会议记录）：0.6 - 0.8
- **一般场景**：0.4 - 0.6
- **宽松场景**（如嘈杂环境）：0.3 - 0.5
- **注意**：阈值越高，漏检越多；阈值越低，误检越多

#### min_silence_duration（最小静音时长）

- **快速响应**：0.25 - 0.5 秒
- **一般场景**：0.5 - 1.0 秒
- **稳定场景**：1.0 - 2.0 秒
- **注意**：值越大，语音片段合并越多；值越小，片段分割越细

#### min_speech_duration（最小语音时长）

- **过滤短片段**：0.25 - 0.5 秒
- **一般场景**：0.1 - 0.25 秒
- **注意**：短于此长度的片段会被丢弃，用于过滤噪声

#### max_speech_duration（最大语音时长）

- **长语音场景**：20 - 60 秒
- **一般场景**：10 - 20 秒
- **短语音场景**：5 - 10 秒
- **注意**：超过此值时，系统会自动提高阈值到0.9以强制分段

## 核心API

### C++ API

```cpp
class VoiceActivityDetector {
  // 构造函数
  explicit VoiceActivityDetector(
      const VadModelConfig &config,
      float buffer_size_in_seconds = 60);

  // 接收音频数据
  void AcceptWaveform(const float *samples, int32_t n);

  // 计算当前窗口的语音概率
  float Compute(const float *samples, int32_t n);

  // 检查是否有语音片段
  bool Empty() const;

  // 获取并移除第一个语音片段
  void Pop();

  // 获取第一个语音片段（不移除）
  const SpeechSegment &Front() const;

  // 清空所有片段
  void Clear();

  // 检查当前是否检测到语音
  bool IsSpeechDetected() const;

  // 获取当前正在检测的语音片段
  SpeechSegment CurrentSpeechSegment() const;

  // 重置状态
  void Reset();

  // 刷新缓冲区，处理最后一个片段
  void Flush();
};
```

### Python API

```python
import sherpa_onnx

# 创建配置
config = sherpa_onnx.VadModelConfig()

# 配置 Silero VAD
config.silero_vad.model = "./silero_vad.onnx"
config.silero_vad.threshold = 0.5
config.silero_vad.min_silence_duration = 0.5
config.silero_vad.min_speech_duration = 0.25
config.silero_vad.window_size = 512
config.silero_vad.max_speech_duration = 20

# 或配置 Ten VAD
config.ten_vad.model = "./ten-vad.onnx"
config.ten_vad.threshold = 0.5
config.ten_vad.min_silence_duration = 0.5
config.ten_vad.min_speech_duration = 0.25
config.ten_vad.window_size = 256
config.ten_vad.max_speech_duration = 20

# 通用配置
config.sample_rate = 16000
config.num_threads = 1
config.provider = "cpu"
config.debug = False

# 创建VAD检测器
vad = sherpa_onnx.VoiceActivityDetector(
    config, 
    buffer_size_in_seconds=60
)

# 接收音频数据
vad.accept_waveform(samples)

# 检查是否有语音片段
while not vad.empty():
    segment = vad.front()
    print(f"语音片段：起始位置={segment.start}, 长度={len(segment.samples)}")
    vad.pop()

# 刷新缓冲区（处理最后一个片段）
vad.flush()

# 重置状态
vad.reset()
```

### Java API

```java
import com.k2fsa.sherpa.onnx.*;

// 创建配置
SileroVadModelConfig sileroConfig = SileroVadModelConfig.builder()
    .setModel("./silero_vad.onnx")
    .setThreshold(0.5f)
    .setMinSilenceDuration(0.5f)
    .setMinSpeechDuration(0.25f)
    .setWindowSize(512)
    .setMaxSpeechDuration(20.0f)
    .build();

VadModelConfig config = VadModelConfig.builder()
    .setSileroVadModelConfig(sileroConfig)
    .setSampleRate(16000)
    .setNumThreads(1)
    .setProvider("cpu")
    .setDebug(false)
    .build();

// 创建VAD检测器
Vad vad = new Vad(config);

// 接收音频数据
vad.acceptWaveform(samples);

// 检查并获取语音片段
while (!vad.empty()) {
    SpeechSegment segment = vad.front();
    System.out.println("语音片段：起始=" + segment.start + 
                      ", 长度=" + segment.samples.length);
    vad.pop();
}
```

## 使用示例

### 示例1：实时麦克风VAD检测

```python
import sherpa_onnx
import sounddevice as sd
import numpy as np

# 配置VAD
config = sherpa_onnx.VadModelConfig()
config.silero_vad.model = "./silero_vad.onnx"
config.silero_vad.threshold = 0.5
config.silero_vad.min_silence_duration = 0.5
config.silero_vad.min_speech_duration = 0.25
config.sample_rate = 16000

vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=30)

# 从麦克风读取音频
sample_rate = 16000
samples_per_read = int(0.1 * sample_rate)  # 100ms

with sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate) as s:
    while True:
        samples, _ = s.read(samples_per_read)
        samples = samples.reshape(-1)
        
        # 输入音频到VAD
        vad.accept_waveform(samples)
        
        # 检查是否有完整的语音片段
        while not vad.empty():
            segment = vad.front()
            print(f"检测到语音片段：长度={len(segment.samples)/sample_rate:.2f}秒")
            vad.pop()
```

### 示例2：离线音频文件VAD检测

```python
import sherpa_onnx
import soundfile as sf
import numpy as np

# 加载音频文件
audio, sample_rate = sf.read("audio.wav")
if len(audio.shape) > 1:
    audio = audio[:, 0]  # 转为单声道

# 重采样到16kHz（如果需要）
if sample_rate != 16000:
    import librosa
    audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
    sample_rate = 16000

# 配置VAD
config = sherpa_onnx.VadModelConfig()
config.silero_vad.model = "./silero_vad.onnx"
config.silero_vad.threshold = 0.5
config.silero_vad.min_silence_duration = 0.5
config.silero_vad.min_speech_duration = 0.25
config.sample_rate = sample_rate

vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=3600)

# 分块处理音频
window_size = config.silero_vad.window_size
for i in range(0, len(audio), window_size):
    chunk = audio[i:i+window_size]
    if len(chunk) < window_size:
        break
    vad.accept_waveform(chunk)

# 刷新缓冲区，处理最后一个片段
vad.flush()

# 获取所有语音片段
segments = []
while not vad.empty():
    segment = vad.front()
    segments.append(segment)
    vad.pop()

print(f"检测到 {len(segments)} 个语音片段")
for i, seg in enumerate(segments):
    print(f"片段 {i+1}: 起始={seg.start/sample_rate:.2f}秒, "
          f"长度={len(seg.samples)/sample_rate:.2f}秒")
```

### 示例3：VAD + ASR 实时识别

```python
import sherpa_onnx
import sounddevice as sd

# 1. 配置VAD
vad_config = sherpa_onnx.VadModelConfig()
vad_config.silero_vad.model = "./silero_vad.onnx"
vad_config.silero_vad.threshold = 0.5
vad_config.silero_vad.min_silence_duration = 0.5
vad_config.sample_rate = 16000

vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=30)

# 2. 配置ASR
asr_config = sherpa_onnx.OnlineRecognizerConfig()
# ... 配置ASR模型 ...
recognizer = sherpa_onnx.OnlineRecognizer(asr_config)

# 3. 实时处理
sample_rate = 16000
samples_per_read = int(0.1 * sample_rate)

with sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate) as s:
    while True:
        samples, _ = s.read(samples_per_read)
        samples = samples.reshape(-1)
        
        vad.accept_waveform(samples)
        
        # 处理检测到的语音片段
        while not vad.empty():
            segment = vad.front()
            
            # 对语音片段进行ASR识别
            stream = recognizer.create_stream()
            stream.accept_waveform(sample_rate, segment.samples)
            recognizer.decode_stream(stream)
            text = stream.result.text
            
            print(f"识别结果: {text}")
            
            vad.pop()
```

### 示例4：使用 Ten VAD

```python
import sherpa_onnx

# 配置 Ten VAD
config = sherpa_onnx.VadModelConfig()
config.ten_vad.model = "./ten-vad.onnx"
config.ten_vad.threshold = 0.5
config.ten_vad.min_silence_duration = 0.5
config.ten_vad.min_speech_duration = 0.25
config.ten_vad.window_size = 256  # 或 160
config.ten_vad.max_speech_duration = 20
config.sample_rate = 16000

vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=60)

# 使用方式与 Silero VAD 相同
# ...
```

## 模型对比

### Silero VAD vs Ten VAD

| 特性 | Silero VAD | Ten VAD |
|------|------------|---------|
| **窗口大小** | 512 样本（32ms @ 16kHz） | 256/160 样本（16ms/10ms @ 16kHz） |
| **窗口移位** | 512 样本 | 256/128 样本 |
| **延迟** | 较高（32ms） | 较低（16ms或10ms） |
| **准确率** | 高 | 高 |
| **模型大小** | 中等 | 较小 |
| **适用场景** | 一般场景 | 低延迟场景 |
| **推荐用途** | 离线处理、会议记录 | 实时交互、语音助手 |

### 选择建议

- **选择 Silero VAD**：
  - 需要高准确率
  - 离线处理场景
  - 对延迟不敏感

- **选择 Ten VAD**：
  - 需要低延迟
  - 实时交互场景
  - 资源受限环境

## 最佳实践

### 1. 参数调优

```python
# 根据场景调整阈值
# 嘈杂环境：降低阈值
config.silero_vad.threshold = 0.3

# 安静环境：提高阈值
config.silero_vad.threshold = 0.7

# 快速响应：减小最小静音时长
config.silero_vad.min_silence_duration = 0.25

# 稳定检测：增大最小静音时长
config.silero_vad.min_silence_duration = 1.0
```

### 2. 缓冲区大小

```python
# 短音频（< 1分钟）
vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=60)

# 中等音频（1-10分钟）
vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=600)

# 长音频（> 10分钟）
vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=3600)
```

### 3. 音频预处理

```python
# 确保单声道
if len(audio.shape) > 1:
    audio = audio[:, 0]

# 确保采样率为16kHz
if sample_rate != 16000:
    import librosa
    audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

# 归一化到[-1, 1]
audio = audio.astype(np.float32)
if audio.max() > 1.0 or audio.min() < -1.0:
    audio = audio / max(abs(audio.max()), abs(audio.min()))
```

### 4. 错误处理

```python
# 验证配置
if not config.validate():
    raise ValueError("Invalid VAD config")

# 检查模型文件
import os
if not os.path.exists(config.silero_vad.model):
    raise FileNotFoundError(f"Model not found: {config.silero_vad.model}")

# 检查音频片段
while not vad.empty():
    segment = vad.front()
    if len(segment.samples) < 0.5 * sample_rate:  # 过滤过短片段
        vad.pop()
        continue
    # 处理片段
    vad.pop()
```

### 5. 性能优化

```python
# 使用多线程加速
config.num_threads = 4

# 使用GPU加速（如果支持）
config.provider = "cuda"

# 批量处理音频
window_size = config.silero_vad.window_size
for i in range(0, len(audio), window_size * 10):  # 每次处理10个窗口
    chunk = audio[i:i+window_size*10]
    vad.accept_waveform(chunk)
```

## 应用场景

### 1. 实时语音识别

**场景**：实时从麦克风捕获语音并进行识别

**流程**：
1. VAD检测语音片段
2. 对每个片段进行ASR识别
3. 输出识别结果

**优势**：
- 自动分割语音片段
- 减少无效音频处理
- 提高识别效率

### 2. 会议记录

**场景**：自动记录会议中的语音内容

**流程**：
1. VAD检测语音片段
2. 对每个片段进行ASR识别
3. 结合说话人识别，标注发言人
4. 生成会议记录

**配置建议**：
- 使用较高的阈值（0.6-0.7）
- 增大最小静音时长（1.0-2.0秒）
- 增大最大语音时长（30-60秒）

### 3. 音频预处理

**场景**：从长音频中提取语音片段

**流程**：
1. 加载完整音频文件
2. VAD检测所有语音片段
3. 保存或处理每个片段

**优势**：
- 自动去除静音部分
- 分割为独立语音片段
- 便于后续处理

### 4. 说话人分离辅助

**场景**：结合说话人识别，进行说话人分离

**流程**：
1. VAD检测语音片段
2. 对每个片段提取声纹嵌入
3. 使用聚类算法分离说话人

**优势**：
- 先分割再识别，提高效率
- 减少无效片段处理

### 5. 语音助手

**场景**：智能音箱、语音助手等实时交互场景

**流程**：
1. 实时VAD检测
2. 检测到语音后唤醒ASR
3. 识别用户指令

**配置建议**：
- 使用 Ten VAD（低延迟）
- 降低阈值（0.3-0.4）
- 减小最小静音时长（0.25-0.5秒）

## 相关资源

### 代码文件

- **C++核心实现**：
  - `sherpa-onnx/csrc/voice-activity-detector.h`、`.cc`
  - `sherpa-onnx/csrc/vad-model.h`、`.cc`
  - `sherpa-onnx/csrc/silero-vad-model.h`、`.cc`
  - `sherpa-onnx/csrc/ten-vad-model.h`、`.cc`
  - `sherpa-onnx/csrc/vad-model-config.h`、`.cc`

- **Python绑定**：
  - `sherpa-onnx/python/csrc/voice-activity-detector.cc`
  - `sherpa-onnx/python/csrc/vad-model-config.cc`

### 使用示例

- **实时VAD**：`python-api-examples/vad-microphone.py`
- **离线VAD**：`python-api-examples/vad-remove-non-speech-segments.py`
- **VAD + ASR**：`python-api-examples/vad-with-non-streaming-asr.py`
- **VAD + 说话人识别**：`python-api-examples/speaker-identification-with-vad-non-streaming-asr.py`

### 模型下载

- **Silero VAD**：
  ```bash
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
  ```

- **Ten VAD**：
  ```bash
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/ten-vad.onnx
  ```

### 相关组件

- **ASR (Automatic Speech Recognition)**：语音识别
- **Speaker Embedding**：说话人嵌入提取
- **FastClustering**：说话人聚类

## 总结

VAD框架是 sherpa-onnx 项目中用于语音活动检测的核心组件，支持 Silero VAD 和 Ten VAD 两种模型。通过灵活的配置参数和丰富的API接口，可以满足实时和离线场景下的语音检测需求。合理选择模型和参数，结合下游任务（ASR、说话人识别等），可以构建高效的语音处理系统。

---

**文档版本**：1.0  
**最后更新**：2024年  
**维护者**：sherpa-onnx团队

