# K2 Speaker Diarization 功能说明文档

## 目录

- [概述](#概述)
- [功能原理](#功能原理)
- [实现架构](#实现架构)
- [核心代码位置](#核心代码位置)
- [配置参数](#配置参数)
- [使用示例](#使用示例)
- [与 VAD 分割音频片段的区别](#与-vad-分割音频片段的区别)
- [应用场景](#应用场景)
- [最佳实践](#最佳实践)

## 概述

K2 Speaker Diarization（说话人分离）是 sherpa-onnx 项目中用于自动识别和分离音频中不同说话人的核心功能。该功能能够自动检测音频中每个时间段对应的说话人，为多人对话场景（如会议录音、电话录音、播客等）提供说话人级别的音频分析。

### 主要特性

- ✅ **端到端处理**：从原始音频直接输出带说话人标签的时间段
- ✅ **三阶段处理**：分割（Segmentation）→ 嵌入提取（Embedding）→ 聚类（Clustering）
- ✅ **无需预设说话人数量**：支持基于阈值的自适应聚类
- ✅ **高质量分割**：基于 Pyannote 分割模型，支持重叠说话人检测
- ✅ **跨平台支持**：提供 C++、Python、Java、Kotlin、Go、C#、JavaScript 等多种语言接口

### 核心组件

Speaker Diarization 由三个核心组件组成：

1. **分割模型（Segmentation Model）**：基于 Pyannote 的说话人分割模型，检测音频中每个时间段的说话人活动
2. **嵌入提取器（Embedding Extractor）**：提取每个语音片段的声纹嵌入向量
3. **聚类算法（Clustering）**：使用 K2 FastClustering 算法将相似声纹聚类，识别同一说话人

## 功能原理

### 三阶段处理流程

Speaker Diarization 采用三阶段处理流程：

```
原始音频
  │
  ├─> 阶段1：说话人分割（Segmentation）
  │     └─> 使用 Pyannote 模型检测每个时间段的说话人活动
  │          输出：每个 chunk 的说话人标签矩阵
  │
  ├─> 阶段2：声纹嵌入提取（Embedding Extraction）
  │     └─> 为每个检测到的说话人片段提取声纹嵌入向量
  │          输出：声纹嵌入矩阵 (N x embedding_dim)
  │
  └─> 阶段3：聚类（Clustering）
        └─> 使用 FastClustering 将相似声纹聚类
            输出：每个片段对应的说话人ID
```

### 详细工作流程

#### 阶段1：说话人分割

使用 Pyannote 分割模型对音频进行分块处理：

```94:109:sherpa-onnx/csrc/offline-speaker-diarization-pyannote-impl.h
  OfflineSpeakerDiarizationResult Process(
      const float *audio, int32_t n,
      OfflineSpeakerDiarizationProgressCallback callback = nullptr,
      void *callback_arg = nullptr) const override {
    std::vector<Matrix2D> segmentations = RunSpeakerSegmentationModel(audio, n);
    // segmentations[i] is for chunk_i
    // Each matrix is of shape (num_frames, num_powerset_classes)
    if (segmentations.empty()) {
      return {};
    }

    std::vector<Matrix2DInt32> labels;
    labels.reserve(segmentations.size());

    for (const auto &m : segmentations) {
      labels.
```

**关键步骤**：
1. 将音频分割为固定大小的窗口（window_size）
2. 使用滑动窗口（window_shift）处理长音频
3. 对每个窗口运行分割模型，得到 powerset 类别概率
4. 将 powerset 类别转换为多标签说话人标签

**Powerset 编码**：
- 模型输出的是 powerset 类别（如：无说话人、说话人1、说话人2、说话人1+说话人2等）
- 通过 powerset_mapping 转换为多标签格式，支持同时检测多个说话人

#### 阶段2：声纹嵌入提取

为每个检测到的说话人片段提取声纹嵌入：

```466:521:sherpa-onnx/csrc/offline-speaker-diarization-pyannote-impl.h
  Matrix2D ComputeEmbeddings(
      const float *audio, int32_t n,
      const std::vector<std::vector<Int32Pair>> &sample_indexes,
      std::vector<int32_t> *valid_indexes,
      OfflineSpeakerDiarizationProgressCallback callback,
      void *callback_arg) const {
    const auto &meta_data = segmentation_model_.GetModelMetaData();
    int32_t sample_rate = meta_data.sample_rate;
    Matrix2D ans(sample_indexes.size(), embedding_extractor_.Dim());

    auto IsNaNWrapper = [](float f) -> bool { return std::isnan(f); };

    int32_t k = 0;
    int32_t cur_row_index = 0;
    for (const auto &v : sample_indexes) {
      auto stream = embedding_extractor_.CreateStream();
      for (const auto &p : v) {
        int32_t end = (p.second <= n) ? p.second : n;
        int32_t num_samples = end - p.first;

        if (num_samples > 0) {
          stream->AcceptWaveform(sample_rate, audio + p.first, num_samples);
        }
      }

      stream->InputFinished();
      if (!embedding_extractor_.IsReady(stream.get())) {
        SHERPA_ONNX_LOGE(
            "This segment is too short, which should not happen since we have "
            "already filtered short segments");
        SHERPA_ONNX_EXIT(-1);
      }

      std::vector<float> embedding = embedding_extractor_.Compute(stream.get());

      if (std::none_of(embedding.begin(), embedding.end(), IsNaNWrapper)) {
        // a valid embedding
        std::copy(embedding.begin(), embedding.end(), &ans(cur_row_index, 0));
        cur_row_index += 1;
        valid_indexes->push_back(k);
      }

      k += 1;

      if (callback) {
        callback(k, ans.rows(), callback_arg);
      }
    }

    if (k != cur_row_index) {
      auto seq = Eigen::seqN(0, cur_row_index);
      ans = ans(seq, Eigen::all);
    }

    return ans;
  }
```

**关键步骤**：
1. 从分割结果中提取每个说话人片段的样本索引
2. 排除重叠区域（ExcludeOverlap），只处理单一说话人片段
3. 为每个片段创建嵌入提取流，提取声纹嵌入向量
4. 过滤无效嵌入（NaN 值）

#### 阶段3：聚类

使用 FastClustering 算法将相似声纹聚类：

```160:166:sherpa-onnx/csrc/offline-speaker-diarization-pyannote-impl.h
    std::vector<int32_t> cluster_labels = clustering_->Cluster(
        &embeddings(0, 0), embeddings.rows(), embeddings.cols());

    if (cluster_labels.empty()) {
      SHERPA_ONNX_LOGE("No speakers found in the audio samples");
      return {};
    }
```

**聚类过程**：
1. 将所有声纹嵌入向量进行 L2 归一化
2. 计算余弦不相似度矩阵
3. 使用层次聚类（Complete Linkage）构建聚类树
4. 根据配置（固定聚类数或阈值）切割聚类树
5. 为每个片段分配说话人ID

**结果后处理**：
1. 将聚类结果映射回原始标签
2. 合并相邻的同一说话人片段（根据 min_duration_off）
3. 过滤过短片段（根据 min_duration_on）
4. 生成最终的时间段结果

## 实现架构

### 整体架构

```
OfflineSpeakerDiarization (公共接口)
    └── OfflineSpeakerDiarizationImpl (抽象基类)
        └── OfflineSpeakerDiarizationPyannoteImpl (Pyannote实现)
            ├── OfflineSpeakerSegmentationPyannoteModel (分割模型)
            ├── SpeakerEmbeddingExtractor (嵌入提取器)
            └── FastClustering (聚类算法)
```

### 核心数据结构

#### OfflineSpeakerDiarizationSegment

```15:41:sherpa-onnx/csrc/offline-speaker-diarization-result.h
class OfflineSpeakerDiarizationSegment {
 public:
  OfflineSpeakerDiarizationSegment(float start, float end, int32_t speaker,
                                   const std::string &text = {});

  // If the gap between the two segments is less than the given gap, then we
  // merge them and return a new segment. Otherwise, it returns null.
  std::optional<OfflineSpeakerDiarizationSegment> Merge(
      const OfflineSpeakerDiarizationSegment &other, float gap) const;

  float Start() const { return start_; }
  float End() const { return end_; }
  int32_t Speaker() const { return speaker_; }
  const std::string &Text() const { return text_; }
  float Duration() const { return end_ - start_; }

  void SetText(const std::string &text) { text_ = text; }

  std::string ToString() const;

 private:
  float start_;       // in seconds
  float end_;         // in seconds
  int32_t speaker_;   // ID of the speaker, starting from 0
  std::string text_;  // If not empty, it contains the speech recognition result
                      // of this segment
};
```

#### OfflineSpeakerDiarizationConfig

```21:27:sherpa-onnx/kotlin-api/OfflineSpeakerDiarization.kt
data class OfflineSpeakerDiarizationConfig(
    var segmentation: OfflineSpeakerSegmentationModelConfig = OfflineSpeakerSegmentationModelConfig(),
    var embedding: SpeakerEmbeddingExtractorConfig = SpeakerEmbeddingExtractorConfig(),
    var clustering: FastClusteringConfig = FastClusteringConfig(),
    var minDurationOn: Float = 0.2f,
    var minDurationOff: Float = 0.5f,
)
```

## 核心代码位置

### C++ 核心实现

| 文件 | 说明 |
|------|------|
| `sherpa-onnx/csrc/offline-speaker-diarization.h` | 公共接口定义 |
| `sherpa-onnx/csrc/offline-speaker-diarization.cc` | 接口实现 |
| `sherpa-onnx/csrc/offline-speaker-diarization-impl.h` | 抽象基类 |
| `sherpa-onnx/csrc/offline-speaker-diarization-pyannote-impl.h` | Pyannote 实现（核心算法） |
| `sherpa-onnx/csrc/offline-speaker-diarization-result.h` | 结果数据结构 |
| `sherpa-onnx/csrc/offline-speaker-segmentation-pyannote-model.h` | 分割模型封装 |
| `sherpa-onnx/csrc/fast-clustering.h` | 聚类算法（参考 K2_SPEAKER_CLUSTERING.md） |

### Python 绑定

| 文件 | 说明 |
|------|------|
| `sherpa-onnx/python/csrc/offline-speaker-diarization.cc` | Python 绑定实现 |
| `python-api-examples/offline-speaker-diarization.py` | Python 使用示例 |

### Kotlin/Java 绑定

| 文件 | 说明 |
|------|------|
| `sherpa-onnx/kotlin-api/OfflineSpeakerDiarization.kt` | Kotlin API 定义 |
| `android/SherpaOnnxSpeakerDiarization/app/src/main/java/com/k2fsa/sherpa/onnx/speaker/diarization/SpeakerDiarizationObject.kt` | Android 应用示例 |

### 其他语言绑定

- **Go**: `go-api-examples/non-streaming-speaker-diarization/main.go`
- **Swift**: `swift-api-examples/speaker-diarization.swift`
- **Dart**: `dart-api-examples/speaker-diarization/bin/speaker-diarization.dart`
- **HarmonyOS**: `harmony-os/SherpaOnnxSpeakerDiarization/entry/src/main/ets/workers/SpeakerDiarizationWorker.ets`

## 配置参数

### OfflineSpeakerDiarizationConfig

#### segmentation（分割模型配置）

```9:14:sherpa-onnx/kotlin-api/OfflineSpeakerDiarization.kt
data class OfflineSpeakerSegmentationModelConfig(
    var pyannote: OfflineSpeakerSegmentationPyannoteModelConfig = OfflineSpeakerSegmentationPyannoteModelConfig(),
    var numThreads: Int = 1,
    var debug: Boolean = false,
    var provider: String = "cpu",
)
```

##### 配置参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| **pyannote.model** | string | "" | **必需**。分割模型文件路径（如 `model.onnx`）。模型文件通常位于下载的压缩包中，解压后找到 `model.onnx` 文件 |
| **numThreads** | int | 1 | 推理线程数。多线程可加速处理，建议设置为 CPU 核心数。例如：4核CPU可设置为4 |
| **debug** | bool | false | 是否启用调试日志。启用后会输出模型元数据（采样率、窗口大小等）和推理过程的详细信息，有助于排查问题 |
| **provider** | string | "cpu" | 执行提供者，指定模型运行的后端。可选值：<br>- `"cpu"`: CPU执行（默认，通用）<br>- `"cuda"`: NVIDIA GPU执行（需要CUDA支持）<br>- `"coreml"`: Apple CoreML执行（macOS/iOS）<br>- `"rknn"`: RKNN设备执行（瑞芯微等） |

##### 模型元数据参数

分割模型文件（ONNX格式）中包含以下元数据，这些参数在模型加载时自动读取，**无需手动设置**：

| 元数据参数 | 类型 | 说明 | 典型值（Pyannote 3.0） | 含义 |
|-----------|------|------|----------------------|------|
| **sample_rate** | int32_t | 模型期望的音频采样率（Hz） | 16000 | 输入音频必须重采样到此采样率 |
| **window_size** | int32_t | 处理窗口大小（样本数） | 160000 | 每个窗口对应 10 秒音频（160000/16000） |
| **window_shift** | int32_t | 窗口移动步长（样本数） | 16000 | 相邻窗口重叠 9 秒，移动 1 秒 |
| **receptive_field_size** | int32_t | 感受野大小（样本数） | 991 | 模型实际"看到"的音频范围，约 62ms |
| **receptive_field_shift** | int32_t | 感受野移动步长（样本数） | 270 | 决定时间分辨率，约 16.875ms |
| **num_speakers** | int32_t | 模型支持的最大说话人数量 | 3 | 模型最多可同时检测 3 个说话人 |
| **powerset_max_classes** | int32_t | Powerset 编码的最大类别数 | 2 | 支持同时检测最多 2 个说话人（重叠说话） |
| **num_classes** | int32_t | 输出类别总数 | 7 | 包括：无说话人(1) + 单个说话人(3) + 两个说话人组合(3) = 7 |

**关键概念详解**：

1. **Window Size（窗口大小）**
   - **作用**：模型每次处理固定时长的音频窗口
   - **计算**：窗口时长 = window_size / sample_rate
   - **示例**：window_size=160000, sample_rate=16000 → 窗口时长 = 10秒
   - **长音频处理**：通过滑动窗口方式处理，相邻窗口有重叠

2. **Window Shift（窗口移动步长）**
   - **作用**：控制相邻窗口之间的重叠量
   - **计算**：通常为 window_size 的 10%
   - **示例**：window_size=160000 → window_shift=16000（1秒）
   - **影响**：重叠越多，边界处理越平滑，但计算量也越大

3. **Receptive Field（感受野）**
   - **作用**：模型实际用于预测的时间范围
   - **特点**：通常小于 window_size，因为模型需要边界上下文信息
   - **示例**：receptive_field_size=991（约62ms）< window_size=160000（10秒）
   - **意义**：模型在窗口内使用感受野进行逐帧预测

4. **Receptive Field Shift（感受野移动步长）**
   - **作用**：决定输出的时间分辨率
   - **计算**：时间分辨率 = receptive_field_shift / sample_rate
   - **示例**：receptive_field_shift=270, sample_rate=16000 → 时间分辨率 ≈ 16.875ms
   - **影响**：步长越小，时间分辨率越高，但计算量越大

5. **Powerset 编码**
   - **作用**：将多个说话人的组合编码为单一类别
   - **原理**：使用 powerset（幂集）方式表示说话人组合
   - **示例**（3个说话人，max_classes=2）：
     - 类别 0：无说话人
     - 类别 1：说话人1
     - 类别 2：说话人2
     - 类别 3：说话人3
     - 类别 4：说话人1+说话人2
     - 类别 5：说话人1+说话人3
     - 类别 6：说话人2+说话人3
   - **优势**：支持同时检测多个说话人（重叠说话场景）

6. **时间分辨率**
   - **定义**：模型输出结果的最小时间单位
   - **计算**：receptive_field_shift / sample_rate
   - **示例**：270 / 16000 ≈ 16.875ms
   - **意义**：决定了说话人切换检测的精度

##### 使用示例

**Python 配置示例**：

```python
import sherpa_onnx

# 基本配置
config = sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
    pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
        model="./sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
    ),
    num_threads=4,      # 使用4个线程加速推理
    debug=True,         # 启用调试日志，查看模型元数据
    provider="cpu"      # 使用CPU执行
)

# 在 Speaker Diarization 中使用
sd_config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
    segmentation=config,
    embedding=embedding_config,
    clustering=clustering_config
)
```

**Kotlin 配置示例**：

```kotlin
val segmentation = OfflineSpeakerSegmentationModelConfig(
    pyannote = OfflineSpeakerSegmentationPyannoteModelConfig(
        model = "segmentation.onnx"  // 模型文件路径
    ),
    numThreads = 4,     // 多线程加速
    debug = true,       // 调试模式
    provider = "cpu"    // CPU执行
)
```

**直接使用分割模型（仅分割，无聚类）**：

参考 `k2_speaker_segmentation_cut.py` 示例，直接使用 ONNX Runtime 加载模型：

```python
import onnxruntime as ort

# 加载模型
model = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])

# 读取元数据
meta = model.get_modelmeta().custom_metadata_map
window_size = int(meta["window_size"])        # 160000
sample_rate = int(meta["sample_rate"])       # 16000
num_speakers = int(meta["num_speakers"])      # 3
```

##### 参数调优建议

| 场景 | 推荐配置 | 说明 |
|------|---------|------|
| **快速处理** | `num_threads = CPU核心数` | 充分利用多核CPU，显著加速 |
| **调试问题** | `debug = true` | 查看模型元数据和推理过程 |
| **GPU加速** | `provider = "cuda"` | 如果有NVIDIA GPU，可显著加速 |
| **移动端** | `num_threads = 2`, `provider = "cpu"` | 平衡性能和功耗 |
| **服务器端** | `num_threads = 8+`, `provider = "cuda"` | 最大化性能 |

#### embedding（嵌入提取器配置）

参考 `SpeakerEmbeddingExtractorConfig`：
- **model**: 声纹嵌入模型路径（必需）
- **numThreads**: 推理线程数
- **debug**: 是否启用调试日志
- **provider**: 执行提供者

#### clustering（聚类配置）

```16:19:sherpa-onnx/kotlin-api/OfflineSpeakerDiarization.kt
data class FastClusteringConfig(
    var numClusters: Int = -1,
    var threshold: Float = 0.5f,
)
```

- **numClusters**: 固定聚类数量（>0 时使用固定数量，<=0 时使用阈值）
- **threshold**: 聚类阈值（仅当 numClusters <= 0 时使用，默认：0.5）

详细说明参考 [K2_SPEAKER_CLUSTERING.md](./K2_SPEAKER_CLUSTERING.md)

#### minDurationOn

- **类型**: `float`
- **默认值**: `0.2` 秒
- **说明**: 如果片段时长小于此值，则被丢弃。设置为 0 则不丢弃任何片段

#### minDurationOff

- **类型**: `float`
- **默认值**: `0.5` 秒
- **说明**: 如果同一说话人的两个片段之间的间隔小于此值，则合并为一个片段。递归执行

## 使用示例

### Python 示例

#### 基本使用

```57:93:python-api-examples/offline-speaker-diarization.py
def init_speaker_diarization(num_speakers: int = -1, cluster_threshold: float = 0.5):
    """
    Args:
      num_speakers:
        If you know the actual number of speakers in the wave file, then please
        specify it. Otherwise, leave it to -1
      cluster_threshold:
        If num_speakers is -1, then this threshold is used for clustering.
        A smaller cluster_threshold leads to more clusters, i.e., more speakers.
        A larger cluster_threshold leads to fewer clusters, i.e., fewer speakers.
    """
    segmentation_model = "./sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
    embedding_extractor_model = (
        "./3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
    )

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=segmentation_model
            ),
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=embedding_extractor_model
        ),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_speakers, threshold=cluster_threshold
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        raise RuntimeError(
            "Please check your config and make sure all required files exist"
        )

    return sherpa_onnx.OfflineSpeakerDiarization(config)
```

#### 处理音频

```102:132:python-api-examples/offline-speaker-diarization.py
def main():
    wave_filename = "./0-four-speakers-zh.wav"
    if not Path(wave_filename).is_file():
        raise RuntimeError(f"{wave_filename} does not exist")

    audio, sample_rate = sf.read(wave_filename, dtype="float32", always_2d=True)
    audio = audio[:, 0]  # only use the first channel

    # Since we know there are 4 speakers in the above test wave file, we use
    # num_speakers 4 here
    sd = init_speaker_diarization(num_speakers=4)
    
    # Resample audio to match the expected sample rate
    target_sample_rate = sd.sample_rate
    audio, sample_rate = resample_audio(audio, sample_rate, target_sample_rate)
    
    if sample_rate != sd.sample_rate:
        raise RuntimeError(
            f"Expected samples rate: {sd.sample_rate}, given: {sample_rate}"
        )

    show_progress = True

    if show_progress:
        result = sd.process(audio, callback=progress_callback).sort_by_start_time()
    else:
        result = sd.process(audio).sort_by_start_time()

    for r in result:
        print(f"{r.start:.3f} -- {r.end:.3f} speaker_{r.speaker:02}")
        #  print(r) # this one is simpler
```

### Kotlin 示例

```41:66:android/SherpaOnnxSpeakerDiarization/app/src/main/java/com/k2fsa/sherpa/onnx/speaker/diarization/SpeakerDiarizationObject.kt
    fun initSpeakerDiarization(assetManager: AssetManager? = null) {
        synchronized(this) {
            if (_sd != null) {
                return
            }
            Log.i(TAG, "Initializing sherpa-onnx speaker diarization")

            val config = OfflineSpeakerDiarizationConfig(
                segmentation = OfflineSpeakerSegmentationModelConfig(
                    pyannote = OfflineSpeakerSegmentationPyannoteModelConfig(
                        segmentationModel
                    ),
                    debug = true,
                ),
                embedding = SpeakerEmbeddingExtractorConfig(
                    model = embeddingModel,
                    debug = true,
                    numThreads = 2,
                ),
                clustering = FastClusteringConfig(numClusters = -1, threshold = 0.5f),
                minDurationOn = 0.2f,
                minDurationOff = 0.5f,
            )
            _sd = OfflineSpeakerDiarization(assetManager = assetManager, config = config)
        }
    }
```

### 模型下载

#### 分割模型

在 [speaker-segmentation-models](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-segmentation-models) 发布页面提供了多种分割模型，主要区别如下：

##### 分割模型配置参数

分割模型配置通过 `OfflineSpeakerSegmentationModelConfig` 进行设置：

```9:14:sherpa-onnx/kotlin-api/OfflineSpeakerDiarization.kt
data class OfflineSpeakerSegmentationModelConfig(
    var pyannote: OfflineSpeakerSegmentationPyannoteModelConfig = OfflineSpeakerSegmentationPyannoteModelConfig(),
    var numThreads: Int = 1,
    var debug: Boolean = false,
    var provider: String = "cpu",
)
```

**参数详细说明**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| **pyannote.model** | string | "" | **必需**。分割模型文件路径（如 `model.onnx`） |
| **numThreads** | int | 1 | 推理线程数。多线程可加速处理，建议设置为 CPU 核心数 |
| **debug** | bool | false | 是否启用调试日志。启用后会输出模型元数据和推理信息 |
| **provider** | string | "cpu" | 执行提供者。可选值：`"cpu"`、`"cuda"`（需要 GPU）、`"coreml"`（macOS）、`"rknn"`（RKNN 设备） |

##### 分割模型元数据

分割模型包含以下元数据（从模型文件中自动读取，无需手动设置）：

| 元数据参数 | 类型 | 说明 | 典型值 |
|-----------|------|------|--------|
| **sample_rate** | int32_t | 模型期望的音频采样率（Hz） | 16000 |
| **window_size** | int32_t | 处理窗口大小（样本数）。每个窗口的音频时长 = window_size / sample_rate | 160000（10秒） |
| **window_shift** | int32_t | 窗口移动步长（样本数）。通常为 window_size 的 10% | 16000（1秒） |
| **receptive_field_size** | int32_t | 感受野大小（样本数）。模型实际"看到"的音频范围 | 991 |
| **receptive_field_shift** | int32_t | 感受野移动步长（样本数）。决定时间分辨率 | 270 |
| **num_speakers** | int32_t | 模型支持的最大说话人数量 | 3 |
| **powerset_max_classes** | int32_t | Powerset 编码的最大类别数。通常为 2（支持同时检测 2 个说话人） | 2 |
| **num_classes** | int32_t | 输出类别总数。包括：无说话人 + 单个说话人组合 + 多个说话人组合 | 7 |

**关键概念说明**：

1. **Window Size（窗口大小）**：
   - 模型每次处理固定时长的音频窗口
   - 例如：window_size=160000（16kHz采样率）对应 10 秒音频
   - 长音频通过滑动窗口方式处理

2. **Window Shift（窗口移动步长）**：
   - 相邻窗口之间的重叠量
   - 通常为 window_size 的 10%，用于平滑处理边界

3. **Receptive Field（感受野）**：
   - 模型实际用于预测的时间范围
   - 通常小于 window_size，因为模型需要边界上下文

4. **Powerset 编码**：
   - 模型输出使用 powerset 编码表示说话人组合
   - 例如：类别 0=无说话人，类别 1=说话人1，类别 2=说话人2，类别 3=说话人1+说话人2
   - 支持同时检测多个说话人（重叠说话）

5. **时间分辨率**：
   - 由 `receptive_field_shift` 决定
   - 例如：receptive_field_shift=270（16kHz）对应约 16.875ms 的时间分辨率

##### 使用示例

**Python 示例**：

```python
import sherpa_onnx

config = sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
    pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
        model="./sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
    ),
    num_threads=4,  # 使用4个线程加速
    debug=True,     # 启用调试日志
    provider="cpu"  # 使用CPU执行
)
```

**Kotlin 示例**：

```kotlin
val segmentation = OfflineSpeakerSegmentationModelConfig(
    pyannote = OfflineSpeakerSegmentationPyannoteModelConfig(
        model = "segmentation.onnx"
    ),
    numThreads = 4,
    debug = true,
    provider = "cpu"
)
```

##### 模型选择

##### 1. Pyannote Segmentation 3.0

**模型文件**：`sherpa-onnx-pyannote-segmentation-3-0.tar.bz2` (约 6.64 MB)

**特点**：
- 基于 Pyannote Audio 的 segmentation-3.0 模型
- 通用场景，适合大多数标准音频环境
- 模型较小，推理速度快
- 适合清晰录音、电话录音等场景

**下载**：
```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
tar xvf sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
```

##### 2. Reverb Diarization v1

**模型文件**：`sherpa-onnx-reverb-diarization-v1.tar.bz2` (约 10.4 MB)

**特点**：
- 来自 Revai 的 reverb-diarization-v1 模型
- **专门针对有混响（reverb）环境的音频优化**
- 适合会议室、大厅等有回声的录音环境
- 在混响场景下表现更好

**下载**：
```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-reverb-diarization-v1.tar.bz2
tar xvf sherpa-onnx-reverb-diarization-v1.tar.bz2
```

##### 3. Reverb Diarization v2

**模型文件**：`sherpa-onnx-reverb-diarization-v2.tar.bz2` (约 242 MB)

**特点**：
- Reverb Diarization 的升级版本
- **模型更大，性能更强**
- 在混响环境下的准确率更高
- 适合对准确率要求高的混响场景

**下载**：
```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-reverb-diarization-v2.tar.bz2
tar xvf sherpa-onnx-reverb-diarization-v2.tar.bz2
```

##### 模型选择建议

| 场景 | 推荐模型 | 原因 |
|------|---------|------|
| 清晰录音、电话录音 | Pyannote Segmentation 3.0 | 模型小、速度快、通用性好 |
| 会议室录音（轻微混响） | Reverb Diarization v1 | 针对混响优化，模型适中 |
| 大厅、大空间录音（强混响） | Reverb Diarization v2 | 最强性能，适合复杂混响环境 |
| 移动端应用 | Pyannote Segmentation 3.0 | 模型小，资源占用少 |
| 服务器端应用 | Reverb Diarization v2 | 性能优先，资源充足 |

**注意**：所有模型的使用方式相同，只需在配置中指定对应的模型路径即可。

#### 嵌入模型

下载声纹嵌入模型：
```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx
```

更多模型请访问：
- 分割模型：https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-segmentation-models
- 嵌入模型：https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models

## 与 VAD 分割音频片段的区别

### 核心区别

| 特性 | VAD (Voice Activity Detection) | Speaker Diarization |
|------|-------------------------------|---------------------|
| **主要功能** | 检测语音/非语音 | 检测语音并识别说话人 |
| **输出信息** | 语音片段（无说话人信息） | 带说话人ID的时间段 |
| **处理目标** | 区分语音和静音/噪声 | 区分不同说话人 |
| **应用场景** | 语音预处理、静音移除 | 多人对话分析、说话人识别 |
| **技术复杂度** | 相对简单 | 更复杂（三阶段处理） |
| **模型需求** | 只需 VAD 模型 | 需要分割模型 + 嵌入模型 |

### 详细对比

#### 1. 功能目标

**VAD**：
- 目标：检测音频中的语音活动
- 输出：语音片段的时间范围（start, end）
- 不区分说话人，只区分语音和非语音

**Speaker Diarization**：
- 目标：识别音频中每个时间段对应的说话人
- 输出：带说话人ID的时间段（start, end, speaker_id）
- 不仅检测语音，还识别说话人身份

#### 2. 处理流程

**VAD 流程**：
```
音频输入
  └─> VAD 模型检测
       └─> 输出：语音片段列表
```

**Speaker Diarization 流程**：
```
音频输入
  ├─> 分割模型（检测说话人活动）
  ├─> 嵌入提取（提取声纹特征）
  └─> 聚类算法（识别说话人）
       └─> 输出：带说话人ID的时间段列表
```

#### 3. 输出格式

**VAD 输出示例**（来自 `k2_vad_cut.py`）：
```python
# 只包含时间信息，无说话人信息
Segment audio_1000_2000 [1.00s-3.00s] duration=2.00s
Segment audio_5000_3000 [5.00s-8.00s] duration=3.00s
```

**Speaker Diarization 输出示例**：
```python
# 包含时间信息和说话人ID
0.000 -- 2.345 speaker_00
2.500 -- 5.678 speaker_01
5.500 -- 8.901 speaker_00
```

#### 4. 代码对比

**VAD 使用示例**（`k2_vad_cut.py`）：
```203:231:python-api-examples/k2_vad_cut.py
    for i in range(0, len(wav_16k), window_n):
        vad.accept_waveform(wav_16k[i : i + window_n])

        while not vad.empty():
            seg = vad.front
            samples = np.asarray(seg.samples, dtype=np.float32)
            
            # No minimum duration limit - process all VAD segments
            offset_samples = seg.start
            offset_ms = int(offset_samples / vad_sr * 1000)
            duration_ms = int(len(samples) / vad_sr * 1000)
            duration_seconds = duration_ms / 1000.0

            # Generate filename: audio_base_name_offsetms_durationms.wav
            wav_filename = f"{audio_base_name}_{offset_ms}_{duration_ms}.wav"
            wav_filepath = wav_output_dir / wav_filename
            segment_id = wav_filename.replace(".wav", "")  # Use filename without extension as segment_id

            sf.write(str(wav_filepath), samples, vad_sr)

            segments_info.append((segment_id, wav_filepath))
            segment_durations.append(duration_seconds)

            print(
                f"Segment {segment_id} [{offset_ms/1000:.2f}s-{(offset_ms + duration_ms)/1000:.2f}s] "
                f"duration={duration_ms/1000:.2f}s saved to {wav_filepath}"
            )

            vad.pop()
```

**Speaker Diarization 使用示例**：
```125:132:python-api-examples/offline-speaker-diarization.py
    if show_progress:
        result = sd.process(audio, callback=progress_callback).sort_by_start_time()
    else:
        result = sd.process(audio).sort_by_start_time()

    for r in result:
        print(f"{r.start:.3f} -- {r.end:.3f} speaker_{r.speaker:02}")
        #  print(r) # this one is simpler
```

#### 5. 应用场景对比

**VAD 适用场景**：
- 语音预处理：移除静音片段
- 音频分割：将长音频分割为语音片段
- ASR 预处理：只对语音片段进行识别
- 说话人识别预处理：分割后再提取声纹

**Speaker Diarization 适用场景**：
- 会议记录：识别每个发言者
- 电话录音分析：区分主叫和被叫
- 播客分析：识别不同嘉宾
- 多说话人转录：为每个说话人生成独立转录

#### 6. 性能对比

| 指标 | VAD | Speaker Diarization |
|------|-----|---------------------|
| **处理速度** | 快（单模型推理） | 较慢（三阶段处理） |
| **内存占用** | 低 | 较高（需要存储嵌入向量） |
| **模型大小** | 小（几MB） | 大（分割模型+嵌入模型，几十MB） |
| **计算复杂度** | O(n) | O(n²) 或 O(n³)（聚类阶段） |

#### 7. 组合使用

在实际应用中，VAD 和 Speaker Diarization 可以组合使用：

```
长音频
  └─> VAD 分割（快速过滤静音）
       └─> 只对语音片段进行 Speaker Diarization
            └─> 提高处理效率
```

示例：`speaker-identification-offline-long-audio.py` 中先使用 VAD 分割，再对每个片段提取声纹。

### 总结

- **VAD**：专注于语音/非语音检测，输出简单的时间段信息，适合预处理和快速分割
- **Speaker Diarization**：专注于说话人识别，输出带说话人ID的时间段，适合多人对话分析

两者可以互补使用：VAD 用于快速预处理，Speaker Diarization 用于精细的说话人分析。

## 应用场景

### 1. 会议记录

**场景**：多人会议录音，需要识别每个发言者

**流程**：
1. 使用 Speaker Diarization 识别每个时间段的说话人
2. 结合 ASR 为每个说话人生成独立转录
3. 生成带说话人标签的会议记录

### 2. 电话录音分析

**场景**：客服电话录音，需要区分客户和客服

**流程**：
1. 使用 Speaker Diarization 识别两个说话人
2. 分析每个说话人的发言时长和内容
3. 生成对话分析报告

### 3. 播客分析

**场景**：多人播客，需要识别不同嘉宾

**流程**：
1. 使用 Speaker Diarization 识别所有说话人
2. 统计每个说话人的发言时长
3. 生成说话人分布图

### 4. 多说话人转录

**场景**：需要为每个说话人生成独立的转录文本

**流程**：
1. Speaker Diarization 识别说话人
2. 对每个说话人片段进行 ASR
3. 生成带说话人标签的转录结果

## 最佳实践

### 1. 参数选择

#### 已知说话人数量

如果知道音频中的说话人数量，强烈建议使用固定聚类数：

```python
clustering = FastClusteringConfig(num_clusters=4)  # 已知有4个说话人
```

#### 未知说话人数量

如果不知道说话人数量，使用阈值模式：

```python
clustering = FastClusteringConfig(threshold=0.5)  # 需要根据数据调整
```

**阈值调整建议**：
- 说话人差异明显：`threshold = 0.6-0.8`
- 说话人差异较小：`threshold = 0.3-0.5`
- 需要更多说话人：降低阈值
- 需要更少说话人：提高阈值

### 2. 音频预处理

- **采样率匹配**：确保音频采样率与模型要求一致（通常 16kHz）
- **单声道**：转换为单声道音频
- **音频质量**：使用高质量音频，避免过度压缩

### 3. 性能优化

- **多线程**：设置合适的 `numThreads` 参数
- **GPU 加速**：如果支持，使用 GPU 执行提供者
- **批量处理**：对于多个音频文件，考虑批量处理

### 4. 结果后处理

- **合并相邻片段**：使用 `min_duration_off` 合并同一说话人的相邻片段
- **过滤短片段**：使用 `min_duration_on` 过滤过短片段
- **结果验证**：检查说话人数量是否合理，必要时调整参数

### 5. 常见问题

#### 说话人数量不准确

- **问题**：检测到的说话人数量与实际不符
- **解决**：
  - 如果知道实际数量，使用 `num_clusters`
  - 如果使用阈值，调整 `threshold` 参数
  - 检查音频质量，确保说话人声音清晰

#### 同一说话人被分割为多个

- **问题**：同一说话人的片段被分配了不同的ID
- **解决**：
  - 降低聚类阈值
  - 检查嵌入模型质量
  - 增加 `min_duration_off` 以合并相邻片段

#### 处理速度慢

- **问题**：处理长音频耗时过长
- **解决**：
  - 增加 `numThreads`
  - 使用 GPU 加速
  - 先使用 VAD 过滤静音片段

## 相关文档

- [K2_SPEAKER_CLUSTERING.md](./K2_SPEAKER_CLUSTERING.md) - 聚类算法详细说明
- [VAD_FRAMEWORK.md](./VAD_FRAMEWORK.md) - VAD 框架说明
- [SPEAKER_EMBEDDING_MANAGER.md](./SPEAKER_EMBEDDING_MANAGER.md) - 说话人嵌入管理

## 总结

K2 Speaker Diarization 是 sherpa-onnx 项目中用于说话人分离的核心功能，通过三阶段处理（分割、嵌入提取、聚类）实现端到端的说话人识别。与 VAD 相比，Speaker Diarization 不仅检测语音活动，还能识别每个时间段对应的说话人，适用于多人对话场景的分析和处理。

通过合理选择配置参数（固定聚类数或阈值）、使用高质量模型和适当的后处理，可以获得准确的说话人分离结果。

---

**文档版本**：1.0  
**最后更新**：2024年  
**维护者**：sherpa-onnx团队

