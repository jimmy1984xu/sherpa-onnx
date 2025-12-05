# K2声纹聚类算法说明文档

## 目录

- [概述](#概述)
- [算法原理](#算法原理)
- [实现架构](#实现架构)
- [核心代码位置](#核心代码位置)
- [配置参数](#配置参数)
- [使用示例](#使用示例)
- [算法特点](#算法特点)
- [性能优化](#性能优化)
- [应用场景](#应用场景)

## 概述

K2声纹聚类算法是sherpa-onnx项目中用于说话人识别和说话人分离（Speaker Diarization）的核心算法。该算法基于**层次聚类（Hierarchical Clustering）**方法，使用**余弦不相似度（Cosine Dissimilarity）**作为距离度量，能够自动将声纹嵌入向量聚类成不同的说话人。

### 主要特性

- ✅ **无需预设说话人数量**：支持基于阈值的自适应聚类
- ✅ **高效实现**：使用fastcluster库进行优化
- ✅ **完全链接**：使用Complete Linkage方法，产生紧凑的聚类
- ✅ **跨平台支持**：提供C++、Python、Java、Kotlin、Go等多种语言接口

## 算法原理

### 1. 层次聚类基础

K2声纹聚类采用**凝聚式层次聚类（Agglomerative Hierarchical Clustering）**算法：

1. **初始化**：每个声纹嵌入向量作为独立的聚类
2. **迭代合并**：逐步合并最相似的聚类对
3. **构建树状结构**：形成完整的聚类树（Dendrogram）
4. **切割聚类树**：根据配置参数切割树，得到最终聚类结果

### 2. 距离度量

算法使用**余弦不相似度**作为距离度量：

```
余弦相似度 = (A · B) / (||A|| × ||B||)
余弦不相似度 = 1 - 余弦相似度
```

**关键优化**：
- 所有嵌入向量预先进行**L2归一化**（L2-norm = 1）
- 归一化后，余弦相似度 = 向量点积（A · B）
- 余弦不相似度范围：[0, 2]，其中0表示完全相同

### 3. 链接方法

使用**完全链接（Complete Linkage）**方法：

- 两个聚类之间的距离 = 两个聚类中**最远点对**之间的距离
- 倾向于产生**紧凑的球形聚类**
- 对异常值相对不敏感
- 适合说话人识别场景

### 4. 聚类树切割策略

提供两种切割策略：

#### 策略1：固定聚类数量（`num_clusters > 0`）

```cpp
fastclustercpp::cutree_k(num_rows, merge.data(), config_.num_clusters, labels.data());
```

- **适用场景**：已知说话人数量
- **优点**：结果确定，无需调参
- **推荐使用**：当你知道音频中有多少个说话人时

#### 策略2：距离阈值（`num_clusters <= 0`）

```cpp
fastclustercpp::cutree_cdist(num_rows, merge.data(), height.data(), 
                             config_.threshold, labels.data());
```

- **适用场景**：未知说话人数量
- **优点**：自适应，自动发现说话人
- **注意**：需要根据数据调整阈值

## 实现架构

### 核心类结构

```
FastClustering (公共接口)
    └── Impl (内部实现)
        ├── FastClusteringConfig (配置)
        └── fastcluster库 (底层算法)
```

### 设计模式

- **PIMPL模式**：隐藏实现细节，提供稳定的ABI
- **配置驱动**：通过配置类控制算法行为
- **就地修改**：特征矩阵在聚类过程中被归一化（in-place）

## 核心代码位置

### C++核心实现

| 文件 | 说明 | 行数 |
|------|------|------|
| `sherpa-onnx/csrc/fast-clustering.h` | 头文件，定义公共接口 | 1-44 |
| `sherpa-onnx/csrc/fast-clustering.cc` | 核心实现，包含聚类算法 | 1-84 |
| `sherpa-onnx/csrc/fast-clustering-config.h` | 配置类定义 | 1-40 |
| `sherpa-onnx/csrc/fast-clustering-config.cc` | 配置类实现 | 1-46 |
| `sherpa-onnx/csrc/fast-clustering-test.cc` | 单元测试 | 1-70 |

### Python绑定

| 文件 | 说明 |
|------|------|
| `sherpa-onnx/python/csrc/fast-clustering.cc` | Python绑定实现 |
| `sherpa-onnx/python/tests/test_fast_clustering.py` | Python测试用例 |

### 其他语言绑定

- **Java**: `sherpa-onnx/java-api/src/main/java/com/k2fsa/sherpa/onnx/FastClusteringConfig.java`
- **Kotlin**: `sherpa-onnx/kotlin-api/OfflineSpeakerDiarization.kt`
- **Go**: 通过C API调用
- **C#**: `scripts/dotnet/FastClusteringConfig.cs`

## 配置参数

### FastClusteringConfig

```cpp
struct FastClusteringConfig {
  int32_t num_clusters = -1;  // 固定聚类数量（>0时忽略threshold）
  float threshold = 0.5;      // 距离阈值（仅当num_clusters <= 0时使用）
};
```

### 参数说明

#### `num_clusters`

- **类型**：`int32_t`
- **默认值**：`-1`
- **说明**：
  - 如果 `> 0`：使用固定聚类数量，`threshold` 参数被忽略
  - 如果 `<= 0`：使用阈值聚类，根据 `threshold` 自动确定聚类数量
- **推荐**：如果知道说话人数量，强烈建议设置此参数

#### `threshold`

- **类型**：`float`
- **默认值**：`0.5`
- **范围**：`[0, 2]`（余弦不相似度范围）
- **说明**：
  - **越小**：产生更多聚类（更多说话人）
  - **越大**：产生更少聚类（更少说话人）
- **注意**：需要根据实际数据调整，建议范围 `[0.3, 0.8]`

### 参数选择建议

| 场景 | 推荐配置 |
|------|----------|
| 已知说话人数量（如会议记录） | `num_clusters = N` |
| 未知说话人数量（如电话录音） | `threshold = 0.5`（需调参） |
| 说话人差异明显 | `threshold = 0.6-0.8` |
| 说话人差异较小 | `threshold = 0.3-0.5` |

## 使用示例

### Python示例

#### 示例1：固定聚类数量

```python
import sherpa_onnx
import numpy as np

# 准备声纹嵌入向量（num_segments x embedding_dim）
embeddings = np.array([
    [0.1, 0.2, 0.3, ...],  # 说话人1的片段1
    [0.15, 0.25, 0.35, ...],  # 说话人1的片段2
    [0.8, 0.9, 0.7, ...],  # 说话人2的片段1
    [0.85, 0.95, 0.75, ...],  # 说话人2的片段2
], dtype=np.float32)

# 配置：已知有2个说话人
config = sherpa_onnx.FastClusteringConfig(num_clusters=2)
clustering = sherpa_onnx.FastClustering(config)

# 执行聚类（就地修改embeddings，进行归一化）
labels = clustering(embeddings)

# labels: [0, 0, 1, 1] 表示前两个片段属于说话人0，后两个属于说话人1
print(f"Clustering labels: {labels}")
```

#### 示例2：阈值聚类

```python
# 配置：使用阈值，自动确定说话人数量
config = sherpa_onnx.FastClusteringConfig(threshold=0.5)
clustering = sherpa_onnx.FastClustering(config)

labels = clustering(embeddings)
print(f"Found {len(set(labels))} speakers")
print(f"Labels: {labels}")
```

#### 示例3：完整工作流（VAD + 声纹提取 + 聚类）

```python
# 1. VAD分割音频
# 2. 提取每个片段的声纹嵌入
# 3. 聚类

embeddings = []
for segment in vad_segments:
    emb = extractor.compute(segment)
    embeddings.append(emb)

embeddings = np.array(embeddings, dtype=np.float32)

# 聚类
config = sherpa_onnx.FastClusteringConfig(num_clusters=3)
clustering = sherpa_onnx.FastClustering(config)
labels = clustering(embeddings)

# 为每个片段分配说话人ID
for i, (segment, label) in enumerate(zip(vad_segments, labels)):
    segment.speaker_id = label
    print(f"Segment {i}: Speaker {label}")
```

### C++示例

```cpp
#include "sherpa-onnx/csrc/fast-clustering.h"

// 准备特征矩阵（行主序，num_segments x embedding_dim）
std::vector<float> features = {
    // segment 0
    0.1f, 0.2f, 0.3f, ...,
    // segment 1
    0.15f, 0.25f, 0.35f, ...,
    // ...
};

// 配置
sherpa_onnx::FastClusteringConfig config;
config.num_clusters = 2;  // 或使用 config.threshold = 0.5f;

// 创建聚类器
sherpa_onnx::FastClustering clustering(config);

// 执行聚类
int32_t num_segments = 10;
int32_t embedding_dim = 512;
auto labels = clustering.Cluster(features.data(), num_segments, embedding_dim);

// 使用标签
for (size_t i = 0; i < labels.size(); ++i) {
    std::cout << "Segment " << i << ": Speaker " << labels[i] << "\n";
}
```

### Java示例

```java
import com.k2fsa.sherpa.onnx.FastClusteringConfig;
import com.k2fsa.sherpa.onnx.FastClustering;

// 配置
FastClusteringConfig config = FastClusteringConfig.builder()
    .setNumClusters(3)  // 或 .setThreshold(0.5f)
    .build();

FastClustering clustering = new FastClustering(config);

// 执行聚类（features是float[][]，每行是一个嵌入向量）
int[] labels = clustering.cluster(features);
```

### 实际应用示例

参考项目中的完整示例：

1. **说话人分离（Speaker Diarization）**
   - 文件：`python-api-examples/cluster-speaker-segments.py`
   - 功能：对VAD分割的音频片段进行说话人聚类

2. **离线说话人分离**
   - 文件：`python-api-examples/offline-speaker-diarization.py`
   - 功能：端到端的说话人分离流程

3. **测试用例**
   - 文件：`sherpa-onnx/python/tests/test_fast_clustering.py`
   - 功能：算法正确性验证

## 算法特点

### 优势

1. **无需预设聚类数量**
   - 支持阈值模式，自动发现说话人数量
   - 适合未知说话人数的场景

2. **确定性结果**
   - 不依赖随机初始化
   - 相同输入产生相同输出

3. **层次结构**
   - 提供完整的聚类层次
   - 可以获取不同粒度的聚类结果

4. **高效实现**
   - 使用fastcluster库优化
   - 支持大规模数据

5. **适合声纹特征**
   - 余弦相似度适合高维嵌入向量
   - L2归一化确保稳定性

### 局限性

1. **计算复杂度**
   - 时间复杂度：O(n³)（n为片段数量）
   - 空间复杂度：O(n²)（距离矩阵）
   - 不适合超大规模数据（>10000片段）

2. **参数调优**
   - 阈值模式需要根据数据调整
   - 不同数据集可能需要不同阈值

3. **对噪声敏感**
   - 一旦错误合并，无法修正
   - 需要高质量的声纹嵌入

## 性能优化

### 1. 向量归一化优化

```cpp
// 所有向量预先归一化，简化余弦相似度计算
m.rowwise().normalize();  // Eigen库高效实现
```

### 2. 距离矩阵优化

```cpp
// 只存储上三角矩阵，节省50%内存
std::vector<double> distance((num_rows * (num_rows - 1)) / 2);
```

### 3. 使用fastcluster库

- 优化的C++实现
- 支持多种链接方法
- 高效的聚类树构建和切割

### 4. 内存管理

- 就地修改特征矩阵（避免额外内存分配）
- 使用Eigen库的Map避免数据拷贝

## 应用场景

### 1. 说话人分离（Speaker Diarization）

**场景**：会议录音、电话录音、播客等多人对话场景

**流程**：
1. VAD分割音频为语音片段
2. 提取每个片段的声纹嵌入
3. 使用FastClustering聚类
4. 为每个片段分配说话人ID

**示例代码位置**：
- `python-api-examples/offline-speaker-diarization.py`
- `python-api-examples/cluster-speaker-segments.py`

### 2. 说话人识别（Speaker Identification）

**场景**：识别音频中的说话人身份

**流程**：
1. 提取待识别音频的声纹嵌入
2. 与已知说话人库进行聚类匹配
3. 确定说话人身份

### 3. 音频分析

**场景**：音频内容分析、转录等

**流程**：
1. 分割音频
2. 聚类说话人
3. 为每个说话人单独进行ASR识别

## 算法流程详解

### 完整流程

```
输入：声纹嵌入矩阵 (N x D)
  │
  ├─> 1. L2归一化所有向量
  │     └─> 确保 ||v|| = 1
  │
  ├─> 2. 构建距离矩阵
  │     ├─> 计算所有向量对的余弦不相似度
  │     └─> 存储为上三角矩阵
  │
  ├─> 3. 层次聚类
  │     ├─> 使用完全链接方法
  │     ├─> 构建聚类树（Dendrogram）
  │     └─> 生成merge和height数组
  │
  └─> 4. 切割聚类树
        ├─> 如果 num_clusters > 0: 切割为N个聚类
        └─> 如果 threshold > 0: 基于距离阈值切割
            │
            └─> 输出：聚类标签数组 (N,)
```

### 代码实现细节

#### 步骤1：归一化

```cpp
Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
    m(features, num_rows, num_cols);
m.rowwise().normalize();  // 每行归一化
```

#### 步骤2：距离计算

```cpp
for (int32_t i = 0; i != num_rows; ++i) {
    auto v = m.row(i);
    for (int32_t j = i + 1; j != num_rows; ++j) {
        double cosine_similarity = v.dot(m.row(j));  // 点积 = 余弦相似度
        double cosine_dissimilarity = 1 - cosine_similarity;
        distance[k] = cosine_dissimilarity;
        ++k;
    }
}
```

#### 步骤3：层次聚类

```cpp
fastclustercpp::hclust_fast(num_rows, distance.data(),
                            fastclustercpp::HCLUST_METHOD_COMPLETE,
                            merge.data(), height.data());
```

#### 步骤4：切割树

```cpp
if (config_.num_clusters > 0) {
    fastclustercpp::cutree_k(num_rows, merge.data(), config_.num_clusters, labels.data());
} else {
    fastclustercpp::cutree_cdist(num_rows, merge.data(), height.data(),
                                 config_.threshold, labels.data());
}
```

## 最佳实践

### 1. 参数选择

- **已知说话人数量**：使用 `num_clusters`，更准确
- **未知说话人数量**：使用 `threshold`，需要调参
- **阈值范围**：通常 `[0.3, 0.8]`，根据数据调整

#### `threshold` 参数设置指南

**默认值**：`0.5`（适合大多数场景的起点）

**推荐值范围**：`[0.3, 0.8]`

**不同场景的建议值**：

| 场景 | 推荐阈值 | 说明 |
|------|---------|------|
| **通用场景**（未知说话人数量） | `0.5` | 默认值，适合大多数情况 |
| **说话人差异明显**（男女、不同年龄段） | `0.6 - 0.8` | 说话人特征差异大，可以使用较高阈值 |
| **说话人差异较小**（同性别、相似音色） | `0.3 - 0.5` | 需要更严格的相似度要求才能区分 |
| **需要更多说话人**（避免过度合并） | `0.3 - 0.4` | 降低阈值，产生更多聚类 |
| **需要更少说话人**（避免过度分割） | `0.6 - 0.8` | 提高阈值，合并相似说话人 |
| **电话录音**（音质一般） | `0.4 - 0.6` | 考虑音质影响，适当降低阈值 |
| **会议录音**（多人对话） | `0.5 - 0.7` | 根据实际说话人数量调整 |
| **播客/访谈**（2-4人） | `0.5 - 0.6` | 中等阈值，平衡准确性和稳定性 |

**调参方法**：

1. **从默认值开始**：先用 `0.5` 测试，观察聚类结果
2. **根据结果调整**：
   - 如果说话人数量**偏少**（过度合并）：**降低阈值**（如 0.4 → 0.3）
   - 如果说话人数量**偏多**（过度分割）：**提高阈值**（如 0.5 → 0.6 → 0.7）
3. **验证方法**：
   - 检查同一说话人的片段是否被正确聚类
   - 检查不同说话人是否被错误合并
   - 结合ASR结果验证说话人标签的准确性
4. **经验值**：
   - 大多数实际应用中，`0.5 - 0.6` 是较好的起点
   - 如果数据质量高、说话人差异明显，可以尝试 `0.7 - 0.8`
   - 如果数据质量一般或说话人相似，建议使用 `0.4 - 0.5`

**注意事项**：
- 阈值是**余弦不相似度**，值越大表示距离越远
- 阈值设置需要根据**实际数据**调整，没有万能值
- 如果可能，优先使用 `num_clusters`（已知说话人数量时）

### 2. 数据预处理

- 确保声纹嵌入向量质量高
- 过滤过短的音频片段（< 0.5秒）
- 使用高质量的声纹提取模型

### 3. 性能优化

- 对于大量片段（>1000），考虑分批处理
- 使用固定聚类数量可以略微提升性能
- 考虑使用GPU加速声纹提取（如果支持）

### 4. 结果验证

- 检查聚类数量是否合理
- 验证同一说话人的片段是否被正确聚类
- 根据ASR结果验证聚类质量

## 相关资源

### 代码文件

- **核心实现**：`sherpa-onnx/csrc/fast-clustering.*`
- **Python绑定**：`sherpa-onnx/python/csrc/fast-clustering.cc`
- **测试用例**：`sherpa-onnx/python/tests/test_fast_clustering.py`
- **使用示例**：`python-api-examples/cluster-speaker-segments.py`

### 依赖库

- **fastcluster**：高效的层次聚类C++库
- **Eigen**：线性代数库，用于矩阵运算

### 相关算法

- **SpeakerEmbeddingManager**：说话人嵌入管理（用于说话人识别）
- **VAD (Voice Activity Detection)**：语音活动检测
- **Speaker Embedding Extraction**：声纹嵌入提取

## 总结

K2声纹聚类算法是sherpa-onnx项目中用于说话人分离的核心算法，基于层次聚类方法，使用余弦不相似度作为距离度量。该算法具有无需预设聚类数量、确定性结果、高效实现等优点，广泛应用于说话人分离、说话人识别等场景。

通过合理选择配置参数（固定聚类数量或阈值），结合高质量的声纹嵌入向量，可以获得准确的说话人聚类结果。

---

**文档版本**：1.0  
**最后更新**：2024年  
**维护者**：sherpa-onnx团队

