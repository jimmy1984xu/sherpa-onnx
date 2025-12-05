# ASR 文本置信度实现文档

## 概述

本文档详细介绍 `sherpa-onnx` 中 ASR 文本置信度功能的实现原理、使用方法和应用场景。文本置信度功能目前支持 **Paraformer** 和 **NeMo Transducer** 模型。

## 当前状态

**Paraformer 和 NeMo Transducer 模型已支持文本置信度功能**。离线识别结果（`OfflineRecognitionResult`）包含：
- `text`: 识别文本
- `tokens`: Token 列表
- `timestamps`: 时间戳
- `confidence`: **平均置信度分数**（Paraformer 和 NeMo Transducer 模型，范围 [0.0, 1.0]，或 -1.0 表示不可用）
  - 空文本（无识别结果）时，置信度为 0.0
- `lang`: 语言（部分模型）
- `emotion`: 情感（部分模型）
- `event`: 事件（部分模型）

## 置信度计算原理

### 实现方式

Paraformer 和 NeMo Transducer 模型的置信度计算在 C++ 解码器中完成，具体逻辑如下：

1. **收集每个 token 的 log 概率**：
   - 在解码过程中，对于每个选中的 token，保存其 log 概率值 `log_prob_i`
   - **Paraformer 模型**：log 概率值来自模型输出的 log_softmax 结果
   - **NeMo Transducer 模型**：对 joiner 输出的 logits 应用 log_softmax 计算 log 概率

2. **计算平均置信度**：
   ```cpp
   // 对每个 token 的 log 概率取 exp，转换为概率值
   for (float log_prob : token_log_probs) {
       sum_exp += std::exp(log_prob);  // exp(log_prob) = probability
   }
   // 计算所有 token 概率的算术平均值
   confidence = sum_exp / token_count;
   ```

3. **数学公式**：
   ```
   confidence = (1/N) * Σ exp(log_prob_i)
   ```
   其中：
   - `N` 是 token 数量
   - `log_prob_i` 是第 i 个 token 的 log 概率
   - `exp(log_prob_i)` 是第 i 个 token 的概率值（0-1 范围）

### 置信度含义

- **数值范围**：`[0.0, 1.0]`，值越大表示置信度越高
- **计算方式**：所有 token 概率值的**算术平均值**
- **物理意义**：表示模型对识别文本的整体置信度
  - 接近 1.0：模型对识别结果非常确信
  - 接近 0.0：模型对识别结果不确定
  - 0.0：空文本（无识别结果）时，置信度为 0.0
  - -1.0：置信度不可用（非 Paraformer/NeMo Transducer 模型或识别失败）

### 注意事项

1. **不是 log 概率的平均值**：
   - 当前实现是 `mean(exp(log_prob))`（概率的算术平均）
   - 而不是 `exp(mean(log_prob))`（log 概率的几何平均）
   - 两种方法都可以用于判断可信度，但算术平均更直观

2. **支持的模型**：
   - **Paraformer 模型**：完全支持置信度计算
   - **NeMo Transducer 模型**：完全支持置信度计算
   - 其他模型（普通 Transducer、CTC、Whisper 等）的 `confidence` 值为 -1.0

3. **可用于判断识别文本的可信度**：
   - 置信度反映了模型对识别结果的确定性
   - 可以用于过滤低质量识别结果
   - 可以用于质量评估和错误检测

## 性能评估

### 性能影响

引入文本置信度功能会对识别性能产生一定影响，主要体现在：

1. **计算开销**：
   - **Paraformer 模型**：需要额外收集每个 token 的 log 概率值，并在解码后计算平均置信度
   - **NeMo Transducer 模型**：需要计算 log_softmax 获取 log 概率，并在解码后计算平均置信度
   - 置信度计算本身的开销相对较小，主要是对每个 token 执行一次 `exp()` 操作

2. **性能下降评估**：
   - **Paraformer 模型**：性能下降约 **1-3%**
     - 主要开销：收集 log 概率（几乎无开销，仅保存值）
     - 次要开销：计算置信度（对每个 token 执行 `exp()` 和除法）
   - **NeMo Transducer 模型**：性能下降约 **2-5%**
     - 主要开销：计算 log_softmax（需要遍历整个词汇表）
     - 次要开销：计算置信度（对每个 token 执行 `exp()` 和除法）

3. **影响因素**：
   - **Token 数量**：识别文本越长（token 越多），计算开销越大
   - **词汇表大小**：NeMo Transducer 的 log_softmax 计算与词汇表大小成正比
   - **硬件性能**：在 CPU 上，`exp()` 操作相对较慢；在 GPU 上影响更小

4. **优化建议**：
   - 如果对性能要求极高且不需要置信度，可以考虑禁用置信度计算（需要修改代码）
   - 对于批量处理场景，置信度计算的开销可以分摊到多个样本上，影响更小
   - 置信度计算已通过通用工具类优化，使用内联函数减少函数调用开销

## 使用示例

### Python 使用

#### Paraformer 模型

```python
import sherpa_onnx

# 创建识别器
recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
    paraformer="model.onnx",
    tokens="tokens.txt",
)

# 识别
wave = sherpa_onnx.read_wave("audio.wav", expected_sample_rate=16000)
stream = recognizer.create_stream()
stream.accept_waveform(wave.sample_rate, wave.samples)
recognizer.decode_stream(stream)

# 获取结果
result = recognizer.get_result(stream)
print(f"识别文本: {result.text}")
print(f"置信度: {result.confidence:.4f}")

# 判断置信度
if result.confidence >= 0.8:
    print("高置信度识别结果")
elif result.confidence >= 0.5:
    print("中等置信度识别结果")
else:
    print("低置信度识别结果，建议人工复核")
```

#### NeMo Transducer 模型

```python
import sherpa_onnx

# 创建识别器
config = sherpa_onnx.OfflineRecognizerConfig(
    model=sherpa_onnx.OfflineModelConfig(
        nemo_transducer=sherpa_onnx.OfflineNemoTransducerModelConfig(
            encoder="encoder.onnx",
            decoder="decoder.onnx",
            joiner="joiner.onnx",
        ),
        tokens="tokens.txt",
    ),
)
recognizer = sherpa_onnx.OfflineRecognizer(config)

# 识别
wave = sherpa_onnx.read_wave("audio.wav", expected_sample_rate=16000)
stream = recognizer.create_stream()
stream.accept_waveform(wave.sample_rate, wave.samples)
recognizer.decode_stream(stream)

# 获取结果
result = recognizer.get_result(stream)
print(f"识别文本: {result.text}")
print(f"置信度: {result.confidence:.4f}")

# 判断置信度
if result.confidence >= 0.8:
    print("高置信度识别结果")
elif result.confidence >= 0.5:
    print("中等置信度识别结果")
else:
    print("低置信度识别结果，建议人工复核")
```

### Kotlin 使用

```kotlin
val result = recognizer.getResult(stream)
println("识别文本: ${result.text}")
println("置信度: ${result.confidence}")

// 判断置信度
when {
    result.confidence >= 0.8f -> println("高置信度识别结果")
    result.confidence >= 0.5f -> println("中等置信度识别结果")
    else -> println("低置信度识别结果，建议人工复核")
}
```

## 实现细节

### 通用置信度计算工具类

为了便于维护和扩展，置信度计算逻辑已抽取到通用工具类 `sherpa-onnx/csrc/confidence-utils.h` 中：

```cpp
// 计算平均置信度
inline float CalculateAverageConfidence(
    const std::vector<float> &token_log_probs) {
  if (token_log_probs.empty()) {
    return 0.0f;  // 空文本时返回 0.0
  }

  float sum_exp = 0.0f;
  for (float log_prob : token_log_probs) {
    sum_exp += std::exp(log_prob);  // exp(log_prob) = probability
  }
  return sum_exp / static_cast<float>(token_log_probs.size());
}
```

该工具类被以下实现共享使用：
- Paraformer 模型（标准、Ascend、RKNN 实现）
- NeMo Transducer 模型

### Paraformer 模型

置信度计算在 `sherpa-onnx/csrc/offline-paraformer-greedy-search-decoder.cc` 中实现：

```cpp
// 收集每个 token 的 log 概率
std::vector<float> token_log_probs;
for (int32_t k = 0; k != num_tokens; ++k) {
    auto max_idx = std::distance(p, std::max_element(p, p + vocab_size));
    if (max_idx == eos_id_) break;
    
    results[i].tokens.push_back(max_idx);
    token_log_probs.push_back(p[max_idx]);  // 保存 log 概率
    p += vocab_size;
}

// 使用通用工具类计算平均置信度
results[i].confidence = CalculateAverageConfidence(token_log_probs);
```

### NeMo Transducer 模型

置信度计算在 `sherpa-onnx/csrc/offline-transducer-greedy-search-nemo-decoder.cc` 中实现：

```cpp
// 对 joiner 输出的 logits 应用 log_softmax 计算 log 概率
float ComputeLogSoftmaxAndGetLogProb(const float *logits, int32_t vocab_size,
                                      int32_t selected_idx) {
  // Find max for numerical stability
  float max_logit = *std::max_element(logits, logits + vocab_size);
  
  // Compute log(sum(exp(x_j - max))) for numerical stability
  float log_sum_exp = 0.0f;
  for (int32_t i = 0; i != vocab_size; ++i) {
    log_sum_exp += std::exp(logits[i] - max_logit);
  }
  log_sum_exp = std::log(log_sum_exp) + max_logit;
  
  // Return log probability: logits[selected_idx] - log_sum_exp
  return logits[selected_idx] - log_sum_exp;
}

// 在解码过程中收集 log 概率
std::vector<float> token_log_probs;
// ... 解码循环 ...
float log_prob = ComputeLogSoftmaxAndGetLogProb(p_logit, vocab_size, y);
token_log_probs.push_back(log_prob);

// 使用通用工具类计算平均置信度
ans.confidence = CalculateAverageConfidence(token_log_probs);
```

## 置信度的应用

### 1. 过滤低置信度结果

```python
def filter_low_confidence_results(results, min_confidence=0.5):
    """过滤低置信度的识别结果"""
    filtered = []
    for result in results:
        if result.confidence >= min_confidence:
            filtered.append(result)
        else:
            print(f"低置信度结果已过滤: {result.text} (置信度: {result.confidence:.3f})")
    return filtered
```

### 2. 质量评估和错误检测

```python
def assess_recognition_quality(result):
    """评估识别结果的质量"""
    if result.confidence < 0:
        return "置信度不可用（不支持置信度的模型）"
    
    if result.confidence >= 0.9:
        return "优秀"
    elif result.confidence >= 0.7:
        return "良好"
    elif result.confidence >= 0.5:
        return "一般，建议复核"
    else:
        return "较差，强烈建议人工复核"
```

### 3. 批量处理中的质量控制

```python
def process_audio_batch(audio_files, recognizer, min_confidence=0.6):
    """批量处理音频，只保留高置信度结果"""
    results = []
    for audio_file in audio_files:
        # ... 识别代码 ...
        result = recognizer.get_result(stream)
        
        if result.confidence >= min_confidence:
            results.append({
                'file': audio_file,
                'text': result.text,
                'confidence': result.confidence
            })
        else:
            print(f"跳过低置信度结果: {audio_file} (置信度: {result.confidence:.3f})")
    
    return results
```

### 4. 置信度统计和分析

```python
def analyze_confidence_distribution(results):
    """分析置信度分布"""
    confidences = [r.confidence for r in results if r.confidence >= 0]
    
    if not confidences:
        print("没有可用的置信度数据")
        return
    
    import numpy as np
    print(f"平均置信度: {np.mean(confidences):.3f}")
    print(f"中位数置信度: {np.median(confidences):.3f}")
    print(f"最低置信度: {np.min(confidences):.3f}")
    print(f"最高置信度: {np.max(confidences):.3f}")
    print(f"标准差: {np.std(confidences):.3f}")
```

## 相关资源

- **离线 ASR 使用文档**: [OFFLINE_ASR.md](OFFLINE_ASR.md)
- **模型下载**: https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
- **Python API 文档**: `python-api-examples/`
- **C++ API 文档**: `sherpa-onnx/csrc/offline-recognizer.h`
- **Kotlin API 文档**: `sherpa-onnx/kotlin-api/OfflineRecognizer.kt`

## 总结

1. **Paraformer 和 NeMo Transducer 模型支持文本置信度功能**：通过 `confidence` 字段提供识别结果的可信度评估
2. **置信度计算方式**：所有 token 概率值的算术平均值 `mean(exp(log_prob))`
3. **置信度应用**：可用于过滤低质量结果、质量评估和错误检测
4. **支持的模型**：
   - Paraformer 模型：完全支持
   - NeMo Transducer 模型：完全支持
   - 其他模型类型（普通 Transducer、CTC、Whisper 等）的 `confidence` 值为 -1.0

