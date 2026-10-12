# Speaker Mask Fusion and Clean-Cluster Assignment Design

**日期：** 2026-09-20
**状态：** 已获用户确认，进入实现

## 1. 目标

扩展 `SherpaOnnxSpeakerSegmentation`，在现有 speaker-count span 的基础上完成跨滑窗 local track 对齐和 mask 融合，并在每个输出 span 中返回：

- `local_speaker_mask`：对齐到当前 session 全局 local track 坐标的三 bit mask；
- `local_speaker_mask_confidence`：融合 mask 的概率/置信度，范围 `[0, 1]`。

同时修改 `python-jimmy/offline-long-audio-pipeline-asr-speaker-segmentation.py` 及其 pipeline 模块：

- 只使用高质量 clean span 建立全局 speaker cluster；
- 低质量片段不能创建新 speaker；
- 利用 local mask 连续性提升 overlap/静音前后短句的 speaker ID 准确性；
- 记录每个结果片段的 clean span、mask、置信度和 speaker ID 来源；
- 使用固定测试集对优化前后版本进行 ASR WER、说话人 DER 和断句准确性对比，并生成报告。

## 2. 语义定义

### 2.1 local mask

`local_speaker_mask` 不是全局 speaker ID。它表示经过跨窗口 track permutation 对齐后的 session-local track 活动状态：

```text
bit 0 -> session local track 0
bit 1 -> session local track 1
bit 2 -> session local track 2
```

合法值来自 Pyannote 3-track / max-2-speaker Powerset 类别：

```text
000, 001, 010, 100, 011, 101, 110
```

`000` 表示无活动，不表示 unknown。

### 2.2 confidence

`local_speaker_mask_confidence` 是当前融合 Powerset 类别的概率，范围 `[0, 1]`。它不是全局 speaker embedding 相似度，也不是 track permutation 的唯一置信度。若需要诊断 alignment ambiguity，内部保留 best/second-best permutation margin；对外首版只输出 mask confidence。

### 2.3 跨窗口对齐

第一个有效窗口建立 session-local canonical track。后续窗口使用与已融合 timeline 的有效重叠帧，枚举 3! 个 permutation，依据 track 活动的交集和冲突选择最佳映射。对齐不足或 best/second-best margin 过低时，仍输出最佳 mask，但降低 confidence，不强制把它当作确定换人证据。

### 2.4 融合

优先保留每帧 7 类 Powerset 概率。窗口完成 permutation 后，将类别概率按 permutation 重映射到 canonical track 坐标，并对覆盖同一全局 frame 的窗口概率进行平均。最终选择融合概率最大的 Powerset 类别并转换为 mask，保证不会产生非法 `111`。

如果实现阶段因现有 C++ forward 接口限制暂时只能使用二值 mask，则使用对齐后的 per-track 多数票，并限制最多两个 active track；最终实现必须保留 confidence。

时间平滑在 mask 对齐和跨窗口融合之后执行，避免不同窗口的 local label 交换污染融合结果。输出 span 在融合 mask 或 speaker count 稳定变化处切分，并继续保留现有 `CONTINUE`、`SPEAKER_COUNT_CHANGED`、`SINGLE_SPEAKER_CHANGED`、`INPUT_FINISHED` 语义。

## 3. Pipeline speaker assignment

### 3.1 clean span

全局聚类只接受连续、高质量 clean span。默认条件：

- 单人活动；
- 不包含 overlap；
- 不包含 unknown activity 或另一个 speaker；
- 连续 clean 时长至少 3 秒；
- ASR 有效。

最终结果使用 `clean_spans: list[[start_ms, end_ms]]` 保存可用于提取声纹的连续区间。不同 clean span 不跨 overlap/不确定区间拼接后参与一次 embedding。

### 3.2 低质量片段

低质量片段按以下顺序赋予 speaker ID：

1. 如果与已有可靠片段之间的 aligned local mask 连续性足够强，则继承对应 speaker ID；
2. 否则与 clean cluster 的 centroid 比较；
3. 达不到 centroid 阈值则保持 `unknown`；
4. 低质量片段永远不能创建或更新全局 cluster。

local mask 继承必须受到最大关联 gap、mask confidence、alignment confidence 和相邻 clean/已解析 speaker ID 的约束。无法判断时宁可 fallback 或 unknown，不强制继承。

结果中记录：

```text
speaker_assignment_source = clean_cluster | local_mask_inherit | centroid_match | unknown
```

## 4. 对外 API

C++ span 增加：

```cpp
uint8_t local_speaker_mask;
float local_speaker_mask_confidence;
```

Python binding 只读暴露同名属性。保留现有字段和 flag 常量，确保旧调用仍可读取 start/end/speaker_count/flag。

## 5. 测试与评测

### 5.1 单元测试

覆盖：

- 跨窗口 permutation 对齐；
- Powerset 概率重映射和 mask 融合；
- 同人 `001 -> overlap -> 001`；
- 换人 `001 -> overlap -> 010`；
- 低 coverage/低 margin confidence；
- 不能输出非法 `111`；
- reset、input finished、尾部裁剪和旧 span flag 回写不回归。

Python pipeline 覆盖：

- clean span >= 3 秒才能参与 cluster；
- overlap/unknown/短片段不能创建新 cluster；
- local mask 连续时短片段继承长片段 speaker ID；
- local mask 不连续或低置信度时 fallback centroid；
- 输出 clean_spans、mask、confidence 和 assignment source。

### 5.2 远程评测

使用 `sherpa-onnx-remote-build-test` 固定服务器和远程仓库流程。测试集：

```text
Z:\ProMax\测试集\音频测试集\中文\说话人测试集.txt
```

结果目录：

```text
C:\Users\admin\Downloads\断句pipeline-python脚本测试-0920
```

报告必须包含：

- baseline 与 optimized 的 commit、模型路径和完整命令；
- CMake/build/test 日志；
- 每条音频的 ASR WER；
- 说话人 DER、miss、false alarm、confusion；
- 断句边界命中率、漏切、错切；
- 片段数、overlap 数、speaker ID 分布、unknown 数；
- clean span 覆盖率、local mask 继承次数、centroid fallback 次数；
- 优化前后差异和失败/未执行项目。

## 6. 非目标

- 不把 local track 直接映射为跨音频永久 speaker ID；
- 不让低质量片段参与新的全局 cluster；
- 不修改 unrelated 的 ASR/VAD 模型或默认参数；
- 不以降低 WER 为唯一成功标准；DER 和断句指标必须同时报告。
