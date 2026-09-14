# Pyannote Segmentation 流式输入 API 设计

**日期：** 2026-09-14
**分支：** `codex/pyannote-segmentation-streaming-api`
**状态：** 已确认，待实施

## 1. 目标

在 sherpa-onnx 提供独立的 Pyannote speaker segmentation 能力。它只负责基于
`pyannote-segmentation-3.0` 的活动人数和单人换人候选边界检测，不集成 speaker
embedding、聚类或全局 speaker ID。

调用方以连续小块 PCM float 音频（例如 32 ms）输入同一个对象；对象按模型 metadata
维护缓存、执行滑窗推理和融合，并输出严格时间不重叠的 span：

```text
[start, end, speaker_count, flag]
```

其中 `speaker_count` 是 `0`（无活动）、`1`（单人）或 `2`（重叠说话）。`flag` 描述
当前 span 的右边界 `end`：它既区分真实的断句候选边界，也允许在长时间无边界时安全地
增量发布已确定的结果。

## 2. 范围和非目标

### 本次范围

- 加载和运行本地 Pyannote segmentation ONNX 模型；
- 使用模型 metadata 驱动窗口长度、窗口步长和帧步长；
- 接收连续小块音频并在对象内维护有界缓存；
- 融合多窗口的 `speaker_count`；
- 融合每窗口内的单人换人候选边界；
- 通过 C++、C API、Python API 输出增量的、已最终确定的 spans；
- 提供 Python 流式测试脚本，以及与现有 offline-long-audio-pipeline 对照的测试报告。

### 明确不做

- 不加载 speaker embedding 模型；
- 不进行 clustering，不输出全局 speaker ID；
- 不把 local mask / local track ID 暴露给上层；
- 不替代现有 `OfflineSpeakerDiarization`；
- 不下载模型、Python 包或其他依赖；不创建新的 Python 环境；
- 不新增 diart 集成或流式 diarization 验证。

## 3. 模型约束和时间网格

第一阶段使用本地模型：

```text
D:\TransAI\audio_model\full\audio_models\speaker_segmentation\sherpa-onnx-pyannote-segmentation-3-0\model.onnx
```

当前模型 metadata 为：

| 字段 | 值 | 用途 |
|---|---:|---|
| `sample_rate` | 16000 Hz | API 输入采样率 |
| `window_size` | 160000 samples / 10 s | 每次 ONNX 输入窗口 |
| `window_shift` | 16000 samples / 1 s | 相邻模型窗口步长 |
| `receptive_field_shift` | 270 samples / 16.875 ms | 融合和 span 边界网格 |
| `num_speakers` | 3 | 每窗口 local track 数 |
| `num_classes` | 7 | powerset 输出类别数 |
| `powerset_max_classes` | 2 | 一帧最多两个活跃 speaker |

实现不得把 `10 s` 或 `1 s` 写死为算法常量；窗口和帧参数从 metadata 读取。第一版
仅支持与 Pyannote 3.0 兼容的 3-track / 7-class / max-2-speaker powerset metadata，遇到
其他模型应在创建对象时明确报错。

span 的边界位于由 `receptive_field_shift` 定义的 0 起始帧网格上，并使用半开区间
`[start, end)`；时间单位为秒（C++/C）和 Python `float` 秒。最后一个 span 的 `end`
不超过实际接收音频时长。

## 4. 对外结果语义

### 4.1 唯一输出类型

```text
[start, end, speaker_count, flag]
```

- `start`、`end`：相对于本次对象 session 起点的秒数；
- `start < end`；
- 全部结果按 `start` 严格递增；
- 两个相邻结果不得时间重叠；
- `speaker_count ∈ {0, 1, 2}`；
- 除 `InputFinished()` 的末尾补零窗口外，已输出 span 的时间、人数和 flag 不会被后续输入
  修改或撤回；
- `flag` 属于 span 的右边界 `end`，不属于 `start`。

### 4.2 `flag` 定义

`flag` 为位掩码；首版公开以下值：

```text
0x00  CONTINUE
0x01  SPEAKER_COUNT_CHANGED
0x02  SINGLE_SPEAKER_CHANGED
0x04  INPUT_FINISHED
```

语义如下：

| flag | `end` 的语义 | 上层处理建议 |
|---|---|---|
| `CONTINUE` | 只是已最终确定的增量发布位置，不是 segmentation 断句边界 | 与后续 span 继续拼接 |
| `SPEAKER_COUNT_CHANGED` | `end` 前后人数不同，例如 `1→2`、`2→1`、`1→0` | 作为人数/overlap/静音状态变化边界处理 |
| `SINGLE_SPEAKER_CHANGED` | `end` 前后人数均为 `1`，但多窗口确认了单人换人候选 | 作为上层 ASR 断句候选 |
| `INPUT_FINISHED` | `end` 为实际输入音频终点 | flush 当前上层 pending 数据；该 bit 可与其他 bit 组合 |

`SPEAKER_COUNT_CHANGED` 与 `SINGLE_SPEAKER_CHANGED` 在首版不会同时设置：后者只适用于
`1→1`，前者适用于人数变化。`INPUT_FINISHED` 可与其余 bit 组合。

### 4.3 增量发布约定

当某一时间区域已不可能被未来滑窗覆盖时，它可以立即入结果队列。若区域中没有真实
segmentation 边界，仍按不超过一个 `window_shift` 的已确定区间输出，并设置
`flag = CONTINUE`。

例如单人从 `0 s` 持续讲话到 `40 s`，对当前 10 s window / 1 s shift 模型，结果将逐步
包含：

```text
[0, 1, 1, CONTINUE]
[1, 2, 1, CONTINUE]
[2, 3, 1, CONTINUE]
...
```

因此上层约 10 秒后就能获得稳定的人员数，不需要等待 40 秒讲话结束。上层只能在
`flag != CONTINUE` 的右边界做 segmentation 建议断句；`CONTINUE` span 是可安全拼接的
进度结果。

例如在 `5 s` 确认单人换人，前一条覆盖该位置的 span 应为：

```text
[4, 5, 1, SINGLE_SPEAKER_CHANGED]
```

后续 `[5, 6, 1, CONTINUE]` 仍可增量输出。

## 5. 融合算法

### 5.1 每窗口推理和 powerset 解码

当对象缓存到从某个窗口起点开始的完整 `window_size` 样本时，执行一次 ONNX 推理。
每帧对 7-class logits 做 softmax / argmax，解码为三条 local binary track：

```text
class 0: 000       class 1: 100       class 2: 010       class 3: 001
class 4: 110       class 5: 101       class 6: 011
```

每帧的窗口内人数为 local mask 的 popcount，即 `0`、`1` 或 `2`。

窗口按 metadata `window_shift` 递进。模型推理可在一次 `AcceptWaveform()` 中批量处理
多个已经完整的窗口，但必须按时间顺序将推理结果写入融合器。

### 5.2 speaker_count 融合

这部分复用现有 `OfflineSpeakerDiarization::ComputeSpeakersPerFrame()` 的语义，而不复用
其 embedding/clustering 流程：

1. 将每个窗口的帧映射到 session 全局帧轴；
2. 将该窗口每帧 local mask 的 popcount 累加到 `count_sum[frame]`；
3. `coverage[frame] += 1`；
4. 融合人数为：

```text
fused_count[frame] = round(count_sum[frame] / coverage[frame])
```

因此，local bit 在两个窗口间互换不会影响 overlap 判断。

### 5.3 窗口内 local track 稳定化和单人换人候选提取

不尝试对齐不同窗口的 local bit。每个窗口仅在自身 local track 语义有效的 10 秒范围内
进行稳定化，复用现有 diarization 的参数命名和默认值：

```text
min_duration_on  = 0.30 s
min_duration_off = 0.50 s
```

对每条 local track：

1. 连续 active island 短于 `min_duration_on` 的，不作为稳定活动；
2. 同一 local track 两个 active island 之间的 inactive gap 短于 `min_duration_off` 的，
   在该窗口内闭合为连续活动；
3. 该处理只用于提取换人证据，**不直接改写最终 fused speaker_count 时间线**。

随后在每个窗口独立判断边界 `t` 前后是否满足：

```text
左侧：稳定后持续单人，唯一活跃 local bit 为 A；
右侧：稳定后持续单人，唯一活跃 local bit 为 B；
A != B。
```

满足时，产生无方向、无身份语义的候选事件：

```text
single_speaker_change_candidate(t) = 1
```

例如 W0 报告 `bit0 -> bit1`、W1 报告 `bit1 -> bit0`，二者都是对同一时间点的
“单人换人”支持票，而不是互相冲突的事件。

`min_duration_on` 同时保证边界两侧不是瞬时 local-track 抖动；`min_duration_off` 仅消除
单窗口、同一 local track 的短暂掉帧。由于本模块没有 global speaker ID，绝不能用
`min_duration_off` 擅自跨最终 `count 1 -> count 0 -> count 1` 时间线合并，否则可能把
`A -> 短静音 -> B` 错误视为同一人。

### 5.4 跨窗口 change-boundary voting

每个候选事件投影到全局帧轴。为容纳不同窗口的少量帧级定位差，候选事件对以该帧为
中心、半径为一个 `receptive_field_shift` 的全局帧邻域投票。

对每个全局帧维护：

```text
change_vote[frame]      # 支持该位置附近存在单人换人的窗口数
change_coverage[frame]  # 覆盖且能够观察该位置左右稳定区的窗口数
```

确认规则：

```text
change_vote / change_coverage >= change_vote_threshold
```

第一版默认：

```text
change_vote_threshold = 0.50
```

即在可观察该边界的窗口中，至少半数支持该边界。若只有一个有效窗口覆盖（例如流开头
或 `InputFinished()` 后的尾部），该窗口的有效候选可确认；这保证短音频仍可产出结果。

只保留局部最大值；两个确认事件距离小于 `min_duration_on` 时，仅保留支持率更高的一个。
这能把不同窗口的 `4.98 s`、`5.00 s`、`5.03 s` 聚合为一个边界。

### 5.5 输出 span 构造和 flag 赋值

最终边界候选有三类：

1. `fused_count` 改变的位置；
2. 已确认的、且边界两侧 `fused_count == 1` 的单人换人事件；
3. 已最终确定区域的增量发布 checkpoint。

按边界拆分出 `[start, end, speaker_count, flag]`：

- count 改变：前一个 span 的 `flag |= SPEAKER_COUNT_CHANGED`；
- 确认单人换人：前一个 span 的 `flag |= SINGLE_SPEAKER_CHANGED`；
- 仅因输出进度 checkpoint 拆分：前一个 span 的 `flag = CONTINUE`；
- `InputFinished()` 的实际音频末尾：最后一个 span 的 `flag |= INPUT_FINISHED`。

count 相同的内部帧可合并直到遇到真实边界或增量发布 checkpoint。checkpoint 只服务于
低延迟结果消费，不能被解释为断句。所有最终输出 span 都不重叠，且 `end` 恰好等于下一
span 的 `start`。

## 6. 流式缓存、最终性和延迟

### 6.1 输入

`AcceptWaveform()` 接收任意正长度的一维 `float32` PCM 块，典型块大小为 32 ms / 512
samples（16 kHz）。输入必须是模型要求的采样率；对象不重采样。

每个对象拥有独立的模型会话、音频缓存、帧融合器、pending 输出和结果队列；不存在
全局音频缓存或全局 session 状态。

### 6.2 有界缓存

对象只保留：

- 下一次模型窗口所需的未丢弃音频，约一个 `window_size`；
- 仍可能被未来窗口覆盖的帧级 count / change 投票数据，约一个模型窗口；
- 尚未发布的帧及很小的结果队列。

已经最终确定且已经发布的时间区域会被丢弃，因此连续长音频不会按总时长线性增长内存。

### 6.3 何时结果可最终输出

某个时间帧只能在“所有还可能覆盖它的滑窗都已推理”后输出。例如当前模型为 10 s
窗口、1 s 步长：

```text
输入达到 10 s：可最终确定并发布 [0 s, 1 s)；
输入达到 11 s：可最终确定并发布 [1 s, 2 s)；
输入达到 15 s：可最终确定并发布 [5 s, 6 s)。
```

因此稳定输出的算法延迟约为一个模型窗口，即约 10 秒，而不是额外等待 9 个窗口。

若当前已确定区间内没有 count change 或 confirmed single-speaker-change，仍会发布
`CONTINUE` span；不会等待一段长讲话的右边界出现。调用方可在保证时间顺序的同时实时
获得 activity / overlap 结果。

### 6.4 结束输入和尾部

`InputFinished()` 表示当前 session 不再接收音频：

- 处理所有能由实际音频组成的完整窗口；
- 按现有 offline diarization 的尾窗语义，必要时建立一个从下一个 `window_shift` 起点
  开始的零填充窗口；
- 只保留实际音频时长内的 frames / spans；
- flush 所有最终帧和结果队列；最后一个 span 设置 `INPUT_FINISHED`；
- 在 `Reset()` 前拒绝新的 `AcceptWaveform()`。

`Reset()` 清空当前 session 的缓存、融合状态、结果队列和时间基准，但保留已经加载的
模型会话，可用于下一条音频流。

## 7. C++ API

新增独立类型，不修改或替代 `OfflineSpeakerDiarization`：

```cpp
enum SpeakerSegmentationSpanFlag : int32_t {
  kSpeakerSegmentationContinue = 0,
  kSpeakerSegmentationSpeakerCountChanged = 1 << 0,
  kSpeakerSegmentationSingleSpeakerChanged = 1 << 1,
  kSpeakerSegmentationInputFinished = 1 << 2,
};

struct SpeakerSegmentationConfig {
  OfflineSpeakerSegmentationModelConfig model;
  float min_duration_on = 0.30f;
  float min_duration_off = 0.50f;
  float change_vote_threshold = 0.50f;

  bool Validate() const;
};

struct SpeakerSegmentationSpan {
  float start;
  float end;
  int32_t speaker_count;
  int32_t flag;
};

class SpeakerSegmentation {
 public:
  explicit SpeakerSegmentation(const SpeakerSegmentationConfig &config);

  int32_t SampleRate() const;
  void AcceptWaveform(const float *samples, int32_t n);
  void InputFinished();
  bool Empty() const;
  const SpeakerSegmentationSpan &Front() const;
  void Pop();
  void Reset();
};
```

约束：

- `Front()` 只能在 `Empty() == false` 时调用；
- `Front()` 返回的引用在下一次任意对象方法调用前有效；
- `InputFinished()` 幂等；
- `AcceptWaveform()` 在 `InputFinished()` 后应记录错误并拒绝数据，调用方需先 `Reset()`；
- `speaker_count` 不承诺、更不映射为 global speaker ID；
- `flag` 只描述 `end` 边界，`CONTINUE` 不可被解释为断句。

## 8. C API

C API 采用现有 VoiceActivityDetector 的结果队列模式，类型前缀为
`SherpaOnnxSpeakerSegmentation`：

```c
typedef enum SherpaOnnxSpeakerSegmentationSpanFlag {
  SherpaOnnxSpeakerSegmentationContinue = 0,
  SherpaOnnxSpeakerSegmentationSpeakerCountChanged = 1 << 0,
  SherpaOnnxSpeakerSegmentationSingleSpeakerChanged = 1 << 1,
  SherpaOnnxSpeakerSegmentationInputFinished = 1 << 2,
} SherpaOnnxSpeakerSegmentationSpanFlag;

typedef struct SherpaOnnxSpeakerSegmentationConfig {
  SherpaOnnxOfflineSpeakerSegmentationModelConfig model;
  float min_duration_on;
  float min_duration_off;
  float change_vote_threshold;
} SherpaOnnxSpeakerSegmentationConfig;

typedef struct SherpaOnnxSpeakerSegmentationSpan {
  float start;
  float end;
  int32_t speaker_count;
  int32_t flag;
} SherpaOnnxSpeakerSegmentationSpan;

typedef struct SherpaOnnxSpeakerSegmentation SherpaOnnxSpeakerSegmentation;

const SherpaOnnxSpeakerSegmentation *SherpaOnnxCreateSpeakerSegmentation(
    const SherpaOnnxSpeakerSegmentationConfig *config);
void SherpaOnnxDestroySpeakerSegmentation(
    const SherpaOnnxSpeakerSegmentation *segmentation);

int32_t SherpaOnnxSpeakerSegmentationGetSampleRate(
    const SherpaOnnxSpeakerSegmentation *segmentation);
void SherpaOnnxSpeakerSegmentationAcceptWaveform(
    const SherpaOnnxSpeakerSegmentation *segmentation,
    const float *samples, int32_t n);
void SherpaOnnxSpeakerSegmentationInputFinished(
    const SherpaOnnxSpeakerSegmentation *segmentation);
int32_t SherpaOnnxSpeakerSegmentationEmpty(
    const SherpaOnnxSpeakerSegmentation *segmentation);
const SherpaOnnxSpeakerSegmentationSpan *SherpaOnnxSpeakerSegmentationFront(
    const SherpaOnnxSpeakerSegmentation *segmentation);
void SherpaOnnxSpeakerSegmentationPop(
    const SherpaOnnxSpeakerSegmentation *segmentation);
void SherpaOnnxSpeakerSegmentationReset(
    const SherpaOnnxSpeakerSegmentation *segmentation);
```

`Front()` 的返回指针由对象所有；调用方不得释放，且必须在下一次该对象 API 调用前复制
需要的数据。C API 不暴露 local mask、raw logits、窗口结果或 embedding。

## 9. Python API

pybind11 暴露与 C++ 同名的类：

```python
config = sherpa_onnx.SpeakerSegmentationConfig()
config.model.pyannote.model = model_path
config.model.num_threads = 4
config.min_duration_on = 0.30
config.min_duration_off = 0.50
config.change_vote_threshold = 0.50

segmenter = sherpa_onnx.SpeakerSegmentation(config)
assert segmenter.sample_rate == 16000

for samples in pcm_blocks_32ms:
    segmenter.accept_waveform(samples)
    while not segmenter.empty():
        span = segmenter.front
        # span.start, span.end, span.speaker_count, span.flag
        segmenter.pop()

segmenter.input_finished()
while not segmenter.empty():
    span = segmenter.front
    segmenter.pop()
```

Python `SpeakerSegmentationSpan` 提供只读属性：`start`、`end`、`speaker_count`、`flag`。
`front` 返回 Python 值对象（复制），不暴露 C++ 引用生命周期问题。耗时的
`accept_waveform()` 和 `input_finished()` 释放 GIL。

## 10. 测试与验证

### 单元测试

不引入 pytest；使用当前环境可运行的 Python `unittest`，以及项目现有 C++ 测试方式。
重点覆盖：

1. powerset decode 与每帧 popcount；
2. 多窗口 count 融合等价于现有 `ComputeSpeakersPerFrame()`；
3. W0 为 `bit0 -> bit1`、W1 为 `bit1 -> bit0` 时，融合为一个相同时间的单人换人边界；
4. local track 的短 active island 被 `min_duration_on` 抑制；同一 local track 的短 inactive
   gap 被 `min_duration_off` 闭合；二者均不直接重写 fused count 时间线；
5. 只有一个窗口报告的短抖动，不满足稳定时长或投票阈值时不产生换人边界；
6. overlap（`count == 2`）正确输出；
7. `CONTINUE` 允许长时间无断句时按最终确定区域增量发布；
8. `SPEAKER_COUNT_CHANGED`、`SINGLE_SPEAKER_CHANGED` 和 `INPUT_FINISHED` 的 end-boundary
   flag 语义与组合规则；
9. 32 ms 分块输入与一次性输入得到相同最终 frames、边界和 flag 序列；
10. `InputFinished()` 的尾窗补零、实际音频长度裁剪和 `Reset()` 复用；
11. 多对象状态隔离。

### 远程 Linux 编译与集成验证

实现提交后，远程 Linux 是 Python 集成测试的主验证环境。严格按照
`sherpa-onnx-remote-build-test` 固定环境执行：本地当前分支正常（非 force）推送到
`jimmy`，远端仓库仅使用 `/speech_store/jimmy/k2_origin/sherpa-onnx/`，确认远端工作树
干净、分支同名且可 `git pull --ff-only` 后，在其 `build/` 目录配置并编译最新提交。

Windows 的 PCM 测试文件可复制到用户授权的专用目录：

```text
/speech_store/jimmy/ai_testset/pyannote-segmentation-streaming-api/input/
```

远端测试结果只写入同一专用目录的 `output/` 子目录，完成后复制回：

```text
C:\Users\admin\Downloads\pyannote-segmentaion测试
```

不下载模型、依赖或 Python 包，也不将模型复制到远端。远程执行前必须以只读检查确认
远端已经存在可读取的 `sherpa-onnx-pyannote-segmentation-3-0/model.onnx`；若不存在，停止
远程集成测试并向用户报告缺失路径，不自动传输或下载模型。远端固定构建命令中的
`SHERPA_ONNX_ENABLE_C_API=OFF` 用于 Python 集成验证；C API 的编译/API 兼容性由新增的
本地 C/C++ 构建测试覆盖，不能把远程 Python 通过误报为 C API 已验证。
### Python 测试脚本和报告

新增一个类似 `python-jimmy/offline-long-audio-pipeline-asr-speaker.py` 的脚本：

- 以 32 ms PCM 块调用新 Python API；
- 使用本地模型，不下载任何模型；
- 从测试目录递归处理 PCM；
- 保存原始 span、分块时序、换人 `1 -> 1` 边界、overlap 区间和运行统计；
- 与 `offline-long-audio-pipeline-asr-speaker.py` 输出对照：WER、最终片段数量、标注文件
  中 multi/overlap 片段及断开情况。

输入目录：

```text
\\10.88.0.243\xainas\ProMax\测试集\音频测试集\中文\内部会议\双人安静咨询\promax
```

输出和报告目录：

```text
C:\Users\admin\Downloads\pyannote-segmentaion测试
```

若当前 Python 环境缺少测试所需模块、或测试步骤需要下载/安装任何内容，必须先停止并向
用户确认；不得自行安装、下载或新建环境。

## 11. 与既有实现的关系

现有 `OfflineSpeakerDiarization` 继续用于“segmentation + embedding + clustering + global
speaker ID”的完整离线 diarization。它的 `ComputeSpeakersPerFrame()` 可作为本设计的
count 融合参考，但不能直接提供单人换人边界，因为它只融合每帧 active speaker 数，
不保留 local mask 的变化证据。

本设计不采用 Python offline-long-audio-pipeline 当前的 local-track permutation matching
和 7-class probability remap 作为对外结果语义；它只借鉴该 pipeline 的 powerset decode、
滑窗和时间轴处理方式。新对象内部的 local mask 仅用于生成 identity-free 的换人投票。
