# Pyannote Segmentation 断句对照评测设计

**日期：** 2026-09-17  
**分支：** `v1.13.2_transai_dev`  
**状态：** 已获用户确认，待编写实施计划  
**结果目录：** `C:\Users\admin\Downloads\sherpa-onnx断句优化测试-0917`

## 1. 目标

在相同音频、相同 VAD / ASR / pyannote / 声纹模型、相同非 pyannote 参数和相同自动聚类参数下，对比两种 sherpa-onnx 长音频处理方式的断句与说话人效果；并以 Diart 的开源实现作为不运行推理的第三种融合/断句逻辑对照。

本评测要回答：在两条中文内部会议音频上，哪一种 pyannote 断句方式更准确，以及其对 ASR WER 与说话人指标的影响。

## 2. 对照范围

### 2.1 方式一：现有离线 pyannote 流程

入口：

```text
python-jimmy/offline-long-audio-pipeline-asr-speaker.py
```

采用当前基线流程：Silero VAD、pyannote activity 切分、ASR、Titanet speaker embedding 和聚类。

### 2.2 方式二：Sherpa-ONNX 流式 SpeakerSegmentation

入口：

```text
python-jimmy/offline-long-audio-pipeline-asr-speaker-segmentation.py
```

该入口复用方式一的 VAD、ASR、声纹、聚类、输出及其配置；唯一替换项为 pyannote activity / 单人换人边界提供者。它通过新建的 `SpeakerSegmentation` 流式 API 输出 `[start, end, speaker_count, flag]` spans，再映射回方式一的 VAD 时间线。

模型 metadata 的窗口长度为 10 秒、窗口步长为 1 秒；中间时刻可被约 10 个重叠窗口覆盖，新的实现负责融合这些覆盖窗口的说话人数与单人换人候选边界。`--segmentation-chunk-ms=32` 仅决定向流式 API 送入 PCM 的块大小，不改变模型的 10 秒 / 1 秒推理窗口定义。

### 2.3 方式三：Diart 逻辑分析（不实测）

不在本次两条音频上运行 Diart，不将其纳入 WER、DER 或断句命中率数值排名。

实施时克隆并记录 Diart 的固定上游 commit；分析以下代码路径和行为：

- rolling audio buffer、推理步长和 latency；
- local segmentation tracks 的聚合；
- overlap-aware embedding weighting；
- online/incremental clustering 及 cannot-link 约束；
- speaker turn / segment 发射条件。

报告必须将 Diart 与方式二的职责边界明确区分：Diart 把局部轨道聚合与在线全局聚类连接在同一在线流程中；方式二只负责融合 speaker count / 单人换人边界，随后继续使用现有 Titanet 及聚类流程。

## 3. 固定输入与公平性约束

### 3.1 测试数据

| ID | PCM | 标注 |
|---|---|---|
| `23_asr_1782715267098` | `\\10.88.0.243\xainas\ProMax\测试集\音频测试集\中文\内部会议\双人安静咨询\promax\23_asr_1782715267098.pcm` | `\\10.88.0.243\xainas\ProMax\测试集\音频测试集\中文\内部会议\双人安静咨询\promax\23_asr_1782715267098_label.txt` |
| `asr_1788402212076` | `\\10.88.0.243\xainas\ProMax\测试集\音频测试集\中文\内部会议\信息流投放实习生面试\asr_1788402212076.pcm` | `\\10.88.0.243\xainas\ProMax\测试集\音频测试集\中文\内部会议\信息流投放实习生面试\asr_1788402212076_label.txt` |

输入统一按 PCM s16le、16 kHz、单声道读取。实施前将原始 PCM 和 label 复制到结果目录的受控输入区，并记录 SHA-256；原始共享盘文件不可修改。

### 3.2 强制完全一致的条件

方式一与方式二必须使用：

- 相同的 ASR 模型目录、模型文件、线程数和解码参数；
- 相同的 VAD 模型目录、模型文件与 VAD 参数；
- 相同的 `pyannote-segmentation-3.0/model.onnx`；
- 相同的 Titanet / speaker embedding 模型、线程数及特征处理；
- 相同的聚类算法与所有聚类参数；
- 相同的音频格式、标注、评价脚本和评价参数；
- 相同的代码基线 commit，除方式二依赖的 SpeakerSegmentation 实现本身外不允许混入额外改动。

聚类不固定说话人数量，统一明确传入：

```text
--num-clusters=-1 --cluster-threshold=0.6
```

除上述确认项外，采用两个入口脚本的当前默认值。每项实际解析出的模型路径、模型 SHA-256 和完整命令均需写入结果报告；默认值不可只以“默认”字样代替。

### 3.3 唯一允许的处理差异

- 入口脚本不同；
- 方式二固定额外参数 `--segmentation-chunk-ms=32`，并使用流式 SpeakerSegmentation 的 10 秒窗 / 1 秒步长融合；
- 方式一保持当前离线 pyannote activity 切分实现。

若在运行前发现任一项模型路径、参数、VAD 输出、聚类路径或结果格式不能对齐，停止对照运行，在报告中记录阻塞原因，不能将不公平结果作为对比结论。

## 4. 评测方法

### 4.1 ASR WER

以同一 label 为 reference，对每个方案、每条音频计算 WER，并保存：

- reference 与 hypothesis 文本；
- WER 总值；
- 参考词数；
- 插入、删除、替换错误数；
- 文本合并规则及详细结果。

WER 用于检测断句变化是否使识别文本退化；它不是断句优劣的单一判据。

### 4.2 说话人和断句指标

从下列源文件复制评价脚本及必要测试依赖到受控评测工具目录：

```text
D:\code\proMax\tz-llm-sdk\agent-sdk-test\python\speaker_diarization_metrics.py
```

统一固定：

```text
--collar-ms=500
--boundary-tolerance-ms=500
```

每个方案和每条音频都计算并保存：

- DER；
- 漏检（miss）、误报（false alarm）、说话人混淆（confusion）；
- 标注跨说话人边界数；
- 预测命中数、漏失数及命中率；
- 边界匹配明细。

### 4.3 判定规则

“断句最准确”的排序规则固定如下：

1. 跨说话人断句命中率更高者优先；
2. 命中率相近时，DER 更低者优先；
3. 若仍无明显差异，比较错切与漏切样例、断句前后 ASR 文本连续性及 WER 是否恶化；
4. 两条音频与总体汇总同时给出；若两条音频结论相反，报告为数据集有限、结论不稳定，不宣称单一方案绝对胜出。

## 5. 可复现执行与远程验证

方式二使用了 sherpa-onnx 的 C++/Python API 改动，必须在固定 Linux 服务器构建、运行。执行流程：

1. 在 Windows 检查干净的工作树、完整 diff、分支和目标 commit；
2. 对代码改动仅提交评测所需文件，普通非强制推送到 `jimmy` 的同名分支；
3. 在固定远程仓库确认工作树、分支和 fast-forward 状态后拉取；
4. 使用 Python-enabled shared-library CMake 配置和 `make -j$(nproc)` 完整构建；
5. 在远程 `build` 目录执行两种固定命令；
6. 将原始结果、日志和派生产物取回 Windows 结果目录；
7. 本地以同一份评价工具生成指标、汇总表和中文报告。

发生模型缺失、远程工作树脏、分支不一致、构建失败、推理失败或评价脚本失败时：保留已有日志和原始结果，生成报告且明确写出失败阶段、原始错误和下一步条件；不得用改变模型或参数的重试隐藏问题。

## 6. 结果目录结构

```text
C:\Users\admin\Downloads\sherpa-onnx断句优化测试-0917\
├── 01_input\
├── 02_baseline_offline\
├── 03_streaming_segmentation\
├── 04_diart_source_analysis\
├── 05_comparison\
└── 报告.md
```

每个测试方案目录按音频 ID 分目录保存命令、stdout/stderr、运行日志、元数据、ASR 结果、VAD / segmentation spans（若有）、声纹 / 聚类结果与 WER 明细。`05_comparison` 保存机器可读的 JSON/CSV 汇总、边界差异清单和指标横向表。

## 7. 最终报告内容

最终中文 `报告.md` 必须包含：

1. 评测目标、范围和不纳入评分的 Diart 分析边界；
2. 代码版本、Git 分支、commit、dirty/clean 状态；
3. 两条实际执行命令和逐项参数对照表；
4. VAD / ASR / pyannote / speaker 模型的实际路径、SHA-256 和加载状态；
5. 每条音频的时长、运行耗时、RTF、片段数和 speaker ID 分布；
6. WER 与错误明细；
7. DER、miss、false alarm、confusion、跨说话人断句命中率及边界差异样例；
8. Diart 固定版本、源码逻辑分析及其与两种 sherpa-onnx 方式的架构对比；
9. 基于固定判定规则的结论、置信度、限制与后续建议；
10. 失败和未执行项目（如存在），含可定位日志路径。
