# 长音频 ASR 与说话人评测工具包设计

**日期：** 2026-09-21
**状态：** 已确认，待最终审阅与实施计划
**范围：** `python-jimmy/evaluation/`

## 1. 背景与目标

当前两个离线识别脚本：

- `python-jimmy/offline-long-audio-pipeline-asr-speaker.py`
- `python-jimmy/offline-long-audio-pipeline-asr-speaker-segmentation.py`

只应执行长音频识别与说话人分段，并继续输出现有的详细 `result.json`、人工可读 `result.txt` 与运行元数据。它们不应承担 WER、DER、断句指标、Excel、测试报告或机器可读评测结果的职责。

新增一个可独立复制的评测工具目录：

```text
python-jimmy/evaluation/
```

工具包读取识别结果与标准标注，统一生成 ASR、说话人与片段差异评测产物。目录不依赖 Windows 绝对路径；应可在 Linux 远程测试环境独立运行。稳定后，可以整体迁移到 `D:\code\proMax\tz-llm-sdk\agent-sdk-test\python`，作为各长音频测试工程的统一评测实现。

## 2. 不在本次范围内

- 不把评测调用重新嵌回两个 pipeline 脚本。
- 不改变 pipeline 的识别、VAD、Pyannote、聚类或 speaker mask 算法。
- 不改变既有 `result.json` 的紧凑展示约定，也不改变 pipeline 当前人工可读 `result.txt` 的格式或生成行为。评测工具在独立 `--output-dir` 中生成同名、机器可读的 `result.txt`，二者不会覆盖。
- 不在本阶段建立第三方公共 Python package、发布流程或跨仓库安装机制。
- 不修改 `agent-sdk-test` 工作区；仅复用其已经验证的评测协议和实现语义。

## 3. 目录与组件

建议目录结构如下：

```text
python-jimmy/evaluation/
├── evaluate_long_audio_asr_speaker.py
├── evaluation.py
├── speaker_diarization_metrics.py
├── README.md
└── tests/
    ├── test_evaluate_long_audio_asr_speaker.py
    ├── test_evaluation.py
    └── test_speaker_diarization_metrics.py
```

### 3.1 `evaluate_long_audio_asr_speaker.py`

唯一的统一 CLI 入口，负责：

1. 发现一个或多个识别 `result.json`；
2. 校验输入、标签和输出目录；
3. 从 `result.json` 生成统一、公开的 `result.txt`；`*_asr.txt` 仅作为调用既有 speaker 指标脚本所需的内部投影；
4. 生成 WER 的整音频 `wer_label.txt` 和 `wer_hyp.txt`；
5. 调用同目录 `evaluation.py`；
6. 调用同目录 `speaker_diarization_metrics.py`；
7. 生成片段级 ASR 差异 Excel；
8. 写入机器可读状态、日志和 Markdown 报告。

它不实现第二套 WER 或 DER 算法；WER 和说话人评测分别委托给下面两个已有实现。

### 3.2 `evaluation.py`

兼容 `agent-sdk-test/python/evaluation.py` 的 WER 对齐接口：

```text
--label --hyp --language --detail
```

用于写出逐 utterance 的文本对齐明细 `wer_detail.txt`。若可用，使用 `kaldialign`；不可用时保留纯 Python 编辑距离兜底，保证工具包可运行。

### 3.3 `speaker_diarization_metrics.py`

兼容 `agent-sdk-test/python/speaker_diarization_metrics.py` 的说话人指标接口：

```text
--results-dir --labels-dir --output-dir
--boundary-tolerance-ms --collar-ms
```

使用 `pyannote.metrics` 计算 DER，并输出跨说话人边界命中率、逐文件详情和 CSV。其 `multi` / overlap 的 DER 语义保持为 `skip_overlap=True`。

## 4. 输入契约

### 4.1 识别结果

入口接受：

```text
--results-dir <单次 run 目录或包含多个 run 的根目录>
```

递归发现 pipeline 输出的 `result.json`。`result.json` 是评测工具的唯一 pipeline 输入；评测工具负责在自己的输出目录生成 `result.txt`。每个可评测的 `result.json` 至少必须包含：

- 顶层 `audio_name`；
- `segments` 数组；
- 每段 `segment_id`、`duration_ms`、`speaker_id`、`asr_text`；
- 可从 `segment_id` 或显式字段推导的 `start_ms`。

评测脚本只读取 pipeline 的原始 `result.json`，不会修改该文件、pipeline 当前的人工可读 `result.txt` 或任何 pipeline 运行日志；机器可读 `result.txt` 是评测脚本在独立输出目录中新建的产物。

### 4.2 标签

入口接受：

```text
--labels-dir <包含 *_label.txt 的目录>
```

单文件调试可额外支持：

```text
--label <单个 *_label.txt>
```

标签行沿用当前长音频格式：

```text
<file_id>_<start_ms>_<duration_ms> (<speaker_id>) <text>
```

`multi` 是 overlap 标注的 speaker 标识。标签匹配优先使用同名 `<file_id>_label.txt`；只有单音频模式且显式给出 `--label` 时允许使用该参数覆盖自动发现。

### 4.3 可调参数

```text
--language ZH
--boundary-tolerance-ms 500
--collar-ms 500
--output-dir <目录>
```

默认语言为 `ZH`，边界容差及 DER collar 都为 500 ms。输出目录必须由调用者显式指定，避免重复评测时误覆盖既有产物。

## 5. 统一评测结果格式

pipeline 继续在 run 目录生成现有人工可读 `result.txt`。评测入口读取 `result.json` 后，在**独立评测输出目录**根路径生成唯一、公开、可移植的：

```text
result.txt
```

每行格式严格为：

```text
<segment_id> <speaker_id> <asr_text>
```

其中：

```text
segment_id = <file_id>_<start_ms>_<duration_ms>
```

示例：

```text
23_asr_1782715267098_100478_2530 speaker_00 你好请问现在方便沟通吗
23_asr_1782715267098_103008_4281 speaker_01 可以你说
```

评测输出目录内的 `result.txt` 是与 `*_label.txt` 对应的公共文本交换格式，也是后续迁移到 Agent SDK 测试工程时必须保持不变的结果格式。它与 pipeline run 目录内的人类可读同名文件不共享路径，不存在覆盖关系。评测脚本从该格式按 `file_id` 拆分出内部 `inputs/<file_id>_asr.txt`，仅用于调用既有 `speaker_diarization_metrics.py`；这些内部投影不是 pipeline 输出契约。

## 6. WER 口径

### 6.1 整音频 WER

每个音频生成一条 WER utterance：

```text
<file_id> <该音频标签文本按时间顺序拼接>
<file_id> <该音频识别文本按时间顺序拼接>
```

分别写入：

```text
inputs/wer_label.txt
inputs/wer_hyp.txt
```

随后运行：

```text
python evaluation.py --label ... --hyp ... --language ZH --detail .../asr/wer_detail.txt
```

整音频汇总避免将合法的切分边界差异误算为文本错误；分段差异由 Excel 和说话人边界指标表达。

### 6.2 overlap / `multi`

- 整体 ASR WER：保留 `multi` 标注文本，反映真实混音场景的转写表现。
- DER：由 `speaker_diarization_metrics.py` 按 `skip_overlap=True` 处理。
- 单人片段 WER 汇总：不计入 `multi` 标注行；Excel 仍保留该行，`scoring_status=overlap_not_scored`。

## 7. 片段级匹配与 Excel

生成：

```text
asr_segment_diff.xlsx
```

依赖 `openpyxl`。包含至少两个工作表。

### 7.1 `summary`

每音频一行，包含：

- 文件 ID；
- 全局 WER、插入、删除、替换、参考 token 数；
- DER；
- speaker change hit rate；
- 标注片段数、识别片段数、可评分单人片段数；
- 未匹配标注片段数、未匹配识别片段数；
- overlap 标注片段数；
- 各子评测状态。

### 7.2 `segment_details`

每个标注片段一行，包含：

```text
file_id
reference_segment_id
reference_start_ms
reference_end_ms
reference_duration_ms
reference_speaker_id
reference_text
reference_is_overlap
matched_prediction_ids
prediction_start_ms
prediction_end_ms
prediction_speaker_ids
prediction_text
time_overlap_ms
reference_coverage_ratio
prediction_coverage_ratio
match_status
segment_wer
insertions
deletions
substitutions
scoring_status
```

匹配规则：

1. 计算预测片段与标注片段的时间交集；
2. 一个标注段可聚合多条有交集的预测段；
3. 一个预测段可覆盖多条标注段，保留覆盖率和主匹配信息；
4. 无有效交集的标注段标记 `unmatched_reference`；
5. 无有效交集的预测段单独输出为 `unmatched_prediction` 行；
6. `multi` 行标记 `overlap_not_scored`，不进入单人片段 WER 汇总。

## 8. 标准输出目录

```text
<output-dir>/
├── evaluation_manifest.json
├── evaluation_status.json
├── evaluation_report.md
├── result.txt
├── inputs/
│   ├── <file_id>_asr.txt
│   ├── wer_label.txt
│   └── wer_hyp.txt
├── asr/
│   ├── wer_detail.txt
│   ├── wer_summary.json
│   ├── evaluation.stdout.log
│   └── evaluation.stderr.log
├── speaker/
│   ├── speaker_diarization_summary.json
│   ├── speaker_diarization_per_file.json
│   ├── speaker_diarization_boundary_details.csv
│   ├── speaker_metrics.stdout.log
│   └── speaker_metrics.stderr.log
└── asr_segment_diff.xlsx
```

### 8.1 `evaluation_manifest.json`

记录：输入目录、标签目录、输出目录、实际命令、语言、容差、collar、发现的 `result.json`、由评测工具生成的 `result.txt`、标签匹配关系以及评测工具版本。

### 8.2 `evaluation_status.json`

每个子任务独立记录 `success`、`failed` 或 `skipped`，并给出错误摘要与产物路径。不得因某一项失败而静默缺失文件。

### 8.3 `evaluation_report.md`

给人阅读的总览：输入、版本、样本数、ASR WER、DER、speaker change hit rate、失败/跳过原因及所有产物路径。

## 9. 错误处理与退出码

| 条件 | 行为 |
| --- | --- |
| 无匹配 label | 仍生成公共 `result.txt` 与内部 `*_asr.txt`；WER 和 speaker 指标标记 `skipped`。 |
| 缺少 `pyannote.metrics` | 继续完成标准化、WER、Excel；speaker 标记 `failed`。 |
| 缺少 `openpyxl` | 继续完成标准化、WER、speaker；Excel 标记 `failed`。 |
| `evaluation.py` 失败 | 保存 stdout/stderr；WER 标记 `failed`。 |
| 单个音频无效 | 记录该文件错误；继续处理其余音频。 |
| 请求的任一评测项目失败 | 统一入口返回非零；原始 pipeline 结果绝不删除或覆盖。 |

## 10. 验收标准

1. 两个 pipeline 脚本的 `result.json`、当前人工可读 `result.txt` 及运行元数据输出均保持不变；均不新增 WER、DER、Excel 或评测用机器可读 `result.txt` 写入逻辑。
2. 单个 `result.json` 与单个标签可以生成完整标准目录及标准化 `result.txt`。
3. 多个 `result.json` 可以生成聚合 `result.txt`、按音频拆分的内部 `*_asr.txt` 与聚合报告。
4. `wer_detail.txt` 由 `evaluation.py` 产生，且标签 metadata 不进入 WER 文本。
5. speaker 指标文件名、JSON 字段和 CSV 编码兼容 Agent SDK 现有 `speaker_diarization_metrics.py`。
6. Excel 含 `summary` 和 `segment_details`，并可识别 split、merge、unmatched 与 overlap_not_scored。
7. 缺少可选依赖或标签时，存在明确状态与日志，而不是静默少文件。
8. 新增单元测试覆盖标准化、标签解析、WER 输入、时间匹配、`multi` 口径、依赖失败状态与 Excel 字段。
9. 在远程 Linux 服务器完成一次实际 `23_asr_1782715267098` 评测，返回所有生成产物以供人工检查。

## 11. 迁移边界

工具包不依赖 sherpa-onnx pipeline 内部模块；它只依赖公开 `result.json` schema 和 `*_label.txt` 标注格式。迁移到 `agent-sdk-test/python` 时，保持 CLI、评测输出目录中的公共 `result.txt`、输出目录和 JSON/CSV/XLSX schema 不变即可。

迁移前应在两个工程对同一份固定结果与标签运行回归测试，确认 WER、DER、边界命中率和 Excel 汇总一致。
