# 会议级长音频评测输出设计

## 背景

`python-jimmy/evaluation/evaluate_long_audio_asr_speaker.py` 当前把多个会议的 WER、DER 产物聚合到同一输出目录，并额外保留 `inputs/`、`asr/`、`speaker/` 子目录、执行日志、manifest 和 status 文件。该布局既混合了多个会议，也暴露了无需交付的中间产物。

本设计仅调整独立评测工具的输出布局；不修改两个长音频 pipeline 脚本的输出职责或其 `result.json` schema。

## 目标

1. 每个发现的 `result.json` / 会议均在输出根目录下拥有一个独立子目录；即使只有一个会议也必须创建子目录。
2. 每个会议目录的文件名固定、目录内扁平化，便于其他工程按固定相对路径读取。
3. 根目录只保留跨会议的 `evaluation_report.md`。
4. 保留说话人边界明细 CSV，删除评测中间输入、日志、manifest、status 和重复 `result.txt` 投影。

## 目录与命名

输入会议的稳定标识为评测工具当前的 `file_id`（由 `segment_id` 推导）。输出路径为：

```text
<output-dir>/
├── evaluation_report.md
├── <file_id-A>/
│   ├── asr.txt
│   ├── wer_detail.txt
│   ├── wer_summary.json
│   ├── segment_asr_detail.xlsx
│   ├── speaker_summary.json
│   └── speaker_diarization_boundary_details.csv
└── <file_id-B>/
    └── ...相同固定文件名...
```

`file_id` 必须是可安全用作单个目录名的值。若两个已发现结果产生相同 `file_id`，工具应在写入前报错，避免会议结果互相覆盖。

## 每会议产物语义

| 文件 | 条件 | 内容 |
| --- | --- | --- |
| `asr.txt` | 始终生成 | 当前会议的识别片段投影，格式保持 `segment_id speaker_id asr_text`，按时间排序。 |
| `wer_detail.txt` | 匹配到标注且 WER 成功 | 当前会议的逐条 WER 明细，由本目录的 `evaluation.py` 生成。 |
| `wer_summary.json` | 匹配到标注且 WER 成功 | 当前会议 WER 汇总，包括错误数、参考 token 数与 WER。 |
| `segment_asr_detail.xlsx` | 匹配到标注且 ASR 片段对齐成功 | 当前会议的 Agent SDK 六列表格、500 ms 对齐 ASR 片段差异。 |
| `speaker_summary.json` | 匹配到标注且 speaker 评测成功 | 原 `speaker_diarization_summary.json` 内容，文件名标准化。 |
| `speaker_diarization_boundary_details.csv` | 匹配到标注且 speaker 评测成功 | 继续保留，用于边界和说话人错误定位。 |

在一个会议没有匹配标注时，仍创建其会议目录并只写 `asr.txt`。该会议不生成 WER、speaker、Excel 产物；根目录报告明确记录其为 `skipped_no_label`。

## 执行与中间文件

每个有标注会议独立执行 WER 与说话人评测。WER label/hyp 输入使用 Python 临时目录，在进程结束时自动删除。子进程 stdout/stderr 不再作为文件保留；失败摘要写入根目录报告，并且任一请求的评测失败时主程序返回非零。

工具不再输出以下路径：

```text
result.txt
evaluation_manifest.json
evaluation_status.json
inputs/
asr/
speaker/
evaluation.stdout.log
evaluation.stderr.log
speaker_diarization.stdout.log
speaker_diarization.stderr.log
speaker_diarization_per_file.json
```

## 根目录汇总报告

`evaluation_report.md` 是唯一根目录产物，按会议列出：

- `file_id`、源 `result.json`、标签文件（若匹配）；
- 会议目录路径；
- `asr.txt` 和可用评测产物；
- WER / DER 核心数值；
- `success`、`skipped_no_label` 或 `failed` 状态与错误摘要；
- 全部会议的计数和可评分会议数。

报告可以展示跨会议汇总指标，但不额外生成全局 `wer_detail.txt`、`wer_summary.json` 或 `speaker_summary.json`，以避免混淆会议级结果。

## 验收标准

1. 单会议和多会议输入都按 `<output-dir>/<file_id>/` 创建目录。
2. 每个有标注会议只含六类约定产物中的成功项，不含子目录与旧中间文件。
3. 无标注会议目录只含 `asr.txt`，根目录报告清楚记录跳过原因。
4. 多会议 WER / DER 互不混合：每个会议都有自己的 detail、summary、Excel 和 CSV。
5. `speaker_diarization_boundary_details.csv` 存在并保持当前 CSV schema。
6. 根目录只输出 `evaluation_report.md`。
7. 现有 `result.json` 解析、WER 计算、DER 计算与 Agent SDK 时间对齐算法不改变。
