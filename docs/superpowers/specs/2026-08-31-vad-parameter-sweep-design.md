# VAD 参数递进搜索测试设计

## 1. 背景与目标

在分支 `v1.13.2_transai_vad` 上，使用 `python-jimmy/pipeline-vad-asr-wer.py` 对 ProMax 中文长音频测试集进行 VAD 参数搜索。目标是在保证四个录音整体识别效果稳定的前提下，找到 WER 最低的参数组合。

测试集目录：

`/speech_store/jimmy/ai_testset/promax录音v1`

测试集包含 4 个 `.pcm` 音频，每个音频有同名 `_label.txt` 参考文本。PCM 按脚本约定为 16 kHz、单声道、S16LE。

本方案不一次性运行全部参数笛卡尔积，而是采用逐阶段锁定参数的递进搜索。最多运行 40 次完整的 VAD + ASR + WER 流程，避免重复计算。

## 2. 固定配置

- VAD 模型：`/speech_store/jimmy/ai_models/silero_vad.onnx`
- ASR 模型：`/speech_store/jimmy/ai_models/asr/sherpa-onnx-seaco-paraformer-zh-20260824`
- 语言：`zh`
- VAD threshold：`0.5`
- `min_speech_duration`：`0.25` 秒
- `max_speech_duration`：`20.0` 秒
- 短句合并的 `short_segment_duration`：脚本默认 `6.0` 秒
- 短句合并的 `max_merged_duration`：脚本默认 `30.0` 秒
- 每个音频均使用其同名 label 文件计算 WER

未启用合并时，不传 `--merge-gap-duration`，脚本使用 `k2-vad_cut.py`。启用合并时传入 `--merge-gap-duration`，脚本使用 `k2-vad-cut-merge.py`。

## 3. 四阶段测试流程

### 阶段一：基线

对全部 4 个音频运行：

- `pre_speech_pad_duration=0.0`
- `min_silence_duration=0.7`
- 不启用短句合并

记录每个音频的 WER、参考词数、错误词数、删除数和插入数，作为后续比较基线。

运行配置数为 1，覆盖 4 个音频，共 4 次完整运行。

### 阶段二：搜索预留时长

固定 `min_silence_duration=0.7`，测试：

- `pre_speech_pad_duration=0.1`
- `pre_speech_pad_duration=0.2`
- `pre_speech_pad_duration=0.3`

阶段一的 `pre_speech_pad_duration=0.0` 作为比较项保留。四个配置均覆盖全部音频，共 16 次累计运行，其中阶段二新增 12 次。

选择阶段二的最优预留时长，作为阶段三和阶段四的固定参数。

### 阶段三：搜索最小静音时长

固定阶段二选出的 `pre_speech_pad_duration`，测试：

- `min_silence_duration=0.8`
- `min_silence_duration=0.9`
- `min_silence_duration=1.0`

阶段一的 `min_silence_duration=0.7` 作为比较项保留。四个配置均覆盖全部音频，共 16 个配置结果，其中阶段三新增 12 次运行。

选择阶段三的最优静音时长，作为阶段四的固定参数。

### 阶段四：搜索短句合并间隔

固定阶段二和阶段三选出的参数，测试：

- `--merge-gap-duration=0.8`
- `--merge-gap-duration=1.0`
- `--merge-gap-duration=1.2`

阶段三的“不合并”结果作为比较项保留。四个配置均覆盖全部音频，共 16 个配置结果，其中阶段四新增 12 次运行。

最终选择合并间隔，并形成完整的最终参数组合：

`min_silence_duration` + `pre_speech_pad_duration` + 合并模式/`merge_gap_duration`

## 4. 结果评价与选择规则

### 必须记录的指标

每个参数配置都必须记录以下结果：

1. 4 个音频各自的 WER，明确标注音频名称；
2. 4 个音频合并后的整体 WER；
3. 参考词数、错误词数、删除数和插入数，分别记录单音频值和整体汇总值。

单音频 WER 用于定位某个录音是否出现异常，不能用单音频 WER 的简单平均替代整体 WER。

### 主指标与最终选择

以四个音频合计计算的整体 WER 作为唯一的参数选择主指标：

`overall_wer = 100 * sum(err_words) / sum(ref_words)`

使用错误词数和参考词数重新计算，避免对不同长度音频做简单平均造成偏差。每个阶段选择整体 WER 最低的候选；第四阶段最终推荐也只以整体 WER 最低为准。

### 稳定性检查

虽然最终判断使用整体 WER，但报告必须展示每个音频的 WER。若整体 WER 最低的配置导致某个音频明显异常，应在报告中标注该现象；除非用户另行确认，不用单音频结果替换整体 WER 的排序结论。

### 阶段选择

每个阶段只在本阶段候选项和上一阶段已选配置之间比较。阶段选择完成后，后续阶段不重新展开已淘汰参数，以控制运行数量并保持搜索路径可解释。

## 5. 输出目录与文件

结果根目录：

`/speech_store/jimmy/temp/vad_test`

每次测试使用独立的运行子目录，例如：

`/speech_store/jimmy/temp/vad_test/v1.13.2_transai_vad_<commit>_<date>/`

目录结构约定：

```text
<run>/
  stage1-baseline/
  stage2-pre-speech-pad/
  stage3-min-silence/
  stage4-merge-gap/
  all_results.tsv
  stage_summary.tsv
  vad_parameter_report.md
  command.txt
```

每个阶段目录下按音频名称分目录；每次单次运行保留脚本生成的 `segments/`、`asr.txt`、`merged.txt` 和 `wer_detail.txt`。汇总文件至少包含：阶段、音频、`min_silence_duration`、`pre_speech_pad_duration`、`merge_gap_duration`、单音频 WER、整体 WER、参考词数、错误词数、删除数、插入数和明细文件路径。整体汇总行的音频字段固定为 `ALL`，用于明确区分单音频记录和整个测试集结果。

`vad_parameter_report.md` 需要包含：

1. 分支、commit、模型和数据集信息；
2. 四个阶段的候选参数、每个音频的 WER 和每个候选的整体 WER；
3. 每个阶段选中的参数及选择依据；
4. 最终推荐组合和整体 WER；
5. 异常音频、失败运行或中断信息。

## 6. 运行与中断策略

- 运行前确认远端仓库处于目标分支且工作树干净，并确认当前 commit。
- 每个阶段完成后立即写入阶段汇总，避免中断后丢失已完成结果。
- 单次运行失败时停止当前阶段，保留日志和已完成结果，不自动重试或覆盖目录。
- 用户中断时停止子进程，保留部分输出；下次运行使用新的运行子目录，不覆盖旧结果。
- 本设计只生成测试结果和报告，不修改测试集、模型或仓库源代码。

## 7. 运行次数核算

| 阶段 | 配置数（含比较项） | 音频数 | 累计完整运行数 | 本阶段新增 |
|---|---:|---:|---:|---:|
| 阶段一：基线 | 1 | 4 | 4 | 4 |
| 阶段二：预留时长 | 4 | 4 | 16 | 12 |
| 阶段三：最小静音时长 | 4 | 4 | 28 | 12 |
| 阶段四：合并间隔 | 4 | 4 | 40 | 12 |

这里的“配置数”包含上一阶段选中的配置，因此每阶段都能直接比较，不需要从历史文件推断缺失的基线。
