# 离线长音频 Pipeline 设计方案

本文描述 `python-jimmy/offline-long-audio-pipeline` 的目标、架构、切句规则、声纹与 ASR 行为、测试设计，以及若干真实音频上的最终对照结果。

入口脚本：`python-jimmy/offline-long-audio-pipeline-asr-speaker.py`。

## 1. 目标与约束

产品目标：对一段本地长音频（PCM / WAV，16 kHz）产出**按说话人断句**的结果，每段同时带：

- 说话人 ID（Titanet 聚类 / 质心回填）
- ASR 文本（中文 Paraformer）

硬约束：

- VAD 区间是硬边界：不跨 VAD 合并；最终段首尾贴齐所属 VAD。
- 默认切句模式是 `vad-pyannote`：在 VAD 语音区内用 pyannote 3.0 活动轨切句。`--segmentation-mode vad` 只保留 Silero 段，用于对照。
- 最终段必须同时走声纹和 ASR；不允许只出时间轴、不出文本。
- 不做 Titanet `speaker_turn` 否决切点，也不接入完整 `OfflineSpeakerDiarization`。
- 评测 WER 用文件级全文拼接，而不是按段对齐标注。说话人库存（聚出多少个 ID）与 WER 分开看。

## 2. 目录与模块

| 路径 | 职责 |
|---|---|
| `offline-long-audio-pipeline-asr-speaker.py` | CLI |
| `pipeline.py` | 编排：读音频 → 建运行时 → VAD → 切句 → 声纹 → ASR → 落盘 |
| `audio_io.py` | PCM/WAV 读取、可选段 WAV 写出 |
| `models.py` | 解析本地模型目录并构造 sherpa-onnx 运行时 |
| `vad.py` | Silero 段收集；`SpeechSegment` 领域对象 |
| `segmentation.py` | 本地 pyannote 3.0 ONNX：powerset class → 局部 3 轨 mask |
| `diarization.py` | 平滑、与 VAD 求交、折叠 overlap/unknown、弱切点规则 |
| `speaker.py` | Titanet 嵌入、受控聚类、质心回填 |
| `asr.py` | 逐段 Paraformer |
| `output.py` | `runs/<时间戳>_<label>_<音频名>/` 下的 `result.json` / `result.txt` / `run_metadata.json` |
| `tests/` | 无模型单测 |

## 3. 默认模型与参数

默认模型根目录 `D:\TransAI\audio_model\full\audio_models\`：

| 角色 | 子目录 |
|---|---|
| ASR | `asr/paraformer-zh` |
| VAD | `vad/silero_vad` |
| 声纹 | `speaker/nemo_en_titanet_large` |
| 切分 | `speaker_segmentation/sherpa-onnx-pyannote-segmentation-3-0` |

默认输出根：`C:\Users\admin\Downloads\python语音Pipeline优化`。

| 参数 | 默认 | 含义 |
|---|---:|---|
| `vad_threshold` | 0.5 | Silero 语音门限 |
| `min_silence_duration` | 0.8 s | VAD 静音切段 |
| `min_speech_duration` | 0.25 s | 最短语音 |
| `max_speech_duration` | 25.0 s | 最长语音（超时强制切） |
| `pre_speech_pad_duration` | 0 | 不向前垫语音 |
| `diarization_min_duration_on` | 0.5 s | 同一 mask 轨最短保留岛 |
| `diarization_min_duration_off` | 0.5 s | 同一 mask 轨填缝上限 |
| `cluster_threshold` | 0.5 | FastClustering 阈值（`num_clusters=-1` 时） |
| `min_cluster_duration` | 1.0 s | 固定：只有净时长 ≥1s 的单人段可建簇 |
| `centroid_assignment_similarity_threshold` | 0.5 | 短段/重叠宿主回填质心的下限 |

代码常量（不暴露 CLI）：

| 常量 | 值 | 位置 |
|---|---:|---|
| `MERGE_GAP_MAX_MS` | 2000 | overlap / unknown 并入邻句的最大空隙 |
| `MIN_SINGLE_CUT_MS` | 1000 | **不同 mask** 弱切点：两边净时长都 ≥1s 才留刀 |
| `MIN_SAME_MASK_CUT_MS` | 2000 | **相同 mask**：两边净时长都 ≥2s 才留刀 |
| `POST_OVERLAP_PAD_MS` | 1000 | 声纹跳过 overlap 后再跳过 1s |

## 4. 端到端流程

```
PCM/WAV
  → Silero VAD 语音区间
  → [vad-pyannote] pyannote 活动轨
  → 时间轴解析（平滑 → 与 VAD 求交 → 折叠 → 最终段）
  → Titanet 嵌入 + 受控聚类
  → Paraformer ASR
  → result.json / result.txt / run_metadata.json / run.log
```

`vad` 模式跳过 pyannote，每个 VAD 区间直接成为最终段（`cut_left/cut_right = vad`）。

## 5. 分层心智模型

切句分四层，不要混为一谈。

### 5.1 pyannote 原始轴

窗口内每一帧一个 **powerset class**（互斥，一条轴）：

| class | mask |
|---|---|
| 0 | 无人 |
| 1 / 2 / 3 | 局部轨 A / B / C |
| 4 / 5 / 6 | A+B / A+C / B+C |

这是**局部** 3 轨，不是全文件说话人 ID。对外输出统一使用 bit mask：A=1（0b001）、B=2（0b010）、C=4（0b100），A+B=3、A+C=5、B+C=6。因此 `result.json` 的 `pyannote_mask` 是平滑**前**局部 bit mask 的 RLE，例如 `[5726,4][507,6]…`；这里的第二个数字不再是模型 powerset class index。

### 5.2 按 mask 轨平滑

`smooth_activity_spans`：把 span 按非空 mask 分轨。对**同一条非空 mask 轨**：

1. `min_duration_off`：轨内空隙 ≤ 该值则填缝（防切断，不是 VAD 静音断句）。
2. `min_duration_on`：填缝后仍短于该值的岛丢弃。

空洞 class 0 不参与填缝。重叠 mask（class 4–6）是独立一轨，不是“三个人同时开着”。

### 5.3 与 VAD 求交

每个 VAD 区间内，用平滑后的 span 边界切原子区间，并标 composition：

- `single_speaker`：mask 基数 1
- `overlapped_speakers`：mask 基数 ≥ 2
- `unknown_activity`：无人 / 对不上轨

空洞若夹在可判定的单人轨之间，可回填为单人。短于 `min_duration_on` 的原子并入邻岛。

### 5.4 折叠与留刀（当前默认）

在单个 VAD 内、不跨 VAD：

1. **同 mask 无条件合并**（折叠 overlap 之前）。
2. **任意长度 overlap 并入后句**；没有后句则并前句。空隙 > 2s 不并。宿主 composition 仍为 `single_speaker`，内部保留 `overlap_regions` 供净时长和声纹裁剪使用。
3. **unknown ≤2s** 同样并后句（否则前句）；**>2s** 保持 `unknown_activity`。
4. **同 mask** 在 fold overlap 之后：短岛仍并；**两边净时长（墙钟减 overlap）都 ≥ 2s** 才保留切点。
5. **不同 mask** 的弱单人切点：两边净时长都 ≥ **1s** 才保留；否则短岛优先并后句。

净时长：`duration_ms - overlap_regions`。

示例（Case1 关心的切开）：

- `0026_149406_155132`：`pyannote_mask` 含 `[5726,4]`（局部轨 C）
- `0027_155132_172277`：overlap `155132–155639`（507ms class 6 并入后句）

## 6. 声纹

1. `samples_for_embedding` 跳过 `overlap_regions` **以及 overlap 结束后 1s**。裁剪后为空则**不**回退到整段混音（该段嵌入失败，说话人保持 `unknown`）。
2. 仅 `speaker_composition == single_speaker` 且净时长 ≥ 1s 的段 **建簇**（`is_cluster_eligible`）。重叠宿主因跳过 overlap+1s 后净时长变短，通常不建簇。
3. FastClustering 只吃合格嵌入；簇标签按首次出现稳定成 `speaker_00`、`speaker_01`…
4. 不合格段只与最终质心比余弦：≥ 0.5 回填该 ID，否则 `unknown`。不合格段不能改库存。

说话人库存偏多（安静双人仍可能聚出十余类）是 Titanet + 阈值聚类的已知现象，**不作为 WER 优化目标**。

## 7. ASR 与产物

- 每段独立识别；失败写入 `asr_error`，文本可空。
- `result.json` 每段含：`segment_id`、`time_range`、`duration_class`、`speaker_composition`、`cut_left` / `cut_right`（`vad` | `pyannote`）、`asr_text`、`speaker_id`、邻段相似度、`pyannote_mask`、`clean_spans`、`local_speaker_mask` 及其置信度。`overlap_regions` 仅作为内部解析字段，不再展示在 `result.json`；`clean_spans` 以紧凑字符串输出，例如 `[249214,252399] [252788,254070]`。成功提取 speaker embedding 的片段额外写入结构化 `embedding_audio_spans`，记录原始音频时间轴上实际送入 embedding 提取器的区间；未提取、跳过或失败的片段省略该字段。`clean_spans` 是诊断用 clean-source 区间，不等同于实际 embedding 输入区间。
- `run_metadata.json` 含段数、组成计数、聚类库存、RTF、模型路径。

## 8. 运行方式

```text
python python-jimmy/offline-long-audio-pipeline-asr-speaker.py --audio <pcm或wav>
```

对照纯 VAD：

```text
python python-jimmy/offline-long-audio-pipeline-asr-speaker.py --audio <pcm> --segmentation-mode vad --run-label vad_only
```

依赖：本机已安装 `sherpa-onnx` Python 包与 `onnxruntime`（见 `requirements.txt`）。单测不加载真实模型。

## 9. 测试设计

### 9.1 无模型单测

目录：`python-jimmy/offline-long-audio-pipeline/tests/`。

```text
python -m unittest discover -s python-jimmy/offline-long-audio-pipeline/tests
```

覆盖：

| 层 | 文件 | 要点 |
|---|---|---|
| CLI | `test_entrypoint.py` | 默认 `vad-pyannote`、on/off=0.5、不暴露危险窗口参数 |
| 编排 | `test_pipeline.py` | `min_cluster_duration` 锁定 1.0；模式校验 |
| VAD 对象 | `test_vad.py` | 聚类资格、overlap 后净时长 |
| 切句 | `test_diarization.py` | VAD 硬边、overlap 并后、unknown 规则、同 mask 2s 留刀、不同 mask 1s 弱切、mask RLE |
| 声纹 | `test_speaker.py` | 跳过 overlap+1s、空裁剪不回退、质心回填 |
| 切分 / ASR / IO / 产物 | `test_segmentation.py` 等 | powerset、写出字段 |

### 9.2 真实音频 WER

口径：

1. Pipeline 把各段 `asr_text` 按时间序拼成一篇 hyp。
2. 标注全文拼接为 ref；去掉行内说话人标签 `(hui|leslie|multi|speaker)` 等。
3. `python-jimmy/evaluation/evaluation.py --language zh`（字级 WER）。
4. 主指标是全文 WER，不是 DER。`asr_error` 必须为 0 才承认该次跑数。

固定测试集（NAS：`\\10.88.0.243\xainas\ProMax\测试集\音频测试集\中文\`）：

| 编号 | 场景 | 文件 | 时长 |
|---|---|---|---:|
| Case1 | 内部会议 / 双人安静咨询 | `23_asr_1782715267098.pcm` | ~699 s |
| Case2 | 外部会议 / 深圳展会 | `12_asr_1782805252933.pcm` | ~1388 s |
| Cafe | 外部会议 / 咖啡店1 | `7_asr_1782376553140.pcm` | ~300 s |

实验根目录：`C:\Users\admin\Downloads\python语音Pipeline优化\tests\`。

对照轴：

- **纯 VAD**：`--segmentation-mode vad`（ASR 上限参考）。
- **规则消融**（主要在 Case1）：overlap 并后 + 同 mask 一律并；同 mask 两边净时长 ≥1s 留刀；**当前默认 ≥2s 留刀**。
- Case2 / Cafe 用默认 `vad-pyannote` 与纯 VAD 对照。Case2 最新默认跑是同 mask ≥1s 留刀那次，未再用 2s 阈值重跑（见下表脚注）。

## 10. 测试报告

以下数字均来自已落盘的 `run_metadata.json` 与 `wer/wer_detail.txt`（或三文件对照的 `SUMMARY.json`）。`asr_error` 均为 0。

### 10.1 最终汇总（当前代码默认 vs 纯 VAD）

当前代码默认：`vad-pyannote`，`min_duration_on/off=0.5`，overlap 任意长度并后句，不同 mask 两边净时长 ≥1s 留刀，**同 mask 两边净时长 ≥2s 留刀**。

| 音频 | 纯 VAD 段数 / WER / 说话人 | 默认 vad-pyannote 段数 / WER / 说话人 | Δ WER | 说明 |
|---|---|---|---:|---|
| Case1 双人安静咨询 `23_asr_…7098` | 78 / **9.58** / 9 | **104 / 10.60 / 15** | +1.02 | 当前默认；run `20260911-111915_keep_long_same_mask_cut_2s_case1_…` |
| Case2 深圳展会 `12_asr_…2933` | 58 / **14.20** / 8 | **267 / 17.61 / 81** | +3.41 | 同 mask ≥1s 留刀，**未**用 2s 重跑；run `20260911-110813_keep_long_same_mask_cut_case2_…` |
| Cafe 咖啡店1 `7_asr_…3140` | 36 / **9.87** / 3 | 36 / **9.87** / 3 | 0.00 | 默认几乎不额外切；三文件对照 `20260910-195753` |

结论：

- 三条音频的**全文 ASR 都没有优于纯 VAD**。切句换来说话人边界，WER 有代价。
- 安静双人：过切有限（78→104），WER +1 左右，可接受作为“能切开换人”的代价。
- 展会嘈杂：过切严重（58→267），假换人多，库存膨胀到 81，WER 伤害最大；主因是厅噪上的假活动轨，不是 `asr_error`。
- 咖啡店口述：VAD 已足够，pyannote 几乎不改时间轴，WER 持平。

### 10.2 Case1 规则消融（同一条 pcm）

| 阶段 | 段数 | WER | 聚类说话人 | 备注 |
|---|---:|---:|---:|---|
| 纯 VAD | 78 | **9.58** | 9 | ASR 最好 |
| 同 mask 一律并（overlap 并后） | 95 | 10.03 | 13 | `20260911-091230_merge_same_mask_after_overlap_case1_…` |
| 同 mask 两边净时长 ≥1s 留刀 | 109 | 10.57 | 16 | `20260911-110638_keep_long_same_mask_cut_case1_…` |
| **同 mask ≥2s 留刀（当前）** | **104** | **10.60** | **15** | `20260911-111915_…_2s_case1_…` |

较早三文件对照里，当时默认是 111 段 / WER 10.64 / 17 人；之后 overlap 并后 + 同 mask 规则把段数压到 104。2s 相对 1s：少切 5 段，WER 几乎不变（10.57→10.60），库存 16→15。

### 10.3 Case2 过程对照

| 阶段 | 段数 | WER | 说话人 |
|---|---:|---:|---:|
| 纯 VAD | 58 | **14.20** | 8 |
| 较早 exclusive-merge 后默认 | 276 | 17.61 | 81 |
| 同 mask ≥1s 留刀（表 10.1 所用） | 267 | 17.61 | 81 |

展会 WER 平台在 ~17.6，段数从 276 降到 267 几乎不改善识别。未跑 2s 同 mask 重测；预期对 WER 影响小于对段数的影响。

### 10.4 运行开销（参考）

| 音频 | 时长 | 默认总耗时 | RTF |
|---|---:|---:|---:|
| Case1 | 699 s | ~81 s | ~0.12 |
| Case2 | 1388 s | ~181 s | ~0.13 |
| Cafe | 300 s | ~32 s | ~0.11 |

耗时主要在 ASR，其次 segmentation 与 Titanet。

## 11. 已知限制与非目标

- pyannote 局部 3 轨在嘈杂厅会假换人；本方案用 VAD 硬边 + 2s 同 mask 留刀抑制，**不**用声纹否决切点。
- 聚类库存不是说话人数估计器；产品若需要“恰好 N 人”，应另设 `num_clusters` 或后处理，而不是改切句。
- 重叠语音并入后句后，宿主仍标 `single_speaker`；ASR 听的是含重叠波形，声纹则尽量躲开重叠。
- 已移除仅用于人工听音的入口：`offline-long-audio-pipeline-export-review-wavs.py`、`offline-long-audio-pipeline-vad-pyannote-review.py`、`offline-long-audio-pipeline-titanet-speaker-turn.py`。
