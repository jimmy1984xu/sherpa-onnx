# 会议级长音频评测输出 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将统一长音频评测工具改为按会议独立目录输出固定命名的最终评测产物，并在根目录仅保留跨会议汇总报告。

**Architecture:** `evaluate_long_audio_asr_speaker.py` 将每个 `ResultRecord` 作为独立评测单元：先创建 `<output-dir>/<file_id>/asr.txt`，如有匹配标签则在该会议目录内独立运行 WER、speaker 和片段 Excel。WER 的 label/hyp 输入使用 `tempfile.TemporaryDirectory`，不会写入交付目录。所有会议的状态和指标仅在根目录 `evaluation_report.md` 汇总。

**Tech Stack:** Python 3.10 标准库、现有 `evaluation.py`、`speaker_diarization_metrics.py`、`asr_segment_detail_diff.py`、`unittest`。

---

### Task 1: 以输出目录契约替换集成测试

**Files:**
- Modify: `python-jimmy/evaluation/tests/test_evaluate_long_audio_asr_speaker.py:RunnerIntegrationTest`
- Test: `python-jimmy/evaluation/tests/test_evaluate_long_audio_asr_speaker.py`

- [ ] **Step 1: 写入单会议固定目录的失败测试**

将当前断言 `output/result.txt`、`output/asr/`、`output/speaker/`、manifest 和 status 的测试改为：

```python
meeting_dir = output / "meeting"
self.assertEqual(
    {path.name for path in meeting_dir.iterdir()},
    {
        "asr.txt",
        "wer_detail.txt",
        "wer_summary.json",
        "segment_asr_detail.xlsx",
        "speaker_summary.json",
        "speaker_diarization_boundary_details.csv",
    },
)
self.assertEqual(
    {path.name for path in output.iterdir()},
    {"evaluation_report.md", "meeting"},
)
```

并断言 `speaker_summary.json` 和 `speaker_diarization_boundary_details.csv` 可读，报告包含会议目录名和 `meeting/wer_summary.json`。

- [ ] **Step 2: 运行测试确认失败**

Run:

```powershell
python -m unittest python-jimmy/evaluation/tests/test_evaluate_long_audio_asr_speaker.py -v
```

Expected: FAIL，因为当前实现仍把 `wer_detail.txt` 写到 `asr/`，speaker 产物写到 `speaker/`，并生成 `result.txt`、manifest 和 status。

- [ ] **Step 3: 写入多会议隔离与无标注会议的失败测试**

新增一个含两个递归 `result.json` 的集成用例：一个会议有 `*_label.txt`，另一个没有。断言：

```python
self.assertTrue((output / "labeled" / "asr.txt").is_file())
self.assertTrue((output / "labeled" / "wer_detail.txt").is_file())
self.assertTrue((output / "unlabeled" / "asr.txt").is_file())
self.assertEqual(
    {path.name for path in (output / "unlabeled").iterdir()},
    {"asr.txt"},
)
self.assertIn("skipped_no_label", (output / "evaluation_report.md").read_text(encoding="utf-8"))
```

- [ ] **Step 4: 运行测试确认失败**

Run:

```powershell
python -m unittest python-jimmy/evaluation/tests/test_evaluate_long_audio_asr_speaker.py -v
```

Expected: FAIL，因为当前实现会将 `*_asr.txt` 放入共享 `inputs/`，并聚合所有会议的 WER / speaker 产物。

- [ ] **Step 5: 提交测试契约**

```powershell
git add python-jimmy/evaluation/tests/test_evaluate_long_audio_asr_speaker.py
git commit -m "test: define meeting-scoped evaluation outputs"
```

### Task 2: 按会议写入 ASR、WER 与片段差异产物

**Files:**
- Modify: `python-jimmy/evaluation/evaluate_long_audio_asr_speaker.py:write_public_result_txt`, `write_internal_asr_files`, `main`
- Test: `python-jimmy/evaluation/tests/test_evaluate_long_audio_asr_speaker.py`

- [ ] **Step 1: 新增会议输出路径与 ASR 投影 helper**

在 `evaluate_long_audio_asr_speaker.py` 新增：

```python
def meeting_output_dir(output_dir: Path, file_id: str) -> Path:
    if not file_id or file_id in {".", ".."} or Path(file_id).name != file_id:
        raise ValueError(f"unsafe meeting file_id for output directory: {file_id}")
    return output_dir / file_id


def write_meeting_asr_txt(record: ResultRecord, meeting_dir: Path) -> Path:
    path = meeting_dir / "asr.txt"
    _write_lines(
        path,
        [f"{segment.segment_id} {segment.speaker_id} {segment.asr_text}" for segment in record.segments],
    )
    return path
```

在写入前验证所有 `ResultRecord.file_id` 唯一；删除对 `result.txt` 和共享 `inputs/*_asr.txt` 的调用。

- [ ] **Step 2: 将 WER 改为逐会议执行且临时输入不落盘**

为单个 `record` 和对应 `LabelRecord` 新建 helper，使用：

```python
with tempfile.TemporaryDirectory(prefix="long-audio-evaluation-") as temp_dir:
    temp_root = Path(temp_dir)
    label_path = temp_root / "wer_label.txt"
    hyp_path = temp_root / "wer_hyp.txt"
    detail_path = meeting_dir / "wer_detail.txt"
```

用当前 `build_whole_audio_wer_inputs([record], {record.file_id: label})` 写入临时 label/hyp；调用本目录 `evaluation.py`，成功时把 `summarize_wer_detail(detail_path)` 写到 `meeting_dir / "wer_summary.json"`。不创建 stdout/stderr 日志文件，失败时保留错误字符串供报告使用。

- [ ] **Step 3: 将 Excel 改为会议目录中的固定文件名**

对每个有标签会议调用现有 `build_agent_sdk_segment_detail_rows` 和 `write_agent_sdk_segment_detail_workbook`：

```python
workbook_path = meeting_dir / "segment_asr_detail.xlsx"
rows = build_agent_sdk_segment_detail_rows(label.segments, record.segments, args.language, record.file_id)
write_agent_sdk_segment_detail_workbook(workbook_path, rows)
```

不改变六列表头、500 ms 对齐或 Agent SDK 差异计算。

- [ ] **Step 4: 运行测试确认通过**

Run:

```powershell
python -m unittest python-jimmy/evaluation/tests/test_evaluate_long_audio_asr_speaker.py -v
```

Expected: PASS，单会议和多会议断言均在会议目录内找到固定命名的 ASR、WER 和 Excel 文件。

- [ ] **Step 5: 提交会议级 ASR 产物实现**

```powershell
git add python-jimmy/evaluation/evaluate_long_audio_asr_speaker.py python-jimmy/evaluation/tests/test_evaluate_long_audio_asr_speaker.py
git commit -m "refactor: write evaluation artifacts per meeting"
```

### Task 3: 按会议写入说话人指标并生成唯一根报告

**Files:**
- Modify: `python-jimmy/evaluation/evaluate_long_audio_asr_speaker.py:run_speaker_metrics`, `_write_report`, `main`
- Modify: `python-jimmy/evaluation/README.md:输出结构`
- Test: `python-jimmy/evaluation/tests/test_evaluate_long_audio_asr_speaker.py`

- [ ] **Step 1: 为 speaker 评测写入固定会议级文件名**

逐会议调用现有 `speaker_diarization_metrics.py`，将输出参数固定为：

```python
summary_path = meeting_dir / "speaker_summary.json"
boundary_path = meeting_dir / "speaker_diarization_boundary_details.csv"
```

保留当前 JSON schema 与 CSV schema；不要生成 `speaker_diarization_per_file.json` 和 subprocess stdout/stderr 文件。

- [ ] **Step 2: 用会议状态集合生成根目录报告**

将 `_write_report` 的输入改为按 `file_id` 存储的会议记录，至少包含：源 result、标签路径、会议目录、ASR/WER/speaker/Excel 状态、成功产物和错误摘要。根目录只写：

```python
report_path = args.output_dir / "evaluation_report.md"
```

报告中的会议章节应展示：

```markdown
## meeting
- Status: success
- Output: `meeting/`
- WER: ...
- Speaker DER: ...
- Artifacts: `asr.txt`, `wer_detail.txt`, `wer_summary.json`, `segment_asr_detail.xlsx`, `speaker_summary.json`, `speaker_diarization_boundary_details.csv`
```

无标注会议显示 `Status: skipped_no_label` 和仅有 `asr.txt`。

- [ ] **Step 3: 删除旧输出控制流**

从 `main` 移除 manifest、status、共享 `inputs_dir`、`asr_dir`、`speaker_dir`、`result.txt` 和所有日志路径的创建。将子任务失败汇总为报告状态，任一有标签会议的 WER、speaker 或 Excel 执行失败时返回非零。

- [ ] **Step 4: 更新 README 的输出目录示例**

用固定名会议目录示例替换旧的 `inputs/asr/speaker` 目录说明，并明确 `speaker_diarization_boundary_details.csv` 保留、根目录只有 `evaluation_report.md`。

- [ ] **Step 5: 运行全部评测测试**

Run:

```powershell
python -m unittest discover -s python-jimmy/evaluation/tests -v
python -m compileall -q python-jimmy/evaluation
git diff --check
```

Expected: 全部测试通过，编译和 whitespace 检查无错误。

- [ ] **Step 6: 提交完成的目录重构**

```powershell
git add python-jimmy/evaluation/evaluate_long_audio_asr_speaker.py python-jimmy/evaluation/README.md python-jimmy/evaluation/tests/test_evaluate_long_audio_asr_speaker.py
git commit -m "refactor: standardize meeting evaluation outputs"
```

### Task 4: 真实结果回归检查

**Files:**
- Verify: 已生成的本地或远程评测输出目录

- [ ] **Step 1: 使用一个会议结果运行统一评测入口**

Run:

```powershell
python python-jimmy/evaluation/evaluate_long_audio_asr_speaker.py --results-dir <run-dir> --labels-dir <labels-dir> --output-dir <output-dir> --language ZH
```

Expected: `<output-dir>/<file_id>/` 存在；根目录仅有 `evaluation_report.md`。

- [ ] **Step 2: 检查固定文件集合**

Run:

```powershell
Get-ChildItem <output-dir>\<file_id> -File | Select-Object -ExpandProperty Name
```

Expected: 有标注会议恰好为 `asr.txt`、`wer_detail.txt`、`wer_summary.json`、`segment_asr_detail.xlsx`、`speaker_summary.json`、`speaker_diarization_boundary_details.csv`。

- [ ] **Step 3: 检查无旧布局残留**

Run:

```powershell
Get-ChildItem <output-dir> -Directory | Select-Object -ExpandProperty Name
Test-Path <output-dir>\inputs
Test-Path <output-dir>\asr
Test-Path <output-dir>\speaker
```

Expected: 目录列表仅含会议目录，后三个 `Test-Path` 均为 `False`。
