# Agent SDK Six-Column ASR Detail Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Replace the wide `asr_segment_diff.xlsx` diagnostic workbook with the same time-aligned six-column ASR-detail workbook used by `agent-sdk-test/python/wer.py`.

**Architecture:** `python-jimmy/evaluation/evaluate_long_audio_asr_speaker.py` remains the unified entry point and continues writing WER/DER/report artifacts. Its Excel layer is an adapter around a verbatim local copy of Agent SDK's timed grouping, local WER invocation, status calculation, and standard-library XLSX writer. Pipeline `result.json` and label text are adapted to the six-column detail types; pipeline scripts remain untouched.

**Tech Stack:** Python 3.10 standard library (`dataclasses`, `zipfile`, XML), existing local `evaluation.py`, `unittest`.

---

### Task 1: Add Agent SDK-compatible alignment behavior with failing tests first

**Files:**
- Modify: `python-jimmy/evaluation/tests/test_evaluate_long_audio_asr_speaker.py`
- Modify: `python-jimmy/evaluation/evaluate_long_audio_asr_speaker.py`

- [x] **Step 1: Write failing tests for six-column time alignment and dependency-free workbook output.**

Add a test that creates the following Agent-SDK-compatible segments and labels:

```python
predictions = [
    self._segment("audio_0_500", "speaker_00", "甲"),
    self._segment("audio_500_500", "speaker_00", "乙"),
    self._segment("audio_2500_500", "speaker_01", "多余"),
]
references = [
    self._segment("audio_0_1000", "alice", "甲乙"),
    self._segment("audio_1500_500", "bob", "漏失"),
]
```

Assert that `build_agent_sdk_segment_detail_rows` returns three groups: a one-to-many correct group, a reference-only `漏识别` group, and a hypothesis-only `多识别` group. Add a second test where a label ends at 1000 ms and the hypothesis starts at 1500 ms to assert the Agent SDK 500 ms threshold joins them into one group. Add an XLSX test using `zipfile.ZipFile` and XML extraction to assert the first six text cells exactly equal:

```python
["标注片段ID", "标注ASR文本", "识别片段ID", "识别ASR文本", "对齐状态", "差异文本"]
```

Also assert the generated archive does not require importing `openpyxl` and contains only `xl/worksheets/sheet1.xml`.

- [x] **Step 2: Run the focused test module and verify it fails because the new Agent SDK-compatible public helpers do not exist.**

Run:

```powershell
python -m unittest python-jimmy/evaluation/tests/test_evaluate_long_audio_asr_speaker.py -v
```

Expected: failure mentioning missing `build_agent_sdk_segment_detail_rows` and/or `write_agent_sdk_segment_detail_workbook`.

- [x] **Step 3: Implement the minimal Agent SDK-compatible detail adapter.**

Add `asr_segment_detail_diff.py` as a verbatim local copy of the Agent SDK detail implementation, including its frozen data classes and time functions with their original semantics:

```python
_SEGMENT_ALIGN_TOLERANCE_MS = 500
_segment_overlap(...)
_segment_distance(...)
_segments_are_time_aligned(...)
_time_alignment_groups(...)
_fallback_alignment_groups(...)
```

Adapt existing `TimedSegment` data from `result.json` and labels, call the local `evaluation.py` through the existing Python executable to calculate each non-empty-reference group's local difference row, and apply Agent SDK statuses (`正确`, `识别错误`, `漏识别`, `多识别`, `混合`). Implement the six-column `.xlsx` writer using `zipfile` and XML like Agent SDK `wer.py:write_segment_detail_workbook`, preserving embedded newlines in multi-segment cells.

- [x] **Step 4: Re-run the focused test module and verify it passes.**

Run:

```powershell
python -m unittest python-jimmy/evaluation/tests/test_evaluate_long_audio_asr_speaker.py -v
```

Expected: all tests in the module pass.

- [x] **Step 5: Commit the completed behavior and tests.**

```powershell
git add python-jimmy/evaluation/evaluate_long_audio_asr_speaker.py python-jimmy/evaluation/tests/test_evaluate_long_audio_asr_speaker.py
git commit -m "feat: align ASR detail workbook with agent SDK"
```

### Task 2: Wire the new workbook into the unified evaluation output and verify a real audio run

**Files:**
- Modify: `python-jimmy/evaluation/evaluate_long_audio_asr_speaker.py`
- Modify: `python-jimmy/evaluation/tests/test_evaluate_long_audio_asr_speaker.py`
- Modify: `python-jimmy/evaluation/README.md`

- [x] **Step 1: Write a failing integration assertion for the output file contract.**

In `RunnerIntegrationTest.test_cli_writes_standardized_evaluation_artifacts`, replace the old expectation for `asr_segment_diff.xlsx` with:

```python
self.assertTrue((output / "sample_segment_asr_detail.xlsx").is_file())
self.assertFalse((output / "asr_segment_diff.xlsx").exists())
```

Add an assertion that `evaluation_status.json` lists the per-audio six-column workbook artifact.

- [x] **Step 2: Run the focused integration test and verify it fails because the runner still emits `asr_segment_diff.xlsx`.**

Run:

```powershell
python -m unittest python-jimmy/evaluation/tests/test_evaluate_long_audio_asr_speaker.py -v
```

Expected: failure stating `sample_segment_asr_detail.xlsx` does not exist.

- [x] **Step 3: Replace the old wide workbook call with per-audio Agent SDK-style workbooks.**

Remove `write_segment_diff_workbook`, its 23-column row construction, `summarize_segment_rows`, and the `openpyxl` requirement. For every result record with a resolved label, write:

```python
output_dir / f"{record.file_id}_segment_asr_detail.xlsx"
```

Use `build_agent_sdk_segment_detail_rows` and `write_agent_sdk_segment_detail_workbook`. Keep whole-audio WER and speaker metrics unchanged. Update report artifact names and `evaluation_status.json` artifact paths. If labels are absent, retain normalization and mark ASR/speaker work as skipped without generating a per-audio Excel file.

- [x] **Step 4: Update README output documentation.**

Replace `asr_segment_diff.xlsx` with `<file_id>_segment_asr_detail.xlsx`, document the six columns and the fixed 500 ms Agent SDK time-alignment threshold, and remove the `openpyxl` install requirement.

- [x] **Step 5: Run all evaluation tests and verify they pass.**

Run:

```powershell
python -m unittest discover -s python-jimmy/evaluation/tests -v
git diff --check
```

Expected: all tests pass and `git diff --check` produces no output.

- [x] **Step 6: Run the real `23_asr_1782715267098` evaluation locally.**

Run:

```powershell
python python-jimmy/evaluation/evaluate_long_audio_asr_speaker.py `
  --results-dir "C:\Users\admin\Downloads\断句pipeline-python脚本测试-0920\runs\23\latest-compact-display-20260921-090944" `
  --labels-dir "C:\Users\admin\Downloads\断句pipeline-python脚本测试-0920\labels" `
  --output-dir "C:\Users\admin\Downloads\断句pipeline-python脚本测试-0920\runs\23\latest-compact-display-20260921-090944\evaluation-agent-sdk-detail" `
  --language ZH `
  --boundary-tolerance-ms 500 `
  --collar-ms 500
```

Expected: successful normalization/WER/speaker/Excel tasks; a file named `23_asr_1782715267098_segment_asr_detail.xlsx`; no `asr_segment_diff.xlsx`; WER and DER identical to the previous evaluation because only the diagnostic presentation layer changed.

- [x] **Step 7: Commit the integration, README, and test changes.**

```powershell
git add python-jimmy/evaluation/evaluate_long_audio_asr_speaker.py python-jimmy/evaluation/tests/test_evaluate_long_audio_asr_speaker.py python-jimmy/evaluation/README.md
git commit -m "docs: standardize six-column ASR detail output"
```
