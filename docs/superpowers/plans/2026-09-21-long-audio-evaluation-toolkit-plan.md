# 长音频 ASR 与说话人评测工具包 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增可独立复制的 `python-jimmy/evaluation/` 评测工具，将 pipeline 的 `result.json` 与 `*_label.txt` 转换为标准化评测结果、WER/DER/断句指标和片段级 Excel，而不修改两个 pipeline 脚本。

**Architecture:** 统一 CLI 递归读取 result.json，解析 ID 中的起始时间/时长，单独输出机器可读 result.txt 和供现有评测器使用的内部投影。WER 使用本目录可移植的 `evaluation.py`；DER、说话人切换和边界数据使用本目录可移植的 `speaker_diarization_metrics.py`；入口分别记录每项任务的状态、日志和报告，因此任一可选依赖或单音频错误都不会静默丢失。

**Tech Stack:** Python 3.10+、标准库、`kaldialign`（带纯 Python 兜底）、`pyannote.core`/`pyannote.metrics`、`openpyxl`、`unittest`。

---

## File structure

- Create `python-jimmy/evaluation/__init__.py`：声明可移植评测工具包。
- Create `python-jimmy/evaluation/evaluation.py`：从 Agent SDK 复用且保留 kaldialign 兜底的 WER CLI。
- Create `python-jimmy/evaluation/speaker_diarization_metrics.py`：从 Agent SDK 复用、接受 `*_asr.txt` 与 `*_label.txt` 的 DER/边界 CLI。
- Create `python-jimmy/evaluation/evaluate_long_audio_asr_speaker.py`：公开统一入口、解析/匹配/投影/子评测编排、Excel、状态与报告。
- Create `python-jimmy/evaluation/README.md`：CLI、输入契约、输出目录和依赖说明。
- Create `python-jimmy/evaluation/tests/test_evaluate_long_audio_asr_speaker.py`：入口单元与集成测试。
- Create `python-jimmy/evaluation/tests/test_evaluation.py`：WER 正常化及 CLI 明细测试。
- Create `python-jimmy/evaluation/tests/test_speaker_diarization_metrics.py`：DER、overlap 跳过与边界测试。

### Task 1: 建立可移植 WER 与 DER 基础实现

**Files:**
- Create: `python-jimmy/evaluation/__init__.py`
- Create: `python-jimmy/evaluation/evaluation.py`
- Create: `python-jimmy/evaluation/speaker_diarization_metrics.py`
- Create: `python-jimmy/evaluation/tests/test_evaluation.py`
- Create: `python-jimmy/evaluation/tests/test_speaker_diarization_metrics.py`

- [ ] **Step 1: 写失败测试，锁定 WER 细节和 speaker overlap 语义。**

```python
# test_evaluation.py
self.assertEqual(text_normalization("你好，ABC", "ZH"), ["你", "好", "abc"])
self.assertEqual(main(["--label", str(label), "--hyp", str(hyp), "--language", "ZH", "--detail", str(detail)]), 0)
self.assertIn("%WER", detail.read_text(encoding="utf-8"))

# test_speaker_diarization_metrics.py
report = evaluate_paths(results_dir, labels_dir, boundary_tolerance_ms=500, collar_ms=0)
self.assertEqual(report.summary["metrics"]["overlap_segments"], 1)
self.assertEqual(report.evaluated_count, 1)
```

- [ ] **Step 2: 运行失败测试，确认新模块不存在。**

Run: `python -m unittest discover -s python-jimmy/evaluation/tests -p "test_evaluation.py" -v`

Expected: `ModuleNotFoundError`。

- [ ] **Step 3: 复制并最小化调整已验证的实现。**

```powershell
Copy-Item 'D:\code\proMax\tz-llm-sdk\agent-sdk-test\python\evaluation.py' 'python-jimmy\evaluation\evaluation.py'
Copy-Item 'D:\code\proMax\tz-llm-sdk\agent-sdk-test\python\speaker_diarization_metrics.py' 'python-jimmy\evaluation\speaker_diarization_metrics.py'
Set-Content 'python-jimmy\evaluation\__init__.py' -Value '"""Portable long-audio ASR and speaker evaluation tools."""'
```

保持两个 CLI 的现有参数与输出文件名，不依赖 Agent SDK 的其他模块。

- [ ] **Step 4: 运行基础测试，确认通过。**

Run: `python -m unittest discover -s python-jimmy/evaluation/tests -p "test_evaluation.py" -v; python -m unittest discover -s python-jimmy/evaluation/tests -p "test_speaker_diarization_metrics.py" -v`

Expected: 全部 PASS。

- [ ] **Step 5: 提交基础工具。**

```powershell
git add python-jimmy/evaluation
git commit -m "feat: add portable WER and speaker metric tools"
```

### Task 2: 实现 result.json 和标签解析及机器可读投影

**Files:**
- Create: `python-jimmy/evaluation/evaluate_long_audio_asr_speaker.py`
- Create: `python-jimmy/evaluation/tests/test_evaluate_long_audio_asr_speaker.py`

- [ ] **Step 1: 写失败测试，覆盖 result.json 解析和 label 自动匹配。**

```python
records = discover_result_records(results_dir)
self.assertEqual(records[0].file_id, "23_asr_1782715267098")
self.assertEqual(records[0].segments[0].start_ms, 1118)
self.assertEqual(records[0].segments[0].end_ms, 3426)
labels = resolve_labels(records, labels_dir, explicit_label=None)
self.assertEqual(labels[records[0].file_id].name, "23_asr_1782715267098_label.txt")
write_public_result_txt(records, output_dir / "result.txt")
self.assertEqual((output_dir / "result.txt").read_text(encoding="utf-8"),
                 "23_asr_1782715267098_1118_2308 speaker_02 来吧我们聊正常聊天就行\\n")
```

- [ ] **Step 2: 运行失败测试。**

Run: `python -m unittest python-jimmy/evaluation/tests/test_evaluate_long_audio_asr_speaker.py -v`

Expected: 导入错误或属性缺失。

- [ ] **Step 3: 实现最小解析模型与投影函数。**

```python
@dataclass(frozen=True)
class TimedSegment:
    file_id: str
    segment_id: str
    start_ms: int
    duration_ms: int
    speaker_id: str
    asr_text: str

SEGMENT_ID_RE = re.compile(r"^(?P<file_id>.+)_(?P<start_ms>\\d+)_(?P<duration_ms>\\d+)$")

def parse_segment_id(segment_id: str) -> tuple[str, int, int]: ...
def discover_result_records(results_dir: Path) -> list[ResultRecord]: ...
def parse_label_file(path: Path) -> list[ReferenceSegment]: ...
def resolve_labels(...): ...
def write_public_result_txt(records, path): ...
def write_internal_asr_files(records, inputs_dir): ...
```

`result.txt` 每行必须是 `<segment_id> <speaker_id> <asr_text>`，空 ASR 文本也必须保留前两列。不得修改 pipeline 原目录的任何文件。

- [ ] **Step 4: 运行解析测试。**

Run: `python -m unittest python-jimmy/evaluation/tests/test_evaluate_long_audio_asr_speaker.py -v`

Expected: 至少解析/投影测试 PASS。

- [ ] **Step 5: 提交。**

```powershell
git add python-jimmy/evaluation/evaluate_long_audio_asr_speaker.py python-jimmy/evaluation/tests/test_evaluate_long_audio_asr_speaker.py
git commit -m "feat: normalize long audio pipeline results for evaluation"
```

### Task 3: 实现整音频 WER 输入和时间重叠的片段映射

**Files:**
- Modify: `python-jimmy/evaluation/evaluate_long_audio_asr_speaker.py`
- Modify: `python-jimmy/evaluation/tests/test_evaluate_long_audio_asr_speaker.py`

- [ ] **Step 1: 写失败测试，覆盖 split、merge、unmatched、multi。**

```python
rows = build_segment_detail_rows(reference_segments, prediction_segments)
self.assertEqual(rows[0]["match_status"], "matched")
self.assertEqual(rows[1]["match_status"], "overlap_not_scored")
self.assertEqual(rows[2]["match_status"], "unmatched_reference")
self.assertEqual(rows[-1]["match_status"], "unmatched_prediction")
label_lines, hyp_lines = build_whole_audio_wer_inputs(records, labels)
self.assertEqual(label_lines, ["23_asr_1782715267098 参考一参考二"])
self.assertEqual(hyp_lines, ["23_asr_1782715267098 预测一预测二"])
```

- [ ] **Step 2: 运行失败测试。**

Run: `python -m unittest python-jimmy/evaluation/tests/test_evaluate_long_audio_asr_speaker.py -v`

Expected: 映射函数不存在或断言失败。

- [ ] **Step 3: 实现按区间交集的映射与单人片段 WER。**

```python
def interval_overlap_ms(left_start: int, left_end: int, right_start: int, right_end: int) -> int:
    return max(0, min(left_end, right_end) - max(left_start, right_start))

def build_segment_detail_rows(reference_segments, prediction_segments):
    # 每个 reference 选择所有正交集 prediction；multi 固定为 overlap_not_scored。
    # 无交集 reference/prediction 分别输出 unmatched_*。
    ...

def compute_text_error_counts(reference: str, hypothesis: str, language: str) -> dict[str, int]: ...
```

全局 WER 的 ID 固定为 file_id，按起始时间拼接文本；标签中的 `multi` 文本保留在全局 WER，但不计单人片段 WER。

- [ ] **Step 4: 运行映射测试。**

Run: `python -m unittest python-jimmy/evaluation/tests/test_evaluate_long_audio_asr_speaker.py -v`

Expected: 全部 PASS。

- [ ] **Step 5: 提交。**

```powershell
git add python-jimmy/evaluation
git commit -m "feat: add overlap-aware ASR segment evaluation"
```

### Task 4: 编排 WER、DER、状态、报告和 Excel

**Files:**
- Modify: `python-jimmy/evaluation/evaluate_long_audio_asr_speaker.py`
- Modify: `python-jimmy/evaluation/tests/test_evaluate_long_audio_asr_speaker.py`

- [ ] **Step 1: 写失败集成测试，断言目录、状态与可选依赖失败可见。**

```python
exit_code = main(["--results-dir", str(results), "--labels-dir", str(labels), "--output-dir", str(output)])
self.assertEqual(exit_code, 0)
self.assertTrue((output / "asr" / "wer_detail.txt").is_file())
self.assertTrue((output / "speaker" / "speaker_diarization_summary.json").is_file())
self.assertTrue((output / "asr_segment_diff.xlsx").is_file())
status = json.loads((output / "evaluation_status.json").read_text(encoding="utf-8"))
self.assertEqual(status["tasks"]["wer"]["status"], "success")
```

另用 `unittest.mock.patch` 将 `subprocess.run` 返回 non-zero，断言 `stdout`、`stderr` 日志写入且对应任务状态为 `failed`；标签缺失时断言 WER/DER 是 `skipped`，但 `result.txt` 仍存在。

- [ ] **Step 2: 运行失败集成测试。**

Run: `python -m unittest python-jimmy/evaluation/tests/test_evaluate_long_audio_asr_speaker.py -v`

Expected: 缺少 `main()` 编排、状态或文件。

- [ ] **Step 3: 实现子进程编排及产物。**

```python
def run_subprocess_task(name: str, command: list[str], stdout_path: Path, stderr_path: Path) -> TaskStatus: ...
def write_segment_diff_workbook(path: Path, summary_rows: list[dict], detail_rows: list[dict]) -> None: ...
def write_report(path: Path, manifest: dict, status: dict, wer: dict, speaker: dict) -> None: ...
def main(argv: list[str] | None = None) -> int: ...
```

主流程必须先写 `evaluation_manifest.json` 和标准 `result.txt`；随后写 `inputs/wer_label.txt`、`inputs/wer_hyp.txt`，调用 `sys.executable evaluation.py`，读取 `wer_detail.txt` 计算/写入 `wer_summary.json`；再调用 `speaker_diarization_metrics.py`；最后独立尝试 Excel。所有子进程 stdout/stderr 必须落盘。请求的任一评测失败时返回非零。

- [ ] **Step 4: 运行完整单元测试。**

Run: `python -m unittest discover -s python-jimmy/evaluation/tests -v`

Expected: 全部 PASS。

- [ ] **Step 5: 提交。**

```powershell
git add python-jimmy/evaluation
git commit -m "feat: add unified long audio evaluation runner"
```

### Task 5: 文档和真实样本验收

**Files:**
- Create: `python-jimmy/evaluation/README.md`
- Modify: `python-jimmy/evaluation/tests/test_evaluate_long_audio_asr_speaker.py`

- [ ] **Step 1: 写失败 CLI help/README 契约测试。**

```python
completed = subprocess.run([sys.executable, str(CLI), "--help"], text=True, capture_output=True)
self.assertEqual(completed.returncode, 0)
self.assertIn("--results-dir", completed.stdout)
self.assertIn("--label", completed.stdout)
```

- [ ] **Step 2: 运行失败测试。**

Run: `python -m unittest python-jimmy/evaluation/tests/test_evaluate_long_audio_asr_speaker.py -v`

Expected: CLI help 或参数缺失。

- [ ] **Step 3: 完成 README。**

README 要列出安装：`pip install openpyxl kaldialign pyannote.core pyannote.metrics`；单文件命令；递归多 run 命令；输入 `result.json` schema；评测目录与 pipeline 的人工可读 `result.txt` 不同；`multi` 的 WER/DER/片段 WER 口径；失败状态解释。

- [ ] **Step 4: 运行全部本地测试和真实样本评测。**

Run:

```powershell
python -m unittest discover -s python-jimmy/evaluation/tests -v
python python-jimmy/evaluation/evaluate_long_audio_asr_speaker.py `
  --results-dir 'C:\Users\admin\Downloads\断句pipeline-python脚本测试-0920\runs\23\latest-compact-display-20260921-090944' `
  --labels-dir 'C:\Users\admin\Downloads\断句pipeline-python脚本测试-0920\labels' `
  --output-dir 'C:\Users\admin\Downloads\断句pipeline-python脚本测试-0920\runs\23\latest-compact-display-20260921-090944\evaluation' `
  --language ZH --boundary-tolerance-ms 500 --collar-ms 500
```

Expected: 退出码为 0；输出 `result.txt`、`asr/wer_detail.txt`、`speaker/speaker_diarization_summary.json`、`speaker/speaker_diarization_boundary_details.csv`、`asr_segment_diff.xlsx`、`evaluation_report.md`、`evaluation_status.json`。

- [ ] **Step 5: 提交文档和验收测试。**

```powershell
git add python-jimmy/evaluation
git commit -m "docs: document long audio evaluation toolkit"
```

### Task 6: Linux 远程回归验收

**Files:**
- No source file changes.

- [ ] **Step 1: 检查工作树、提交和远端分支。**

Run: `git status --short --branch; git log -1 --oneline; git push jimmy HEAD:codex/speaker-mask-fusion-clean-cluster`

Expected: 工作树干净，正常非强推成功。

- [ ] **Step 2: 按远程 skill 安全切换并 ff-only 拉取。**

Run: `ssh -i 'C:\Users\admin\.ssh\codex_ed25519' -p 21022 ps@159.135.196.86 'cd /speech_store/jimmy/k2_origin/sherpa-onnx && git status --short && git fetch jimmy codex/speaker-mask-fusion-clean-cluster && git switch codex/speaker-mask-fusion-clean-cluster && git pull --ff-only jimmy codex/speaker-mask-fusion-clean-cluster && git rev-parse HEAD'`

Expected: 无 tracked 改动、远端 HEAD 等于本地提交。

- [ ] **Step 3: 用远端已有 23 样本结果/标签执行相同 CLI。**

仅在确认远端结果与标签的实际路径后执行，输出目录限定在远端仓库内，例如 `build/evaluation-23`。保存命令、退出码、产物清单、`evaluation_status.json` 与报告关键指标。

- [ ] **Step 4: 复核输出，不伪造远端测试。**

Run: `Get-Content '<evaluation-output>\evaluation_status.json'; Get-Content '<evaluation-output>\evaluation_report.md'; Get-ChildItem -Recurse '<evaluation-output>'`

Expected: ASR、speaker、Excel 和公共 result.txt 均存在；任何失败在状态中可见。

## Plan self-review

- [x] 设计中的“只新增评测目录、绝不修改 pipeline”由 Tasks 1–5 覆盖。
- [x] `result.json` 到独立标准 `result.txt`、label 自动发现、显式 label、整音频 WER、DER、边界、Excel、manifest/status/report 都有明确任务。
- [x] `multi` 的全局 WER 保留、DER skip overlap、片段 WER 不计均由 Task 3 测试和实现固定。
- [x] 无标签、可选依赖、子进程失败的可观测状态由 Task 4 覆盖。
- [x] 计划不包含 TODO/TBD 占位项；每一改动任务先 RED、后 GREEN、再提交。
