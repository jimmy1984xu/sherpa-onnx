# Pyannote Segmentation 断句对照评测 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 生成一个可复现的两方案 pyannote 断句对照评测工具链，完成远程构建/推理、WER 与说话人指标计算、Diart 源码分析，并把审计材料及中文报告输出到 `C:\Users\admin\Downloads\sherpa-onnx断句优化测试-0917`。

**Architecture:** 在 `python-jimmy` 增加一个纯 Python 评测/汇总模块，负责固定两种命令、输入与模型哈希清单、将 pipeline 的 `result.json` 规范化为供指标脚本读取的三列 `*_asr.txt`、计算中文 WER、生成边界差异表和报告。PowerShell 驱动仅完成 Windows 输入固化、SSH/SCP、远端 build/两次推理与回收，不计算指标；Windows 汇总模块以相同的受控输入和远端原始结果生成最终报告。

**Tech Stack:** Python 3.10 标准库、现有 `python-jimmy/offline-long-audio-pipeline`、现有 `python-jimmy/evaluation.py` 中文规范化、`speaker_diarization_metrics.py`、PowerShell 7、Git、SSH/SCP、Linux CMake/Make、远端已存在的 sherpa-onnx Python extension 和模型。

---

## 文件结构

| 文件 | 责任 |
|---|---|
| Create `python-jimmy/segmentation-punctuation-evaluation.py` | 单一评测 CLI：生成 manifest；验证两方案公平性；把 `result.json` 转为标准三列预测文件；从 label/预测生成 WER；执行拷贝的 DER/边界指标；汇总差异边界、模型 hash、耗时和中文 `报告.md`。所有 JSON/CSV/Markdown 均使用 UTF-8。 |
| Create `python-jimmy/test_segmentation_punctuation_evaluation.py` | `unittest` 覆盖命令等价性、manifest 公平性、结果标准化、中文 WER、边界差异和失败报告保留行为；不加载 ONNX 或访问网络。 |
| Create `python-jimmy/run-segmentation-punctuation-evaluation.ps1` | Windows 编排入口：建立 `01_input`、复制固定两条输入与标注、计算输入/Windows 模型 hash、生成 manifest、检查远端、推送当前 commit、远端构建、运行两入口脚本、回收结果、调用 Python 汇总并保留失败日志。 |
| Create `python-jimmy/diart-source-analysis.py` | 克隆 Diart 到输出目录临时源码区或读取已有固定 checkout；写出 commit、许可证、关键模块/函数定位和以结构化 JSON 供总报告消费的分析，不运行 Diart 推理。 |
| Modify `docs/superpowers/specs/2026-09-17-pyannote-segmentation-punctuation-evaluation-design.md` | 将“状态”更新为已实施/已运行，追加最终的实际 commit、执行状态和链接到结果目录；若运行被前置条件阻断，只记录阻断阶段而不更改技术结论。 |
| Create `C:\Users\admin\Downloads\sherpa-onnx断句优化测试-0917\...`（运行时产物，不纳入 Git） | 固化输入、原始远端输出、指标、Diart 分析、机器可读汇总和最终报告。 |

### 共同常量和数据格式

在 `segmentation-punctuation-evaluation.py` 中定义以下不可变常量，作为唯一数据集真源：

```python
OUTPUT_ROOT = Path(r"C:\Users\admin\Downloads\sherpa-onnx断句优化测试-0917")
TEST_CASES = (
    TestCase(
        file_id="23_asr_1782715267098",
        pcm=Path(r"\\10.88.0.243\xainas\ProMax\测试集\音频测试集\中文\内部会议\双人安静咨询\promax\23_asr_1782715267098.pcm"),
        label=Path(r"\\10.88.0.243\xainas\ProMax\测试集\音频测试集\中文\内部会议\双人安静咨询\promax\23_asr_1782715267098_label.txt"),
    ),
    TestCase(
        file_id="asr_1788402212076",
        pcm=Path(r"\\10.88.0.243\xainas\ProMax\测试集\音频测试集\中文\内部会议\信息流投放实习生面试\asr_1788402212076.pcm"),
        label=Path(r"\\10.88.0.243\xainas\ProMax\测试集\音频测试集\中文\内部会议\信息流投放实习生面试\asr_1788402212076_label.txt"),
    ),
)
COMMON_ARGUMENTS = (
    "--audio-format", "pcm", "--sample-rate", "16000", "--channels", "1",
    "--sample-width", "2", "--asr-engine", "paraformer", "--num-clusters", "-1",
    "--cluster-threshold", "0.6", "--segmentation-mode", "vad-pyannote",
)
METRIC_ARGUMENTS = ("--collar-ms", "500", "--boundary-tolerance-ms", "500")
```

每份 manifest 必须记录两种方案的完整 argv、Git commit、`git status --porcelain`、输入/label SHA-256、每个模型角色（ASR、VAD、speaker、segmentation）的本地和远端绝对路径、模型文件 SHA-256、线程数以及所有默认参数的显式值。模型角色只允许同角色路径不同于操作系统，但其哈希必须相同。

标准预测文件每行严格为：

```text
<file_id>_<start_ms>_<duration_ms> <speaker_id_without_parentheses> <asr_text>
```

时间、speaker 和文本来自 `result.json` 的顺序 segment；空文本保留为空字符串，非法 `end_ms <= start_ms` 必须使该方案在该音频标记失败并保留其原始 `result.json`。

### Task 1: 建立并验证纯 Python 评测核心

**Files:**
- Create: `python-jimmy/segmentation-punctuation-evaluation.py`
- Test: `python-jimmy/test_segmentation_punctuation_evaluation.py`

- [ ] **Step 1: 编写命令公平性与 pipeline 结果适配的失败测试。**

  在 `python-jimmy/test_segmentation_punctuation_evaluation.py` 通过 `importlib.util.spec_from_file_location` 加载带连字符的 CLI 文件。加入以下测试；全部只使用 `tempfile.TemporaryDirectory()`：

  ```python
  class InvocationTest(unittest.TestCase):
      def test_variants_share_all_common_arguments_and_only_streaming_adds_chunk(self):
          common = module.build_common_arguments(model_paths(), output_root=Path("out"))
          baseline = module.build_invocation("baseline", Path("a.pcm"), common)
          streaming = module.build_invocation("streaming", Path("a.pcm"), common)
          self.assertEqual(module.common_option_map(baseline), module.common_option_map(streaming))
          self.assertNotIn("--segmentation-chunk-ms", baseline)
          self.assertEqual(
              streaming[streaming.index("--segmentation-chunk-ms") + 1], "32"
          )
          self.assertEqual(module.option_value(baseline, "--num-clusters"), "-1")
          self.assertEqual(module.option_value(streaming, "--cluster-threshold"), "0.6")

  class ResultAdapterTest(unittest.TestCase):
      def test_result_json_becomes_metrics_compatible_three_column_asr_file(self):
          payload = {"segments": [
              {"start_ms": 1000, "end_ms": 2500, "speaker_id": "(spk_1)", "asr_text": "你好"},
              {"start_ms": 2600, "end_ms": 3000, "speaker_id": "spk_0", "asr_text": "世界"},
          ]}
          output = module.write_metrics_asr_file(Path(self.idir), "sample", payload)
          self.assertEqual(output.read_text(encoding="utf-8").splitlines(), [
              "sample_1000_1500 spk_1 你好", "sample_2600_400 spk_0 世界"
          ])
  ```

- [ ] **Step 2: 运行测试，确认因实现未定义而失败。**

  Run:

  ```powershell
  python -m unittest python-jimmy/test_segmentation_punctuation_evaluation.py -v
  ```

  Expected: `ImportError` 或 `AttributeError`，指向尚未创建的 `segmentation-punctuation-evaluation.py`、`build_common_arguments`、`build_invocation` 或 `write_metrics_asr_file`；不应访问 UNC、SSH 或模型。

- [ ] **Step 3: 实现可导入的核心数据模型、参数构造和结果适配。**

  创建 CLI，使用以下最小实现形状；所有模型目录均从 `ModelPaths` 显式传入，不能在远端依赖 Windows 默认路径：

  ```python
  @dataclass(frozen=True)
  class ModelPaths:
      asr_dir: Path
      vad_dir: Path
      speaker_dir: Path
      segmentation_dir: Path

  def build_common_arguments(paths: ModelPaths, output_root: Path) -> list[str]:
      return [
          "--output-root", str(output_root), "--audio-format", "pcm",
          "--sample-rate", "16000", "--channels", "1", "--sample-width", "2",
          "--asr-dir", str(paths.asr_dir), "--vad-dir", str(paths.vad_dir),
          "--speaker-dir", str(paths.speaker_dir), "--segmentation-dir", str(paths.segmentation_dir),
          "--asr-num-threads", "1", "--speaker-num-threads", "2",
          "--vad-threshold", "0.5", "--min-silence-duration", "0.8",
          "--min-speech-duration", "0.25", "--max-speech-duration", "25.0",
          "--pre-speech-pad-duration", "0.0", "--cluster-threshold", "0.6",
          "--num-clusters", "-1", "--min-cluster-duration", "1.0",
          "--centroid-assignment-similarity-threshold", "0.5",
          "--diarization-min-duration-on", "0.5", "--diarization-min-duration-off", "0.5",
          "--asr-engine", "paraformer", "--segmentation-mode", "vad-pyannote",
      ]

  def build_invocation(kind: str, pcm: Path, common: list[str]) -> list[str]:
      script = {
          "baseline": "offline-long-audio-pipeline-asr-speaker.py",
          "streaming": "offline-long-audio-pipeline-asr-speaker-segmentation.py",
      }[kind]
      argv = [sys.executable, str(SCRIPT_DIR / script), "--audio", str(pcm), *common]
      if kind == "streaming":
          argv.extend(["--segmentation-chunk-ms", "32"])
      return argv

  def write_metrics_asr_file(output_dir: Path, file_id: str, result: dict[str, object]) -> Path:
      rows = []
      for item in result["segments"]:
          start_ms, end_ms = int(item["start_ms"]), int(item["end_ms"])
          if end_ms <= start_ms:
              raise ValueError(f"invalid segment interval: {start_ms}-{end_ms}")
          speaker = str(item.get("speaker_id", "-")).strip().strip("()") or "-"
          text = str(item.get("asr_text", "")).replace("\r", " ").replace("\n", " ").strip()
          rows.append(f"{file_id}_{start_ms}_{end_ms - start_ms} {speaker} {text}".rstrip())
      path = output_dir / f"{file_id}_asr.txt"
      path.parent.mkdir(parents=True, exist_ok=True)
      path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
      return path
  ```

  增加 `common_option_map()`、`option_value()` 和 `validate_variant_fairness()`：后者将两条 argv 转换为选项映射，要求除脚本路径、`--audio`、`--run-label`、`--segmentation-chunk-ms` 外逐项相同；baseline 不得出现 chunk 参数，streaming 必须等于 `32`，`num_clusters=-1`、`cluster_threshold=0.6` 不可变。

- [ ] **Step 4: 重新运行核心测试。**

  Run:

  ```powershell
  python -m unittest python-jimmy/test_segmentation_punctuation_evaluation.py -v
  ```

  Expected: 当前新增测试全部 `ok`；不产生 `C:\Users\admin\Downloads\sherpa-onnx断句优化测试-0917`。

- [ ] **Step 5: 提交核心评测框架。**

  ```powershell
  git add python-jimmy/segmentation-punctuation-evaluation.py python-jimmy/test_segmentation_punctuation_evaluation.py
  git commit -m "feat: add segmentation evaluation core"
  ```

### Task 2: 实现 manifest、输入冻结、中文 WER 和边界差异汇总

**Files:**
- Modify: `python-jimmy/segmentation-punctuation-evaluation.py`
- Modify: `python-jimmy/test_segmentation_punctuation_evaluation.py`

- [ ] **Step 1: 为 hash、label 解析、中文 WER 与公平性失败编写测试。**

  增加以下测试。WER 需复用仓库 `python-jimmy/evaluation.py` 的 `text_normalization(text, "zh")`，再按其空格 token 计算 Levenshtein，确保中文标点、英文大小写和中文数字处理规则与现有工具一致：

  ```python
  def test_label_and_predictions_generate_zh_wer_counts(self):
      label = Path(self.idir) / "sample_label.txt"
      label.write_text("sample_0_1000 (alice) 你好，世界！\n", encoding="utf-8")
      predicted = Path(self.idir) / "sample_asr.txt"
      predicted.write_text("sample_0_1000 spk_0 你好 世界\n", encoding="utf-8")
      summary = module.compute_wer(label, predicted, language="zh")
      self.assertEqual(summary["reference_tokens"], 4)
      self.assertEqual(summary["errors"], 0)
      self.assertEqual(summary["insertions"], 0)
      self.assertEqual(summary["deletions"], 0)
      self.assertEqual(summary["substitutions"], 0)

  def test_manifest_rejects_same_role_model_hash_mismatch(self):
      manifest = valid_manifest()
      manifest["models"]["remote"]["vad"]["sha256"] = "f" * 64
      with self.assertRaisesRegex(ValueError, "VAD.*SHA-256"):
          module.validate_manifest(manifest)
  ```

- [ ] **Step 2: 运行测试，确认缺失函数和 hash 检查导致失败。**

  Run:

  ```powershell
  python -m unittest python-jimmy/test_segmentation_punctuation_evaluation.py -v
  ```

  Expected: 新增测试报 `AttributeError`，指出 `compute_wer` / `validate_manifest` 未定义；先前 Task 1 测试仍通过。

- [ ] **Step 3: 实现输入复制、SHA-256、manifest 和 WER。**

  在 CLI 中加入 subcommands：`prepare`, `validate-manifest`, `normalize-results`, `compute-wer`, `summarize`。`prepare` 只从 `TEST_CASES` 复制两份 PCM 与两份 label 到 `<output>/01_input/{pcm,labels}`，若目标 hash 已相同则不覆盖；否则写临时文件后用 `Path.replace()` 原子替换。对于每个输入和 label 用 1 MiB 块实现 `sha256_file()`。

  label 解析必须严格使用三列格式，不接受不含 segment ID 或 speaker 的行：

  ```python
  def parse_three_column_file(path: Path) -> list[TimedText]:
      records = []
      for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
          if not line.strip():
              continue
          fields = line.split(maxsplit=2)
          if len(fields) < 2:
              raise ValueError(f"{path}:{number}: expected id speaker [text]")
          records.append(TimedText(fields[0], fields[1].strip().strip("()"), fields[2] if len(fields) == 3 else ""))
      return records
  ```

  对 WER，按开始时间排序 reference 和 hypothesis，再分别拼接文本；内部动态规划的每个 cell 保存 `(error_count, insertions, deletions, substitutions)`，用替换、删除、插入三种转移产生完整错误分解。输出 `<scheme>/<file_id>/metrics/wer.json` 和 `wer_detail.md`，内容包含归一化后的 reference/hypothesis、token 数、I/D/S、errors、wer。WER 的定义固定为 `errors / max(reference_tokens, 1)`。

  `manifest.json` 格式包含 `schema_version: 1`、创建时间、Windows/remote repo 版本、测试样本、两个完整 commands、两端 model inventory、`metrics_parameters` 和 `status`。`validate_manifest()` 要验证所列文件 ID 恰好为两条固定样本、两个方案的 common options 相等、音频规范相同、模型四个角色的 hash 相同、自动聚类参数和评价参数固定；违反时报告具体 role 或 option。

- [ ] **Step 4: 执行本模块测试与现有离线 pipeline 回归测试。**

  Run:

  ```powershell
  python -m unittest python-jimmy/test_segmentation_punctuation_evaluation.py -v
  python -m unittest discover -s python-jimmy/offline-long-audio-pipeline/tests -v
  ```

  Expected: 新模块测试全部通过。第二条命令在本工作树当前 Windows 解释器可预期仅 `test_public_symbols_and_span_shape` 失败，因为它导入的是未含 `SpeakerSegmentationConfig` 的已安装旧 `sherpa_onnx`；把该基线失败的完整输出存到最终报告的“预检限制”，不得把它归因于本任务。远端构建后必须在 `PYTHONPATH=<remote build>/lib:<repo>/sherpa-onnx/python` 下复跑并要求成功。

- [ ] **Step 5: 提交输入、WER 和 manifest 能力。**

  ```powershell
  git add python-jimmy/segmentation-punctuation-evaluation.py python-jimmy/test_segmentation_punctuation_evaluation.py
  git commit -m "feat: add reproducible segmentation evaluation metrics"
  ```

### Task 3: 接入拷贝的说话人指标、边界差异和报告生成

**Files:**
- Modify: `python-jimmy/segmentation-punctuation-evaluation.py`
- Modify: `python-jimmy/test_segmentation_punctuation_evaluation.py`
- Create: `C:\Users\admin\Downloads\sherpa-onnx断句优化测试-0917\tools\speaker_diarization_metrics.py`（运行时拷贝）

- [ ] **Step 1: 为外部指标命令和 Markdown 结论优先级写测试。**

  使用假 `speaker_diarization_summary.json` 和两个小型 `speaker_diarization_boundary_details.csv`，断言：

  ```python
  def test_winner_prefers_boundary_hit_rate_then_lower_der(self):
      winner = module.select_winner({
          "baseline": {"hit_rate": 0.80, "der": 0.10},
          "streaming": {"hit_rate": 0.80, "der": 0.08},
      })
      self.assertEqual(winner, "streaming")

  def test_boundary_difference_keeps_text_and_both_matched_times(self):
      row = module.make_boundary_difference(
          reference_boundary_ms=5000, baseline_match_ms=None, streaming_match_ms=5100,
          tolerance_ms=500, before_text="甲", after_text="乙",
      )
      self.assertEqual(row["classification"], "streaming_only_hit")
      self.assertEqual(row["after_text"], "乙")
  ```

- [ ] **Step 2: 运行测试，确认外部指标集成函数尚未定义。**

  Run:

  ```powershell
  python -m unittest python-jimmy/test_segmentation_punctuation_evaluation.py -v
  ```

  Expected: 新测试因 `select_winner` / `make_boundary_difference` 缺失失败；此前行为保持通过。

- [ ] **Step 3: 实现指标调用、横向表与报告。**

  实现 `install_metrics_tool(source, tools_dir)`，仅当 `source.name == "speaker_diarization_metrics.py"` 且 SHA-256 已记录时复制该单文件。实现 `run_speaker_metrics()`，调用：

  ```python
  command = [
      sys.executable, str(tools_dir / "speaker_diarization_metrics.py"),
      "--results-dir", str(normalized_result_root),
      "--labels-dir", str(input_labels_dir),
      "--output-dir", str(metrics_dir),
      "--boundary-tolerance-ms", "500", "--collar-ms", "500",
  ]
  completed = subprocess.run(command, text=True, capture_output=True, check=False)
  (metrics_dir / "speaker_metrics.stdout.log").write_text(completed.stdout, encoding="utf-8")
  (metrics_dir / "speaker_metrics.stderr.log").write_text(completed.stderr, encoding="utf-8")
  ```

  返回码非零时必须保留两个日志并把该 scheme 记录为 `metrics_failed`；只要 `speaker_diarization_summary.json` 缺失就不可计算胜者。`make_boundary_difference()` 从两方案 CSV 按 `文件ID` / reference boundary 合并，分类为 `both_hit`、`baseline_only_hit`、`streaming_only_hit`、`both_miss`，并从相邻 reference label 段提取前后文本。写出 `05_comparison/boundary_differences.csv` 和 `comparison.json`。

  `write_report()` 必须逐项写入实际命令、models 的路径和 SHA、每条音频的音频时长/运行耗时/RTF/segment count/speaker distribution、WER、DER/miss/false-alarm/confusion/命中率、差异边界表最多前 50 行、Diart JSON 摘要、失败与限制。只有两条音频都被成功评价时，才根据 `select_winner()` 生成“本次数据集排名”；否则报告必须写“未形成有效总体排名”。命中率差值绝对值小于 `0.01` 时视为相近，使用 DER 决胜；DER 仍相同则输出“无明显差异”。

- [ ] **Step 4: 运行所有新增测试与指标脚本自测。**

  Run:

  ```powershell
  python -m unittest python-jimmy/test_segmentation_punctuation_evaluation.py -v
  python 'D:\code\proMax\tz-llm-sdk\agent-sdk-test\python\test_speaker_diarization_metrics.py'
  ```

  Expected: 新评测测试全部通过。第二条命令通过或清楚地因 Python 环境缺少 `pyannote.metrics` 依赖而失败；若失败，将完整依赖错误写入预检报告，且不安装新环境或用不同版本替换指标库。

- [ ] **Step 5: 提交说话人指标和报告汇总。**

  ```powershell
  git add python-jimmy/segmentation-punctuation-evaluation.py python-jimmy/test_segmentation_punctuation_evaluation.py
  git commit -m "feat: report segmentation punctuation comparison"
  ```

### Task 4: 实现 Windows 到固定 Linux 服务器的可审计执行器

**Files:**
- Create: `python-jimmy/run-segmentation-punctuation-evaluation.ps1`
- Modify: `python-jimmy/test_segmentation_punctuation_evaluation.py`

- [ ] **Step 1: 为 PowerShell 命令生成和停止策略写纯文本测试。**

  测试 Python 的 `render_remote_run_script()` 只允许生成以下远端变更位置：`/speech_store/jimmy/k2_origin/sherpa-onnx/` 及其内部的 `evaluation-artifacts/`；不得包含 `git reset --hard`、`git clean`、`--force` 或外部远端输出目录。还要断言远端命令同时运行 baseline 与 streaming，并为两者传相同的 `--asr-dir --vad-dir --speaker-dir --segmentation-dir --num-clusters -1 --cluster-threshold 0.6`。

- [ ] **Step 2: 运行测试，确认远端渲染函数未实现。**

  Run:

  ```powershell
  python -m unittest python-jimmy/test_segmentation_punctuation_evaluation.py -v
  ```

  Expected: 对 `render_remote_run_script` 的测试失败，且没有连接 SSH。

- [ ] **Step 3: 实现 PowerShell 编排和远端命令渲染。**

  PowerShell 参数固定如下：

  ```powershell
  param(
    [string]$OutputRoot = 'C:\Users\admin\Downloads\sherpa-onnx断句优化测试-0917',
    [string]$MetricsSource = 'D:\code\proMax\tz-llm-sdk\agent-sdk-test\python\speaker_diarization_metrics.py',
    [string]$RemoteAsrDir,
    [string]$RemoteVadDir,
    [string]$RemoteSpeakerDir,
    [string]$RemoteSegmentationDir,
    [switch]$SkipPush
  )
  ```

  四个 remote model 参数必须全部显式给出；脚本先用 `sha256sum` 获取每个角色实际模型文件的 hash，再和 Windows default model file 的 hash 比较。不相同、找不到模型、找不到解释器、远端分支/工作树不匹配时，写 `<output>/05_comparison/preflight_failure.md` 后退出非零，绝不开始推理。

  依次实施：

  1. 调用 Python `prepare` 固化输入；调用 `validate-manifest` 但在 remote inventory 补齐前允许模型字段为 `pending`；复制指标脚本到 `<output>/tools/`；保存 Windows `git status --short --branch`、`git rev-parse HEAD`、PowerShell 版本和本地模型 SHA。
  2. 检查完整 diff、仅提交上述任务文件；普通 `git push jimmy HEAD:<current-branch>`，验证 `jimmy/<branch>` 等于本地 HEAD。`-SkipPush` 只能用于已验证 `jimmy/<branch>` 与 HEAD 相同的重跑。
  3. SSH 只读预检 `/speech_store/jimmy/k2_origin/sherpa-onnx`：远端 worktree 必须 clean、当前分支必须和 Windows 一致。成功后只在该 repository 内建立 `evaluation-artifacts/input`、`evaluation-artifacts/raw`、`evaluation-artifacts/logs`。
  4. 使用 SCP 将 `01_input/pcm` 和 `01_input/labels` 复制到 remote repository 的 `evaluation-artifacts/input`；远端算 hash，并与 local manifest 输入 hash 对比。
  5. 在远端仓库执行 `git pull --ff-only`；在既有 `build` 目录用以下配置运行 CMake 和 `make -j"$(nproc)"`：

     ```bash
     cmake -DSHERPA_ONNX_ENABLE_PYTHON=ON -DBUILD_SHARED_LIBS=ON \
       -DSHERPA_ONNX_ENABLE_CHECK=OFF -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
       -DSHERPA_ONNX_ENABLE_C_API=OFF -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF ..
     make -j"$(nproc)"
     ```

     然后 `export PYTHONPATH="$PWD/lib:/speech_store/jimmy/k2_origin/sherpa-onnx/sherpa-onnx/python${PYTHONPATH:+:$PYTHONPATH}"`，运行 `python3 -m unittest discover -s python-jimmy/offline-long-audio-pipeline/tests -v` 并要求通过，确认实际使用新 extension。
  6. 对每条音频分别运行两条由 manifest 渲染的命令，输出到远端 `evaluation-artifacts/raw/baseline` 与 `evaluation-artifacts/raw/streaming`。每个命令用 `/usr/bin/time -v`，stdout/stderr/time 分别记录；每种方案额外传 `--run-label baseline` 或 `--run-label streaming`。任一命令失败后不运行其后续音频，但回收已有文件并继续生成失败报告。
  7. SCP 整个远端 `evaluation-artifacts` 至 Windows `<output>/remote-artifacts`，再调用 Python `normalize-results`, `compute-wer`, `summarize`。不以 stdout 成功替代 output file 检查：每个成功运行须有 `run_metadata.json`、`result.json`、`result.txt`、远端日志和对应规范化的 `*_asr.txt`。

- [ ] **Step 4: 静态验证 PowerShell 脚本和 Python 命令渲染测试。**

  Run:

  ```powershell
  [void][scriptblock]::Create((Get-Content -Raw python-jimmy/run-segmentation-punctuation-evaluation.ps1))
  python -m unittest python-jimmy/test_segmentation_punctuation_evaluation.py -v
  ```

  Expected: 第一条命令无语法异常；第二条命令通过。没有提供四个 remote model dir 时运行脚本应在连接 SSH 前输出明确缺失参数错误。

- [ ] **Step 5: 提交远程执行器。**

  ```powershell
  git add python-jimmy/run-segmentation-punctuation-evaluation.ps1 python-jimmy/segmentation-punctuation-evaluation.py python-jimmy/test_segmentation_punctuation_evaluation.py
  git commit -m "feat: automate remote segmentation evaluation"
  ```

### Task 5: 进行 Diart 固定版本源码分析（不运行模型）

**Files:**
- Create: `python-jimmy/diart-source-analysis.py`
- Modify: `python-jimmy/test_segmentation_punctuation_evaluation.py`

- [ ] **Step 1: 编写本地 checkout 分析器测试。**

  在临时目录构造只含 `README.md`、`diart/blocks/diarization.py` 和 `diart/blocks/clustering.py` 的 fake checkout。测试分析器在这些文件含目标符号时写入 JSON，并在 `git rev-parse HEAD` 失败时拒绝生成“已固定版本”的分析：

  ```python
  def test_diart_analysis_records_commit_and_required_logic_locations(self):
      report = module.analyze_diart_checkout(fake_checkout)
      self.assertEqual(report["commit"], "0123456789abcdef")
      self.assertIn("aggregation", report["locations"])
      self.assertIn("online_clustering", report["locations"])
  ```

- [ ] **Step 2: 运行测试，确认 Diart 分析器尚未定义。**

  Run:

  ```powershell
  python -m unittest python-jimmy/test_segmentation_punctuation_evaluation.py -v
  ```

  Expected: 新测试因缺失 `analyze_diart_checkout` 失败；无网络访问。

- [ ] **Step 3: 实现固定 commit 的源码读取和结构化说明。**

  CLI 参数为 `--checkout`, `--output-dir`；PowerShell 在第一次运行时执行：

  ```powershell
  git clone https://github.com/juanmc2005/diart.git "$OutputRoot\04_diart_source_analysis\source"
  ```

  若 source 已存在，要求其为 Git checkout；记录 `git remote get-url origin`、`git rev-parse HEAD`、`git status --porcelain` 和 `LICENSE` 的 SHA-256。Python 分析器只能读取 checkout，不可 import `diart` 或执行模型。以候选文件和符号搜索以下概念，找不到任意一个时把分析标为 `incomplete`：

  - `SpeakerSegmentation`, `OverlappedSpeechPenalty`, `SpeakerEmbedding`, `DelayedAggregation`；
  - `OnlineSpeakerDiarization`, `OnlineSpeakerClustering`, `cannot_link`；
  - rolling buffer、latency、step、aggregation strategy。

  输出 `diart_analysis.json` 和 `diart_analysis.md`，每项列出仓库相对路径、行号、函数/类名、压缩的逻辑释义和它对应/不对应 sherpa 新 API 的部分。Markdown 明确写出：Diart 在同一在线闭环中做 local tracks 的聚合、overlap-aware embedding 和 incremental global clustering；本项目 streaming 方法只融合 speaker count / 单人换人边界，之后继续用现有 Titanet 聚类。不得在 JSON 或报告中捏造当前 checkout 不包含的函数。

- [ ] **Step 4: 运行 Diart 分析器测试。**

  Run:

  ```powershell
  python -m unittest python-jimmy/test_segmentation_punctuation_evaluation.py -v
  ```

  Expected: 所有本地单元测试通过；测试不需要 clone 或安装 Diart。

- [ ] **Step 5: 提交 Diart 分析工具。**

  ```powershell
  git add python-jimmy/diart-source-analysis.py python-jimmy/test_segmentation_punctuation_evaluation.py
  git commit -m "feat: analyze diart segmentation logic"
  ```

### Task 6: 远程真实评测、回收结果与最终验收

**Files:**
- Modify: `docs/superpowers/specs/2026-09-17-pyannote-segmentation-punctuation-evaluation-design.md`
- Create: `C:\Users\admin\Downloads\sherpa-onnx断句优化测试-0917\报告.md`（运行时）
- Create: `C:\Users\admin\Downloads\sherpa-onnx断句优化测试-0917\05_comparison\comparison.json`（运行时）

- [ ] **Step 1: 完整本地代码验证并检查仅预期文件待提交。**

  Run:

  ```powershell
  python -m unittest python-jimmy/test_segmentation_punctuation_evaluation.py -v
  git diff --check
  git status --short
  ```

  Expected: 新测试全通过，`git diff --check` 无输出。若 `git status` 含用户未关联改动，停止并分离，不将其加入任何 commit。

- [ ] **Step 2: 生成 Diart 分析。**

  Run:

  ```powershell
  $root = 'C:\Users\admin\Downloads\sherpa-onnx断句优化测试-0917'
  $source = Join-Path $root '04_diart_source_analysis\source'
  if (-not (Test-Path (Join-Path $source '.git'))) { git clone https://github.com/juanmc2005/diart.git $source }
  python python-jimmy/diart-source-analysis.py --checkout $source --output-dir (Join-Path $root '04_diart_source_analysis')
  ```

  Expected: `diart_analysis.json`、`diart_analysis.md` 存在且记录 commit；不出现音频推理输出。

- [ ] **Step 3: 运行完整远程对照。**

  使用 Task 4 的 preflight 产生的 `<output>\manifest.json` 中 `models.remote` 四个已验证路径，赋给以下 PowerShell 变量；变量值必须是 `Test-Path` 和 `sha256sum` 已验证的实际目录，而不是重新手填的模型名。然后执行：

  ```powershell
  $manifest = Get-Content -Raw 'C:\Users\admin\Downloads\sherpa-onnx断句优化测试-0917\manifest.json' | ConvertFrom-Json
  $remoteAsrDir = $manifest.models.remote.asr.directory
  $remoteVadDir = $manifest.models.remote.vad.directory
  $remoteSpeakerDir = $manifest.models.remote.speaker.directory
  $remoteSegmentationDir = $manifest.models.remote.segmentation.directory
  .\python-jimmy\run-segmentation-punctuation-evaluation.ps1 `
    -RemoteAsrDir $remoteAsrDir `
    -RemoteVadDir $remoteVadDir `
    -RemoteSpeakerDir $remoteSpeakerDir `
    -RemoteSegmentationDir $remoteSegmentationDir
  ```

  Expected: manifest 指明两方案仅有入口和 streaming 的 `--segmentation-chunk-ms 32` 差异；远端 CMake / Make / 114 个 pipeline tests 成功；两种方案各对两条音频产生原始结果；所有日志和结果均已回收。模型 SHA 或 remote prerequisite 不通过时，不运行本步骤的推理；PowerShell 仍生成带精确原因的 `报告.md`。

- [ ] **Step 4: 验收评测产物及内容。**

  Run:

  ```powershell
  $root = 'C:\Users\admin\Downloads\sherpa-onnx断句优化测试-0917'
  @(
    "$root\manifest.json",
    "$root\报告.md",
    "$root\04_diart_source_analysis\diart_analysis.json",
    "$root\05_comparison\comparison.json",
    "$root\05_comparison\boundary_differences.csv"
  ) | ForEach-Object { if (-not (Test-Path $_)) { throw "Missing expected artifact: $_" } }
  Get-Content -Raw "$root\报告.md"
  ```

  Expected: 报告包含两条音频的 WER、DER、miss、false alarm、confusion、边界 hit/hit-rate、所有实际模型 SHA 和路径、实际命令、运行时间/RTF、Diart commit/分析、胜者或未形成排名的原因。任何失败必须在“失败与限制”章节引用本地或远端日志的绝对路径。

- [ ] **Step 5: 更新设计状态并提交文档/工具的最终状态。**

  将 design 的状态改为“已运行”或“前置条件阻断（包含阶段）”，追加最终 commit 和报告绝对路径。然后：

  ```powershell
  git add docs/superpowers/specs/2026-09-17-pyannote-segmentation-punctuation-evaluation-design.md
  git commit -m "docs: record segmentation evaluation execution"
  git status --short --branch
  ```

  Expected: status clean；运行时下载目录及 remote artifacts 不进入 Git。

## 计划自检

- **Spec coverage:** Task 1/2 固定输入、模型清单、命令公平性和 WER；Task 3 覆盖要求复制的说话人指标、DER 和断句命中率；Task 4 覆盖固定 Linux 构建、远程执行和结果回收；Task 5 完成不实测的 Diart 分析；Task 6 汇总中文报告、验证与版本记录。
- **Fairness guard:** `validate_variant_fairness()` 与 `validate_manifest()` 同时检查 common argv 和模型 SHA，阻止模型/参数差异伪装为断句结论。
- **Failure behavior:** 所有步骤将 stdout/stderr 和已有产物留在结果目录；没有任何步骤采用 force push、reset、clean、删除 remote build 或自动更换模型/参数。
- **Known baseline limitation:** 在 2026-09-17 Windows worktree 的默认 Python 3.10 环境，`python -m unittest discover -s python-jimmy/offline-long-audio-pipeline/tests -v` 已运行 114 项而仅 `test_public_symbols_and_span_shape` 失败，因导入旧安装版 `sherpa_onnx` 缺少 `SpeakerSegmentationConfig`。Task 4 通过远端 build 的 `PYTHONPATH` 重新验证；此问题在报告中作为环境前置条件而非本任务回归。

