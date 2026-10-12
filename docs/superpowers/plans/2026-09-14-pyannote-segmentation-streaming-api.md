# Pyannote Segmentation 流式输入 API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 sherpa-onnx 新增对象级、流式输入的 Pyannote segmentation API，向 C++、C 和 Python 返回 `[start, end, speaker_count, flag]`，并在远程 Linux 用最新 Python 包完成真实 PCM 集成验证。

**Architecture:** `SpeakerSegmentation` 保持 16 kHz PCM 与未完成窗口的有界缓存，按现有 Pyannote 3.0 model metadata 创建 10 s（由 metadata 决定）的滑窗并调用现有 `OfflineSpeakerSegmentationPyannoteModel`。内部 fusion 层分别聚合每帧 `speaker_count` 与无身份的单人换人投票；只有不会再被未来窗口覆盖的帧才可作为不可撤回的 span 发布。C API 和 pybind11 仅镜像该对象的队列式接口，不暴露 local mask、embedding 或 global speaker ID。

**Tech Stack:** C++17、ONNX Runtime、现有 sherpa-onnx Pyannote segmentation model、GoogleTest、C API、pybind11、Python `unittest`/标准库、SSH/SCP 远程 Linux 验证。

---

## 已锁定的文件结构

- `sherpa-onnx/csrc/speaker-segmentation.h` — 公开的 C++ config、span flag、span 值类型和队列式对象 API。
- `sherpa-onnx/csrc/speaker-segmentation.cc` — config 校验、metadata 校验、对象级音频缓存、模型驱动、尾窗补零、结果队列和 PImpl 转发。
- `sherpa-onnx/csrc/speaker-segmentation-fusion.h` — 仅供 core/test 使用的确定性融合单元：powerset decode、count/coverage、local-track 稳定化、换人 voting、span 发布。
- `sherpa-onnx/csrc/speaker-segmentation-fusion.cc` — fusion 单元实现；不依赖 ONNX，供合成输入测试。
- `sherpa-onnx/csrc/speaker-segmentation-test.cc` — GoogleTest 合成 local-mask 单元测试与对象状态契约测试。
- `sherpa-onnx/csrc/CMakeLists.txt` — 将新 core 源和测试注册到现有 speaker-diarization 条件构建分支。
- `sherpa-onnx/c-api/c-api.h` — C config、flag enum、span、opaque handle 和所有函数声明/ownership 文档。
- `sherpa-onnx/c-api/c-api.cc` — C config 到 C++ config 的转换与 API 转发。
- `sherpa-onnx/c-api/sherpa-onnx-symbols-c.exp` — 导出新 C API 符号。
- `sherpa-onnx/python/csrc/speaker-segmentation.h` / `.cc` — Python config/span/class pybind11 绑定。
- `sherpa-onnx/python/csrc/offline-speaker-diarization.h` / `.cc` — 将现有 segmentation model config binding 提升为可复用函数，避免重复注册。
- `sherpa-onnx/python/csrc/sherpa-onnx.cc` / `CMakeLists.txt` — 模块注册并参与 Python extension 编译。
- `sherpa-onnx/python/sherpa_onnx/__init__.py` — 公开导入新 Python 名称。
- `python-jimmy/test-pyannote-segmentation-streaming.py` — 真实 PCM 的 32 ms 输入测试、JSONL/JSON 结果、统计和退出码。
- `python-jimmy/offline-long-audio-pipeline-asr-speaker-segmentation.py` — 复用现有 ASR/VAD/上层 global embedding + clustering 流程，但由新 span 边界进行断句的对照 runner；不修改原 baseline runner。
- `python-jimmy/offline-long-audio-pipeline/tests/test_streaming_segmentation_runner.py` — `unittest` 覆盖报告解析、flag 边界和 baseline/new 对照摘要，不依赖网络、模型或 pytest。
- `docs/superpowers/specs/2026-09-14-pyannote-segmentation-streaming-api-design.md` — 已确认设计，补充远程验证约束。

## Task 1: 为融合语义建立可脱离 ONNX 的失败测试

**Files:**
- Create: `sherpa-onnx/csrc/speaker-segmentation-fusion.h`
- Create: `sherpa-onnx/csrc/speaker-segmentation-fusion.cc`
- Create: `sherpa-onnx/csrc/speaker-segmentation-test.cc`
- Modify: `sherpa-onnx/csrc/CMakeLists.txt: speaker diarization sources and SHERPA_ONNX_ENABLE_TESTS list`

- [ ] **Step 1: 写入 GoogleTest 用例，先固定 count 融合与 flag 语义。**

  在 `speaker-segmentation-test.cc` 写入以下最小合成测试组；每个 frame 以 `uint8_t` local mask 表示，`0b001`/`0b010` 是不同 local track，`0b011` 是 overlap：

  ```cpp
  TEST(SpeakerSegmentationFusion, CountsUseCoverageAverageAndRound) {
    SpeakerSegmentationFusion fuser(/*frame_shift_samples=*/270,
                                    /*window_shift_samples=*/16000,
                                    /*sample_rate=*/16000,
                                    /*min_on=*/0, /*min_off=*/0,
                                    /*threshold=*/0.5F);
    fuser.AddWindow(/*start_frame=*/0, {0b001, 0b011});
    fuser.AddWindow(/*start_frame=*/0, {0b001, 0b001});
    auto frames = fuser.FinalizeBefore(/*frame_exclusive=*/2);
    ASSERT_EQ(frames.size(), 2);
    EXPECT_EQ(frames[0].speaker_count, 1);
    EXPECT_EQ(frames[1].speaker_count, 2);  // round(1.5) == 2
  }

  TEST(SpeakerSegmentationFusion, ReversedLocalTracksVoteForOneChange) {
    // W0: bit0 -> bit1; W1: bit1 -> bit0. Both must create the same
    // identity-free 1 -> 1 boundary instead of cancelling each other.
  }

  TEST(SpeakerSegmentationFusion, MinDurationsOnlyChangeEvidence) {
    // Assert short on islands and short off gaps disappear from change votes,
    // while raw count frames remain unchanged.
  }

  TEST(SpeakerSegmentationFusion, SpanFlagsDescribeRightBoundary) {
    // Assert CONTINUE, SPEAKER_COUNT_CHANGED, SINGLE_SPEAKER_CHANGED and
    // INPUT_FINISHED (including INPUT_FINISHED | count-change) occur on end.
  }
  ```

- [ ] **Step 2: 只配置目标并运行测试，确认当前失败原因是源文件/类型不存在。**

  Run:

  ```powershell
  cmake --build build --target speaker-segmentation-test
  ```

  Expected: build failure that names the missing `speaker-segmentation-*` source or declaration, not a silent skipped test.

- [ ] **Step 3: 定义 fusion 的明确内部契约。**

  在 `speaker-segmentation-fusion.h` 定义非导出的 `SpeakerSegmentationFusion` 和如下值类型：

  ```cpp
  struct FinalizedSpeakerFrame {
    int64_t frame_index;
    int32_t speaker_count;  // 0, 1, or 2
    bool single_speaker_changed_before;
  };

  struct SpeakerSegmentationFusionConfig {
    int32_t frame_shift_samples;
    int32_t window_shift_samples;
    int32_t sample_rate;
    float min_duration_on;
    float min_duration_off;
    float change_vote_threshold;
  };
  ```

  `AddWindow(start_frame, raw_local_masks)` 必须：保留 raw mask 的 `popcount` 用于 count；对每条 local bit 单独应用 on/off 稳定化，仅从稳定后的 local bit 序列中提取 `one-active-bit A -> one-active-bit B, A != B` 证据；对候选中心 frame ±1 个 model-frame 投票；所有覆盖该 frame 的窗口均增加 change coverage。`FinalizeBefore(frame_exclusive)` 仅返回已不可能再被后续窗口覆盖的 frame，并从内部累积器删去该前缀。

- [ ] **Step 4: 实现最小 fusion。**

  在 `.cc` 中实现以下不可替代规则：

  ```cpp
  const int32_t raw_count =
    static_cast<int32_t>((raw_mask & 1) + ((raw_mask >> 1) & 1) +
                         ((raw_mask >> 2) & 1));
  count_sum[global_frame] += std::min(raw_count, 2);
  ++count_coverage[global_frame];
  fused_count = std::clamp(
      static_cast<int32_t>(std::round(count_sum / count_coverage)), 0, 2);
  changed = change_vote / change_coverage >= change_vote_threshold;
  ```

  `min_duration_on` 使用 frame 数 `ceil(seconds * sample_rate / frame_shift_samples)` 删除短 active island；`min_duration_off` 在同一 local bit 中闭合短 inactive gap。不得把稳定化后的 mask 回写 count 累积器。相距少于 `min_duration_on` 的确认 change event 只保留 vote ratio 较高的局部峰值，相同支持率时保留更早的 frame。

- [ ] **Step 5: 重新运行 fusion 测试。**

  Run:

  ```powershell
  cmake --build build --target speaker-segmentation-test; .\build\bin\speaker-segmentation-test.exe
  ```

  Expected: 所有四组 GoogleTest 通过；若本地 build 配置不存在，则只记录该前置条件并在 Task 7 的 Linux 构建中执行同一测试 target，不安装任何工具。

- [ ] **Step 6: 提交融合层。**

  ```powershell
  git add sherpa-onnx/csrc/speaker-segmentation-fusion.h sherpa-onnx/csrc/speaker-segmentation-fusion.cc sherpa-onnx/csrc/speaker-segmentation-test.cc sherpa-onnx/csrc/CMakeLists.txt
  git commit -m "feat: add speaker segmentation fusion"
  ```

## Task 2: 实现对象级 Pyannote window runner 和不可撤回 span 队列

**Files:**
- Create: `sherpa-onnx/csrc/speaker-segmentation.h`
- Create: `sherpa-onnx/csrc/speaker-segmentation.cc`
- Modify: `sherpa-onnx/csrc/CMakeLists.txt`
- Modify: `sherpa-onnx/csrc/speaker-segmentation-test.cc`

- [ ] **Step 1: 添加失败测试，锁定输入、尾窗和对象隔离。**

  增加 fake-model seam：`SpeakerSegmentation` 的私有 implementation 接受只在 test 编译中使用的 segmentation-forward callback。写入测试：

  ```cpp
  TEST(SpeakerSegmentation, ChunkedAndOneShotHaveSameFinalSpans);
  TEST(SpeakerSegmentation, TenSecondFirstWindowPublishesFirstOneSecond);
  TEST(SpeakerSegmentation, InputFinishedPadsTailButClipsToAudioEnd);
  TEST(SpeakerSegmentation, ResetAndTwoObjectsDoNotShareState);
  TEST(SpeakerSegmentation, RejectsAcceptAfterInputFinishedUntilReset);
  ```

  第一项用同一 12.3 s 合成 PCM 分别一次性和每 512 samples（32 ms）输入，断言全部 `start/end/count/flag` 完全相等；第二项断言满第一个 metadata window 时结果队列出现 `[0, window_shift, 1, CONTINUE]`。

- [ ] **Step 2: 运行该目标并确认失败。**

  Run:

  ```powershell
  cmake --build build --target speaker-segmentation-test; .\build\bin\speaker-segmentation-test.exe --gtest_filter=SpeakerSegmentation.*
  ```

  Expected: tests fail because public API、window runner 或 fake-model seam 尚未实现。

- [ ] **Step 3: 定义 C++ 公共 API 和 validation。**

  在 `speaker-segmentation.h` 定义：

  ```cpp
  enum SpeakerSegmentationSpanFlag : int32_t {
    kSpeakerSegmentationContinue = 0,
    kSpeakerSegmentationSpeakerCountChanged = 1 << 0,
    kSpeakerSegmentationSingleSpeakerChanged = 1 << 1,
    kSpeakerSegmentationInputFinished = 1 << 2,
  };

  struct SpeakerSegmentationConfig {
    OfflineSpeakerSegmentationModelConfig model;
    float min_duration_on = 0.30F;
    float min_duration_off = 0.50F;
    float change_vote_threshold = 0.50F;
    bool Validate() const;
    std::string ToString() const;
  };

  struct SpeakerSegmentationSpan { float start; float end; int32_t speaker_count; int32_t flag; };
  ```

  `Validate()` 必须调用 `model.Validate()`，拒绝负 duration 和不在 `(0, 1]` 的 threshold。创建时读取 `OfflineSpeakerSegmentationPyannoteModel::GetModelMetaData()`，仅接受 `num_speakers == 3`、`num_classes == 7`、`powerset_max_classes == 2`、所有时间参数正值；失败时输出具体 metadata 字段和值。

- [ ] **Step 4: 实现滑窗和发布规则。**

  `AcceptWaveform()` 仅接受 `n >= 0`，追加至对象私有缓存；只要 `next_window_start + window_size <= received_samples`，就将恰好 `window_size` 个 samples 传给现有模型，powerset argmax 解码为 3-bit mask，交给 fusion，然后将 `next_window_start` 增加 metadata `window_shift`。处理起点为 `s` 的窗口后，finalize `[s, s + window_shift)` 对应的 model-frame 前缀；这正是未来窗口起点必然大于该区域、不能再覆盖它的时刻，产生约一个 model-window 的延迟。

  将 finalized frames 合并为半开 spans：count 变化在前 span 的 `end` 置 `kSpeakerSegmentationSpeakerCountChanged`；确认的 `1 -> 1` change 在前 span 的 `end` 置 `kSpeakerSegmentationSingleSpeakerChanged`；没有边界但达到一个 window shift 的已确定 run 置 `kSpeakerSegmentationContinue`。`InputFinished()` 对所有仍未运行且起点小于真实音频长度的 shift-grid 窗口右侧补零，finalize 后把最后 span 截断至 `received_samples / sample_rate`，并在其 `flag` OR `kSpeakerSegmentationInputFinished`。重复 `InputFinished()` 不改变队列。`Reset()` 清空输入、fusion、队列、sample counter 和 finished 状态，不重新加载模型。

- [ ] **Step 5: 运行 unit test，并检查格式。**

  Run:

  ```powershell
  cmake --build build --target speaker-segmentation-test; .\build\bin\speaker-segmentation-test.exe
  git diff --check
  ```

  Expected: `speaker-segmentation-test` 全部通过，且没有 whitespace error。

- [ ] **Step 6: 提交对象 API。**

  ```powershell
  git add sherpa-onnx/csrc/speaker-segmentation.h sherpa-onnx/csrc/speaker-segmentation.cc sherpa-onnx/csrc/speaker-segmentation-test.cc sherpa-onnx/csrc/CMakeLists.txt
  git commit -m "feat: add streaming speaker segmentation api"
  ```

## Task 3: 暴露并测试 C API

**Files:**
- Modify: `sherpa-onnx/c-api/c-api.h: offline speaker diarization section`
- Modify: `sherpa-onnx/c-api/c-api.cc: opaque wrapper declarations and C++ conversions`
- Modify: `sherpa-onnx/c-api/sherpa-onnx-symbols-c.exp`
- Create: `c-api-examples/speaker-segmentation-c-api.c`
- Modify: `c-api-examples/CMakeLists.txt`

- [ ] **Step 1: 写入会失败的 C consumer 示例。**

  `speaker-segmentation-c-api.c` 用 `memset` 零初始化 config，设置 `config.model.pyannote.model`、threads、duration 和 threshold；读取每个 `SherpaOnnxSpeakerSegmentationSpan` 后检查：

  ```c
  assert(span->start < span->end);
  assert(span->speaker_count >= 0 && span->speaker_count <= 2);
  assert((span->flag & ~(SherpaOnnxSpeakerSegmentationSpeakerCountChanged |
                          SherpaOnnxSpeakerSegmentationSingleSpeakerChanged |
                          SherpaOnnxSpeakerSegmentationInputFinished)) == 0);
  ```

  以 512 samples 分块输入，再 `InputFinished()`、drain queue、Destroy。该示例不释放 `Front()` 的返回指针。

- [ ] **Step 2: 用 C API 打开时的构建配置验证失败。**

  Run:

  ```powershell
  cmake -S . -B build-c-api -DSHERPA_ONNX_ENABLE_C_API=ON -DSHERPA_ONNX_ENABLE_PYTHON=OFF -DSHERPA_ONNX_ENABLE_TESTS=ON
  cmake --build build-c-api --target speaker-segmentation-c-api
  ```

  Expected: compile failure cites missing `SherpaOnnxSpeakerSegmentation*` declarations/symbols.

- [ ] **Step 3: 增加 C ABI。**

  在 `c-api.h` 声明 `SherpaOnnxSpeakerSegmentationSpanFlag`、`SherpaOnnxSpeakerSegmentationConfig`、`SherpaOnnxSpeakerSegmentationSpan` 与 opaque handle，并提供 `Create/Destroy/GetSampleRate/AcceptWaveform/InputFinished/Empty/Front/Pop/Reset`。`Front()` 文档必须声明对象拥有指针、调用方不能 free、有效期至下次同一对象 API 调用。

  在 `c-api.cc` 以 `std::unique_ptr<SpeakerSegmentation>` 作为 opaque wrapper，逐字段从 C config 拷贝 model path、threads/debug/provider 和三个新参数，所有 queue 方法直接转发。把九个新增 C 函数列入 `.exp`，保持按字母序位置。

- [ ] **Step 4: 重新编译并运行 C 示例。**

  Run:

  ```powershell
  cmake --build build-c-api --target speaker-segmentation-c-api
  .\build-c-api\bin\speaker-segmentation-c-api.exe --help
  ```

  Expected: C 编译和链接成功；`--help` 输出 model/audio 选项，不读取模型。若当前 Windows 没有可用 CMake/ORT 依赖，只记录未执行原因，禁止安装，并在代码审查中保留该命令供具备依赖的环境执行。

- [ ] **Step 5: 提交 C API。**

  ```powershell
  git add sherpa-onnx/c-api/c-api.h sherpa-onnx/c-api/c-api.cc sherpa-onnx/c-api/sherpa-onnx-symbols-c.exp c-api-examples/speaker-segmentation-c-api.c c-api-examples/CMakeLists.txt
  git commit -m "feat: expose speaker segmentation c api"
  ```

## Task 4: 暴露并测试 Python API

**Files:**
- Create: `sherpa-onnx/python/csrc/speaker-segmentation.h`
- Create: `sherpa-onnx/python/csrc/speaker-segmentation.cc`
- Modify: `sherpa-onnx/python/csrc/sherpa-onnx.cc`
- Modify: `sherpa-onnx/python/csrc/CMakeLists.txt`
- Modify: `sherpa-onnx/python/sherpa_onnx/__init__.py`

- [ ] **Step 1: 写最小 Python API smoke test，先确认 import/属性会失败。**

  在 `python-jimmy/offline-long-audio-pipeline/tests/test_streaming_segmentation_runner.py` 加入：

  ```python
  class SpeakerSegmentationApiTest(unittest.TestCase):
      def test_public_symbols_and_span_shape(self):
          import sherpa_onnx
          self.assertTrue(hasattr(sherpa_onnx, "SpeakerSegmentationConfig"))
          self.assertTrue(hasattr(sherpa_onnx, "SpeakerSegmentation"))
          self.assertTrue(hasattr(sherpa_onnx, "SpeakerSegmentationSpan"))
  ```

- [ ] **Step 2: 运行测试确认失败。**

  Run:

  ```powershell
  python -m unittest python-jimmy.offline-long-audio-pipeline.tests.test_streaming_segmentation_runner.SpeakerSegmentationApiTest
  ```

  Expected: failure only because新符号尚未绑定；如当前解释器无法 import 当地 build extension，记录该环境限制，不安装包，并在 Task 7 使用 remote build extension 运行。

- [ ] **Step 3: 实现 pybind11 绑定。**

  先将 `offline-speaker-diarization.cc` 中当前 static 的 `PybindOfflineSpeakerSegmentationModelConfig` 提升为在对应 `.h` 声明的可复用函数；从 `PybindOfflineSpeakerDiarization` 删除其内部重复调用。在 module 初始化的 diarization enabled 分支中按顺序调用一次 config binding、`PybindSpeakerSegmentation` 和 `PybindOfflineSpeakerDiarization`。`speaker-segmentation.cc` 只消费已注册的 model config，并绑定 config 的三个字段、只读 span 四字段和 `SpeakerSegmentation`。`accept_waveform` 接受 `std::vector<float>`；`front` 用按值 lambda 返回 copy；`accept_waveform` 与 `input_finished` 使用 `py::call_guard<py::gil_scoped_release>()`。`empty`、`pop`、`reset` 和 `sample_rate` 与 VAD 命名一致。

- [ ] **Step 4: 注册并导出 Python 名称。**

  在 Python CMake sources 加入 `.cc`；在 module initialization 的 diarization enabled 分支调用 `PybindSpeakerSegmentation(&m)`；在 `__init__.py` 的 import list 加入 `SpeakerSegmentationConfig`、`SpeakerSegmentationSpan`、`SpeakerSegmentation` 和三个非零 flag 常量（`CONTINUE` 为 0 的模块常量也一并导出）。禁用 diarization 时沿用现有 empty-symbol 模式，确保 import 行为一致。

- [ ] **Step 5: 运行 smoke test 与 Python syntax check。**

  Run:

  ```powershell
  python -m py_compile python-jimmy\offline-long-audio-pipeline\tests\test_streaming_segmentation_runner.py
  python -m unittest python-jimmy.offline-long-audio-pipeline.tests.test_streaming_segmentation_runner
  ```

  Expected: syntax 成功；在 extension 可用的环境中 unittest 通过。

- [ ] **Step 6: 提交 Python API。**

  ```powershell
  git add sherpa-onnx/python/csrc/speaker-segmentation.h sherpa-onnx/python/csrc/speaker-segmentation.cc sherpa-onnx/python/csrc/offline-speaker-diarization.h sherpa-onnx/python/csrc/offline-speaker-diarization.cc sherpa-onnx/python/csrc/sherpa-onnx.cc sherpa-onnx/python/csrc/CMakeLists.txt sherpa-onnx/python/sherpa_onnx/__init__.py python-jimmy/offline-long-audio-pipeline/tests/test_streaming_segmentation_runner.py
  git commit -m "feat: add python speaker segmentation api"
  ```

## Task 5: 新增 PCM 流式验证与 pipeline 对照脚本

**Files:**
- Create: `python-jimmy/test-pyannote-segmentation-streaming.py`
- Create: `python-jimmy/offline-long-audio-pipeline-asr-speaker-segmentation.py`
- Modify: `python-jimmy/offline-long-audio-pipeline/tests/test_streaming_segmentation_runner.py`

- [ ] **Step 1: 用 `unittest` 写报告文件契约的失败测试。**

  测试一个临时目录中的 synthetic spans，断言 runner 的 report writer 输出：`spans.jsonl` 每行含 `start/end/speaker_count/flag`；`summary.json` 含 `audio_seconds/span_count/count_0_seconds/count_1_seconds/count_2_seconds/single_speaker_change_count`; `report.md` 含 baseline/new 的 WER、片段数和 multi/overlap 断开表。对 `CONTINUE` 断言报告将相邻可拼接 span 合并为 display run，而 nonzero boundary flag 保留为切分点。

- [ ] **Step 2: 运行 unittest，确认报告模块不存在而失败。**

  Run:

  ```powershell
  python -m unittest python-jimmy.offline-long-audio-pipeline.tests.test_streaming_segmentation_runner
  ```

  Expected: import failure names新 runner/report helper。

- [ ] **Step 3: 实现直接 API 测试脚本。**

  `test-pyannote-segmentation-streaming.py` 的 argparse 必须包含：

  ```text
  --input-dir --output-dir --model --sample-rate --channels --sample-width
  --chunk-ms --num-threads --min-duration-on --min-duration-off
  --change-vote-threshold
  ```

  默认 `--chunk-ms=32`、`--sample-rate=16000`、`--channels=1`、`--sample-width=2`、`--num-threads=4`、duration/threshold 采用设计默认值。递归处理 `*.pcm`，以 little-endian signed 16-bit PCM 转 `float32 [-1, 1]`，每个 512 samples 调用 `accept_waveform`，每次调用后 drain queue，最后 `input_finished` 后再 drain。对每个 PCM 写 `<stem>.spans.jsonl` 和 `<stem>.summary.json`；一次 run 写总 `summary.json`。任何 span overlap、倒序、count 越界或非末尾 span 带 `INPUT_FINISHED` 必须返回 exit code 2。

- [ ] **Step 4: 实现对照 runner，保持 baseline 未修改。**

  `offline-long-audio-pipeline-asr-speaker-segmentation.py` 接受 existing baseline 的模型目录/音频参数及 `--segmentation-model`。它从新 API 读取 spans：`SPEAKER_COUNT_CHANGED` 和 `SINGLE_SPEAKER_CHANGED` 是候选切分边界，`CONTINUE` 仅拼接；之后复用现有 pipeline 的 VAD、ASR、global embedding + clustering 模块，生成同格式标注。它不得调用 `OfflineSpeakerDiarization`，不得把 local speaker index 解释成 global speaker ID。`--baseline-run-dir` 指向原 `offline-long-audio-pipeline-asr-speaker.py` 结果时，读取双方标注并将 WER、最终片段数量、标注中 multi/overlap 的范围及断开次数写到 `comparison.json` 与 `report.md`。

- [ ] **Step 5: 重新运行 unittest 和脚本 help。**

  Run:

  ```powershell
  python -m unittest python-jimmy.offline-long-audio-pipeline.tests.test_streaming_segmentation_runner
  python python-jimmy\test-pyannote-segmentation-streaming.py --help
  python python-jimmy\offline-long-audio-pipeline-asr-speaker-segmentation.py --help
  ```

  Expected: unittest 通过，两个 help 退出 0；不会下载模型、运行模型或创建环境。

- [ ] **Step 6: 提交测试及对照脚本。**

  ```powershell
  git add python-jimmy/test-pyannote-segmentation-streaming.py python-jimmy/offline-long-audio-pipeline-asr-speaker-segmentation.py python-jimmy/offline-long-audio-pipeline/tests/test_streaming_segmentation_runner.py
  git commit -m "test: add streaming segmentation pipeline runner"
  ```

## Task 6: 审查、提交最终文档并准备远程 Linux 验证

**Files:**
- Modify: `docs/superpowers/specs/2026-09-14-pyannote-segmentation-streaming-api-design.md`
- Create: `docs/superpowers/plans/2026-09-14-pyannote-segmentation-streaming-api.md`

- [ ] **Step 1: 执行本地不可下载检查和完整 diff review。**

  Run:

  ```powershell
  git status --short
  git diff --check HEAD~5..HEAD
  rg -n -i "\b(TODO|TBD|FIXME|XXX)\b|pip install|conda create|venv|python -m pip" sherpa-onnx python-jimmy docs
  ```

  Expected: intended files only；没有 whitespace error；没有任何新增的自动安装、下载或环境创建行为。

- [ ] **Step 2: 执行可用的本地验证矩阵。**

  Run:

  ```powershell
  cmake --build build --target speaker-segmentation-test
  .\build\bin\speaker-segmentation-test.exe
  python -m unittest python-jimmy.offline-long-audio-pipeline.tests.test_streaming_segmentation_runner
  python -m py_compile python-jimmy\test-pyannote-segmentation-streaming.py python-jimmy\offline-long-audio-pipeline-asr-speaker-segmentation.py
  ```

  Expected: each available command exits 0; unavailable pre-existing local dependencies are listed exactly and are not remedied by installation.

- [ ] **Step 3: 提交文档状态及 implementation plan。**

  ```powershell
  git add docs/superpowers/specs/2026-09-14-pyannote-segmentation-streaming-api-design.md docs/superpowers/plans/2026-09-14-pyannote-segmentation-streaming-api.md
  git commit -m "docs: plan streaming segmentation implementation"
  ```

## Task 7: 按固定 remote-build-test 流程构建、执行真实 PCM 和回拷报告

**Files:**
- Remote test input/output only: `/speech_store/jimmy/ai_testset/pyannote-segmentation-streaming-api/`
- Windows result only: `C:\Users\admin\Downloads\pyannote-segmentaion测试`

- [ ] **Step 1: 发布最终 commit，严禁 force push。**

  在 Windows worktree 运行：

  ```powershell
  git status --short
  $branch = git branch --show-current
  $commit = git rev-parse HEAD
  git push jimmy "HEAD:$branch"
  git rev-parse "jimmy/$branch"
  ```

  Expected: status clean；`$branch` 为 `codex/pyannote-segmentation-streaming-api`；最后一个 commit hash 等于 `$commit`。push 被拒绝或 ref 不匹配时停止，不 force、不重写 ref。

- [ ] **Step 2: 远端只读预检代码分支、模型和解释器。**

  Run:

  ```powershell
  ssh -i "C:\Users\admin\.ssh\codex_ed25519" -p 21022 ps@159.135.196.86 "set -e; cd /speech_store/jimmy/k2_origin/sherpa-onnx; test -z \"`$(git status --porcelain)\"; test \"`$(git branch --show-current)\" = 'codex/pyannote-segmentation-streaming-api'; find /speech_store/jimmy -type f -path '*sherpa-onnx-pyannote-segmentation-3-0/model.onnx' -print -quit; command -v python3"
  ```

  Expected: 输出一个既有的远端 model path 和 `python3`。若 model path 为空，立即停止远程测试并报告；不得下载、复制本地模型或新建环境。

- [ ] **Step 3: 同步 PCM 输入到用户授权的远端测试集目录。**

  Run:

  ```powershell
  ssh -i "C:\Users\admin\.ssh\codex_ed25519" -p 21022 ps@159.135.196.86 "mkdir -p /speech_store/jimmy/ai_testset/pyannote-segmentation-streaming-api/input /speech_store/jimmy/ai_testset/pyannote-segmentation-streaming-api/output"
  scp -i "C:\Users\admin\.ssh\codex_ed25519" -P 21022 -r "\\10.88.0.243\xainas\ProMax\测试集\音频测试集\中文\内部会议\双人安静咨询\promax\*" "ps@159.135.196.86:/speech_store/jimmy/ai_testset/pyannote-segmentation-streaming-api/input/"
  ```

  Expected: only the user-authorized testset directory changes outside the remote repository. If Windows cannot access the UNC source, stop and report the exact SCP failure.

- [ ] **Step 4: 快进更新远端分支并按 skill 固定 CMake 参数构建。**

  Run:

  ```powershell
  ssh -i "C:\Users\admin\.ssh\codex_ed25519" -p 21022 ps@159.135.196.86 "set -e; cd /speech_store/jimmy/k2_origin/sherpa-onnx; git pull --ff-only; mkdir -p build; cd build; cmake -DSHERPA_ONNX_ENABLE_PYTHON=ON -DBUILD_SHARED_LIBS=ON -DSHERPA_ONNX_ENABLE_CHECK=OFF -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF -DSHERPA_ONNX_ENABLE_C_API=OFF -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF ..; make -j\`$(nproc)\`; ./bin/speaker-segmentation-test"
  ```

  Expected: `git pull --ff-only`、CMake、`make` 和 GoogleTest target 全部 exit 0。不得使用 `git reset --hard`、`git clean -fd` 或删除 remote build。

- [ ] **Step 5: 使用远端 build 的最新 Python extension 跑 32 ms 真实 PCM。**

  将 Step 2 输出的实际 model path 赋值给 shell variable `MODEL`，再运行（仅在该 path 已存在时）：

  ```bash
  cd /speech_store/jimmy/k2_origin/sherpa-onnx/build
  export PYTHONPATH="$PWD/lib:/speech_store/jimmy/k2_origin/sherpa-onnx/sherpa-onnx/python${PYTHONPATH:+:$PYTHONPATH}"
  python3 /speech_store/jimmy/k2_origin/sherpa-onnx/python-jimmy/test-pyannote-segmentation-streaming.py \
    --input-dir /speech_store/jimmy/ai_testset/pyannote-segmentation-streaming-api/input \
    --output-dir /speech_store/jimmy/ai_testset/pyannote-segmentation-streaming-api/output/new-api \
    --model "$MODEL" --sample-rate 16000 --channels 1 --sample-width 2 \
    --chunk-ms 32 --num-threads 4 --min-duration-on 0.30 --min-duration-off 0.50 \
    --change-vote-threshold 0.50
  ```

  Expected: exit 0 and `summary.json` plus every discovered PCM 的 `.spans.jsonl` / `.summary.json` 存在。对照 runner 只有当远端已有它需要的 ASR、VAD、embedding 模型和 baseline result 时才执行；任何一个缺失即明确报告“new API integration passed; WER comparison not run because <exact path> is missing”，不下载或复制模型。

- [ ] **Step 6: 生成对照报告并回拷 Windows。**

  若对照 prerequisites 已满足，运行新 pipeline runner，输出至远端 `output/comparison/`；否则由直接 API runner 生成仅 segmentation report。随后运行：

  ```powershell
  New-Item -ItemType Directory -Force 'C:\Users\admin\Downloads\pyannote-segmentaion测试' | Out-Null
  scp -i 'C:\Users\admin\.ssh\codex_ed25519' -P 21022 -r 'ps@159.135.196.86:/speech_store/jimmy/ai_testset/pyannote-segmentation-streaming-api/output/*' 'C:\Users\admin\Downloads\pyannote-segmentaion测试\'
  Get-ChildItem -Recurse 'C:\Users\admin\Downloads\pyannote-segmentaion测试' | Select-Object FullName,Length
  ```

  Expected: Windows output contains remote `summary.json`, JSONL spans, report markdown and—若 prerequisites 齐全—`comparison.json`。报告中明确列出 WER、最终片段数和 multi/overlap 断开比较，或明确列出无法执行对照的唯一缺失前置条件。

- [ ] **Step 7: 最终验证和报告。**

  Run:

  ```powershell
  Get-Content 'C:\Users\admin\Downloads\pyannote-segmentaion测试\new-api\summary.json'
  git status --short
  git log -1 --oneline
  ```

  Expected: copied summary 可读、local worktree clean。最终报告必须含：local commit、pushed branch、remote commit、CMake/make/test command 与 exit code、实际远端 model path、回拷路径、new API result 和 WER 对照是否执行/原因。不得把缺失对照前置条件写成成功。