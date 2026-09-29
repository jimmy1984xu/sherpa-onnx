# Speaker Mask Fusion and Clean-Cluster Assignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add aligned Pyannote local-mask fusion with confidence to `SpeakerSegmentation`, use it to improve clean-only global clustering and short-segment speaker assignment, then run a reproducible baseline-vs-optimized evaluation.

**Architecture:** Preserve Pyannote Powerset probabilities through C++ window inference. Align every new window's three local tracks to a session-canonical track order using the overlapping fused timeline, remap the seven Powerset classes, and aggregate probabilities per global frame. Emit fused mask/confidence spans. In Python, derive clean spans from the fused timeline, cluster only continuous clean spans of at least three seconds, and assign non-clean spans by local-mask continuity first and centroid fallback second without allowing them to create clusters.

**Tech Stack:** C++17, ONNX Runtime, pybind11, Python 3, NumPy, unittest/pytest-compatible tests, CMake, remote Linux build via `sherpa-onnx-remote-build-test`.

---

## Task 1: Add regression tests for the public span and fusion semantics

**Files:**
- Modify: `D:\code\opensource\ai\sherpa-onnx\sherpa-onnx\csrc\speaker-segmentation-test.cc`
- Modify: `D:\code\opensource\ai\sherpa-onnx\python-jimmy\offline-long-audio-pipeline\tests\test_speaker_segmentation_pipeline.py`
- Create: `D:\code\opensource\ai\sherpa-onnx\python-jimmy\offline-long-audio-pipeline\tests\test_mask_fusion.py`

- [ ] **Step 1: Add C++ test expectations for fused mask/confidence fields.**

  Extend the test helper output checks so a single-speaker test expects `local_speaker_mask` to be one of the single-bit masks and confidence in `[0, 1]`. Add a two-window injected-forward case where the second window swaps local labels but overlap alignment should keep the same canonical mask.

- [ ] **Step 2: Add Python failing tests for mask-driven short-segment assignment.**

  Add tests with synthetic `SpeechSegment` objects and fake embeddings/clusterer that assert:
  - a 1-second segment with the same local mask as a resolved 5-second segment inherits its speaker ID;
  - a different local mask does not inherit the previous ID;
  - a short/overlap segment never appears in the clusterer's input;
  - output serialization contains `clean_spans`, `local_speaker_mask`, `local_speaker_mask_confidence`, and `speaker_assignment_source`.

- [ ] **Step 3: Run the focused tests before implementation.**

  Run:

  ```powershell
  python -m pytest D:\code\opensource\ai\sherpa-onnx\python-jimmy\offline-long-audio-pipeline\tests\test_mask_fusion.py D:\code\opensource\ai\sherpa-onnx\python-jimmy\offline-long-audio-pipeline\tests\test_speaker_segmentation_pipeline.py -q
  ```

  Expected: the new tests fail because the fields and assignment path do not exist yet. Do not modify production code before recording this red result.

- [ ] **Step 4: Commit the red tests only.**

  ```powershell
  git add sherpa-onnx/csrc/speaker-segmentation-test.cc python-jimmy/offline-long-audio-pipeline/tests/test_speaker_segmentation_pipeline.py python-jimmy/offline-long-audio-pipeline/tests/test_mask_fusion.py
  git commit -m "test: define local mask fusion and clean assignment behavior"
  ```

## Task 2: Preserve Powerset probabilities and align C++ local tracks

**Files:**
- Modify: `D:\code\opensource\ai\sherpa-onnx\sherpa-onnx\csrc\speaker-segmentation-fusion.h`
- Modify: `D:\code\opensource\ai\sherpa-onnx\sherpa-onnx\csrc\speaker-segmentation-fusion.cc`
- Modify: `D:\code\opensource\ai\sherpa-onnx\sherpa-onnx\csrc\speaker-segmentation.cc`
- Modify: `D:\code\opensource\ai\sherpa-onnx\sherpa-onnx\csrc\speaker-segmentation.h`
- Modify: `D:\code\opensource\ai\sherpa-onnx\sherpa-onnx\python\csrc\speaker-segmentation.cc`

- [ ] **Step 1: Define an internal Powerset-window representation.**

  Replace the mask-only forward seam with a representation containing per-frame seven-class probabilities and retain a mask-only adapter for existing unit-test seams. Validate the probability shape `(frames, 7)`, finite values, and normalize logits with stable softmax when the ONNX model returns logits.

- [ ] **Step 2: Implement six-permutation overlap alignment.**

  Keep a bounded canonical-track reference for the recent overlap timeline. For every new window enumerate the six permutations, score each local/global track pairing over valid covered frames using weighted intersection minus disagreement, select the best mapping, and retain the best/second-best margin for confidence. If there is no valid overlap, use identity mapping and mark alignment confidence low.

- [ ] **Step 3: Remap Powerset classes after permutation.**

  For each local class mask, construct its canonical global mask and accumulate its probability into the corresponding canonical Powerset class. This must happen before frame-level window aggregation so the resulting fused class remains one of `000,001,010,100,011,101,110`.

- [ ] **Step 4: Fuse per-global-frame class probabilities.**

  Extend the frame accumulator to retain seven-class probability sums and coverage. Emit the argmax class, its probability as `local_speaker_mask_confidence`, and the converted three-bit `local_speaker_mask`. Do not use bitwise OR across windows and do not derive identity from speaker count alone.

- [ ] **Step 5: Apply post-fusion temporal smoothing and keep existing flags.**

  Run min-duration on/off smoothing on fused masks, split spans on stable mask changes, keep count-change and single-speaker-change flags, and keep input-finished clipping and checkpoint flag write-back behavior.

- [ ] **Step 6: Expose the fields through C++ and Python bindings.**

  Add read-only `local_speaker_mask` and `local_speaker_mask_confidence` properties. Keep old fields and constants unchanged.

- [ ] **Step 7: Run focused C++/Python tests and fix until green.**

  Run the repository's speaker-segmentation unit target and the focused Python tests. Expected: all new tests pass and existing streaming tests remain green.

- [ ] **Step 8: Commit the C++ API and fusion implementation.**

  ```powershell
  git add sherpa-onnx/csrc/speaker-segmentation-fusion.h sherpa-onnx/csrc/speaker-segmentation-fusion.cc sherpa-onnx/csrc/speaker-segmentation.cc sherpa-onnx/csrc/speaker-segmentation.h sherpa-onnx/python/csrc/speaker-segmentation.cc sherpa-onnx/csrc/speaker-segmentation-test.cc
  git commit -m "feat: fuse aligned local speaker masks with confidence"
  ```

## Task 3: Implement clean-span extraction and mask-aware speaker assignment

**Files:**
- Modify: `D:\code\opensource\ai\sherpa-onnx\python-jimmy\offline-long-audio-pipeline\segmentation.py`
- Modify: `D:\code\opensource\ai\sherpa-onnx\python-jimmy\offline-long-audio-pipeline\diarization.py`
- Modify: `D:\code\opensource\ai\sherpa-onnx\python-jimmy\offline-long-audio-pipeline\vad.py`
- Modify: `D:\code\opensource\ai\sherpa-onnx\python-jimmy\offline-long-audio-pipeline\speaker.py`
- Modify: `D:\code\opensource\ai\sherpa-onnx\python-jimmy\offline-long-audio-pipeline\output.py`
- Modify: `D:\code\opensource\ai\sherpa-onnx\python-jimmy\offline-long-audio-pipeline-asr-speaker-segmentation.py`

- [ ] **Step 1: Extend `SpeechSegment` diagnostics.**

  Add fields for `clean_spans`, `local_speaker_mask`, `local_speaker_mask_confidence`, `speaker_assignment_source`, and any local-continuity score required by the assignment routine. Keep existing JSON fields compatible.

- [ ] **Step 2: Build clean spans from atomic timeline intervals.**

  During timeline resolution, retain contiguous single-speaker intervals only when they are not overlap, not unknown, and not contaminated by another track. Store absolute `[start_ms, end_ms]` ranges. Make cluster eligibility require at least one contiguous clean span of at least `3_000 ms`; do not concatenate separated spans across overlap/uncertain ranges for cluster creation.

- [ ] **Step 3: Change embedding extraction to return the source span.**

  Add a helper that extracts from the selected clean span and records the exact range. For excluded segments, retain the existing overlap-safe waveform extraction only for centroid matching; excluded embeddings must never enter the clusterer.

- [ ] **Step 4: Add local-mask continuity assignment.**

  Add a chronological post-cluster pass that searches the previous/next resolved single-speaker segment within a bounded gap. Require compatible masks and confidence above configured thresholds. If the local mask is continuous and the neighboring speaker ID is known, assign that ID and mark `speaker_assignment_source="local_mask_inherit"`. If masks differ or confidence is insufficient, leave the segment for centroid fallback.

- [ ] **Step 5: Keep centroid fallback read-only.**

  Cluster only clean eligible embeddings. Compare all remaining valid embeddings to final normalized centroids; assign only above the existing threshold. Never add excluded embeddings to centroids and never generate a new ID from them.

- [ ] **Step 6: Update the streaming runner to preserve fused fields.**

  Extend span normalization, runtime collection, timeline adaptation, artifacts, and metadata in `offline-long-audio-pipeline-asr-speaker-segmentation.py` so mask and confidence survive from the Python binding to final `result.json` and diagnostic span artifacts.

- [ ] **Step 7: Run the Python unit suite.**

  Run:

  ```powershell
  python -m pytest D:\code\opensource\ai\sherpa-onnx\python-jimmy\offline-long-audio-pipeline\tests -q
  ```

  Expected: all existing and new tests pass.

- [ ] **Step 8: Commit the pipeline implementation.**

  ```powershell
  git add python-jimmy/offline-long-audio-pipeline python-jimmy/offline-long-audio-pipeline-asr-speaker-segmentation.py
  git commit -m "feat: assign short segments from clean clusters and local masks"
  ```

## Task 4: Add reproducible baseline-vs-optimized evaluation and report generation

**Files:**
- Create: `D:\code\opensource\ai\sherpa-onnx\python-jimmy\run-speaker-segmentation-evaluation.py`
- Create: `D:\code\opensource\ai\sherpa-onnx\python-jimmy\tests\test_speaker_segmentation_evaluation.py`
- Modify: `D:\code\opensource\ai\sherpa-onnx\python-jimmy\speaker_diarization_metrics.py` only if required to consume the existing test-set format
- Create runtime output under: `C:\Users\admin\Downloads\断句pipeline-python脚本测试-0920\`

- [ ] **Step 1: Add a manifest parser test.**

  Test the exact `说话人测试集.txt` format using a temporary manifest with audio path, reference label path, and optional ASR text path. The parser must preserve the input order and reject missing audio/label paths with a clear error.

- [ ] **Step 2: Implement evaluation orchestration.**

  The runner must:
  - read the supplied manifest without changing its paths;
  - run baseline via `offline-long-audio-pipeline-asr-speaker.py` and optimized via `offline-long-audio-pipeline-asr-speaker-segmentation.py`;
  - write commands, stdout/stderr, metadata, raw results, and per-file metrics under separate baseline/optimized directories;
  - calculate ASR WER, DER, and boundary accuracy using the repository's existing evaluation utilities and `speaker_diarization_metrics.py`;
  - write machine-readable JSON/CSV summaries and a Chinese Markdown report under the requested result directory.

- [ ] **Step 3: Run the orchestration parser/unit tests locally.**

  Run:

  ```powershell
  python -m pytest D:\code\opensource\ai\sherpa-onnx\python-jimmy\tests\test_speaker_segmentation_evaluation.py -q
  ```

- [ ] **Step 4: Commit the evaluation runner.**

  ```powershell
  git add python-jimmy/run-speaker-segmentation-evaluation.py python-jimmy/tests/test_speaker_segmentation_evaluation.py
  git commit -m "test: add baseline and optimized segmentation evaluation"
  ```

## Task 5: Verify locally, publish, build remotely, run the supplied test set, and generate the report

**Files:**
- No additional source files unless verification exposes a regression.
- Output: `C:\Users\admin\Downloads\断句pipeline-python脚本测试-0920\`

- [ ] **Step 1: Run a clean local verification pass.**

  Run the complete focused C++/Python tests, then inspect `git diff`, `git status`, and all generated metadata keys. Do not include the pre-existing unrelated untracked files in implementation commits.

- [ ] **Step 2: Create the final implementation commit and verify it.**

  Run `git diff --check`, the relevant unit/build checks, and inspect the staged diff before committing. Record the commit hash.

- [ ] **Step 3: Push the current branch normally to `jimmy`.**

  Use exactly:

  ```powershell
  git push jimmy HEAD:v1.13.2_transai_dev
  git rev-parse HEAD
  git rev-parse jimmy/v1.13.2_transai_dev
  ```

  Stop on any non-fast-forward or branch mismatch. Never force push.

- [ ] **Step 4: On the fixed remote repository, verify branch/worktree and pull fast-forward only.**

  Use the SSH/key/port and remote path defined by `sherpa-onnx-remote-build-test`. Confirm the remote branch is `v1.13.2_transai_dev`, worktree is clean, and pull with `git pull --ff-only`.

- [ ] **Step 5: Configure and build the Python-enabled shared library.**

  Run from `/speech_store/jimmy/k2_origin/sherpa-onnx/build/`:

  ```bash
  cmake -DSHERPA_ONNX_ENABLE_PYTHON=ON -DBUILD_SHARED_LIBS=ON -DSHERPA_ONNX_ENABLE_CHECK=OFF -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF -DSHERPA_ONNX_ENABLE_C_API=OFF -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF ..
  make -j"$(nproc)"
  ```

- [ ] **Step 6: Run the supplied test set remotely with the exact declared command.**

  Before execution, resolve the actual command from the supplied manifest/config and record it verbatim in the report. Do not invent model paths or rewrite arguments. If the supplied manifest references a Windows/network path unavailable on Linux, stop the remote run and report the exact missing path plus the commands already completed.

- [ ] **Step 7: Copy raw results/logs back and calculate metrics.**

  Preserve originals. Generate WER, DER, and boundary metrics in separate baseline and optimized directories, then generate the comparison Markdown report and summary JSON/CSV.

- [ ] **Step 8: Run final verification before claiming completion.**

  Verify exit status for every build/test/metric command, inspect report files, confirm both baseline and optimized results exist, and report any unavailable metric explicitly instead of substituting another metric.
