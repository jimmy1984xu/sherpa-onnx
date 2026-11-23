# 720 财务长音频 SIGSEGV：`Convert()` BPE/@@ 合并缺陷

**分支**：`v1.13.2_transai_debug_paraformer_segv_720`  
**修复提交**：`296f5189`（诊断提交 `f9a7fcac`）  
**关联模型**：Seaco Paraformer CPU `0100010001`（720 `audio_models`）

---

## 1. 现象（用户侧）

| 项 | 说明 |
|----|------|
| 场景 | 720 设备、财务 AI 培训会长音频 `asr_1788339431818.pcm`，Pipeline（VAD + 离线 ASR）**1× 实时推流** |
| 失败点 | 约 **20 分钟**处（`offsetMs≈1208478`），进程 **SIGSEGV**，fault `0x434c4e47432b2b00` |
| 表象 | 日志有 `ASR_DECODE_BEGIN`，**无** `ASR_DECODE_END`；易被误判为「25s 单块过大」或 ORT 算子问题 |
| 对照 | 同文件其他时刻、同长度 **399328 样点** 的切段可成功；**同一长度、不同起点** 决定成败 → 与 **解码 token 内容** 强相关 |

## 2. 排查路径（本任务做了什么）

1. **不改** `kMaxSamplesPerCall`，用 `run_asr_model` 做多长度 / 多起点扫频 → 排除「任意 25s 必崩」。
2. 在 **测试分支** 加 `sherpa-onnx-diag` 日志 + 远端 Linux 编 Android `libsherpa-onnx-c-api.so`。
3. 720 上 **FAIL / PASS** 两段隔离 PCM（同 399328 样点）对比 logcat。

### 2.1 诊断日志结论（修复前 `f9a7fca`）

FAIL 段典型顺序：

```text
SherpaOnnxDecodeOfflineStream begin
ParaformerDecode: num_frames=415, Forward done, out0[1,75,8404]
ParaformerGreedyDecode: batch=1 tokens=75 ...
ParaformerDecode: decode done batch_results=1
（无 DecodeOfflineStream done）
→ SIGSEGV
```

PASS 段：tensor `out0[1,117,8404]`，随后 **有** `DecodeOfflineStream done` 与 `ASR_DECODE_END`。

**结论**：ORT Forward 与 Greedy 主循环 **已正常结束**；崩溃发生在 **Greedy 之后、C-API 返回之前** → 落在 **结果后处理**（`Convert` → ITN → HomophoneReplacer → `SetResult`）。

## 3. 根因（代码级）

### 3.1 出错函数

`offline-recognizer-paraformer-impl.h` 中的 **`Convert(OfflineParaformerDecoderResult, SymbolTable)`**  
（在线版 `online-recognizer-paraformer-impl.h` 同源逻辑。）

### 3.2 业务背景：Paraformer + BPE + `@@`

- Greedy 输出一串 **token id**；`SymbolTable[id]` 得到 **子词符号**（可能以 `@@` 结尾表示与下一子词**无空格拼接**）。
- `Convert()` 用启发式判断「是否以 `@@` 结尾」决定：剥掉 `@@` 并设置 **mergeable**，用于中英混排空格规则。

### 3.3 缺陷逻辑（修复前）

伪代码：

```cpp
if (sym 不以 @@ 规则结尾) {
  按 ASCII/中文规则拼 text;
} else {
  // 认为是 "xxx@@" 续写片
  sym = string(sym.data(), sym.size() - 2);  // 去掉末尾 "@@"
  拼 text;
}
```

对 **FAIL 段**，Greedy 打出 **token_id=7251**，词表符号为 **单个字符 `sym='@'`**（长度为 1）：

1. `sym.back() == '@'` 为真 → 误入 **`@@` 分支**（实际并非 `piece@@`）。
2. `sym.size() - 2` 在 `size==1` 时，**`size_t` 无符号下溢** → 长度接近 `SIZE_MAX`。
3. `std::string(sym.data(), 巨大长度)` → **堆破坏 / 非法访存** → SIGSEGV。  
   fault 地址常表现为「像字符串碎片」的数值（如 `0x434c4e47432b2b00`），是内存被破坏后的访问，不是业务指针。

### 3.4 为何不是其它模块

| 假设 | 证据 |
|------|------|
| `kMaxSamplesPerCall` 顶格 | 同长度对照段 PASS；26s 二分块 PASS |
| ORT CPU 算子 | Forward 已打印 `done` + 合法 shape |
| Greedy stride 越界 | 已打 stride；decode done 已打印 |
| agent-sdk buffer | 隔离 `run_asr_model` 仍崩（修复前） |

## 4. 修复（`296f5189`）

1. **token id**：`SymbolTable::Contains(id)`，非法 id 跳过并 `LOGE`。
2. **空符号**：`sym.empty()` 不再调用 `sym.back()`。
3. **`@@` 分支**：仅当 `sym.size() >= 3`（至少 `x@@`）才剥 `@@`；否则 **跳过并打日志**  
   `ParaformerConvert: skip malformed @@ piece token_id=... sym='...'`。
4. 离线 **post-decode** 分段 diag 日志；Greedy 输出 `token_ids` 摘要（前 32 个）。

### 4.1 修复后 720 验证

| 用例 | 结果 |
|------|------|
| 原 FAIL 隔离段 | **通过**；日志出现 `skip ... token_id=7251 sym='@'`，有完整 ASR 文本 |
| 原 PASS 隔离段 | **通过** |

跳过畸形 `@@` 可能使个别英文子词边界少一个字符，**优于进程崩溃**；后续可改为把单 `@` 当字面量拼接（可选增强）。

## 5. 发布与合入

- **测试分支**：`v1.13.2_transai_debug_paraformer_segv_720`（含 `f9a7fcac` + `296f5189`）。
- **目标主分支**：`v1.13.2_transai`（与 720 现网 TransAI so 对齐）。
- **合入条件**：全长财务长音频 1× 跑通 + 结果指标归档；再 **ff-merge / merge** 测试分支，远端 `build-android-transai-arm64-v8a`（或等价）出 **正式 c-api so**，更新 agent-sdk `jniLibs`。

## 6. 相关文件

| 文件 | 变更类型 |
|------|----------|
| `sherpa-onnx/csrc/offline-recognizer-paraformer-impl.h` | `Convert` 加固 + 可选 diag |
| `sherpa-onnx/csrc/online-recognizer-paraformer-impl.h` | 同上 |
| `sherpa-onnx/csrc/offline-paraformer-greedy-search-decoder.cc` | diag |
| `sherpa-onnx/csrc/macros.h`、`c-api/c-api.cc` | diag（`f9a7fcac`） |

---

*本文档随测试分支维护；合入主分支时请一并 cherry-pick 或保留 `docs/transai/` 副本。*