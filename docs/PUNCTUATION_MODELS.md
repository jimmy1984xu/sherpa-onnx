# 标点模型（Punctuation Models）总览

本文梳理 `sherpa-onnx` 仓库内与标点恢复相关的所有主要代码与示例，并补充 Android 平台的集成建议，便于快速定位实现与开展二次开发。

## 模型类型与能力

| 模式 | 模型文件 | 典型路径 | 说明 |
| --- | --- | --- | --- |
| 离线标点（Offline） | `ct_transformer` ONNX | `sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12/model.onnx` | 双语（中英）Transformer，批量文本标点恢复，接口 `OfflinePunctuation::AddPunctuation`。 |
| 在线标点（Online） | `cnn_bilstm` ONNX + `bpe.vocab` | `sherpa-onnx-online-punct-en-2024-08-06/model.onnx`, `bpe.vocab` | 轻量 CNN + BiLSTM，支持流式文本/词流，接口 `OnlinePunctuation::AddPunctuationWithCase`。 |

模型发布页面：<https://github.com/k2-fsa/sherpa-onnx/releases/tag/punctuation-models>。离线模型主要适配多语场景，在线模型强调低延迟与自动大小写。

## 模型下载

### 下载方式

所有标点模型均从 GitHub Releases 页面下载：<https://github.com/k2-fsa/sherpa-onnx/releases/tag/punctuation-models>

### 离线标点模型（Offline Punctuation）

**下载命令：**
```bash
# 中英双语模型（推荐）
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2

# 解压
tar xvf sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2

# 清理压缩包（可选）
rm sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2
```

**模型文件结构：**
```
sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12/
└── model.onnx          # ONNX 模型文件（必需）
```

**使用路径：**
- 解压后，模型文件路径为：`./sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12/model.onnx`
- 在代码中指定该路径即可使用

### 在线标点模型（Online Punctuation）

**下载命令：**
```bash
# 英文在线模型
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-online-punct-en-2024-08-06.tar.bz2

# 解压
tar xvf sherpa-onnx-online-punct-en-2024-08-06.tar.bz2

# 清理压缩包（可选）
rm sherpa-onnx-online-punct-en-2024-08-06.tar.bz2
```

**模型文件结构：**
```
sherpa-onnx-online-punct-en-2024-08-06/
├── model.onnx          # ONNX 模型文件（必需）
└── bpe.vocab           # BPE 词表文件（必需）
```

**使用路径：**
- 模型文件：`./sherpa-onnx-online-punct-en-2024-08-06/model.onnx`
- BPE 词表：`./sherpa-onnx-online-punct-en-2024-08-06/bpe.vocab`
- 两个文件都需要在配置中指定

### 其他下载方式

**使用 curl：**
```bash
# 离线模型
curl -L -O https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2

# 在线模型
curl -L -O https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-online-punct-en-2024-08-06.tar.bz2
```

**手动下载：**
1. 访问 <https://github.com/k2-fsa/sherpa-onnx/releases/tag/punctuation-models>
2. 找到对应的模型压缩包（`.tar.bz2` 格式）
3. 点击下载并解压到本地目录

### 注意事项

- 所有模型文件均为 `.tar.bz2` 格式，需要使用 `tar` 命令解压
- Windows 系统可使用 7-Zip 或 WinRAR 解压
- 模型文件较大（通常几十到几百 MB），请确保网络连接稳定
- 解压后请检查模型文件是否存在且可读

## 代码分布速查

| 层级 | 关键文件/目录 | 作用 |
| --- | --- | --- |
| 核心推理 | `sherpa-onnx/csrc/offline-punctuation*.{h,cc}`<br>`sherpa-onnx/csrc/offline-punctuation-ct-transformer-impl.h`<br>`sherpa-onnx/csrc/offline-ct-transformer-model*.{h,cc}` | Offline 标点配置、校验与 Transformer 模型封装。 |
| 核心推理 | `sherpa-onnx/csrc/online-punctuation*.{h,cc}`<br>`sherpa-onnx/csrc/online-punctuation-cnn-bilstm-impl.h`<br>`sherpa-onnx/csrc/online-cnn-bilstm-model*.{h,cc}` | Online 标点配置、流式实现与 CNN-BiLSTM 子图。 |
| C API | `sherpa-onnx/c-api/c-api.h`（`SherpaOnnxOfflinePunctuationConfig` 等）<br>`c-api-examples/add-punctuation-c-api.c` | 标点配置与推理的 C 语言封装，便于与 C/C++ 工程对接。 |
| C++ 示例 | `cxx-api-examples/punctuation-cxx-api.cc` | 展示 `OfflinePunctuation` 的 C++ 使用方式。 |
| Python 绑定 | `sherpa-onnx/python/csrc/offline-punctuation.{h,cc}`<br>`sherpa-onnx/python/csrc/online-punctuation.{h,cc}` | Pybind11 导出离线/在线类。 |
| Python 示例 | `python-api-examples/add-punctuation.py`（离线）<br>`python-api-examples/add-punctuation-online.py`（在线） | 常用脚本示例，可直接运行验证模型。 |
| Go | `scripts/go/_internal/add-punctuation`<br>`go-api-examples/add-punctuation` | Go 语言 CLI 示例。 |
| Node.js | `nodejs-addon-examples/test_offline_punctuation.js`<br>`test_online_punctuation.js` | NAPI 扩展示例。 |
| Flutter/Dart | `flutter/sherpa_onnx/lib/src/offline_punctuation.dart`<br>`flutter/sherpa_onnx/lib/src/online_punctuation.dart`<br>`dart-api-examples/add-punctuations` | Flutter 插件与 Dart 脚本。 |
| Swift/iOS | `swift-api-examples/add-punctuation-online.swift` 等 | iOS 侧调用样例。 |
| Java/Kotlin/Android | `sherpa-onnx/jni/offline-punctuation.cc`、`jni/online-punctuation.cc`<br>`sherpa-onnx/kotlin-api/OfflinePunctuation.kt`、`OnlinePunctuation.kt`<br>`java-api-examples/OfflineAddPunctuation.java`、`OnlineAddPunctuation.java`<br>`kotlin-api-examples/test_online_punctuation.kt` | Android NDK 层桥接、Kotlin 封装与 JVM 侧示例。 |
| 鸿蒙 | `harmony-os/SherpaOnnxHar/.../punctuation.cc` | HarmonyOS C++ 侧封装。 |

以上目录构成了标点模型的全栈实现：C++ 核心 → 多语言绑定 → 示例/工具。

## 关键调用流程

1. **配置解析**：`OfflinePunctuationConfig` / `OnlinePunctuationConfig` 负责校验必需字段（模型路径、BPE 词表等），并支持线程数、Provider、调试信息等可选项。
2. **实现分发**：`OfflinePunctuationImpl::Create` 与 `OnlinePunctuationImpl::Create` 根据配置选择具体实现（如 `OnlinePunctuationCNNBiLSTMImpl`）。Android 构建时会走 AAssetManager overload，直接从 APK 资源读取模型。
3. **推理接口**：`OfflinePunctuation::AddPunctuation` 针对整段文本；`OnlinePunctuation::AddPunctuationWithCase` 支持在线场景并同步补全大小写。
4. **多语言导出**：C API → Python/Java/Kotlin/Node/Go 等语言层共享同一底层实现，保证行为一致。

## 示例入口

- **Python**：`python-api-examples/add-punctuation.py`、`add-punctuation-online.py`，适合快速验证模型输出。
- **C/C++**：`cxx-api-examples/punctuation-cxx-api.cc`、`c-api-examples/add-punctuation-c-api.c`。
- **JVM**：`java-api-examples/OfflineAddPunctuation.java`、`OnlineAddPunctuation.java`，外加 Kotlin 版 `kotlin-api-examples/test_online_punctuation.kt`。
- **其他语言**：Go (`scripts/go/_internal/add-punctuation`)、Node (`nodejs-addon-examples`)、Flutter/Dart (`flutter/sherpa_onnx/lib/src/*punctuation.dart`)、Swift (`swift-api-examples`)，可根据项目语言选择参考。

## Android 集成指南

1. **准备依赖**  
   在顶层 `settings.gradle(.kts)` 中确保引入 JitPack：
   ```kotlin
   dependencyResolutionManagement {
       repositories {
           google()
           mavenCentral()
           maven("https://jitpack.io")
       }
   }
   ```
   在模块 `build.gradle(.kts)` 中添加 Sherpa-ONNX AAR（版本号与仓库 `android/*/app/build.gradle` 保持一致，示例 v1.12.14）：
   ```kotlin
   dependencies {
       implementation("com.github.k2-fsa:sherpa-onnx:v1.12.14")
   }
   ```

2. **放置模型**  
   - 离线：将解压后的 `model.onnx`（如 `sherpa-onnx-punct-ct-transformer-zh-en-.../model.onnx`）复制到 `app/src/main/assets/`。  
   - 在线：除 `model.onnx` 外，还需 `bpe.vocab`。  
   Android 端通过 `AssetManager` 读取，因此无需额外路径权限。

3. **构造配置并创建实例**  
   Kotlin 推荐写法（离线）：
   ```kotlin
   val offlineConfig = OfflinePunctuationConfig(
       model = OfflinePunctuationModelConfig(
           ctTransformer = "sherpa-onnx-punct-ct-transformer-.../model.onnx",
           numThreads = 4,
           provider = "cpu",
           debug = false,
       ),
   )
   val punct = OfflinePunctuation(assetManager, offlineConfig)
   val result = punct.addPunctuation(rawText)
   ```
   在线模式类似，只需改为 `OnlinePunctuationConfig` 并设置 `cnnBilstm` / `bpeVocab`。示例可参考 `kotlin-api-examples/test_online_punctuation.kt`。

4. **JNI 生命周期**  
   `sherpa-onnx/kotlin-api/*.kt` 内部通过 `sherpa-onnx-jni` 加载本地库；对象释放调用 `release()`（或 `close` 包装）即可触发 `delete(ptr)`，避免内存泄漏。

5. **线程与调度**  
   - `numThreads` 控制 ONNX Runtime 推理线程，UI 线程只负责调用 Kotlin 封装，推理应放在 `CoroutineDispatcher.IO` 或业务线程池。  
   - 若要切换 GPU/NNAPI，可调整 `provider` 字段并确保对应 EP 已在 AAR 中启用。

6. **调试与日志**  
   配置中的 `debug=true` 时，会通过 Android logcat 打印 ONNX Runtime 日志；JNI 层 (`sherpa-onnx/jni/*punctuation*.cc`) 也会在配置校验失败时输出详细错误。

## 进一步工作

- 若需自定义模型，可复用 `offline-ct-transformer-model` 或 `online-cnn-bilstm-model` 的解析逻辑，只需保证 ONNX I/O 与当前实现保持一致。
- Flutter、Node、Go 等其它封装默认依赖同一底层配置结构，迁移 Android 业务时可先在 Python/CLI 验证输出，以确保模型权重正确。

以上内容可作为标点模型开发、排查与文档化的统一入口。若需更新，请同步维护对应示例，便于不同语言的用户快速对齐配置。

