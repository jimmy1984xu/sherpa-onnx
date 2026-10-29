# 带真实分数的 Mali GPU 插件

v2/v3 runner 和 TDT kernel 派生自 ORIGIN.json 所列原包，代码来源 SHA 见 PROVENANCE.json。scored-sdk.inc 为两版本共用的带分数 ABI；只导出 create、recognize_ex、destroy。

core/ 只包含实际编译使用的 GPU dispatch、command buffer、profiling 及 OpenCL Headers。模型权重、前端和运行 kernel 位于交付的 app/models/。在 sherpa-onnx 的 Android arm64-v8a 构建中显式设置 SHERPA_ONNX_ENABLE_MALI_G720=ON，即从本目录源码构建并安装 v2/v3 插件；build-android-arm64-v8a.sh 支持透传同名环境变量。所需原始 SDK 或外部项目路径均已消除。

mali_loader.h 为设备 OpenCL 加载器；驱动不随包复制。
