# Mali-G720 Parakeet 后端

v2与v3各使用一套GPU运行库，结构和词表兼容的微调模型共用相应运行库。私有插件导出create、recognize_ex、recognize_beam、destroy；sherpa公共C API结构保持一致。

默认greedy使用模型自带GPU程序。modified_beam_search使用随运行库封装的GPU可执行程序，执行TDT矩阵运算、四路候选排序/剪枝、热词图跳转与文本生成；不逐步回读logits，不回退CPU。热词配置在推理前编码与上传。热词奖励只参与候选排序，输出分数为未加奖励的声学token log-prob。

源码与GPU程序身份见PROVENANCE.json及beam-programs.json。重建入口app/android/build.py使用交付app、sherpa-onnx及本包quantization/neon-load/export-ort.cc；重建时校验GPU源码和随包可执行程序的哈希。每个模型由一个独立 `.aot` 文件提供，不嵌入运行库。

公共C API不增加字段或方法。MaliG720配置使用 `transducer.encoder=<model.aot>`，decoder、joiner和tokens留空；model_type仍选择parakeet_v2_mali_g720或parakeet_v3_mali_g720。非Mali provider的配置和实现不变。

单文件使用TAIAOT01头、受限索引和64KiB对齐的原始资产，包含model.json、词表、权重、执行计划及五个GPU可执行程序。初始化直接pread小资产、映射Encoder/TDT权重并阻塞上传；没有解包、临时模型文件、运行时内容哈希或源码编译回退。索引范围/重复项、资产尺寸、架构和GPU/驱动检查保留。

相同流级热词及其不可变图使用有界单项缓存，减少重复BPE编码和图序列化；GPU仍执行全部模型与搜索计算。PCM边界转换使用与原lround相同舍入规则的ARM SIMD转换、饱和及有限范围检查。热词搜索在现有槽位保留声学轨迹，防止blank/短路径排除正常文本。V2普通候选须完成热词才可替代声学轨迹；V3结合最后发声帧限制长度奖励，保留后续语句并避免插入词获益。输出声学分数不含热词奖励。

Greedy以二进制大小的多步命令批次组合原有轮数预算；beam将相邻两个八轮批次合并，奇数个批次保留八轮尾项。V3从原预算启动，利用已经读回的最终包中的搜索轮数调整后续预算，并按需录制命令批次，减少低并行度尾段；不增加中间读回或CPU搜索。预算具有请求历史，验收固定顺序与预热方式。两种既有GPU归约布局的末位FP32声学概率可能有舍入差，输出格式、文本和任务质量独立核对。

当前结果与交付身份见发布根目录 `docs/ASR_HOTWORDS_K70_ACCEPTANCE_20261111.md`、`docs/V3_HOTWORD_SPEED_K70_20261112.md`。GPU程序来源清单中的编译设备临时路径只记录编译来源，加载与重建不读取这些路径。
