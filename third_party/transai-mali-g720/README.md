# Mali-G720 Parakeet 后端

v2与v3各使用一套GPU运行库，结构和词表兼容的微调模型共用相应运行库。私有插件导出create、recognize_ex、recognize_beam、destroy；sherpa公共C API结构保持一致。

默认greedy使用模型自带GPU程序。modified_beam_search使用随运行库封装的GPU可执行程序，执行TDT矩阵运算、四路候选排序/剪枝、热词图跳转与文本生成；不逐步回读logits，不回退CPU。热词配置在推理前编码与上传。热词奖励只参与候选排序，输出分数为未加奖励的声学token log-prob。

源码与GPU程序身份见PROVENANCE.json及beam-programs.json。重建入口app/android/build.py只依赖交付app与sherpa-onnx；重建时校验GPU源码和随包可执行程序的哈希。每个模型由一个独立 `.aot` 文件提供，不嵌入运行库。

公共C API不增加字段或方法。MaliG720配置使用 `transducer.encoder=<model.aot>`，decoder、joiner和tokens留空；model_type仍选择parakeet_v2_mali_g720或parakeet_v3_mali_g720。非Mali provider的配置和实现不变。

单文件使用TAIAOT01头、受限索引和64KiB对齐的原始资产，包含model.json、词表、权重、执行计划及五个GPU可执行程序。初始化直接pread小资产、映射Encoder/TDT权重并阻塞上传；没有解包、临时模型文件、运行时内容哈希或源码编译回退。索引范围/重复项、资产尺寸、架构和GPU/驱动检查保留。

相同流级热词及其不可变图使用有界单项缓存，减少重复BPE编码和图序列化；GPU仍执行全部模型与搜索计算。PCM边界转换使用与原lround相同舍入规则的ARM SIMD转换、饱和及有限范围检查。候选数、剪枝、GPU程序和输出分数定义均不变。

Greedy以二进制大小的多步命令批次组合原有轮数预算；beam将相邻两个八轮批次合并，奇数个批次保留八轮尾项。两者减少队列提交及未使用的完成事件，不使用同时重入同一记录的扩展。相同in-order队列仍按原顺序执行完全相同数量的GPU核，候选数及数值计算保持一致。
