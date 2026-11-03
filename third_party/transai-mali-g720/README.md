# Mali-G720 Parakeet 后端

v2与v3各使用一套GPU运行库，结构和词表兼容的微调模型共用相应运行库。私有插件导出create、recognize_ex、recognize_beam、destroy；sherpa公共C API结构保持一致。

默认greedy使用模型自带GPU程序。modified_beam_search使用随运行库封装的GPU可执行程序，执行TDT矩阵运算、四路候选排序/剪枝、热词图跳转与文本生成；不逐步回读logits，不回退CPU。热词配置在推理前编码与上传。热词奖励只参与候选排序，输出分数为未加奖励的声学token log-prob。

源码与GPU程序身份见PROVENANCE.json及beam-programs.json。重建入口app/android/build.py只依赖交付app与sherpa-onnx；重建时校验GPU源码和随包可执行程序的哈希。两套模型权重保持独立资产，不嵌入运行库。
