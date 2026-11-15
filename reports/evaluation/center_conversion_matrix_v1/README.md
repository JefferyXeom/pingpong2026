# 正式模型的尺寸与转换精度/速度评估

批次：2026-11-09完成；12项。

范围：使用pingpong_yolo26n_center_v1，即baseline第28轮last.pt；全量val1,272张＋test830张；960×768与原图1280×1024，PyTorch/ONNX/TensorRT FP32/FP16。

本目录内的历史报告保留当时结论；最新状态以[总览](../../../docs/training_and_evaluation_overview.zh-CN.md)为准。

| 文件 | 用途 |
| --- | --- |
| [center_conversion_matrix_comparison.png](center_conversion_matrix_comparison.png) | batch2球心F1、P95和RTX3090 Graph吞吐可视化。 |
| [center_conversion_matrix_progress.json](center_conversion_matrix_progress.json) | 此转换矩阵最终完成状态；12项及最终复测均完成。 |
| [center_conversion_matrix_summary.csv](center_conversion_matrix_summary.csv) | 12项指标的可筛选表格，包含val/test球心及框AP。 |
| [center_conversion_matrix_summary.json](center_conversion_matrix_summary.json) | 转换矩阵的后端误差、同batch提速和原图变化摘要。 |
| [center_conversion_matrix_v1.json](center_conversion_matrix_v1.json) | 正式模型12项完整精度、转换差异、初测/复测速率主报告。 |
| [center_conversion_matrix_verification.json](center_conversion_matrix_verification.json) | 权重哈希、12项缓存、测速次数与原分辨率示例校验。 |
| [center_conversion_regressions.json](center_conversion_regressions.json) | FP16新增漏检逐图复查，含候选置信度和球心距离。 |
| [conversion_original_examples/01_val_native_fp16.jpg](conversion_original_examples/01_val_native_fp16.jpg) | 原图1280×1024标注示例，TensorRT FP16 batch2；源图片见同目录index.json。 |
| [conversion_original_examples/02_val_native_fp16.jpg](conversion_original_examples/02_val_native_fp16.jpg) | 原图1280×1024标注示例，TensorRT FP16 batch2；源图片见同目录index.json。 |
| [conversion_original_examples/03_val_native_fp16.jpg](conversion_original_examples/03_val_native_fp16.jpg) | 原图1280×1024标注示例，TensorRT FP16 batch2；源图片见同目录index.json。 |
| [conversion_original_examples/04_val_native_fp16.jpg](conversion_original_examples/04_val_native_fp16.jpg) | 原图1280×1024标注示例，TensorRT FP16 batch2；源图片见同目录index.json。 |
| [conversion_original_examples/05_test_native_fp16.jpg](conversion_original_examples/05_test_native_fp16.jpg) | 原图1280×1024标注示例，TensorRT FP16 batch2；源图片见同目录index.json。 |
| [conversion_original_examples/06_test_native_fp16.jpg](conversion_original_examples/06_test_native_fp16.jpg) | 原图1280×1024标注示例，TensorRT FP16 batch2；源图片见同目录index.json。 |
| [conversion_original_examples/07_test_native_fp16.jpg](conversion_original_examples/07_test_native_fp16.jpg) | 原图1280×1024标注示例，TensorRT FP16 batch2；源图片见同目录index.json。 |
| [conversion_original_examples/08_test_native_fp16.jpg](conversion_original_examples/08_test_native_fp16.jpg) | 原图1280×1024标注示例，TensorRT FP16 batch2；源图片见同目录index.json。 |
| [conversion_original_examples/index.json](conversion_original_examples/index.json) | 8张均匀抽样原分辨率标注图的源图片、哈希和推理配置。 |
| [conversion_original_image_crops.jpg](conversion_original_image_crops.jpg) | 固定均匀抽取8张的局部放大对照；不是评估输入裁剪。 |
