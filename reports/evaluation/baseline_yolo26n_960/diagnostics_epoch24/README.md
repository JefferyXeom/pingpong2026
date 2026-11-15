# 第24轮中断阶段诊断

批次：2026-11-08；恢复训练之前。

范围：使用保存的best第3轮、last第24轮快照；训练子集384张、验证子集256张；不使用test。这里的last.pt不是最终第28轮。

本目录内的历史报告保留当时结论；最新状态以[总览](../../../../docs/training_and_evaluation_overview.zh-CN.md)为准。

| 文件 | 用途 |
| --- | --- |
| [baseline_center_diagnostics.json](baseline_center_diagnostics.json) | 同一第3/24轮诊断子集的球心和框尺度分析。 |
| [baseline_diagnostics/best.pt](baseline_diagnostics/best.pt) | 诊断用第3轮best快照，保留当时checkpoint状态；不是当前球心候选。 |
| [baseline_diagnostics/last.pt](baseline_diagnostics/last.pt) | 诊断用第24轮last快照；不是训练结束的第28轮last。 |
| [baseline_diagnostics/train_subset.txt](baseline_diagnostics/train_subset.txt) | 诊断训练子集384张图片路径（每个训练批次128张）。 |
| [baseline_diagnostics/train_subset.yaml](baseline_diagnostics/train_subset.yaml) | 上述训练诊断子集的Ultralytics数据配置。 |
| [baseline_diagnostics/val_subset.txt](baseline_diagnostics/val_subset.txt) | 诊断验证子集256张图片路径。 |
| [baseline_diagnostics/val_subset.yaml](baseline_diagnostics/val_subset.yaml) | 上述验证诊断子集的Ultralytics数据配置。 |
| [baseline_diagnostics.json](baseline_diagnostics.json) | 第3/24轮快照在384张train与256张val上的框指标及检测头对比。 |
| [baseline_prediction_crops.jpg](baseline_prediction_crops.jpg) | 第24轮诊断阶段的原始标签/预测局部对照。 |
| [train_val_domain_preview.jpg](train_val_domain_preview.jpg) | 训练与验证录制批次抽样对照。 |
| [train_val_domain_review.json](train_val_domain_review.json) | 各录制批次的框尺度、球心位置和32张抽样亮度；未使用test。 |
