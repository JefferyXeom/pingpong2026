# 训练结束后的完整球心评估与选模

批次：2026-11-08；完成28轮之后。

范围：完整val1,272张对比best第3轮和last第28轮；选定last后，完整test830张只评估last。

本目录内的历史报告保留当时结论；最新状态以[总览](../../../../docs/training_and_evaluation_overview.zh-CN.md)为准。

| 文件 | 用途 |
| --- | --- |
| [baseline_full_test_centers.json](baseline_full_test_centers.json) | 选模后仅第28轮last在完整test上的球心结果。 |
| [baseline_full_test_centers_last_frames.json](baseline_full_test_centers_last_frames.json) | 完整test、第28轮last逐帧计数与误差。 |
| [baseline_full_val_centers.json](baseline_full_val_centers.json) | 完整val上第3/28轮球心对比及按固定阈值选模结果。 |
| [baseline_full_val_centers_best_frames.json](baseline_full_val_centers_best_frames.json) | 完整val、第3轮best的逐帧计数与匹配误差。 |
| [baseline_full_val_centers_last_frames.json](baseline_full_val_centers_last_frames.json) | 完整val、第28轮last的逐帧计数与匹配误差。 |
| [trained_model_selection.json](trained_model_selection.json) | 候选模型来源、SHA256、选择依据和初始完整球心评估。 |
