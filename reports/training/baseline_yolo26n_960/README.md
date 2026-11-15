# 正式基线训练

批次：运行ID baseline_yolo26n_960；2026-11-08完成。

范围：YOLO26n，6,618张train、1,272张val；从24轮恢复，28轮早停。result中的最终验证对应best.pt第3轮，不能当作last.pt第28轮指标。

本目录内的历史报告保留当时结论；最新状态以[总览](../../../docs/training_and_evaluation_overview.zh-CN.md)为准。

| 文件 | 用途 |
| --- | --- |
| [baseline_interruption_review.json](baseline_interruption_review.json) | 恢复前停在24轮的状态快照；后来用户确认手动中断，已恢复完成。 |
| [baseline_launch_history.jsonl](baseline_launch_history.jsonl) | 初次启动与断点恢复的追加历史。 |
| [baseline_learning_curves.png](baseline_learning_curves.png) | 正式基线损失与框指标曲线。 |
| [baseline_process.json](baseline_process.json) | 最后一次启动的PID、命令及日志位置；历史记录不代表进程仍运行。 |
| [baseline_training_analysis.json](baseline_training_analysis.json) | 28轮训练过程、数据量、耗时、最佳/末轮指标分析。 |
| [baseline_yolo26n_960_progress.json](baseline_yolo26n_960_progress.json) | 训练最终状态及最后一次回调；final_best_validation对应best。 |
| [baseline_yolo26n_960_result.json](baseline_yolo26n_960_result.json) | 训练正常结束报告；返回的是best.pt第3轮的最终框验证。 |
