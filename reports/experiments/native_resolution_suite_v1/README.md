# 原图实验队列与组合汇总

批次：实时状态见progress.json。

范围：顺序训练n/v1、n/合并、s/合并；1280×1024，球心F1选模，原图推理优先。

本目录内的历史报告保留当时结论；最新状态以[总览](../../../docs/training_and_evaluation_overview.zh-CN.md)为准。

| 文件 | 用途 |
| --- | --- |
| [all_combinations.csv](all_combinations.csv) | 训练模型×数据×推理尺寸×后端×精度×batch的已有结果。 |
| [center_precision_recall.csv](center_precision_recall.csv) | 球心专表：48个推理组合×val/test共96行，P/R、TP/FP/FN、同batch转换变化，不含IoU指标。 |
| [center_precision_recall.json](center_precision_recall.json) | 球心P/R专表的完整精确数值、配置、源权重哈希及尚无INT8结果的说明。 |
| [completion_verification.json](completion_verification.json) | 本目录批次完成核验：组合数、权重/engine/ONNX哈希、缓存、保留集顺序、训练选模和最终测速次数。 |
| [final_analysis.json](final_analysis.json) | 本目录批次实际训练耗时、选中权重、原图FP32/FP16结果及完成核验。 |
| [native_results.png](native_results.png) | 本目录批次训练曲线及原图TensorRT FP16精度/速度对照，RTX3090。 |
| [native_results.svg](native_results.svg) | 训练和原图精度/速度对照的可导出矢量图。 |
| [process.json](process.json) | 队列进程ID与创建时间，用于防重复启动。 |
| [progress.json](progress.json) | 训练/评估/队列状态；实际批次张量及轮数用于判断进展。 |
| [queue.lock](queue.lock) | 操作系统持有的互斥锁；文件存在不等于队列仍运行。 |
| [summary.json](summary.json) | 旧基线及三组新实验全部可用组合汇总，保留未完成状态。 |
