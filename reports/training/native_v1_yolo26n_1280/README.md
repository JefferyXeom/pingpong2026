# YOLO26n原图训练

批次：已完成；运行ID native_v1_yolo26n_1280。

范围：数据 native_v1；1280×1024、batch8；53轮早停，评估使用第33轮球心best。

本目录内的历史报告保留当时结论；最新状态以[总览](../../../docs/training_and_evaluation_overview.zh-CN.md)为准。

| 文件 | 用途 |
| --- | --- |
| [configuration.json](configuration.json) | 本次训练参数、初始化权重哈希、数据范围及原图输入尺寸。 |
| [epoch_metrics.jsonl](epoch_metrics.jsonl) | 逐轮完整val球心、框指标和误差，包含用于早停的fitness。 |
| [preflight.json](preflight.json) | 正式优化前的实际训练张量、数据量及验证尺寸检查。 |
| [progress.json](progress.json) | 训练/评估/队列状态；实际批次张量及轮数用于判断进展。 |
| [result.json](result.json) | 训练完成记录、最佳球心epoch、权重路径与SHA256；仅完成后生成。 |
