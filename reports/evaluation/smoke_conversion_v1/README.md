# 小规模权重的转换流程验证

批次：早期流程批次；不是正式模型精度评估。

范围：使用smoke第1轮best.pt；方形960×960，抽样8张，比较传统NMS与免NMS导出；原始转换报告在artifacts的两个smoke目录。

本目录内的历史报告保留当时结论；最新状态以[总览](../../../docs/training_and_evaluation_overview.zh-CN.md)为准。

| 文件 | 用途 |
| --- | --- |
| [conversion_comparison.json](conversion_comparison.json) | 两个smoke转换批次的预测空间一致性；不是正式模型P/R。 |
