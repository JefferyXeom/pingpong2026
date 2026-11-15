# 首轮原始数据整理 pingpong_v1

批次：数据版本v1。

范围：桌面yolov5data；8,720张、10,270球框，实际用于smoke和baseline。

本目录内的历史报告保留当时结论；最新状态以[总览](../../../docs/training_and_evaluation_overview.zh-CN.md)为准。

| 文件 | 用途 |
| --- | --- |
| [archive_inventory.json](archive_inventory.json) | 历史ZIP清单与缺失配对复查，未重复解压图片。 |
| [dataset_detail.json](dataset_detail.json) | 源数据详细目录、标签类别与原有Python环境探测。 |
| [dataset_inventory.json](dataset_inventory.json) | 源目录文件数量、扩展名、容量和初始盘点。 |
| [dataset_preparation.json](dataset_preparation.json) | 独立副本准备记录、筛选决策和逐项来源。 |
| [dataset_preview.jpg](dataset_preview.jpg) | 首轮数据集抽样画面预览。 |
| [dataset_verification.json](dataset_verification.json) | 整理后图片、标签、清单和集合隔离校验。 |
| [pairing_review.json](pairing_review.json) | 原图与标注配对、缺失标注/图片复查。 |
