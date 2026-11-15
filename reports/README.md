# 报告分类与批次导航

最新已完成：[Baidu新增数据原图n/s最终报告](../docs/experiments/native_merged_v2_baiduout_20261113_results.zh-CN.md)，本轮24项；[含历史72组合](experiments/native_merged_v2_baiduout_20261113/all_models_combinations.csv)。旧三组实验保留原批次。

先看[训练与评估总览](../docs/training_and_evaluation_overview.zh-CN.md)，再按下表打开对应批次。`reports`根目录仅放导航、实验登记和整理校验；报告实体按用途及实验ID保存。

| 目录 | 用途与批次 | 模型、数据及范围 |
| --- | --- | --- |
| [environment/windows_rtx3090](environment/windows_rtx3090/README.md) | 训练机环境；初始环境快照；不以文件移动时间推断安装时间。 | Windows / RTX3090，供所有现有实验参考；不是Ubuntu/2060 Super环境。 |
| [datasets/pingpong_v1](datasets/pingpong_v1/README.md) | 首轮原始数据整理 pingpong_v1；数据版本v1 | 桌面yolov5data；8,720张、10,270球框，实际用于smoke和baseline。 |
| [datasets/pingpong_additional_v1](datasets/pingpong_additional_v1/README.md) | 新增历史数据备份；2026-11-08整理记录 | D:/yolov5-master/VOCData及history_VOCOdata；10,786张、14,042球框；未参与旧960基线，新原图合并实验通过独立列表使用。 |
| [training/smoke_yolo26n_960](training/smoke_yolo26n_960/README.md) | 小规模训练流程验证；运行ID smoke_yolo26n_960；独立1轮 | YOLO26n，192张train、64张val，batch8；只验证训练链路。 |
| [training/baseline_yolo26n_960](training/baseline_yolo26n_960/README.md) | 正式基线训练；运行ID baseline_yolo26n_960；2026-11-08完成 | YOLO26n，6,618张train、1,272张val；从24轮恢复，28轮早停。result中的最终验证对应best.pt第3轮，不能当作last.pt第28轮指标。 |
| [evaluation/baseline_yolo26n_960/diagnostics_epoch24](evaluation/baseline_yolo26n_960/diagnostics_epoch24/README.md) | 第24轮中断阶段诊断；2026-11-08；恢复训练之前 | 使用保存的best第3轮、last第24轮快照；训练子集384张、验证子集256张；不使用test。这里的last.pt不是最终第28轮。 |
| [evaluation/baseline_yolo26n_960/center_selection_v1](evaluation/baseline_yolo26n_960/center_selection_v1/README.md) | 训练结束后的完整球心评估与选模；2026-11-08；完成28轮之后 | 完整val1,272张对比best第3轮和last第28轮；选定last后，完整test830张只评估last。 |
| [evaluation/smoke_conversion_v1](evaluation/smoke_conversion_v1/README.md) | 小规模权重的转换流程验证；早期流程批次；不是正式模型精度评估 | 使用smoke第1轮best.pt；方形960×960，抽样8张，比较传统NMS与免NMS导出；原始转换报告在artifacts的两个smoke目录。 |
| [evaluation/center_conversion_matrix_v1](evaluation/center_conversion_matrix_v1/README.md) | 正式模型的尺寸与转换精度/速度评估；2026-11-09完成；12项 | 使用pingpong_yolo26n_center_v1，即baseline第28轮last.pt；全量val1,272张＋test830张；960×768与原图1280×1024，PyTorch/ONNX/TensorRT FP32/FP16。 |
| [datasets/pingpong_baidu_out_v1](datasets/pingpong_baidu_out_v1/README.md) | Baidu新增数据审计；2026-11-13；源目录只读 | E:/BaiduNetdiskDownload/pingpongdata----out；字节/解码像素去重及逐图标注核验。 |
| [datasets/native_merged_v2_baiduout_20261113](datasets/native_merged_v2_baiduout_20261113/README.md) | 合并v2数据列表；版本pingpong_merged_v2 | v1原train＋additional_v1＋Baidu有效新图；原val/test固定。 |
| [experiments/native_merged_v2_baiduout_20261113](experiments/native_merged_v2_baiduout_20261113/README.md) | 合并v2原图训练队列；2026-11-13独立批次 | 按n、s顺序训练/评估；不覆盖旧原图三组实验。 |
| [training/native_merged_v2_baiduout_yolo26n_1280](training/native_merged_v2_baiduout_yolo26n_1280/README.md) | YOLO26n合并v2原图训练；native_merged_v2_baiduout_yolo26n_1280 | 数据pingpong_merged_v2，原图1280×1024、batch8、最多100轮、球心F1早停20轮。 |
| [evaluation/native_merged_v2_baiduout_yolo26n_1280_conversion_v1](evaluation/native_merged_v2_baiduout_yolo26n_1280_conversion_v1/README.md) | YOLO26n合并v2转换评估；native_merged_v2_baiduout_yolo26n_1280_conversion_v1 | 对应球心best，全val1272/test830，两尺寸12项；P/R/F1和定位误差为重点。 |
| [training/native_merged_v2_baiduout_yolo26s_1280](training/native_merged_v2_baiduout_yolo26s_1280/README.md) | YOLO26s合并v2原图训练；native_merged_v2_baiduout_yolo26s_1280 | 数据pingpong_merged_v2，原图1280×1024、batch8、最多100轮、球心F1早停20轮。 |
| [evaluation/native_merged_v2_baiduout_yolo26s_1280_conversion_v1](evaluation/native_merged_v2_baiduout_yolo26s_1280_conversion_v1/README.md) | YOLO26s合并v2转换评估；native_merged_v2_baiduout_yolo26s_1280_conversion_v1 | 对应球心best，全val1272/test830，两尺寸12项；P/R/F1和定位误差为重点。 |
| [datasets/native_resolution_suite_v1](datasets/native_resolution_suite_v1/README.md) | 原图实验派生列表；2026-11-09；不复制图片 | v1 train6618张，合并train17404张；val1272/test830固定；合并含2997背景变体。 |
| [experiments/native_resolution_suite_v1](experiments/native_resolution_suite_v1/README.md) | 原图实验队列与组合汇总；实时状态见progress.json | 顺序训练n/v1、n/合并、s/合并；1280×1024，球心F1选模，原图推理优先。 |
| [training/native_v1_yolo26n_1280](training/native_v1_yolo26n_1280/README.md) | YOLO26n原图训练；已完成；运行ID native_v1_yolo26n_1280 | 数据 native_v1；1280×1024、batch8；53轮早停，评估使用第33轮球心best。 |
| [evaluation/native_v1_yolo26n_1280_conversion_v1](evaluation/native_v1_yolo26n_1280_conversion_v1/README.md) | YOLO26n原图模型转换评估；native_v1_yolo26n_1280_conversion_v1 | 此训练球心best；全val1272/test830；两尺寸、6后端/batch组合每档。 |
| [training/native_v1_additional_yolo26n_1280](training/native_v1_additional_yolo26n_1280/README.md) | YOLO26n原图训练；已完成；运行ID native_v1_additional_yolo26n_1280 | 数据 native_v1_additional；1280×1024、batch8；44轮早停，评估使用第24轮球心best。 |
| [evaluation/native_v1_additional_yolo26n_1280_conversion_v1](evaluation/native_v1_additional_yolo26n_1280_conversion_v1/README.md) | YOLO26n原图模型转换评估；native_v1_additional_yolo26n_1280_conversion_v1 | 此训练球心best；全val1272/test830；两尺寸、6后端/batch组合每档。 |
| [training/native_v1_additional_yolo26s_1280](training/native_v1_additional_yolo26s_1280/README.md) | YOLO26s原图训练；已完成；运行ID native_v1_additional_yolo26s_1280 | 数据 native_v1_additional；1280×1024、batch8；50轮早停，评估使用第30轮球心best。 |
| [evaluation/native_v1_additional_yolo26s_1280_conversion_v1](evaluation/native_v1_additional_yolo26s_1280_conversion_v1/README.md) | YOLO26s原图模型转换评估；native_v1_additional_yolo26s_1280_conversion_v1 | 此训练球心best；全val1272/test830；两尺寸、6后端/batch组合每档。 |

`runs/`保存Ultralytics原始args.yaml、results.csv、训练图和权重；`artifacts/`保存ONNX、TensorRT、预测缓存与构建参数；`logs/`保存执行日志。这里的报告通过路径引用这些产物，不重复复制。

查找原则：先确定实验ID，再确定权重来源和数据范围。`baseline_diagnostics/last.pt`是第24轮快照；`runs/detect/baseline_yolo26n_960/weights/last.pt`及当前模型包是第28轮。

新增训练使用`reports/training/<run_name>/`；新增评估使用`reports/evaluation/<experiment_id>/`。`evaluate_centers.py`提供`--report-dir`，新实验应显式选择新目录。每个新批次应记录源权重/SHA256、数据版本、split、图片数、输入尺寸、阈值、后端/精度、日期与限制，登记到实验总览。

[机器可读文件目录](catalog.json) · [实验登记](experiment_registry.json) · [本次路径迁移记录](report_layout_migration.json) · [整理校验](report_layout_verification.json)
