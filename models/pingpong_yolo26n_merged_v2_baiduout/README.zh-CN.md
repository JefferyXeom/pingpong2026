# pingpong_yolo26n_merged_v2_baiduout

来自`native_merged_v2_baiduout_yolo26n_1280`，merged_v2完整28401张train/36093球，原图1280×1024，batch8。训练32轮，球心F1选中第12轮；PT/ORT/TRT矩阵已完成。权重引用原best.pt，SHA见[模型卡](model_card.json)。

[本轮最终报告](../../docs/experiments/native_merged_v2_baiduout_20261113_results.zh-CN.md)含全部配置、val/test P/R/F1、定位误差、转换位移及测速。没有INT8结果；目标2060 Super未验证。
