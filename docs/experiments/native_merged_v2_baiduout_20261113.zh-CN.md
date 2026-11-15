# Baidu增量数据与原图训练：pingpong_merged_v2

数据本身的框选质量可查看[Baidu标注质量可视化](../../reports/datasets/pingpong_baidu_out_v1/label_quality_review.zh-CN.md)：96张样本、原图框/球心、无损局部放大及12页拼图；没有模型预测，未修改标注。

来源审计日期2026-11-13。本页由scripts/summarize_followup_suite.py依据实际产物更新，未完成结果不提前宣称完成。

增量来源为E:/BaiduNetdiskDownload/pingpongdata----out。按文件SHA256及解码RGB像素SHA256比对所有v1/additional图片；有效新图独立复制到datasets/pingpong_baidu_out_v1。已用图不重复加入，有标注冲突、缺标注、疑似保留集派生图均排除。源目录和此前训练/评估不修改。

## 数据身份与数量

| 项目 | 图片 | 球框 |
| --- | --- | --- |
| 原始发现目录 | 18,086 | 以逐图审计为准 |
| 本次有效唯一新增 | 10,997 | 13,912 |
| 合并v2 / train | 28,401 | 36,093 |
| 合并v2 / val | 1,272 | 1,272 |
| 合并v2 / test | 830 | 859 |

原有验证集20240527_3（1272图/1272球）与测试集20240527_2（830图/859球）保持不变；新图只加train。已知文件名父图候选均在train，但历史录制/背景派生关系和跨场地独立性不能仅由哈希证明；源图中的后缀变体不可当作全部独立采集。缺标签及空标注不自动当作无球负样本。

[源目录重复/新增统计](../../reports/datasets/pingpong_baidu_out_v1/audit_summary.json) · [逐图决策](../../reports/datasets/pingpong_baidu_out_v1/audit_decisions.jsonl) · [数据清单与哈希](../../reports/datasets/native_merged_v2_baiduout_20261113/data_preparation.json)

## 训练配置与模型命名

保持与此前原图实验相同的策略：YOLO26n后接YOLO26s顺序训练；对应官方预训练分别初始化，全量合并数据训练，并非恢复已结束B/C的优化器。这样可与旧B/C比较新增数据影响。实际输入宽1280×高1024，张量B×3×1024×1280，不缩放/补边/裁剪；batch8（末批按数据余数）、最多100轮，完整val conf=.25且原图5px球心F1选best，连续20轮无严格提升早停。训练AMP、验证FP32、关闭TF32。

MuSGD lr0=.01、lrf=.01、momentum=.937、weight_decay=.0005、warmup3、nbs64；seed20261109与此前相同；workers2、cache=False、deterministic=True；mosaic/几何scale/translate/旋转/mixup/copy_paste关闭；fliplr=.5、HSV .015/.7/.4。save_period=-1，仅best/last；预留至少5GiB磁盘。完整实际设置和权重SHA见各configuration.json及args.yaml。

| 训练ID | 架构 | 状态 | 完成epoch | 选中epoch | 转换评估 |
| --- | --- | --- | --- | --- | --- |
| native_merged_v2_baiduout_yolo26n_1280 | YOLO26n | training_complete | 32 | 12 | complete |
| native_merged_v2_baiduout_yolo26s_1280 | YOLO26s | training_complete | 42 | 22 | complete |

每组训练完成后依次进行两尺寸/12项PT、ORT CPU FP32、TRT FP32/FP16完整评估和GPU最终复测，然后下一模型。P/R/F1与定位中位数/P95/P99均为重点；新增漏检和转换位移同样保留。INT8未纳入本次任务。3090速度不代表2060 Super达标。

## 已有原图batch2评估

| 模型 | 后端 | 集合 | 球心P | 球心R | F1 | 中位 / P95 / P99 px |
| --- | --- | --- | --- | --- | --- | --- |
| native_merged_v2_baiduout_yolo26n_1280 | pytorch_fp32 | val | 98.59% | 99.21% | 98.90% | 1.225 / 2.695 / 3.351 |
| native_merged_v2_baiduout_yolo26n_1280 | pytorch_fp32 | test | 93.40% | 97.21% | 95.27% | 1.184 / 2.444 / 3.118 |
| native_merged_v2_baiduout_yolo26n_1280 | onnx_cpu_fp32 | val | 98.59% | 99.21% | 98.90% | 1.225 / 2.695 / 3.351 |
| native_merged_v2_baiduout_yolo26n_1280 | onnx_cpu_fp32 | test | 93.40% | 97.21% | 95.27% | 1.184 / 2.444 / 3.118 |
| native_merged_v2_baiduout_yolo26n_1280 | tensorrt_fp32 | val | 98.59% | 99.21% | 98.90% | 1.225 / 2.695 / 3.351 |
| native_merged_v2_baiduout_yolo26n_1280 | tensorrt_fp32 | test | 93.40% | 97.21% | 95.27% | 1.184 / 2.444 / 3.118 |
| native_merged_v2_baiduout_yolo26n_1280 | tensorrt_fp16 | val | 98.67% | 99.21% | 98.94% | 1.250 / 2.738 / 3.325 |
| native_merged_v2_baiduout_yolo26n_1280 | tensorrt_fp16 | test | 93.30% | 97.21% | 95.21% | 1.176 / 2.465 / 3.060 |
| native_merged_v2_baiduout_yolo26s_1280 | pytorch_fp32 | val | 99.14% | 99.61% | 99.37% | 1.201 / 2.683 / 3.314 |
| native_merged_v2_baiduout_yolo26s_1280 | pytorch_fp32 | test | 94.28% | 97.79% | 96.00% | 1.200 / 2.467 / 3.078 |
| native_merged_v2_baiduout_yolo26s_1280 | onnx_cpu_fp32 | val | 99.14% | 99.61% | 99.37% | 1.201 / 2.683 / 3.314 |
| native_merged_v2_baiduout_yolo26s_1280 | onnx_cpu_fp32 | test | 94.28% | 97.79% | 96.00% | 1.200 / 2.467 / 3.078 |
| native_merged_v2_baiduout_yolo26s_1280 | tensorrt_fp32 | val | 99.14% | 99.61% | 99.37% | 1.201 / 2.683 / 3.314 |
| native_merged_v2_baiduout_yolo26s_1280 | tensorrt_fp32 | test | 94.28% | 97.79% | 96.00% | 1.200 / 2.467 / 3.078 |
| native_merged_v2_baiduout_yolo26s_1280 | tensorrt_fp16 | val | 99.06% | 99.61% | 99.33% | 1.250 / 2.719 / 3.272 |
| native_merged_v2_baiduout_yolo26s_1280 | tensorrt_fp16 | test | 94.49% | 97.79% | 96.11% | 1.186 / 2.504 / 3.121 |

## 相对上一版合并模型（原图batch2、test）

| 新模型 | 后端 | 旧版P / R / F1 | 新版P / R / F1 | 旧版P95 / 新版P95 px |
| --- | --- | --- | --- | --- |
| native_merged_v2_baiduout_yolo26n_1280 | pytorch_fp32 | 94.87% / 96.86% / 95.85% | 93.40% / 97.21% / 95.27% | 2.456 / 2.444 |
| native_merged_v2_baiduout_yolo26n_1280 | onnx_cpu_fp32 | 94.87% / 96.86% / 95.85% | 93.40% / 97.21% / 95.27% | 2.456 / 2.444 |
| native_merged_v2_baiduout_yolo26n_1280 | tensorrt_fp32 | 94.87% / 96.86% / 95.85% | 93.40% / 97.21% / 95.27% | 2.456 / 2.444 |
| native_merged_v2_baiduout_yolo26n_1280 | tensorrt_fp16 | 94.87% / 96.86% / 95.85% | 93.30% / 97.21% / 95.21% | 2.485 / 2.465 |
| native_merged_v2_baiduout_yolo26s_1280 | pytorch_fp32 | 95.45% / 97.67% / 96.55% | 94.28% / 97.79% / 96.00% | 2.620 / 2.467 |
| native_merged_v2_baiduout_yolo26s_1280 | onnx_cpu_fp32 | 95.45% / 97.67% / 96.55% | 94.28% / 97.79% / 96.00% | 2.620 / 2.467 |
| native_merged_v2_baiduout_yolo26s_1280 | tensorrt_fp32 | 95.45% / 97.67% / 96.55% | 94.28% / 97.79% / 96.00% | 2.620 / 2.467 |
| native_merged_v2_baiduout_yolo26s_1280 | tensorrt_fp16 | 95.45% / 97.67% / 96.55% | 94.49% / 97.79% / 96.11% | 2.601 / 2.504 |

没有评估行即表示尚未执行，不用计划数值代替结果。置信度.25，5px一对一中心匹配算P/R/F1；误差分位数只含10px内匹配，必须同时看FN；框IoU不作为主判据。

## 查看状态

```powershell
& ./.venv/Scripts/python.exe scripts/check_native_suite_status.py --suite native_merged_v2_baiduout_20261113
& ./.venv/Scripts/python.exe scripts/summarize_followup_suite.py --suite native_merged_v2_baiduout_20261113
# 队列停止且无独立训练进程时才恢复：
& ./.venv/Scripts/python.exe scripts/run_native_suite.py --suite native_merged_v2_baiduout_20261113 --launch
```

[队列状态](../../reports/experiments/native_merged_v2_baiduout_20261113/progress.json) · [汇总JSON](../../reports/experiments/native_merged_v2_baiduout_20261113/summary.json)

各训练进度reports/training/<run_id>/progress.json，30秒刷新；完成后result.json记录权重路径与SHA。模型卡在完成训练后生成于models/pingpong_yolo26n_merged_v2_baiduout及对应s目录，引用原始best.pt节省空间。
