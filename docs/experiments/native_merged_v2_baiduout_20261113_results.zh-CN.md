# Baidu新增数据：原图 n/s 训练与转换最终报告

查看数据标注本身：[Baidu标注质量可视化](../../reports/datasets/pingpong_baidu_out_v1/label_quality_review.zh-CN.md)，含96张新增/已用样本的标注框、球心和无损局部图；未修改数据或训练结果。

本轮全部完成于北京时间 **2026-11-14 06:47:47**（Asia/Shanghai，UTC+08:00）。两个模型均正常早停，训练、全量转换评估和最终GPU复测全部完成。8个队列阶段、24个推理组合、48行val/test球心结果；每组合评估2,102张图片。

**新增数据提高了测试召回率，但增加了额外预测，测试F1未全面超过旧版。** 不能将新模型直接视为全面升级。本轮s的测试定位P95较旧s改善，n/s仍需与旧B/C一起保留，在独立新录制与目标机上评估。阈值仍固定.25；未用test重选权重或调阈值。

[本轮48行球心表](../../reports/experiments/native_merged_v2_baiduout_20261113/center_results.csv) · [24组合完整表](../../reports/experiments/native_merged_v2_baiduout_20261113/all_combinations.csv) · [含历史共72组合](../../reports/experiments/native_merged_v2_baiduout_20261113/all_models_combinations.csv) · [核验记录](../../reports/experiments/native_merged_v2_baiduout_20261113/completion_verification.json) · [精确结果JSON](../../reports/experiments/native_merged_v2_baiduout_20261113/final_analysis.json)

## 数据版本、去重与保留集

源目录 `E:/BaiduNetdiskDownload/pingpongdata----out` 共有18,086张PNG及同数XML，约24.92GiB。经文件SHA256与解码RGB像素SHA256比对：7,089张已在additional_v1 train中使用，标注一致；10,997张有效新图含13,912个球框，新备份命名为 `pingpong_baidu_out_v1`，图片约15.20GiB。桌面、D盘历史目录和E盘源文件保持不动，新合并列表引用已有备份图片，未复制整份合并数据。

| 数据组成 / 集合 | 图片数 | 球框数 | 用途 |
| --- | --- | --- | --- |
| v1 train | 6,618 | 8,139 | 原始训练图 |
| additional_v1 | 10,786 | 14,042 | 含2,997张白背景变体 |
| baidu_out_v1新增 | 10,997 | 13,912 | 仅加入train |
| merged_v2 train | 28,401 | 36,093 | n/s均完整使用 |
| 固定val：20240527_3 | 1,272 | 1,272 | 选模与早停；左1122 / 右150 |
| 固定test：20240527_2 | 830 | 859 | 训练后报告；左629 / 右201 |

val/test列表与上一批逐字节相同，转换评估图片顺序及图片哈希记录也一致；同一录制左右目留在同集合。新增图仅加入train。精确内容去重不能证明跨场地、跨录制或近似图独立；新目录的后缀变体不能全部当独立采集，未知历史派生关系仍待核实。旧test已多次报告，下一次验收应补充独立新录制。

数据审计：[摘要](../../reports/datasets/pingpong_baidu_out_v1/audit_summary.json)、[逐图决策](../../reports/datasets/pingpong_baidu_out_v1/audit_decisions.jsonl)、[实际列表及SHA](../../reports/datasets/native_merged_v2_baiduout_20261113/data_preparation.json)。本轮没有已确认无球负样本，缺标签图片没有当负样本使用。

## 实际训练配置与停止条件

单类别 `0: ball`，YOLO26n与YOLO26s分别从官方预训练初始化，使用完整merged_v2重新训练；没有接着旧B/C的优化器续训。**原图宽1280×高1024，实际训练/验证张量B×3×1024×1280，无缩放、补边、裁剪**；imgsz=1280、rect=True、multi_scale=0。训练batch=8，每轮3551批，末批1张。progress最后张量[1,3,1024,1280]是末批；preflight的[2,3,1024,1280]是两图检查样本，都不能当正式batch。

| 模型 / 本轮简称 | 运行ID | 完成 / 上限epoch | 球心best epoch | 选中时val F1 | CSV训练时间 / 墙钟时间 |
| --- | --- | --- | --- | --- | --- |
| YOLO26n / D | native_merged_v2_baiduout_yolo26n_1280 | 32 / 100 | 12 | 98.94% | 5.042h / 5.047h |
| YOLO26s / E | native_merged_v2_baiduout_yolo26s_1280 | 42 / 100 | 22 | 99.37% | 6.869h / 6.876h |

CSV训练/逐轮验证共 **11.912小时**，不包含之后导出、TRT构建及测速。完整队列墙钟约13.174小时。两个模型在best之后连续20轮无严格球心F1提升而早停；最多100轮。使用完整val、conf=.25、原图5px球心F1选best，平分保留最新权重。评估使用best.pt，last.pt仅作续训/追溯。

MuSGD：lr0=.01、lrf=.01、momentum=.937、weight_decay=.0005、warmup3、nbs64；训练AMP、验证FP32，关闭TF32；seed20261109、deterministic=True、workers2、cache=False。关闭mosaic、scale、translate、旋转、shear、perspective、mixup、copy_paste与上下翻转；保留fliplr=.5、HSV=.015/.7/.4；max_det300、nms=False、save_period=-1、plots=False、device0。仍优化YOLO检测损失，未新增球心损失。

训练回调验证与独立转换矩阵分开保存：n选中时val F1为98.9416%，独立PT矩阵val为98.9020%（同一best权重、固定.25）；两条执行路径的边界输出有小差异，不能混用为转换退化。下文所有后端变化均以独立矩阵同尺寸/同batch PT结果为参考。s选模与矩阵F1本轮一致。

## 评估协议

每个模型12项：原图1280×1024和全图缩小960×768，各包含PT FP32 b1/b2、ONNX Runtime CPU FP32 b2、TRT FP32 b2、TRT FP16 b1/b2。没有裁切。conf=.25；在原图坐标以≤5px一对一贪心最近匹配计算P/R/F1，TP为成功匹配、FP为额外预测、FN为未匹配标注球。FP/FN可能包括定位偏差导致的匹配失败，并不等于纯无球误检数/模型完全没输出。

定位误差是预测球心到GT标注球心的欧氏距离；中位/P95/P99仅统计≤10px成功匹配，需同时看TP/FP/FN。不同模型匹配集合可不同，不能仅靠分位数证明逐球都更准。二维像素误差不等于三维精度。框IoU/mAP保存在原始矩阵，本报告不按其排名。

## 原图球心精度：所有batch2后端

### 验证集1272图 / 1272球

| 模型 | 后端 | P | R | F1 | TP / FP / FN | 中位 / P95 / P99 px |
| --- | --- | --- | --- | --- | --- | --- |
| YOLO26n | pytorch_fp32 | 98.59% | 99.21% | 98.90% | 1262 / 18 / 10 | 1.225 / 2.695 / 3.351 |
| YOLO26n | onnx_cpu_fp32 | 98.59% | 99.21% | 98.90% | 1262 / 18 / 10 | 1.225 / 2.695 / 3.351 |
| YOLO26n | tensorrt_fp32 | 98.59% | 99.21% | 98.90% | 1262 / 18 / 10 | 1.225 / 2.695 / 3.351 |
| YOLO26n | tensorrt_fp16 | 98.67% | 99.21% | 98.94% | 1262 / 17 / 10 | 1.250 / 2.738 / 3.325 |
| YOLO26s | pytorch_fp32 | 99.14% | 99.61% | 99.37% | 1267 / 11 / 5 | 1.201 / 2.683 / 3.314 |
| YOLO26s | onnx_cpu_fp32 | 99.14% | 99.61% | 99.37% | 1267 / 11 / 5 | 1.201 / 2.683 / 3.314 |
| YOLO26s | tensorrt_fp32 | 99.14% | 99.61% | 99.37% | 1267 / 11 / 5 | 1.201 / 2.683 / 3.314 |
| YOLO26s | tensorrt_fp16 | 99.06% | 99.61% | 99.33% | 1267 / 12 / 5 | 1.250 / 2.719 / 3.272 |

### 测试集830图 / 859球

| 模型 | 后端 | P | R | F1 | TP / FP / FN | 中位 / P95 / P99 px |
| --- | --- | --- | --- | --- | --- | --- |
| YOLO26n | pytorch_fp32 | 93.40% | 97.21% | 95.27% | 835 / 59 / 24 | 1.184 / 2.444 / 3.118 |
| YOLO26n | onnx_cpu_fp32 | 93.40% | 97.21% | 95.27% | 835 / 59 / 24 | 1.184 / 2.444 / 3.118 |
| YOLO26n | tensorrt_fp32 | 93.40% | 97.21% | 95.27% | 835 / 59 / 24 | 1.184 / 2.444 / 3.118 |
| YOLO26n | tensorrt_fp16 | 93.30% | 97.21% | 95.21% | 835 / 60 / 24 | 1.176 / 2.465 / 3.060 |
| YOLO26s | pytorch_fp32 | 94.28% | 97.79% | 96.00% | 840 / 51 / 19 | 1.200 / 2.467 / 3.078 |
| YOLO26s | onnx_cpu_fp32 | 94.28% | 97.79% | 96.00% | 840 / 51 / 19 | 1.200 / 2.467 / 3.078 |
| YOLO26s | tensorrt_fp32 | 94.28% | 97.79% | 96.00% | 840 / 51 / 19 | 1.200 / 2.467 / 3.078 |
| YOLO26s | tensorrt_fp16 | 94.49% | 97.79% | 96.11% | 840 / 49 / 19 | 1.186 / 2.504 / 3.121 |

PT FP32、ORT CPU FP32、TRT FP32在本轮原图主指标一致，仍可能有微小坐标差。ORT使用CPU两线程作为数值参考，不能与GPU速度作公平比较。FP16是半精度/混合精度推理；本轮未做INT8校准、训练或评估，暂无INT8结果。

## 转换位移、精度变化与新增漏检

这里的位移比较同权重、同尺寸、同batch的PT FP32预测与转换预测，覆盖val+test 2102张，仅对两侧conf≥.25且预测中心≤10px的匹配预测对统计，也可能含共同误检；无法配对输出另计。P95为95百分位，max为本次已匹配预测对的最大位移，不是未来所有画面的误差上限。它们与相对GT的定位误差是两种统计。

| 模型 | FP16 batch | test ΔP / ΔR / ΔF1（百分点） | 位移中位 / P95 / P99 / max px | 新增5px漏匹配 / 恢复 | 未配对PT / TRT输出 |
| --- | --- | --- | --- | --- | --- |
| YOLO26n | 1 | +0.1046 / +0.0000 / +0.0544 | 0.0796 / 0.2025 / 0.2806 / 0.4528 | 0 / 0 | 1 / 1 |
| YOLO26n | 2 | -0.1044 / +0.0000 / -0.0543 | 0.0802 / 0.1992 / 0.2806 / 0.4528 | 0 / 0 | 1 / 1 |
| YOLO26s | 1 | +0.0000 / +0.0000 / +0.0000 | 0.0782 / 0.2047 / 0.2850 / 0.4627 | 0 / 0 | 1 / 1 |
| YOLO26s | 2 | +0.2121 / +0.0000 / +0.1098 | 0.0781 / 0.2047 / 0.2850 / 0.4627 | 0 / 0 | 2 / 1 |

新增漏匹配与恢复是逐GT对照，不能只从总召回率推断；完整置信度差、变化目标路径和两侧未配对输出保存在各矩阵against_pytorch中。FP16量化舍入会改变坐标和阈值附近置信度，指标略升或略降都不代表重新训练了模型。

## 同协议与上一版合并模型比较

旧B/C分别为上一版合并数据（v1＋additional_v1）的n/s，train17404图/22181球；本轮D/E增加10997图。共同官方预训练、seed和训练策略不变，但每轮优化步数、最佳epoch及样本分布都变化，不能单凭一次结果确定误检增加的原因。下表原图TRT FP16 batch2、test，阈值均.25。

| 模型版本 | P / R / F1 | TP / FP / FN | 中位 / P95 / P99 px | 本机普通GPU批次ms |
| --- | --- | --- | --- | --- |
| n / 旧合并v1 | 94.87% / 96.86% / 95.85% | 832 / 45 / 27 | 1.205 / 2.485 / 3.105 | 3.134 |
| n / 新合并v2 | 93.30% / 97.21% / 95.21% | 835 / 60 / 24 | 1.176 / 2.465 / 3.060 | 3.199 |
| s / 旧合并v1 | 95.45% / 97.67% / 96.55% | 839 / 40 / 20 | 1.231 / 2.601 / 3.267 | 4.917 |
| s / 新合并v2 | 94.49% / 97.79% / 96.11% | 840 / 49 / 19 | 1.186 / 2.504 / 3.121 | 4.919 |

- n：test P / R / F1变化为-1.573 / +0.349 / -0.642个百分点；新增匹配3个球。
- s：test P / R / F1变化为-0.961 / +0.116 / -0.438个百分点；新增匹配1个球。

本轮val F1提高而test F1下降，表明增量收益没有在现有test上全面体现；并非“数据越多必然越好”。旧n继续作为速度优先目标机候选保留，新s的定位尾部和召回值得进一步对照；没有依据直接替换全部旧模型。各配置仅单次训练，微小差异未做多seed显著性检验。

## RTX3090最终纯GPU速度

最终复测使用独立非默认CUDA stream；预热200次，普通执行和CUDA Graph分别1000次。批次ms为CUDA event均值，图/s为包含调用及同步开销的墙钟吞吐；表中P95是批次耗时P95。输入输出已在显存，排除解码、预处理、H2D/D2H、后处理、相机、网络、双目匹配和三维计算。Graph需部署程序实现。

| 模型 | 后端 | batch | 普通均值 / P95 ms | 普通图/s | Graph均值 ms / 图/s |
| --- | --- | --- | --- | --- | --- |
| YOLO26n | pytorch_fp32 | 2 | 8.707 / 9.776 | 228.4 | 7.037 / 281.7 |
| YOLO26n | tensorrt_fp32 | 2 | 5.630 / 5.732 | 351.9 | 5.012 / 394.3 |
| YOLO26n | tensorrt_fp16 | 1 | 2.307 / 2.944 | 426.7 | 1.474 / 659.5 |
| YOLO26n | tensorrt_fp16 | 2 | 3.199 / 3.500 | 616.5 | 2.407 / 814.0 |
| YOLO26s | pytorch_fp32 | 2 | 17.053 / 17.238 | 116.7 | 15.874 / 125.3 |
| YOLO26s | tensorrt_fp32 | 2 | 12.237 / 12.348 | 162.4 | 11.569 / 171.8 |
| YOLO26s | tensorrt_fp16 | 1 | 3.150 / 3.604 | 313.0 | 2.472 / 397.1 |
| YOLO26s | tensorrt_fp16 | 2 | 4.919 / 5.030 | 401.8 | 4.315 / 457.7 |

- n：同batch2从PT FP32到TRT FP16普通GPU耗时降低63.3%，约2.72倍；离线PNG解码流水线实测48.1图/s，后者不是已优化相机串流性能。
- s：同batch2从PT FP32到TRT FP16普通GPU耗时降低71.2%，约3.47倍；离线PNG解码流水线实测46.4图/s，后者不是已优化相机串流性能。

目标120对/秒=240图/秒，每对平均间隔8.33ms；离线batch2并不保证图片来自同一时刻的左右目。上述3090 GPU结果不能证明Ubuntu/2060 Super端到端达标，也不能按硬件比例可靠外推。当前尚无目标机测试结果。

## 降分辨率对照（TRT FP16 batch2）

| 模型 | 输入宽×高 | test P / R / F1 | 中位 / P95 / P99 px | 普通GPU ms / 图/s |
| --- | --- | --- | --- | --- |
| YOLO26n | 1280×1024 | 93.30% / 97.21% / 95.21% | 1.176 / 2.465 / 3.060 | 3.199 / 616.5 |
| YOLO26n | 960×768 | 90.56% / 94.88% / 92.67% | 1.275 / 2.831 / 3.500 | 2.360 / 835.3 |
| YOLO26s | 1280×1024 | 94.49% / 97.79% / 96.11% | 1.186 / 2.504 / 3.121 | 4.919 / 401.8 |
| YOLO26s | 960×768 | 90.25% / 92.67% / 91.44% | 1.803 / 3.360 / 4.058 | 3.459 / 570.8 |

960×768是全图等比缩到75%，没有ROI裁切；其原图坐标误差已还原。缩小推理降低计算时间，同时损失球心检出指标，仍优先原图1280×1024。本轮没有裁切收益实测，不能据此保证ROI一定适用。

## 权重身份、转换配置与复现


| 模型 | 评估使用的best.pt | SHA256 |
| --- | --- | --- |
| YOLO26n | [runs/detect/native_merged_v2_baiduout_yolo26n_1280/weights/best.pt](../../runs/detect/native_merged_v2_baiduout_yolo26n_1280/weights/best.pt) | `d25c43e4a05b7e1fc72614341b572a0a22882c295409fe039508a3d3dd566e1c` |
| YOLO26s | [runs/detect/native_merged_v2_baiduout_yolo26s_1280/weights/best.pt](../../runs/detect/native_merged_v2_baiduout_yolo26s_1280/weights/best.pt) | `433c47f818aaf07b4e1d34df64a7bad3ee744d1fb21b1f1d00b08915af9a89f0` |


| 模型 | 官方预训练初始化 | 源SHA256 |
| --- | --- | --- |
| YOLO26n | models/yolo26n.pt | `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` |
| YOLO26s | models/yolo26s.pt | `646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b` |

完整固定配置：[suite JSON](../../configs/native_merged_v2_baiduout_20261113.json)。每模型实际参数保存在runs/detect/同名目录args.yaml，逐轮训练CSV在results.csv；配置及源哈希在reports/training/同名目录configuration.json，球心曲线在epoch_metrics.jsonl，选模与结束时间在result.json。模型包models/pingpong_yolo26n_merged_v2_baiduout和对应s目录仅引用原始best，不重复存储权重。

ONNX opset17、dynamic=True、simplify=True、免NMS；TensorRT每个engine的min=opt=max固定[B,3,H,W]，workspace2GiB、builder优化级别3、关闭TF32；FP16 mixed，检测头相关Sigmoid保留FP32（准确节点清单见build.json）。每模型2份ONNX、6份engine。原生engine没有Ultralytics文件头，动态ONNX不代表现有固定engine可任意改尺寸或batch。

转换文件位于artifacts/<run_id>_conversion_v1/h1024_w1280（以及h768_w960），包含weights.onnx、trt_fp16_b2.engine、对应.build.json及逐图NPZ缓存；source.pt在该实验artifacts根目录。部署应迁移PT/ONNX与构建参数，在Ubuntu/2060 Super核对Turing兼容软件并重建engine，不能直接沿用Windows/3090 engine。

精度矩阵预处理：OpenCV INTER_LINEAR仅用于960档，原图档不缩放；BGR→RGB、CHW、FP32除255，无padding；结果还原到1280×1024坐标。训练环境Windows RTX3090 24GiB、Python3.12.3、torch2.11.0+cu128/CUDA12.8、Ultralytics8.4.207、TensorRT10.16.1.11、ONNX1.23.2、ORT1.31.1 CPU参考；版本原始记录在每个matrix的versions。

实际队列顺序是每模型训练→完整矩阵→最终复测→汇总，再进入下个模型。下面列出已执行命令的模板用于学习与追溯，**本轮已完成，无需重新启动**：

```powershell
& ./.venv/Scripts/python.exe scripts/run_native_suite.py --suite native_merged_v2_baiduout_20261113 --launch
& ./.venv/Scripts/python.exe scripts/train_native_experiment.py --suite native_merged_v2_baiduout_20261113 --run <run_id>
& ./.venv/Scripts/python.exe scripts/evaluate_conversion_matrix.py --experiment <run_id>_conversion_v1 --weights runs/detect/<run_id>/weights/best.pt --resume
& ./.venv/Scripts/python.exe scripts/rebenchmark_conversion_matrix.py --experiment <run_id>_conversion_v1
```

只刷新现有报告、核验缓存和生成曲线（不训练、不推理）：

```powershell
& ./.venv/Scripts/python.exe scripts/finalize_followup_suite.py --suite native_merged_v2_baiduout_20261113
```

本次核验通过：24份缓存完整、12份engine及4份ONNX哈希一致、20个GPU组合完成200预热/每模式1000次测试、2组训练轮数与选模一致、holdout列表与历史一致。核验时D盘空余33.52GiB，满足≥5GiB要求。未重新逐字节读取源目录图片；源数据审计以此前逐图记录为准。

## 曲线与对照

![训练球心曲线与原图FP16对照](../../reports/experiments/native_merged_v2_baiduout_20261113/native_results.png)

[导出SVG](../../reports/experiments/native_merged_v2_baiduout_20261113/native_results.svg)

## 下一阶段边界

反光眼睛、手表及白色小物体专项仍缺已确认无球数据。应采集真实相机连续片段，人工确认无球后创建同名空YOLO标签文件；包含真球的困难画面需完整标注全部球，不能把漏标真球作为负样本。按录制/场景整体分train、val和新独立test，避免相邻帧及左右目泄漏；训练加入困难负样本，保留无球测试集统计每图/每分钟误检及有球片段召回。当前模型的FP不能替代这项专项结论。

目前没有INT8、目标卡端到端、相机串流或三维定位的完成证据。下一步先在目标卡以原图、batch1/2、TRT FP16测真实流水线；若做INT8，校准只用train代表样本，再以固定协议比较球心P/R/F1、分位误差和新增漏检。
