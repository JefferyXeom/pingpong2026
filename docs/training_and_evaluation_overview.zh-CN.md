# 当前训练、模型与评估总览

最新批次`native_merged_v2_baiduout_20261113`已于北京时间**2026-11-14 06:47:47**完成训练、转换评估和最终测速，见[本轮详细最终报告](experiments/native_merged_v2_baiduout_20261113_results.zh-CN.md)及[含历史72组合CSV](../reports/experiments/native_merged_v2_baiduout_20261113/all_models_combinations.csv)。新增10,997张/13,912球，合并train28,401张/36,093球，固定val1,272张/1,272球、test830张/859球；n/s实际训练32/42轮，选第12/22轮best。原图训练/推理宽1280×高1024，batch8训练；本轮独立24推理组合。TRT FP16 batch2测试n：P93.30%、R97.21%、F1 95.21%、误差中位/P95/P99=1.176/2.465/3.060px；s：94.49%/97.79%/96.11%、1.186/2.504/3.121px。新模型召回略升但误检增加，未全面超过旧B/C，旧目标机候选继续保留。模型卡位于`models/pingpong_yolo26n_merged_v2_baiduout`及对应s目录。以下为此前批次的历史详情。

截至2026-11-10，三组原图训练及全部转换评估均已完成；实际训练53/44/50轮，分别选中第33/24/30轮。含旧基线共48项组合。优先阅读[原图最终结果分析](experiments/native_resolution_suite_v1_results.zh-CN.md)及[48组合总表](experiments/native_resolution_suite_v1.zh-CN.md)。本页后续保留旧960基线的完整历史细节。

当前目标机优先验证候选为原图合并数据YOLO26n第24轮best：`runs/detect/native_v1_additional_yolo26n_1280/weights/best.pt`，优先1280×1024＋TRT FP16。合并YOLO26s第30轮作为漏检/速度对照，尚未在2060 Super定版。旧`models/pingpong_yolo26n_center_v1/weights.pt`仍是第28轮历史模型包；其旧评估文件保留原模型身份。

## 报告与产物的阅读顺序

1. [reports分类导航](../reports/README.md)：按环境、数据集、训练、评估批次找文件；每批次有自己的README。
2. [实验登记](../reports/experiment_registry.json)：每次训练/评估的模型ID、数据量、参数及源权重哈希。
3. [Baidu增量最终报告](experiments/native_merged_v2_baiduout_20261113_results.zh-CN.md)为最新n/s比较；[此前原图三组结果](experiments/native_resolution_suite_v1_results.zh-CN.md)、[旧基线评估](conversion_evaluation_results.zh-CN.md)与[协议](conversion_evaluation_protocol.zh-CN.md)保留旧模型证据。
4. `runs/`为原始训练配置、CSV、权重；`artifacts/`为ONNX、TensorRT、预测缓存及构建记录；`logs/`为执行日志。报告目录只组织说明和分析，不重复复制数据集。

## 数据版本与使用范围

| 数据版本/集合 | 图片 | 球框 | 图片容量 | 使用范围（此表v1划分同旧基线） |
| --- | ---: | ---: | ---: | --- |
| pingpong_v1 / train | 6,618 | 8,139 | 10.88 GiB | 优化权重 |
| pingpong_v1 / val | 1,272 | 1,272 | 1.90 GiB | 验证/选模 |
| pingpong_v1 / test | 830 | 859 | 1.23 GiB | 选模后独立测试 |
| pingpong_v1 合计 | 8,720 | 10,270 | 14.00 GiB | 全部为原始1280×1024图片 |
| pingpong_additional_v1 | 10,786 | 14,042 | 13.55 GiB | 已用于原图合并n/s训练；未用于旧960基线 |

主数据源为桌面`yolov5data`；独立整理副本为`datasets/pingpong_v1`。train包含june01_recording1（1,520张）、june01_recording2（2,997张）、may27_recording1（2,101张）；val整批留出may27_recording3，test整批留出may27_recording2。左右目留在同一集合；文件内容哈希没有跨集合重复。目录批次不能证明不同场地隔离，也不能排除所有近似画面泄漏。

标注从VOC转换为YOLO，沿用旧项目约定：中心`((min+max)/2-1)/size`，宽高`(max-min)/size`。图片/标注独立复制并核对，原始XML保留。主数据4,185张缺标注图片未作为负样本；当前已验证无球负样本数为0。

新增历史源为`D:/yolov5-master/VOCData`及`history_VOCOdata`，去重备份10,786张：7,789张其他原图及2,997张_white换背景变体。变体均匹配已有train原图。新增数据已由原图实验派生列表全部加入合并train；未改变原val/test，未参与旧960基线。桌面processed_images/waste_data未加入首轮。

## 历史960基线的两次训练（新三组见最终结果）

| 项目 | smoke_yolo26n_960 | baseline_yolo26n_960 |
| --- | --- | --- |
| 初始化模型 | models/yolo26n.pt | models/yolo26n.pt |
| 每轮train | 192张／234球框 | 6,618张／8,139球框 |
| 每轮val | 64张／64球框 | 1,272张／1,272球框 |
| test是否参与训练 | 否 | 否 |
| batch / 每轮训练批数 | 8 / 24 | 16 / 414（末批10张） |
| 训练输入宽×高 | **960×960**（imgsz=960，rect=False） | **960×960**（imgsz=960，rect=False） |
| 计划/实际epochs | 1 / 1 | 最多100 / 实际28 |
| 结束原因 | 达到1轮 | best在第3轮，连续25轮框fitness未改善，patience=25早停 |
| 中断恢复 | 无 | 手动中断后，从已完成第24轮恢复至28轮；同一次训练 |
| CSV记录耗时 | 23.18秒，非完整启动总耗时 | 约63.83分钟，含轮内验证，不含暂停 |
| 当前用途 | 验证训练/导出/推理流程 | 历史center_v1候选来源 |

**原图宽×高为1280×1024；smoke和baseline的训练网络输入均为960×960。** 依据两份实际args.yaml：imgsz=960、rect=False、multi_scale=0。正式训练的一批张量通常为16×3×960×960（batch、RGB通道、高、宽），末批为10张；smoke为8×3×960×960。

仅做等比缩放与letterbox补边时，1280×1024先缩为960×768，再上下各补96像素，形成960×960，不会直接把原图拉伸成正方形。实际训练还使用mosaic、随机缩放/平移和翻转，因此每个训练样本不都等于这条简单补边路径，但最终送入网络的空间尺寸固定为960×960。

正式推理评估另测960×768（等比缩小、不补边）和1280×1024（原图、不缩放）。训练输入与推理输入尺寸分别记录；不能把960×768评估尺寸写成这次训练输入。模型日志记录训练结构为260层、2,504,190参数。28轮重复使用同一6,618张train。

## smoke、精度与运行时分别是什么

**smoke是smoke test（冒烟测试）**，用于以少量数据和很短训练检查读取标签、GPU训练、保存权重、导出与推理能否走通。本项目为192张train、64张val、1轮；smoke不是另一种YOLO架构，正式baseline从预训练yolo26n.pt单独开始，没有接着smoke权重训练。

**FP32/FP16描述数值表示与计算精度。** 旧基线正式矩阵各后端均来源于相同第28轮候选；新矩阵分别来自对应原图训练best，无须分别重训。FP16使用更少位数表示部分权重/中间结果，TensorRT还进行算子融合、内核选择等优化；网络的检测任务和学习来源保持相同，数值与结果可能变化。这不同于剪枝、减少网络层或换成更小的YOLO型号。实际FP16引擎允许混合精度，不是所有运算/输入输出都为FP16。训练本身AMP=True，也不能称为从头到尾纯FP32训练。INT8尚未做，通常另需校准，部分方案会做量化感知训练。

**ONNX是模型交换格式，ONNX Runtime（ORT）是加载和执行ONNX模型的推理引擎。** 同一.pt可导出.onnx，由ORT运行，也可由TensorRT解析ONNX并构建.engine。ONNX文件不是又训练出来的一份独立模型，ORT也不是训练算法。ORT可使用CPU或受支持的GPU后端；本次只用CPU执行作数值参考，最终GPU加速比较使用TensorRT。

实际配置：GPU0 RTX3090，workers=2，AMP=True，cache=False，seed=20261108，deterministic=True，fraction=1.0，不冻结层，multi_scale=0，验证开启，save_period=-1，仅保留best/last。基线nms=False。smoke原始args.yaml的nms=null；后续脚本已显式指定False，不能用现行脚本值覆盖早期运行事实。

优化器入口为optimizer=auto，实际日志选择MuSGD，初始lr=0.01；weight_decay=0.0005，lrf=0.01，warmup_epochs=3，cos_lr=False。原始启动日志记录momentum=0.9，恢复日志为0.937，恢复后args.yaml记录0.937；不将两阶段写成完全相同的动量配置。实际学习率变化见results.csv。

增强：mosaic=0.25、scale=0.25、translate=0.1、fliplr=0.5、hsv_h/s/v=0.015/0.7/0.4；degrees=0、flipud=0、mixup=0、copy_paste=0、shear=0、perspective=0。baseline的close_mosaic=10按计划训练末段关闭；实际在28/100早停，不能声称已经经历最后10轮关闭mosaic阶段。smoke的close_mosaic=0。

完整原始参数：[baseline args.yaml](../runs/detect/baseline_yolo26n_960/args.yaml)、[smoke args.yaml](../runs/detect/smoke_yolo26n_960/args.yaml)。实际逐轮损失、框P/R、mAP、学习率见[baseline results.csv](../runs/detect/baseline_yolo26n_960/results.csv)。启动与恢复日志为logs/baseline.stdout.log、baseline.stderr.log。

训练完成后Ultralytics剥离了best/last的优化器状态；这两份最终权重可用于推理或新一轮微调初始化，不能作为保留原优化器状态的直接resume断点。诊断快照保留的是中断阶段状态，不代表应从旧第24轮重新训练。

## 权重身份与评估对应关系

| 模型ID | epoch | 文件 | 用途 |
| --- | ---: | --- | --- |
| pretrained_yolo26n | 预训练 | `models/yolo26n.pt` | 预训练初始化；不是已训练的乒乓球模型 |
| smoke_epoch1 | 1 | `runs/detect/smoke_yolo26n_960/weights/best.pt` | 小规模训练及早期转换流程验证 |
| baseline_best_epoch3 | 3 | `runs/detect/baseline_yolo26n_960/weights/best.pt` | 按验证框mAP选择的best；完成后的推理权重 |
| diagnostic_best_epoch3 | 3 | `reports/evaluation/baseline_yolo26n_960/diagnostics_epoch24/baseline_diagnostics/best.pt` | 第24轮中断阶段保留的best诊断快照 |
| diagnostic_last_epoch24 | 24 | `reports/evaluation/baseline_yolo26n_960/diagnostics_epoch24/baseline_diagnostics/last.pt` | 第24轮中断阶段last诊断快照；不是第28轮 |
| baseline_last_epoch28 | 28 | `runs/detect/baseline_yolo26n_960/weights/last.pt` | 最终last；按完整验证球心指标选中 |
| pingpong_yolo26n_center_v1 | 28 | `models/pingpong_yolo26n_center_v1/weights.pt` | 历史center_v1模型包；旧矩阵的唯一源模型 |
| native_v1_yolo26n_1280_best | 33 | `runs/detect/native_v1_yolo26n_1280/weights/best.pt` | 原图训练；按完整val球心F1选中，用于同名转换矩阵。 |
| native_v1_additional_yolo26n_1280_best | 24 | `runs/detect/native_v1_additional_yolo26n_1280/weights/best.pt` | 原图训练；按完整val球心F1选中，用于同名转换矩阵。 |
| native_v1_additional_yolo26s_1280_best | 30 | `runs/detect/native_v1_additional_yolo26s_1280/weights/best.pt` | 原图训练；按完整val球心F1选中，用于同名转换矩阵。 |

历史center_v1候选SHA256：`fb770bff46a07b46234fd32b816842ad9e6d65880cbbd038e988d6cbe579db3f`。它与最终第28轮last.pt、正式转换矩阵保存的source.pt相同。所有权重的文件哈希已登记到experiment_registry.json；第3轮诊断checkpoint与训练结束剥离优化器后的best即使来自相同epoch，整个文件哈希也可能不同。

框best依据验证框mAP50:95选择：第3轮CSV mAP50=15.126%、mAP50:95=4.012%；训练正常结束后重新验证best，result.json约为14.974%/3.993%。这些是第3轮验证框指标，不是当前第28轮候选的测试指标。球心候选只在完整val对比best与last，并不是保存/搜索了28轮全部checkpoint。

## 已执行的评估批次

| 批次 | 使用模型 | 数据范围 | 预处理/后端 | 回答的问题 |
| --- | --- | --- | --- | --- |
| smoke_conversion_v1 | smoke第1轮best | val/test均匀抽样8张 | 固定方形960×960、batch1，PyTorch/ONNX CPU/TRT FP16；传统头与免NMS各一份 | 转换、加载和真实图片推理是否走通；不是精度认证 |
| diagnostics_epoch24 | 保存的best第3轮＋last第24轮 | train384张、val256张，test未用 | imgsz960、rect=True、batch16；免NMS，并附第24轮val传统NMS对照 | 低框AP原因、训练/验证差异、框尺寸与球心 |
| center_selection_v1 | 完整val对比第3轮best和第28轮last；test只用选中的第28轮 | val1,272张、test830张 | PyTorch，imgsz960、rect=True、batch16、nms=False；conf0.05/0.25/0.5 | 按固定conf0.25球心F1选择候选，再独立测试 |
| center_conversion_matrix_v1 | **只有第28轮候选center_v1** | 每项完整val1,272张＋test830张 | 两档尺寸、PyTorch FP32 b1/b2、ONNX CPU FP32 b2、TRT FP32 b2、TRT FP16 b1/b2，共12项 | 原图输入收益、转换精度变化、GPU耗时 |

正式矩阵的960档为宽960×高768，OpenCV INTER_LINEAR等比缩小；原图档宽1280×高1024不缩放。两档均不补边、不裁剪，BGR→RGB、CHW、FP32/255归一化；输出恢复到原始坐标。所有输入都是原始数据，不包含换背景图。batch2按离线顺序组成，不保证同步左右目。

球心指标用原图坐标一一贪心最近匹配，5像素用于P/R/F1，10像素成功匹配用于误差分位数；漏检和额外预测另计。矩阵保存conf≥0.001候选，统计0.05/0.25/0.5阈值，主结果固定0.25。框AP采用IoU0.5:0.95，与球心指标分开。初始center_selection使用Ultralytics预处理，矩阵使用显式cv2预处理，数值可能有微小差异；应在同一评估协议内比较。

## 历史center_v1候选的全量结果

| 输入宽×高 | 后端 / batch | 数据 | 球心P | 球心R | 球心F1 | 中位误差 px | P95 px |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 960×768 | pytorch_fp32 / 2 | val | 96.87% | 97.33% | 97.10% | 1.254 | 2.764 |
| 960×768 | pytorch_fp32 / 2 | test | 93.96% | 95.93% | 94.93% | 1.210 | 2.461 |
| 960×768 | tensorrt_fp16 / 2 | val | 96.87% | 97.33% | 97.10% | 1.250 | 2.730 |
| 960×768 | tensorrt_fp16 / 2 | test | 93.96% | 95.93% | 94.93% | 1.213 | 2.505 |
| 1280×1024 | pytorch_fp32 / 2 | val | 96.83% | 96.07% | 96.45% | 1.310 | 2.837 |
| 1280×1024 | pytorch_fp32 / 2 | test | 92.65% | 95.34% | 93.98% | 1.317 | 2.735 |
| 1280×1024 | tensorrt_fp16 / 2 | val | 96.67% | 95.99% | 96.33% | 1.288 | 2.828 |
| 1280×1024 | tensorrt_fp16 / 2 | test | 92.54% | 95.34% | 93.92% | 1.317 | 2.742 |

同尺寸ONNX/TRT FP32的球心P/R/F1与PyTorch一致，坐标差为微小浮点数值量级。FP16 b2在960档没有新增5像素漏检，球心位移P95约0.213像素；原图档位移P95约0.200像素，但val新增1个漏检。该例候选置信度从0.25565降至0.24219，被0.25阈值过滤；位置误差仍约1.25像素。没有为了修复此例而调阈值。

原图PyTorch test球心F1为93.98%，低于960档94.93%，P95也从2.461增至2.735像素。本次是960训练权重换推理尺寸，不能推出原图尺寸重新训练没有价值。标准框指标仍低，960候选test mAP50约0.23%；标签框与预测框宽高存在约两倍差异，球心F1不能称为mAP。

## 转换与测速的具体设置

每档导出FP32 ONNX：opset17、simplify=True、dynamic=True、batch=2、nms=False。TensorRT用该ONNX分别建立固定min=opt=max的b1/b2及高宽profile；两种空间尺寸的固定engine不能任意换输入。输出为B×300×6免NMS检测。每档生成FP32 b2、FP16 b1、FP16 b2共3个原生engine；两档ONNX合计2份、engine合计6份。原生engine不含Ultralytics文件头，需使用本工作区Runner或相容的原生TensorRT加载器，不能假设普通YOLO(engine)直接可用。

TensorRT workspace=2GiB、builder_optimization_level=3，保存timing cache、构建参数/耗时与模型哈希。FP32关闭TF32；FP16允许混合精度，检测头概率Sigmoid保留FP32。未执行INT8量化或校准。

ONNX Runtime使用CPUExecutionProvider、2线程作数值参考。本机ORT GPU需要CUDA13 DLL，而PyTorch为CUDA12.8，未为它改动训练环境；CPU耗时不参与GPU加速比。

GPU最终复测在全部构建结束后顺序执行，PyTorch/TRT都使用独立非默认CUDA stream：普通执行预热200次、测量1000次；Graph捕获前另行预热，捕获后回放10次预热、1000次测量。CUDA event耗时与包含调用/同步开销的墙钟吞吐分列，P50/P95/P99保存在主报告。初测30/200留在initial_benchmark。

| 输入宽×高 | 后端，batch2 | 普通批次 ms | 普通同步吞吐 图/s | Graph批次 ms | Graph同步吞吐 图/s |
| --- | --- | ---: | ---: | ---: | ---: |
| 960×768 | pytorch_fp32 | 6.339 | 313.2 | 4.709 | 419.1 |
| 960×768 | tensorrt_fp32 | 4.177 | 473.6 | 3.367 | 585.8 |
| 960×768 | tensorrt_fp16 | 2.421 | 813.4 | 1.567 | 1243.3 |
| 1280×1024 | pytorch_fp32 | 8.726 | 227.6 | 7.187 | 275.6 |
| 1280×1024 | tensorrt_fp32 | 5.804 | 341.4 | 5.147 | 384.6 |
| 1280×1024 | tensorrt_fp16 | 3.165 | 624.1 | 2.413 | 814.0 |

GPU基准输入已在显存，不含图片解码、预处理、H2D/D2H和后处理；精度遍历另记离线PNG流程（2个CPU工作线程）总耗时。后者不是优化相机流水线的上限。全部速度来自Windows RTX3090，不能外推为2060 Super的实测结果。

环境：Python3.12.3、PyTorch2.11.0+cu128、CUDA12.8、Ultralytics8.4.207、TensorRT10.16.1.11、ONNX1.23.2、ONNX Runtime1.31.1；RTX3090 24GB、驱动591.86。[依赖快照](../reports/environment/windows_rtx3090/environment_freeze.txt)。目标Ubuntu＋2060 Super由用户提供，尚未核验或实测。

## 命令、产物与复现边界

以下在工作区根目录PowerShell执行，列出各阶段入口。它们用于理解已经做过的工作；现有目录有防覆盖检查，训练已完成，不应把整个代码块当作需要立即重跑的流水线。完整参数同时保存在args.yaml和报告中。

```powershell
# 数据审计、独立复制与验证（数据已存在，不需再次复制）
py -3.12 scripts/audit_dataset.py
& D:/ProgramData/anaconda3/python.exe scripts/prepare_dataset.py
& D:/ProgramData/anaconda3/python.exe scripts/verify_dataset.py
# 两次训练；baseline恢复为历史操作，最终权重已剥离优化器
& ./.venv/Scripts/python.exe scripts/train.py --stage smoke
& ./.venv/Scripts/python.exe scripts/start_baseline.py
& ./.venv/Scripts/python.exe scripts/start_baseline.py --resume
# 完整val比较best/last，test只评估选定的last
& ./.venv/Scripts/python.exe scripts/evaluate_centers.py --split val --name baseline_full_val_centers --weights runs/detect/baseline_yolo26n_960/weights/best.pt runs/detect/baseline_yolo26n_960/weights/last.pt
& ./.venv/Scripts/python.exe scripts/evaluate_centers.py --split test --name baseline_full_test_centers --weights runs/detect/baseline_yolo26n_960/weights/last.pt
# 正式尺寸/转换矩阵；已有12项时resume会跳过，不能据此称为重新推理
& ./.venv/Scripts/python.exe scripts/evaluate_conversion_matrix.py --resume
& ./.venv/Scripts/python.exe scripts/rebenchmark_conversion_matrix.py
# 从缓存更新可视化/报告及做完整性检查，不重新训练
& ./.venv/Scripts/python.exe scripts/review_conversion_regressions.py
& ./.venv/Scripts/python.exe scripts/summarize_conversion_matrix.py
& ./.venv/Scripts/python.exe scripts/review_conversion_predictions.py
& ./.venv/Scripts/python.exe scripts/verify_conversion_report.py
```

正式导出产物：`artifacts/center_conversion_matrix_v1/h768_w960`与`h1024_w1280`；各目录保存ONNX、固定engine、build.json及预测npz。`image_order.json`记录完整val/test顺序与图片哈希；`source.pt`是同哈希的第28轮候选副本。

训练/转换脚本有磁盘余量检查；要求D盘保留至少5GiB，训练低于5.5GiB停止，正式转换低于8GiB停止。此次分类采用移动，不复制大图片或权重；旧/新路径及迁移前后哈希见reports/report_layout_migration.json。

## 当前结论与尚未覆盖的工作

原图三组已经完成，当前优先验证合并数据YOLO26n＋1280×1024＋TRT FP16；合并YOLO26s保留为对照。旧960基线仅作历史比较。目标为120对双目/秒，即240图/秒，每对到达间隔8.33ms；3090纯GPU结果不能保证目标卡端到端达标。目标机需要重建相容2060 Super/Turing的软件与engine，并测队列、丢帧、持续吞吐和延迟。

仍需真实无球/反光难负样本测试、统一标注质量、真实双目同步/三角化与串流验证。新增YOLO26s、原图及合并数据对照均已完成，结果见原图最终结果分析。二维像素误差不是三维物理定位精度。

旧诊断报告保留当时的“原因未确定”等历史判断；用户后来确认手动中断，基线已恢复完成。模型包说明和此总览已更新FP16评估状态；INT8及无球专项仍未完成。
