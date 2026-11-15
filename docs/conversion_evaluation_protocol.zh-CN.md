# 正式模型的尺寸、转换精度与耗时评估

脚本为 `scripts/evaluate_conversion_matrix.py`，输入为第 28 轮球心定位候选 `models/pingpong_yolo26n_center_v1/weights.pt`。报告路径为 `reports/evaluation/center_conversion_matrix_v1/center_conversion_matrix_v1.json`，逐阶段进度为 `reports/evaluation/center_conversion_matrix_v1/center_conversion_matrix_progress.json`，模型、原始检测缓存及构建参数在 `artifacts/center_conversion_matrix_v1`。脚本会保存源权重 SHA256 及包含源图哈希的图片顺序清单；不会修改训练数据、标注或 checkpoint。

## 数据与输入

固定使用完整验证集 1,272 张、完整测试集 830 张，共 2,102 张，分别报告指标。所有图像原始尺寸为宽 1280、高 1024。

- 960 长边：固定网络高×宽为 768×960，使用 OpenCV INTER_LINEAR 等比例缩小，不补边、不裁剪。
- 原图：固定网络高×宽为 1024×1280，不做空间缩放，保留全部采集像素。

两档都执行 BGR→RGB、HWC→CHW、FP32/255 归一化。输出框按准确缩放比例恢复为原图坐标。原图档使用当前 960 训练权重评估，不是重新训练的 1280 模型。

对同一档输入比较：PyTorch FP32 batch=1/2、ONNX CPU FP32 batch=2、TensorRT FP32 batch=2、TensorRT FP16 batch=1/2，共 12 项。batch=2 是离线图像批处理能力测试，现有图片的顺序不保证同步左右目，不能据此声明双目三维精度或相机端到端帧率。

## 转换与数值设置

每个空间尺寸导出一份动态 ONNX（FP32、opset17、simplify、nms=False），由同一计算图构建固定 batch/高宽的 TensorRT profile，min=opt=max。引擎为原生 TensorRT 文件，不含 Ultralytics 元数据文件头，测试器直接绑定输入输出并执行。

PyTorch 禁用 TF32，TensorRT FP32 清除 TF32 标志，避免在 RTX3090 上把 TF32 加速误标成严格 FP32 比较。FP16 引擎允许混合精度，检测头概率 Sigmoid 保留 FP32；不把所有 backbone SiLU 的 Sigmoid 强制设为 FP32。构建器 workspace=2GiB、optimization_level=3，共享同精度 timing cache，保存参数、哈希及构建耗时。

环境中 ONNX Runtime GPU 1.31.1 需要 CUDA13 库，而当前 PyTorch 使用 CUDA12.8。为保持训练环境，ONNX 使用 CPUExecutionProvider、2个CPU线程，仅作数值参考。其耗时单独标为 CPU，不参与 GPU 提速结论。

## 精度指标

所有后端保存 conf≥0.001 的免 NMS 检测结果，max_det=300；在固定 conf=0.05/0.25/0.5 下分别报告球心距离指标，不按测试结果调整阈值。主比较使用 conf=0.25。

球心匹配在原图坐标计算，使用一一贪心最近匹配，距离阈值为5和10像素。报告球心精确率/召回率/F1、成功匹配球的误差P50/P95/P99、漏检帧及额外预测帧。误差分位数只统计10像素范围内的成功匹配，未匹配目标数量另报，不能把漏检排除后只宣传小误差。

框 AP 采用 IoU0.5:0.95、按IoU降序贪心唯一匹配及 Ultralytics AP 积分；与球心指标分开。原始框松紧不一致仍会影响 AP，不修改标注或后处理缩放框以提高分数。

同尺寸后端对比报告：预测中心位移P50/P95/P99/最大值、置信度绝对差、未匹配预测数量、以真值球为单位的新增漏检与恢复检出。预测行次序可能因 TopK 改变，因此按空间位置匹配，不能直接逐行相减。batch=2 PyTorch 也与 batch=1 比较。

## 耗时指标

GPU 测试使用预分配显存输入输出，初次为30次预热、200次同步 CUDA event 测量。全部引擎构建完成后，使用 `scripts/rebenchmark_conversion_matrix.py` 顺序复测所有GPU后端：普通执行200次预热、1000次测量；Graph捕获前另行预热，捕获后10次回放预热、1000次测量。复测结果写入 `benchmark`，初测保留在 `initial_benchmark`，并记录 `timing_repeated_after_all_builds`。报告批次平均耗时、P50/P95/P99、图像吞吐及同步循环的实际吞吐。原生 PyTorch/TensorRT 的 CUDA Graph 回放与普通执行分列。纯 GPU 测试不包含相机、CPU图像解码、预处理或H2D/D2H。

精度遍历同时记录离线 PNG 解码、2个CPU预处理线程、归一化、传输、推理、结果回传与过滤的总时间。它是实际实现的离线文件处理速度，尚未做相机/GPU预处理流水线优化，不能当成未来串流吞吐上限。ONNX CPU 基准使用3次预热、30次 CPU 墙钟测量，单独标注。

最终复测对 PyTorch 和 TensorRT 都使用独立非默认 CUDA stream，避免 TensorRT 在默认 stream 上额外隐式同步的已知开销。普通执行与 Graph 使用相同的输入形状和batch，速度比按同执行方式计算；FP32显式禁用TF32，因此不代表所有PyTorch精度配置中的最快速度。

所有 GPU 数值来自 Windows RTX3090。它们能评估后端精度变化与本机相对耗时，但不能保证 Ubuntu RTX2060 Super 达到240张图/秒；目标机需重建引擎，并测真实左右目同步、队列、丢帧、完整流水线及延迟分布。目前没有已确认无球数据，也没有实际三角化评估。

## 执行与恢复

```powershell
& .\.venv\Scripts\python.exe -u scripts/evaluate_conversion_matrix.py
# 中断或处理依赖问题后续跑：
& .\.venv\Scripts\python.exe -u scripts/evaluate_conversion_matrix.py --resume
# 完成12项精度评估后，复测耗时并生成报告和原图局部对照：
& .\.venv\Scripts\python.exe -u scripts/rebenchmark_conversion_matrix.py
& .\.venv\Scripts\python.exe scripts/review_conversion_regressions.py
& .\.venv\Scripts\python.exe scripts/summarize_conversion_matrix.py
& .\.venv\Scripts\python.exe scripts/review_conversion_predictions.py
& .\.venv\Scripts\python.exe scripts/verify_conversion_report.py
```

源模型、图片顺序、已完成后端结果和引擎均保存。续跑跳过已记录完成项，目标目录首次创建时拒绝覆盖已有内容。每次构建前保留至少8GiB空间；不会创建新的图片数据集副本。
