# 乒乓球 YOLO26n：球心定位候选 v1

训练已正常完成 28 轮，因连续 25 轮框 mAP 未提升而早停。数据为 pingpong_v1：6,618 张训练、1,272 张验证、830 张测试。新增历史备份和换背景图片未加入本轮。

原始图片宽×高为 **1280×1024**；本轮训练网络输入宽×高为 **960×960**（imgsz=960、rect=False、multi_scale=0），batch=16，启用AMP混合精度训练。后续正式评估的960×768与原图1280×1024是推理输入尺寸，使用同一份已训练权重。

候选由完整验证集在固定 conf=0.25、原图球心距离不超过 5 像素的 F1 选择，来自第 28 轮 `last.pt`。测试集没有参与权重或阈值选择。它与 Ultralytics 按框 mAP 选择的第 3 轮 best.pt 的用途评价不同，两份原始训练权重均保留。

| 完整数据集 | 球心精确率 | 球心召回率 | 球心 F1 | 匹配误差中位数 | 匹配误差 P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 验证 1,272 张 | 96.87% | 97.33% | 97.10% | 1.25 px | 2.76 px |
| 测试 830 张 | 93.96% | 95.93% | 94.93% | 1.21 px | 2.46 px |

球心精确率/召回率使用 5 像素距离阈值；误差分位数统计 10 像素以内成功匹配的球，不包括未匹配目标。坐标都在原始 1280×1024 图片上计算。它们是补充指标，不能替代或改称标准框 mAP。

验证集中预测框宽、高的中位数约为对应标注框的 0.57、0.52 倍。即使球心准确，框尺寸不同仍会让 IoU 很低。没有自动放大预测框或修改验证标签来提高分数，需要统一后续标注规范。

推理示例，在工作区根目录的 Python 环境执行：

```python
from ultralytics import YOLO
model = YOLO('models/pingpong_yolo26n_center_v1/weights.pt')
results = model.predict('your_image.png', imgsz=960, conf=0.25,
                        nms=False, rect=True, device=0)
for result in results:
    centers_xy = result.boxes.xywh[:, :2]  # 原图坐标；可能有多个候选
```

2026-11-09已使用这份同哈希第28轮权重完成正式转换评估：完整val1,272张＋test830张，960×768与原图1280×1024，PyTorch/ONNX/TensorRT FP32/FP16共12项。960档FP16 batch2没有新增5像素漏检；原图档FP16 batch2在val新增1例置信度跨阈值漏检。详见[正式结果](../../docs/conversion_evaluation_results.zh-CN.md)、[训练与评估总览](../../docs/training_and_evaluation_overview.zh-CN.md)。正式矩阵使用显式矩形预处理和原生TensorRT加载器。

当前仍是单帧检测候选，不包含无球专项评估、实际左右目配对、三维重建或串流。固定960×960方形引擎、INT8及目标Ubuntu/2060 Super均未完成正式全量验证；当前速度来自Windows RTX3090。

训练正常结束后，Ultralytics 已剥离优化器；这里的权重可用于推理和新一轮微调的初始化，不能直接作为含完整优化器状态的断点续训。
