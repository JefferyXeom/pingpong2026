# 负样本、输入尺寸与 TensorRT 精度

本文解释原理及数据准备方法；已经执行的原图/转换测试、使用模型和完整数据范围见[训练与评估总览](training_and_evaluation_overview.zh-CN.md)及[正式结果](conversion_evaluation_results.zh-CN.md)。

2026-11-09 的约束是定位精度优先，尽量不降低空间分辨率，部署目标仍为 RTX 2060 Super 上 120 对双目/秒（240 张图/秒）。当前 960 长边本身已经把 1280×1024 原图缩小；768×960 矩形输入相对 960×960 只是减少填充，不进一步缩小球。必要时单独比较原图大小 1024×1280。

## 是否需要负样本

YOLO 可以只用有球图片训练，因为这些图片中未匹配目标的候选也接受背景监督。但对于已知的眼睛/手表反光误检，本项目应补充真实无球与困难背景样本。现有训练数据没有经过确认的无球样本，之前排除的缺标签图片不能直接作为负样本。

整帧经过人工确认没有目标球时，保存原图并创建同名、零字节 TXT；有真球和反光物时，标注所有目标球，不标反光物。保持一个类别 `0: ball`，不增加背景类别，不给手表画 ball 框。不能把模糊但真实存在的球误留为空标签，否则会抑制真实球检测。

第一轮可收集约 1,000–2,000 张有差异的全幅画面，优先包含真实部署视角、眼睛/眼镜、手表、衣服/鞋、球拍、灯光及地面反射，不用大量重复空场充数。当前 6,618 张训练图，可先加入约 800–1,600 张已确认负样本，再通过固定阈值下的球心召回及无球误检决定配比；该范围是实验起点，不是 YOLO 的强制比例。按录制片段分组，同一时刻的左右目及相邻帧不跨集合；另留独立负样本验证/测试片段。误检输出只用于找候选，不能直接作为人工标签。

空标签示例：

```text
images/train/no_ball_watch_clip01_000123.png
labels/train/no_ball_watch_clip01_000123.txt  # 真正空文件，不能写“0”
```

负样本不改变模型结构，通常不会增加单帧推理计算量；它增加训练数据和训练时间。换背景有球图仍是正样本，不等同于真实无球负样本。比较加负样本前后时固定预处理、阈值和录制划分，同时看有球召回、球心误差、无球帧误检率、每分钟误检次数。

## YOLO26 的尺寸支持

PyTorch YOLO26 权重支持不同空间输入，不被训练 imgsz 永久锁定。当前模型使用最大 stride 32，常用输入高宽取 32 的倍数，由 Ultralytics 缩放并 letterbox 填充。一个普通张量 batch 内形状一致。multi_scale 是训练增强功能，当前训练 multi_scale=0.0；支持多尺寸推理不要求当时启用多尺度训练。

训练 imgsz=960 与推理 imgsz=1280 可以不同，通常无需为改变尺寸重新训练；但目标像素尺度及上下文会变化，不能假设精度相同或更高，应独立验证，必要时微调。原图、网络张量和输出坐标是三个概念：原图 1280×1024，960 方形训练通常缩放为 960×768 并补边到 960×960；矩形推理可只使用 960×768。Ultralytics Results.boxes 的坐标恢复到原图，原生 TensorRT 实现要自己移除补边并反缩放；有 ROI 时再加回裁剪偏移。双目投影/三角化必须与所用坐标系及内参一致。

推理示例，尺寸元组的顺序是高、宽：

```python
from ultralytics import YOLO
model = YOLO('models/pingpong_yolo26n_center_v1/weights.pt')
results = model.predict('your_image.png', imgsz=(768, 960),
                        rect=False, conf=0.25, nms=False, device=0)
# 要评估原图大小，可以另用 imgsz=(1024, 1280)。
```

以上是 API 示例，未执行新的图片推理。

## ONNX 和 TensorRT

固定 ONNX/engine 的输入张量尺寸由导出/构建决定；例如固定 2×3×768×960 只能接受这个 batch/空间尺寸。它可以先接收不同大小的相机原图，再由程序预处理为固定尺寸；这不代表引擎自身动态。不能仅在 predict 中改 imgsz 就让固定引擎接受 1280。

动态 ONNX 和 TensorRT 可以改变输入形状，但 TensorRT 必须在构建时设置有效 optimization profile 的 min/opt/max，运行时选中合适 profile 并设置输入形状，调整绑定/输出缓冲。batch 也要在范围内。动态不表示任意大小，维度通常仍要求 stride 对齐及图中相关约束一致。当前安装版本 Ultralytics 的 dynamic=True 导出会标记 batch、高、宽，并自动创建 profile；其默认范围不等于本项目希望的紧凑范围，正式部署应检查或自行构建 profile。

下面是独立目录内导出的示例，尚未执行；当前脚本 export_validate.py 仍固定方形、batch=1、dynamic=False，需要扩展后才能用于矩形/动态完整验证。

```python
from pathlib import Path
import shutil
from ultralytics import YOLO

out = Path('artifacts/example_rect_export')
out.mkdir(parents=True, exist_ok=False)
weights = out / 'weights.pt'
shutil.copy2('models/pingpong_yolo26n_center_v1/weights.pt', weights)

# 固定左右目 batch=2，FP16，宽960高768。
YOLO(str(weights)).export(format='engine', imgsz=(768, 960), batch=2,
                        dynamic=False, quantize=16, nms=False,
                        workspace=2, device=0)

# 动态版应使用另一个独立产物目录：
# YOLO(str(other_weights)).export(format='engine', imgsz=(768, 960),
#     batch=2, dynamic=True, quantize=16, nms=False, workspace=2, device=0)
```

动态引擎未必必然比固定引擎慢；在固定相机尺寸及严格吞吐要求下，先比较固定引擎或少量明确 profile 更容易优化。实际性能取决于形状、profile、kernel/tactic 和切换成本。在 3090/Windows 上构建和测量不替代 Ubuntu/2060 Super 的重建/实测。

## 转换精度损失与耗时收益怎么测

先将 FP32 参考、ONNX FP32、TensorRT FP32、TensorRT FP16 固定在相同有效输入、预处理、阈值和数据上比较。FP32 转换也可能因算子融合、浮点计算顺序和 TopK 等出现差异；FP16 有更少数值精度，实际定位变化必须测量，不能保证无损。FP16 不需要 INT8 校准；若做 INT8，需使用训练或专用校准录制，覆盖暗光、反光、小球、模糊、有球/无球，单独验证量化后的误差和召回，最终测试集不用于调量化参数。

除了中心精确率/召回及匹配误差中位数/P95/P99，还记录同一张图不同后端预测中心的位移、未匹配预测、置信度变化和无球误检。不能只在两个后端都成功匹配的球上报告平均误差而隐去新增漏检。双目后端中心误差需分别看左右目及视差变化；Z=fB/d 时，小视差下相同的像素误差可能带来更大的深度误差，原图约1像素的2D误差不等于已有3D精度保证。

速度在目标机同一输入/batch 下充分预热，使用 CUDA event 或同步后计时，测大量连续调用的均值/P50/P95/P99及吞吐，分别报告纯 GPU 与完整流程。加速比=原耗时/新耗时，耗时下降比例=(原耗时-新耗时)/原耗时；例如10ms到6ms才是1.67倍和40%下降，该数字只是算术示例。2026-11-09已完成正式模型在RTX3090上的12项全量转换/尺寸精度及速度评估，见[正式结果](conversion_evaluation_results.zh-CN.md)；2060 Super收益仍待实测。早期smoke的8图检查与正式评估分批归档。

优化顺序为：保持有效图像尺寸、减少补边、选择 FP16、比较同时间左右目 batch=2、预分配与数据复用、GPU 预处理/异步流水线，再决定是否开展 INT8 或其他方案。若精度与240张图/秒无法同时达到，应报告两者实测结果，而非默认降低分辨率。
