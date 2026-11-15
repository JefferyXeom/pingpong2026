# 本次乒乓球 YOLO26 训练与 TensorRT 转换说明

当前状态及每个评估批次的权重、数据量、参数见[训练与评估总览](training_and_evaluation_overview.zh-CN.md)，文件查找见[报告导航](../reports/README.md)。本文按阶段保留操作教学和历史判断；截至2026-11-09，baseline已完成28轮，正式矩形/原图FP16评估已完成，详见[正式结果](conversion_evaluation_results.zh-CN.md)。

本文对应 `scripts/` 中实际实现以及 `runs/detect/baseline_yolo26n_960/args.yaml`。命令在 Windows 训练机的工作区 PowerShell 中执行。已经完成的整理或正在运行的训练不要重复启动。

## 1. 机器、环境与命令含义

当前是通过 VS Code SSH 访问 Windows RTX 3090 训练机；部署用 Ubuntu 是后续的另一台机器。

```powershell
Set-Location D:\workspace\pingpong_yolo26
hostname
Get-Location
Get-PSDrive -Name C,D
nvidia-smi
py -0p
```

`Set-Location` 设置当前目录。`hostname` 和 `Get-Location` 确认操作位置。`Get-PSDrive` 检查磁盘容量。`nvidia-smi` 查询 NVIDIA 驱动/GPU；它显示的 CUDA Version 是驱动支持的 CUDA 上限，不能证明安装了相同版本 CUDA Toolkit。`py -0p` 列出 Python 解释器。实际框架 CUDA 版本看 `torch.version.cuda`。

独立环境创建入口：

```powershell
py -3.12 -u scripts/bootstrap_environment.py
```

内部依次进行以下操作（这里将环境创建展开成等效命令，实际脚本使用 `venv.EnvBuilder`）：

```powershell
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --no-cache-dir --upgrade pip
& .\.venv\Scripts\python.exe -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128
& .\.venv\Scripts\python.exe -m pip install --no-cache-dir 'ultralytics>=8.4.0' onnx onnxslim onnxruntime-gpu
& .\.venv\Scripts\python.exe -m pip install --no-cache-dir 'tensorrt-cu12>=10,<11'
& .\.venv\Scripts\python.exe -m pip check
```

`&` 是 PowerShell 调用运算符；`-m pip` 确保给指定解释器安装包；`-u` 禁用 Python 输出缓冲，让日志及时刷新；`--no-cache-dir` 减少持久下载缓存。显式调用 `.venv` 的解释器无需 activate，也不会安装到旧 Conda 环境。

实际安装版本为 Python 3.12.3、PyTorch 2.11.0+cu128、Ultralytics 8.4.207、TensorRT 10.16.1.11、ONNX 1.23.2、ONNX Runtime 1.31.1。安装命令使用版本范围，今后重跑可能解析到新版本；本次确切版本记录在 `reports/environment/windows_rtx3090/environment_freeze.txt`。复现 CUDA 环境还需保留 CUDA wheel 索引、驱动/GPU 和平台信息，单靠 freeze 不能保证跨平台相同。

## 2. 数据是如何准备的

先运行初步盘点：

```powershell
& .\scripts\Inspect-YoloDataset.ps1
py -3.12 scripts/audit_dataset.py
py -3.12 scripts/inspect_pairing.py
```

三个入口分别检查容量与目录、详细图片/XML/类别/环境、未配对 XML 与 ZIP 内文件目录。最后一个脚本不会解压 ZIP。源目录总计约 46.34 GiB，混有重复图片、压缩备份、旧缓存与背景变体；22,793 张图片不是 22,793 张独立可用的标注图片。

实际整理命令：

```powershell
& D:\ProgramData\anaconda3\python.exe -u scripts/prepare_dataset.py
```

这里使用已有 Anaconda base 的 Pillow 解码图片。脚本同样可在已装好 Pillow 的 `.venv` 中运行。`prepare_dataset.py` 的操作顺序为：

1. 按明确的图片目录/XML 目录配对同名文件，不跨混乱目录猜测标签。
2. 只选择约定的五个原始采集批次，排除 `waste_data` 和背景替换图片。
3. 解码每张候选图片，检查 XML 尺寸与真实尺寸一致。
4. 验证类别只能为 `ball`，框坐标有限、宽高大于零且不越界。
5. 按旧项目的 VOC 约定转换成归一化 YOLO 框，不直接沿用旧 TXT 中的绝对路径。
6. 计算图片 SHA256；完全相同的图片只选一次，若同图标签冲突则排除并记录。
7. 核对入选 `img_data` 图片与 `VOCData/images` 对应副本内容一致。
8. 按完整录制批次划分 train/val/test，给输出文件添加批次前缀避免撞名。
9. 检查容量，在目标目录创建进行中标记；复制图片及原 XML，验证目标 SHA256 后才完成单个文件。
10. 写入 YOLO 标签、来源清单 `manifest.jsonl`、`data.yaml` 和汇总，成功后移除进行中标记。

脚本拒绝覆盖已存在的 `pingpong_v1`。它创建的是筛选后独立数据集副本，未完整复制整个 46.34 GiB 源目录；桌面源目录仍保留所有未入选数据。

| 源图片目录 | XML 目录 | 集合 | 有效图片 |
| --- | --- | --- | ---: |
| `data/060101` | `data/060101/text` | train | 1,520 |
| `data/img_data/images` | `data/img_data/Annotations` | train | 2,997 |
| `data/img_data.0527/img_data/images` | 同批次 `labels` | train | 2,101 |
| `data/标记2/images/20240527_3` | `data/标记2/annotations/0527_3` | val | 1,272 |
| `data/标记2/images/20240527_2` | `data/标记2/annotations/0527_2` | test | 830 |

共 8,720 张、10,270 个框，图片约 14.00 GiB。4,185 张图片缺标签；1,295 份 XML 没有找到对应图片，ZIP 目录也没有命中。它们均未作为负样本处理。入选样本的图片解码、尺寸、框校验没有发现异常。

YOLO 标签每行一个球：

```text
0 0.5000000000 0.5000000000 0.0100000000 0.0120000000
```

依次是类别编号、归一化中心 x/y、归一化宽/高。XML 是像素坐标，TXT 是相对于原图宽高的比例；训练加载器之后会随 letterbox、翻转、平移等操作变换标签。

本次沿用旧代码：`xc=((xmin+xmax)/2-1)/W`、`yc=((ymin+ymax)/2-1)/H`、`w=(xmax-xmin)/W`、`h=(ymax-ymin)/H`。中心减一是旧项目的像素坐标约定，不应将它当作所有 VOC 数据通用公式。未来混入新标注工具数据时应核实工具的 0/1 起点和边界约定；乒乓球很小，一像素也会影响 IoU。

输出结构：

```text
datasets/pingpong_v1/
  data.yaml
  dataset_summary.json
  manifest.jsonl
  images/train/*.png
  images/val/*.png
  images/test/*.png
  labels/train/*.txt
  labels/val/*.txt
  labels/test/*.txt
  annotations/voc/*.xml
```

`data.yaml`：

```yaml
train: images/train
val: images/val
test: images/test
names:
  0: ball
```

校验及预览：

```powershell
& D:\ProgramData\anaconda3\python.exe scripts/verify_dataset.py
```

逐项检查文件、标签、清单和集合隔离，并生成 `reports/datasets/pingpong_v1/dataset_preview.jpg`。按批次留出可以减少相邻帧泄漏，但没有证明这些批次来自不同场地，也没有证明所有近似帧都完全隔离；SHA256 只判断文件内容完全相同。

## 3. 两次训练、batch 与 epoch

第一次是流程测试：

```powershell
& .\.venv\Scripts\python.exe -u scripts/train.py --stage smoke
```

均匀抽取 192 张训练图片、64 张验证图片，batch=8，1 epoch，训练加载 24 个 batch。下载并加载 `models/yolo26n.pt` 预训练权重，替换为 1 类检测任务。权重用于导出流程测试，不作为最终模型。首次 smoke 执行时尚未显式指定 `nms=False`；当前脚本已显式指定，重跑的新测试会使用免 NMS 模式。已存在的 run 不允许覆盖。

正式训练通过以下入口启动：

```powershell
& .\.venv\Scripts\python.exe -u scripts/start_baseline.py
```

它后台运行的核心命令是：

```powershell
& .\.venv\Scripts\python.exe -u scripts/train.py --stage baseline
```

两条是不同启动方式，不要同时执行，否则是两份训练。启动脚本保存 PID、进程创建时间和日志，并检查重复运行。正式训练从原始预训练 `yolo26n.pt` 重新开始，不接着那 1 轮 smoke 继续。

正式训练每轮使用完整 6,618 张训练图，1,272 张验证图只评估、不反向传播，830 张 test 不参与每轮验证。转换测试曾对其中少量 test 图片推理，所以这里的 test 只能作为独立评估参考；若后续依它反复调整模型，应另保留最终未使用的录制数据。

batch=16 表示一次前向/反向处理约 16 张，不是整个训练只用 16 张。每个 epoch 有 `ceil(6618/16)=414` 个 batch，最后一个只有 10 张。每轮训练顺序打乱、增强随机，Mosaic 还会从训练集读取其他图片，因此物理图片读取次数不一定恰好等于 6,618。

默认 `nbs=64` 会将梯度累积设为 `64/16=4`：预热后通常约四个 batch 更新一次权重，名义有效 batch 约 64，不能说 414 次加载就等于 414 次优化器更新。前三轮 warmup 会调整学习率、动量和累积次数。

本次主要训练设置：

| 参数 | 本次值 | 用途 |
| --- | --- | --- |
| 模型 | YOLO26n | 先建立小模型基线，利于后续受限服务器部署 |
| imgsz | 960 | 减少小球被缩小后的细节损失 |
| batch | 16 | RTX 3090 显存负载与吞吐折中 |
| epochs | 100 | 训练上限 |
| patience | 25 | 验证 fitness 连续未改善的容忍轮数 |
| device | 0 | 第一张 CUDA GPU |
| workers | 2 | CPU 数据加载进程数，不是 GPU 数 |
| cache | False | 不建立整套图片 RAM/磁盘缓存；小型标签缓存仍会生成 |
| amp | True | 混合精度训练；不等于 INT8，也不等于导出 FP16 引擎 |
| nms | False | 显式启用支持的 YOLO26 免 NMS 推理路径 |
| optimizer | auto | 实际正式训练选中 MuSGD，日志显示基础 lr=0.01、momentum=0.9 |
| seed | 20261108 | 固定随机种子，配合 deterministic 尽量复现 |
| save_period | -1 | 不额外按固定间隔保存每一轮 checkpoint，仍保存 best/last |
| val | True | 每轮结束验证 |

球框最短边中位数为 12 原图像素。宽边从 1280 缩到 640 时，887 个框最短边不足 4 像素；缩到 960 时为 134 个。输入 960 需要更多计算，但更适合作为当前小目标基线。

原图 1280×1024 在方形 960 输入中保持比例缩成约 960×768，再填充上下边，不是拉伸成方形。训练标签随图像变换。

增强包括 Mosaic 概率 0.25、缩放幅度 0.25、水平翻转概率 0.5、默认平移 0.1 和 HSV 色彩变化；不使用垂直翻转、旋转、MixUp 或 Copy-Paste。计划最后 10 轮关闭 Mosaic，是相对于 100 轮上限的最后 10 轮；如果提前早停，不保证经历关闭 Mosaic 的阶段。这些是在线增强，不将海量增强图片写盘。

每一轮是：加载训练数据和标签 → 增强与 letterbox → GPU 前向 → 计算分类/定位损失 → 反向传播与梯度累积 → 更新权重 → 使用验证集计算指标 → 保存 checkpoint 和 CSV → 检查早停。

训练日志的 `box_loss`、`cls_loss`、`l1_loss` 是训练目标不同部分，不同损失数值不能直接横向比较。验证指标 Precision=TP/(TP+FP)，Recall=TP/(TP+FN)。mAP50 用 IoU=0.5 判断框匹配，mAP50–95 对 0.50 到 0.95 多个 IoU 阈值求平均，框很小会对一像素误差敏感。这些指标不是“检测正确的图片占比”。

停止条件：达到 100 epoch；或验证 fitness（本版本检测任务为 mAP50–95）超过最好值连续 25 轮没有发生；或磁盘低于 5.5 GiB 触发脚本异常；或 OOM/系统中断/人工停止等异常。空间异常不是优雅早停，当前轮可能未保存，通常从上一完整轮的 last.pt 恢复。没有配置“精度达到某百分比立即停止”，也没有配置小时数上限。

```powershell
& .\.venv\Scripts\python.exe scripts/status.py
Get-Content logs/baseline.stdout.log -Tail 10 -Wait
Get-Content reports/training/baseline_yolo26n_960/baseline_yolo26n_960_progress.json
```

在 `Get-Content -Wait` 中按 Ctrl+C 只停止查看日志，后台训练继续。若训练进程已经退出且 checkpoint 有效，续训：

```powershell
& .\.venv\Scripts\python.exe -u scripts/train.py --stage baseline --resume
```

`last.pt` 是最近保存的完整轮，用于续训；`best.pt` 是验证 fitness 最好的轮，用于评估/导出。两者可能不同。最终应另外对 test 评估，不能只报告训练损失。

## 4. TensorRT 转换

`.pt` 保存 PyTorch 权重与模型信息，ONNX 表达可交换计算图，TensorRT 根据 GPU/软件环境选择内核、融合算子并构建引擎。导出会改变表示/计算精度，不会重新学习区分眼睛和球。

```text
YOLO26 best.pt
  ├─ FP32 ONNX → ONNX checker / ONNX Runtime 检查
  └─ 导出 ONNX → TensorRT 构建 FP16 engine → 加载引擎真实图片推理
```

实际已运行过传统外部 NMS 和免 NMS 两次转换。免 NMS 的命令为：

```powershell
& .\.venv\Scripts\python.exe -u scripts/export_validate.py --name smoke_yolo26n_960_e2e
```

实际完整基线训练结束后，再运行：

```powershell
& .\.venv\Scripts\python.exe -u scripts/export_validate.py --weights runs/detect/baseline_yolo26n_960/weights/best.pt --name baseline_yolo26n_960
```

已有产物目录不能覆盖，需要使用新的 `--name`。导出脚本先把 checkpoint 复制到产物目录，避免引擎/ONNX 混入训练目录。

核心 Python 接口：

```python
from ultralytics import YOLO

model = YOLO('best.pt')
model.export(format='onnx', imgsz=960, batch=1, dynamic=False,
             opset=17, simplify=True, quantize=32, nms=False, device=0)

model = YOLO('best.pt')
model.export(format='engine', imgsz=960, batch=1, dynamic=False,
             opset=17, simplify=True, quantize=16, nms=False,
             workspace=2, device=0)
```

参数解释：`format` 指产物类型；`imgsz` 为空间输入尺寸；`batch=1` 固定单帧；`dynamic=False` 固定形状；`opset=17` 指 ONNX 算子集版本；`simplify=True` 简化计算图；`quantize=32/16` 分别选择 FP32/FP16；`nms=False` 选免 NMS 检测头；`workspace=2` 给 TensorRT 构建策略的临时工作空间限制约 2 GiB，不等于引擎文件或推理总显存；`device=0` 选择用于构建的 GPU。

这里两份文件名中 `model_fp16.onnx` 是为 FP16 引擎构建生成的中间 ONNX，并不保证其图本身已经存成 FP16；真正精度配置是 TensorRT builder 的 FP16 设置，部分层仍可能使用 FP32。

当前免 NMS 导出输入 `1×3×960×960`，输出 `1×300×6`。每个候选通常是 `[x1,y1,x2,y2,score,class]`。300 是候选上限，不是认为画面有 300 个球。需置信度过滤、移除 letterbox 填充并映射回原图。960 方形预处理为：BGR→RGB，HWC→CHW，连续内存，归一化到 0..1；自行部署时必须复现这些步骤。

导出测试先调用一次预热，再在 8 张真实图片上运行三种后端，记录有限输出、候选框与推理阶段耗时。它不是稳定吞吐基准，耗时未包含完整相机采集、传输、解码和业务处理，也没有做正式置信区间统计。

首轮 smoke 仅训练一轮，候选质量很弱。已保存检测在 IoU≥0.95 且分数差≤0.005 的严格匹配条件下，免 NMS ONNX 与 PyTorch 的参考框匹配比例约 95.9%，FP16 TensorRT 约 73.9%；这不能解释为真实召回率，更不能声称 FP16 已无精度损失。TopK 排序和阈值边缘候选会影响逐行比较，需按类别与空间位置配对；正式模型还需比较 mAP、召回、误检和小球中心误差。

FP16 不需要校准集，INT8 通常需要代表性的已预处理图像用于校准；校准数据应来自训练/专用校准录制，覆盖真实曝光、背景、模糊、小球和无球画面，不用最终测试集调量化参数。INT8 是否值得做还需目标 GPU 上速度与精度测试，不能只看引擎大小。

部署 Ubuntu 时迁移 `.pt`、FP32 `.onnx`、代码和版本/构建参数，在目标机器重建 engine。Windows 上的 `.venv` 不能整体复制为 Linux 环境，当前 engine 不应直接当作跨 GPU/OS 通用产物。双目可左右分别 batch=1，或另构建 batch=2 引擎；当前固定 batch=1 不能直接喂两个画面。

Ultralytics 导出的 engine 带元数据封装，适合通过其加载器使用；自行写原生 TensorRT 反序列化程序需要处理封装，或者从 ONNX 直接构建原生 engine，不能假定所有 `.engine` 的文件头相同。

## 5. 无球与易误检样本

建议补充。当前 8,720 张均有至少一个球框，正样本中未标注的背景也会提供背景监督，但缺少专门针对眼睛、手表和反光物的无球/困难背景覆盖。

| 画面 | 正确标注 |
| --- | --- |
| 无球，有反光眼睛/手表 | 与图片同名的空 TXT，零行 |
| 有真球，也有眼睛/手表 | 标出全部应该检测的真球，眼睛/手表不画框 |
| 很暗或模糊，无法判断是否有球 | 人工复核或暂时排除，不能直接放空标签 |

类别仍然只有 `0: ball`，不增加 `background` 类，也不对整张图画一个背景框。空文件不能写 `0`，因为那不是有效五列检测标签。模型误检输出不能直接当作人工真值。

建议第一轮收集约 1,000–2,000 张有差异的真实无球/困难背景图片，按独立录制片段划分后训练集先加入约 800–1,600 张，其余留作负样本验证/测试。这是初始实验配比，不是强制标准。每秒 1–2 帧可作为抽样起点，但反光是短暂事件，还需额外保留触发误检的帧；避免大量几乎相同的连续帧。

覆盖眼睛/眼镜、手表、白鞋/衣服、球台反光、地面亮点、球拍与手部快速运动、不同光照/曝光、左右目、不同远近和角度。保留全幅原始画面与采集批次；只裁出手表再放大训练会改变它相对球的尺度和上下文。需要局部增强时应单独设计。

可以先对新录像运行已有 baseline 的 best.pt 挖掘误检（下方是后续命令示例，未执行）：

```powershell
& .\.venv\Scripts\yolo.exe predict model=runs/detect/baseline_yolo26n_960/weights/best.pt source=incoming/no_ball/clip01.mp4 imgsz=960 device=0 nms=False conf=0.10 save=True save_txt=True save_conf=True project=runs/hard_negative_review name=clip01
```

低阈值用于候选采集，不是最终部署阈值。人工回看原图，确认检测对象是否为球；若整帧没有真球，作为空标签候选；有真球则只标真球。不同来源片段不要随机逐帧混入 train/val/test。若验证/测试发现误检并把这些图片加入训练，应补充新的独立评估数据。

后续以新数据版本或独立训练清单加入，不在当前后台训练中改动 `pingpong_v1`。同名空标签示例，必须在人工确认后创建：

```text
images/train/negative_watch_clip01_000123.png
labels/train/negative_watch_clip01_000123.txt  # 0 字节
```

评估时除 mAP/召回，还测固定置信度下无球帧误检率、每分钟误检、眼睛/手表专项误检和中心坐标误差；同一固定阈值对比加负样本前后。提高阈值可能降低误检，也可能漏掉暗处小球，所以要联合看召回。部署后可进一步加入双目极线/视差、运动连续性与有效空间范围约束，前提是不把合法球运动排除。

原计划的换背景图片训练仍保留：`scripts/add_background_variant.py` 生成原始+背景变体训练清单，验证/测试保持原始画面。这属于正样本背景扩展，与真实眼睛/手表困难负样本互补。

## 6. Codex 审批说明

实际配置为 `sandbox_mode="workspace-write"`、`approval_policy="on-request"`，Windows sandbox 已为 `elevated`。默认沙箱执行创建进程超时，批准沙箱外相同命令即可运行；因此本轮审批多并非桌面数据每次都需要手工确认。

固定脚本/受限命令前缀的持久授权可以减少重复审批，宜复用 `scripts/status.py` 等明确入口，避免反复把不同检查拼成新的整段 PowerShell。若决定给 Codex 完整机器访问，可在客户端权限界面选择对应模式，或在确认客户端支持和组织策略允许后调整顶层 `sandbox_mode="danger-full-access"`；`approval_policy="on-request"` 可继续保留，`"never"` 用于不请求执行审批。单独设 `approval_policy="never"` 不会解除坏掉的沙箱。

完整访问覆盖训练机，而不只覆盖工作区。配置需要在运行 Codex 的远程主机侧生效，新建会话后核对实际权限；托管策略可能限制或覆盖个人配置。本轮没有修改用户配置。OpenAI 官方页面请求返回 HTTP 403，以上故障定位来自本机对照测试；最新配置/权限细节可在官方页面自行核对：

- https://developers.openai.com/codex/security
- https://developers.openai.com/codex/config-reference

## 7. 新发现的 D 盘历史数据如何整理（2026-11-08）

新增来源为 `D:\yolov5-master\VOCData` 和 `D:\yolov5-master\history_VOCOdata`。本次按以下顺序操作，所有源文件均只读：

1. `scripts/audit_additional_data.py` 盘点目录，解码 24,766 张图片，检查图片尺寸，计算图片 SHA256。为每张图片查找同名 XML/TXT，验证类别、框范围及图片与 XML 尺寸是否一致，比较两种标注是否相符。
2. 标注比较最初使用四舍五入后的列表，遇到了浮点边界问题，例如 `0.378125` 和 `0.37812500000000004` 可能被误判。现改为逐框一一匹配、类别完全相同、坐标绝对差不超过 `1e-7`，不要求框顺序相同。`scripts/refresh_additional_audit.py` 使用已经解析的审计记录重算比较，无需再次读入约 35 GiB 源数据。原先的 2,426 个冲突全部消除，没有发现真实冲突。
3. `scripts/inspect_additional_provenance.py` 核对文件名族、XML 内历史路径和背景原图对应关系。2,997 张 `_white` 图均对应现有训练集原图，标签一致。ZIP 不解压，直接在流中计算每张图片 SHA256，确认其中 700 张图片全部已有相同内容的散文件。
4. `scripts/review_additional_data.py` 生成分文件名族的图片抽样预览，绿色框标出球。抽样框位置可用；这只说明抽查结果，不能替代所有画面的人工标签复核。
5. `scripts/prepare_additional_dataset.py` 将 10,786 对独特有效样本复制到 `datasets/pingpong_additional_v1`。预检在复制后至少留 8 GiB，每次复制保留至少 5.5 GiB。逐张核对目标图片 SHA256、原始标注 SHA256，并将 YOLO 标签写入同名 TXT；生成清单保留原始路径、框、哈希、文件名族及换背景原图线索。
6. `scripts/verify_additional_dataset.py` 检查全部输出清单、文件数量、标签数值、图片去重、背景对应关系和空间余量。图片和原始标注复制时已经逐文件校验哈希，此处不重复读完整副本计算哈希。

对应命令如下；审计、整理已经执行，整理脚本拒绝覆盖已有版本。复查时只需运行最后一行。

```powershell
& .\.venv\Scripts\python.exe -u scripts/audit_additional_data.py
& .\.venv\Scripts\python.exe scripts/refresh_additional_audit.py
& .\.venv\Scripts\python.exe scripts/inspect_additional_provenance.py
& .\.venv\Scripts\python.exe scripts/review_additional_data.py
& .\.venv\Scripts\python.exe -u scripts/prepare_additional_dataset.py
& .\.venv\Scripts\python.exe scripts/verify_additional_dataset.py
```

其中 7,789 张来自 `VOCData/images`，2,997 张为换背景版本；已与当前数据集相同的 2,997 张原图和历史目录内部的 7,089 张重复图没有再次复制。另有 3,894 张缺标注图片被排除，不能直接充当无球样本。之前桌面数据中缺图的 1,295 份 XML，在此次散文件的文件名匹配中仍未找到图片候选。

补充副本暂存 `images/unassigned`、`labels/unassigned`，未加入旧数据集，也没有生成训练配置。文件内容不同只能排除完全重复，无法证明不同时间、左右目或近邻帧互相独立。后续需要按录制来源确定划分，背景版本必须跟随对应原图；这里确认的 2,997 张背景图原图都属于训练集，因此具备后续仅用于训练增强的来源条件。

本次同时记录了原基线中断状态：完整完成 24 轮，日志停在第 25 轮中途，进程已不存在，stderr 为空，尚未确定退出原因。第 24 轮 checkpoint 含优化器状态；最佳验证仍在第 3 轮，mAP50=0.15126，mAP50–95=0.04012，后续轮数显著下降。它不能作为已训练好的部署模型，需要先诊断训练/验证问题。`scripts/inspect_baseline_checkpoint.py` 只读取 CSV、checkpoint 和进程信息，报告保存在 `reports/training/baseline_yolo26n_960/baseline_interruption_review.json`，不会启动训练。

## 8. 恢复训练、完整评估及对低 mAP 的修正（2026-11-08）

用户随后确认此前是手动强制退出。实际执行的恢复命令为：

```powershell
& .\.venv\Scripts\python.exe -u scripts/start_baseline.py --resume
```

启动器恢复已有 `last.pt`，向原日志追加输出，保存新的进程信息和旧启动记录，拒绝重复启动同一轮训练。第 25 轮尚未完成的 batch 不会逐步复原，因此从第 24 轮完整 checkpoint 开始重跑第 25 轮。优化器、学习率计划、EMA 和早停状态从 checkpoint 恢复，数据仍为 `pingpong_v1`。

本轮已在第 28 轮正常结束：以框 mAP50–95 计算的最佳轮为第 3 轮，连续 25 轮没有超过它，达到 `patience=25`。恢复部分 CSV 计时约 533 秒，恢复前累计约 3,296 秒，总计约 63.83 分钟，不含手动暂停和后续额外分析。CSV 中 time 在续训时重新从零计算，不能直接把最后一行当成全部累计时间。每轮通常约 2.2–2.3 分钟，跑满 100 轮约需 3.7–4 小时。

当前数据集不是 smoke 子集：train 6,618 张、8,139 个框，图片 10.88 GiB；val 1,272 张、1,272 个框，1.90 GiB；test 830 张、859 个框，1.23 GiB。共 8,720 张、10,270 个框，图片约 14.00 GiB。原图都是 1280×1024，训练输入为 960，batch=16，每轮约 414 个训练 batch。新增历史备份 10,786 张没有参与这一轮，换背景图和已确认无球图也没有加入。

### 为什么不能只用低 mAP 判断没有找到球

固定抽样检查曾得到：第 24 轮模型在训练抽样的 mAP50 为 97.24%，验证抽样约为 0.053%。这最初提示泛化问题，但不足以判定模型无法定位球。随后查看原图、标注放大图，并计算预测球心与标注球心的距离，发现预测中心通常正确，而预测框约为验证标注框的一半宽、一半高。

例如同心的 9×9 预测框与 17×17 标注框，即使中心误差为零，IoU 也只有 `81/289≈0.28`，无法通过 IoU≥0.5 的框匹配。原有训练/验证标注的松紧程度存在明显差异，需要统一后续标注规范。这里保持原标签不变，没有自动放大预测框来提高分数，也没有把球心距离得分冒充框 mAP。

`reports/evaluation/baseline_yolo26n_960/diagnostics_epoch24/baseline_prediction_crops.jpg` 中绿色为原标签，紫色为预测，右侧显示球附近放大区域。`reports/evaluation/baseline_yolo26n_960/diagnostics_epoch24/train_val_domain_review.json` 同时记录了录制批次的球框尺寸与抽样亮度。该证据说明框尺寸差异是低框指标的重要原因；仍不能排除其他泛化问题。

### 完整评估与候选选择

执行以下命令完成完整验证集比较，按预定 conf=0.25、中心距离不超过 5 原图像素的 F1 选择权重，再评估测试集。两个报告均已存在，脚本拒绝覆盖，复现实验须换新的 `--name`。

```powershell
& .\.venv\Scripts\python.exe -u scripts/evaluate_centers.py --weights runs/detect/baseline_yolo26n_960/weights/best.pt runs/detect/baseline_yolo26n_960/weights/last.pt --split val --name baseline_full_val_centers
& .\.venv\Scripts\python.exe -u scripts/evaluate_centers.py --weights runs/detect/baseline_yolo26n_960/weights/last.pt --split test --name baseline_full_test_centers
```

完整验证选择了第 28 轮的 `last.pt`，候选独立保存为 `models/pingpong_yolo26n_center_v1/weights.pt`。第 3 轮 `best.pt` 仍保留，它是以框 mAP 选出的模型。测试集没有参与本次权重和 conf=0.25 的选择。

| 完整集合，conf=0.25 | 球心精确率 | 球心召回率 | 球心 F1 | 匹配误差中位数 | 匹配误差 P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| val，1,272 张 | 96.87% | 97.33% | 97.10% | 1.25 px | 2.76 px |
| test，830 张 | 93.96% | 95.93% | 94.93% | 1.21 px | 2.46 px |

精确率为中心落在 5 像素范围内的匹配预测数/全部预测数；召回率为匹配球数/全部标注球数，均采用一一匹配。误差分位数针对 10 像素范围内匹配成功的球，不含漏检；没有把漏检当成零误差。距离基于原始 1280×1024 坐标和原始标签中心，不能直接解释为真实三维坐标精度。

评估使用 imgsz=960、batch=16、rect=True 的矩形 letterbox、nms=False。固定 960×960 TensorRT 输入的方形预处理、FP16 或 INT8 精度需要后续重新验证，不能直接沿用此表作为引擎精度结论。当前也没有已确认的无球测试片段，因此尚不能量化眼睛/手表专项误检或无球视频每分钟误检率。

训练正常结束后，Ultralytics 对 best/last 剥离优化器；它们可用于推理或新一轮微调的初始化，当前完成状态不能再用 `--resume` 恢复完整优化器。如果新实验仍主要关注球心，应先统一标注规范和模型选择指标，再设计训练及停止条件。

分析脚本 `scripts/analyze_training.py` 输出 `reports/training/baseline_yolo26n_960/baseline_training_analysis.json` 与 `reports/training/baseline_yolo26n_960/baseline_learning_curves.png`。模型候选的完整配置、来源哈希、验证/测试结果和使用限制见 `models/pingpong_yolo26n_center_v1/model_card.json` 与同目录 `README.zh-CN.md`。

