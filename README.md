# 乒乓球 YOLO26 训练与转换工作区

跨机器接手先读[项目交接状态](docs/status.md)和[Git及大文件规则](docs/repository_workflow.zh-CN.md)。Git只保存代码、配置、模型卡和必要评估摘要；数据、权重、TensorRT engine、录像及大型可视化另行传输，忽略规则不会删除本地文件。部署端先完成新相机同步/标定，再进行TRT验证和主球关联。

本工作区位于 SSH 连接的 Windows 训练机。原始数据为 `C:\Users\Administrator\Desktop\yolov5data`，整理脚本只读访问源目录；训练、标注转换、权重与导出产物位于当前工作区。

当前状态、模型身份和每个测试批次见[训练与评估总览](docs/training_and_evaluation_overview.zh-CN.md)。查找报告请从[reports分类导航](reports/README.md)进入；报告已按环境、数据版本、训练运行、评估批次分别归档，每个目录均附用途说明。

最新完成：**Baidu增量原图n/s批次**已于北京时间2026-11-14 06:47:47结束，见[详细最终报告](docs/experiments/native_merged_v2_baiduout_20261113_results.zh-CN.md)。merged_v2每轮28,401图/36,093球，n训练32轮选第12轮，s训练42轮选第22轮；本轮24项转换组合，连同历史共[72组合](reports/experiments/native_merged_v2_baiduout_20261113/all_models_combinations.csv)。测试召回略升、额外预测增加，F1未全面超过旧B/C；保留旧候选与新模型对照，未验证目标2060 Super，INT8未执行。下面的旧批次细节继续保留。

部署重点比较**球心precision、recall、F1及定位误差中位数/P95/P99**；各PT/ORT/TensorRT FP32/FP16版本见[球心P/R专表](docs/experiments/native_center_precision_recall.zh-CN.md)，定位误差与转换位移说明见[完整结果](docs/experiments/native_resolution_suite_v1_results.zh-CN.md)。框IoU/mAP暂不作为部署选择重点，INT8尚未评估。

三组原图训练（n/v1、n/v1+additional、s/v1+additional）及转换评估已于2026-11-10完成，含旧基线共48项组合。先读[最终结果分析](docs/experiments/native_resolution_suite_v1_results.zh-CN.md)，配置和完整表见[全部组合](docs/experiments/native_resolution_suite_v1.zh-CN.md)。实际训练/验证输入为1280×1024，按完整验证集球心F1选模与早停；配置为`configs/native_resolution_suite_v1.json`。目标机优先验证合并数据YOLO26n＋原图＋TRT FP16，保留YOLO26s比较漏检和计算预算，2060 Super尚未实测。项目实验记录约定见[项目skill](skills/pingpong-experiments/SKILL.md)。

## 数据集

整理后的独立副本为 `datasets/pingpong_v1`，图片约 14.00 GiB。共有 8,720 张 1280×1024 图片、10,270 个 `ball` 框，类别编号为 `0`。

| 集合 | 图片数 | 采集批次 |
| --- | ---: | --- |
| train | 6,618 | `060101`、`img_data`、`img_data.0527` |
| val | 1,272 | `20240527_3` |
| test | 830 | `20240527_2` |

按完整录制批次划分，左右目画面留在同一集合。划分依据是原始目录中的批次信息；这些目录不能证明训练、验证和测试来自不同场地。完全相同的图片内容没有跨集合重复。

`data.yaml` 省略绝对根目录，由 YAML 所在目录确定数据根路径，方便后续迁移。`manifest.jsonl` 记录每张图片的来源、集合、SHA256、输出路径、原始框及未来换背景图片路径。原始 XML 副本保存在 `annotations/voc`。

本次转换沿用旧项目的 VOC 约定：归一化中心为 `((min+max)/2-1)/size`，宽高为 `(max-min)/size`。所有实际配对的图片都已解码检查，图片尺寸与 XML 一致，标签均有效，复制时逐文件校验 SHA256。

本次没有复制 ZIP、旧缓存、重复的 `VOCData` 图片副本、`waste_data` 或 `processed_images`。4,185 张原始图片缺标注，未作为负样本使用；另有 1,295 份 `060101` XML 在本次扫描的原始图片目录和两份 ZIP 目录中没有找到对应图片。详细路径见 `reports/datasets/pingpong_v1/dataset_preparation.json`、`reports/datasets/pingpong_v1/pairing_review.json` 和 `reports/datasets/pingpong_v1/archive_inventory.json`。整理后的数据没有已确认的无球负样本，后续需要补充并验证误检表现。

## 新增历史数据备份（2026-11-08）

新增源目录为 `D:\yolov5-master\VOCData` 和 `D:\yolov5-master\history_VOCOdata`。共扫描到 24,766 张散文件图片，其中 20,872 张具有有效标注、3,894 张缺少同名标注。XML/TXT 比较采用归一化坐标绝对容差 `1e-7`，避免浮点小数舍入造成假冲突；本次没有发现真实标注冲突，也没有确认的无球样本。

去除与 `pingpong_v1` 以及历史目录内部的图片内容重复后，补充副本为 `datasets/pingpong_additional_v1`，共 10,786 张图片，图片占用约 13.55 GiB：

| 来源 | 去重后补充图片 | 说明 |
| --- | ---: | --- |
| `VOCData/images` | 7,789 | 700 张数字文件名、379 张短 `img` 文件名、6,710 张带后缀索引文件名；采集关系待核实 |
| `history_VOCOdata/VOCData_1/images` | 2,997 | `_white` 换背景版本，均匹配现有训练集原图，标签一致 |

`VOCDatahk` 的 7,089 张图片均已有相同内容的 `VOCData` 文件；另外 2,997 张原图与现有训练集相同。历史 ZIP 的 700 张图片通过逐项 SHA256 验证，均可在散文件中找到相同内容，没有再次解压或复制。

补充副本保留原始 XML/TXT、转换后的 YOLO 标签、来源路径、内容哈希和换背景原图对应关系。所有图片和原始标注复制时校验 SHA256，转换标签与清单另行核对。图片和标签仍保存在 `images/unassigned`、`labels/unassigned`，原图合并实验通过 `configs/datasets/native_resolution_suite_v1/native_v1_additional.yaml` 和派生列表引用它们；目录名unassigned不代表没有参与训练。已知背景变体继承原图train划分，原有 `pingpong_v1` val/test保持原样。

审计详情见 `reports/datasets/pingpong_additional_v1/additional_data_review.json`，画面抽检见 `reports/datasets/pingpong_additional_v1/additional_data_preview.jpg`，来源核对见 `reports/datasets/pingpong_additional_v1/additional_data_provenance.json`，副本验证见 `reports/datasets/pingpong_additional_v1/additional_dataset_verification.json`。整理脚本拒绝覆盖已经存在的补充副本；复查命令为：

```powershell
& .\.venv\Scripts\python.exe scripts/verify_additional_dataset.py
```

## 环境与运行

独立环境为 `.venv`，使用 Python 3.12、CUDA 版 PyTorch、Ultralytics、ONNX 和 TensorRT。实际安装版本记录在 `reports/environment/windows_rtx3090/environment_freeze.txt`。旧 Conda 环境的包未被安装脚本更改。

在工作区 PowerShell 中执行：

```powershell
# 盘点、整理与校验；prepare_dataset 拒绝覆盖已存在的数据集版本。
py -3.12 scripts/audit_dataset.py
& D:\ProgramData\anaconda3\python.exe scripts/prepare_dataset.py
& D:\ProgramData\anaconda3\python.exe scripts/verify_dataset.py

# 小规模训练：192 张 train、64 张 val，1 epoch。
& .\.venv\Scripts\python.exe scripts/train.py --stage smoke

# 完整基线：YOLO26n，960 输入，batch 16，最多 100 epochs，patience 25。
& .\.venv\Scripts\python.exe scripts/train.py --stage baseline

# 或在持久日志中运行完整基线，进程信息写入 reports/training/baseline_yolo26n_960/baseline_process.json。
& .\.venv\Scripts\python.exe scripts/start_baseline.py

# 已有基线中断后续训；完成的训练无需再次 resume。
& .\.venv\Scripts\python.exe scripts/train.py --stage baseline --resume
```

训练使用预训练 `models/yolo26n.pt`、固定 seed `20261108`、AMP、免 NMS 推理头、2 个数据加载 worker，关闭大图片缓存，仅保留 best/last 权重。输入选择 960 是因为原始球框最短边中位数为 12 像素：缩至 640 时有 887 个框最短边不足 4 像素，960 时为 134 个。

旧960基线的`runs/detect/<run>/weights/best.pt`按验证框指标选择，新原图三组的best.pt按完整验证集球心F1选择，两者不能混用。训练尚未正常结束时，`last.pt`含断点状态；正常结束后Ultralytics会剥离优化器，此时best/last可用于推理或新训练初始化，不能直接按原优化器状态续训。`results.csv`保存在同一运行目录。旧基线后台日志位于`logs/baseline.stdout.log`、`logs/baseline.stderr.log`；新原图阶段日志在`logs/native*`，进度和结果按实验ID放在`reports/training/<run_id>/`。

2026-11-08 用户确认前次中断为手动强制退出。已执行 `scripts/start_baseline.py --resume`，从第 24 轮恢复，第 28 轮触发 patience=25 的早停并正常结束。实际累计训练约 63.83 分钟（不含暂停），恢复部分约 8.89 分钟；正常结束报告见 `reports/training/baseline_yolo26n_960/baseline_yolo26n_960_result.json`。

需要修正仅按低 mAP 判定球检测失败的早期结论：完整球心评估显示第 28 轮模型仍能较准确定位球。验证框普遍比预测框宽松，预测宽/高约为标注的 0.57/0.52 倍，框 IoU 很低。原始标注保持原样；球心指标作为补充，未改称标准 mAP。按照完整验证集在固定 conf=0.25、原图距离 5 像素内的球心 F1，选择第 28 轮 `last.pt`，之后才评估测试集。

已整理候选为 `models/pingpong_yolo26n_center_v1/weights.pt`，同目录附模型说明与 SHA256。完整验证集球心精确率/召回率为 96.87%/97.33%，完整测试集为 93.96%/95.93%；匹配误差中位数分别为 1.25/1.21 原图像素。模型说明区分球心与框指标、匹配误差统计范围及后续TensorRT评估；无球与实际双目仍未覆盖。初始球心报告见 `reports/evaluation/baseline_yolo26n_960/center_selection_v1/baseline_full_val_centers.json`、`reports/evaluation/baseline_yolo26n_960/center_selection_v1/baseline_full_test_centers.json` 和 `reports/evaluation/baseline_yolo26n_960/center_selection_v1/trained_model_selection.json`。

```powershell
Get-Content logs/baseline.stderr.log -Tail 10 -Wait
Get-Content reports/training/baseline_yolo26n_960/baseline_yolo26n_960_progress.json
& .\.venv\Scripts\python.exe scripts/status.py
```

## ONNX 与 TensorRT

`scripts/export_validate.py` 导出 FP32 ONNX、FP16 TensorRT，并检查 ONNX、原始 FP32 输出与真实图片推理。默认启用免 NMS 头；当前安装的 Ultralytics 用 `nms=False` 显式选择此模式，用 `quantize=16` 指定 FP16。`--external-nms` 可用于对比传统后处理方式。

```powershell
# 小规模训练的流程验证权重
& .\.venv\Scripts\python.exe scripts/export_validate.py --name smoke_yolo26n_960_e2e

# 完整基线训练完成后，用实际 best 权重再次转换、评估。
& .\.venv\Scripts\python.exe scripts/export_validate.py --weights runs/detect/baseline_yolo26n_960/weights/best.pt --name baseline_yolo26n_960
```

转换产物位于 `artifacts/<name>`，报告为 `conversion_report.json`。固定输入为 `1×3×960×960`，使用相同的方形 letterbox 预处理进行后端验证。小规模训练产物仅验证流程；正式精度以及 FP16/未来 INT8 的精度变化，应使用完整训练的权重和保留的数据评估。

首轮已生成传统外部 NMS 和免 NMS 两种测试产物，免 NMS 目录为 `artifacts/smoke_yolo26n_960_e2e`，输出形状为 `1×300×6`。PyTorch、ONNX CPU 和 TensorRT GPU 均已在 8 张真实图片上运行。`reports/evaluation/smoke_conversion_v1/conversion_comparison.json` 对保存的检测结果按空间位置匹配，避免把 TopK 输出行次序变化误当作逐元素坐标误差。当前测试权重只训练了 1 轮，存在大量低置信度候选框及后端差异；这些检查没有建立正式模型的精度一致性。

```powershell
& .\.venv\Scripts\python.exe scripts/compare_conversion.py
```

Windows RTX 3090 上构建的引擎用于本机验证。后续迁移保留 `.pt`、FP32 `.onnx`、环境版本和构建参数，在目标 Ubuntu GPU/软件环境重新构建并验证引擎。串流、双目同步和相机接入留到部署阶段。

## 正式模型的原图与转换评估（2026-11-09）

针对球心候选 `models/pingpong_yolo26n_center_v1/weights.pt`，固定完整验证集1,272张、测试集830张，在960×768及不缩放的原图1280×1024上分别比较PyTorch FP32、ONNX CPU FP32、TensorRT FP32/FP16以及batch1/2，共12项。没有重新训练、改变标签或按测试结果调阈值。原图档使用现有960训练权重，不等于原图训练模型。

[评估结果](docs/conversion_evaluation_results.zh-CN.md)列出球心精确率、召回率、F1、原图像素误差、标准框mAP、FP16新增漏检及本机耗时；[评估协议](docs/conversion_evaluation_protocol.zh-CN.md)说明预处理、匹配口径、TensorRT构建参数、测速范围及复现指令。完整报告为 `reports/evaluation/center_conversion_matrix_v1/center_conversion_matrix_v1.json`，筛选表为 `reports/evaluation/center_conversion_matrix_v1/center_conversion_matrix_summary.csv`。

原图对照见[局部放大图](reports/evaluation/center_conversion_matrix_v1/conversion_original_image_crops.jpg)，保持1280×1024的8张完整标注图位于 `reports/evaluation/center_conversion_matrix_v1/conversion_original_examples`，对应源图见该目录的 `index.json`。这些画面按固定位置均匀抽取；评估本身使用全量保留集。

所有速度来自RTX3090，不能据此认定2060 Super满足每秒120对双目画面。当前没有已确认的无球评估集，眼睛、手表等专项误检仍需补充数据验证。

## 后续换背景训练

首轮仅使用原始图片。用户确认后续仍需要换背景图片训练，已准备显式启用脚本：

```powershell
& .\.venv\Scripts\python.exe scripts/add_background_variant.py
& .\.venv\Scripts\python.exe scripts/train.py --stage baseline --data datasets/pingpong_v1/data_background.yaml --name baseline_background_yolo26n_960
```

脚本只为属于训练集且有有效标注的原图复制换背景版本，生成独立的 `data_background.yaml` 与清单。验证和测试继续使用原始画面，避免原图与其背景变体跨集合。现有背景替换脚本只改变颜色，沿用原图标签；后续训练前仍需检查替换质量。

## 空间管理

保留 D 盘至少 5 GiB 空闲。整理时逐次检查余量并验证复制；训练在每批次检查空间，低于 5.5 GiB 时停止以保留余量。环境安装禁用 pip 持久下载缓存，训练禁用图片缓存，不批量保留每轮 checkpoint。盘点、缺失配对、数据验证和环境报告集中在 `reports`。
