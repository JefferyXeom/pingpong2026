# 项目交接状态

状态记录日期：2026-11-19（Asia/Shanghai）。本页为两台机器的共同入口，详细历史见实验报告。

部署包`deployment_handoff_20261118_v1`已于11月19日经SSH/SFTP传输至用户指定的Ubuntu接收目录`/home/pingpong/pingpong-transfer/incoming/deployment_handoff_20261118_v1`，状态`complete_verified`。两个ZIP的远端SHA256一致，部署端逐成员校验通过：模型包4287文件、图片包2104文件；传输回执也已上传核验。内容为4组候选PT＋原图ONNX、24份参考预测、完整val/test2102原图与标注、128图快速清单。模型包约158MiB，图片包约3.13GiB，均在Git忽略的transfers目录，未提交大文件。[部署接手步骤](deployment_transfer_20261119.zh-CN.md)、[部署包说明](deployment_package_20261118.zh-CN.md)、[带SHA256的清单](../configs/deployment_handoff_20261118_v1.json)。本次完成文件交接；解压、本机TRT构建、目标机精度/性能及相机验证由部署端继续。

## 机器职责与当前目标

- 训练端：Windows / RTX3090，数据整理、训练、补训、离线评估与候选模型产出。
- 部署端：Ubuntu / 用户报告RTX2060 Super，实际环境待核实；负责TensorRT本机构建、硬件同步、标定、相机串流、左右关联与三维计算。
- 目标：120对双目画面/秒，即240图/秒；推理优先宽1280×高1024，保持球心定位质量。吞吐目标不是已达到的结果，端到端延迟上限尚未指定。
- 短期只跟踪当前对打球，忽略其他球；长期再扩展所有球的三维身份。

## 已完成的模型与评估

此前原图三组训练、含旧基线48组合已完成，新增Baidu数据n/s训练及24组合也已完成，共72组合。Baidu有效新图10997张/13912球，合并v2 train28401张/36093球；固定val1272张/1272球、test830张/859球。

| 候选 | 选中epoch | 原图TRT FP16 b2 test P / R / F1 | 作用 |
| --- | --- | --- | --- |
| native_v1_additional_yolo26n_1280 | 24 | 94.87% / 96.86% / 95.85% | 保留速度优先部署候选 |
| native_v1_additional_yolo26s_1280 | 30 | 95.45% / 97.67% / 96.55% | 旧s对照 |
| native_merged_v2_baiduout_yolo26n_1280 | 12 | 93.30% / 97.21% / 95.21% | 新n对照 |
| native_merged_v2_baiduout_yolo26s_1280 | 22 | 94.49% / 97.79% / 96.11% | 新s定位/召回对照 |

权重路径均为`runs/detect/<候选ID>/weights/best.pt`；SHA256见[实验登记](../reports/experiment_registry.json)。不能把最后一轮last替代所选best。新数据召回略升、额外预测增加，没有全面超过旧模型，暂不自动替换。用户目视认为Baidu质量优于additional，此为人工抽样观察，未完成全量质量打分，不据此自动删数据或开启训练。

- [最新详细结果](experiments/native_merged_v2_baiduout_20261113_results.zh-CN.md)
- [72组合总表](../reports/experiments/native_merged_v2_baiduout_20261113/all_models_combinations.csv)
- [数据标注可视化说明](../reports/datasets/pingpong_baidu_out_v1/label_quality_review.zh-CN.md)：HTML/图片是本地派生产物，Git不携带。

## 部署端待办（未完成）

1. 接收代码、模型清单及单独传输的PT/ONNX/核验图片，先验证SHA256。历史绝对路径仅供追溯，重新配置本机路径。
2. 核实GPU/驱动/软件、相机实际采集模式；接入用户已有硬件同步和标定方案，新设备目前未完成同步与标定。
3. 在支持Turing SM7.5的软件栈上重建TRT FP16；固定图片验证P/R/F1、中位/P95/P99定位误差、相对FP32位移和新增漏检。INT8尚未执行。
4. 同步录像离线实现候选检测→球心去畸变/校正→极线筛选→共享主球轨迹约束左右配对→三角化。歧义时不强行输出新实测深度；单目丢失/预测需标明状态。
5. 接入相机并测目标卡完整流水线吞吐、延迟、积压、丢帧及错误配对/跟踪中断，不以3090纯GPU结果代替验收。

## 模型接口与约束

标准输入`[B,3,1024,1280]`，BGR→RGB、CHW、FP32/255，无补边；现有原图输入不缩放。实际新相机分辨率待核实，标定参数与图像变换需一致。当前导出免NMS，输出`[B,300,6]`为xyxy/conf/class，class0=ball；保留多个候选再关联。现有FP16 engine接口仍为FP32输入/输出，固定engine不能任意改batch/高宽；Windows/3090 engine不可当目标机已验证产物。

现有数据无已确认无球负样本；需采集反光、手表、白物等困难场景。历史test多次用于报告，后续另留独立录制验收集。未完成相机串流、目标卡达标、INT8或三维精度验收。

## Git与产物交接

见[仓库管理与大文件规则](repository_workflow.zh-CN.md)。用户已指定GitHub仓库`https://github.com/JefferyXeom/pingpong2026`，训练端准备仓库专用SSH Deploy Key，origin使用`git@github.com:JefferyXeom/pingpong2026.git`。用户已在该仓库添加可写Deploy Key，SSH访问及实际推送已验证。首次检查远端为空；已将本地main首次上传并核对提交`e21480a`，后续交接更新同样通过main同步。当前同步状态以`git status -sb`及`git ls-remote origin refs/heads/main`为准。私钥及主机配置均留在被忽略的`.tools/ssh/`，不随Git交接，部署机使用独立密钥。首次版本作者为用户提供的JefferyXeom，仓库级配置邮箱jefferyxeom@gmail.com。代码提交号用Git记录查询，勿将尚未存在的提交号写成已完成。
