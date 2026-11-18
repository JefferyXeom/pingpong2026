# 部署交接包：deployment_handoff_20261118_v1

2026-11-18（北京时间）已完成打包和完整校验；11月19日已传至Ubuntu部署机，并完成两个ZIP的远端SHA256和全部成员校验，状态`complete_verified`。本机目录为`transfers/deployment_handoff_20261118_v1`，受Git忽略，不上传到代码仓库。[实际传输记录与部署端接手步骤](deployment_transfer_20261119.zh-CN.md)。

| 文件 | 大小 | 用途 |
| --- | --- | --- |
| models_and_references.zip | 165,633,073字节，约158MiB | 4份选中PT、4份原图ONNX、模型卡、源训练/构建配置、24份完整原图预测缓存、标签/XML、可移植校验和评估工具 |
| holdout_images.zip | 3,358,043,139字节，约3.13GiB | 全val1272/test830，共2102张原始1280×1024 PNG |
| SHA256SUMS | 文本 | 两个ZIP的整包校验值 |
| verify_deployment_package.py | Python标准库工具 | 解压前/后逐成员SHA256校验 |
| README.zh-CN.md | 说明 | 目录结构、解压、复算、目标机TRT构建与接口 |
| package_receipt.json | 打包快照 | 本机核验通过、打包时剩余空间和当时的传输准备状态 |
| transfer_receipt.json | 传输回执 | 11月19日的远端文件SHA256、包内成员核验和完成状态；本机与远端各一份 |

模型B/C是此前合并v1的n/s第24/30轮best；D/E是新增Baidu合并v2的n/s第12/22轮best。所有PT与ONNX核对源哈希，ONNX通过checker；4模型×6个原图后端/batch缓存重新计算球心指标和转换位移，与原始报告一致。首选B开始目标机测试，其他保留比较，不把新模型视为全面替代。

原图与原XML逐张核对SHA，图片均解码检查1280×1024。quick128按split、左右目、目标数分层抽取，指向同一份图片，不额外复制；仅检查流程，不替代完整验收。包内standalone评估脚本已从ZIP提取并执行，B的quick128和完整FP16 b2参考复算均通过。

两个ZIP均逐成员回读校验通过。没有训练图片、Windows engine、私钥或Token，也未修改源数据。ZIP采用STORE，避免无收益地重压PNG，接收端应准备至少10GiB空间供压缩包、解压和引擎使用。目标机TRT构建脚本使用TensorRT10 API，已语法检查和配置审查，尚未在Ubuntu/2060Super运行，不预设当前版本支持Turing。

输入[B,3,1024,1280]、RGB/CHW/FP32除255；输出[B,300,6]为xyxy/conf/class，免NMS。现有mixedFP16 engine接口仍是FP32。full/quick新预测缓存均需按image_order/quick128_indices顺序保存；标准球心conf=.25、匹配5px，定位分位数仅对10px匹配，转换与同batch源PT对照；缺少目标时不当零误差。包内README详述命令。

历史holdout不构成已验证的同步双目验收录像。部署端仍须完成新硬件同步/标定、主球关联、目标机推理一致性和120对/秒完整流水线测试；INT8尚未执行，不能用holdout做校准。

[机器可读包清单](../configs/deployment_handoff_20261118_v1.json)包含两个ZIP的SHA256及完成状态；[接手文档](deployment_transfer_20261119.zh-CN.md)可转发给部署机Codex。实际接收目录为`/home/pingpong/pingpong-transfer/incoming/deployment_handoff_20261118_v1`。两个ZIP及全部6391个清单成员已在部署端核验通过，传输回执也已上传核验；压缩包尚未解压，目标机推理尚未测试。`package_receipt.json`保留打包时快照，以单独的`transfer_receipt.json`记录网络完成状态。
