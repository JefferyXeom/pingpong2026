# 部署包传输与接手步骤（2026-11-19）

交接包为`deployment_handoff_20261118_v1`。打包日期是11月18日，实际网络传输在11月19日执行；这两个日期不代表不同模型版本。传输已完成，状态为`complete_verified`：两个ZIP的远端SHA256一致，包内4287＋2104个清单成员全部校验通过，传输回执也已上传并核对SHA256。记录见[交接清单](../configs/deployment_handoff_20261118_v1.json)及接收目录中的`transfer_receipt.json`。

## 接收位置与验证依据

用户指定的部署端接收目录：

```text
/home/pingpong/pingpong-transfer/incoming/deployment_handoff_20261118_v1
```

训练端使用独立传输密钥，经SSH/SFTP传送，严格核对用户提供的ED25519主机指纹`SHA256:klk5Qej7Eg/eFMNkpqQFeX3NKLLe61I2huNSM8Vt57I`。网络地址、执行日志和传输明细保存在受Git忽略的本机`transfers/deployment_handoff_20261118_v1/transfer_receipt.json`；远端另有一份完整回执。

上传先写带内容哈希的`.part`，大小和完整SHA256符合预期后才生成正式文件；已有同名正式文件必须核对一致，不覆盖不同内容。断线重试时先核对远端部分文件与本机相同长度前缀的SHA256，再续传。两个ZIP完成后，再在部署端运行标准库工具逐一校验包内清单、大小和SHA256。

| 压缩包 | 字节数 | SHA256 |
| --- | ---: | --- |
| models_and_references.zip | 165,633,073 | `3fcb682aa29a4663490bff92cd5d2fa188c5e7a53c8df65230b95eb8f858943f` |
| holdout_images.zip | 3,358,043,139 | `63b59b9e8a08d835deb5f4b2fe277af719394cb1b2b60ec888cb95c1ce6b4028` |

`package_receipt.json`是打包时的固定快照，其中“等待传输”的状态描述打包时刻；网络完成状态由单独的`transfer_receipt.json`记录。保留这个区别，避免修改已经校验并传输的打包快照。

## 模型及测试数据

| 包内模型ID | 源训练ID | 数据版本 | 球心best轮次 | 用途 |
| --- | --- | --- | ---: | --- |
| B_n_merged_v1 | native_v1_additional_yolo26n_1280 | v1＋additional_v1，train17,404图 | 24 | 第一目标机验证候选 |
| C_s_merged_v1 | native_v1_additional_yolo26s_1280 | 同上 | 30 | s模型精度和速度对照 |
| D_n_merged_v2 | native_merged_v2_baiduout_yolo26n_1280 | 合并Baidu后的v2，train28,401图 | 12 | 新数据n模型对照 |
| E_s_merged_v2 | native_merged_v2_baiduout_yolo26s_1280 | 同上 | 22 | 新数据s模型对照 |

每组提供PT、动态ONNX、训练与导出配置、模型卡，以及6份原图参考预测。精确权重SHA256和数据身份见压缩包`core/package.json`与[实验登记](../reports/experiment_registry.json)。打包时的仓库基线提交为`37ea111f214fd9c5f026a91ab3e19dc16f75cf81`；这不表示历史训练在该提交时执行。包内打包/评估/构建工具另存当时源码快照及文件哈希，后续文档提交号也不改变已打包模型与工具的身份。

固定holdout共2,102张宽1280×高1024原图：val1,272图/1,272球，test830图/859球；YOLO标签和原始XML放在模型包内。quick128是这批图片的索引子集，适合先查流程。没有训练图片，没有已确认的无球负样本，图片尚未核实为同步双目帧。

## 可转发给部署端Codex的接手任务

请阅读仓库`AGENTS.md`、`docs/status.md`和本文，确认当前机器为Ubuntu部署端。若仓库无本地未提交修改，拉取最新main；有修改时先检查并妥善保留。检查接收目录的`transfer_receipt.json`状态为`complete_verified`，然后在该目录执行：

```bash
sha256sum -c SHA256SUMS
python3 verify_deployment_package.py models_and_references.zip
python3 verify_deployment_package.py holdout_images.zip
mkdir -p core testdata
python3 -m zipfile -e models_and_references.zip core
python3 -m zipfile -e holdout_images.zip testdata
python3 core/tools/verify_deployment_package.py core
python3 core/tools/verify_deployment_package.py testdata
```

解压前确认`core`、`testdata`中没有需要保留的其他版本文件。首次解压完成后，先用numpy复算B模型缓存，核对评估链路：

```bash
python3 core/tools/evaluate_deployment_predictions.py --core core --model B_n_merged_v1 --case h1024_w1280_tensorrt_fp16_b2 --subset quick128
python3 core/tools/evaluate_deployment_predictions.py --core core --model B_n_merged_v1 --case h1024_w1280_tensorrt_fp16_b2
```

随后核实本机GPU、驱动、CUDA、TensorRT版本及Turing SM7.5支持，在本机重建B模型FP16 batch2及batch1。命令、输出缓存规范见接收目录的`README.zh-CN.md`；构建工具使用TensorRT10 API，需要在实际环境验证。输入为`[B,3,1024,1280]`，BGR转RGB、CHW、FP32除255，原图不缩放、不补边、不裁切；输出`[B,300,6]`为`xyxy/conf/class`，class0=ball，免NMS。混合FP16 engine的输入/输出仍是FP32。

目标机须独立生成新预测，报告固定conf=.25、原图5px匹配的球心precision/recall/F1及TP/FP/FN，10px内成功匹配的定位误差中位/P95/P99，并报告相对同batch PT的转换位移和新增漏检。完整val/test用于一致性与固定协议评估，不能用于INT8校准或新超参数选择；quick128不替代完整评估。

目标机推理、测速、硬件同步、标定与相机串流仍需部署端完成。120对/秒即240图/秒是待验证目标，必须测完整流水线、延迟、积压和丢帧。历史holdout不构成已验证的同步双目录像，不能直接用于左右配对或三维精度验收。INT8尚未执行。
