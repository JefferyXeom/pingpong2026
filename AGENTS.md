# 本项目的实验约定

处理本工作区的YOLO训练、数据准备、转换、评估和结果说明时，先阅读并应用项目skill：

`skills/pingpong-experiments/SKILL.md`

最新用户指令优先。正在执行的实验和报告按实验ID管理，先看`reports/README.md`、`reports/experiment_registry.json`及当前实验配置，避免重复启动GPU任务。

跨机器协作开始前再读`docs/status.md`、`docs/repository_workflow.zh-CN.md`及相关配置。确认本机是Windows/3090训练端还是Ubuntu/2060 Super部署端；另一台机器的绝对路径、驱动和运行状态不能直接套用。代码与交接记录用Git，模型/数据/录像另行传输；没有用户指定远端和发布指令时，不自行创建远端或推送。

阶段完成后更新status及实验记录，模型变更记录模型ID、权重SHA256、数据版本和对应代码提交。执行`python scripts/audit_git_content.py`核对暂存/已跟踪清单，不能用`git add -f`绕过大文件规则。完整历史报告中的权重、图片、缓存链接依赖本地或单独传输的产物；Git克隆不包含这些文件。
