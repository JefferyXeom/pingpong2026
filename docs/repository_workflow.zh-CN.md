# Git仓库与大文件交接

本项目使用轻量Git仓库保存代码和可检查的交接记录。原始及派生数据、模型二进制和大量运行产物独立保留，忽略规则不删除这些本地文件。当前不引入Git LFS/DVC；未来有需要时再决定。

| 纳入Git | 留在本地/单独传输 |
| --- | --- |
| scripts、skills、AGENTS、文档、训练/目标机配置 | datasets、runs、artifacts、logs、虚拟环境和缓存 |
| models中的model_card.json和README | PT/ONNX/engine/预测NPZ/视频/图片/压缩包 |
| reports中的必要JSON/CSV、摘要、曲线SVG | 大型逐图审计JSON/JSONL、内嵌图片的37MB HTML |
| 少量逐轮epoch_metrics.jsonl | configs/datasets中本机生成的绝对路径列表和YAML |

具体以根目录`.gitignore`为准。已入Git的旧文件不会因后来加忽略规则自动退出，所以每次提交都要检查暂存内容。

## 两端如何协作

每端各自使用Codex会话，开始先读`AGENTS.md`和`docs/status.md`。远端Git仓库建立后，两端clone同一仓库；每次修改前拉取最新提交，完成后提交并在用户授权下推送。避免两边同时改同一状态段落，交接以模型ID、哈希、代码提交及数据版本为依据。

本地首次提交已完成；用户现已指定远端`https://github.com/JefferyXeom/pingpong2026`。训练端使用仓库专用SSH Deploy Key，origin为`git@github.com:JefferyXeom/pingpong2026.git`；GitHub公钥已登记并授予写入，SSH访问、实际推送及远端提交一致性均已验证。首次提交作者配置仅作用于当前仓库，不改系统全局身份。

Deploy Key是授权给单个仓库的SSH公私钥，不是个人账号Token。训练端私钥留在`.tools/ssh/`并限制当前Windows用户访问，不设口令以支持自动Git操作；GitHub只登记公钥。它不是真正的硬件绑定，复制私钥也会复制权限，因此每台机器使用不同密钥，GitHub可单独撤销。仓库级core.sshCommand指定该密钥，GitHub主机公钥从HTTPS官方api.github.com/meta获取，受限时改用GitHub官方指纹文档；两者都不可用则保持严格校验并待后续补齐主机公钥，不跳过校验。不改用户全局SSH配置。部署端另建只读密钥。

## Windows使用方式

系统Git可用时直接使用`git`。本机使用被忽略的`.tools/git/`中的官方MinGit，可以直接调用可执行文件，无需修改PowerShell脚本执行策略；Ubuntu可直接用系统Git：

```powershell
& ./.tools/git/cmd/git.exe status
& ./.tools/git/cmd/git.exe diff --cached --stat
& ./.venv/Scripts/python.exe scripts/audit_git_content.py
& ./.tools/git/cmd/git.exe commit -m "Describe the actual change"
```

`.vscode/settings.json`中的Git路径为机器本地配置，不进仓库。便携Git同样不进仓库，部署机自行安装Git。需要时重新加载VS Code以识别新Git路径。

## 提交前检查

`scripts/audit_git_content.py`读取Git索引中的实际暂存内容，列出总量与最大文件；检查每文件≤2MiB及整体≤20MiB，并拒绝被排除的二进制、运行目录、环境目录或疑似密钥文件。超限时应筛选摘要/单独传输，不能直接`git add -f`绕过。该检查用于本项目代码仓库体积控制，不代替对内容和密钥的人工审查。

## 模型/数据如何交接

Git clone不包含权重与数据，完整训练、转换和本地历史验证需要另外接收对应产物。第一批至少交接选中best.pt、必要ONNX、模型SHA256和一套固定核验图片/标注/训练端预测；不默认传输全部训练数据。TensorRT engine在目标Ubuntu/2060 Super按兼容版本重建。

原报告中指向本地权重、图片、生成列表及缓存的链接继续保留历史身份，克隆后这些链接可能缺失；它们是独立产物引用，不表示文件已经上传。原`verify_report_layout.py`需要完整训练端产物；在只有Git克隆的机器上使用独立的`audit_git_content.py`检查仓库内容。迁移时不要把Windows历史绝对路径当部署路径。

## 当前同步入口

训练端main已上传到上述GitHub仓库。部署端接手时在本机配置独立只读Deploy Key，再执行：

```bash
git clone git@github.com:JefferyXeom/pingpong2026.git
cd pingpong2026
```

随后阅读AGENTS.md和docs/status.md，按部署端待办继续；权重与核验图片另行接收。当前训练机可使用仓库专用密钥拉取/推送，无需个人账号网页登录。
