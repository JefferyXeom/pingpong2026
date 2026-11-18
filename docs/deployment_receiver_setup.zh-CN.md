# 部署端文件接收准备

训练端已整理专用模型/参考结果包和完整holdout原图包，独立于Git传输。推荐SSH/SFTP：部署端接收，训练端推送；不使用GitHub Deploy Key登录部署端，不把模型/数据提交进Git。

## 可直接转发给部署端Codex的任务

请在Ubuntu部署机准备SSH/SFTP接收训练端文件。先确认现有SSH服务和登录用户，优先沿用现有配置；没有服务时安装并启动openssh-server。使用普通用户目录，不开放root登录，不必启用密码登录，不要为了本任务开放公网端口。如果两台机器不在同一内网且无可达私网，可先通过用户认可的Tailscale网络联通，再使用其地址。

接收目录建议`~/pingpong-transfer/incoming/deployment_handoff_20261118_v1`，创建后确认当前普通用户可写，至少有10GiB可用空间供压缩包、解压及后续引擎使用。训练端将以.part文件上传，部署端完整SHA256核对通过后才能使用。

将下面这条训练端专用传输公钥加入接收用户的`~/.ssh/authorized_keys`；同一公钥已存在就不要重复添加，不覆盖其他密钥。保持`.ssh`权限700、authorized_keys权限600。前缀`restrict`关闭此密钥的转发/PTY等能力，但允许SFTP及校验所需的普通远程命令：

```text
restrict ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAINObvZUuGT9BhnunAdNXpXetPHd9iZNU0+onOs9rm6gg pingpong2026-training-transfer
```

这是SSH登录授权，与GitHub仓库的Deploy keys页面无关；不要把GitHub的私钥复制过来。传输完成后可以保留供下轮模型交接，或删除这一条来撤销训练端登录权限。

完成后请返回以下非敏感信息，由用户转告训练端：

```text
部署机可达IP或SSH主机名：
SSH端口：
接收用户名：
接收目录（展开~后的绝对路径）：
该用户可用空间：
SSH ED25519主机公钥指纹（SHA256）：
```

主机指纹可在部署端执行：

```bash
ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub -E sha256
```

请核实用户所属组/安全策略没有禁止此密钥登录或SFTP。不要返回密码、私钥或Token。

## 训练端接下来的操作

收到上述信息后，训练端先核对目标主机公钥与部署端返回指纹，使用`.tools/ssh/pingpong_transfer_ed25519`，显式指定密钥和严格主机校验。先做只读连通及磁盘空间检查，再传送SHA256SUMS、校验脚本、模型包、图片包及说明。

优先SFTP `reput`续传，临时文件以包SHA256标识，避免不同版本的部分文件误续传；校验已有部分文件前缀后再续传。完成后在部署端逐包计算SHA256，与训练端清单一致才将.part改为正式ZIP名称；再运行标准库校验工具核对每个成员。没有部署端SHA回读就不宣称传输已验证。

首次联通和传输由训练端根据实际地址执行；本文件不是“已经传输成功”的记录。
