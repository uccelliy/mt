# Slurm 服务器任务外部监听器

## 概要

只做学校服务器版本。监听器独立运行在 Slurm 登录节点，不负责启动训练，也不接触模型代码、日志或数据。

流程为：

1. 提交 Slurm 作业。
2. 执行一次 `slurm-watch add <JobID>` 登记。
3. 训练程序只更新一个本地 `progress.json`。
4. 监听器每 1–2 分钟读取 Slurm 状态和进度文件。
5. 在 25%、50%、70%、90% 以及作业完成、失败、超时或取消时发邮件。

Slurm 的 `sacct` 能提供作业状态、耗时和退出码，因此结束状态不依赖训练程序主动通知。[Slurm sacct 文档](https://slurm.schedmd.com/sacct.html)

## 核心实现

- 提供一个轻量 Python 命令行工具：
  - `slurm-watch doctor --email <地址>`：检查 `squeue`、`sacct`、cron 和服务器本地发信能力，并发送测试邮件。
  - `slurm-watch add <JobID> --label <别名> --progress-file <路径>`：登记一个任务。
  - `slurm-watch list`：查看任务、状态和最近进度。
  - `slurm-watch remove <JobID>`：停止监听。
  - `slurm-watch poll`：执行一次查询，供 cron 定期调用。
  - `slurm-watch watch`：前台持续监听，方便测试。
- 使用 SQLite 保存登记任务、已发送阈值和最终状态，重复执行 `poll` 不会重复发邮件。
- 通过 `squeue` 获取排队/运行状态，通过 `sacct -X` 获取最终状态、退出码和耗时，并忽略 `.batch`、`.extern` 等作业步骤。
- 状态处理：
  - `COMPLETED`：完成邮件。
  - `FAILED`、`OUT_OF_MEMORY`、`NODE_FAIL`、`BOOT_FAIL`：失败邮件。
  - `TIMEOUT`：超时邮件。
  - `CANCELLED`：取消邮件。
  - `REQUEUED`：继续监听，不当作最终失败。
- 默认只监听手动登记的 Job ID；作业数组需逐个登记具体数组任务 ID。

## 真实进度接口

训练代码仅负责写进度，不包含任何邮件或密钥逻辑：

```python
from slurm_watch.progress import ProgressFile

progress = ProgressFile("progress.json", total=num_steps)

for step, batch in enumerate(loader, start=1):
    train_step(batch)
    progress.update(step)
```

进度文件格式固定为：

```json
{
  "current": 700,
  "total": 1000,
  "updated_at": "2026-07-30T12:00:00Z"
}
```

- 使用临时文件加原子替换，避免监听器读到写了一半的 JSON。
- helper 仅在进度至少变化 1% 或间隔超过 60 秒时落盘，避免每个 step 都产生磁盘写入。
- 阈值默认为 25%、50%、70%、90%；100% 不单独通知，只有 Slurm 真正进入 `COMPLETED` 后才发完成邮件。
- 如果一次轮询从 20% 跳到 80%，只发送最新的 70% 邮件，不补发已经过时的 25% 和 50% 邮件。
- 邮件只包含任务别名、Job ID、进度、Slurm 状态、耗时和退出码；进度文件路径、命令、日志和真实节点名不会进入邮件。

## 运行与发信

- 首先运行 `doctor` 验证登录节点上的 `mail`、`mailx` 或 `sendmail`。
- 如果本地邮件可用，直接使用学校邮件基础设施，不保存个人邮箱密码或第三方 API Token。
- 优先使用 cron 每两分钟执行一次 `slurm-watch poll`；安装 cron 前显示将写入的配置并要求确认。
- 如果服务器不允许 cron，则保留前台 `watch` 模式，并明确说明退出 SSH 后可能停止监听；不擅自在登录节点运行长期后台进程。
- 如果服务器本地邮件不可用，第一版停止在诊断结果，不自动引入第三方发信服务。

## 测试与验收

- 小型成功作业依次写入进度，验证收到 25%、50%、70%、90% 和最终完成邮件。
- 作业在 60% 主动退出，验证收到 25%、50% 和失败邮件，且包含正确退出码。
- 分别验证超时、取消、重排和节点失败状态不会误判或重复发送。
- 连续多次执行 `poll`，同一阈值和最终状态只能通知一次。
- 进度文件短暂缺失、内容不完整或 Slurm accounting 延迟时，监听器继续重试而不误报完成。
- 检查数据库、运行日志和邮件均不包含训练日志、命令行、模型路径或邮件凭据。

## 假设

- 服务器使用 Slurm，登录节点可运行 `squeue` 和 `sacct`。
- 作业与监听器能访问同一共享文件系统中的 `progress.json`。
- 第一版只支持服务器，不处理个人电脑，也不检测训练指标好坏。
- 是否允许 cron 以及本地邮件是否可用，由 `doctor` 实测后确定。
