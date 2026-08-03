# Slurm 服务器任务邮件提醒

## 1. 范围与已定决策

只做学校 ULHPC 服务器版本。V1 的运行拓扑固定为：

- 一个 Slurm Job ID 只运行一个分析。
- 一个 Python 分析进程同时使用四张 GPU，模型分片到四张卡上。
- 不启动四个分析进程，不把数据拆成四个 shard。
- 每个新 Job ID 都创建新的进度通知器；重新收到相同阈值邮件是可接受行为。
- 不支持作业数组、多分析阶段或多进程发送者。

因此 V1 不需要外部监听器、cron、SQLite、`progress.json` 或 Job ID 登记命令。
通知分为两个互相独立的通道：

1. **Slurm 原生邮件**负责开始、结束、失败和 walltime 提醒。
2. **分析进程内通知器**负责 25%、50%、70%、90% 的工作进度邮件。

**拓扑不由通知系统决定，由模型的显存需求决定**，两种都必须支持：

| 情况 | 拓扑 | 吞吐 |
|---|---|---|
| 模型装得下单卡（如 8B NF4，5.68 GB） | 4 进程数据并行，各占一卡 | ≈4× |
| 模型装不下单卡（如 70B NF4，~40 GB） | 单进程模型分片到 4 卡 | ≈1×，但跑得动 |

（依据见 `docs/Server test design.md` §8。）

⇒ **通知器不能绑定任何一种拓扑。** 一条规则同时覆盖两者：**只有 shard 0
发通知**。

- 模型分片模式：只有一个进程、没有 `--shard`，它即 shard 0 → 发送。
- 数据并行模式：4 个进程中只有 shard 0 发送 → 不会收到 4 份重复邮件。

分片用的是 `_common.py:61` 的 `rows[k::n]` **轮询交错**取样（shard 0 取第
0、4、8… 个 session），因此其子集在统计上代表全体——shard 0 到 25% 时整体也在
25% 附近，对"到 25% 提醒我一下"这个需求精度绰绰有余。

**因此不需要改动 launcher。**

## 2. 实现前先验证邮件能力

仓库提供 `scripts/hpc_mail_probe.sh`。它只依赖 Bash 和服务器系统命令，不依赖
module、Python、项目 venv 或模型环境。

### 2.1 登录节点能力检查

先做完全无副作用的检查：

```bash
bash scripts/hpc_mail_probe.sh --check-only
```

它会显示 `sendmail`、`mailx`、`mail`、`sbatch` 和 `timeout` 是否存在，不发送邮件，
也不提交作业。

### 2.2 登录节点直接发信

```bash
bash scripts/hpc_mail_probe.sh --send-test <你的邮箱>
```

脚本按固定优先级只选择一个本地邮件后端，避免一次测试发出多封直接邮件。命令返回 0
只表示本地邮件系统接受了消息，不能证明已经送达；必须人工检查收件箱和垃圾箱。

### 2.3 Slurm 与计算节点测试

```bash
bash scripts/hpc_mail_probe.sh --submit-slurm-test <你的邮箱>
```

该命令在 `interactive/debug` 上提交一个 2 分钟、1 CPU、128 MB 的小作业，不申请
GPU。正常情况下会收到：

1. Slurm `BEGIN` 邮件；
2. 计算节点通过本地邮件命令发送的 direct probe；
3. Slurm `END` 邮件。

如果计算节点没有本地邮件命令，probe 作业会非零退出，此时应至少收到 Slurm
`FAIL` 邮件。提交命令只证明作业进入队列，最终仍需检查邮箱，并用脚本打印的 Job ID
查询：

```bash
squeue -j <JobID>
sacct -X -j <JobID> --format=JobID,State,ExitCode,Elapsed
```

若集群配置变化，可通过 `MT_MAIL_PROBE_PARTITION` 和 `MT_MAIL_PROBE_QOS` 覆盖测试
分区与 QOS。也可以显式执行 `--all <邮箱>`，依次完成能力检查、登录节点直发和
Slurm 测试。

实际进度邮件会从 GPU 计算节点发送。在以后已有的 GPU interactive 会话中还应再运行
一次 `--check-only` 和 `--send-test`；不要只为邮件测试单独占用 GPU。

## 3. V1 通知流程

### 3.1 Slurm 终态与时间提醒

真实作业脚本使用学校的原生邮件设施，例如：

```bash
#SBATCH --mail-user=<你的邮箱>
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90
```

`TIME_LIMIT_90` 表示已经使用 90% 的 walltime，不代表分析完成了 90%。实际启用前应
先在服务器上确认当前 Slurm 版本接受该 mail type；不接受时先退回
`BEGIN,END,FAIL`。

Slurm 邮件只说明调度器看到的作业状态。`END` 或 `COMPLETED` 不能写成“科学结果已
验证通过”；应用层仍可能跳过失败样本或产生不完整输出。

### 3.2 进程内进度提醒

分析 runner 在结果成功落盘后更新通知器：

```python
from mt.utils.slurm_progress import ProgressNotifier

notifier = ProgressNotifier(
    total=len(sessions),
    label="centaur-evaluation",
    recipient="name.surname@uni.lu",
)

for completed, session in enumerate(sessions, start=1):
    result = analyze(session)
    save_result(result)
    notifier.update(completed)
```

`ProgressNotifier` 是后续要实现的接口，本次邮件探测不实现它。契约为：

- `total > 0`，`current` 必须处于 `[0, total]` 且单调不减。
- 阈值固定为 25%、50%、70%、90%。
- 同一 Python 进程内，每个阈值最多成功发送一次。
- 20% 一次跳到 80% 时只尝试最新的 70%，不补发已经过时的 25% 和 50%。
- 100% 不发进度邮件；最终状态由 Slurm `END`/`FAIL` 邮件负责。
- 邮件失败不得让分析失败；输出简短警告，并在下一次 `update()` 时重试当前阈值。
- 新 Job ID 启动新进程，阈值状态重新开始。

阈值状态只保存在当前进程内。若 Slurm 用同一个 Job ID requeue 并重启 Python，可能
再次发送已经发过的阈值；V1 接受这个限制。只有以后确实需要跨进程去重时，才考虑一
个小型状态文件，不预先引入 SQLite。

## 4. 邮件内容与安全边界

进度邮件只包含：

- 分析标签；
- `SLURM_JOB_ID`；
- 当前量、总量和百分比；
- 当前分析进程的已运行时间。

邮件不包含节点名、命令行、模型路径、数据路径或日志内容。实现必须：

- 保守校验收件地址，拒绝换行、控制字符和以 `-` 开头的值；
- 使用参数列表调用固定的邮件程序，不使用 `shell=True`、`eval` 或拼接 shell 命令；
- 给邮件命令设置短超时；
- 不保存邮箱密码或第三方 API Token；
- 不因邮件命令缺失、超时或非零退出而中止科学作业。

## 5. 根据探测结果决定下一步

| Slurm 原生邮件 | GPU 计算节点 direct mail | 决策 |
|---|---|---|
| 可用 | 可用 | 实现完整 V1：Slurm 终态 + 进程内进度 |
| 可用 | 不可用 | 先启用终态邮件；若仍需要进度，再设计登录节点轮询 + 小型 JSON 状态 |
| 不可用 | 可用 | 进度可发，但终态没有可靠兜底；先向 ULHPC 确认邮件配置 |
| 不可用 | 不可用 | 停止实现，不自动引入第三方邮件凭据或 API |

登录节点 direct mail 可用不能代替 GPU 计算节点实测。

### 5.1 实测结果（2026-08-03）

登录节点 `--check-only`：bash 4.4.20，`timeout` / `sendmail` / `mailx` / `mail` /
`sbatch` **全部存在**。

| 测试 | 结果 |
|---|---|
| 登录节点 direct → `@uni.lu` | ✅ 收到 |
| 登录节点 direct → Gmail | ⚠️ **送达了，但进了垃圾箱** |
| Slurm probe 作业（`interactive/debug`） | ✅ BEGIN、计算节点 direct、END **三封全收到** |

⇒ **落在决策表第一行，实现完整 V1。**

**两条由此确立的约束：**

1. **优先用 `@uni.lu` 地址。** 校外投递（Gmail）实际是通的，但**被判为垃圾邮件**
   ——首次验收时差点误判为"没送达"。若要长期用校外地址，须先在收件端把发件域
   加入白名单，否则进度提醒会静默沉进垃圾箱、失去"不用主动查"的全部意义。
   这也实证了本文档反复强调的一点：**邮件命令返回 0 只代表本地邮件系统收下了**，
   验收必须以收件箱为准，**而且要连垃圾箱一起看**。
2. **计算节点直发已确认，但用的是 CPU 节点**（probe 作业 1 CPU / 128 MB，
   不占 GPU）。GPU 节点尚未实测——按 §2.3 的约定，**并入下一次已有的 GPU
   interactive 会话顺手验一次**，不单独为此占用 GPU。

## 6. 测试与验收

### 6.1 通知器单元测试

- `24,25,49,50,69,70,89,90,100` 只产生 25%、50%、70%、90% 四次通知。
- 重复调用相同进度不重复发送。
- 20% → 80% 只发送 70%。
- 100% 不发送进度邮件。
- 新建通知器后允许重新发送全部阈值。
- 负数、超过 total、进度回退和 `total=0` 被明确拒绝。
- 邮件命令失败或超时不抛出到分析循环，并留下简短警告。
- 邮件内容不泄露节点、路径、命令或日志。

### 6.2 服务器验收

- 登录节点 `--check-only` 不发送邮件、不提交任务。
- 登录节点 direct probe 被本地邮件系统接受且实际收到。
- 小型 Slurm probe 收到 BEGIN、计算节点 direct 和 END 邮件。
- 小型主动失败作业收到 Slurm FAIL 邮件。
- 小型分析依次触发 25%、50%、70%、90%，每个阈值只收到一次。
- 分析进程异常退出后不再有进度邮件，但仍收到 Slurm 终态邮件。
- 新 Job ID 重提后允许再次收到相同进度阈值。
- 运行期间只有一个 Python 分析进程和一个通知发送者，即使模型占用四张 GPU。
- 验收后没有 SQLite、`progress.json`、cron 条目或长期后台进程。
