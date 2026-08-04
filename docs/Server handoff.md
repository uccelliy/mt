# 服务器交接（2026-08-04）

写给**下一段对话：正式设计实验**。这份只讲服务器现状与它对实验设计的约束，
不讲科学内容。

**一句话状态**：ULHPC 环境已打通并数值验证，邮件通知已实现并实测；
断点续跑机制（L3）尚未验证；70B 相关一切尚未尝试。

---

## 1. 设计实验时会真正卡住你的六条

这一节是本文档的重点。其余章节都是它的依据。

### 1.1 `gpu` 分区 2 天硬上限，且我们不用长作业 QOS

单作业 walltime 上限 **2 天**。账号关联里确实有 `iris-gpu-long`（14 天），
但**已决定不用**（2026-07-31）。

⇒ **任何预计超过 47 小时的作业，必须设计成能分片接力**：`--shard k/n`
+ `--resume`，超时后原样重提。排期时要先算清楚 `总时长 ÷ 47h = 提交几次`，
而这个数字只能靠实测吞吐外推。

⇒ **`--resume` / 分片 / merge 这套机制至今没有端到端验证过**（Server test
design §5.4b 的 L3）。第一个长作业之前必须补上，否则是在没有安全网的情况下
赌 47 小时。

### 1.2 显存决定拓扑，拓扑决定吞吐

| 配置 | 权重 | 单卡放得下？ | 拓扑 | 吞吐 |
|---|---|---|---|---|
| 8B NF4 | 5.68 GB | 16G ✓ | 4 进程数据并行 | **≈4×** |
| 8B FP16 | ~16 GB | 16G ✗ / 32G ✓（窄） | 单卡，`-C volta32` | ≈4×（4 卡 4 进程） |
| 70B NF4 | ~40 GB | ✗ | 单进程模型分片，2–4 卡 | ≈1× |
| 70B FP16 | ~140 GB | ✗ | **本集群做不到**（4×32G = 128 GB） | — |

- `-C volta16` = 16GB 卡，**18 个节点**，排队快
- `-C volta32` = 32GB 卡，**6 个节点**，排队慢
- `-C volta` **两种都匹配**，不要用它来指定 16GB

⇒ **70B 只能走 NF4**，而 NF4 在 V100 上已验证可用（§2）。
⇒ **70B 需要 ~140 GB 磁盘**：`--load 4bit` 是"下完整 BF16 权重再加载时量化"，
NF4 省显存不省磁盘。scratch 有 10 TB，够，但要预留时间下载。

### 1.3 E3 类作业在 16GB 卡上 `--batch-tokens` 最高 8192

实测：16384 需要一个 **6.1 GiB 连续块**，OOM；8192 和 4096 通过。
不是碎片化（`expandable_segments` 同样失败），是真容量不够——E0 峰值本就
占到整卡 96%。

⚠️ **`--batch-tokens` 会改变数值**（8192 vs 4096 在某窗口均值差 0.019），
所以**所有要互相比较的运行必须用同一个值**。若新结果要与本地那批 E3 网格
合并，用 `-C volta32` + 16384 才协议一致。

### 1.4 训练目前只能用一张卡

`src/mt/models/llm/finetuning.py:171` 是
`device_map = {"": local_rank}`——把整个模型钉在一张卡上，而仓库里没有
torchrun/accelerate 启动器，`LOCAL_RANK` 恒为 0。

⇒ **任何"多卡训练"的实验设计，前置项是先改这一行**（最小改动：换成
`"auto"`，得到 naive pipeline parallel；正规做法是 FSDP/DeepSpeed + 启动器）。
QLoRA 的 `--gradient-checkpointing` 默认已开，显存估算见 Server test design §4.6。

### 1.5 计费与 fairshare 会反噬

- GPU 分区的计费权重是 `CPU=1.0, Mem=0.037G, GRES/gpu=50.0`——**一张 GPU
  抵 50 个核**，所以核数内存怎么要都是零头，贵的是 GPU 数和时长。
- 每张 GPU 的配额是 **7 核 + 192 GB**，超一点就按多一整张 GPU 计费。
- **walltime 估不准两头挨打**：进 fairshare 的效率评分，还让 backfill
  塞不进空隙。别习惯性顶格要时间。
- `seff <jobid>` 看完成作业的真实 CPU/内存效率，用来把下次的 `--mem`
  和 `--time` 调准。

### 1.6 数值精度地板

跨硬件跑同一配置，**逐 choice 的 NLL 会差 ~0.019 nat**（fp16 累加顺序不同），
但**逐实验平均只差 6.046e-4**——零均值噪声，聚合抵消。

⇒ 协议纪律（详见 `centaur-eval-design.md` §9.5）：

1. **显式写 `--dtype fp16`**，不要用 `auto`——它在 MPS 上返回 **bf16**，
   那是系统性精度差异，不是零均值噪声。
2. 要比较的运行必须固定 `--dtype` / `--load` / `--batch-tokens` / 硬件类别，
   并随结果记录。
3. **任何效应量小于地板 5 倍时不得直接断言。** 现有结论里 E5 在 $w\ge1$
   的交互项（0.0006–0.0033）正好贴在这条线上。
4. 6.046e-4 是**单任务**实测；全量聚合的地板按 $1/\sqrt N$ 外推约 3e-5，
   **但那是外推**，要在实际报告的层面直接实测。

---

## 2. 已验证的（可以放心依赖）

| 项 | 值 / 结论 |
|---|---|
| 集群 | ULHPC `iris`，GPU 分区 24 节点（18×16G + 6×32G V100） |
| module | `env/release/2023b` + `lang/Python/3.11.5-GCCcore-13.2.0` |
| Python | 3.11.5（`requires-python` 已放宽到 `<3.12`，`uv lock` 零包版本变动） |
| torch | **2.13.0+cu126**，arch list 含 **`sm_70`** |
| transformers / peft / pandas | 5.14.1 / 0.20.0 / 3.0.5（**均比本地新**，见 §4） |
| **bitsandbytes NF4 在 sm_70** | **可用**（0.50.0，`Linear4bit` 前向输出有限值）→ **70B 路线活着** |
| 注意力后端 | FLASH/cuDNN 在 sm_70 不可用，**mem-efficient 正常选中**；40k token 峰值 0.15 GiB。已内置进 `preflight.py`，每次作业自动检查 |
| L0（preflight + pytest） | 通过：11 项检查，6561 sessions / 75 experiments，4 分片完整 |
| **L1（数值对拍）** | **通过**：逐实验平均 NLL 差 **6.046e-4** |
| 存储 | `$SCRATCH` 10T/11T 配额、inode 100 万；home 已用 21G/500G |
| 邮件 | BEGIN / END / FAIL / TIME_LIMIT_90 **四种全通**；**超时发的是 FAIL** |
| 进度通知 | `ProgressNotifier` 已实现 + 19 个单测；GPU 节点实跑已确认 |

**关于 L1 那个数字**：最初的判据是"逐行 max\|Δ\| < 1e-3"，失败了；改成判
**逐实验平均**后通过。**这不是把阈值调松**——6.046e-4 在原来的 1e-3 下同样
通过，改的是判哪个量。同时加了四道闸防止变成橡皮图章：单行最大差 < 0.25、
r > 0.99、`num_tokens` 必须完全一致、报告带符号平均差。换错模型会差 0.166
（超阈值 33 倍），照样挂。

## 3. 未验证的（不要假设它能用）

| 项 | 说明 |
|---|---|
| **L3：`--resume` / 分片 / merge** | **从没端到端跑过。** 见 §1.1——第一个长作业前必须补 |
| §6.2 的两个脆弱点 | `append_records` 非原子写（SIGKILL 可能留半行）；`.failed.csv` 污染（`--resume` 会永久跳过）。随 L3 一并验 |
| 全量数值地板 | 现有 6.046e-4 是单任务的，报告层面的地板要实测 |
| E3 的进度通知 | 代码已接通，但首次实跑时漏了，**下次跑才会验证到** |
| 分析异常退出 / requeue 后的通知行为 | Server Alarm §6.2 里标 ⬜ 的几项 |
| 70B 任何环节 | 下载、加载、`device_map="auto"` 跨卡分片，全部未试 |
| FP16 臂 | `MT_LOAD=none` + `-C volta32` 的路径没跑过 |

## 4. 一个必须记住的差异源

集群解析到的依赖**比本地新**：transformers 5.14.1（本地 5.10.2）、
peft 0.20.0（0.19.1）、**pandas 3.0.5（本地 2.3.3，跨大版本）**。
项目只声明下限，pip 每次拿最新。

⇒ 这是两边仅有的非受控差异。**将来对拍出现意料外偏差时，这是第一批要查的
地方。** 真要排除，就在集群上钉死版本重跑。

---

## 5. 最短可用路径

一次性设好（在集群 `~/.bashrc`）：

```bash
export SBATCH_MAIL_USER=<你的>@uni.lu
```

之后：

```bash
cd ~/mt && sbatch scripts/smoke_e0_e3.slurm
```

邮件两个通道自动生效（`sbatch` 读它当 `--mail-user`；`--export=ALL` 默认行为
把它带进作业，脚本转成 `--notify-email`）。**用 `@uni.lu`**——Gmail 能收到
但进垃圾箱。

换模型重新验证集群：

```bash
sbatch --export=ALL,MT_MODEL=<hf-id>,MT_EXPERIMENT=<task/exp.csv> \
       scripts/smoke_e0_e3.slurm
```

没有本地基线时对拍步骤自动跳过，退化为端到端冒烟测试。

**写新作业脚本**：从 `scripts/template_gpu_job.slurm` 起步（五段骨架，
每段一句注释说明为什么），再把中间三段换成一行 `source scripts/hpc_env.sh`。
教学说明在 Server test design §10。

## 6. 脚本清单

| 文件 | 作用 |
|---|---|
| `scripts/hpc_env.sh` | **唯一的站点设置入口**：module、`HF_HOME`、显存监控、仓库定位 |
| `scripts/template_gpu_job.slurm` | 学习用最小模板 |
| `scripts/smoke_e0_e3.slurm` | 集群验证（可换模型复用） |
| `scripts/e0_e3_minitaur.slurm` | 4 卡数据并行的生产模板 |
| `scripts/merge_shards.slurm` | 分片合并（`batch` 分区，**必须 `-C skylake`**） |
| `scripts/hpc_probe.sh` / `hpc_mail_probe.sh` | 集群/邮件能力探测 |
| `scripts/mail_notify_test.slurm` | 邮件类型验证（故意超时） |
| `scripts/experiments/preflight.py` | 提交前自检（含长上下文注意力检查） |
| `scripts/experiments/compare_scoring.py` | 数值对拍判定 |
| `src/mt/utils/slurm_progress.py` | 进度通知器 |

⚠️ **venv 绑死在 skylake 架构上**（AVX-512）。GPU 节点本身就是 skylake，
自动满足；但 `batch` 分区有一半是 broadwell，落上去会 `Illegal instruction`。
CPU 作业一律加 `-C skylake`。

## 7. 该读什么

1. **本文件** —— 约束与状态
2. `docs/Server test design.md` —— 环境细节、故障速查、作业脚本教学（§10）
3. `docs/Server Alarm.md` —— 通知系统设计与验收
4. `docs/centaur-eval-design.md` **§9.5** —— 数值复现性协议（影响科学结论）
5. `docs/centaur-eval-handoff.md` —— 科学侧现状（与本文件正交）

## 8. 建议的下一步顺序

1. **实验设计定稿**（新对话的主题）——决定跑什么模型、什么条件
2. 定稿后：**补 L3**（`--resume` 端到端），这是长作业的安全网
3. 同时：在报告层面**实测数值地板**，那个数字要写进论文
4. 若涉及 70B：先确认磁盘、再试加载、再谈跑分
5. 若涉及多卡训练：先改 `finetuning.py:171`
