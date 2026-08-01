# 学校 HPC 首次上机测试设计

## 0. 本文档的范围

第一次使用学校 HPC，本文档只覆盖**环境打通与验证**：把仓库跑起来，并用一个
**已经在本地跑过的实验做数值对拍**，确认集群上的结果可信。

不在本文档范围内：

- **正式科学作业的设计**（Adapter E3、E6 任务信息消融、FP16 锚点等）——见
  `docs/centaur-eval-handoff.md` §7 的优先级列表，另行设计。
- **作业状态通知**（原需求："服务器可以根据任务运行状态、时间向我发送提示"）
  ——确实需要，但有专门任务处理，本文档不设计实现。仅记一条事实备用：ULHPC
  官方示例里就有 `#SBATCH --mail-user=` / `--mail-type=end,fail`，最省事的
  一档不需要写任何代码。本文档的脚本只用 `echo` + 时间戳留进度可读性。

产出物：一份可以照着敲的上机流程，以及修好的 SLURM 脚本
（`scripts/smoke_e0_e3.slurm`、`scripts/e0_e3_minitaur.slurm`）。

---

## 1. 集群事实核对表

集群是 **ULHPC（卢森堡大学）**，两套系统 `iris` 和 `aion`；**GPU 全在 iris 上**
（aion 是纯 CPU 的 AMD epyc 节点）。§1.1 来自官方文档，§1.2 是仍需实测的。

### 1.1 已确认（ULHPC 官方文档）

| 项目 | 值 |
|---|---|
| GPU 分区 | `gpu`，最多 **4 节点**，walltime **2 天** |
| **16GB V100 节点** | `iris-[169-186]`，18 个，feature **`volta16`**（`volta` 两种都匹配！） |
| **32GB V100 节点** | `iris-[191-196]`，6 个，feature **`volta32`** |
| ~~更新的 GPU 节点~~ | 集群里有 `hopper`(4×H100 96G) 和 `l40s`(4×L40S 46G)，但**我们没有权限，只能用 `gpu` 分区的 V100** |
| GPU 节点规格 | 2 socket × 14 核 = **28 核**，**756 GB** 内存，4×V100 SXM2，驱动 580.159.04 |
| GPU 请求写法 | **`--gpus-per-task=N`**（官方 AI/DL launcher 示例的写法） |
| CPU 配比规定 | **每张 GPU 配 1/4 节点核数 = 7 核**（官方明确要求） |
| 普通 QOS | `low`(prio 10) / `normal`(100) / `high`(200) / `urgent`(1000) |
| 我的 QOS | `normal` / `low` / `debug` / `besteffort` / `iris-gpu-long` / `iris-batch-long` / `iris-bigmem-long` / `aion-batch-long`（**无 `high`/`urgent`**，`normal` 就是优先级天花板） |
| 长作业 QOS | `iris-gpu-long`（14 天）关联里**有**，但我们**按 48h 规划**，见 §6.3 |
| 快速测试分区 | `interactive`：2 节点，**2h**，**只允许 `--qos=debug`**；含 GPU 节点，可交互调试 |
| 抢占式 QOS | `besteffort`，会被任何其它 QOS 打断，需自带 checkpoint |
| Account | UL 附属用户有默认 account，可不写 `-A` |
| module 系统 | **LMod + EasyBuild**（模块名带类别前缀，如 `numlib/cuDNN`） |
| `$HOME` | `/home/users/$USER`，GPFS，**无备份**，有 SSD 缓存（小文件快） |
| `$SCRATCH` | `/scratch/users/$USER`，**Lustre**，无备份 |
| `$PROJECTHOME` | `/work/projects/<项目名>`，GPFS，部分备份（仅 `backup` 子目录） |
| 超线程 | **关闭**，#核 = #线程 |
| 脚本头 | 必须 `#!/usr/bin/bash --login`（官方反复强调别忘 `-l`） |
| 线程数写法 | 官方建议 `OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}` |

**两条会改变做法的**：

1. **`module` 在登录节点上根本不存在**——官方原话是它在 access/login 服务器上
   被刻意禁用，只在计算节点的 slurm 作业里可用。所以 §5 的 L0
   **不能在登录节点跑**，必须进 `interactive` 分区。
2. **16G/32G 靠 feature 区分，但 `volta` 不是 16GB 的意思**：16GB 节点的
   feature 是 `skylake,gpu,volta,volta16`，32GB 的是 `...,volta,volta32`
   ——**`-C volta` 两种都会匹配**。要精确指定必须用 **`-C volta16`** 或
   **`-C volta32`**。不需要 `--nodelist`，排队不受影响。

**我们按 `gpu` 分区的 2 天上限来规划。** `sacctmgr` 显示账号关联里确实有
`iris-gpu-long`（MaxWall 14 天，最多 2 节点、4 作业/用户），但这是**主动
不用**的决定，不是权限问题——记在这里以免以后再查一遍。
⇒ §6 的分片续跑不是保险措施，是**必需机制**：任何超过 2 天的全量作业
都必须靠重复提交接力完成。

### 1.2 仍待实测

| # | 项目 | 探测命令 | 值 |
|---|---|---|---|
| C9 | 有没有 Python **3.10**（2023b 工具链配的是 3.11.5，与本项目 `<3.11` 冲突） | `module spider lang/Python`（**类别要写全**，否则会匹配到 GitPython/TopHat） | 2023b 树 = **3.11.5**；其它工具链树待查 |
| ~~C10~~ | ~~CUDA / cuDNN 模块~~ | **已确认（GPU 节点）**：`system/CUDA/12.6.0`、`numlib/cuDNN/9.5.0.50-CUDA-12.6.0`、`lib/NCCL/2.20.5-...-CUDA-12.6.0`。**CUDA 12.x，不是 13——Volta 仍受支持** | ✓ |
| ~~C11~~ | ~~集群 PyTorch~~ | **已确认**：`ai/PyTorch/2.3.0-foss-2023b-CUDA-12.6.0`（需先 `env/release/2023b`）→ python 3.11.5 / torch 2.3.0 / cuda 12.6 / **`arch ['sm_70']`** / `available True`。**但 torch 2.3.0 对本仓库太老，见 §4.2b** | ✓ |
| ~~C12~~ | ~~登录节点外网~~ | **已确认：huggingface.co / pypi.org 均 HTTP 200，登录节点有外网** | ✓ |
| ~~C14~~ | ~~配额~~ | **已确认**：`$SCRATCH=/scratch/users/$USER` 配额 10T 软 / 11T 硬，**inode 100 万软 / 110 万硬**；home 已用 21G（上限 500G）。对我们绰绰有余 | ✓ |
| C15 | `/work/projects/acnets`（`pcardoso` 名下）我是否有读写权 | `ls -ld /work/projects/acnets`；`id` 看在不在 acnets 组 | |
| ~~C19~~ | ~~QOS 授权范围~~ | **已确认**：见 §1.1「我的 QOS」行 | ✓ |
| C16 | `/scratch` 的清理策略与保留期 | 官方文档存储章节 | |

> **C14 的 inode 上限值得单独确认。** 官方在 Conda 那节提到配额同时限制
> "storage space **and number of files**"。一个 venv 有几千个小文件——空间够
> 但 inode 用尽是 HPC 上很常见的翻车方式。

### 1.3 怎么跑探测脚本

`scripts/hpc_probe.sh` 填 §1.2 的表。**它要跑两遍**：ULHPC 在登录节点上刻意
禁用了 `module`，所以 C9/C10/C11 三行只有在计算节点上才有输出。

脚本全部只读（`sinfo` / `sacctmgr show` / `module avail` / `quota` / `df` /
`ls` / `curl -I`），不提交、不加载、不写任何东西，可以反复跑。它也**故意没有
`set -e`**——某一段失败要能继续跑完后面的。

**① 本机：把脚本传上去**（这时仓库还不用传，只要这一个文件）

```bash
scp -P 8022 scripts/hpc_probe.sh <你的login>@access-iris.uni.lu:~/
```

> `scp` 的端口参数是**大写 `-P`**，`ssh` 是小写 `-p`。ULHPC 的 access 端口
> 通常是 8022，以你开户邮件里的为准。如果你已经配好了 `~/.ssh/config` 的
> `Host iris-cluster`，直接 `scp scripts/hpc_probe.sh iris-cluster:~/`。

**② 登录**

```bash
ssh -p 8022 <你的login>@access-iris.uni.lu
```

**③ 第一遍：在登录节点上跑**

```bash
bash --login ~/hpc_probe.sh 2>&1 | tee ~/hpc_probe_login.txt
```

这一遍拿到分区、QOS、账号、配额、文件系统、外网连通性。
**C8 那段会打印 "no 'module' command"——这是预期的，不是故障。**

**④ 申请一个交互作业**

```bash
salloc -p interactive --qos=debug -N 1 --ntasks-per-node=1 -c 1 -C skylake --time=0:30:00
```

⚠️ **必须是 `--qos=debug`。** `interactive` 分区的 `AllowQos` 只有 `debug`
一个，写 `--qos=normal` 会直接报 `Invalid qos specification`。

`interactive` 分区上限 2h、优先级高，通常几秒到几分钟就分到。分到之后
**命令提示符会变成计算节点的名字**（形如 `iris-0xx`），这就是判据。

顺带：`interactive` 分区**包含 GPU 节点**（iris-[169-186,191-196]），
所以调试 L1 时可以直接要一张卡交互着试，不必每次 sbatch 等排队：

```bash
salloc -p interactive --qos=debug -N 1 --ntasks-per-node=1 -c 7 -G 1 \
       -C volta16 --time=2:00:00
```

**⑤ 第二遍：在计算节点上跑**

```bash
bash --login ~/hpc_probe.sh 2>&1 | tee ~/hpc_probe_compute.txt
```

这一遍的重点是 C9/C10/C11——`module avail lang/Python`、`system/CUDA`、
`numlib/cuDNN` 和 `module spider PyTorch` 的输出，**这是定 `MT_MODULES` 唯一
需要的东西**。

**⑥ 退出交互作业**（别忘了，占着资源会计费）

```bash
exit
```

**⑦ 把两份结果取回来**

```bash
scp -P 8022 "<你的login>@access-iris.uni.lu:~/hpc_probe_*.txt" .
```

或者直接在终端里 `cat` 出来复制。两份都要。

**可能遇到的情况**

| 现象 | 说明 |
|---|---|
| 第一遍 C8 说没有 `module` | 预期行为，继续第④步 |
| `salloc` 一直排队 | `interactive` 只有 2 节点，可能被占。等一下，或试 ULHPC 的 `si` 封装函数 |
| `salloc: error: Invalid qos specification` | `interactive` 只收 `--qos=debug`，不是 `normal` |
| 终端卡死、Ctrl-C 无效 | 多半是某个命令在网络文件系统上进了不可中断 I/O（D 状态）。新开一个 ssh 连接 `pkill -9 -u $USER <命令名>`，或直接关掉重连——探测脚本只读，不会写坏任何东西。脚本现在给所有查询都套了 `timeout 20` |
| `module avail` 输出被截断 | 脚本每段只取前 14 行。若关键版本没露出来，单独跑 `module avail lang/Python` 看全 |
| `curl` 超时 | 说明登录节点没外网 ⇒ 模型只能从本机 rsync（§3.3） |

---

## 2. module 是什么

（这一节是给完全没用过 module 的人写的。）

### 2.1 一句话

**module 只做一件事：修改当前 shell 的环境变量**（`PATH`、`LD_LIBRARY_PATH`、
`CPATH`、`PKG_CONFIG_PATH` 等），让某个预装在集群公共目录里的软件"变得可见"。

它**不是包管理器**。它不下载、不编译、不装 Python 包。集群管理员事先把
CUDA 12.4、Python 3.10、GCC 13 等装在 `/opt/software/...` 之类的地方，
`module load` 就是把对应的 `bin/` 加进 `PATH`、`lib/` 加进 `LD_LIBRARY_PATH`。

### 2.2 六个命令

```bash
module avail                 # 列出当前可见的所有模块
module spider cuda           # 全局搜索（层级式 Lmod 里，有些模块要先 load
                             #   编译器才可见，avail 看不到，spider 看得到）
module load cuda/12.4.0      # 加载。永远写全版本号，不要写裸 `module load cuda`
module list                  # 看当前加载了什么
module show cuda/12.4.0      # 看它到底改了哪些环境变量（排错必用）
module purge                 # 卸载全部，回到干净状态
```

`module avail` 输出可能几百行，用 `module avail 2>&1 | grep -i <关键词>`
过滤（注意 module 把输出写到 **stderr**，所以要 `2>&1`）。

### 2.3 module + venv 怎么配合，以及那个坑

分工很清楚：

| 层 | 由谁提供 | 本项目需要什么 |
|---|---|---|
| 系统层 | module | Python 3.10 解释器、CUDA/cuDNN 运行时 |
| 应用层 | venv (`pip`/`uv`) | torch、transformers、peft、bitsandbytes、本项目 `mt` 包 |

**坑在于：venv 记录的是创建它的那个 python 的绝对路径。**
`.venv/pyvenv.cfg` 里有一行 `home = /opt/software/Python/3.10.13/bin`。
如果下次登录忘了 `module load python/3.10.13`，或者集群把这个版本下线了，
`.venv/bin/python` 就指向一个不存在的文件，报 "bad interpreter" 或者更隐蔽地
用错解释器。

⇒ **铁律**：任何 sbatch 脚本、任何交互式会话，顺序都必须是

```bash
module purge
module load python/3.10.x  cuda/12.x  cudnn/8.x     # 版本号写死，不许浮动
source .venv/bin/activate
```

**先 purge，再 load，最后 activate。顺序不可反。**
`module purge` 是必要的——登录 shell 可能有站点默认加载的模块，不清掉会和
你 load 的版本打架。

### 2.4 ULHPC 特有的两层结构（实测踩过）

**(a) 模块树按节点架构分。** GPU 节点上的 `MODULEPATH` 是：

```
/opt/apps/easybuild/systems/iris/rhel810-20250803/2023b/gpu/modules/all
/opt/apps/easybuild/systems/iris/rhel810-20250803/2023b/skylake/modules/all
/opt/apps/easybuild/systems/binary/rhel810-20250803/2023b/generic/modules/all
```

而 `batch` 节点上是 `.../2023b/broadwell/modules/all`，**没有那个 `gpu` 树**
——所以 `system/CUDA`、`numlib/cuDNN`、带 CUDA 的 PyTorch 在 CPU 节点上
全都查不到。⇒ **任何 module 探测和安装都必须在 GPU 节点上做。**

**(b) `env/release/<工具链>` 是个网关。** 很多模块 `spider` 能搜到，
直接 `module load` 却报

```
Lmod has detected the following error: These module(s) or extension(s)
exist but cannot be loaded as requested: "ai/PyTorch/2.6.0-foss-2024a"
```

原因是它属于另一个工具链树，要先加载对应的 `env/release/*` 才可见：

```bash
module purge
module load env/release/2023b      # 或 env/release/default
module load ai/PyTorch/2.3.0-foss-2023b-CUDA-12.6.0
```

`module spider <全名>` 会直接告诉你需要先加载哪个 `env/release/*`——
遇到这个报错第一件事就是 spider 全名看提示。

---

## 3. 存储与配额

### 3.1 需求量

| 内容 | 大小 | 说明 |
|---|---|---|
| `marcelbinz/Llama-3.1-Minitaur-8B` HF 快照 | **15 GB** | 本次唯一需要的模型，非 gated |
| `data/psych-101-test/prompts_testing_t1.jsonl` | 92 MB | 唯一的打分输入 |
| 代码仓库 | < 50 MB | |
| 每份全量 score CSV | 40–70 MB | 本次验证只产生小文件 |
| （将来）`meta-llama/Llama-3.1-8B` | ~16 GB | **gated**，本次不下 |
| （将来）`marcelbinz/Llama-3.1-Centaur-70B` | **~140 GB** | 见下面的警告 |

> ⚠️ **NF4 省的是显存，不是磁盘。** `--load 4bit` 走的是
> "下载完整 BF16 权重 → 加载时量化"（`_common.py:227-228` 的
> `BitsAndBytesConfig`），HF 上没有预量化的 4bit 权重。所以 70B NF4 虽然只占
> **~40 GB 显存**，却要先在磁盘上放下 **~140 GB** 的原始检查点。
> 上 70B 之前必须先确认 C15 的大空间够——这往往比显存更早成为瓶颈。

### 3.2 HF 缓存位置

**空间不是问题**：ULHPC 的价目表写明 home **500 GB 免费**、scratch
**10 TB 免费**、project 1 TB 免费。15 GB 的模型放哪都够。真正要确认的只剩
**inode（文件数）上限**（C14）——一个 venv 有几千个小文件，空间够而 inode
用尽是 HPC 上很常见的翻车方式。

仓库现在**任何地方都没有引用 `HF_HOME`**，默认会往 `~/.cache/huggingface`
塞。`hpc_env.sh` 已经把它默认指到 `$SCRATCH/hf`。选放哪时权衡：
home 有 SSD 缓存、小文件随机读快，但 scratch 空间大 20 倍且都无备份。

```bash
# 写进 ~/.bashrc 和每个 sbatch 脚本
export HF_HOME=/scratch/$USER/hf          # 或其它大空间
export HF_HUB_OFFLINE=1                   # 计算节点断网时必须
```

注意：`TRANSFORMERS_CACHE` 在 transformers v5 / huggingface_hub v1 里已废弃，
**用 `HF_HOME`**。如果 scratch 会定期清理（C16），模型要么放 home、要么接受
每次重传。

### 3.3 把模型和数据弄上去

**实测：登录节点和计算节点都有外网**（`huggingface.co` / `pypi.org` 均 HTTP 200，
`pip install` 在 GPU 交互作业里直接成功）。所以模型直接在交互作业里下即可，
不需要 rsync 15 GB。

⚠️ 下载前注意 `hpc_env.sh` 会设 `HF_HUB_OFFLINE=1`（跑作业时要的），
**下载那一步必须先 `unset HF_HUB_OFFLINE`**，否则 `hf download` 会拒绝联网。

**下模型（非 gated，不需要登录）**：

```bash
export HF_HOME=/scratch/$USER/hf
hf download marcelbinz/Llama-3.1-Minitaur-8B
hf download --repo-type dataset marcelbinz/Psych-101-test   # 需先在 HF 网页同意条款
```

**若登录节点断网**——从本机 rsync。**这里有个坑**：HF cache 的
`snapshots/<sha>/model-0000x.safetensors` 是**指向 `blobs/` 的符号链接**。

```bash
# 正确：-a 保留符号链接结构（15 GB）
rsync -avP ~/.cache/huggingface/hub/models--marcelbinz--Llama-3.1-Minitaur-8B \
  <user>@<login>:/scratch/<user>/hf/hub/

# 数据文件
rsync -avP data/psych-101-test/prompts_testing_t1.jsonl \
  <user>@<login>:~/mt/data/psych-101-test/
```

不要用裸 `rsync -r`（不保留链接，会传成一堆空文件或直接失败）。
若目标文件系统不支持符号链接，改用 `rsync -avPL`（解引用），但会占双倍空间。

传完后用 `preflight.py`（§4.3）验证缓存完整性——它会读
`*.safetensors.index.json` 逐个核对分片是否存在且非空。

---

## 4. 环境安装

### 4.1 一个必须先讲清楚的矛盾

`docs/centaur-eval-handoff.md:764` 写的 HPC 装法是
"保留集群模块提供的 CUDA/PyTorch，用 `uv pip install -e . --no-deps` 装本项目"。
这条**单独一条不够**，有两个原因：

1. `scripts/experiments/preflight.py:60-67` **硬性要求 transformers ≥ 5**
   （runner 用了 v5 才有的 `dtype=` 加载参数，见 `_common.py:210`）。
   集群模块基本不可能提供 transformers 5.x → 应用层必须自己装。
2. `uv.lock` 锁的 `torch 2.12.0` 依赖 `nvidia-*-cu13`，即 **CUDA 13 构建**。
   CUDA 13 已经放弃 Volta (sm_70) 的离线编译 → 直接 `uv sync` 装出来的 torch
   **大概率跑不了 V100**。

所以判据不是"能不能 `import torch`"，而是：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda); \
           print(torch.cuda.get_arch_list())"
```

**输出里必须有 `sm_70`。** 没有就说明这个 torch 编译时没带 Volta 内核，
在 V100 上要么直接报 "no kernel image is available for execution on the device"，
要么走 PTX JIT（慢且不保证成功）。

### 4.1b 探测后新增的两个障碍

**(a) EasyBuild 模块树是按 CPU 架构分的。** 实测到的 `MODULEPATH` 形如

```
/opt/apps/easybuild/systems/iris/rhel810-20250803/2023b/broadwell/modules/all
                                                          ^^^^^^^^^
```

`batch` 节点是 broadwell，**GPU 节点是 skylake**——两者看到的模块集不同。
⇒ **所有 module 相关的探测和安装都必须在 GPU 节点（或至少 skylake 节点）上做。**
第一次在 broadwell 上探测时 `CUDA` / `cuDNN` 都是 "No module found"，
这**不能**说明 GPU 节点上也没有。

**(b) broadwell 树只有 Python 3.11.5，而本项目锁的是 `>=3.10,<3.11`**
（`pyproject.toml:10`；`uv.lock` 更严，是 `==3.10.*`）。两条出路：

1. **先在 GPU 节点上 `module spider Python`** 看有没有 3.10——不同工具链树
   （2023b / 2024a / …）带的 Python 版本不同，很可能有。
2. 没有的话，把 `requires-python` 放宽到 `>=3.10,<3.12`。这个上界看不出技术
   理由（代码没有 3.10-only 语法，ruff 的 `target-version = py310` 只影响 lint
   规则，torch/transformers 都支持 3.11），但**这是改动全项目的配置**，
   动之前要确认本地 5060 Ti 和 Mac 环境不会因此漂移。

### 4.1c 实测结论：集群 PyTorch 有 sm_70，但版本太老

GPU 节点上实测（`env/release/2023b` + `ai/PyTorch/2.3.0-foss-2023b-CUDA-12.6.0`）：

```
python 3.11.5
torch 2.3.0  cuda 12.6
arch ['sm_70']
available True
```

**arch list 只有 `sm_70`**——ULHPC 是专门为自己的 V100 机队编的。CUDA 12.6
不是 13，Volta 完整受支持。这一条彻底排除了 §4.1 第 2 点的风险。

**但 torch 2.3.0 用不了**：`src/mt/evaluation/transcript_scoring.py:203-213`
的 `_cuda_sdpa_context` 依赖

```python
backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
return sdpa_kernel(backends, set_priority=True)
```

`SDPBackend.CUDNN_ATTENTION`、`set_priority=` 以及"传一个 backend 列表"都是
torch 2.5/2.6 之后才有的。这是**每次 CUDA 打分都会经过的热路径**。
⇒ **走路线 B（自己装 torch）**，目标版本 **≥2.6 且 CUDA 12.x**（12.x 才带
Volta；13 已经放弃）。装完仍然要用 §4.1 的判据复验 `sm_70`。

> 想省掉这一步的话，另一条路是把 `_cuda_sdpa_context` 按 torch 版本做降级
> 兼容。但那是为了迁就一个 2024 年的 torch 而改科学代码的热路径，
> 不划算——除非路线 B 也失败。

**Python 版本：已解决。** transformers 5.10.2 和 peft 0.19.1 的
`Requires-Python` 都是 `>=3.10.0` **无上界**，torch 也没有硬约束，所以
3.11.5 对依赖侧没问题。唯一的阻塞是本项目自己的上界，已放宽：

```toml
requires-python = ">=3.10,<3.12"     # pyproject.toml:10
```

这是**放宽**，本地 Mac（3.10.20）和 5060 Ti 环境都不受影响。`uv lock` 重新
生成后**零个包版本变动**——只有 `requires-python` 和 resolution markers 变了。
`AGENTS.md` 已同步。

> 顺带澄清一处历史说法：handoff 里"HPC 用 `uv pip install -e . --no-deps` +
> 集群 torch"的方案，在本集群上不成立（集群 torch 2.3.0 太老）。`uv pip
> install` 不读 `uv.lock`，全项目也没有一处用 `uv sync`，所以锁文件对实际
> 安装流程是惰性的。

> **好消息：没有 CUDA 模块并不致命。** 现代 PyTorch 的 pip wheel
> **自带 CUDA 运行时**（以 `nvidia-*-cu12` 一系列包的形式装进 venv），
> 节点上只需要 NVIDIA **驱动**，而驱动总是有的。所以 §4.2 的**路线 B 根本不
> 依赖任何 CUDA module**。只有路线 A（复用集群 torch）才需要，而且还要额外
> 验证集群那个 `ai/PyTorch/*-foss-*` 到底带没带 CUDA——`foss` 工具链
> （GCC+OpenMPI+OpenBLAS）编出来通常是 **CPU-only**，带 CUDA 的 EasyBuild
> 命名一般会有 `-CUDA-12.x` 后缀。

### 4.2 安装步骤（路线 B，已定）

§4.1c 已经排除了路线 A（集群 torch 2.3.0 太老）。下面是实际要敲的。
**全部在一个 GPU 交互作业里做**——登录节点没有 `module`，CPU 节点没有 GPU 树。

```bash
salloc -p interactive --qos=debug -N 1 --ntasks-per-node=1 -c 7 -G 1 \
       -C volta16 --time=2:00:00
```

**① 只加载 Python，不加载集群 torch**

```bash
module purge
module load env/release/2023b
module load lang/Python/3.11.5-GCCcore-13.2.0
python --version        # 期望 Python 3.11.5
```

不需要 `system/CUDA` 或 `numlib/cuDNN`：pip 的 torch wheel **自带 CUDA 运行时**
（`nvidia-*-cu12` 那一堆包），节点上只要有 NVIDIA 驱动就够（实测 580.159.04）。

**② 建 venv 并装 torch —— 这一步是卡点**

```bash
cd ~/mt
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

选 **cu126** 而不是默认源：默认源现在给的是 CUDA 13 构建，**已放弃 Volta**。
需要的是 **torch ≥2.6**（`sdpa_kernel(..., set_priority=True)` 和
`SDPBackend.CUDNN_ATTENTION` 从 2.5/2.6 才有）**且 CUDA 12.x**。

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_arch_list())"
```

**判据：arch list 必须含 `sm_70`，torch 版本必须 ≥2.6。** 两者缺一就换版本重来
（可以显式指定，例如 `pip install "torch==2.7.*" --index-url .../cu126`）。

**③ 装应用层**

```bash
pip install -e ".[dev]"
```

`pyproject.toml` 已放宽到 `<3.12`，3.11.5 能过。这一步会按 lock 之外的解析
拉 transformers/peft/pandas 等；**注意别让它把 torch 顶掉**——装完复验一次：

```bash
python -c "import torch, transformers, peft; print(torch.__version__, transformers.__version__, peft.__version__); print(torch.cuda.get_arch_list())"
```

若 torch 被换成了 CUDA 13 版本，改用 `pip install --no-deps -e .` 再手工补齐
`transformers>=5.8 peft>=0.19 datasets>=4.8 pandas numpy matplotlib tqdm schedulefree`。

**④ bitsandbytes（走 4bit 才需要，但这是 70B 路线的总闸门）**

```bash
pip install "bitsandbytes>=0.49.2"
python -c "import bitsandbytes, torch; print(bitsandbytes.__version__); import torch; x=torch.zeros(1).cuda(); print('cuda ok', x.device)"
```

> **注意：V100 上 4bit 不是"更好的跑法"，只是"能对拍的跑法"。** 见 §4.4。
> 它能不能在 sm_70 上跑是 §5.6 说的总闸门，装上了也要等 L1 实测才算数。

### 4.2b 装到哪、哪些节点用得上

**pip 装的一切都进 `~/mt/.venv/`，而家目录是全局的。** ULHPC 官方原话是
"All UL HPC systems use global home directories"——`/home/users/$USER`（GPFS）
和 `/scratch/users/$USER`（Lustre）都经 Infiniband 挂在**每个节点**上。
⇒ **装一次，所有作业都能用**，不需要每个计算节点重装；登录节点也看得见同一份
（只是那里没有 `module`，跑不起来）。

放置策略（和 §3.2 的选择一致）：

| 内容 | 位置 | 理由 |
|---|---|---|
| `.venv/`（数万个小文件，约 6 GB） | **`$HOME`** | 官方：home 有 SSD 缓存，"significantly faster for random and small file I/O" |
| HF 模型快照（15 GB，少量大文件） | **`$SCRATCH`** | Lustre 适合大文件；`hpc_env.sh` 已默认 `$SCRATCH/hf` |
| pip 缓存 | `$SCRATCH` | 默认在 `~/.cache/pip`，会吃 home 配额 |

```bash
export PIP_CACHE_DIR="$SCRATCH/pipcache"     # 建议写进 ~/.bashrc
```

**⚠️ 已确认：venv 绑死在 skylake 上。**
software 目录和 module 树一样按架构分——GPU 节点上实测：

```
/mnt/aiongpfs/apps/easybuild/systems/iris/rhel810-20250803/2023b/skylake/
    software/Python/3.11.5-GCCcore-13.2.0/bin/python3.11
```

文件在共享的 `aiongpfs` 上，每个节点都**看得见**；问题不是路径，是**指令集**：
skylake 编出来的二进制带 AVX-512，而 `batch` 分区有一半是 broadwell
（iris-[001-108]，只有 AVX2），跑上去会 `Illegal instruction (core dumped)`。

⇒ **纪律：凡是用到这个 venv 的作业，都必须落在 skylake 节点上。**

| 作业 | 怎么保证 |
|---|---|
| GPU 作业（L1/L2/L3、正式打分） | GPU 节点本身 feature 就含 `skylake`，**自动满足** |
| `merge_shards.slurm`（`batch` 分区） | 已加 `#SBATCH --constraint=skylake` → iris-[109-168] |
| `interactive` 交互作业 | salloc 要加 `-C skylake`（本文档所有 salloc 已加） |
| 将来的 CPU 作业（E2 基线、画图） | 同样要 `-C skylake` |

> 想彻底躲开这个约束，可以改在 broadwell 节点上建 venv（AVX2 二进制在
> skylake 上也能跑），但那会让 GPU 作业损失 AVX-512。我们的负载是 GPU-bound，
> CPU 侧只做 I/O 和 CSV 拼接，其实两种都行——**选 skylake 是因为 GPU 作业
> 占绝对多数，别让主作业将就次要作业**。

**⑤ `MT_MODULES` —— 已经填好，不用改**

`scripts/hpc_env.sh` 里现在是：

```bash
MT_MODULES="${MT_MODULES:-env/release/2023b lang/Python/3.11.5-GCCcore-13.2.0}"
```

版本写死，以后所有 sbatch 脚本靠它复现同一个解释器（§2.3 的铁律）。
注意里面**没有** `ai/PyTorch`（torch 走 pip，理由见 §4.1c），
也**没有** `system/CUDA` / `numlib/cuDNN`（pip wheel 自带运行时）。

> **常见错误：`Package 'mt' requires a different Python: 3.11.5 not in
> '<3.11,>=3.10'`** —— 集群上那份 `pyproject.toml` 还是旧的。本仓库已把上界
> 放宽到 `<3.12`（§4.1c），把改动同步到集群即可；来不及提交时先就地改：
> `sed -i 's/>=3.10,<3.11/>=3.10,<3.12/' pyproject.toml`

### 4.3 安装验收

⚠️ **整个 §4.2 的安装过程都不能在登录节点做**——ULHPC 刻意在 access/login
节点上禁用了 `module`，没有 module 就没有 Python 解释器可以建 venv。
先开一个交互作业：

```bash
salloc -p interactive --qos=debug -N 1 --ntasks-per-node=1 -c 4 -C skylake --time=1:00:00
```

`interactive` 分区上限 2h、优先级高，正是为这种事准备的。装完再退出。
（ULHPC 还提供了 `si` / `si-gpu` 之类的封装函数，效果一样，更短。）

装完在同一个交互作业里验收：

```bash
cd ~/mt && source scripts/hpc_env.sh
python scripts/experiments/preflight.py \
  --model marcelbinz/Llama-3.1-Minitaur-8B \
  --data data/psych-101-test/prompts_testing_t1.jsonl
```

期望 `PREFLIGHT PASSED`。它检查：torch/transformers/pandas 可导入、
transformers 主版本 ≥5、`mt` 包可导入、数据文件可解析且每个 session 都有非空
`<<>>` 标记、切分无损（E3 前提）、**HF 缓存分片完整**、tokenizer span 映射正常、
输出目录可写、GPU 报告。

`interactive` 分区默认不带 GPU，所以会打印
`note: no GPU visible`，**这不算失败**。但**此时不要加 `--load 4bit`**——
那一支会检查 `torch.cuda.is_available()` 并判失败。bitsandbytes 在 sm_70 上
的实机验证放在 L1 作业里（§4.4 解释了为什么那是总闸门）。

### 4.4 为什么在 V100 上仍然主用 NF4

**前提：本项目在这台集群上，显存是硬约束，时间不是。** 不拆小就根本装不下更大的
模型，慢一点可以接受。下面的取舍全部建立在这个前提上。

**V100 没有 INT8/INT4 tensor core**（那是 Turing sm_75 才引入的），只有 FP16
tensor core。由此：

- **`--load 8bit` 不要用。** `LLM.int8()` 依赖 int8 tensor core，sm_70 上没有。
- **`--load 4bit`（NF4）可以用。** 它不是 int4 矩阵乘：权重按 4 bit 存，
  算之前逐块**反量化成 `bnb_4bit_compute_dtype`（fp16）**，再做标准 fp16 GEMM
  （`_common.py:220-223`）。不需要 int4 硬件。

**代价要认清**：我们的打分负载是 `use_cache=False` 的大 prefill，属**计算密集**；
量化真正提速的是**访存密集**场景（逐 token 生成）。所以 NF4 在这里
**没有速度收益，还要付每次前向的反量化开销**。省的纯粹是显存——而这正是我们要的。

**装得下装不下（单作业上限 1 节点 = 4 GPU）**：

| 配置 | 权重 | 4×16G = 64 GB | 4×32G = 128 GB |
|---|---|---|---|
| 8B FP16 | ~16 GB | ✗ 单卡装不下 | ✓ 单卡 |
| 8B NF4 | 5.7 GB | ✓ 单卡 | ✓ 单卡 |
| 70B FP16 | ~140 GB | ✗ | **✗ 超出 128 GB** |
| 70B NF4 | ~40 GB | ✓ 需 4 卡 | ✓ **2 卡即可** |

⇒ **NF4 不是权宜之计，是这台集群上唯一能碰 70B 的路。** 70B FP16 在物理上
就做不到（单作业最多一个节点）。

⇒ **由此，L1 的真正意义不是"跑通一个小实验"，而是验证 bitsandbytes NF4 在
sm_70 上到底能不能用。** 这是后续所有"更大模型"计划的总闸门：
`bitsandbytes` 近几个版本一直在抬高最低架构要求，0.49.x 是否还发 sm_70 kernel
必须实测，不能假设。

### 4.5 训练侧的现状（本次不跑，但要记录）

`mt-finetune-llm`（`src/mt/models/llm/finetuning.py`）已经是标准 QLoRA：

- `--load-in-4bit` + `--bnb-4bit-quant-type nf4` + `prepare_model_for_kbit_training`
  （:184-188）
- **`--gradient-checkpointing` 默认就是 `True`**（:62），且正确调用了
  `enable_input_require_grads()`——这是训练显存最大的一根杠杆，已经拉下来了
- `--batch-size` 默认 1、`--gradient-accumulation-steps` 16、`--seq-len` 4000，
  已经是保守配置

**但有一个硬阻塞**：`finetuning.py:171` 写的是

```python
model_kwargs["device_map"] = {"": local_rank} if torch.cuda.is_available() else None
```

`{"": local_rank}` 是 DDP 的写法——**把整个模型钉在一张卡上**，每个 rank 一份
完整副本。而仓库里没有任何 torchrun / accelerate / deepspeed 启动器，
`LOCAL_RANK` 恒为 0 ⇒ **当前训练只能用一张卡，模型再大也拆不开。**

要在 4 张卡上训练一个装不下的模型，需要改代码，两条路：

1. **最小改动**：`{"": local_rank}` → `"auto"`。得到 naive pipeline parallel，
   HF Trainer 原生支持（会识别 `model.is_parallelizable`）。慢，但装得下——
   符合"显存优先于时间"的取舍。
2. **正规做法**：FSDP 或 DeepSpeed ZeRO-3 + `torchrun` 启动器。快得多，
   但要引入启动器、配置文件和一整套调试。

本次不做，但这是"训练更大模型"的**真正前置项**，应该记进
`docs/centaur-eval-handoff.md` §7 的待办里。

### 4.6 训练显存估算

**先说结论：2B 和 7B 不一样，但差距是 ~2 倍，不是 3.5 倍。** 因为训练显存里
有一大块**根本不随模型大小变化**。

按仓库当前默认值估算：`--load-in-4bit`、`--lora-r 8`、7 个 projection 全上、
`--seq-len 4000`、`--batch-size 1`、`--gradient-checkpointing`（默认开）、
双重量化开。

| 组成 | ~2B | ~8B (Llama-3.1) | 是否随模型规模变 |
|---|---|---|---|
| NF4 冻结权重 | ~1.1 GB | **~5.7 GB** | **是**，线性 |
| LoRA 参数 + 梯度 + AdamW 状态 | ~0.13 GB | ~0.34 GB | 是，但都很小 |
| 检查点激活 | ~0.5 GB | ~1.0 GB | 是，但次线性 |
| **logits + 交叉熵** | **~2.4–3.6 GB** | **~3–4 GB** | **否！** 只看词表×序列长×batch |
| workspace / 碎片 / CUDA 上下文 | ~0.8 GB | ~1.0–1.5 GB | 略微 |
| **合计** | **~5–6.5 GB** | **~11–14 GB** | |

几个值得注意的点：

- **权重那一行是唯一严格线性的**，而且 8B 的 5.7 GB 是**实测值**
  （handoff 记的推理常驻 5.68 GiB），不是估的——所以这一行可信。
- **logits 那一行是"隐形大户"，且与模型大小无关。** Llama-3.1 词表 128,256，
  seq_len 4000 时仅 fp16 logits 就是 1.03 GB，交叉熵通常还要升到 fp32 再留
  梯度缓冲，实际 3–4 GB。**这就是 2B 省不下 4 倍的原因**——甚至反过来：
  Gemma 系列词表 256k，一个 2B 模型的 logits 比 8B Llama 还大。
- LoRA 那一行小到可以忽略：r=8 全 projection 在 8B 上只有 **~21 M 可训练参数
  （0.26%）**，参数+梯度+优化器状态加起来约 340 MB。
  **QLoRA 的省法不在优化器，在"冻结的基座不产生梯度和优化器状态"。**

**对比：如果做全量微调**（本仓库不支持，仅作参照）——混合精度 AdamW 约
**16 字节/参数**：2B → ~32 GB，8B → ~128 GB。所以在 V100 上全量微调 8B
是彻底不可能的，QLoRA 不是"省一点"，是"从不可能变成可能"。

**真正决定显存的四个旋钮，按影响排序：**

| 旋钮 | 影响 | 说明 |
|---|---|---|
| `--seq-len` | **最大**。激活和 logits 都线性 | 4000→2000 能让 8B 从 ~12 GB 降到 ~8 GB |
| `--gradient-checkpointing` | **关掉会炸** | 关掉后激活从"每层边界存一份"变成"全部保留"，8B 上多出 10 GB 量级。**V100 上永远不要关** |
| `--batch-size` | 激活侧全部线性 | 这就是默认 `batch=1 + accum=16` 的原因：等效 batch 16 而显存只按 1 算 |
| 词表大小 | 决定 logits 那一行 | 由选哪个模型决定，不可调 |

**外推到 70B QLoRA**：40 GB 权重 + ~5 GB 检查点激活 + ~4 GB logits + ~1 GB
LoRA 优化器 + workspace ≈ **50–55 GB** ⇒ 需要 **2×32G**，
但**当前被 §4.5 的 `device_map={"": local_rank}` 卡死**，必须先改代码。

> 以上除权重行外都是估算，误差约 ±30%。真实数字用仓库自带的
> `mt-monitor-hardware`（`src/mt/utils/hardware_monitor.py`）在小步数试跑时
> 实测，填进 §9。

---

## 5. 分级验证

四级，逐级加码，每级都有**明确的通过判据**。核心思路：**用本地已经跑完的结果
做数值对拍**，而不是"跑起来没报错就算过"。

### 5.1 对拍锚点（本地已有）

| 用途 | 本地文件 | 锚点 |
|---|---|---|
| L1 | `outputs/scoring/minitaur8b_e0_full_4bit.csv` | `kool2016when/exp1.csv` = 20 sessions / 2407 choice 行 |
| L2 | `outputs/scoring/minitaur8b_e0_longest_4bit_cudnn_probe_summary.csv` | `xiong2023neural/exp1.csv` participant **28**，4800 choices，**paper_token_nll = 0.472271** |
| L3 | `outputs/scoring/minitaur8b_e3_e0grid5_4bit.csv` | 全量 5 锚点窗口网格 |

三份都是 **`--load 4bit` / Minitaur-8B / NF4** 配置，所以集群上对拍也必须
`--load 4bit`。NF4 kernel 在 sm_70 与 Blackwell 上实现不同，**不可能逐位相同**，
判据是容差不是相等。

对拍工具：`scripts/experiments/compare_scoring.py`（本次新增）。它按
`(experiment, participant, choice_index)` 对齐（E3 自动加上 `window` 和
`target_index`），报行数差、`max|Δnll|`、Pearson r 和只在一侧出现的 session，
超出容差就以非零码退出。默认只判**交集**，所以拿一个只跑了一个 experiment 的
集群结果去比全量本地基线是正常用法；要求两边覆盖完全一致时加 `--strict`。

### 5.2 涉及的脚本

| 文件 | 作用 |
|---|---|
| `scripts/hpc_env.sh` | **所有站点相关设置的唯一入口**：module 列表、`HF_HOME`、显存监控函数。上机后只改这一个文件 |
| `scripts/smoke_e0_e3.slurm` | L1 + L2（含对拍）；`MT_LOAD=none` 切到 FP16 臂 |
| `scripts/e0_e3_minitaur.slurm` | L3 与将来的正式作业；`MT_LIMIT=200` 切到 L3 的有界版本 |
| `scripts/merge_shards.slurm` | 分片合并（纯 CPU，独立作业） |
| `scripts/experiments/preflight.py` | 提交前自检（已有） |
| `scripts/experiments/compare_scoring.py` | 对拍判定（本次新增） |

`#SBATCH` 里的 `--partition` / `--qos` / `--gpus-per-task` / `--constraint`
无法从 `hpc_env.sh` 读取（SLURM 直接解析脚本文件本身），但这几行现在已经按
ULHPC 官方 AI/DL launcher 模板填好了：`-p gpu --qos=normal`、
`--gpus-per-task=N`、每张 GPU 配 7 核（节点 28 核的 1/4，官方硬性要求）。
`hpc_env.sh` 里的 `MT_MODULES` 也已经按实测填好（§4.2 ⑤）。

### 5.3 四级

| 级 | 内容 | 资源 | 通过判据 |
|---|---|---|---|
| **L0** | 登录节点：`preflight.py` + `pytest` | 0 GPU | `PREFLIGHT PASSED`；`pytest` 全绿（本仓库无 GPU 测试，登录节点可跑完） |
| **L1** | 1×V100，`kool2016when/exp1.csv`，`--load 4bit` | 30 min | 20 session 全部产出、`.failed.csv` 为空；对拍 `r > 0.9999` 且 `max\|Δnll\| < 1e-3` |
| **L2** | 1×V100，最长 session 探针（`--experiment xiong2023neural/exp1.csv --participant 28`），`--load 4bit` | 30 min | 不 OOM；`paper_token_nll` 对上 **0.472271**（同容差）；记录显存峰值 |
| **L3** | 4×V100 分片 + 主动 `scancel` + `--resume` 重提 + merge | 1 h | 4 个 shard 都在增长；重提后行数继续增加且 `(experiment, participant)` 无重复；merge 后 session 总数正确 |

**L2 为什么用单 session 探针而不是整个 experiment**：`xiong2023neural/exp1.csv`
的 participant 28 是全测试集最长的 session（53,091 token / 168,968 字符 /
4,800 choices），它才是显存上限的**真正判据**。跑这一个 session 比跑整个
experiment 更快、信息量更高。本地 NF4 下整卡峰值约 15.8 GiB——所以这一步同时
回答"16 GB 节点到底够不够"。

**提交命令**：

```bash
# L0（interactive 分区，不是登录节点——那里没有 module）
salloc -p interactive --qos=debug -N 1 --ntasks-per-node=1 -c 4 -C skylake --time=1:00:00
source scripts/hpc_env.sh
python scripts/experiments/preflight.py \
  --model marcelbinz/Llama-3.1-Minitaur-8B \
  --data data/psych-101-test/prompts_testing_t1.jsonl
pytest

# L1 + L2（NF4，16G 的 volta 节点即可）
sbatch scripts/smoke_e0_e3.slurm

# L1 + L2 的 FP16 对照臂（必须 32G）
sbatch --constraint=volta32 --export=ALL,MT_LOAD=none scripts/smoke_e0_e3.slurm

# L3（有界重跑测试，中途 scancel 后原样重提）
sbatch --time=01:00:00 --export=ALL,MT_LIMIT=200 scripts/e0_e3_minitaur.slurm
scancel <jobid>          # 跑 20 分钟后手动取消
sbatch --time=01:00:00 --export=ALL,MT_LIMIT=200 scripts/e0_e3_minitaur.slurm

# 合并（纯 CPU）
sbatch --export=ALL,MT_STEMS="outputs/scoring/hpc_e0_minitaur8b_4bit outputs/scoring/hpc_e3_minitaur8b_e0grid5_4bit" \
  scripts/merge_shards.slurm
```

**显存峰值**由 `hpc_env.sh` 的 `mt_watch_memory` / `mt_report_memory` 自动记录
（后台 `nvidia-smi` 轮询，按卡取最大值）。用**整卡**占用而不是
`torch.cuda.max_memory_allocated()`，因为后者不含 CUDA 上下文和 workspace，
而 16 GB / 32 GB 这条线卡的是整卡。

**还有一件必须自己确认的事**：**SDPA 实际用了哪个后端**。V100 是 sm_70，
FlashAttention 和 cuDNN attention 都要 sm_80+，会落到 mem-efficient 内核
（`transcript_scoring.py:204-213` 的优先级链）。**必须确认没掉到 math 路径**
——Windows 上掉过一次，GQA 直接请求 336 GiB。L1 的显存峰值如果远超预期，
第一个要查的就是这个。

### 5.4 附加臂：FP16 对照（32G 节点，一个 experiment）

在 L1 通过之后，用 **同一个 experiment**（`kool2016when/exp1.csv`）在 32G 节点上
再跑一次 `--load none`。这一臂有两个独立价值：

1. **量出 NF4↔FP16 的差**。同数据同模型，唯一变量是量化。这正是
   handoff 反复点名缺失的 "BF16/FP16 锚点" 的先导数据——现有全部结论
   都带着 "runtime NF4 ≠ BF16，尚无锚点" 的免责声明。
2. **量出 FP16 的显存峰值与吞吐**，用来判断将来 handoff §7 第 6 项的全量 FP16 对照要几个
   48h 窗口。

注意事项：

- **必须上 32GB 节点（191–196）**。FP16 的 8B 权重就 16 GB，16G 节点装不下。
- **不要传 `--dtype bf16`**：V100 无 bf16 单元。`resolve_dtype("auto","cuda")`
  已经返回 fp16（`_common.py:189-200`），保持 `--dtype auto` 即可。
  handoff §7 第 6 项说的 "原生 BF16/FP16 环境" 在 V100 上兑现为 **FP16**。
- 与本地 NF4 基线的对比走**宽容差**（§5.5），因为 NF4 vs FP16 本来就该有差。
  脚本已经按 `MT_LOAD` 自动切换容差，不用手动传。

### 5.5 两档容差

| 对比 | 容差 | 判定什么 |
|---|---|---|
| 集群 NF4 ↔ 本地 NF4 | `max\|Δnll\| < 1e-3`，`r > 0.9999` | **数值正确性**。同配置，差异只应来自 kernel 实现 |
| 集群 FP16 ↔ 本地 NF4 | `max\|Δnll\| < 0.02`，`r > 0.99` | **结构完整性**。差异本身是要测的量；这档只用来排除 tokenizer 错、span 映射错、加载错模型、attention 塌成 math 等**量级 0.1–1.0 nat** 的粗错 |

### 5.6 bitsandbytes 在 sm_70 上的总闸门 —— ✅ 已通过

这曾是整个计划最大的单点风险：跑不了就意味着这台集群上 **70B 完全没戏**
（70B FP16 需 140 GB > 4×32G 上限），"拆开加载更大模型"的整条路线不成立，
QLoRA 训练也一并受影响（§4.5）。

**2026-08-01 实测通过**，用的是直接调 NF4 内核的最小验证，不必等 L1：

```bash
python -c "
import torch, bitsandbytes as bnb
from bitsandbytes.nn import Linear4bit
print('bnb', bnb.__version__, '| gpu', torch.cuda.get_device_name(0))
lin = Linear4bit(2048, 2048, bias=False, compute_dtype=torch.float16, quant_type='nf4').cuda()
x = torch.randn(4, 2048, dtype=torch.float16, device='cuda')
y = lin(x)
print('NF4 matmul:', tuple(y.shape), y.dtype, '| finite:', bool(torch.isfinite(y).all()))
"
```

```
bnb 0.50.0 | gpu Tesla V100-SXM2-16GB
NF4 matmul: (4, 2048) torch.float16 | finite: True
```

⇒ §4.4 的取舍成立：NF4 是这台集群上唯一能碰 70B 的路，而它可用。

> 若将来换机器或升级 bnb 后这一条失效：先分清是**装不上**还是**装上跑不了**
> （分别记报错原文），试 `pip install bitsandbytes==0.50.*` 钉版本；
> 实在不行则退化为纯 FP16 单臂（§5.4）+ 宽容差（§5.5 第二行），
> 并重新评估全部 70B 计划。

---

## 6. 48 小时上限与断点续跑

### 6.1 现有机制

session 级续跑，够用但要理解边界：

- `--resume` 时 runner 读输出 CSV，把已完成的 `(experiment, participant)`
  跳过（`_common.py:156-167` 的 `completed_sessions`）。
- 每个 shard 有自己的 CSV，**各自独立续跑**，所以 4 卡作业超时后原样重提即可。
- E0 每 `--chunk-size`（默认 8）个 session 落盘一次；E3 **每个 session 落盘**
  ——对抢占更友好。
- **粒度是一个 session**：跑到一半被杀的 session 会从头重来。E3 的 full 窗口
  session 很长，这是实打实的浪费。

### 6.2 两个已知脆弱点（L3 要主动验一次）

1. **非原子写**。`append_records` 用 `to_csv(mode='a')`（`_common.py:174-180`），
   被 SIGKILL 可能留下半行；而 `completed_sessions` 只捕获 `EmptyDataError`，
   **不捕获 `ParserError`**。⇒ L3 里手工把某个 shard CSV 的最后一行截断，
   再带 `--resume` 提交，看是崩掉还是能自愈。崩掉就说明需要加固
   （最小改法：`completed_sessions` 加 `on_bad_lines='skip'` + 捕获
   `ParserError`）。
2. **`.failed.csv` 污染**。`--resume` 会把 `.failed.csv` 里的 session
   **永久跳过**（`run_transcript_scoring.py:87-90`）。本地就踩过：
   `minitaur8b_e0_full.failed.csv` 里存的是宿主内存导致的失败，不是真的超长
   （见 handoff §"续跑注意"）。
   ⇒ **纪律：集群上一律用全新的 output stem**，不要复用本地跑过的文件名。

### 6.3 时间预算

- **`gpu` 分区 2 天上限对我们是硬的**（`iris-gpu-long` 无权限）。所以任何
  全量作业都要按"多次提交接力"来规划，§6.1 的分片续跑是必需机制。
- `--time` **不要顶满 48h**，用 `1-23:00:00` 留一小时余量：最后一个 session
  被硬杀在半路是纯浪费，因为续跑粒度是整个 session。
- **merge 拆成了独立小作业**（`scripts/merge_shards.slurm`，跑在 `batch`
  分区）。原先它写在 `e0_e3_minitaur.slurm` 末尾，主作业一超时就永远执行
  不到，分片白攒。
- 排期时要算清楚：**总时长 ÷ 47h = 需要提交多少次**。这个数字要用 L1/L2
  实测的吞吐外推（§9），不能拍脑袋。

### 6.4 计费与 fairshare：为什么"多要一点"会反噬

ULHPC 用 Fair Tree 算法排优先级，**过去的用量会压低你未来的优先级**。
两条具体规则直接影响怎么写 `#SBATCH`：

**(1) GPU 按"GPU 等价数"计费，不是按你写的数字。** 一张 GPU 的配额是
**7 核 + 192 GB**（节点 4 GPU / 28 核 / 768 GB 的 1/4）。**超一点点就按整张
多余的 GPU 收费**——官方原话的例子是"请求 1 GPU + 8 核，按 2 GPU 计费"。

本仓库的脚本都正好卡在边界上，不要随手改大：

| 脚本 | GPU | 核 | 内存 | 判定 |
|---|---|---|---|---|
| `smoke_e0_e3.slurm` | 1 | 7 | 32G | ✓ 按 1 GPU |
| `e0_e3_minitaur.slurm` | 4 | 28 | 128G | ✓ 按 4 GPU（整节点） |
| `merge_shards.slurm` | – | 2 | 8G | ✓ `batch` 每核约 4.57 GB，2 核≈9.1 GB |

**(2) walltime 估不准会两头挨打。** 一方面它进 fairshare 的效率评分
（官方列的第一项就是 Average Walltime Accuracy），压低你的 share；另一方面
backfill 是靠"核数 × 内存 × 时间"三维找空隙塞任务的，**要得越准越容易被提前
插进去跑**。⇒ **不要习惯性顶格要时间。** L1/L2 存在的意义之一就是拿到能估准
的吞吐数，让后面的正式作业写得出诚实的 `--time`。

**(3) 两个该养成习惯的命令**：

```bash
seff <jobid>        # 作业完成后看 CPU/内存实际效率，直接回填 §9
ulhpcshare          # 看自己当前的 fairshare 分数
```

`seff` 报告的内存峰值可以用来把 `--mem` 调到刚好够——省下的既是钱也是优先级。

### 6.5 备选路径：`besteffort`

`normal` 排队太久、或者赶时间时还有一条路。`besteffort` QOS 的
**MaxWall 是 50 天、并发 300 作业**，代价是**会被任何其它 QOS 的作业随时打断**
（进程直接被杀，不是优雅退出）。

ULHPC 对它的要求原文是：用 besteffort 的可执行程序**必须自带
checkpoint-restart 机制**。⇒ **我们正好满足**：§6.1 的 session 级 `--resume`
就是这个机制，每个 shard 从自己的 CSV 续跑，被杀了重提即可。

```bash
sbatch --qos=besteffort scripts/e0_e3_minitaur.slurm
```

两条注意：

- 被打断的**当前 session 会整个重做**（续跑粒度是一整个 session），
  所以打断越频繁，浪费的比例越高。E3 每个 session 落一次盘，比 E0 的
  `--chunk-size 8` 更抗打断。
- §6.2 的**非原子写**风险在这里被放大：besteffort 是硬杀，正好是可能留下
  半行 CSV 的场景。用它之前先把 L3 里那个截断测试做掉。

本次的验证作业都很短（30–60 分钟），用不上；这一段是给将来的全量作业留的。

UL 内部工作免费，但 PI 会定期收到用量报告，GPU 的指示价是 **1.25€/GPU-hour**
（一个满节点跑满 47h ≈ 235€ 指示性）。不是账单，但值得心里有数。

---

## 7. 故障速查

| 症状 | 原因 | 处理 |
|---|---|---|
| `module: command not found` | **在登录节点上——ULHPC 刻意禁用的，不是故障** | 进 `interactive` 分区：`salloc -p interactive --qos=debug -N 1 --ntasks-per-node=1 -c 4 -C skylake --time=1:00:00` |
| 作业里 `module: command not found` | 脚本首行漏了 `-l` | 必须 `#!/usr/bin/bash --login`（四个脚本都已经是） |
| `bad interpreter` / venv 里的 python 打不开 | 没先 `module load` 创建 venv 时用的 python | 见 §2.3 铁律；实在乱了就删掉 `.venv` 重建 |
| `no kernel image is available for execution on the device` | torch 没编 sm_70 | 回 §4.2 决策树另一支 |
| `get_arch_list()` 不含 `sm_70` | 装到了 CUDA 13 wheel | 卸掉，装 cu12.x |
| bitsandbytes 报错 / 编译不了 | sm_70 支持问题 | 见 §5.6——这会阻断整条 70B 路线，要上报 |
| CUDA OOM | 长 session + 16GB 卡 | 加 `--constraint=volta32` 换 32GB 节点；或调小 `--batch-tokens`；或 `--max-chars` 设门限（会静默丢 session，需检查 `.skipped.csv`） |
| 显存请求几百 GiB | SDPA 掉到 math 路径 | 检查日志里的后端；确认 torch 版本支持 mem-efficient 内核 |
| `can't open file '.../scripts/experiments/preflight.py'` 或 `no .venv found at $HOME` | **`salloc` 也会导出 `SLURM_SUBMIT_DIR`**，记的是你敲 `salloc` 时所在的目录，之后 `cd` 到仓库并不改变它 | `hpc_env.sh` 已改成按 `$MT_ROOT` → `$(pwd)` → `$SLURM_SUBMIT_DIR` 依次挑第一个真的像仓库的；老版本上先 `unset SLURM_SUBMIT_DIR` |
| `Illegal instruction (core dumped)` | venv 建在 skylake 上，作业落到了 broadwell 节点 | 给作业加 `-C skylake`；见 §4.2b |
| `ModuleNotFoundError: _common` | 用了 `python -m` | 必须用路径调用 `python scripts/experiments/xxx.py`，且 CWD 在仓库根 |
| `GatedRepoError` / HTTP 403 | 模型需要许可 | 本次不涉及；将来跑 `meta-llama/Llama-3.1-8B` 要先在 HF 网页接受 Meta 许可并 `hf auth login` |
| 作业秒退、无日志 | `#SBATCH` 里的 partition/qos/account 名字不对 | 回 §1.2 核对 C1/C5/C6 |

---

## 8. 多卡怎么用：数据并行 vs 模型并行

### 8.1 两种并行不是一回事

| | **数据并行**（当前做法） | **模型并行**（`device_map="auto"`） |
|---|---|---|
| 每卡装什么 | 一份**完整**模型 | 模型的 1/4 层 |
| 怎么分工 | 每个进程打分 1/4 的 session | 一个 batch 依次流过 4 张卡 |
| 吞吐 | **≈ 4×** | **≈ 1×**（同一时刻只有一张卡在算，其余空转） |
| 解决什么问题 | 跑得快 | **装得下** |
| 前提 | 模型能塞进单卡 | 模型塞不进单卡 |

现有 4 卡作业走的是数据并行：4 个互不通信的独立进程，各自
`CUDA_VISIBLE_DEVICES=$i` + `--shard i/4`（见 `e0_e3_minitaur.slurm:51-79`）。

### 8.2 代码现状

- **`--load 4bit` / `8bit` 已经支持跨卡**：`_common.py:227-228` 用的就是
  `device_map="auto"`。只要**不设** `CUDA_VISIBLE_DEVICES`，它会自动把层铺到
  所有可见 GPU 上。不需要改任何代码。
- **`--load none`（FP16）是单卡**：`_common.py:209-211` 走
  `from_pretrained(...).to(device)`，没有 `device_map`。跨卡要改代码——改动很小
  （把 `.to(device)` 换成 `device_map="auto"`），但目前没改。

### 8.3 什么时候该拆

| 配置 | 权重占用 | 单卡放得下吗 | 结论 |
|---|---|---|---|
| 8B NF4 | 5.68 GiB | 16G ✓ 32G ✓ | **不要拆**。拆了吞吐 4×→1×，纯亏 |
| 8B FP16 | ~16 GB | 16G ✗ / 32G ✓（余量窄） | 上 32G 节点单卡跑；**只有**被迫用 16G 节点时才值得拆 |
| 70B NF4 | ~40 GB | ✗ | **必须拆**。今天就能跑：`--load 4bit` + 不设 `CUDA_VISIBLE_DEVICES`，2×32G 或 4×16G |
| 70B FP16 | ~140 GB | ✗ | **本集群做不到**。单作业上限 1 节点 = 4×32G = 128 GB < 140 GB。与代码无关 |

### 8.4 NVLink 的实际价值

`device_map="auto"` 给的是 **naive pipeline parallel**（按层切分、串行执行），
**不是 tensor parallel**。它在层边界只传 hidden state，数据量很小，PCIe 就够
——**这种模式下 NVLink 基本用不上**。

真正吃 NVLink 带宽的是 tensor parallel：每一层的矩阵乘都被切开，每层都要
all-reduce。那需要 vLLM / DeepSpeed / Megatron 之类的框架，本仓库没有，
也不在当前研究路线上（teacher-forced 打分是纯 prefill，不是生成，
vLLM 的优势发挥不出来）。

⇒ 原需求里"未来可以探索分布式部署更大的模型"是成立的，但要修正为：
**70B NF4 跨卡今天就能做且不需要改代码；NVLink 在这条路径上不是关键因素。**

### 8.5 顺带：CPU 节点是用得上的

原需求说"暂时想不到会用到 cpu 节点的地方"，但这几个全是纯 CPU、
完全不需要 GPU：

- `scripts/experiments/run_sequence_baselines.py`（E2 序列基线）
- `scripts/experiments/run_population_baselines.py`（E2-pop）
- 全部 `scripts/experiments/build_*_figures.py`（重画图）
- shard merge

把它们放 CPU 分区能省下 GPU 配额，排队也快得多。

---

## 9. 实测基线（上机后回填）

| 项目 | 实测值 | 日期 |
|---|---|---|
| 使用的 module 组合 | `env/release/2023b` + `lang/Python/3.11.5-GCCcore-13.2.0` | 2026-08-01 |
| Python | 3.11.5（EasyBuild skylake 树） | 2026-08-01 |
| torch / CUDA / arch list | **2.13.0+cu126** / 12.6 / `['sm_50','sm_60','sm_70','sm_75','sm_80','sm_86','sm_90']` | 2026-08-01 |
| transformers / peft | **5.14.1 / 0.20.0**（本地 5.10.2 / 0.19.1，见下方注） | 2026-08-01 |
| pandas | **3.0.5**（本地 2.3.3 —— 跨大版本，见下方注） | 2026-08-01 |
| L0（preflight + pytest） | **通过**：11 项检查全 ok，6561 sessions / 75 experiments，4 分片完整，72 tests 全绿 | 2026-08-01 |
| bitsandbytes | **0.50.0** 装成功，`torch.zeros(1).cuda()` 正常 | 2026-08-01 |
| **bitsandbytes NF4 内核在 sm_70 上** | **可用** —— `Linear4bit(2048,2048,nf4)` 前向在 Tesla V100-SXM2-16GB 上输出有限值 | 2026-08-01 |
| SDPA 实际后端 | | |
| L1 用时 / 显存峰值 | | |
| L2 用时 / 显存峰值 / `volta` (16G) 节点是否够 | | |
| L2 `paper_token_nll` 与 0.472271 的差 | | |
| 单 session 平均耗时（E0） | | |
| 全量 E0（6561 session）单卡外推 / 4 卡外推 | | |
| 全量 E3（5 锚点 × 7 窗口）外推 | | |
| $HOME / $SCRATCH 配额（空间 + inode） | | |
| FP16 臂：与 NF4 的差、显存峰值 | | |

> **集群与本地的版本差异**：集群解析到 transformers 5.14.1 / peft 0.20.0，
> 本地是 5.10.2 / 0.19.1（项目只声明 `>=5.8` / `>=0.19`，pip 每次都拿最新）。
> 更值得注意的是 **pandas 2.3.3 → 3.0.5 的跨大版本差异**（copy-on-write 与
> 字符串 dtype 的默认行为都变了）。项目只声明 `pandas>=2.3`，pip 每次拿最新。
> 已知 `read_csv(low_memory=False)` 在 3.0.5 上仍可用——`test_compare_scoring.py`
> 在集群上通过了。
>
> 这几处理论上都不影响 teacher-forced NLL，但它们是两边仅有的非受控差异——
> **L1 对拍若出现意料外的偏差，这是第一批要怀疑的地方**。真要排除就在集群上
> 钉死版本重跑一次。

最后几行的外推是本次验证真正的产出：`gpu` 分区 2 天封顶且没有长作业 QOS，
所以正式作业必须提前算出**要接力提交多少次**（总时长 ÷ 47h），
而这只能靠实测吞吐外推。

---

## 10. 作业脚本模板与自学指南

`scripts/template_gpu_job.slurm` 是一份**刻意最小化**的模板：没有错误检查、
没有进度打印、没有显存监控，只有资源请求、环境准备、代码调用。目的是把骨架
看清楚，之后能自己写新的分析脚本。

### 10.1 五段骨架

| 段 | 作用 | 本集群的硬约束 |
|---|---|---|
| ① `#SBATCH` 头 | 向 SLURM 要资源 | **必须在任何可执行语句之前**，否则被静默忽略 |
| ② `module` | 提供 Python 解释器 | 登录节点没有 `module`；`env/release/2023b` 是网关 |
| ③ `cd` + `activate` | 定位仓库、进 venv | 顺序不可反；venv 记录着创建它的 python 绝对路径 |
| ④ `export` | 缓存位置、线程数 | `HF_HOME` 别指 home；`HF_HUB_OFFLINE=1` |
| ⑤ `python ...` | 真正干活 | 必须**路径调用**，不能 `python -m` |

第一行 `#!/usr/bin/bash --login` 里的 `--login` 不能省——没有它，非交互 shell
不会 source `/etc/profile.d/`，`module` 就不存在。

### 10.2 写新脚本时要改什么

按这个顺序问自己五个问题：

| 问题 | 改哪一行 | 取值 |
|---|---|---|
| 要不要 GPU？ | `--partition` | 要 → `gpu`；纯 CPU（基线、画图、合并）→ `batch` |
| 要几张卡？ | `--gpus-per-task` | 配套改 `--cpus-per-task = 7 × 卡数`（官方要求每卡 1/4 节点核数） |
| 模型装得下 16G 吗？ | `--constraint` | NF4 的 8B → `volta16`；FP16 的 8B → **必须** `volta32` |
| 要跑多久？ | `--time` | **别顶格要**——估不准会压低 fairshare，还让 backfill 塞不进空隙（§6.4） |
| 内存要多少？ | `--mem` | GPU 分区每卡配额 192 GB，超了按多一张卡计费（§6.4） |

改完把第五段的 `python ...` 换成你要跑的东西即可。

### 10.3 三个常见变体

**四卡数据并行**（照搬 `e0_e3_minitaur.slurm` 的做法）：

```bash
#SBATCH --cpus-per-task=28
#SBATCH --gpus-per-task=4

for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i python scripts/experiments/run_transcript_scoring.py \
        --shard "$i/4" --resume --output "outputs/scoring/xxx_shard$i.csv" ... &
done
wait
```

四个进程各占一张卡、各跑 1/4 的 session，互不通信（**不走 NVLink**，见 §8）。
`--shard` + `--resume` 是断点续跑的基础。

**纯 CPU 作业**（E2 基线、画图、合并分片）：

```bash
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --constraint=skylake      # venv 是 skylake 编的，别落到 broadwell（§4.2b）
```

去掉 `--gpus-per-task` 和 `volta*` 约束即可。

**交互调试**（不写脚本，直接开一个会话手敲）：

```bash
salloc -p interactive --qos=debug -N 1 --ntasks-per-node=1 -c 7 -G 1 \
       -C volta16 --time=2:00:00
```

`interactive` 分区**只收 `--qos=debug`**，上限 2h，含 GPU 节点。
调试新脚本时比反复 sbatch 排队快得多。

> ⚠️ `salloc` 也会导出 `SLURM_SUBMIT_DIR`，记的是你敲 `salloc` 时的目录，
> 之后 `cd` 不改变它。模板第三段用 `$SLURM_SUBMIT_DIR` 在 sbatch 下是对的，
> 交互会话里要注意（`hpc_env.sh` 已按 `$(pwd)` 优先处理）。

### 10.4 提交与查看

```bash
cd ~/mt && sbatch scripts/template_gpu_job.slurm    # 必须先 cd 到仓库根
squeue -u $USER                                     # 看排队/运行
scancel <jobid>                                     # 取消
seff <jobid>                                        # 完成后看 CPU/内存实际效率
```

日志按 `%x-%j.out` 落在**提交目录**。`seff` 报的内存峰值可以用来把下次的
`--mem` 和 `--time` 调准——这既省钱也提高 fairshare（§6.4）。

### 10.5 模板和真实脚本差在哪

模板刻意省掉的东西，在真实作业里都是必要的，读的时候可以对照：

| 真实脚本有、模板没有 | 在哪 | 为什么需要 |
|---|---|---|
| `source scripts/hpc_env.sh` | 全部 `.slurm` | 把 ②③④ 三段收敛到一个文件，改一处生效全部 |
| `set -euo pipefail` | 全部 | 中间步骤失败时立刻停，而不是带着坏状态继续 |
| `--resume` + 分片 | `e0_e3_minitaur.slurm` | 48h 上限下唯一的接力手段（§6） |
| `trap ... USR1` | `e0_e3_minitaur.slurm` | 墙钟耗尽前 10 分钟记录进度 |
| 显存轮询 | `hpc_env.sh` | 整卡峰值只能外部采样 |
| `compare_scoring.py` 对拍 | `smoke_e0_e3.slurm` | 证明集群结果可信，不是"跑完没报错" |

自己写新分析时，最省事的做法是**从模板起步、再 `source scripts/hpc_env.sh`
替换掉 ②③④ 三段**——那一行等价于模板里那七行，而且已经按实测填好了。
