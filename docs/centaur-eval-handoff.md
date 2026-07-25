# Centaur 评估工作流：交接文档

**日期**：2026-07-25
**配套文档**：[centaur-eval-design.md](centaur-eval-design.md)（科学设计，实验分解见 §12）
**范围**：第一阶段——只做 LLM 推理侧打分（不微调、不复现认知模型）

---

## 0. 一句话现状

打分引擎（E0/E1/E3）与序列基线（E2）已实现。E0、P0、E2 和 Minitaur 的 E3
全量 runtime-NF4 运行均已完成，E1 初步曲线与 E0×E2 交叉分析已出（§5）。在当前
prompt 重建协议下，E3 的截断干预进一步确认：删除会话历史会损害 Minitaur 的预测，
绝大部分收益来自最近一个 transcript 段，但在 session 中后期及部分任务上仍有
超过 20 段的残余长上下文收益。
HPC 维护期间，24GB Mac 跑 bf16 的 8B **内存不稳定**，本地预览因此改在 RTX
5060 Ti 上使用 NF4。

**当前状态**（2026-07-25）：gated `Psych-101-test` 已获授权并下载。E0 和 E3
均完整覆盖 75 个精确 experiment、6,561 个 session；E3 产物为
`outputs/scoring/minitaur8b_e3_e0grid5_4bit.csv`，完整性审计无缺失、无重复、无
failed/skipped sidecar。另有 P0（paper-like NLL）兼容性轨道：从该因果模型的完整
E0 cache 经 tokenizer cutoff 审计后构造，采用论文 evaluator 的 36 个 task family、
32,768-token 头部截断与 session-mean 聚合。它对**评估协议**兼容，但不是
Centaur-70B/BF16/FlashAttention 论文结果的复现；4-bit 结果始终单列。

---

## 1. 实验状态（对应 design §12）

| # | 内容 | 状态 | 产物 / 备注 |
|---|---|---|---|
| E0 | 75 个精确 experiment 的 full-context 逐 choice NLL | **已全量完成（runtime NF4）** | `minitaur8b_e0_full_4bit.csv`；是 E1/E3 的研究型全上下文输入，不等同论文 evaluator |
| P0 | paper-like NLL：论文 36 family 的 evaluator 协议 | **已完成（runtime NF4）** | `minitaur8b_paperstyle_nf4*.csv`；见 §5，不能与 70B/BF16 论文数字混写 |
| E1 | 逐 trial 位置的适应曲线 | **初步完成（runtime NF4）** | `outputs/analysis_e1_adaptation_curve.csv`；发现与解读见 §5，Llama 对照曲线待做 |
| E2 | 计数序列基线 | **已全量完成** | `outputs/scoring/e2_all_tasks_s50.csv`（+ `_summary.csv`）|
| E3 | 上下文窗口截断 | **已全量完成（runtime NF4）** | `outputs/scoring/minitaur8b_e3_e0grid5_4bit.csv`；结果与审计见 §5 |
| E3a | $w=0\rightarrow1$ 的格式/任务/个体收益分解 | 待做（代表任务试点） | 五条件设计见 design §7.4 |
| E3b | 超过 20 段历史的长度匹配干预 | 待做（代表任务试点） | far shuffle / participant swap / control，见 design §7.5 |
| E4 | 语言表面扰动 | 未开始 | 每个实验需单独设计变换，最费手工 |
| E5 | 上下文×微调因子分解 | 待做（**纯分析**） | 从 E0+E3 的 CSV 计算，见 design §6 |

---

## 2. 代码地图

四层，依赖严格向下，读代码自底向上一遍即可：

```
scripts/*.slurm            作业层：资源声明 + 流程串接，无逻辑
scripts/experiments/*.py   入口层：argparse、文件读写、续跑/分片/容错编排
src/mt/evaluation/*.py     库层：全部科学逻辑，可单测，无 I/O
mt/data/_llm_supervision   既有代码：唯一衔接点 find_target_spans（定位 <<>>）
tests/evaluation, tests/experiments   平行，只测库层与入口 helper
```

**库层**（`src/mt/evaluation/`）：

- `transcript_scoring.py` — E0/E1 核心。teacher-forced 逐 choice NLL。
  链路：`_prepare_marked_text`（字符 span→token 下标，BOS 偏移、上下文守卫）
  → `score_marked_texts`（按长度打包 batch）→ `_score_batch` + `_forward`
  （**只在打分位置上过 lm_head，不 materialize 全词表 logits**）。
- `context_windows.py` — E3。`segment_transcript` 无损切分（header/含 choice
  标记的 transcript 段/tail，通常一段对应一条 trial line；同一行的多个 choice
  不再拆开；重组必须恒等原文否则 RuntimeError）；`build_window_prompt` = header
  + 最近 w 段；`score_window_grid` 批量打分 (target,window) 网格。
  **import 了 transcript_scoring**，是库层内唯一的内部依赖。
- `sequence_baselines.py` — E2。`score_sequence_online`（会话内 prequential
  计数，见下节决策）；`TableBuilder`/`score_sequence` 是群体版备用。

**入口层**（`scripts/experiments/`）：

- `_common.py` — 三个 runner 的共享底座：`load_sessions`（显式 UTF-8、过滤/抽样/分片/
  `max_chars` 跳过 + 逐任务日志）、`load_model`（普通 or bitsandbytes 量化）、
  续跑三件套（`completed_sessions`/`append_records`/`guard_output`）、
  `empty_device_cache`、`resolve_dtype`、`skip_log_for`/`failure_log_for`。
- `run_transcript_scoring.py`（E0）、`run_window_scoring.py`（E3）、
  `run_sequence_baselines.py`（E2）、`build_paper_style_nll.py`（P0）和
  `preflight.py`（提交前自检）。P0 工具用原 tokenizer 审计 32,768-token 截断，
  仅在没有 response span 横跨 cutoff 时复用完整 E0 cache；否则必须改为直接截断后重打分。

**作业层**（`scripts/`）：`smoke_e0_e3.slurm`（30 分钟 1 卡烟测）、
`e0_e3_minitaur.slurm`（主作业，4×V100 数据并行 + 合并分片）。
注：这两个 slurm 是为 HPC 的 Minitaur-8B 写的；本地 CUDA 卡直接用命令行。

---

## 3. 本会话的关键决策（代码里看不出的 why）

1. **必须用 Centaur 原论文的 held-out split**（`Psych-101-test`），不能用本仓库
   splitting 重切——公开 checkpoint 在 ~90% 参与者上微调过，自建 split 会泄漏。
2. **E2 用会话内在线（prequential）计数，不是群体拟合**：Psych-101 给每个参与者
   随机分配按键字母，跨参与者的群体计数在原始标签空间上是噪音（试点 base rate
   ≈ ln 26）。在线计数严格因果、无泄漏，且概念上正好对应"ICL 能从上下文薅到的
   表面统计"。群体版保留为 `--mode population` 备用。
3. **E2 基线看不到逐 trial 的选项集**（如 two-step 每 trial 只给两个选项中的一对），
   所以只能贴 ln(4) 而非 ln(2) 走；对比 LLM 时要记住 LLM 读得到选项。
4. **logits 内存优化**：`_forward` 只在打分位置过 lm_head，避免 `[seq,vocab]`
   全张量（12.8 万词表，长会话仅此就数 GB）。与 dense 路径逐 token 等价
   （测试 + Qwen 真模型 0 差异）。不暴露 `.model`+output embedding 的模型自动回退。
5. **打分只需 `use_cache=False`**（无生成，不需要 KV cache）。
6. **E3 网格抽样**：全量对每个目标×每个 w 重构 prompt 会到数十亿 token。E0/E1
   已显示最大变化发生在开头第 1→2 个 trial/目标（macro NLL 2.75→1.28），因此 E3
   默认改为预先固定的五点 `e0-informed` 网格：第 1、第 2、10%、50% 和最后一个
   **含 choice 标记的 transcript 段**。这通常对应 trial line，但不是“原子 choice”
   的严格同义词：同一行可含多个 `<<...>>`。本次抽中的目标段有 92.4% 恰含一个
   choice，单 choice 目标的敏感性分析保持相同结论。这不是按单个 session 的 NLL
   挑点，不引入结果导向的选择偏差；旧的等距网格仍可用 `--position-grid even`
   复现。早期不同 window 产生的相同 prompt 只前向一次，再把同一分数回填到各
   window label，输出语义不变。
7. **P0 是协议控制轨道，不是 E0 的替代品**：固定官方 36 个 family prefix、
   `add_special_tokens=True` 后保留开头 32,768 token（含 BOS），只监督 `<<...>>`
   内 token。主指标是每个 family 的 session-mean token NLL；token-micro NLL 只作诊断。
   E0 的 75-experiment、无截断输出继续服务 E1/E3。对因果模型，cutoff 前的 logits
   不依赖未来 token；因此本次通过“无 span 横跨 cutoff”审计后可准确复用这些保留 token，
   但任何将来审计失败的数据都必须直接重跑，不能用 `--max-chars` 代替 token 截断。

---

## 4. 本地运行的内存问题始末（用户重点关注）

**目标**：HPC 维护期间在本地用真 Minitaur-8B 跑预览。

**踩过的坑与修法（时间顺序）**：

1. 初版打分对整段序列做 float32 log_softmax → 长会话显存爆。
   **修**：只 gather 目标位置（`_score_batch`）。
2. `model(...).logits` 仍 materialize 全词表 `[seq,vocab]` → 长会话数 GB。
   **修**：logits 内存优化（决策 4）。
3. 24GB Mac 上早期实现跨会话拼 batch，使注意力显存翻倍且会话间显存不及时释放
   → OOM。**修**：runner 改为逐 session 调度，只在同一 session 内按 token budget
   打包目标；同时设置 `use_cache=False`、每会话 `gc.collect()`+清缓存，并逐会话
   捕获 OOM 记入 `.failed.csv` 后跳过而非中断。
4. **仍不稳**：用户观察到 Python 内存运行时突然涨到 30–40GB 冲破物理内存。
   根因：**16GB 的 bf16 模型放在 24GB 机器上，运行时波动一冲即破**，非代码可根治。

**结论与出路**：
- **24GB Mac 跑 bf16 8B 判定为不可行**，别再调。单会话探针能跑（NLL 0.44 合理），
  但全量会因累积/波动 OOM。
- 已加 **`--load {8bit,4bit}`**（bitsandbytes，CUDA only）。RTX 5060 Ti 实测
  4-bit 模型常驻显存 5.68 GiB，短评估峰值 6.27 GiB；加载 BF16 checkpoint 并现场
  量化时，Windows 进程主内存峰值 15.83 GiB，加载完成后回落到 1.86 GiB。测试机
  有 32 GiB 主内存，余量够，但加载前仍应关闭占内存较大的程序。
- 当前打分设置 `use_cache=False`，因此长会话的额外显存主要是 prefill 激活与注意力
  工作区，不是 KV cache。Windows PyTorch wheel 未编译 FlashAttention，原先会在 GQA
  上回退到申请约 336 GiB 的平方内存 math 路径；现已在 CUDA 上显式按
  Flash → cuDNN → memory-efficient → math 排序，Mac/CPU 路径不受影响。
- 测试集最长 session（`xiong2023neural/exp1.csv`，participant 28，168,968 字符）现已
  跑通：E0 得到 4,800 个 choice，full-context response-token micro NLL 0.4723；
  E3 的 0/5/full 窗口探针也通过。`nvidia-smi` 观察到整卡峰值约 15.8/16.3 GiB，
  主内存工作集约 7.6 GiB。因此 16GB **可跑但余量很窄**；本次 E3 全量使用
  `--batch-tokens 16384`、不设 `--max-chars`，最终无失败完成。runner 本身逐
  session 运行，没有 `--chunk-size` 参数。
- 4-bit 会改变 NLL。P0 已作为独立的“paper-like runtime-NF4”协议控制轨道完成：
  固定原始 held-out split、prompt、36-family allowlist、32,768-token 截断、metric 与
  NF4 配置。结果必须标成 `Minitaur-8B BF16 checkpoint, runtime NF4`，不能替代 BF16
  或 70B 主结果。

**⚠️ 未清理的污染**：Mac 上那次失败的运行把一批**本不大**的会话（如 1067 字符）
误记进了 `outputs/scoring/minitaur8b_e0_full.failed.csv`（当时是内存满不是会话大）。
若要在 Mac 上重试，先 `rm -f outputs/scoring/minitaur8b_e0_full.*`，否则 `--resume`
会永久跳过这些好会话。转 CUDA 卡用新文件名则无影响。

---

## 5. 已有结果

### E0：full-context runtime-NF4（研究型轨道）

`outputs/scoring/minitaur8b_e0_full_4bit.csv` 完整覆盖 6,561 / 6,561 个 session、
75 个精确 experiment、1,177,866 个 choice 和 1,410,879 个 response token；没有
failed/skipped session。原文件的全体 response-token 加权 micro NLL 为
**0.59713413**；把 10 个 CP936 误读的 `zorowitz2023data` session 换成 UTF-8 修复
结果后为 **0.59712747**，对应 all-choice session-macro token NLL 为 **0.79965797**。
正式分析应使用修复后的拼接视图。这是无截断全 transcript 的工程/研究诊断，不能
直接与论文的 36-family `eval_loss` 相比。

已知瑕疵：CSV 中 `collsiöö2023MCPL` 的三个 experiment 名被 Windows 运行写成了
`collsi枚枚2023MCPL`（CP936 误解码残留）。已核查其 transcript 文本为纯 ASCII，
**分数本身有效**，但和 E2 等按 experiment join 时必须先归一化名字，或直接修 CSV。

### P0：paper-like NLL（评估协议兼容轨道）

`outputs/scoring/minitaur8b_paperstyle_nf4.csv` 和 `_summary.csv` 使用官方 36 个
family prefix（57 个精确 experiment、5,795 个 session）。tokenizer 在加入 special
tokens 后从右侧截断，即保留开头 32,768 token（含 BOS）；目标严格位于 `<<...>>` 内。
58 个 session 超过 cutoff，合计 28,862 个完整的尾部 choice 因此被排除；没有任何目标
span 横跨 cutoff，故保留 token 可从完整 E0 cache 精确复用。10 个
`zorowitz2023data` session 已由 UTF-8 更正 E0 输出替换，未沿用旧的 CP936 误解码分数。

主产物是 36 个 family 各自的 `official_eval_loss`：先在 session 内以 target token 平均，
再在 family 内对 session 平均，对应官方 `per_device_eval_batch_size=1` 的聚合语义。
本次 36 个 family 分数的**本地等权平均**为 **0.572705322**；它只是便于浏览的汇总，
不是论文报告的全局官方 scalar。诊断性 token-micro NLL 为 **0.581371297**。该结果匹配
评估口径，但模型、NF4/FP16 运行时和 cuDNN SDPA 后端均不同于论文的 Centaur-70B/BF16/
FlashAttention 设置，不能把数值解释为论文复现或直接相减。

### E3：context-window curve（2026-07-24，runtime NF4）

产物：`outputs/scoring/minitaur8b_e3_e0grid5_4bit.csv`。完整性审计结果：

- 覆盖 **6,561 / 6,561 session、75 / 75 experiment**，共 459,102 条 response 记录；
  每个 window 都有 65,586 条 response、68,542 个 response token。
- 每个 window 有 32,672 个抽样目标位置，7 个 window 合计 228,704 个带 window label
  的目标格；按同一 target 下的等价 window label 去重后，实际需要评分的是 133,034
  个 effective prompt cell/input（不是对全文本做全局 unique，也不是 133,034 次
  forward call；模型调用仍会按 `batch-tokens` 打包）。
- 原始数据逐 session/target 重建后 missing=0、unexpected=0、incomplete session=0，
  完整结果键重复=0；没有 `.failed.csv` 或 `.skipped.csv`。
- 6,507 个 session 取得 5 个锚点，其余 54 个短 session 合法地只有 1–4 个；所有
  NLL 有限、非负，token 数有效。

CSV 的 `nll` 是一个 `<<...>>` response 内各 token NLL 的**和**，因此 token-micro
必须用 `sum(nll) / sum(num_tokens)`，不能再次乘 `num_tokens`。下表先报告描述性
**session-macro token NLL**：在每个 session/window 内汇总抽中的 response token，
再对 6,561 个 session 等权。它描述测试集中 session 的经验分布，不是跨 experiment
等权的最终主推断；`Δ vs full` 是同 session 配对差，95% CI 为配对差均值
± 1.96×SE 的正态近似，越低越好：

| window（历史段数） | token-micro NLL | session-macro token NLL | 配对 Δ vs full（正态近似 95% CI） |
|---:|---:|---:|---:|
| 0 | 1.292844 | 2.256869 | +1.152546 [1.128556, 1.176535] |
| 1 | 0.737327 | 1.259799 | +0.155476 [0.147573, 0.163380] |
| 2 | 0.704850 | 1.206035 | +0.101712 [0.094936, 0.108489] |
| 5 | 0.676376 | 1.156379 | +0.052056 [0.046595, 0.057516] |
| 10 | 0.658825 | 1.134835 | +0.030512 [0.026438, 0.034587] |
| 20 | 0.649056 | 1.122534 | +0.018210 [0.015206, 0.021215] |
| full | 0.636964 | 1.104323 | 0 |

按 design §9.2 先在 participant/session 内汇总、再对每个精确 experiment 内的
participant 平均、最后对 75 个 experiment 等权，跨任务 point estimate 同样单调：

| window | 0 | 1 | 2 | 5 | 10 | 20 | full |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 75-experiment task-macro NLL | 2.650945 | 1.526024 | 1.437704 | 1.347940 | 1.309466 | 1.275880 | 1.210492 |

w=20−full 的 task-macro 差为 +0.065388，75 个 experiment 中 64 个为正。正式论文
推断仍应按 §9.2 生成带固定 seed/重复次数的 participant/task bootstrap 产物；当前
文档的区间只是明确标注的正态近似，不能冒充 bootstrap。

在上述**五锚点、session 等权**口径下，从 w=0 到 full 的总改善中，w=1 已捕获
**86.5%**，w=2 为 91.2%，w=5 为 95.5%，w=20 为 98.4%。这说明主要收益来自
最近一个历史段，但不等于“一个原子 trial 就足够”：窗口单位是含 choice 标记的
transcript 段，同一行可能包含多个 choice。仅按论文 `PAPER_TASKS` allowlist 筛出的
36-family E3 子集（仍是五锚点、无 head-32k 截断，**不是 P0 指标**）上，w=20 比
full 高 **0.03527 nat**；36 个 family 配对差的正态近似 95% CI 为
[0.01794, 0.05261]。平均而言仍存在小而稳定的长上下文收益。

为避免五点网格前密后疏造成误读，严格位置分层只用 `n_segments > 5` 的 6,478 个
session。每个 session/position/window 内先算 `sum(nll) / sum(num_tokens)`，再对
6,478 个 session 等权；表中 `w-full gap` 是该位置截断相对 full 多出的 NLL：

| 位置 strata | full NLL | w=1 gap | w=5 gap | w=20 gap |
|---|---:|---:|---:|---:|
| first | 2.093 | 0 | 0 | 0 |
| second | 1.066 | 0 | 0 | 0 |
| 10% | 0.833 | +0.209 | +0.046 | +0.006 |
| 50% | 0.747 | +0.241 | +0.085 | +0.028 |
| last | 0.778 | +0.329 | +0.141 | +0.053 |

first/second 在足够长 window 下与 full 相等是由可用历史长度**构造性决定**，不是额外
证据；second 的 w=0 gap 为 +1.217。真正的信息是到了 10%、50% 和末点，短窗口相对
full 的损失逐渐变大，末点对较长历史最敏感。任务间也有明显异质性：例如
`feng2021dynamics`、`xiong2023neural` 的 w=20→full 改善较大，少数任务则 w=20
略优于 full，可能是上下文干扰或小数值差，不能逐任务过度解释。

结构完整性与聚合一致性的 sanity check 均通过，但不声称逐行 numerical
equivalence：

1. 把 E0 中误编码的 `collsiöö2023MCPL` 名称归一化，并用 UTF-8 修复文件替换 10 个
   `zorowitz2023data` session 后，E3/full 的 65,586 条 response 与 E0 一一匹配，
   token 数完全一致。两者 token-micro 分别为 0.63696353 和 0.63705079，差
   -0.00008726。单行并非 bit-exact（MAE 0.00413，最大 0.242）；这与低精度
   CUDA/cuDNN 对 batch shape/packing 敏感的数值差异相容，但来源尚未用同 prompt
   重复实验单独隔离。这里只能说 key/token 审计和聚合一致性通过，不能说逐行等价。
2. 抽中目标段中 30,195 / 32,672（92.4%）只含一个 choice；限制到单 choice 目标的
   敏感性分析在总体聚合上仍得到同样的单调曲线与“近期收益为主、长历史残余存在”
   的结论。

**结论边界**：E3 在当前 prompt 重建协议下，直接测得了“删除较早历史段”对模型
NLL 的干预效应，比 E1 的观察性位置曲线更强；但截断同时改变可见内容、输入长度和
transcript 连贯性，因此不能把它单独解释为抽象的 memory-span 参数，更不能证明
模型使用了人类相同的认知机制。本结果只适用于 Minitaur-8B runtime NF4、teacher
forcing 和五点位置抽样。E3 五点 `full` scalar 不能与 E0 all-choice global scalar
或 P0 的 36-family/head-32k `official_eval_loss` 直接相减；E3 与 E0 的相同 keys
sanity 对照仍然有效。未来还需 Llama-base 对照和 BF16/HPC 主结果。

### E2：全 75 任务，每任务 ≤50 人抽样，seed=0，43.7 万 choice

macro NLL（trial→参与者→任务→总体）：uniform 1.61 / base_rate 1.44 /
sticky 1.41 / **bigram 1.27**。即纯序列统计能从均匀基线砍掉约 0.34 nat。

分任务两极：`cox2017information`、`popov2023intent`、`garcia2023experiential`
等序列自相关强的任务，bigram 吃掉近 1 nat；`wulff2018description`、
`hebart2023things`、`ruggeri2022globalizability` 等独立 trial 任务增益≈0。
后者是检验 Centaur 真本事的干净战场。这套数字是 Centaur/Llama 结果要超越的底线。

### E1 初步 + E0×E2 交叉分析（2026-07-22，全部 runtime NF4）

产物：`outputs/analysis_e0_vs_e2_by_task.csv`（逐任务对照）与
`outputs/analysis_e1_adaptation_curve.csv`（会话内十分位适应曲线，限制在与 E2
共同的 session 上算）。指标均为本设计的分层 **macro choice NLL**
（trial→参与者→任务），与 P0 的 `official_eval_loss` 口径不同，不可混写。

1. **总体**：Minitaur **0.90** vs uniform 1.61 / base_rate 1.44 / sticky 1.41 /
   bigram 1.27——比最强计数基线好 0.37 nat。但逐任务的 Minitaur 增益与 bigram
   增益相关 **r = 0.79**：模型赢得最多的任务，恰是序列统计能薅到最多的任务。
2. **适应曲线（E1 的预言得到印证）**：decile 0→9 Minitaur 下降 0.23 nat
   （18 个独立 trial 任务上 0.25），bigram 只降 0.07，sticky/base_rate ≈ 0——
   会话内适应远超在线计数可解释的范围。更细看开头：**第 1 个 trial macro NLL
   = 2.75，比 uniform（1.61）差 1.1 nat**；第 2 个 trial 即降至 1.28。即 k=0 时
   模型并无优势，优势需要上下文才出现（design §5 的 ICL 判据）。注意反向
   caveat：E2 的 uniform 白拿了会话标签字母表，而 LLM 需自己从 instructions
   读出合法按键，trial-0 的差距部分是格式发现成本，不全是认知内容。
3. **干净战场上优势消失**：三个旗舰独立任务 Minitaur 全部输给或打平 uniform——
   `wulff2018description` 0.838 vs 0.693、`ruggeri2022globalizability` 0.712 vs
   0.693、`hebart2023things` 1.148 vs 1.099。而赢 bigram 最多的任务多为答案可从
   上下文直接读出/推出的记忆类（`enkavi2019digitspan`、`popov2023intent`、
   `garcia2023experiential`、`krueger2022identifying`）——那是上下文抽取，
   不是认知模拟。18 个独立任务整体 Minitaur 仍平均领先 0.26 nat，故不能说优势
   "全是"表面统计，但对"机制相似性"解读非常不利。
4. **对核心假设的初步判定**：与 design §1 的分解主张一致——full-context 优势
   ≈ 大量早期 ICL 适应 + 与序列统计高度相关的任务增益 + 记忆/复述类任务的
   上下文抽取，而在最能检验认知机制的独立 trial 任务上归零甚至为负。
5. **结论边界**：(a) 缺同协议的 **Llama-3.1-8B base E0/E3**，无法拆分预训练 vs
   微调（E5 缺口，当前性价比最高的下一步）；(b) E1 本身仍是观察性位置曲线，但
   已完成的 E3 直接测量了当前 prompt 重建协议下删除较早历史段的干预效应；它没有
   单独识别抽象记忆跨度，也不等于人类认知机制；(c) 全部数字为 runtime NF4，
   量化可能不均匀影响长尾任务；
   (d) 14 个任务 choice 多于 1 token（`wise2019acomputational` 达 3
   token/choice，其"输给 bigram"部分是口径 artifact）；E2 仅抽样 ≤50 人/任务。

---

## 6. git / 未提交状态

原交接所列内存与量化改动已提交到 `3e65c89`，不是当前未提交状态。2026-07-22
的未提交改动修复运行可靠性与跨设备安装说明：只把真实设备 OOM 记失败、空结果
不再生成坏 CSV、bitsandbytes 作为非 macOS 的可选 extra，并让 E0 同时汇总 token-micro
诊断与本项目的分层 macro choice NLL。另增加精确 `--participant` 探针筛选，并在 CUDA
上优先选择 fused SDPA，避免 Windows 回退到平方内存 math attention。

2026-07-22 新增：库层的超长上下文守卫改抛 `ContextLengthError`，两个 runner
把它当作与 OOM 同级的会话级失败——记入 `.failed.csv` 后跳过，不再让整个运行
崩溃。动机：Mac 上换用小模型（如 0.5B，32k 上下文）时，测试集最长会话
（~35k token）超窗直接把全量跑挂掉。注意小上下文模型会因此静默丢掉最长的
会话，与大模型结果对比时需检查 `.failed.csv` 的丢弃集合。

CUDA wheel 只安装在本机 gitignored `.venv`，没有写入仓库的全平台依赖源。Mac 继续
使用 MPS/`--load none`，HPC 继续使用集群自己的 CUDA PyTorch/`--load none`；需要
量化的 CUDA 环境再显式安装 `.[centaur-eval]`。

安装入口按设备分开，避免互相覆盖：

- Mac：`uv pip install -e ".[dev]"`，运行时用 MPS/`--load none`。
- 5060 Ti 新环境：先从 PyTorch CUDA 13.0 wheel 源安装 `torch`，再执行
  `uv pip install -e ".[dev,centaur-eval]"`。CUDA wheel 源只用于这一步，不写进项目锁文件。
- HPC：保留集群模块提供的 CUDA/PyTorch；环境依赖已由集群准备好时，用
  `uv pip install -e . --no-deps` 安装本项目，运行 `preflight.py --load none` 复核。

后续新增的未提交项包括：`load_sessions` 显式使用 UTF-8，以修复 Windows 默认 CP936
误读 10 个 `zorowitz2023data` prompt；P0 的 cache-to-paper-style 工具及其测试；以及
用户要求移除 `.gitignore` 中的 `outputs/`。生成的结果 CSV 现在会出现在 Git status 中，
但尚未自动暂存或提交。

2026-07-25 的当前工作树还包含 E3 的 E0-informed 网格、等价 prompt 去重/续跑相关
代码与测试、两份文档更新，以及未跟踪的全量结果
`outputs/scoring/minitaur8b_e3_e0grid5_4bit.csv`。该结果约 459k 行；提交前应明确
决定是否真的把大型 CSV 纳入 Git，而不是因为 `outputs/` 已取消 ignore 就顺手全加。

---

## 7. 下一步（建议顺序）

1. **Llama-3.1-8B base 的 E0 + 对应 E3（当前性价比最高）**：使用相同数据、窗口、
   五点网格、NF4 设置与聚合；至少需要 base 的 full 与预注册 matched window，完整
   曲线更利于检查交互。模型换为 `meta-llama/Llama-3.1-8B`（gated，HF 账号需先
   同意条款），输出必须使用新文件名。有了它才能拆分预训练 vs 微调（E5）、给 E1
   加双模型对照，并检验独立 trial 任务上的负结果是否 Minitaur 特有。
2. **检查并保存 P0 的逐 family 结果**：以 `_summary.csv` 的 36 个
   `official_eval_loss` 为比较单位；不要把本地 task-macro 或 token-micro 误当作论文
   唯一全局数字。
3. **E1 正式版**：初步曲线已出（§5），待 Llama base 跑完后画双模型对照曲线；
   仍从无截断 E0 的 CSV 聚合，不要用 P0 替代（P0 尾部按论文协议被截断）。
4. **E3 结果封装与敏感性分析**：主运行已经完整，不需要重跑。下一步把 §5 的窗口
   主曲线、五位置 strata、participant/task bootstrap CI 和单 choice 目标敏感性导出
   成固定 summary/figure，并记录 bootstrap seed 与重复次数。只有论文需要稳健性
   附录时再跑 `--position-grid even`，BF16/HPC 精确版另列，不覆盖本次 NF4 产物。
5. **E3a：拆分 $w=0\rightarrow1$ 的巨大收益**：在相同目标位置依次比较
   instructions only、显式合法 label space、训练分布内的 format-only demonstration、
   同任务/阶段的 matched other-participant trial，以及目标参与者自己的上一段历史。
   相邻差值分别用于估计 response alphabet、格式校准、任务识别与个体适应；只有
   own history 相对 matched other history 的额外收益才初步归入 online phenotyping。
   先在独立 trial、预生成反馈或 yoked trajectory 的代表任务上试点，完整约束见
   design §7.4。
6. **E3b：长度匹配的远端历史干预**：固定目标、instructions、最近 20 段、历史段数
   和 tokenizer token budget，只对更早历史做 within-participant shuffle、
   matched-participant swap，以及在能够保持训练分布和因果合法时加入 neutral/control
   history。不得用无意义 padding 凑长度，也不得在 RL 任务中拆开交换 action/outcome。
   用该实验判断 E3 的 $w=20\rightarrow\mathrm{full}$ 残余来自远端顺序、参与者 profile、
   任务阶段还是单纯长度线索；先做代表任务试点，详见 design §7.5。
7. **E5**：等 Llama base 的 E0/E3 到位后计算上下文增益与交互项，画 Figure B
   瀑布（须显式报告交互项）。
8. **E4**：语言表面扰动（逐任务设计，最费手工），可先只对试点任务做。
9. **HPC 精确对照（未来，需用户启动）**：在原生 BF16/FP16 环境上直接跑论文协议，
   用于与 runtime-NF4 P0 分开报告；不要把这一步自动并入本地作业。
10. **小修**：把 E0 CSV 里 `collsi枚枚2023MCPL` 的 experiment 名改回
   `collsiöö2023MCPL`（分数有效，仅名字误解码，见 §5）。

---

## 8. 起步要读的文件

1. `docs/centaur-eval-design.md`（§2.1 P0 协议控制轨道、§12 实验分解，§14 结论边界）
2. 本文件
3. `scripts/experiments/build_paper_style_nll.py`（P0 cutoff 审计、cache 复用和聚合）
4. `src/mt/evaluation/transcript_scoring.py`（打分核心，重点 `_forward`/`_score_batch`）
5. `scripts/experiments/_common.py`（`load_sessions` 顺序、UTF-8、`load_model`、续跑）
6. `docs/agents/CONVENTIONS.md`（代码风格：80 列、单空行、填满再折行等）
