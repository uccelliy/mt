# Centaur 评估工作流：交接文档

**日期**：2026-07-28
**配套文档**：[centaur-eval-design.md](centaur-eval-design.md)（科学设计，实验分解见 §12）
**范围**：第一阶段——只做 LLM 推理侧打分（不微调、不复现认知模型）

---

## 0. 一句话现状

打分引擎（E0/E1/E3）与序列基线（E2）已实现。E0、P0、E2，以及 Minitaur 与
**Llama-3.1-8B base 的 E3** 全量 runtime-NF4 运行均已完成（§5），E5 因子分解
也已完成。当前最重要的三个定量结论：(a) 两模型的 context-gain 曲线在 $w\ge1$
时几乎重合，微调×上下文交互接近零；(b) 微调主要改善 $w=0$ 冷启动，交互
-0.318 nat；(c) E0 all-choice 下行为微调的全上下文增益只有 ~0.014 nat，且集中在
trial 0——在这个 8B/NF4 设置下，绝大部分优势来自通用预训练+ICL。
HPC 维护期间，24GB Mac 跑 bf16 的 8B **内存不稳定**，本地预览因此改在 RTX
5060 Ti 上使用 NF4。

**当前状态**（2026-07-28）：gated `Psych-101-test` 已获授权并下载。两模型的 E0
和 E3 均完整覆盖 75 个精确 experiment、6,561 个 session；E3 产物为
`outputs/scoring/{minitaur8b,llama31_8b_base}_e3_e0grid5_4bit.csv`，完整性审计
无缺失、无重复、无 failed/skipped sidecar。另有 P0（paper-like NLL）兼容性轨道：
从该因果模型的完整
E0 cache 经 tokenizer cutoff 审计后构造，采用论文 evaluator 的 36 个 task family、
32,768-token 头部截断与 session-mean 聚合。它对**评估协议**兼容，但不是
Centaur-70B/BF16/FlashAttention 论文结果的复现；4-bit 结果始终单列。

---

## 1. 实验状态（对应 design §12）

| # | 内容 | 状态 | 产物 / 备注 |
|---|---|---|---|
| E0 | 75 个精确 experiment 的 full-context 逐 choice NLL | **已全量完成（runtime NF4）** | `minitaur8b_e0_full_4bit.csv`；是 E1/E3 的研究型全上下文输入，不等同论文 evaluator |
| P0 | paper-like NLL：论文 36 family 的 evaluator 协议 | **已完成（runtime NF4）** | `minitaur8b_paperstyle_nf4*.csv`；见 §5，不能与 70B/BF16 论文数字混写 |
| E1 | 逐 trial 位置的适应曲线 | **双模型初步完成（runtime NF4）** | `outputs/analysis_e1_adaptation_curve.csv`、`outputs/analysis_llama_vs_minitaur_curve.csv`；正式 bootstrap/figure 待封装 |
| E0-L | Llama-3.1-8B **base** 对照 E0（同协议） | **已全量完成（runtime NF4）** | `llama31_8b_base_e0_full_4bit.csv` + paperstyle 派生；关键发现见 §5 |
| E2 | 计数序列基线 | **已全量完成** | `outputs/scoring/e2_all_tasks_s50.csv`（+ `_summary.csv`）|
| E3 | 上下文窗口截断 | **双模型均已全量完成（runtime NF4）** | `outputs/scoring/{minitaur8b,llama31_8b_base}_e3_e0grid5_4bit.csv`；结果与审计见 §5 |
| E3a | $w=0\rightarrow1$ 的收益分解 | 待做 | 2026-07-27 修订设计：label-space prompt 条件改为合法 label 重归一化解析诊断 + 四条件阶梯，见 design §7.4 |
| E3b | 超过 20 段历史的长度匹配干预 | **暂缓** | 目标效应 ~0.02 nat，功效存疑；重启门槛与 shuffle-only 试点见 design §7.5 |
| E4 | 语言表面扰动 | 未开始 | 每个实验需单独设计变换，最费手工 |
| E5 | 上下文×微调因子分解 | **已完成（8B/runtime NF4）** | $w=0$ 交互 -0.318 nat；$w\ge1$ 交互接近零，见 §5 |

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

### E3：Minitaur context-window curve（2026-07-24，runtime NF4）

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
模型使用了人类相同的认知机制。本节结果只适用于 Minitaur-8B runtime NF4、teacher
forcing 和五点位置抽样。E3 五点 `full` scalar 不能与 E0 all-choice global scalar
或 P0 的 36-family/head-32k `official_eval_loss` 直接相减；E3 与 E0 的相同 keys
sanity 对照仍然有效。Llama-base 对照与 E5 见下文；论文精度仍需 BF16/HPC 主结果。

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
5. **结论边界**：(a) 同协议 **Llama-3.1-8B base E0/E3** 与 E5 已完成；结果显示
   $w\ge1$ 的 context-gain 交互近零，强化了“主要是通用 ICL、微调主要改善冷启动”
   的解释；(b) E1 本身仍是观察性位置曲线，但
   已完成的 E3 直接测量了当前 prompt 重建协议下删除较早历史段的干预效应；它没有
   单独识别抽象记忆跨度，也不等于人类认知机制；(c) 全部数字为 runtime NF4，
   量化可能不均匀影响长尾任务；
   (d) 14 个任务 choice 多于 1 token（`wise2019acomputational` 达 3
   token/choice，其"输给 bigram"部分是口径 artifact）；E2 仅抽样 ≤50 人/任务。

**勘误（2026-07-27）**：上面第 3 条最初写的 uniform 数字（0.693/0.693/1.099）
是理论 ln(K)，不是 E2 的经验 uniform。经验值更低：`wulff2018description`
0.582、`ruggeri2022globalizability` 0.665、`hebart2023things` 1.091（E2 的
uniform 按 session 实际出现的标签集大小算，参与者从不选某选项时字母表变小）。
结论不变且更强：Minitaur 在这三个任务上输给经验 uniform 的幅度比原文更大。

### E0-L：Llama-3.1-8B base 对照（2026-07-27，runtime NF4）——微调增益极小

产物：`outputs/scoring/llama31_8b_base_e0_full_4bit.csv`（+`_summary.csv`）、
paperstyle 派生 `llama31_8b_base_paperstyle_nf4*.csv`，以及零算力分析
`outputs/analysis_llama_vs_minitaur_by_task.csv` 和 `_curve.csv`。完整性：
6,561 session、1,177,866 choice、1,410,879 token 与 Minitaur E0 **逐 key 完全
对齐**，无 failed/skipped；本次 experiment 名为正确 UTF-8。以下 Minitaur 侧
均已换入 zorowitz UTF-8 修复分数、归一化 collsiöö 名字后配对计算。

**核心发现：行为微调在 8B/NF4 下的全上下文增益只有 ~0.014 nat，而上下文增益
是它的 100 倍。**

1. **总量**：task-macro choice NLL：Llama base **0.9158** vs Minitaur
   **0.9016**。微调增益 = **0.0142 nat**（task 级配对 95% 正态近似
   ±0.0037），**75/75 任务全部为正**——真实、普遍、但极小。token-micro
   0.6043 vs 0.5971，同一图景。
2. **论文口径同样成立**：36-family `official_eval_loss` 均值 Llama 0.586291
   vs Minitaur 0.572705，增益 0.0136，**36/36 family 全部为正**。
3. **对照 E3 的上下文增益**：Minitaur w=0→full 的 task-macro 增益为 1.44 nat。
   以 uniform（1.6096）为原点的全上下文瀑布：uniform → Llama full 提升
   0.694 nat（通用预训练+ICL，其中 bigram 可达部分 0.341）→ Minitaur full
   再提升 0.014 nat（行为微调）。即在本设置下，相对简单基线的优势约 **98%
   来自通用 Llama 预训练 + 上下文学习，约 2% 来自 Psych-101 行为微调**。
4. **微调增益集中在会话开头**：trial-0 macro NLL Llama 3.19 vs Minitaur
   2.75（差 0.44 nat），从 trial 1 起差距骤降到 ~0.01–0.02 并保持；十分位
   曲线 decile 0 差 0.044，其后 0.008–0.012。解读：**微调学到的东西大部分
   等价于一个压缩的格式/反应空间先验，一个 trial 的 ICL 几乎可以完全替代**。
   这正是 E3a 要检验的：字母表重归一化诊断可判定 trial-0 的 0.44 nat 差有
   多少只是合法按键先验。
5. **微调增益与序列统计无关**：corr(微调增益, bigram 增益) = **-0.06**（对照
   总优势的 r = 0.79）。且微调增益最大的恰是独立 trial 的干净战场任务
   （`wulff2018description` 0.109、`ruggeri2022globalizability` 0.049、
   `wulff2018sampling` 0.043）——微调在 ICL 薅不到统计的地方帮忙最多，方向
   上符合"学到了行为先验"；但即便加上微调，两模型在这三个任务上仍全部输给
   经验 uniform（如 wulff：Llama 0.947 / Minitaur 0.838 / uniform 0.582）。
6. **结论边界**：(a) Minitaur-8B 不是 Centaur-70B——作者标注其分布外泛化更
   弱，且没有官方 8B 参考数字；"微调增益极小"目前只能声明到 8B 复制品 +
   runtime NF4 + teacher-forced NLL 这个设置；70B 上微调增益可能更大，这恰
   是将来 HPC 精确对照要回答的。(b) 两模型同 NF4 同协议配对，量化对差值的
   压缩是二阶效应，但 0.014 的绝对量级应谨慎对待。(c) E5 的配对结果见下一节；
   它不改变 70B/BF16 仍需独立验证的边界。

### ⚠️ 官方公布结果对比（2026-07-28 晚，修正主叙事的关键发现）

用户下载了官方 repo `results/` 的逐任务公布结果（36 个 seen family、标准
eval_loss、custom_metric=False；unseen 任务用 custom metric，不可直接比）。
对照脚本 `scripts/experiments/build_official_comparison_figures.py`，产物
fig9/fig10 与 `outputs/analysis_official_vs_ours.csv`。三个结论：

1. **管线获得端到端验证**：我们的 Llama-8B runtime NF4 与官方
   unsloth-8B-bnb-4bit 逐 family 几乎相同——平均 |差| 0.0008、最大 0.0043、
   r = 0.99999。P0 协议、tokenization、截断、聚合全部与官方 evaluator 等价。
2. **Minitaur 不等于官方 Centaur-8B**：官方 Centaur-8B-adapter 36-family 均值
   **0.4555**，我们的 Minitaur（merged 权重 + runtime NF4）为 0.5727，平均差
   +0.117（最大 +0.385，tomov2020discovery）。两个候选解释待判定：
   (H1) Minitaur 与 Centaur-8B-adapter 是不同 checkpoint；(H2) QLoRA adapter
   在 4-bit base 上训练，merge 到 BF16 再运行时 NF4 重量化损伤了 adapter。
3. **官方口径的微调增益是 ~0.13 nat，不是 0.014**：36-family 上
   base8→Centaur-8B 为 0.1305，base70→Centaur-70B 为 0.1248；相比之下
   8B→70B 的规模增益只有 0.017（base）/0.011（Centaur）。认知基线均值
   0.5928（34 family 可得），Centaur-70B 在全部 34 个上胜出；未微调的
   8B base（0.586）平均已与认知基线打平。

**叙事修正**：(a) "微调增益 0.014/2%" 只适用于 Minitaur-merged-NF4 设置，
引用官方数字时应说 0.13/1.44 ≈ 9%——**ICL 主导的大结论不变**（微调增益仍
不到上下文增益的十分之一，规模增益更小），但绝对量级要改口。(b) E5 的
"微调只除冷启动、w≥1 无交互"结论是用 Minitaur 测的，必须用官方配置复检。
(c) E0-L 小节边界 (a) 中"没有官方 8B 参考数字"的说法已被本节推翻。

**判定实验（已就绪）**：两个 runner 已支持 `--adapter`（PeftModel 加载到
量化 base 上，评分核心新增 `_unwrap_base` 解包 wrapper 链，有测试）。用官方
配置复跑 E0：若结果 ≈ 0.4555 则 H2 成立（我们的 Minitaur 轨道要重新标注为
"merged+requantized"变体）；若仍 ≈ 0.57 则 H1 成立。

### 官方逐 choice 原始数组（`custom_metrics_full_log_likelihoods_*.pth`）

官方 `results/` 还有一组 `.pth`（安全性已核：pickle 只引用 numpy 重建函数）：
`dict[task -> 一维 float 数组]`，是**逐 choice 的 NLL 原始值**。覆盖：认知
基线 37 任务、Centaur-8B/70B 与 base-8B/70B 各 46 任务（36 seen + 10
unseen），另有 Hermes/Nemotron/Reflection/Llama-3-Instruct 四个 70B
instruct 对照。已验证的三件事（2026-07-29）：

1. **顺序与我们的逐 choice cache 对齐**：官方 unsloth-8B 数组 vs 我们的
   Llama-8B cache 按位置比，badham r=0.996、bahrami r=0.999、digitspan
   r=0.9996、ruggeri r=0.998（残差 ~0.02/choice 为跨 GPU 数值噪声）。即
   数组顺序 = 测试集 jsonl 顺序 × session 内 choice 顺序，可借我们的
   cache 按位置标注 participant / choice_index。
2. **单位是每 choice 的 token NLL 之和**（用多 token 的 collsiöö 判定：
   对 sum r=0.999，对 mean 仅 0.86），与我们 `nll` 列同单位。
3. **认知基线 26/37 任务长度与我们完全一致**，可位置标注；11 个不匹配
   （kool 两步、部分 enkavi 等，认知模型只给部分 trial 打分）先排除。

使用注意：数组不带标签，任何用途前必须先过逐任务"长度一致"检查；截断任务
（如 wulff2018sampling 108,245 vs 全量 108,585）要配 P0 paperstyle cache
而非全量 E0。解锁的分析（汇报后做，见 §7）：E1 适应曲线加认知基线线
（Figure A 三方齐）、认知基线的分层 macro、unseen 任务对比（collsiöö 长度
恰好全对齐；popov 只挑了 1,892 条，需读官方 custom metric 代码确定 span）、
instruct-70B 对照加入 fig10 阵容。

### E3-L / E5：Llama base 窗口曲线与微调×上下文分解（2026-07-28）

产物：`outputs/scoring/llama31_8b_base_e3_e0grid5_4bit.csv`。完整性审计全部通过：
75 个 experiment、6,561 个 session、459,102 条 response；每个 window 均为
65,586 条 response、68,542 个 response token 和 32,672 个目标位置；每个目标都有
全部 7 个 window，完整键重复为 0，且与 Minitaur 的 459,102 个键逐一匹配、token
数完全一致。没有 `.failed.csv` 或 `.skipped.csv`。30,195/32,672（92.4%）目标位置
仅含一个 choice。

Llama E3/full 与自己的 E0 在相同 65,586 个 response key 上完全匹配且 token 数一致：
token-micro NLL 0.69071382 vs 0.69072503，差 -0.00001171；单行 MAE 0.000279，
最大绝对差 0.1957。它支持 key/token 和聚合一致性，但不能表述为逐行 bit-exact。

Llama base 自身的完整窗口曲线如下；session-macro 的 `Δ vs full` 区间是同 session
配对差的正态近似，仅作描述性审计：

| $w$ | token-micro NLL | session-macro NLL | task-macro NLL | session Δ vs full（95% CI） |
|---:|---:|---:|---:|---:|
| 0 | 1.534687 | 2.723367 | 3.055074 | +1.515237 [1.490267, 1.540207] |
| 1 | 0.791420 | 1.363479 | 1.615063 | +0.155350 [0.147264, 0.163435] |
| 2 | 0.759640 | 1.310735 | 1.527316 | +0.102605 [0.095651, 0.109560] |
| 5 | 0.730783 | 1.259755 | 1.435769 | +0.051625 [0.045937, 0.057314] |
| 10 | 0.713391 | 1.239525 | 1.395948 | +0.031396 [0.027188, 0.035604] |
| 20 | 0.703471 | 1.227265 | 1.363826 | +0.019136 [0.016020, 0.022251] |
| full | 0.690714 | 1.208129 | 1.296670 | 0 |

E5 主表按 participant/session → experiment → 75-experiment task-macro 聚合，交互
定义为 $G_{\mathrm{context}}(\mathrm{Minitaur})-
G_{\mathrm{context}}(\mathrm{base})$：

| $w$ | Llama base NLL | Minitaur NLL | base−Minitaur | base context gain | Minitaur context gain | context-gain 交互 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3.055074 | 2.650945 | +0.404129 | 1.758404 | 1.440453 | -0.317951 |
| 1 | 1.615063 | 1.526024 | +0.089039 | 0.318393 | 0.315532 | -0.002861 |
| 2 | 1.527316 | 1.437704 | +0.089611 | 0.230646 | 0.227212 | -0.003434 |
| 5 | 1.435769 | 1.347940 | +0.087828 | 0.139098 | 0.137448 | -0.001650 |
| 10 | 1.395948 | 1.309466 | +0.086482 | 0.099278 | 0.098974 | -0.000304 |
| 20 | 1.363826 | 1.275880 | +0.087946 | 0.067156 | 0.065388 | -0.001768 |
| full | 1.296670 | 1.210492 | +0.086178 | 0 | 0 | 0 |

participant×task 配对 bootstrap（seed=20260728，5,000 次）：

- $w=0$：base context gain 1.758404 [1.557747, 1.973043]；Minitaur
  1.440453 [1.253219, 1.645093]；交互 **-0.317951
  [-0.347411, -0.286952]**。
- $w=1$：交互 -0.002861 [-0.007126, 0.000997]。
- $w=5$：交互 -0.001650 [-0.004956, 0.001692]。
- $w=20$：交互 -0.001768 [-0.004561, 0.001030]。

**核心判定**：行为微调没有可检测地增强 $w\ge1$ 时的历史利用效率。两模型的
context-gain 曲线几乎重合；任务级 gain 的跨模型相关在 $w=1$ 为 0.9991，在
$w=20$ 为 0.9979。显著交互只出现在 $w=0$：Minitaur 的冷启动损失低很多。因此
微调更像是学到任务接口、response alphabet/格式校准和行为先验，减少对第一段
demonstration 的依赖，而不是创造一种更强的长上下文学习机制。

Llama base 从 $w=0$ 到 full 的 task-macro 总收益中，$w=1/2/5/10/20$ 分别捕获
81.9%/86.9%/92.1%/94.4%/96.2%，故主口径 EC90=$5$、EC95=$20$。两模型在
$w=20$ 后都仍有长上下文残余，不能把它归因于 Psych-101 微调。

E3 五锚点的 full 微调差为 0.086178 nat，但按位置拆开为 first 0.413605、second
0.006556、10% 0.008830、50% 0.006654、last 0.007032；总体差几乎完全由 first
anchor 的高权重拉高，不能与 E0 all-choice 的 ~0.014 nat 直接比较。只保留单-choice
目标时，$w=0$ 交互 -0.338390，而 $w=1/5/20$ 仍仅为
-0.003320/-0.000611/-0.001490，结论不依赖 multi-token response。

结论边界不变：以上只适用于同架构 8B、runtime NF4、teacher-forced NLL 和当前
prompt 重建/五锚点协议；不能外推成 Centaur-70B/BF16 的效应量，也不能从截断曲线
单独推断人类或模型的抽象记忆跨度。

**汇报材料（2026-07-28）**：四张主图 + 数字备忘（中/英）在 `outputs/figures/`
（fig1 E5 交互、fig2 瀑布分解、fig3 早期适应曲线、fig4 任务散点，
`RESULTS_SUMMARY{,_EN}.md` 含每张图的关键数字与必须口头声明的边界）。另有
逐任务附录 fig5–fig8（E0 点图、E1 适应降幅、E2 四基线、E3 截断代价热图）
与合并表 `outputs/analysis_per_task_report.csv`。图由
`scripts/experiments/build_report_figures.py`（综合图 fig1–4）与
`build_per_task_figures.py`（附录图 fig5–8）从 score CSV 一键重生成
（内置 collsiöö 名字归一化与 zorowitz UTF-8 替换）。E5 主表数字已独立复算
核对（key 逐行对齐，交互值一致）。

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

0. **⚡ 官方配置判定实验（最高优先级，PC 上跑；2026-07-30 汇报用现有结果，
   此实验汇报后立即启动）**：用 4-bit base + 官方 adapter 复跑 E0（见 §5
   官方对比节）。跑完派生 P0 对照官方 0.4555，判定 H1/H2；随后**必须**用
   同配置补跑 E3——E5 的"微调只除冷启动、$w\ge1$ 无交互"结论目前只在
   Minitaur 上成立，需要在真 Centaur-8B 配置上复检后才能写进论文。命令：
   `run_transcript_scoring.py --model meta-llama/Llama-3.1-8B
   --adapter marcelbinz/Llama-3.1-Centaur-8B-adapter --load 4bit
   --device cuda --chunk-size 1 --data
   data/psych-101-test/prompts_testing_t1.jsonl --output
   outputs/scoring/centaur8b_adapter_e0_full_4bit.csv --summary
   outputs/scoring/centaur8b_adapter_e0_full_4bit_summary.csv`。
0b. **官方逐 choice 数组的三个零算力分析（汇报后，材料已验证可用，见 §5
   pth 节）**：(i) E1 适应曲线加认知基线线（26 个长度对齐任务，Figure A
   三方齐）；(ii) 认知基线的分层 macro 聚合；(iii) fig10 加入四个 70B
   instruct 对照。unseen 任务对比需先读官方 custom metric 代码确定 span
   选择，单列一步。
1. **封装 E3/E5 固定分析产物与 Figure B/C（零 GPU）**：把双模型窗口主曲线、
   五位置 strata、participant×task bootstrap、单-choice 敏感性和 E5 交互导出为
   版本化 summary/figure；固定 seed=20260728、5,000 次。Figure B 必须显式画出
   $w=0$ 的交互块，不把它默认并入微调或 ICL 主效应。
2. **E1 正式版（零算力，随时可做）**：双模型适应曲线数据已在
   `outputs/analysis_llama_vs_minitaur_curve.csv`；正式版补 participant/task
   bootstrap CI。仍从无截断 E0 聚合，不要用 P0 替代（P0 尾部被截断）。
3. **E3a：拆分 $w=0\rightarrow1$ 的巨大收益（当前最高优先级的新实验）**：E5
   现已直接确认微调交互几乎全部集中在 $w=0$（0.318 nat），E3a 的字母表
   重归一化诊断正好能判定这部分里有多少只是合法按键先验——它现在同时回答
   "ICL 第一段收益的构成"
   和"微调到底教了什么"两个问题。设计已于
   2026-07-27 修订（design §7.4）。原 "label-space only" prompt 条件因在训练
   分布内没有合法形态被移除，字母表成分改为解析测量：每个条件同时算原始 NLL
   与限制到 session 合法 label token 集的重归一化 NLL，其差即字母表未发现惩罚
   （需打分器小改：目标位置额外 gather 合法 label 的 logprob；先限单 token
   response 任务）。prompt 阶梯剩四条件：instructions only / format-only /
   matched other-participant / own history，主分析在重归一化 NLL 上取相邻差。
   runner 写好后 Minitaur 与 Llama base 各跑一遍。
4. **E3b：暂缓**（design §7.5）：两模型 $w=20$ 后都有小残余，且交互近零；
   残余大的任务恰是 swap/control 最难合法实施的序列型任务。
   重启前先过四道门槛：零算力预分析（full−w20 差对位置/长度回归）、功效预算、
   有状态表面特征审计、shuffle-only 试点先行。预分析属零算力，可随时顺手做。
5. **E4**：语言表面扰动（逐任务设计，最费手工），可先只对试点任务做。
6. **HPC 精确对照（未来，需用户启动）**：在原生 BF16/FP16 环境上直接跑论文协议，
   用于与 runtime-NF4 P0 分开报告；不要把这一步自动并入本地作业。
7. **小修**：把 E0 CSV 里 `collsi枚枚2023MCPL` 的 experiment 名改回
   `collsiöö2023MCPL`（分数有效，仅名字误解码，见 §5）。

---

## 8. 起步要读的文件

1. `docs/centaur-eval-design.md`（§2.1 P0 协议控制轨道、§12 实验分解，§14 结论边界）
2. 本文件
3. `scripts/experiments/build_paper_style_nll.py`（P0 cutoff 审计、cache 复用和聚合）
4. `src/mt/evaluation/transcript_scoring.py`（打分核心，重点 `_forward`/`_score_batch`）
5. `scripts/experiments/_common.py`（`load_sessions` 顺序、UTF-8、`load_model`、续跑）
6. `docs/agents/CONVENTIONS.md`（代码风格：80 列、单空行、填满再折行等）
