# Centaur 阶段已完成分析汇总（§11.2 A1–A17）

**日期**：2026-08-13
**用途**：把散在 [Behaverse Benchmark v1](behaverse-benchmark-v1.md) §11.2 登记表各行、
以及 `outputs/analysis/` 各目录下的结果收拢成一份可独立阅读的记录。转向
[trial-level 自有模型](../benchmark%20design.md)后不再扩展这些分析，本文是它们的终点。
**与快照的关系**：[Centaur 阶段结果快照](centaur-results-snapshot.md)冻结的是主对照
（Centaur vs matched base）与产物处置决定；本文补的是**六个模型逐行的实测数字**，
以及每行分别能主张与不能主张什么。快照 §3 的限制在此全部继续有效。

## 0. 共同口径

- runtime **NF4/4-bit**，不得与 published BF16 放在同一结果列；
- 6 个模型 × **73 experiment × 6,499 session × 1,144,236 marked choice**；
- `enkavi2019gonogo` 按 §9 第 7 条排除（合法集只有 1 个元素），`xiong2023neural` 未跑；
- **A3 的 `comparability.csv` 为空**——六个 run 覆盖同一批 session，$L_f$ 逐任务可比。
  这是其余各行成立的前提，先看它；
- 聚合一律按 §4：choice 内求和 → session 内平均 → participant 间平均。跨任务均值只作
  定向参考，逐任务表才是结果。

## 1. 一句话

**六个模型的排序在每一行分析里都相同，而把它们分开的主要是答题接口，不是对人的理解。**
冷启动这一件事以四种互不相同的形式出现在 A4、A9、A14、A17 里；A13 与 A15 说明在接口
之外还剩一个较小、方向一致的差距。

## 2. 逐行结果

### A1 / A2 / A3 — 三个指标与计数基线

`condition=full`，73 个任务的 task-macro $L_f$：

| 模型 | $L_f$ (nat) | vs uniform | vs base rate | vs sticky | vs online bigram |
|---|---:|---:|---:|---:|---:|
| centaur8b | **0.73545** | 1.883× | 1.619× | 1.746× | **1.470×** |
| llama31_8b | 0.93579 | 1.551× | 1.383× | 1.378× | 1.280× |
| llama32_3b | 0.99021 | 1.462× | 1.320× | 1.289× | 1.184× |
| llama32_3b_instruct | 1.06784 | 1.369× | 1.228× | 1.174× | 1.141× |
| llama32_1b | 1.07836 | 1.408× | 1.272× | 1.140× | 1.101× |
| llama32_1b_instruct | 1.25578 | 1.265× | 1.112× | 1.054× | 1.004× |

$R_f$ 取逐任务中位数。Centaur 对 matched base 差 **0.20034 nat**（$e^{0.20034}=1.222\times$），
**73/73 个任务更好**；对 online bigram 在 **71/73** 个任务上更好。最弱的
llama32_1b_instruct 对 bigram 的中位比已降到 1.004×，即与一个二元计数模型基本持平。

### A4 — 上下文曲线（按锚点位置分层）

`target≥20` 的 16,901 个锚点，**排除 `frey2017risk`**（它一行内含一个气球中的大量
choice，marked segment 不是可比窗口单位）后 71 个任务：

| 历史窗口 $w$ | 0 | 1 | 2 | 5 | 10 | 20 | full |
|---|---:|---:|---:|---:|---:|---:|---:|
| centaur8b $L_f$ | 1.7174 | 1.4710 | 1.3229 | 1.1725 | 1.0784 | 1.0237 | 0.8520 |
| base − centaur | **2.4326** | 0.3074 | 0.2247 | 0.1955 | 0.2090 | 0.1932 | **0.1778** |

**一个 marked-choice segment 消除了 $w=0$ 差距的 87.4%**，其后 19 段历史合计只再贡献
约 0.75 nat，而残留 0.18–0.21 nat 跨窗口稳定。第二产出：`target=0` 层上 $w=0$ 与
$w=\infty$ 前缀逐字相同、理论差为 0，实测差即本批的逐 choice 噪声底（p95 ≈ 1e-2），
其余各行断言效应时以它为参照。

### A9 — 「没在答题」时输出了什么

`format_ok=False` 占 1,144,236 个 choice 的 5.1%–5.6%，但**其中绝大部分是结构性的**：

| 模型 | format_ok=False | 其中多-token 结构性 | 真·off_task | 对任务并集仍 off_task |
|---|---:|---:|---:|---:|
| centaur8b | 58,278 | 55,545 | **175** | 34 |
| llama31_8b | 59,280 | 54,789 | 1,281 | 1,003 |
| llama32_3b | 60,473 | 54,871 | 2,178 | 1,847 |
| llama32_3b_instruct | 61,184 | 54,506 | 3,355 | 2,906 |
| llama32_1b | 60,308 | 54,081 | 4,026 | 3,475 |
| llama32_1b_instruct | 63,588 | 54,077 | 3,960 | 3,592 |

14 个多-token 任务上只读一个 token 而 `format_ok` 比的是完整选项串（`77.37`、
`turquoise`），结构上恒为假——**那不是没答题**（§2.2、§9 第 8 条）。扣掉它以后，真正
脱离答题格式的比例是 centaur 0.015%、base 0.11%–0.35%，相差一个数量级。

### A13 — 答案分布 vs 人类真实分布

规范码空间，48/73 个任务、4,037 个 session。TVD $=\frac12\sum_o|h-m|$，session 内计算
后按 §4 聚合：

| 模型 | TVD ①（greedy 频次） | TVD ③（期望频次） | ① 最大频次 | ③ 最大频次 |
|---|---:|---:|---:|---:|
| centaur8b | **0.060** | **0.035** | 0.543 | 0.494 |
| llama31_8b | 0.110 | 0.049 | 0.579 | 0.493 |
| llama32_3b | 0.101 | 0.055 | 0.574 | 0.489 |
| llama32_3b_instruct | 0.119 | 0.070 | 0.584 | 0.494 |
| llama32_1b | 0.112 | 0.059 | 0.585 | 0.502 |
| llama32_1b_instruct | 0.110 | 0.083 | 0.560 | 0.502 |
| *i.i.d. 零模型 / 人类* | *0.054* | *0.038* | *0.513* | *0.503* |

Centaur 在 36/48（①）和 39/48（③）个任务上是六个里最接近人类的。**退化检测给的是
否定答案**：base 模型的 greedy 读数确实比人类集中（0.56–0.585 vs 0.513），但 ③ 上六个
模型全部 ≈0.49，与人类的 0.503 一致——押众数的倾向只存在于 argmax 读数里，概率本身
没有塌。这正是 §2.3 说只有画分布才看得出来的东西。

方法上值得留下的两件事：transcript 的行是 parquet 表行的**任务相关子集**（三条通用
规则 raw / `forced==0` / `code>=0` 覆盖 47 个 experiment，7 个对不上）；`pred_options`
每 trial 列的是 **session 并集**而非该 trial 的选项，多 block 任务的每个 trial 都带着
别的 block 的键，单独记为 `off_block_mass`（centaur 0.023，base 0.030–0.036）。

### A14 — 合法选项上的概率质量

63 个任务有 ③。按 session 内位置：

| `choice_index` | 0 | 1 | 2 | 3 |
|---|---:|---:|---:|---:|
| centaur8b | **0.948** | 0.944 | 0.962 | 0.965 |
| llama31_8b | 0.262 | 0.903 | 0.944 | 0.953 |
| llama32_3b | 0.331 | 0.873 | 0.907 | 0.935 |
| llama32_3b_instruct | 0.236 | 0.886 | 0.901 | 0.938 |
| llama32_1b | 0.111 | 0.884 | 0.889 | 0.921 |
| llama32_1b_instruct | **0.056** | 0.892 | 0.882 | 0.910 |

**冷启动时 base 模型只把 5.6%–33% 的概率放在合法选项上，一个 in-context 示例把它拉到
0.87–0.90**；Centaur 从第一个 choice 起就在 0.95。全程均值六个模型都在 0.979–0.993，
说明这是纯粹的开头现象。`jansen2021dunningkruger` 在第 24 个 trial 换答题格式，出现
二次塌方——逐任务图能看见，跨任务曲线看不见。

### A15 — 对人类的准确率

54 个 experiment（15 个无规范表，`hebart2023things` 与 `jansen2021dunningkruger` 的
规范并集比 session 并集更差而剔除）：

| 模型 | accuracy | chance | lift |
|---|---:|---:|---:|
| centaur8b | **0.7662** | 0.3705 | **2.87×** |
| llama31_8b | 0.7028 | 0.3705 | 2.58× |
| llama32_3b | 0.6867 | 0.3705 | 2.52× |
| llama32_3b_instruct | 0.6656 | 0.3705 | 2.44× |
| llama32_1b | 0.6656 | 0.3705 | 2.44× |
| llama32_1b_instruct | 0.6342 | 0.3705 | 2.28× |

Centaur 在 54/54 个任务上高于 base，六个模型全部高于随机。**`chance` 必须用规范表的
选项数**：按键逐参与者随机化，98% 的 session 并集小于其任务，196 个 session 只有 1 个
选项（此时高于随机结构上不可能），旧口径把 lift 夸大 82%（5.23 → 2.87）。

### A17 — session 内学习曲线

`condition=full`，按 session 内十分位的 $L$：

| 十分位 | 0 | 1 | 4 | 7 | 9 | 0→9 落差 |
|---|---:|---:|---:|---:|---:|---:|
| centaur8b | 0.905 | 0.762 | 0.667 | 0.623 | 0.792 | **0.113** |
| llama31_8b | 1.259 | 0.969 | 0.837 | 0.798 | 0.972 | 0.287 |
| llama32_1b_instruct | 1.803 | 1.239 | 1.071 | 1.050 | 1.321 | 0.482 |
| bigram（计数基线） | 1.445 | 1.307 | 1.227 | 1.201 | 1.404 | 0.041 |
| base_rate（计数基线） | 1.480 | 1.404 | 1.409 | 1.384 | 1.709 | −0.228 |
| uniform / sticky | — | — | — | — | — | ≈0 |

**几乎全部改善发生在第一个十分位**：llama31_8b 仅 0→1 一段就下降 0.290，比 0→9 的总
落差 0.287 还大，此后曲线是平的。**第 9 个十分位的回升不是效应而是构成变化**——计数
基线在同一位置同样回升，是长 session 与短 session 的混合比例在变。对照组（论文的认知
模型全体共享一套参数、结构上被禁止个体适应）在此是平的。

## 3. 反复出现的同一件事

四行分析从不同角度指向同一个结论：**模型的冷启动是格式失败，不是知识失败。**

- A14：第一个 choice 上 base 只有 5.6%–33% 的概率落在合法选项内；
- A4：$w=0$ 时差 2.43 nat，一个 segment 后掉到 0.31；
- A17：session 内的改善几乎全部发生在第一个十分位；
- A9：真正脱离答题格式的 choice，centaur 175 个 vs base 1,281–4,026 个。

Centaur 的 adapter 主要买到的就是这一件事。但它不是全部：残留的 0.18–0.21 nat 跨窗口
稳定（A4），A13 的分布贴合与 A15 的准确率优势也在接口之外，只是量级远小于冷启动项。

## 4. 每行不能主张什么

沿用快照 §3，逐行展开：

| 行 | 不能主张 |
|---|---|
| A1/A2 | NF4 数字不能与 published BF16 并列；跨任务均值不是主结果，逐任务表才是 |
| A4 | 曲线是观察性的，$w$ 与锚点位置的分层已做，但任务难度与 session 长度仍混杂 |
| A9 | 分类是对 `raw_generation` 的启发式归类，不是标注；类别边界未做一致性检验 |
| A13 | **只说明 session 级边际分布的贴合**。低于 i.i.d. 零模型不能证明逐 trial tracking——需要 trial shuffle / lag 破坏或逐 trial proper-score 对照，本批没做。另外只覆盖 48/73 个任务，且规范码对齐的自检只在人类按过 ≥2 个键的 block（54.9%）上有效 |
| A14 | 分母/聚合不规范（合法集是打分时冻结的 session 并集，且为 micro/session-weighted），不能当 task-macro 主结果 |
| A15 | 只覆盖 54 个任务，且 mixed multi-token 任务只用了单-token 部分，不能当 73-task leaderboard |
| A17 | 观察性 position curve，trial 顺序、难度与参与者流失混杂，不是因果 in-context learning 证据 |
| 全部 | 新批次尚未重新完成 official 36-family `r=1.00000` 对拍（旧批次做过，新旧共享任务分数高度一致） |

## 5. 没做的行

A8 / A10 / A11（答题率的总体、逐任务、逐窗口三个切面，需先统一「没在答题」口径）、
A12（温度校准，本可把 0.200 nat 拆成判别力与校准误差）、A16（任务的历史依赖度剖面与
Xie & Zhu 三分类的预注册对照）、A18（$L_f$ 矩阵的模型间相关与主成分）、
A19（合法选项在 top-20 中的排名，需在集群上聚合 `pred_topk`）。
§11.1 另有两项待补：多-token 任务的 continuation pass，以及 `legal_mass` 的 re-score。

**这些缺口正是停止扩展的理由**：补完能让 Centaur audit 更严谨，但不会转化为自有训练
方法或可识别的科学发现。

## 6. 产物位置

```
outputs/analysis/trackp-greedy1tok-v1-8c072183d8d6/
  metrics_by_task.csv  probability_ratio.csv  comparability.csv   A1 A2 A3
  a4/   a4_curve_unsaturated.csv  a4_curve_by_anchor.csv  + 3 图    A4
  a9/   a9_categories.csv  a9_totals.csv  a9_examples.csv          A9
  a13/  a13_by_task.csv  a13_distribution.csv  a13_alignment.csv + 3 图
  a14/  a14_by_task.csv  a14_by_position.csv  + 2 图
  a15/  a15_accuracy_by_task.csv  + 1 图
  a17/  within_session_curve.csv  + 3 图
outputs/baselines/psych-101-test/sequence_online{,_summary}.csv    A2
```

脚本一行一个，都在 `scripts/experiments/`：`build_benchmark_metrics.py`、
`build_context_curve.py`、`build_offtask_readout.py`、`build_answer_distribution.py`、
`build_legal_mass.py`、`build_accuracy.py`、`build_within_session_curve.py`。
规范码的对齐与按键字典在 `mt.models.baselines.canonical_tables`
（`align_sessions` / `label_map`），带测试，**是本阶段唯一对新路线仍可能有用的代码**：
任何需要跨参与者选项同一性（而不只是选项计数）的分析都要用它。

按快照 §4 的处置原则，上表中 A13/A14/A15/A17、A9 与含 Frey 的旧 A4 图表可删；本文
保留了它们的结论与数字，删图不丢结果。
