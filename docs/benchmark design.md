# P500 Trial-Native Behavior Modeling 三篇论文路线图（当前）

**日期**：2026-08-13
**状态**：当前设计记录（design of record）
**历史阶段**：[Behaverse Benchmark v1](archive/behaverse-benchmark-v1.md)
**历史结果**：[Centaur 阶段结果快照](archive/centaur-results-snapshot.md)
**论文 A 设计**：[Trial-Native State vs. Language Serialization](p500-structured-input-paper.md)
**论文 B 设计**：[Shared Response Dynamics Across Cognitive Tasks](p500-choice-rt-paper.md)
**论文 C 设计**：[Unity and Diversity of Executive Control Across Language Models](p500-llm-executive-function-paper.md)

本文负责三篇论文共享的数据协议、边界、先后顺序和清理策略；三份 paper design 负责各自的
具体条件、estimands 与停止规则。发生冲突时，shared contract 以本文为准，论文内实验选择
以对应 paper design 为准。

## 0. 研究决策

从本设计开始，新的核心任务、行为训练数据和模型施测都来自自己的 **P500**。Psych-101 与
Centaur 不再进入后续实验矩阵；它们只作为已经冻结的背景。论文 C 允许少量预注册、协议
冻结的外部 benchmark outcomes 作为 criterion validity，但它们不参与 P500 因子定义、选题
或模型选择。旧阶段显示 full-transcript teacher forcing 混合了答题接口、历史利用、校准与
行为预测，继续扩大旧 benchmark 不能直接告诉我们自己的模型应当怎样训练。

项目推进三个基于 P500、但具有独立科学命题的研究：

1. **论文 A：Trial-native state vs. language serialization。** 把异质认知任务压成
   自然语言，是否改变行为模型的归纳偏置、数据效率与跨任务泛化？主比较固定为
   choice-only，在等信息、同输出头条件下比较 text view 与 structured view。
2. **论文 B：Shared response dynamics across cognitive tasks。** 能否建立一个跨任务、
   同时解释 choice、RT 与未反应事件的模型，并证明共享的时间动力学可以迁移到新参与者、
   低数据任务或未见 task family？它不是“给 choice 模型多加一个 RT head”的文章。
3. **论文 C：LLM Unity–Diversity of executive functions。** 大量现有 LLM/checkpoints
   在无泄漏 P500 agent battery 上的行为协方差，是否复现人类执行功能的 unity/diversity
   结构？这些维度是否跨 model lineage 稳定，并对交互式外部能力提供增量预测？

**历史预算**只是一项辅助诊断。若进入论文 A，必须对 text 与 structured 都比较相同的
short/full 条件，检验 representation×history，而不能只在目标模型上改变历史；它不构成
独立主线。论文 B 固定一个历史预算，不把 context 长度与 response dynamics 混在一起；
论文 C 根据每个范式的理论需求固定可见历史，不用 Paper A 的 context ablation 代替记忆测验。

论文 A/B 共享 P500 Trial-State、participant split、candidate-action interface 与部分 backbone
代码。论文 C 只复用 task ontology、pre-response state、canonical actions 和泄漏检查，另建
agent-facing battery；它把 model checkpoint 而不是 human participant/trial 作为协方差分析
单位。默认顺序是先完成论文 A、冻结结构化接口，再正式开展论文 B；RT 字段审计与论文 C
的 construct/interface audit 都可并行进行。

目标是建立自己的模型与可证伪结论，不再写一篇 Centaur 检验文章。

---

## 1. 已有结果与转向依据

旧阶段最稳的初步结果是：本地 8B/NF4 下，Centaur full-context task-macro NLL 为
0.73545，matched Llama-3.1-8B base 为 0.93579，Centaur 在当前 73/73 个任务上更好。
排除 multi-token choice 任务后差距仍为 0.16488 nat，59/59 个任务更好。

但修正 E3 的任务口径后，base−Centaur 差距从无历史时的 2.4326 nat，在看到一个
marked-choice segment 后降到 0.3074 nat；到 full history 仍有 0.1778 nat。第一 choice
上 base 只把约 0.262 概率质量放在观测到的答案集合，一个示例后变为 0.903。这说明最显眼
的差距很大程度与冷启动/答题接口有关，同时保留一个较小 residual；它不能说明 residual
对应什么认知机制，也不能给出训练方法。

P500 允许直接避开这个识别问题：原始实验数据有明确的 trial、stimulus、option、response、
session 与 RT 字段，不需要从自然语言 token 或 `<<>>` marker 反推决策边界。Centaur 结果
以后只承担“为什么从 transcript 转向 trial-native 建模”的动机作用。

---

## 2. P500 的现有基础与限制

当前转换产物覆盖：

- 100 名参与者；
- 21 个 task families、62 个 instruments；
- 6,588 个 participant×session×task runs；
- 777,403 个 trials、934,410 个 choice targets；
- 标准表 1,008,651 个 decision-step rows，`trial_uid` 无重复。

现有 `standard_trials.csv` 已包含 participant/session/task/run/trial/step IDs、反应、正确性
和 `response_time_ms`。但它是为自然语言渲染建立的：实际 stimulus/option 的一部分结构已被
压成 `stimulus_text` / `options_text`，不能把现有 JSON 字符串序列化直接称为“结构化模型”。
新的数据版本必须从 P500 原始表重新保留 typed state。

RT 也不是每个 choice row 都有可直接使用的标签。初步审计中，73.59% 的 choice rows 有
大于 0 ms 的数值，25.01% 为 null，另有 PC family 的 13,080 个 0 ms 值需要核查。这里
“大于 0”只是因为后续要建模 $\log RT$，并不表示数据中出现了“负反应时”；需要处理的是
缺失值和 0。Sequence tasks 多数只有整个 trial 的 RT，当前被放在第一个 press，后续 step
为空。因此 RT 覆盖率应在 trial 和 family 层重新统计，不能把空值当 0，也不能声称每个
decision step 都有独立 RT。

---

## 3. 三个关键操作定义

### 3.1 Text 与 structured 必须等信息

从同一个 canonical $S_t$ 生成两个 view：

- **text view**：把字段确定性渲染为自然语言；
- **structured view**：直接编码 typed categorical/numeric/set/sequence fields，不先变成
  自然语言或 JSON tokens。

两者必须包含相同信息，使用相同 canonical action IDs、candidate-action head、训练 targets、
数据 split 和预算。Text 与 structured 的差异只落在输入 adapter；共享同一序列 backbone
和对应论文的输出 heads，并尽量匹配 trainable parameters 与 compute。论文 A 固定
choice-only；论文 B 不再重新比较这两个 views。

若 text 用自由 token generation、structured 用合法 action classifier，结果首先反映输出
接口，不是输入表示。主实验统一使用 candidate-action scoring；raw token generation 仅在
未来确有需要时作为单独接口诊断。

### 3.2 论文 B 建模 response process，不只把 RT 当 auxiliary loss

当前 trial 的 RT 永远只在 target 侧出现，不能把“after 600 ms”写回当前输入。一个可实现的
起点是：

$$
p(y_t\mid S_t)\;p(\log RT_t\mid y_t,S_t),
$$

其中 $y_t$ 是 choice 或同一 trial 内的 choice sequence。Choice head 对合法 actions 做归一化；
RT 部分预测完整条件分布，辅以 log-RT MAE 和 quantile calibration。正式论文还必须显式处理
response/no-response/timeout 等 event；若 deadline 已知，未发生反应不能一律作为普通 missing
row 丢弃，而应使用相应的 censoring 或 event likelihood。

- 只对有效且大于 0 ms 的 observed RT 计算 log-RT loss；不填补 null 或 0；
- sequence task 的 RT 按 trial-level target 计算一次，不复制到每个 press；
- 合法 no-press 是显式 action；invalid timeout/omission 单独标记，不再因为“没有 token”丢掉；
- PC 的 0 ms 与各 family 的计时语义在训练前完成 QA；
- 若 joint training 改善 choice，再用 task×condition 内 shuffled RT 和 matched auxiliary
  control 检验增益来自 RT 的行为内容，还是任意 auxiliary loss 的正则化；
- 论文 B 的主要成功条件是 RT/event 分布拟合和跨任务迁移。Choice 是否改善是重要但非必要的
  次级问题，不能用一个 choice NLL 增益代替 response-dynamics 证据。

### 3.3 历史预算是辅助条件

对 trial $t$，定义：

- $S_t$：参与者在作答前真正可见、且完成当前反应所必需的任务状态；
- $H_t^{(k)}$：此前 $k$ 个已完成 trials 中的人类 choice、RT、feedback/outcome 历史。

历史条件改变 $k$，但始终完整保留 $S_t$。以下内容属于 task state，不能被当作“历史”删掉：

- n-back 所需的前 $n$ 个刺激；
- sequence recall 当前 trial 的记忆序列，以及同一 trial 内已生成的 response prefix；
- adaptive task 当前实际难度/时间限制；
- task instruction、block/phase、当前 stimulus 与合法 action set；
- 多阶段 trial 当前 step 之前的同一 trial 状态。

历史实验只在论文 A 的核心 representation effect 稳定后追加。若追加，text 与 structured
都比较一个预先固定的 short-history 条件与 full-session 条件；具体 $k$ 在 Trial-State v1
审计后冻结。若需要解释，再增加 $k=0$ 或 $k=1$，不重新铺开
`0/1/2/5/10/20/full` 窗口曲线。论文 B 全程固定同一个 $K_{ref}$。

不同 history 条件必须使用相同总 target trials、优化步数和有效 choice targets。“每个样本
含几个历史 trials”不得与“模型总共见过多少训练 trials”混在一起。

---

## 4. P500 Trial-State v1

训练前先从原始 stimulus/option/response 表生成新的 choice-level 或 trial-step-level 数据集。
每条记录至少包含：

### 4.1 只用于索引和 split

- `participant_id`, `session_id`, `task_run_id`；
- `task_family`, `task_id`, `block_id/phase`；
- `trial_uid`, `trial_index`, `trial_step_index`。

这些 ID 默认不作为主模型输入；尤其 participant ID 不进入 state-only 主条件。

### 4.2 输入 state

- instruction/rule ID 与必要 task parameters；
- 当前 stimulus 的 typed fields；
- candidate actions 及 mask；
- 当前 block/condition/adaptive state；
- family-specific minimal sufficient state；
- 同一 trial 内必要的 step prefix。

每个 family 必须登记“充分状态”与来源列。不能用 `expected_response`、`correct`、当前/future
response、当前 RT 或作答后 feedback 构造输入。

### 4.3 Targets 与 observation mask

- observed action/action sequence；
- trial-level valid RT（大于 0 ms）和 `rt_observed`；
- no-press、timeout、skip/censoring 状态；
- correctness 只用于行为现象分析，不作为 choice target 的替代。

Text view 必须从上述 state 自动生成；不维护一个信息不同的手写文本数据集。每个 prediction
都保留 `trial_uid`，不再仅靠 marker 与 RT list 的位置对应。

### 4.4 P500 Agent Battery v1 是独立 view

论文 C 不能直接把上述“预测人类反应”的 records/renderer 当作给 LLM 施测的材料。它从同一
task ontology 和 pre-response state 出发，但另建无泄漏、可交互的 agent-facing view：不含
expected response、correctness、答案反推描述或未来反馈；视觉/记忆/自适应任务保留真实信息
约束，并由被测模型自己的反应推进状态。

Agent Battery 还要保存 model revision、base lineage、interface、prompt、decoding seed、
parallel form、item/condition 和 response。LLM wall-clock latency 不映射成人类 RT。论文 C
的完整 contract 与 construct map 由对应 paper design 负责。

Primary Paper C battery 只包含能无损符号化、且不会改变核心 manipulation 的范式，并使用
可审计的 stateful streaming 执行 updating tasks：stimulus 只呈现一次，后续不重放 transcript
或提供外部 scratchpad/retrieval。真实视觉任务与 VLM 另作 multi-group sensitivity，不和
text-capable checkpoints 的非重叠题目直接混入一个主因子模型。

---

## 5. 三篇论文的实验边界

不再把 text/structured × choice/choice+RT 四个 cell 当作必须完成的单篇 $2\times2$。
论文 A/B 只共享一个 **S-C（structured + choice-only）锚点**和底层数据/模型接口；没有
独立科学用途时，不自动运行 text+RT 条件。论文 C 使用独立 agent battery 和多模型 roster，
不把 A/B 的训练 checkpoints 当作足以估计因子协方差的样本。

### 5.1 论文 A：Trial-native state vs. language serialization

**问题**：把相同的 task state 序列化为语言，是否改变多任务行为模型的归纳偏置、数据效率
与对未见 instrument/task family 的泛化？

先做一个主 backbone、一个规模、3–5 seeds，所有条件固定 choice-only 和同一个历史预算
$K_{ref}$：

| ID | 输入 | 监督 | 作用 |
|---|---|---|---|
| A-T1 | canonical text renderer | choice | 主要文本参照 |
| A-T2 | 第二个等信息 text renderer | choice | 排除单一措辞/模板效应 |
| A-S | typed structured state | choice | trial-native 系统 |

三者使用同一个 canonical state、candidate-action head、participant split、target trials、训练
更新数与 choice loss。主结果按相同训练 examples 比较，并补一个尽可能匹配 compute/FLOPs
的敏感性分析；text 通常更长，两种公平口径必须分开报告，不能假装它们天然等价。

论文 A 的最小完整证据还包括：

- held-out participants，以及至少一个真正的任务迁移层：held-out instruments，最好再有
  grouped held-out families；
- 预先冻结的数据量曲线，回答 representation 是否改变 sample efficiency；
- family-level effect 与 state-field ablation，说明哪些任务结构产生差异；
- 至少一个第二 backbone、规模或 pretrained/from-scratch 条件上的定向复现；
- 两个 renderer 下方向一致。只在一个模板上成立时，结论降级为 renderer effect。

RT target 不进入论文 A 的主训练矩阵。如果追加历史条件，必须形成
text/structured × short/full 的完整交叉；只改 A-S 不能支持 representation×history 结论。

### 5.2 论文 B：Shared response dynamics across cognitive tasks

**问题**：选择、反应时间与未反应事件是否包含跨认知任务共享、可迁移的 response dynamics？

论文 A 完成后冻结结构化 view、choice interface 与一个主 backbone。论文 B 固定输入表示和
历史预算，不再把 text/structured 当主轴：

| ID | 输入 | 监督 | 作用 |
|---|---|---|---|
| B-C | structured | choice | choice-only 锚点，可复用 A-S |
| B-R | structured | RT/event | response-process 单目标参照 |
| B-CR | structured | choice + RT/event | 联合模型 |
| B-CR-shuffle | structured | choice + shuffled RT/event | 行为内容控制 |

根据首轮结果增加 matched random auxiliary target 或 stop-gradient/separate-head control，区分
共享表征、额外参数与普通正则化。RT/event 模型至少与 task×condition 的
shifted-lognormal/lognormal 分布、简单 structured regression/MLP 和 independent per-family
models 比较；DDM/LBA 只用于语义适用的二选一速度任务，不强行覆盖全部 families。

论文 B 的主结果不是“加 RT 后 choice NLL 有没有下降”，而是：

- held-out participant 上的 RT/event predictive density 与分布校准；
- shared multitask model 相对 independent/per-family model 的低数据收益；
- 对未见 instrument 或 held-out family 的时间动力学迁移；
- condition、difficulty、accuracy–RT/speed–accuracy、switch/congruency 等适用行为效应；
- no-press、timeout/omission 和 sequence trial-level RT 的正确概率语义。

Choice gain 是有价值的次级结果；即使它为零，只要存在稳定、校准且可迁移的共享时间动力学，
论文 B 仍可成立。反之，若 RT 只在已见任务上容易拟合、没有迁移或独立行为发现，就不足以
包装成跨任务 response-dynamics 论文。

### 5.3 论文 C：LLM Unity–Diversity of executive functions

**问题**：跨相对独立 LLM/checkpoints 的 P500 行为协方差，是否更符合单因子、原始
Inhibition/Updating/Shifting 三相关因子，还是修订的 Common EF + Updating/Shifting-specific
结构？这些维度能否与人类 measurement structure 对应，并在 model lineage 外提供外部效度？

当前阶段包括：

- 逐范式完成 task→construct 与泄漏审计，每个构念至少 3 个不同范式；
- 从 task definitions 构建 compact parallel agent battery，而不是运行全部人类 trials；
- 先用 6–10 个结构差异明显的 checkpoints 做 C0 interface/state/leakage smoke；
- 先用 30–50 checkpoints 做 reliability、ceiling/floor、method variance 和 power pilot；
- 正式 roster 以约 200 checkpoints、至少约 25–30 base lineages 为预算起点，最终由
  Monte Carlo power 冻结；
- 用 trial-level confirmatory MIRT/GLLVM 比较 unitary、correlated-three-factor、修订
  Unity–Diversity 与 method-factor models；
- 检验 human–LLM factor congruence；只有 linked-scale gate 通过才检验
  configural/metric/partial invariance；并估计控制 common ability、规模、release/training
  type 和 lineage 后的外部增量预测。

排行榜只承担 criterion validity，不单拆文章。普通 prompts/seeds 是测量重复而非独立被试；
因子结构描述模型群体的表现协方差，不证明单个 LLM 内部具有对应的人类认知机制。
External headline 预先限定为 interactive/dynamic outcomes；传统知识、数学、语言 composite
主要作为 common-capability control 与次级结果。

### 5.4 明确不做

三篇都不重新引入 Psych-101 或 Centaur 新分析。论文 A/B 不恢复旧大模型 roster、70B、
完整 context sweep、open-loop 或经典模型大全；论文 A 不同时比较所有
ICL/LoRA/full-finetuning 组合，论文 B 不把所有 family 强塞进同一种 RT likelihood。论文 C
当前不开发 nonlinear CFA/VAE 方法论文，不抓取大量排行榜事后筛选相关，也不用 API latency
冒充反应时。若 A/B 的统一模型出现稳定负迁移，再测试一个 modular/expert 方案。

---

## 6. Splits 与评估

### 6.1 主 split：held-out participants

100 名参与者按 participant 分组为 train/validation/test；同一参与者的所有 session、tasks
只能进入一个 split。最简单可靠的实现就是在训练前冻结一份
`participant_id → train/validation/test` manifest，再用它生成三个数据集；标准化可以先做，
但 sample/window/rendering 与训练 loader 都只能读取已经分好的 records。现有 finetuner
内部的随机 row split 必须关闭。论文 A/B 学习的是 population-level
$p(behavior\mid state,history)$，不是依靠 participant identity 做个体化。

### 6.2 论文 A/B 的任务迁移检验

论文 A 先检验 held-out instrument（family 已见、具体 instrument 未见），再做预先冻结的
grouped held-out-family folds。论文 B 只在 RT 语义和覆盖率通过 gate 的 families 上检验
low-data instrument 与 held-out-family response-dynamics transfer。Family 的 state schema 和
合法 action 编码可以已知，但对应 family 的人类训练 targets 不得出现。

只有任务层迁移成功时才使用“跨任务 foundation/shared dynamics”表述；仅
held-out-participant 成功时，只称多任务 population model。

### 6.3 论文 C 的样本与验证 split

论文 C 的独立单位是 model checkpoint，base lineage 是聚类/外推单位。Prompt、seed、量化或
同一 base 的微调不能当作独立模型。Pilot roster 与 confirmatory roster 分开；正式 model/item
test 不能参与 item 筛选。主要泛化使用 leave-one-lineage-out，条件允许时再保留未来发布模型
作 temporal holdout。

C1 后冻结 target population、lineage 定义、每 lineage 配额与分析权重；当前预算默认每个
base lineage 最多 8 个 C2 checkpoints、任一 lineage 不超过总分析权重的 5%，最终由显式包含
within-lineage ICC 的 Monte Carlo power 决定。C2 主样本必须有 immutable revision/checksum；
无法冻结的 API 只作补充观察。

每个 lineage-held-out fold 中，measurement parameters 只从其余 lineages 学习；新 checkpoint
只用 Form A/anchor items 校准 factor score，再预测 Form B/held-out items，不能用同一 responses
同时估计和评价 latent score。

同一 P500 人类样本用于 multi-group measurement comparison；现有 100 人是否足以支持复杂
invariance constraints 必须通过与实际模型匹配的 Monte Carlo calibration 判断，不能因为
trial 多就忽略人/模型层样本量。

现有人类与 LLM 若使用不同 modality/renderer，只支持 indicator-level factor congruence。
Metric/partial invariance 只有在共同 anchor items、共同 task-relevant information，并通过
same-modality 或 linked-form calibration 后才进入正式结论。

### 6.4 聚合

- 论文 A：choice categorical NLL 为主，accuracy、calibration、sample/compute efficiency 为辅；
- 论文 B：RT/event held-out log predictive density 和 calibration 为主，choice NLL、
  log-RT MAE/quantiles 为辅；
- 论文 C：item-response predictive density、factor variance/correlation、measurement
  comparability（linking gate 通过时才含 invariance/DIF）、factor reliability 与外部增量预测；
- 论文 A/B 先在 participant 内聚合，再到 task/instrument 和 task-family；论文 C 的重复
  responses 先形成 checkpoint×paradigm 测量，再以 base lineage 为聚类与外推单位；
- choice 与 RT 分开报告，不拼成单一总分；
- 论文 A/B 的 task family/seed 与论文 C 的 lineage/form 是推断重点；不能把百万个 trial
  当作百万个独立重复。

论文 A 的简单 baseline 是 task×condition base-rate choice 与简单 state classifier/sequence
model。论文 B 另加 task×condition RT distribution、structured RT regression、independent
per-family model，以及少数适用任务的认知 timing baseline。论文 C 的基线包括 task score
CFA、单因子 generalized model、raw task totals/PCA，以及带 lineage controls 的外部预测。

---

## 7. 三篇论文的主要 estimands

论文 A/B 的 loss 差异在同一 held-out human records 上配对计算，loss 越低越好。论文 C
比较模型结构、跨群体对应和 lineage-held-out 预测，不把一组 CFA fit index 当作单一胜负指标。

### 7.1 论文 A：representation、效率与迁移

$$
\Delta_{struct}^{choice}=L_{choice}(\text{T-C})-L_{choice}(\text{S-C})
$$

其中 text renderer 1 和 2 分开报告，不挑选对 structured 最有利的一个。相同差异还要在
预先冻结的数据量、held-out instrument/family 与第二 backbone 条件下估计。只有两个 view
等信息、同 head、同预算时，才能把它解释为两个完整输入系统的差异；由于语言预训练、序列
长度和初始化仍不同，不声称识别了抽象 representation 的纯因果效应。

### 7.2 论文 B：RT skill、共享动力学与 choice 增量

$$
\Delta_{RT\text{-}skill}
=L_{RT}(\text{task/condition baseline})-L_{RT}(\text{B-CR}),
$$

$$
\Delta_{joint}^{choice}=L_{choice}(\text{B-C})-L_{choice}(\text{B-CR}),
$$

$$
\Delta_{RT\text{-}content}^{choice}
=L_{choice}(\text{B-CR-shuffle})-L_{choice}(\text{B-CR}).
$$

共享动力学还通过 low-data/held-out-family 中 shared model 相对 independent per-family model
的 RT/event predictive density 衡量。若 choice gain 为正但不胜 shuffled/matched auxiliary
control，只称普通 auxiliary regularization。若 choice gain 为零，但 RT/event 分布校准和
跨任务迁移成立，论文 B 仍然具有独立结果。

### 7.3 论文 C：执行功能结构、跨群体对应与外部效度

论文 C 不用一个“CFA 是否显著”作为单一 estimand。预先登记：

1. unitary、correlated-three-factor、修订 Unity–Diversity 和 method-factor models 的
   leave-one-lineage-out item-response predictive density；
2. Common EF/specific-factor variance、factor correlations 和跨 parallel forms 的稳定性；
3. human–LLM factor congruence；仅在 linked-scale gate 通过后估计
   loading/discrimination 差异与 DIF；
4. EF-specific factor posterior 在 common ability、规模、release/training type 和 lineage
   controls 之外的增量 out-of-sample predictive log score/$R^2$。

Factor scores 的 posterior uncertainty 必须进入外部预测；全样本两阶段相关只作描述。

### 7.4 论文 A 的可选历史诊断

$$
I_{history}=
[L(\text{text, short})-L(\text{text, full})]
-[L(\text{structured, short})-L(\text{structured, full})]
$$

四个条件使用相同 target trials 和训练预算，并分别在 short/full test context 下报告，
而不是只在各自最有利的测试条件下比较。这个结果回答语言化和结构化系统对 session context
的依赖差异；若不完整交叉，就只作为目标模型 robustness，不用于 representation 主张。

---

## 8. 结果解释与停止规则

### 8.1 论文 A

| 结果 | 可支持的解释 |
|---|---|
| structured 在两个 renderer、任务迁移和复现条件下稳定更好 | trial-native interface 改善 P500 choice 学习与泛化 |
| 只胜一个 renderer | 主要是模板/序列化实现差异，不能概括为 text vs structured |
| seen task 更好、held-out instrument/family 不更好 | 改善任务内拟合，尚无跨任务归纳偏置证据 |
| text 与 structured 在预定等效区间内 | 语言化未造成实质损失，typed interface 不是必要条件 |
| text 更好 | 语言预训练先验的收益超过序列化代价 |

### 8.2 论文 B

| 结果 | 可支持的解释 |
|---|---|
| RT/event 分布校准且 shared model 在低数据/held-out family 更好 | 存在可迁移的共享 response dynamics |
| joint choice 增益且胜 shuffled/matched auxiliary | 真实 RT 含 choice-only 未表达的可复用监督信号 |
| RT 可拟合但没有迁移或行为效应 | 只能称已见任务上的 timing prediction，不够支持共享动力学 |
| 只有少数 family 有稳定效应 | timing structure 是 family-specific，不包装成统一机制 |
| joint 不改善 choice | 不否定 RT 论文；choice 与 timing 可以共享有限，也可能需要不同表征 |

### 8.3 论文 C

| 结果 | 可支持的解释 |
|---|---|
| 修订 Unity–Diversity 稳定胜出，且 linked-scale gate 后至少部分 metric invariance | 跨模型表现具有与人类理论对齐的 unity/diversity structure |
| 单因子足够、specific variance 接近零 | P500 上的模型差异主要是 common/general control axis |
| 加 method factors 后 EF structure 消失 | interface、modality 或 task form 主导原有协方差 |
| linked-scale gate 通过但 human–LLM invariance 失败 | LLM 的 performance structure 不复现人类测量结构；仍是可报告发现 |
| linked-scale gate 不成立 | 只报告 factor congruence，不把 renderer/modality 差异解释成群体 DIF |
| specific factors 在 lineage 外增量预测交互任务 | 因子具有 criterion validity，但不等于相同内部机制 |
| pilot 普遍 ceiling/floor、低信度或指标不足 | 不扩 full roster，先修 battery 或缩小理论问题 |

不能由上述结果声称 structured representation 更像人脑，或直接把 RT 头解释为 processing
speed、effort、evidence accumulation。论文 B 若要做机制主张，必须增加能区分机制的实验与
认知模型对照；论文 C 的因子结构也只描述跨模型表现协方差，不等于相同内部机制。

---

## 9. 开跑前的工程 gate

1. 把 P500 converter、schema、family sufficient-state registry、QA tests 从被忽略的
   `data/` 目录迁入 tracked code/docs；原始私有数据继续忽略。
2. 生成带 choice→`trial_uid/trial_index/trial_step_index` 映射的 Trial-State v1；保存
   schema version、row counts 和 checksum。
3. 结果表使用 `task_run_id` 作为 run key。旧 scorer 只使用 `(experiment, participant)`；
   同一个人若在两个 session 做同一任务，两次运行会被当作同一个。当前 6,588 个实际 runs
   因此只能形成 5,497 个旧 key，少区分 1,091 个 runs。这只是旧评分代码的索引问题，新的
   P500 evaluator 从一开始保留 `task_run_id` 即可。
4. 训练前冻结 participant split manifest，生成独立 train/validation/test records，并加零
   participant leakage 断言。
5. 为每个 family 验证 $S_t$ 足够：特别是 n-back、sequence recall、adaptive tasks 与
   no-press/timeout。
6. **论文 A gate**：实现 shared candidate-action head、canonical renderer、第二 renderer 与
   typed structured adapter；通过等信息审计、参数/compute 记录、小数据过拟合、label shuffle
   和 base-rate baseline 后才提交正式训练。
7. 冻结论文 A 的 structured state、action head、$K_{ref}$ 和主训练配置；论文 B 只能用
   train/validation 选择 RT objective，不能根据同一 test split 的结果继续挑输入表示。
8. **论文 B measurement gate**：逐 family 登记 RT 起点、终点、deadline、no-press、timeout、
   omission、sequence-total-RT 与异常值规则；解决 PC 0 ms，不能用一个全局 mask 掩盖不同
   测量语义。
9. **论文 B model gate**：实现 RT/event distribution、censor/event likelihood、分 family
   coverage/calibration report，以及 B-C/B-R/B-CR/B-CR-shuffle 的容量和训练预算记录；通过
   toy likelihood、极端值和 missingness tests 后才提交正式训练。
10. **论文 C construct gate**：冻结 task→Inhibition/Updating/Shifting map；每个因子至少三个
    无泄漏、不同范式、可重建 stimulus 且有人类 reliability 依据的指标。任一因子失败就补任务
    或缩小理论问题，不用同一 instrument 变体凑数。
11. **论文 C interface gate**：建立独立 task-execution environment、parallel forms 与统一
    response/provenance schema；updating task 用 stimulus-once 的 stateful streaming，不重放
    transcript 或提供外部 memory，adaptive state 随模型自身反应推进，prompt/form 重复不增加
    被试 N。
12. **论文 C C0 smoke gate**：用 6–10 个结构差异明显的 checkpoints 验证 primary symbolic
    interface、stateful streaming、adaptive state、parse、泄漏和明显 ceiling/floor；失败时先修
    battery，不进入模型样本扩张。
13. **论文 C C1 pilot gate**：30–50 checkpoints 完成 parse、floor/ceiling、form/retest reliability、
    prompt/interface method variance 和 lineage-aware power 输入；只有 between-model signal 足够
    且含 lineage ICC 的 Monte Carlo power 通过，才冻结 immutable full roster、lineage cap/weights
    和 confirmatory items。
14. **论文 C analysis gate**：实现 trial-level confirmatory MIRT/GLLVM、lineage-blocked split、
    human–LLM factor congruence、linked-scale gate 与 factor-uncertainty propagation；只有 gate
    通过时才估计 invariance/DIF，普通 task-mean CFA 只作基线。

现有 `format_record_as_marked_text()` 只是把一行表序列化成 JSON token；它可作 smoke test，
但不能充当 structured condition。现有 finetuner 的随机 row split 和现有 scorer 的 session
key 也不能直接复用到 P500 正式实验。

论文 C 还不能复用现有 human-response renderer：某些记录包含 derived ground truth，某些视觉
state 不完整。它必须通过独立 agent-battery gates，不能因为已有文本 JSONL 就直接跑模型 roster。

---

## 10. 产物与清理策略

### P500 暂时不能删

- 原始 `data/P500-100agents/data/` 与 instruments；
- 1.6 GB `processed/standard_trials.csv`，直到 converter 进入 Git、重建成功并记录 checksum；
- conversion code、design/worklog、quality CSV 与 summary。

论文 C 新增的 construct/leakage audit、indicator specification、model-lineage registry、
item/prompt manifests 与 compact QA summaries 应进入 tracked docs/artifacts；大规模模型响应
继续忽略。

新 Trial-State v1 可复现后，旧的 74/89 MB full-session text JSONL 降为可再生历史产物；
旧 TOVA/BCS pilot 与 stale prompt 可删除。

### Psych-101/Centaur 只保留最小锚点

- 保留结果快照和 matched Centaur-8B / Llama-3.1-8B raw anchor（约 260 MB），至少到快照
  数字做完最后一次核对；
- 四个 Llama-3.2 raw runs、当前派生分析、per-choice count baseline 和旧 v0 workspace
  不再服务新主线，可删除或移到冷存储；预计可从活跃工作区释放约 1.25 GB；
- v0 大 CSV 已进入 Git，普通删除只清理工作树，不会缩小 Git 历史；历史重写是另一项任务。

实际删除按“tracked 可从 Git 恢复”与“ignored raw 无法恢复”分批执行，必须列出精确路径，
不递归清空整个 `outputs/`。
