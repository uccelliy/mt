# Paper B — Shared Response Dynamics Across Cognitive Tasks

**日期**：2026-08-13  
**状态**：论文级设计草案；正式训练等待论文 A 冻结表示接口  
**共享协议**：[P500 三篇论文路线图](benchmark%20design.md)  
**上游接口**：[Paper A — Trial-Native State](p500-structured-input-paper.md)

## 0. 科学问题

> 选择、反应时间与未反应事件是否包含跨认知任务共享、可迁移的 response dynamics？

这不是“加 RT auxiliary head 能否改善 choice”的文章。它要建立一个在异质认知任务上联合
预测 choice、time 和 response event 的模型，并检验共享时间表征能否迁移到新参与者、
低数据 instrument 或未见 task family。

Choice 改善是重要的次级证据，但不是论文成立的必要条件；核心是 RT/event 分布本身是否
校准、是否恢复行为规律，以及 shared model 是否产生 task transfer。

## 1. 与论文 A 的边界

论文 B 直接冻结并引用论文 A 的：

- Trial-State v1 与 structured view；
- canonical action IDs、masks 与 candidate-action head；
- participant split、$K_{ref}$、主 backbone 和 choice 训练配置。

论文 B 不重新比较 text/structured，也不能根据 RT test 结果回头选择最有利的表示。它新增的
贡献是 RT/event measurement contract、joint objective、timing baselines 与 shared-dynamics
泛化检验。

## 2. RT/event measurement contract

正式训练前，每个 family 必须登记：

- timer 的起点、终点和单位；
- `response_time`、`response_initiation_time`、`response_datetime` 等字段的实际语义；
- deadline、合法 no-press、timeout、omission 和 invalid response；
- RT 是单个 choice、整个 trial，还是 sequence 总时长；
- 0、null、极端值、暂停/中断和设备异常的处理规则；
- 可用于 RT loss、只可用于 choice loss、或只能进入 censor/event likelihood 的 observation mask。

“大于 0 ms”只是 log-RT 的有效数值条件，不暗示存在负 RT。PC 的 0 ms、sequence trial 的
单一总 RT，以及 no-press task 的无按键事件必须分别处理，不能用全局删行规则掩盖。

若不同 families 的计时含义不兼容，应定义语义一致的 RT subsets 或 family-aware likelihood，
而不是为了覆盖率强迫所有任务共享同一种 target。

## 3. 模型目标

当前 trial 的 RT/event 永远只在 target 侧出现。统一目标写作：

$$
p(y_t,e_t,T_t\mid S_t,H_t),
$$

其中 $y_t$ 是 choice/choice sequence，$e_t$ 是 response、no-response、timeout/omission 等事件，
$T_t$ 是在相应事件语义下观察或 censor 的时间。

一个可实现的起点是：

$$
p(y_t\mid S_t,H_t)\;p(e_t,T_t\mid y_t,S_t,H_t).
$$

- 普通 response trial 预测条件 RT density，而不只回归均值；
- 有 deadline 的 no-response/timeout 使用 event/survival 或明确 censor likelihood；
- sequence task 只预测一次 trial-level total RT，不复制到每个 press；
- family conditioning 可以进入 likelihood/head，但 shared encoder 与 independent per-family
  encoder 必须作为可检验的模型差异。

## 4. 主条件与控制

固定 structured input 和 $K_{ref}$：

| ID | 监督 | 目的 |
|---|---|---|
| B-C | choice | choice-only 锚点，可复用论文 A 的 A-S |
| B-R | RT/event | timing-only 参照 |
| B-CR | choice + real RT/event | 联合模型 |
| B-CR-shuffle | choice + shuffled RT/event | 排除任意 auxiliary regularization |

Shuffle 在 task×condition 和兼容的 event/coverage strata 内完成，保留边际分布与 missingness，
但破坏 trial-level 行为对应。若 B-CR 改善 choice，再增加至少一个 matched random auxiliary
target 或 stop-gradient/separate-head control，区分真实 RT 内容、额外容量与梯度正则化。

Loss weight 只在 validation 上选择，并报告一个预先冻结的局部 sensitivity range。

## 5. Baselines

全局最小集合：

- task×condition shifted-lognormal/lognormal RT distribution；
- 简单 structured regression/MLP；
- RT-only、choice-only、joint 和 shuffled-RT；
- shared multitask 与 independent per-family models。

在语义适用的二选一速度任务上，增加 DDM/LBA 类认知 timing baseline；它用于检验联合
choice–time 分布，不要求强行覆盖全部 21 families。

## 6. Splits、泛化与指标

### 6.1 Held-out participants

主评估使用论文 A 冻结的 participant split。指标先在 participant 内聚合，再到
instrument/family；不能把 trial 数当独立重复。

### 6.2 Low-data 与 task transfer

- 预先冻结低数据 instrument 条件，比较 shared 与 independent models 的学习曲线；
- 在 RT measurement gate 通过的 families 上定义 held-out-family folds；
- held-out family 的 state schema/action ontology 可以已知，但 choice/RT 人类 targets 不可见。

### 6.3 主要指标

- RT/event held-out log predictive density；
- distribution/quantile calibration 和 coverage；
- no-response/timeout event calibration；
- choice NLL 与 calibration；
- family-level paired effects 与 seed uncertainty。

Choice 和 RT 不合成一个总分。

## 7. 主要 estimands

RT skill：

$$
\Delta_{RT\text{-}skill}
=L_{RT}(\text{task/condition baseline})-L_{RT}(\text{B-CR}).
$$

Choice 增量：

$$
\Delta_{joint}^{choice}
=L_{choice}(\text{B-C})-L_{choice}(\text{B-CR}).
$$

真实 RT 内容相对 shuffle：

$$
\Delta_{RT\text{-}content}^{choice}
=L_{choice}(\text{B-CR-shuffle})-L_{choice}(\text{B-CR}).
$$

共享动力学通过 low-data/held-out-family 中 shared model 相对 independent per-family model 的
RT/event predictive-density gain 测量。需要分别报告总体、family 和 event type 结果。

## 8. 行为现象验证

按 family 的实验设计，预先登记适用的现象，而不是训练后挑选：

- condition/difficulty 对 RT 分布的影响；
- correct/error RT 与 speed–accuracy 关系；
- switch cost、congruency/interference 或学习/疲劳效应；
- no-press、timeout/omission 的概率与时间；
- sequence length 与 total RT 的关系。

模型复现相关性不自动识别 processing speed、effort 或 evidence accumulation；机制结论需要
额外干预或可区分的机制基线。

## 9. 成功、降级与停止规则

| 结果 | 论文结论 |
|---|---|
| RT/event 分布校准，shared model 在低数据/held-out family 更好 | 存在可迁移的共享 response dynamics |
| B-CR choice 更好且胜 shuffle/matched controls | 真实 RT 含 choice-only 未表达的可复用监督 |
| RT 可拟合但没有迁移或行为效应 | 只能称已见任务 timing prediction，主论文命题不足 |
| 只有少数 families 稳定 | family-specific timing；缩小论文范围或采用 modular model |
| choice 不改善但 timing 迁移成立 | 论文仍成立；choice 与 timing 的共享表征有限 |
| 所有增益与 shuffled RT 相同 | 普通 auxiliary regularization，不归因于 RT 内容 |

若 RT 只能作为一个容易预测的输出，既无 distributional insight、task transfer，也无独立行为
发现，暂停论文 B，而不是把“增加一个 RT head”当成贡献。

## 10. 执行顺序

1. 在论文 A 期间并行完成逐 family RT/event audit；
2. 冻结 measurement contract、eligible families、event masks 和 held-out folds；
3. 等论文 A 冻结 structured interface、$K_{ref}$ 和主 backbone；
4. 实现并验证 RT/event likelihood 与简单 baselines；
5. 运行 B-C/B-R/B-CR/B-CR-shuffle；
6. 只有真实 RT 产生增量时追加 matched auxiliary/gradient controls；
7. 运行 low-data、held-out-family 和预登记行为现象分析；
8. 根据停止规则决定统一 shared model、modular model 或缩小论文范围。

## 11. 排除项

- 不运行 text+RT 或重新申领论文 A 的 representation 贡献；
- 不改变 history budget；
- 不把 null/0 一律删除或填补；
- 不把 sequence total RT 复制到 choice steps；
- 不用一个 choice NLL 数字代替 RT distribution evidence；
- 不恢复 Psych-101/Centaur 分析或完整经典认知模型套件。
