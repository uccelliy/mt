# 近五年认知-行为 Foundation Models 研究综述（2021-2026）

**编制日期**：2026-06-12  
**范围**：以人类认知、心理实验、社会/经济行为、个体行为轨迹为建模对象的 foundation model 或 foundation-model-adjacent 工作。  
**项目关联**：本综述服务于 `mt` 的核心方向：用结构化 trial data 训练跨任务认知行为模型，并与 Centaur 的自然语言 trial encoding 路线正面对比。

---

## 0. Executive Summary

近五年真正以“人类认知/行为”为目标的 foundation model 仍然很少，领域还处在定义期。主线模型可以压缩为四类：

1. **Centaur**：最接近“foundation model of human cognition”的工作。它把 Psych-101 中 160 个实验、6 万多名被试、1000 万级选择编码为自然语言 trial sequence，微调 LLM 来预测人类 choice，并报告跨任务泛化和神经表征对齐。
2. **Be.FM**：更接近“open behavioral foundation model”。它覆盖行为经济学、调查、文献问答等数据，目标是预测群体行为、推断情境/个体特征、应用行为科学知识。
3. **Human Behavior Atlas / OmniSapiens**：从心理、情感、病理、社会行为的多模态理解出发，训练 social/psychological behavior processing 模型，不是典型 trial-level cognitive model。
4. **HumanLLM**：从在线用户轨迹出发，构建个体化模拟器，目标是 profile、inner thought、action、writing style 和 preference 的个体化生成。

此外还有一些相邻方向：Generative Agents / LLM-simulated participants、Monad 式用户事件 foundation model、wearable behavioral foundation model、humanoid behavioral foundation model。这些模型有“behavioral”或“foundation model”的形式，但多数不是认知科学意义上的 trial-by-trial human cognitive model。

**对 `mt` 的直接判断**：已有模型共同留下一个明显空位：**结构化 trial-level、跨决策与基础认知、同时建模 choice/accuracy/RT、并能与公式化 cognitive baselines 对齐的 foundation cognitive model**。这正是当前项目最有差异化的贡献空间。

---

## 1. 什么算“认知-行为 Foundation Model”

本报告采用一个实用定义：

> 认知-行为 foundation model 是一个在大规模、多任务人类行为数据上训练或适配的通用模型，能够跨任务预测、解释或模拟人类认知/行为，并可迁移到未见任务、未见情境或未见个体。

因此，普通 LLM “能回答心理学问题”不算；只用 LLM prompt 模拟 survey participant 也不完全算。更接近本综述范围的工作通常满足至少两点：

- 训练或评估数据来自真实人类行为，而不只是文本知识；
- 任务覆盖多实验、多情境或多个行为构造；
- 模型目标是预测行为分布、trial-level choice/RT、个体差异或社会/心理状态；
- 明确宣称 foundation model、generalist model、human simulator、behavioral FM，或功能上接近。

---

## 2. 主模型地图

| 模型 / 路线 | 年份 | 训练/评估数据 | 输出能力 | 是否核心认知行为 FM | 与 `mt` 的关系 |
|---|---:|---|---|---|---|
| Turning LLMs into cognitive models | 2023 | 心理实验/决策任务 | 微调 LLM 预测人类行为 | 前身/方法论 | 证明 LLM 可以吃行为数据，但规模较小 |
| **Centaur** | 2024/2025 | Psych-101，160 实验，6 万+ 被试，1000 万+ choices | trial-level choice prediction，跨任务泛化，神经对齐 | 是，最核心 | `mt` 的直接 head-to-head 对照 |
| **Be.FM** | 2025 | 文献、MobLab 经济博弈、Big Five survey | 行为分布预测、个体/群体/情境推断、行为科学知识应用 | 是，偏行为科学 | `mt` 可与其在结构化 trial 和 RT 方向形成差异 |
| Human Behavior Atlas / **OmniSapiens-7B** | 2025 | 多模态心理/社会行为数据 | affective/cognitive/pathological/social behavior understanding | 部分是，偏社会行为理解 | 可借鉴多模态与异质任务训练 |
| **OmniSapiens-7B 2.0** | 2026 | Human Behavior Atlas + HARPO RL | 异质心理行为任务上的 RL 对齐 | 部分是 | 可借鉴异质任务平衡训练 |
| **HumanLLM** | 2026 | Reddit/Twitter/Blogger/Amazon 用户轨迹 | 个体化模拟、profile/action/style/preference | 部分是，偏 online persona simulation | 可借鉴个体条件化与长期轨迹建模 |
| Generative Agents / LLM-simulated participants | 2023-2025 | 预训练 LLM + prompt / persona / survey context | 社会模拟、survey response simulation | 相邻，不是训练出的行为 FM | 提供 agent-based evaluation 思路，但不能替代真实被试 |
| Monad / user-event FM | 2023-2025 | click/like/page view/cart/purchase 等事件 | 用户行为表示、推荐/预测 | 相邻，偏工业行为日志 | 可借鉴 event-tokenization，不足以解释认知机制 |
| Wearable behavioral FM | 2025 | 可穿戴设备行为与健康数据 | 健康/行为状态预测 | 相邻，偏健康行为信号 | 可借鉴长时序、多主体行为表征 |
| Humanoid behavioral FM / Motivo | 2024-2025 | 机器人/人形控制数据 | motor behavior/control | 相邻，偏 embodied control | 不直接覆盖高层认知行为 |

---

## 3. 核心工作详述

### 3.1 Turning LLMs into cognitive models（2023）

Binz 与 Schulz 在 2023 年提出把 LLM 微调成 cognitive model 的路线：不是让 LLM 只凭 prompt 扮演被试，而是用心理实验数据训练模型，使其输出更贴近人类选择。

关键意义：

- 把 LLM 从“文本生成器”转成“行为预测模型”；
- 提供 Centaur 的技术前身；
- 说明心理实验数据可以被序列模型吸收，并支持一定跨任务迁移。

局限：

- 数据规模和任务覆盖有限；
- 主要展示方向可行性，还没有形成统一大规模 benchmark；
- 解释性依然弱，模型内部机制不等于 cognitive process。

参考：Binz & Schulz, 2023, [Turning large language models into cognitive models](https://arxiv.org/abs/2306.03917).

### 3.2 Centaur（2024/2025）

Centaur 是目前最核心的“foundation model of human cognition”工作。它使用 Psych-101 这一大型心理实验库，把不同任务的 trial 编码为自然语言，训练模型预测人类选择。

主要贡献：

- **数据规模**：160 个实验、6 万多名被试、1000 万级 choices；
- **任务广度**：覆盖 bandit、two-step、prospect theory、memory、learning、reasoning 等 Psych-101 任务；
- **泛化测试**：测试新 cover story、结构改变、新任务域；
- **神经对齐**：作者报告 Centaur 的内部表征与人类神经活动有对应关系；
- **统一路线**：把大量实验从“每个任务一个模型”推进到“一个模型覆盖多实验”。

与传统 cognitive model 的差别：

- 传统模型是 formula-first，如 RL、DDM、CPT、Bayesian learner；
- Centaur 是 sequence-model-first，以自然语言 trial 描述作为输入；
- 它预测力强，但机制透明性较弱。

论文中/相关讨论指出的局限与未来方向：

- 需要扩展到更多心理学领域，如 psycholinguistics、social psychology、economic games；
- 需要更好处理 individual differences；
- 需要更深入分析内部表征，而不只是预测分数；
- 需要探索从头训练或不同架构，而不只是在现有 LLM 上微调；
- 当前对 RT 的处理不足，这一点对基础认知任务尤其关键。

外部 critique：

- 有评论认为 Centaur 预测 choice 的能力不等于复现人类 generative cognitive process；
- 在一些生成式行为测试里，模型可能系统性偏离人类，因此还不能类比“心理学中的 AlphaFold”。

参考：

- Binz et al., 2024/2025, [Centaur: a foundation model of human cognition](https://arxiv.org/abs/2410.20268).
- Hammond & Lieder, 2025, [Not Yet AlphaFold for the Mind: Examining the Validity of Centaur's Predicted Cognitive Trajectories](https://arxiv.org/abs/2508.07887).

### 3.3 Be.FM（2025）

Be.FM 将目标从“认知实验选择预测”扩到“human behavior foundation model”。它强调通用 LLM 对行为科学问题存在 population variance、format bias、underrepresented groups 等问题，需要用行为科学数据和实验数据微调。

训练/数据来源：

- 行为科学文献；
- MobLab 行为经济学实验；
- Big Five survey；
- 初版明确不包括 observational data。

主要能力：

- 预测行为分布；
- 从行为推断个体或群体特征；
- 推断实验/社会情境；
- 使用行为科学知识解释现象；
- 支持 behavioral science 应用场景。

局限与未来方向：

- 初版只做 SFT，没有 RL 或更强 reasoning 对齐；
- observational data 尚未纳入；
- 若干能力仍缺少细粒度定量评估；
- 未来可扩展到更多 base LLM、更多真实世界行为数据、更多人群与跨文化数据。

与 `mt` 的关系：

- Be.FM 更“宽”，但 trial-level cognitive contract 不深；
- 它关注行为科学知识和群体行为预测，`mt` 可关注结构化 trial、RT、basic cognition；
- Be.FM 的开放路线说明该领域正在从封闭 benchmark 转向可复用模型和数据资产。

参考：Li et al., 2025, [Be.FM: Open Behavioral Foundation Models](https://arxiv.org/abs/2505.23058).

### 3.4 Human Behavior Atlas / OmniSapiens（2025-2026）

Human Behavior Atlas 试图建立一个统一的 psychological and social behavior understanding benchmark，并训练 OmniSapiens-7B 系列模型。它覆盖 affective、cognitive、pathological、social behavior，并使用 text/audio/vision 等多模态输入。

主要贡献：

- 将心理/社会行为理解组织为统一任务空间；
- 提供 10 类行为、10 万级样本、多模态数据；
- 训练 Omnisapiens SFT、BAM、RL 等模型版本；
- 后续 OmniSapiens 2.0 提出 HARPO，用于处理异质任务训练中的 reward/learning imbalance。

局限：

- 更偏 behavior understanding，而非认知实验中的 trial-by-trial prediction；
- 多模态社会行为任务与传统 cognitive task 的数据结构差异很大；
- 异质任务训练会让强信号任务主导学习，弱但重要的行为信号容易被淹没；
- 仍需要更严谨的机制解释与 OOD 泛化测试。

与 `mt` 的关系：

- 可借鉴其异质任务组织方式；
- HARPO 一类方法提示：跨任务训练不能只做简单混合，需要平衡不同任务的学习动态；
- 但 `mt` 应避免把 trial-level cognitive modeling 稀释成泛社会行为分类。

参考：

- Wu et al., 2025, [Human Behavior Atlas: A Unified Psychological and Social Behavior Understanding Benchmark for Multimodal AI Systems](https://arxiv.org/abs/2510.04899).
- Gao et al., 2026, [OmniSapiens-7B 2.0: Advancing Human Behavior Processing with HARPO](https://arxiv.org/abs/2602.10635).

### 3.5 HumanLLM（2026）

HumanLLM 从在线用户轨迹出发，目标不是认知实验 trial prediction，而是个体化人类模拟。它构建 Cognitive Genome Dataset，覆盖 Reddit、Twitter/X、Blogger、Amazon 等在线行为，训练模型生成或预测 profile、inner thought、action、writing style、preference。

主要贡献：

- 将“人类模拟”从单次 persona prompt 推进到长程个体行为轨迹；
- 强调 user-specific context，而不是 population average；
- 覆盖文本、偏好、行动、风格等多个输出维度；
- 与 LLM agent / social simulator 方向强相关。

局限：

- 在线平台数据代表性有限；
- 隐私与伦理风险很高；
- 合成数据可能 hallucinate、leak 或放大偏差；
- 个体轨迹模拟不等于认知机制解释；
- 与 controlled cognitive experiment 的 trial schema 距离较远。

与 `mt` 的关系：

- 可借鉴个体条件化、history conditioning、longitudinal user representation；
- 不适合作为 `mt` 的直接对照，因为它不以标准心理实验 trial 为主。

参考：Shen et al., 2026, [HumanLLM: Personalized Understanding and Simulation of Human Nature](https://arxiv.org/abs/2601.15793).

---

## 4. 相邻方向：有用但不要混淆

### 4.1 LLM-simulated participants / artificial subjects

大量 2023-2025 工作尝试用 LLM 模拟 survey participant、实验被试或社会群体。这些工作重要，但多数不是 foundation model，因为模型没有在大规模真实行为 trial 上训练，而是利用预训练 LLM 的文本知识和 persona prompt。

代表方向：

- Generative Agents：用 LLM agent 模拟小社会行为；
- Out of One, Many / silicon samples：用 LLM 模拟调查人群；
- LLM-as-participant：让模型直接参加心理实验。

主要发现：

- LLM 可以复现一些平均趋势；
- 但 effect size、群体差异、敏感议题、个体异质性常常偏离真实人类；
- 不能替代真实被试，只能作为 hypothesis generation 或 simulation baseline。

参考：

- Park et al., 2023, [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442).
- Aher et al., 2024, [Can AI Replace Human Subjects?](https://arxiv.org/abs/2409.00128).

### 4.2 Monad / user-event behavioral FM

Be.FM 的相关工作中提到 Monad 这类模型：以 click、like、page view、cart、purchase 等事件序列训练用户行为表示。这类模型与推荐系统和用户建模更接近。

可借鉴点：

- event tokenization；
- long sequence user history；
- behavior embedding；
- multi-domain behavioral transfer。

不适合作为 `mt` 主对照的原因：

- 目标是预测工业行为事件，不是解释 cognitive construct；
- 通常没有实验控制、reward/choice/RT contract；
- 机制解释和心理学可比性弱。

### 4.3 Wearable behavioral foundation model

可穿戴设备方向已经出现大规模 behavioral foundation model，用海量 wearables 行为/生理时间序列做健康预测。它证明“行为时间序列 + foundation model”路线在健康领域可行。

但它的“behavior”主要是 activity/sleep/physiology，不是 cognitive trial behavior。

参考：Shetty et al., 2025, [A wearable behavioral foundation model for health prediction](https://arxiv.org/abs/2507.00191).

### 4.4 Humanoid behavioral foundation model / Motivo

机器人和 embodied AI 领域也有 behavioral foundation model，目标是人形控制、动作生成、motor skill transfer。这与认知行为模型共享“行为序列建模”的形式，但研究对象是 motor control，而不是人类心理实验。

对 `mt` 的启发主要是：

- 多任务行为预训练；
- reward-conditioned behavior modeling；
- offline behavior dataset 的表示学习。

---

## 5. 共同缺陷：文献中反复出现的批评

### 5.1 预测准确率不等于认知机制

Centaur、Be.FM、OmniSapiens 都更像 predictive model，而不是 mechanistic cognitive model。它们能拟合行为，但模型内部是否实现了人类式策略、记忆、注意、价值更新，仍然需要额外验证。

对 `mt` 的启示：

- 不能只报告 held-out accuracy；
- 需要和 RL、DDM、CPT、Bayesian learner 等公式模型做 parameter/process probing；
- 需要做 counterfactual task manipulation，而不是只换 cover story。

### 5.2 平均行为容易，个体差异难

LLM 和大模型常能接近群体均值，但不能稳定复现个体差异、策略差异、临床差异、文化差异。

典型问题：

- effect size 被夸大或缩小；
- underrepresented groups 表现差；
- 个人历史、策略切换、学习率差异难建模；
- prompt/persona 不能稳定替代真实 subject-level latent state。

对 `mt` 的启示：

- 数据 contract 应显式保留 `subject_id`、session、block、task context；
- 训练目标应支持 subject-conditioned prediction；
- evaluation 需要 subject-level split、new-subject split、new-task split 三套指标。

### 5.3 行为变异性不足

一些 LLM participant 研究发现，模型可以复现均值，但行为多样性和搜索路径不像人类。例如 phonemic fluency 任务中，模型可能给出平均上合理的词数，但个体间 retrieval structure 不像人类。

对 `mt` 的启示：

- 需要评估 choice distribution、RT distribution、error pattern、strategy cluster；
- 不能只看 top-1 choice accuracy；
- 生成式评估要比较完整 trajectory，而不是单 trial。

参考：Demszky et al., 2025, [LLMs can simulate standardized human cognitive measures, but miss individual differences](https://arxiv.org/abs/2505.16164).

### 5.4 数据结构混杂

Be.FM 和 OmniSapiens 都强调异质行为数据的问题。不同任务、模态、人群、指标的 loss scale 不同，简单混合训练容易让高频或强信号数据主导模型。

对 `mt` 的启示：

- 任务采样、loss weighting、domain balancing 是核心技术问题；
- 决策任务与基础认知任务应保持同一个 contract，但不能丢失 schema-specific fields；
- RT loss、choice loss、accuracy loss 不应机械相加，需要 calibration。

### 5.5 自然语言编码带来可控性问题

Centaur 的自然语言 trial encoding 很灵活，但也带来问题：

- 相同结构可以被不同文本表述影响；
- 模型可能依赖语言先验而不是任务结构；
- 难以保证跨任务字段的严格一致性；
- 与 tabular baselines、classical ML baselines、formula models 对齐不直接。

这正是 `mt` 的结构化 trial-data 路线的机会。

### 5.6 RT 仍然是空位

Centaur 主目标是 choice。Be.FM 更偏行为分布与知识推断。HumanLLM 更偏 persona/action/style。大多数 foundation-model 工作没有把 RT 当作一等输出。

对基础认知任务而言，RT 是 golden signal：

- inhibition/control：RT 和 accuracy 的 trade-off 是核心；
- memory/search：RT 反映检索和扫描过程；
- decision/RL：RT 能区分 model-free habit、model-based planning、uncertainty deliberation；
- clinical/aging：RT 可能比 choice 更敏感。

---

## 6. 未来趋势

### 6.1 从 prompt simulation 转向 behavior-data training

早期路线是“让 LLM 扮演人”。新路线是“用真实行为数据训练模型”。Centaur、Be.FM、HumanLLM 都体现了这个转向。

### 6.2 从 population average 转向 individual-conditioned model

HumanLLM 代表个体化方向。Centaur 和 Be.FM 也都面对个体差异不足的问题。未来模型需要同时预测：

- population-level regularity；
- subject-level latent traits；
- within-subject learning trajectory；
- context-dependent strategy shift。

### 6.3 从单一 choice head 转向多输出行为建模

未来认知行为 FM 应同时预测：

- choice / response；
- accuracy；
- RT；
- confidence；
- eye movement / cursor / action trace；
- neural or physiological alignment signal。

### 6.4 从语言 trial encoding 转向结构化/混合 encoding

Centaur 证明自然语言 trial encoding 可以工作，但结构化 encoding 更适合：

- 保持字段一致；
- 与 cognitive baselines 对齐；
- 做跨任务 schema transfer；
- 加入 RT、stimulus features、signal/noise、block structure；
- 避免语言先验污染。

一个可能路线是 hybrid encoder：结构化字段为主，文本说明为辅。

### 6.5 从 black-box prediction 转向 mechanistic probing

未来论文需要回答：

- 模型是否学到了 Q-learning、Bayesian updating、DDM-like accumulation、memory decay？
- 模型内部表征是否对应 task state、value、uncertainty、control demand？
- 模型失败时是否像人类失败？
- 模型参数能否 recover classical cognitive model parameters？

### 6.6 从 benchmark accuracy 转向 generative trajectory validity

Centaur critique 提醒：预测 held-out choices 不等于生成合理的人类 trajectory。未来评估应包括：

- full-session trajectory simulation；
- strategy cluster recovery；
- RT distribution fit；
- perturbation/counterfactual tests；
- out-of-domain task family transfer；
- human-vs-model generative distinguishability。

---

## 7. 对 `mt` 的研究定位建议

### 7.1 推荐定位句

`mt` 可以定位为：

> A structured-trial foundation cognitive model for human behavior, trained across complementary decision-making and basic-cognition domains, with explicit choice/accuracy/RT prediction and formula-first cognitive baselines for mechanistic evaluation.

中文版本：

> 本项目构建一个基于结构化 trial contract 的认知行为基础模型，跨决策与基础认知两类互补领域训练，同时预测选择、正确率与反应时，并以公式化 cognitive baselines 作为机制评估锚点。

### 7.2 与 Centaur 的差异

| 维度 | Centaur | `mt` 机会 |
|---|---|---|
| 输入 | natural-language trial encoding | structured/tabular trial contract |
| 领域 | Psych-101，偏决策/学习 | 决策 + 基础认知双域 |
| 输出 | choice 为主 | choice / accuracy / RT |
| 机制对齐 | 神经表征 + 行为预测 | 行为预测 + cognitive formula probing |
| 可控性 | 高灵活但字段不严格 | schema 严格，可直接接 classical baselines |
| 独特性 | LLM 读实验文本 | 模型读实验 contract |

### 7.3 与 Be.FM 的差异

| 维度 | Be.FM | `mt` 机会 |
|---|---|---|
| 目标 | broad behavioral science FM | trial-level cognitive foundation model |
| 数据 | 文献、经济博弈、survey | 标准认知任务 trial |
| 粒度 | 群体/情境/知识推断 | trial/session/subject 轨迹 |
| 输出 | behavior distribution / inference | choice/RT/accuracy |
| 机制 | 行为科学知识为主 | formula-first cognitive baselines |

### 7.4 关键实验路线

1. **Centaur-compatible decision domain**
   - bandit、prospect theory、two-step、RL、risk/delay/social choice；
   - 与 Centaur 在相同或相似数据上做 choice prediction；
   - 比较 natural language vs structured encoding。

2. **Basic cognition domain**
   - perception、attention、memory、inhibition、cognitive control；
   - 强调 RT 与 accuracy；
   - 这是 Centaur 和 Be.FM 的弱覆盖区。

3. **Cross-domain transfer**
   - decision -> basic cognition；
   - basic cognition -> decision；
   - joint training；
   - held-out task family transfer。

4. **Mechanistic evaluation**
   - 与 RL、CPT、DDM、SDT、Bayesian models 对齐；
   - probe hidden states 是否编码 value、uncertainty、drift、threshold、memory load；
   - 做 parameter recovery / model mimicry。

5. **Generative validity**
   - 生成完整 session；
   - 比较 trajectory-level statistics；
   - 比较 RT distribution；
   - 检查是否复现人类 error pattern 和 learning curve。

---

## 8. 建议阅读顺序

1. **Centaur**：先读，因为它是 `mt` 的直接 comparison target。  
   Binz et al., [Centaur: a foundation model of human cognition](https://arxiv.org/abs/2410.20268)

2. **Centaur critique**：用于设计更强 evaluation。  
   Hammond & Lieder, [Not Yet AlphaFold for the Mind](https://arxiv.org/abs/2508.07887)

3. **Be.FM**：理解 behavioral FM 的开放路线。  
   Li et al., [Be.FM: Open Behavioral Foundation Models](https://arxiv.org/abs/2505.23058)

4. **Human Behavior Atlas / OmniSapiens**：理解多模态心理/社会行为统一 benchmark。  
   Wu et al., [Human Behavior Atlas](https://arxiv.org/abs/2510.04899)  
   Gao et al., [OmniSapiens-7B 2.0](https://arxiv.org/abs/2602.10635)

5. **HumanLLM**：理解个体化模拟路线。  
   Shen et al., [HumanLLM](https://arxiv.org/abs/2601.15793)

6. **LLM participant critique**：避免把 prompt simulation 当成真实认知模型。  
   Aher et al., [Can AI Replace Human Subjects?](https://arxiv.org/abs/2409.00128)  
   Demszky et al., [LLMs can simulate standardized human cognitive measures, but miss individual differences](https://arxiv.org/abs/2505.16164)

---

## 9. Reference List

- Aher, G. V., Arriaga, R. I., & Kalai, A. T. (2024). *Can AI Replace Human Subjects?* [https://arxiv.org/abs/2409.00128](https://arxiv.org/abs/2409.00128)
- Binz, M., & Schulz, E. (2023). *Turning large language models into cognitive models.* [https://arxiv.org/abs/2306.03917](https://arxiv.org/abs/2306.03917)
- Binz, M. et al. (2024/2025). *Centaur: a foundation model of human cognition.* [https://arxiv.org/abs/2410.20268](https://arxiv.org/abs/2410.20268)
- Bommasani, R. et al. (2021). *On the Opportunities and Risks of Foundation Models.* [https://arxiv.org/abs/2108.07258](https://arxiv.org/abs/2108.07258)
- Demszky, D. et al. (2025). *LLMs can simulate standardized human cognitive measures, but miss individual differences.* [https://arxiv.org/abs/2505.16164](https://arxiv.org/abs/2505.16164)
- Gao, Z. et al. (2026). *OmniSapiens-7B 2.0: Advancing Human Behavior Processing with HARPO.* [https://arxiv.org/abs/2602.10635](https://arxiv.org/abs/2602.10635)
- Hammond, L., & Lieder, F. (2025). *Not Yet AlphaFold for the Mind: Examining the Validity of Centaur's Predicted Cognitive Trajectories.* [https://arxiv.org/abs/2508.07887](https://arxiv.org/abs/2508.07887)
- Li, X. et al. (2025). *Be.FM: Open Behavioral Foundation Models.* [https://arxiv.org/abs/2505.23058](https://arxiv.org/abs/2505.23058)
- Park, J. S. et al. (2023). *Generative Agents: Interactive Simulacra of Human Behavior.* [https://arxiv.org/abs/2304.03442](https://arxiv.org/abs/2304.03442)
- Shen, Y. et al. (2026). *HumanLLM: Personalized Understanding and Simulation of Human Nature.* [https://arxiv.org/abs/2601.15793](https://arxiv.org/abs/2601.15793)
- Shetty, K. et al. (2025). *A wearable behavioral foundation model for health prediction.* [https://arxiv.org/abs/2507.00191](https://arxiv.org/abs/2507.00191)
- Wu, J. et al. (2025). *Human Behavior Atlas: A Unified Psychological and Social Behavior Understanding Benchmark for Multimodal AI Systems.* [https://arxiv.org/abs/2510.04899](https://arxiv.org/abs/2510.04899)

