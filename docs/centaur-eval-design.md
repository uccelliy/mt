# Centaur 评估设计：分离预测、上下文学习与认知相似性

## 1. 背景与核心问题

Centaur 的主要结果表明，一个在大规模人类行为数据上微调的语言模型，可以在多个心理学任务中取得比经典认知模型更低的负对数似然（negative log-likelihood, NLL）。这一结果有重要价值：它证明了跨任务行为预训练能够构造非常强的通用人类反应预测器。

但原始比较不能直接支持“Centaur 比经典认知模型更好地解释了人类认知过程”。核心原因不是简单的参数量不匹配，而是完整序列上的 teacher-forced NLL 同时混合了多种能力：

1. 大规模语言预训练获得的语义知识和任务先验；
2. 从刺激、选择和反馈历史中学习任务动态；
3. 从同一参与者此前的反应中在线识别个体策略；
4. 利用重复选择、按键偏好和局部序列相关等统计规律；
5. 与经典认知模型所描述的学习、记忆、决策或控制机制相似的计算过程。

因此，本设计的核心主张是：

> 完整上下文下的序列 NLL 是有效的条件预测指标，但不能单独作为人类认知相似性或机制解释力的指标。Centaur 的优势需要被分解为预训练优势、行为微调优势、上下文学习优势和机制相关优势。

本设计不以削弱 Centaur 为目标，而是希望回答更清楚的问题：Centaur 为什么预测得更好，这种优势在多大程度上来自通用知识、个体适应、长上下文统计规律，或可以跨参与者和跨任务泛化的认知结构？

## 2. 原始序列 NLL 的含义与局限

Centaur 的主要评估可以写成：

$$
\mathcal{L}_{\mathrm{full}}(M)
=
-\frac{1}{N}
\sum_{i,t}
\log p_M
\left(
y_{i,t}
\mid
I_i, x_{i,\leq t}, y_{i,<t}
\right),
$$

其中 $I_i$ 表示任务说明，$x_{i,\leq t}$ 表示截至当前 trial 的刺激、状态和反馈，$y_{i,<t}$ 表示真实参与者此前的反应。

该指标在概率意义上是成立的。根据链式法则，逐 trial 条件概率的乘积可以构成观测序列的联合似然。问题在于它测量的是 teacher-forced conditional prediction：模型始终站在真实人类生成的历史上预测下一步，而不是依赖自己的选择继续生成完整轨迹。

因此，它主要回答：

> 在已经观察到这个参与者此前如何反应的条件下，模型能否预测其下一次反应？

它不能单独回答：

> 模型能否依靠与人类相似的心理过程，独立生成具有相似学习曲线、错误模式、探索策略和个体差异的行为轨迹？

此外，按 response 直接平均 NLL 还可能使 trial 更多、session 更长或样本量更大的实验获得更高权重，并弱化 trial 之间的依赖结构。跨任务结论应补充 participant-level 和 task-level 的分层聚合。

## 3. 真正需要匹配的是适应机会，而不只是输入长度

完全删除历史并不是理想方案。人类在序列任务中本来就能观察自己的选择和反馈；Rescorla–Wagner、two-step 和 bandit 模型也必须利用历史来更新价值。真正的不对称在于模型如何利用历史：

- 经典认知模型只允许历史通过预先规定的状态变量起作用，例如价值、prediction error、stickiness 或 model-based weight；
- Centaur 可以通过注意力从完整 transcript 中提取任何具有预测力的规律；
- 原论文中的认知基线使用训练参与者共享的一套参数预测 held-out participants；
- Centaur 不需要更新权重，就能从 held-out participant 已有的反应中进行 in-context 个体识别。

因此，Centaur 的完整上下文优势可能包含 online phenotyping：根据前几个 trial 推断参与者的错误率、风险偏好、重复倾向、model-basedness 或其他稳定特征。这种能力本身具有价值，但应作为独立能力报告，而不应全部归入 cognitive alignment。

未微调 Llama 的比较只能反驳“任何一个拥有长上下文的通用大模型都足以达到 Centaur 的成绩”。它不能排除评估指标本身奖励无约束上下文推断的可能性，因为 Llama 和 Centaur 共享相同的大模型架构和强大的 in-context learning 能力。

## 4. 总体评估框架

建议将 Centaur 评估拆分为三个互补赛道，而不是用一个总 NLL 排序：

### 4.1 条件预测能力

评估模型在给定合法任务历史时预测下一次人类反应的能力。主要指标包括：

- held-out participant NLL；
- Brier score 和概率校准；
- choice accuracy 作为辅助指标；
- response time 的似然、误差或分布校准；
- participant-level 和 task-level macro-average。

该赛道保留 Centaur 原始主图所回答的问题，但避免将其直接称为机制相似性。

### 4.2 适应与上下文利用

评估模型从任务历史和参与者历史中获得了多少增益，以及这些增益是否依赖长上下文、真实人类前缀或自然语言表面规律。

主要方法包括：

- prefix–suffix adaptation curve；
- full、restricted 和 truncated context 对比；
- Llama 与 Centaur 的 context × finetuning 因子设计；
- history swap、history shuffle 和 context-window 操作；
- 首 trial、早期 trial 和晚期 trial 的分段表现。

### 4.3 生成与机制相似性

让模型在实验环境中进行 open-loop simulation，将自己的反应和反馈继续作为后续历史，而不是始终使用真实参与者前缀。比较模型与人类在以下方面的相似性：

- 完整轨迹分布；
- 学习曲线和错误类型；
- win–stay/lose–shift、perseveration 和 recency；
- directed/random exploration；
- model-basedness、风险敏感性和时间折扣；
- accuracy–RT、entropy–RT 和条件 RT 分布；
- 个体差异及参数相关结构。

预测赛道回答“是否猜得准”，生成与机制赛道回答“模型自己运行时是否表现得像人”。

## 5. 核心实验一：prefix–suffix 个体适应曲线

这是匹配 Centaur in-context learning 与经典认知模型个体适应机会的首选方案。

对每一位 held-out participant，将前 $k$ 个 trial 作为 calibration prefix，其余 trial 作为计分 suffix：

1. Llama 和 Centaur 获得完全相同的前 $k$ 个 trial；
2. 认知模型首先在训练参与者上学习群体层级先验 $p(\theta)$；
3. 使用该参与者的前 $k$ 个 trial 更新参数后验 $p(\theta_i\mid D_{i,1:k})$；
4. 三类模型都只在 suffix 上计分；
5. 令 $k\in\{0,1,2,5,10,20,\text{all previous}\}$，绘制 NLL 随 prefix 长度变化的曲线。

认知模型的预测应使用 posterior predictive，而不是单一点估计：

$$
p(y_{i,t}\mid H_{i,t},D_{i,1:k})
=
\int
p(y_{i,t}\mid H_{i,t},\theta_i)
p(\theta_i\mid D_{i,1:k})
d\theta_i.
$$

这一设计可以区分：

- $k=0$：模型的群体先验和跨参与者泛化能力；
- 小 $k$：快速个体适应能力；
- 大 $k$：充分利用参与者历史后的预测能力。

如果 Centaur 主要在大 $k$ 时拉开差距，其优势更可能来自个体识别和 ICL；如果在 $k=0$ 或很小的 $k$ 下仍领先，则更支持它学习到了可跨参与者迁移的行为规律。

## 6. 核心实验二：上下文与微调的因子分解

对 Llama 和 Centaur 同时设置两类上下文：

| 模型 | Matched / restricted context | Full transcript |
|---|---|---|
| Llama | 基础预测能力 | 通用 ICL 增益 |
| Centaur | 行为微调能力 | 微调后的 ICL 增益 |

`full` 使用原始完整 transcript。`matched` 不应简单删除所有历史，而应保留完成任务所必需的信息，并控制模型可利用的额外参与者线索。可使用以下一种或多种操作定义：

1. 固定长度的最近 $k$ 个 trial；
2. 结构化的 trial 字段，而不是包含重复语言线索的完整自然语言 transcript；
3. 任务所需的刺激、选择、状态和反馈，但移除与当前预测无关的叙述信息；
4. 任务特异的 sufficient-state prompt，作为诊断条件而不是唯一主条件；
5. 对不需要序列历史的任务逐 trial 独立预测。

记 $L_{M,c}$ 为模型 $M$ 在上下文条件 $c$ 下的 NLL，则上下文增益为：

$$
G_{\mathrm{context}}(M)
=
L_{M,\mathrm{matched}}
-
L_{M,\mathrm{full}}.
$$

微调是否增强了对历史的利用，可以通过交互项衡量：

$$
G_{\mathrm{interaction}}
=
G_{\mathrm{context}}(\mathrm{Centaur})
-
G_{\mathrm{context}}(\mathrm{Llama}).
$$

原始的 Centaur 相对认知模型优势还可以分解为：

$$
\begin{aligned}
L_{\mathrm{Cog,matched}}-L_{\mathrm{Centaur,full}}
=&
\underbrace{
L_{\mathrm{Cog,matched}}-L_{\mathrm{Llama,matched}}
}_{\text{预训练与架构优势}}\\
&+
\underbrace{
L_{\mathrm{Llama,matched}}-L_{\mathrm{Centaur,matched}}
}_{\text{行为微调增益}}\\
&+
\underbrace{
L_{\mathrm{Centaur,matched}}-L_{\mathrm{Centaur,full}}
}_{\text{额外上下文/ICL 增益}}.
\end{aligned}
$$

该分解能够把原主图中的单个差值转化为三个具有明确含义的来源。

## 7. 核心实验三：history 操作与有效记忆范围

### 7.1 Context-window curve

仅允许模型看到最近 $w$ 个 trial，其中 $w\in\{0,1,2,5,10,20,\text{full}\}$。比较性能随窗口长度的变化，并估计有效记忆范围。

如果 Centaur 的优势随着近乎无界的历史持续增长，而人类行为主要由较短历史解释，则应将这部分优势标记为 long-context statistical gain，而不是直接解释为人类相似性。

### 7.2 History swap

在具有 matched 或 yoked trajectories 的条件下，将参与者 A 的历史替换为参与者 B 的历史，同时预测 A 当前的反应。性能下降可以量化模型对个体历史的依赖。

该操作需要避免破坏任务因果结构。例如，在选择影响后续反馈的 RL 任务中，不能随意交换选择而保留不一致的奖励。可以优先在独立选择任务、预生成反馈序列或实验设计本身允许 yoking 的任务中使用。

### 7.3 History shuffle 与表面线索控制

在保持任务统计量或认知模型状态近似不变的条件下，打乱不应影响预测的历史顺序、叙述措辞、按键标签或冗余文本。若 Centaur 对这些变换高度敏感，而人类行为和经典模型基本不变，则表明其预测利用了额外的表面相关性。

## 8. 将所有模型转换到共同的认知表型空间

原始 NLL 比较的是模型输出概率，不直接比较模型产生的行为结构。建议对人类、Llama、Centaur 和经典模型的 open-loop 轨迹应用同一套分析管线，生成统一的 cognitive phenotype vector：

$$
z=
[
\text{accuracy},
\text{learning slope},
\text{win--stay},
\text{lose--shift},
\text{perseveration},
\text{exploration},
\text{model-basedness},
\text{risk sensitivity},
\text{RT effects},
\ldots
].
$$

比较时不只报告均值，还应比较：

- 表型分布；
- 参与者间方差；
- 不同表型之间的相关矩阵；
- 条件效应和交互效应；
- 极端策略或亚群的出现频率。

进一步，可以把同一个诊断性认知模型分别拟合到人类和各模型模拟轨迹上，得到：

$$
\hat\theta_{\mathrm{human}},
\quad
\hat\theta_{\mathrm{Centaur}},
\quad
\hat\theta_{\mathrm{Llama}},
\quad
\hat\theta_{\mathrm{cognitive\ simulation}}.
$$

随后比较参数分布、参数相关和参数随实验条件变化的模式。这种“共同认知模型投影”不能证明 Centaur 内部实现了相同算法，但能检验它是否生成了相似的认知表型，而不只是提高了下一步预测率。

## 9. NLL 的归一化与分层聚合

### 9.1 归一化预测分数

为了比较固有难度和选择数不同的任务，可定义：

$$
S_M
=
\frac{L_{\mathrm{null}}-L_M}
{L_{\mathrm{null}}-L_{\mathrm{ceiling}}}.
$$

其中：

- $S_M=0$ 表示不优于简单群体或均匀基线；
- $S_M=1$ 表示达到估计的人类可预测上限；
- $S_M<0$ 表示差于基线。

Noise ceiling 必须与上下文条件一致。full-context 模型不能使用 context-independent ceiling 作为同一尺度，否则模型可能通过利用参与者上下文表现出表面上的“超过 ceiling”。

### 9.2 分层聚合

推荐使用以下顺序：

1. 在每位参与者内部平均 trial-level log score；
2. 在每个任务内部平均参与者；
3. 对任务进行 macro-average；
4. 同时报告按 response 加权的 micro-average，作为补充而非唯一结果；
5. 使用 participant bootstrap 和 task bootstrap 估计不确定性。

显著性分析应以参与者或任务为独立单位，避免把高度相关的 trial 当作完全独立观测。

## 10. 不建议直接进行参数量惩罚

不建议把 NLL 除以参数量，也不建议机械地用 AIC 或 BIC 比较 70B 预训练模型与少参数认知模型。原因包括：

- 70B 基础参数并非在 Psych-101 上估计；
- 只计算 LoRA 参数会严重低估 Centaur 的实际容量；
- 计算全部预训练参数又无法反映迁移学习中的有效自由度；
- 经典 AIC/BIC 的渐近假设不适合这种预训练、冻结参数和上下文学习的组合。

更合适的是分别报告多个维度，形成 Pareto frontier：

- 预测 NLL 和校准；
- open-loop 行为相似性；
- 新参与者、新任务和跨域泛化；
- 模型大小、推理成本和训练数据量；
- 参数和计算过程的可解释性；
- 参数恢复、可识别性和反事实可检验性。

如需一个同时考虑数据效率和预测能力的复杂度指标，可探索 prequential minimum description length，但不应将其取代机制分析。

## 11. 推荐的结果图设计

### Figure A：个体适应曲线

横轴为 calibration prefix 长度 $k$，纵轴为 suffix macro-NLL 或归一化预测分数。曲线包括：

- Llama；
- Centaur；
- 固定群体参数的认知模型；
- 层级贝叶斯在线适应的认知模型。

### Figure B：Centaur 优势分解

使用 waterfall plot 展示：

1. 预训练与架构优势；
2. Psych-101 行为微调增益；
3. 完整上下文/ICL 增益；
4. 最终相对经典认知模型的总 NLL 优势。

### Figure C：上下文鲁棒性

展示 context-window curve、history swap、history shuffle、结构化输入和自然语言输入之间的差异。

### Figure D：open-loop 认知表型

以 effect-size forest plot 或参数分布图比较人类、Centaur、Llama 和认知模型。应避免仅使用雷达图展示均值，因为雷达图不容易表达不确定性和个体差异。

## 12. 最小可行实验（MVP）

第一阶段无需立即复现全部 Psych-101，可以选择三到四个互补任务：

1. bandit 或 horizon task：探索、价值学习和历史依赖；
2. two-step task：model-free/model-based 混合与个体差异；
3. risky choice 或 intertemporal choice：较弱的序列依赖，可用于独立 trial 对照；
4. N-back、go/no-go 或其他基础认知任务：检验该框架能否从决策领域迁移到基础认知能力领域。

每个任务优先完成：

1. 相同 held-out participant split；
2. Llama 与 Centaur 的 full/restricted context；
3. 经典模型的固定群体参数与层级在线适应版本；
4. $k\in\{0,5,20,\text{all previous}\}$ 的 prefix–suffix 曲线；
5. participant-level、task-level macro-NLL；
6. 至少一个 open-loop 行为效应；
7. 至少一次共同认知模型投影。

若 MVP 发现 Centaur 的优势大部分由 prefix 长度或 full transcript 解释，就有充分理由扩展 context decomposition。若受限上下文下 Centaur 仍稳定领先，则下一步应研究它在结构化 trial 表征中学到了哪些跨任务规律。

## 13. 与本项目的结合

本项目以结构化 trial 数据和 canonical field registry 为统一输入格式，这为控制 Centaur 式模型的上下文信息提供了天然优势。每次评估可以明确指定允许进入模型的字段，例如：

- 当前刺激和任务条件；
- 历史选择；
- 历史奖励或正确反馈；
- response time；
- trial index、block 和 session；
- participant-level calibration prefix；
- 自然语言说明或结构化任务元数据。

因此，可以在相同数据层上构造多种 view：

- `current_trial`：只包含当前 trial；
- `task_history`：包含完成任务所必需的历史；
- `participant_prefix_k`：包含固定长度的个体前缀；
- `full_session`：完整 session；
- `cognitive_state`：由指定认知模型导出的充分状态；
- `counterfactual_history`：用于 history swap 或 shuffle 的诊断视图。

所有 LLM、经典模型和统计基线共享同一 split、字段定义和评估容器，可以避免自然语言 prompt 与表格预处理之间出现不可追踪的信息差异。

## 14. 可以支持与不能支持的结论

仅凭 full-context teacher-forced NLL，可以支持：

> Centaur 是一个强大的跨任务条件行为预测器，能够利用完整实验历史预测 held-out participants 的下一次反应。

不能直接支持：

> Centaur 实现了与人类相同或更好的认知机制。

如果 Centaur 在 matched context、$k=0$ 或短 prefix、open-loop simulation、history perturbation 和共同认知表型比较中都接近人类，则可以更有力地支持：

> Centaur 学到的并不只是完整历史中的表面预测规律，而包含能够跨参与者、跨任务结构和生成条件稳定表达的人类行为规律。

如果其主要优势只在 full transcript 或长 prefix 下出现，则更准确的结论是：

> Centaur 的突出能力主要是利用长上下文进行在线行为识别和条件模仿，而其机制层面的人类相似性仍需独立验证。

## 15. 参考资料

- Binz, M. et al. (2025). [A foundation model to predict and capture human cognition](https://www.nature.com/articles/s41586-025-09215-4). *Nature*, 644, 1002–1009.
- Binz, M. et al. (2025). [Supplementary Information: domain-specific models and modelling details](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-025-09215-4/MediaObjects/41586_2025_9215_MOESM1_ESM.pdf).

