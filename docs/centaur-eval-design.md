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

### 2.1 paper-like NLL：论文协议控制轨道（P0）

除下文三条科学评估赛道外，第一阶段另保留一条 **P0 / paper-like NLL** 协议控制轨道。
它的目的不是新增机制结论，而是把本项目的运行时量化结果放到与 Centaur 官方 evaluator
相同的任务选择、截断和聚合口径中，隔离“评估实现差异”与“模型/量化差异”。P0 固定：

1. 官方的 36 个 task-family prefix（当前测试集对应 57 个精确 experiment、5,795 个 session）；
2. 同一 tokenizer 的 `add_special_tokens=True`，再右截断到 32,768 token，即保留开头
   32,768 个 token（含 BOS）；
3. 只监督 `<<...>>` 中的 response token；
4. 每个 family 的主指标为先在每个 session 内计算 target-token mean NLL，再在该 family 内
   对 session 等权平均的 `official_eval_loss`；token-micro NLL 仅作诊断；
5. 完整记录模型、prompt/tokenizer、量化配置、compute dtype 与 attention backend。

当前 P0 的标签是 **`Minitaur-8B BF16 checkpoint, runtime NF4 — paper-protocol NLL
derived from full-context cache`**。它不是论文的 Centaur-70B/BF16/FlashAttention 复现，
也不保证逐 bit 等同官方 Trainer；其价值是 evaluator 语义兼容的 runtime control。

本次可从已完成的 full-context runtime cache 安全派生 P0，是因为 decoder-only 因果模型中 cutoff 前 token 的
条件概率不依赖未来 token，并且同一 tokenizer/prompt 的审计确认：没有一个 response span
横跨 32,768-token 边界，所有保留 choice 的 token 数都与重新 tokenization 一致；10 个
`zorowitz2023data` session 则由显式 UTF-8 重评分的 replacement 替换。58 个超窗 session
的 28,862 个完整尾部 choice 被排除。若任一前提不成立，必须直接以 token 级截断重评分，
不得用字符级 `--max-chars` 或简单抽取某个位置的 NLL 代替。

## 3. 真正需要匹配的是适应机会，而不只是输入长度

完全删除历史并不是理想方案。人类在序列任务中本来就能观察自己的选择和反馈；Rescorla–Wagner、two-step 和 bandit 模型也必须利用历史来更新价值。真正的不对称在于模型如何利用历史：

- 经典认知模型只允许历史通过预先规定的状态变量起作用，例如价值、prediction error、stickiness 或 model-based weight；
- Centaur 可以通过注意力从完整 transcript 中提取任何具有预测力的规律；
- 原论文中的认知基线使用训练参与者共享的一套参数预测 held-out participants；
- Centaur 不需要更新权重，就能从 held-out participant 已有的反应中进行 in-context 个体识别。

因此，Centaur 的完整上下文优势可能包含 online phenotyping：根据前几个 trial 推断参与者的错误率、风险偏好、重复倾向、model-basedness 或其他稳定特征。这种能力本身具有价值，但应作为独立能力报告，而不应全部归入 cognitive alignment。

未微调 Llama 的比较只能反驳”任何一个拥有长上下文的通用大模型都足以达到 Centaur 的成绩”。它不能排除评估指标本身奖励无约束上下文推断的可能性，因为 Llama 和 Centaur 共享相同的大模型架构和强大的 in-context learning 能力。

### 3.1 该不对称已由论文方法学与官方数据共同确认（2026-07-29）

上述不对称不再是推测，两条独立证据：

**方法学出处**：补充材料”Modeling details”载明，所有认知基线”对训练集所有
参与者**联合估计一套参数**，而不是为每位参与者单独拟合”。即认知基线被
**结构性禁止**做个体适应；Centaur 则从 transcript 中免费获得在线个体识别。
二者的比较因此不是等适应机会下的比较。

**实证签名**：用官方公布的逐 choice 似然数组，按会话内位置分层（24 个四方
对齐任务，详见 handoff §5 fig15）：

| within-session decile | 0 | 9 |
|---|---:|---:|
| 认知模型 | 0.7126 | 0.7174（**平坦**） |
| Centaur-70B | 0.6547 | 0.4203 |
| Centaur − 认知模型 | **+0.058** | **+0.297** |

Centaur 相对认知模型的平均优势（≈0.17 nat）**大部分是会话内累积的**：开局
仅 0.058，结尾 0.297。绝对位置上更明确——**第 1 个 trial 上认知模型胜过
Centaur**（0.731 vs 0.766），而 trial 0 恰是双方都只有群体知识的公平对照点。
未微调的 base-70B 同样：开局差 0.234，靠读历史在结尾反超 0.185。

**这不构成”Centaur 没有认知价值”的结论**，因为该签名混合了两种来源：
(a) 更好地建模任务学习过程（属机制上的成功），(b) 在线个体识别（§7.2 所指的
online phenotyping，不属机制成功）。分离二者必须做 §7.2 的 history swap；在
该实验完成前，只能声明”优势的相当部分依赖会话内累积的个体历史”，不能声明
它全部是 phenotyping。

**另需注意基线的构成**：14 类基线中，模型 13「理性模型」是”神谕 + 混淆矩阵”
（被直接告知每试次最优答案，只学偏离模式），模型 14 是逐 trial 查找表，二者
共覆盖 7 个任务且被补充材料自己归为”统计性上限/基线”。定量上它们并未拉高
Centaur（17.6% 的任务贡献 14.0% 的优势），但在这些任务上”胜过领域认知模型”
的主张属于**未经检验**——存在真模型而论文未拟合。

## 4. 总体评估框架

建议将 Centaur 评估拆分为三个互补赛道，而不是用一个总 NLL 排序。P0 是上节定义的
protocol-control，不是第四条认知科学赛道，也不产生关于机制相似性的结论：

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

### 5.1 计分窗口修正（重要）

若对整个 suffix 计分，$k$ 的作用会被稀释：suffix 期间历史仍在累积（Centaur 在 suffix 第 $t$ 个 trial 时已见过 $k+t$ 个该参与者的 trial；认知模型的状态变量也在 suffix 上继续更新），长 suffix 平均后不同 $k$ 的差异只体现在 suffix 开头几个 trial。因此：

1. 只对 prefix 后的固定小窗口（trial $k+1$ 到 $k+m$，$m\in[1,5]$）计分，而不是整个 suffix；
2. 对 LLM 而言，full-context 打分下"逐 trial 位置的 NLL 曲线"就是这条适应曲线本身——第 $t$ 个 trial 的预测天然只依赖前 $t-1$ 个 trial。一次打分即可，无需为每个 $k$ 单独构造 prompt。按 $k$ 构造 prompt 只有认知模型的后验更新才真正需要（第三阶段）；
3. 不同 $k$ 对应的计分 trial 集合不同，曲线沿 $k$ 的形状混合了适应增益与 trial 难度漂移；跨模型的干净比较应在固定 $k$ 下进行。

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

**主定义预先注册为操作 1（自然语言格式内的截断）**：保留任务 instructions，将最近 $k$ 个 trial 拼接为一个"session 刚开始"形态的合法 transcript。原因是操作 2–4 改变了 prompt 格式，而 Centaur 是在完整自然语言 transcript 上微调的——这些条件下的性能下降无法区分"信息受限"与"格式分布外"，等于在一次操作里混入了两个自变量（上下文信息量 × 编码格式）。操作 2–5 降级为诊断性附加分析，编码格式本身的对照留给第二阶段（结构化输入微调）。

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

**注意分解的路径依赖。** 上式是望远镜恒等式（$a-d=(a-b)+(b-c)+(c-d)$，中间项两两相消），数值上恒成立，但各项的语义标签依赖分解顺序：微调增益在 matched 条件下测量（$L_{\mathrm{Llama,matched}}-L_{\mathrm{Centaur,matched}}$）与在 full 条件下测量，恰好相差一个交互项 $G_{\mathrm{interaction}}$。按上式的顺序，交互项被隐式归入了 ICL 项。报告时必须显式给出交互项（或同时给出两种分解顺序），Figure B 的 waterfall 中交互项应作为独立一块画出，不得默认合并。

### 6.1 E5 runtime-NF4 结果（2026-07-28）

Llama-3.1-8B base 与 Minitaur-8B 已在完全相同的 E3 协议下完成配对。这里把每个
$w$ 都看作一个 matched 条件，并用
$G_{\mathrm{context}}(M,w)=L_{M,w}-L_{M,\mathrm{full}}$。主推断仍按
participant/session → 精确 experiment → 75-experiment task-macro 聚合：

| $w$ | Llama base NLL | Minitaur NLL | 微调增益（base−Minitaur） | $G_{\mathrm{context}}$(base) | $G_{\mathrm{context}}$(Minitaur) | 交互（Minitaur−base） |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3.055074 | 2.650945 | +0.404129 | 1.758404 | 1.440453 | -0.317951 |
| 1 | 1.615063 | 1.526024 | +0.089039 | 0.318393 | 0.315532 | -0.002861 |
| 2 | 1.527316 | 1.437704 | +0.089611 | 0.230646 | 0.227212 | -0.003434 |
| 5 | 1.435769 | 1.347940 | +0.087828 | 0.139098 | 0.137448 | -0.001650 |
| 10 | 1.395948 | 1.309466 | +0.086482 | 0.099278 | 0.098974 | -0.000304 |
| 20 | 1.363826 | 1.275880 | +0.087946 | 0.067156 | 0.065388 | -0.001768 |
| full | 1.296670 | 1.210492 | +0.086178 | 0 | 0 | 0 |

participant×task 配对 bootstrap（seed=20260728，5,000 次）显示：$w=0$ 的交互为
**-0.317951**，95% CI [-0.347411, -0.286952]；但一旦给出至少一个历史段，交互
几乎归零：$w=1$ 为 -0.002861 [-0.007126, 0.000997]，$w=5$ 为
-0.001650 [-0.004956, 0.001692]，$w=20$ 为 -0.001768
[-0.004561, 0.001030]。因此在这个 8B/NF4 设置下，**Psych-101 微调没有可检测地
增强模型利用已有历史的效率**；它主要改善无历史时的冷启动、任务接口/response
alphabet 校准和行为先验，从而降低对第一段上下文的依赖。

两条曲线在 $w\ge1$ 时几乎平行。Llama base 从 $w=0$ 到 full 的总 task-macro
上下文收益中，$w=1/2/5/10/20$ 分别捕获 81.9%/86.9%/92.1%/94.4%/96.2%；
按主 task-macro 口径，EC90=$5$、EC95=$20$。Minitaur 的相应曲线形状与之相同。
任务级 context gain 的跨模型相关在 $w=1$ 为 0.9991、$w=20$ 为 0.9979。
这说明“近期历史贡献最大、超过 20 段仍有残余”主要是基础模型已有的 ICL
性质，而不是行为微调新造出的上下文利用机制。

五锚点网格刻意给 first/second 较高权重，因而 E3/full 的总体微调差
0.086178 nat 不能与 E0 all-choice 的约 0.014 nat 直接比较。按位置拆开后，Minitaur
相对 base 的 full NLL 优势为：first 0.413605、second 0.006556、10% 0.008830、
50% 0.006654、last 0.007032 nat；即 E3 的总体差几乎完全由 first anchor 拉高。
抽样目标中 30,195/32,672（92.4%）是单 choice；只保留这些目标时，$w=0$ 交互
-0.338390，而 $w=1/5/20$ 交互仍分别仅为 -0.003320/-0.000611/-0.001490，
所以结论不是 multi-token response artifact。

以上是同架构 8B 复制品、runtime NF4、teacher-forced NLL 与五锚点抽样下的因子
分解；它不能外推为 Centaur-70B/BF16 的效应量，也不能把 context truncation
解释成人类或模型的抽象记忆跨度。

## 7. 核心实验三：history 操作与有效记忆范围

### 7.1 Context-window curve

仅允许模型看到最近 $w$ 个含 choice 标记的 transcript 段（通常对应一条 trial
line），其中 $w\in\{0,1,2,5,10,20,\text{full}\}$。比较性能随窗口长度的变化，
并估计在该段定义下的有效记忆范围。

截断后的 prompt 必须仍是模型分布内的合法形态：保留任务 instructions，并将最近
$w$ 个段拼接成"session 刚开始"的 transcript。Psych-101 的训练 transcript 都从
trial 1 加说明开始，从中间截起的裸片段是模型没见过的形态，否则会把格式 OOD
效应误读为历史依赖。

位置抽样采用 E0-informed 的五个预注册锚点：第 1、第 2、session 的 10%、50% 和
最后一个**含 choice 标记的 transcript 段**。依据是 E0/E1 中最大的适应变化出现在
开头第 1→2 个 trial/目标，而原 8 点等距网格通常漏掉第二个位置；10% 表示早期稳定
阶段，50% 与末点保留中后期覆盖。这里的窗口和位置单位不是保证只有一个反应的
“原子 choice”：实现按含 `<<...>>` 的 transcript 行切段，同一行的多个 choice
留在同一段。本次抽样目标段中 92.4% 恰含一个 choice，单 choice 目标敏感性分析保持
同样结论。短 session 对索引做单调 clamp，段数不超过 5 时全部保留。该网格按总体
E0 发现预先固定，不根据每个 session 的 NLL 选择位置。结果应按五个位置 strata
分别报告；不能把前密后疏样本直接当成全 session 的等权平均。旧的等距方案作为
`--position-grid even` 保留，供敏感性分析或历史结果复现。

#### E3 runtime-NF4 全量结果（2026-07-24）

`outputs/scoring/minitaur8b_e3_e0grid5_4bit.csv` 已完整覆盖 75 个 experiment、
6,561 个 session。每个 window 有 32,672 个目标位置、65,586 条 response 记录和
68,542 个 response token；7 个 window 共 459,102 行。按同一 target 下的等价
window label 去重后，需评分 133,034 个 effective prompt cell/input；这不是全文本
全局 unique，也不是 forward-call 次数，模型调用还会按 token budget 打包。按原数据
重建审计没有缺失、额外或不完整 session，没有重复完整结果键，也没有
failed/skipped sidecar。

CSV 的 `nll` 是 response-token NLL 的和。先在每个 session/window 内计算
`sum(nll) / sum(num_tokens)`。session-macro 对 6,561 个 session 等权，描述测试集
经验 session 分布；按 §9.2 的跨任务主 point estimate 则再在每个精确 experiment
内平均 participant/session，最后对 75 个 experiment 等权。token-micro 只作诊断：

| window（历史段数） | token-micro NLL | session-macro NLL | 75-experiment task-macro NLL | session Δ vs full |
|---:|---:|---:|---:|---:|
| 0 | 1.292844 | 2.256869 | 2.650945 | +1.152546 |
| 1 | 0.737327 | 1.259799 | 1.526024 | +0.155476 |
| 2 | 0.704850 | 1.206035 | 1.437704 | +0.101712 |
| 5 | 0.676376 | 1.156379 | 1.347940 | +0.052056 |
| 10 | 0.658825 | 1.134835 | 1.309466 | +0.030512 |
| 20 | 0.649056 | 1.122534 | 1.275880 | +0.018210 |
| full | 0.636964 | 1.104323 | 1.210492 | 0 |

session 配对差的正态近似 95% CI（均值 ± 1.96×SE）对所有非 full 窗口均高于 0；
这不是 bootstrap。75-experiment task-macro 的 w=20−full 为 +0.065388，64/75 个
experiment 为正。正式推断仍需按 §9.2 产出注明 seed 与重复次数的 participant/task
bootstrap。

在**五锚点、session 等权**汇总中，w=1 已捕获 w=0→full 总改善的 86.5%，w=5
捕获 95.5%，w=20 捕获 98.4%。仅按论文 `PAPER_TASKS` allowlist 筛选的 36-family
E3 子集仍得到 w=20−full = **+0.03527 nat**；其 family 配对差正态近似 95% CI 为
[0.01794, 0.05261]。该子集仍使用 E3 五锚点且没有 head-32k 截断，**不是 P0
`official_eval_loss`**。所以“主要依赖最近历史”和“仍有小而稳定的长历史收益”
可以同时成立。

为避免五点网格前密后疏造成混合，位置分析限制在 `n_segments > 5` 的 6,478 个
session。每个 session/position/window 内先算 `sum(nll) / sum(num_tokens)`，再对
6,478 个 session 等权。下表报告 full NLL，以及截断窗口相对 full 的 gap：

| 位置 strata | full NLL | w=1 gap | w=5 gap | w=20 gap |
|---|---:|---:|---:|---:|
| first | 2.093 | 0 | 0 | 0 |
| second | 1.066 | 0 | 0 | 0 |
| 10% | 0.833 | +0.209 | +0.046 | +0.006 |
| 50% | 0.747 | +0.241 | +0.085 | +0.028 |
| last | 0.778 | +0.329 | +0.141 | +0.053 |

first/second 在足够长窗口下与 full 相同是由当时可用历史长度构造性决定；不能把
这些 0 当作额外经验支持。可解释的模式是：越到 session 后部，短窗口相对 full 的
损失越大，末点仍能利用更久历史。任务间存在异质性，少数任务甚至 w=20 略优于
full，因此不应把总体曲线变成每个任务都具有同一记忆长度的主张。

E0 的结构与聚合 sanity check 通过：归一化误编码的 experiment 名并替换 10 个 UTF-8
修复的 `zorowitz2023data` session 后，E3/full 的 65,586 条 response 与 E0 一一对应且
token 数完全一致。两者 token-micro NLL 为 0.63696353 与 0.63705079，差
-0.00008726。单行并非 numerical equivalent（MAE 0.00413，最大差 0.242）；该模式
与低精度 CUDA/cuDNN 对 batch shape/packing 敏感的数值差异相容，但来源尚未用同
prompt 重复实验单独隔离，因此只能确认 key/token 与聚合一致性。

因此，本次 E3 在该 prompt 重建协议下直接测得“删除较早历史段”对 Minitaur-8B
runtime-NF4 NLL 的干预效应。它比观察性位置曲线更强，但截断同时改变可见内容、
输入长度和 transcript 连贯性，不能单独推断抽象记忆跨度，更不能推断人类内部机制。
五点网格又刻意提高了早期位置权重：E3 五点 `full` scalar 不能与 E0 all-choice
global scalar 或 P0 `official_eval_loss` 直接相减；E3 与 E0 相同 keys 的 sanity
对照仍然有效。同协议 Llama-base E0/E3 已完成，E5 结果见 §6.1；对论文精度的主张
仍需要 BF16/HPC 单列验证。

如果 Centaur 的优势随着近乎无界的历史持续增长，而人类行为主要由较短历史解释，则应将这部分优势标记为 long-context statistical gain，而不是直接解释为人类相似性。

### 7.2 History swap

在具有 matched 或 yoked trajectories 的条件下，将参与者 A 的历史替换为参与者 B 的历史，同时预测 A 当前的反应。性能下降可以量化模型对个体历史的依赖。

该操作需要避免破坏任务因果结构。例如，在选择影响后续反馈的 RL 任务中，不能随意交换选择而保留不一致的奖励。可以优先在独立选择任务、预生成反馈序列或实验设计本身允许 yoking 的任务中使用。

### 7.3 History shuffle 与表面线索控制

在保持任务统计量或认知模型状态近似不变的条件下，打乱不应影响预测的历史顺序、叙述措辞、按键标签或冗余文本。若 Centaur 对这些变换高度敏感，而人类行为和经典模型基本不变，则表明其预测利用了额外的表面相关性。

### 7.4 $w=0\rightarrow1$ 的收益分解

E1/E3 显示最大的性能变化发生在第一个可见历史段加入时，但该差值不能整体解释为
行为学习或个体适应。单个 demonstration 同时暴露了合法 response alphabet、trial
格式、输入分布、任务结构和参与者反应。分解设计由一个解析测量加一个四条件
prompt 阶梯构成。

**2026-07-27 修订**：移除了原设计中的 "label-space only" prompt 条件。理由：
Psych-101 训练分布内不存在"单独陈述合法按键"的 transcript 形态，任何措辞都是
分布外的，测得的"字母表收益"会与 OOD 格式惩罚混淆——这正是本节对其余条件
明令禁止的问题。字母表成分改用下述解析测量，不再依赖 prompt 构造。

**解析测量（response alphabet 成分）**：在每个条件下同时计算原始 NLL 与把
softmax 限制到该 session 合法 response label token 集后的重归一化 NLL。两者之
差是模型浪费在非法 token 上的概率质量，即"字母表未发现"惩罚。合法 label 集用
该 session 全部 `<<...>>` response 的并集近似（客观可得，但是真实合法集的下界，
须如实报告）；该诊断先限制在单 token response 的任务上，多 token response 的
重归一化没有唯一定义。实现上需要打分器在目标位置额外 gather 全部合法 label
token 的 logprob，属小改。

**prompt 阶梯（相同目标位置）**：

1. **instructions only**：复用 E3 的 $w=0$；
2. **format only**：加入一个训练分布内、格式合法的 trial demonstration，但
   response 与真实 input--response 映射解耦（以真实匹配 trial 的 response
   randomization 实现，不得插入模型训练时未见过的占位符）；
3. **matched other-participant trial**：加入同任务、同阶段、选项与反馈结构匹配
   的其他参与者 trial，并把按键标签按**选项角色而非频率**一致映射到目标
   session 的 response alphabet（按频率映射会把对方的 base rate 偷渡进来），
   匹配与映射规则须预注册；
4. **own previous trial**：复用 E3 的 $w=1$，使用目标参与者自己的真实上一段
   历史。

主分析在**重归一化 NLL** 上取相邻差值，依次近似量化格式校准、任务/环境识别和
participant-specific adaptation——字母表成分已被重归一化剥离，否则每个含
demonstration 的条件都会重复吸收字母表收益；各条件的原始−重归一化差另行报告
该条件下残余的字母表惩罚。条件 2–3 只能用于无反馈、独立 trial、预生成反馈或
yoked trajectory 的任务：response randomization 在有反馈任务中会产生行为与反馈
不一致的 trial，模型察觉后惩罚的是连贯性而非信息量（与 §7.2 的约束同源）。

该实验的首要报告量不是各条件的绝对 NLL，而是各成分解释了
$L_{w=0}-L_{w=1}$ 的多少比例。只有 `own previous trial` 相对
`matched other-participant trial` 的额外收益，才可初步归入在线个体适应。
runner 实现后应对 Minitaur 与 Llama base 用同一套条件各跑一遍：微调是否改变
$w=0\rightarrow1$ 收益的构成本身就是 §6 交互项最有解释力的版本。

### 7.5 超过 20 段历史的长度匹配干预（暂缓）

**状态（2026-07-27）：暂缓，不进入当前执行队列。** 两个原因：

1. **功效未经论证**。待解释的目标效应很小（session-macro $w{=}20-$full 仅
   +0.018 nat），而 shuffle−swap 这类二阶对比可能只有几个 millinat；E0↔E3 的
   full-cell 对照显示仅 batch shape/packing 的数值非确定性就有单行 MAE ≈0.004。
2. **残余大的任务恰是干预最难合法实施的任务**。残余集中在
   `feng2021dynamics`、`xiong2023neural` 等序列型长会话任务，far swap 与
   neutral control 在这类任务里最难保持因果与表面一致性——拼接缝处的价值跳变、
   累计分数、block/trial 计数器等有状态表面特征都会把干预惩罚污染成
   不连贯惩罚；而干预容易的独立 trial 任务残余本来≈0，无功效。

重启前置条件（按序）：

1. 零算力预分析：在 task 内把 choice 级 $(\mathrm{full}-w{=}20)$ NLL 差对
   位置与会话长度回归，确定残余跟随什么变化，缩小假设空间；
2. 功效预算：给出目标对比的最小可检测效应与所需 session 数；干预与对照必须
   同一次运行、相同 packing 顺序配对打分；
3. 对候选试点任务的 transcript 做有状态表面特征审计（累计分数、计数器等）；
4. 若重启，先只做 full vs within-participant far shuffle（整段打乱不拆
   action/outcome，几乎处处因果合法），swap/control 视 shuffle 结果再定。

以下干预设计保留作为重启时的参考。固定目标、instructions、最近 20 段、历史段数
和 tokenizer token budget，只干预超过 20 段的远端历史：

1. **full**：未经修改的真实历史；
2. **within-participant far shuffle**：只打乱同一参与者的远端历史，保留远端内容与
   近似边际统计；
3. **matched-participant far swap**：替换为同任务、同 session 阶段、长度和基本
   选择/反馈统计匹配的其他参与者远端历史，并一致重映射 response labels；
4. **length-matched neutral/control history**：仅在能构造训练分布内、因果合法的
   control 时使用，用于估计通用任务线索、输入长度或 session 阶段本身的收益。

token 长度匹配应在实际 tokenizer 下完成，不得靠无意义 padding 制造等长 prompt；
无法在预注册容差内找到匹配历史的样本应排除并报告。对选择会改变后续状态或奖励的
RL 任务，不得独立交换 action 和 outcome；主分析优先使用独立 trial、预生成反馈和
yoked trajectory 任务。

`full - far shuffle` 主要检验远端顺序/轨迹信息，`far shuffle - far swap` 检验
参与者特异信息，`far swap - neutral control` 检验一般任务、阶段和长度线索。
如果 shuffle 几乎无损而 swap 有损，远端更像无序的 participant profile；如果两者
都无损，E3 长尾更可能来自通用统计或长度线索；只有保持因果结构的自身有序远端历史
稳定胜出，才支持长期个体状态的解释。

### 7.6 任务信息消融与 shortcut 诊断（E6）

#### 7.6.1 已有工作与它们的结构性弱点

两篇 2025 年的短文已在小样本上建立了"存在性"：

- **Xie & Zhu (2025)**：从原 prompt 中删去大部分任务相关 token、保留选择历史，
  在**空间相关多臂老虎机**与**多线索判断**两个任务上，Centaur 仍胜过领域认知
  模型；移除选择历史时则表现下降。结论：Centaur 可能学到了一条对心理任务
  不敏感的 shortcut。
- **Liu & Ding (2025)**：三种操作——instruction-free、context-free（只留
  `<<J>>` 之类的选择 token）、misleading instruction（"看到 `<<` 就输出 J"）
  ——在**四个任务**上测试。context-free 条件下 Centaur 在 4 个中的 2 个仍胜过
  认知模型；instruction-free 与 misleading 条件下在全部 4 个上胜过认知模型。

**共同的可攻击点**：两者都**按"Centaur 相对认知模型优势最大"来选任务**（原文
明示）。用官方公布数据核算，它们选的任务并不代表全体：

| | 认知基线 | Centaur-70B | 优势 | 其中未微调 base-70B 即得 |
|---|---:|---:|---:|---:|
| 它们的 4 个任务 | 0.809 | 0.473 | 0.336 | **0.227** |
| 全体 34 任务 | 0.593 | 0.423 | 0.170 | 0.053 |

优势为全体均值的 **2.0 倍**，而"未微调即得"的成分是全体的 **4.3 倍**；
`tomov2021multitask` 位于 Centaur 优势的 97 百分位，其认知基线恰是补充材料中
**唯一没有给出技术定义**的模型（模型 11，只有名称与引文）。因此"去掉任务信息
后 Centaur 仍胜过认知模型"这一结论，在它们的样本上有相当部分可由"这些任务的
认知基线本就弱到未微调模型都能大幅超越"解释。另需注意：这些任务的 E2 bigram
可利用性并不突出（分位 39%/48%/24%/54%），即"序列依赖"在其自选样本上并未获得
特别强的支持。

#### 7.6.1b 一条两篇均未使用的独立证据：规模增益近乎为零

shortcut 假说有一个可检验的推论：**若性能主要来自易学的捷径，它应在小模型上
即已饱和，放大参数量不应带来明显收益。** 官方公布数字直接验证了这一点（34 个
可比任务）：

| 干预 | 收益（nat） | 相对规模的倍数 |
|---|---:|---:|
| 上下文（$w{=}0\rightarrow$full） | 1.4405 | ×132 |
| 行为微调（8B 上） | 0.1242 | ×11 |
| 参数量 ×8.75（8B→70B，微调后） | **0.0109** | ×1 |

未微调时规模增益 0.0176，微调后降至 0.0109（二者争夺同一部分可解释方差）；
方向一致但极小（70B 在 33/34 个任务上占优），在 Centaur-70B 相对认知模型的
总优势中只占 **6.4%**。

这条证据的价值在于：**它不需要任何消融**，官方公布数字里就有，因此不受
"消融造成格式 OOD"这一质疑的影响，也不受任务选择的影响（用全部可比任务）。
两篇短文均未使用它。作为论证，它与 E6 的消融互补——消融回答"任务信息的边际
贡献多大"，规模探针回答"这个指标对模型能力本身是否敏感"。

#### 7.6.2 全覆盖能回答而小样本不能的问题

1. **免疫上述选择性批评**；
2. **从存在性升级为量级**：Centaur 报告的总优势中，有多大比例在任务信息被
   移除后仍然存活——这是定量问题，2–4 个任务在结构上无法回答；
3. **产出分类学而非单一论断**：哪些 Psych-101 任务是 shortcut-prone，本身即
   对领域有用的输出（告诉他人哪些任务仍可用于理论检验），也是 benchmark 的前置；
4. **系统性的 base 模型对照**：区分"微调带来的 shortcut"与"任何 LLM 都有的
   上下文利用"。上表显示这一区分至关重要，而两篇均未系统控制。

注意本设计在**历史轴**上已强于二者：E3 是全 75 任务、七个窗口的剂量-反应曲线，
而它们是 2–4 个任务的二元消融；E2/E2-pop 另给出"多少可由纯计数解释"的分解；
§3.1/fig15 的结构性论证更不需要任何消融。**尚缺的只有任务信息这条轴。**

#### 7.6.3 先验分类 vs 实证测量（本实验的核心科学产出）

两篇短文判断一个任务是否具有历史依赖，用的是**基于任务设计的理论分析**。
Xie & Zhu 给出三类易诱发 shortcut 的实验设计：

1. **放大选择惯性**的设计（习惯形成、奖励学习）——序列依赖强；
2. **block 稳定**的设计——任务结构在长序列内不变，过去的选择很快暴露当前
   block 语境；
3. **说明只设定初始行为**的设计——影响第一个选择 $C_1$，其后按马尔可夫转移
   演进，任务信息的影响随时间衰减。

本设计则是**后验实证测量**：E2 的 bigram 增益（纯计数可提取多少）× E3 的
$L_{w=1}-L_{\mathrm{full}}$（截断掉更早历史的真实代价），已产出四区分类
（54 / 3 / 11 / 7，见 handoff 的 fig13）。

**二者的对照本身即一项零算力科学产出**：把 75 个任务按上述三类先验标注，检验
先验标签能否预测两项实证指标。

- 若一致 → 收敛效度成立，该理论分类可用于**预筛未来的任务设计**，这对
  benchmark 的任务选择有直接价值；
- 若不一致 → 说明理论分类漏掉了某些机制，或实证指标捕捉到了设计层面看不出的
  依赖，两种情形都值得单独报告。

标注 75 个任务需人工，但一次性且无需算力，建议在 E6 打分启动前完成，使先验
标签成为**预注册**的而非事后拟合的。

#### 7.6.4 条件设计

固定目标位置，与 §7.1 的窗口维度交叉，构成 2×7（或 3×7）因子设计：

1. **full**：原始完整 transcript（复用 E3 的 `full`）；
2. **instruction-swapped（主条件）**：把 header 换成**另一个任务的 instructions**
   ——保持自然语言 transcript 的训练分布内形态，但任务内容错误；
3. **instruction-free（诊断条件）**：直接删去 header，只保留 trial 段。

**为什么以 swap 而非 free 为主条件**：Psych-101 的 transcript 恒以说明开头，
直接删去 header 会造成**格式分布外**，使"性能下降"无法区分"信息被移除"与
"形态没见过"——这与 §7.4 删去 label-space 条件时的理由同源，而两篇短文均未
处理该混淆（Liu & Ding 的 misleading instruction 方向正确，但其元指令本身
高度 OOD）。instruction-free 保留为诊断量，用于与 swap 相减估计格式 OOD 成本。

swap 的配对规则须预注册：优先选择**动作数相同、选项标签空间可一致重映射**的
其他任务，避免把"合法按键集变了"混进来。

#### 7.6.5 实现与分析产出

工程上几乎零成本：`context_windows.segment_transcript` 已将 header 与含 choice
标记的段分离，`build_window_prompt` 已是 `header + 最近 w 段`；E6 只需替换或
置空 header 参数即可复用现有 runner 与网格。

主要分析产出：

1. 每个条件下的窗口曲线，及**任务信息 × 历史窗口的交互**；
2. Centaur 总优势中在 instruction-swapped 下存活的比例（全 75 任务加权）；
3. 逐任务的 shortcut 易感度排名，与 §7.6.3 的先验分类对照；
4. 与 base 模型同条件对照，分离"微调产生的 shortcut"与"通用 ICL"。

#### 7.6.6 结论边界

消融只能说明"模型在缺少任务信息时仍能预测"，**不能**直接推出"模型从不使用
任务信息"——full 条件下模型仍可能同时使用二者。因此主结论应表述为"任务信息
对预测的**边际贡献**有多大"，而不是"模型是否理解任务"。后者需要 §4.3 的
open-loop 与 §8 的表型比较，属第三阶段。

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

Noise ceiling 必须与上下文条件一致。full-context 模型不能使用 context-independent ceiling 作为同一尺度，否则模型可能通过利用参与者上下文表现出表面上的”超过 ceiling”。

**第一阶段不估计 ceiling。** full-context 条件下的 ceiling 是参与者特异的，没有干净的估计方法。第一阶段的归一化只用 null 基线：报告相对 uniform、群体 base rate、repeat-last-choice 等简单基线（即 E2 基线，一物两用）的提升量，即 $S_M=(L_{\mathrm{null}}-L_M)/L_{\mathrm{null}}$ 或直接报告差值。ceiling 留待后续只在独立 trial 任务上做，用跨参与者对同一刺激的反应一致性估计。

**不得用 Centaur（或任何 SOTA 模型）充当 ceiling**（2026-07-29 决定）。理由
不是操作困难，而是它会使指标失效：(a) noise ceiling 是不可约的随机性，SOTA
是会变的当下最好水平，用后者归一化会在更好的模型出现时产出 $S_M>1$；
(b) 更严重的是，若 Centaur 的优势有相当部分来自 ICL 与序列统计（§3.1 与 E5
结果指向此），按 Centaur 归一化等于**把被测偏差编进分母**——"接近 Centaur"
将等价于"善于复制这些统计"，恰好摧毁本设计要建立的区分。官方数字也支持这一
顾虑：Centaur-70B 相对认知模型 0.170 nat 的优势中，Psych-101 微调贡献 69%。

替代做法两条，可同时使用：

1. **当参考线，不当分母**：在结果图上把 Centaur-70B 画成标注好的参考水平线，
   报告原始差值。视觉上给出"离当下最好的模型还有多远"，指标本身不被污染。
2. **在能算的地方算真 ceiling**：独立 trial 且刺激跨参与者重复的任务，ceiling
   就是同一刺激下的跨参与者反应一致性，完全可估。本测试集中
   `peterson2021using`（1,466 人）、`hebart2023things`（1,218 人）、
   `ruggeri2022globalizability`（1,295 人）样本量均充足。**注意这批任务恰好
   就是 §13 所指的"干净战场"**：ceiling 难算的任务（序列/个体历史依赖）正是
   ICL 混淆最重的任务，而 ceiling 好算的任务正是最能检验机制的任务。因此
   不必求全，只在干净战场上估 ceiling 即可正中要害。

若某任务的 ceiling 确实不可测量，正确处理是**不归一化**（只报相对 null 基线
的原始差值），而不是引入替身——替身不解决不可测量，只是把它藏起来。

### 9.2 分层聚合：micro 与 macro 的定义

**记号**。设任务（experiment）$e$、参与者（session）$i$、该会话内第 $j$ 个
choice。打分器对每个 choice 输出该 choice 内所有 target token 的 NLL **之和**
$\ell_{e,i,j}$ 与 token 数 $n_{e,i,j}$。三个层级的规模记为 $E$ 个任务、任务
$e$ 内 $I_e$ 个参与者、参与者 $(e,i)$ 内 $J_{e,i}$ 个 choice。

#### 9.2.1 三种聚合，彼此不可混用

**(a) token-micro**（把所有 token 倒进一个池子）：

$$
L_{\mathrm{micro}}
=\frac{\sum_{e,i,j}\ell_{e,i,j}}{\sum_{e,i,j}n_{e,i,j}}
$$

即 `sum(nll) / sum(num_tokens)`。**注意**：CSV 中的 `nll` 已是 choice 内的
token NLL 之和，因此不得再乘 `num_tokens`。

**(b) 分层 macro choice NLL**（本设计的**主指标**）：逐层等权，且每一层的单位
是 **choice** 而非 token——

$$
L_{e,i}=\frac{1}{J_{e,i}}\sum_{j}\ell_{e,i,j},
\qquad
L_{e}=\frac{1}{I_e}\sum_{i}L_{e,i},
\qquad
L_{\mathrm{macro}}=\frac{1}{E}\sum_{e}L_{e}.
$$

**(c) P0 的官方 evaluator 口径**：会话内按 token 平均、再对会话等权，见 §9.3。

三者的差别不是精度问题而是**权重定义**问题：micro 让 choice 多的任务与
token 多的 choice 占主导；macro 让每个任务等权；P0 让每个 session 等权。
**任何跨口径的数值相减都是无效的**（§13 的报告纪律）。

#### 9.2.2 为什么以 macro 为主指标

任务规模在本测试集中极度偏斜：最大的三个任务
（`peterson2021using` 1,466 人、`ruggeri2022globalizability` 1,295 人、
`hebart2023things` 1,218 人）占了全部 6,561 个 session 的约六成，而尾部十余个
任务只有 2–6 人。若用 micro，"75 个任务上的平均表现"实际上会退化为"两三个大
任务上的表现"。macro 通过逐层等权显式地拒绝这一点，这也是跨任务比较的通行做法。

#### 9.2.3 为什么仍然必须同时报告 micro

micro 不是 macro 的劣质版本，二者回答**不同的问题**（每个 token 预测得多准
vs 平均每个任务表现如何）。保留 micro 有四个不可替代的用途：

1. **与外部数字对齐**：论文与官方 evaluator 的数字都由 token 级平均得到。本
   项目能与官方公布结果对到 $r=1.00000$，靠的正是保留了同口径的量；只算三层
   macro 则与任何公开数字都对不上，管线验证无从谈起。
2. **一致性检验的首选工具**：micro 就是 `sum(nll)/sum(tokens)`，不依赖任何
   分组决策，同一批行怎么算都是同一个数。E3-full 与 E0 相同 key 的对照
   （0.63696 vs 0.63705）、zorowitz UTF-8 修复前后的对照（0.5971341 vs
   0.5971275）都在 micro 上做——此时 macro 的分组加权反而是干扰。
3. **两者之差本身是信息**：本项目 E0 的 micro 0.60 与 macro 0.90 相差约 0.3
   nat，这说明**choice 数多的大任务恰好是模型预测得好的任务**。若某结论在两个
   口径下都成立（如微调增益），即可确认它不是加权方式的 artifact；若两口径
   打架，则说明效应集中在大任务或小任务，本身就值得追查。
4. **统计稳定性**：macro 给 400 个 choice 的小任务与 19 万 choice 的大任务同样
   的一票，小任务的噪声会明显晃动总分；micro 在百万级样本上几乎不动。

**纪律**：**macro 做科学推断，micro 做对表、查管线与稳健性附注。** 报告时两者
都给，并显式标注口径。

#### 9.2.4 不确定性与显著性

使用 participant bootstrap 与 task bootstrap 估计不确定性，并固定 seed 与重复
次数（当前产物为 seed=20260728、5,000 次）。显著性分析应以参与者或任务为独立
单位，避免把高度相关的 trial 当作完全独立观测。正态近似区间可作描述性审计，
但必须明确标注，不得冒充 bootstrap。

### 9.3 P0 的官方 evaluator 聚合与报告边界

P0 不沿用当前 full-context runtime cache 的 75-experiment token-micro 汇总，也不沿用
本设计的 participant→task macro choice NLL。对 family $f$ 中的每个 session $i$，先计算

$$
\ell_i
=
\frac{\sum_{r\in R_i}\operatorname{NLL}(r)}
{\sum_{r\in R_i}\operatorname{tokens}(r)},
\qquad
L_f^{\mathrm{P0}}
=
\frac{1}{|S_f|}\sum_{i\in S_f}\ell_i.
$$

这与官方 `per_device_eval_batch_size=1` 下的 `eval_loss` 聚合语义对应。因此应逐 family
报告 `L_f^{P0}`；可以另外给出 token-micro NLL 和 36-family 的等权平均以便诊断，但后者
不是官方脚本原本输出的全局 scalar，也不能替代本设计的分层 macro 指标。

### 9.4 计数基线：定义、选择理由与文献依据

E2 的四条基线不是随手挑的，它们是一条**按假设强度递增的嵌套阶梯**，每一级都对应
行为科学里一条被独立确立过的规律。共同点是都只需要数数，不需要任何任务理解——
因此它们共同定义了"**不能算作认知证据**"的地板：凡是查表能达到的水平，都不构成
模型理解了人类认知的证据。设某任务有 $k$ 个选项：

| 基线 | 预测 | 自由参数 | 对应的行为规律与文献 |
|---|---|---|---|
| uniform | $1/k$ | 0 | 纯机遇水平；对应 Centaur 官方 `random` 基线 |
| base rate | $P(c)$ | $k-1$ | 稳定的边际选择偏好；简单 actuarial 基线难以击败的传统 (Dawes 1979; Gigerenzer & Goldstein 1996) |
| sticky | 重复上次选择给 $\theta$，其余均分 $(1-\theta)/(k-1)$ | 1 | perseveration / choice stickiness / win-stay-lose-shift (Nowak & Sigmund 1993; Lau & Glimcher 2005; Ito & Doya 2009; Katahira 2015; Gershman 2016; Miller, Shenhav & Ludvig 2019) |
| bigram | $P(c\mid c_{\text{prev}})$ | $k(k-1)$ | 一阶马尔可夫序列依赖；n-gram 预测与熵基准的经典范式 (Shannon 1951; Chen & Goodman 1999) |

**为什么以 bigram 收尾**：它是纯计数在选择标签空间上能表达的上限——再往上就必须
引入刺激、反馈或任务结构，那已经不是"表面统计"而是任务理解了。因此
"模型 vs bigram"的差值，是"超出局部序列统计的部分"最保守的下界估计。

**为什么不给 sticky 之外的启发式单独留位置**：perseveration 是选择序列上最稳健、
跨任务复现最广的单条规律（上表文献），且只需一个参数；其余启发式（WSLS 的
反馈依赖版、recency 加权等）都需要读取反馈或额外参数，属于第三阶段认知模型的
范围，不应混入"零理解基线"这一层。

#### 9.4.0 计算式（实现即此，见 `mt/models/baselines/sequence.py`）

设某会话的选择序列为 $c_1,\dots,c_T$，标签集 $\mathcal L$（取该会话内实际出现
过的标签，$k=|\mathcal L|$），平滑常数 $\alpha$。所有基线都在**预测第 $t$ 个
choice 时只使用前 $t-1$ 个 choice 的计数**，打完分再更新（prequential，见
§9.4.1）。记前 $t-1$ 个 trial 中：

- $c^{(t)}(x)$ = 标签 $x$ 出现次数；
- $n^{(t)}(x,y)$ = 转移 $x\rightarrow y$ 出现次数，$n^{(t)}(x,\cdot)$ 为其行和；
- $r^{(t)}$ = 重复次数（$c_s=c_{s-1}$ 的个数），$m^{(t)}=t-2$ 为转移总数。

四条基线的预测概率：

$$
\begin{aligned}
p_{\mathrm{uniform}}(c_t) &= \frac{1}{k}
&&\Rightarrow\ \mathrm{NLL}=\ln k \ \text{（常数）}\\[4pt]
p_{\mathrm{base}}(c_t) &= \frac{c^{(t)}(c_t)+\alpha}{(t-1)+\alpha k}\\[4pt]
p_{\mathrm{sticky}}(c_t) &=
\begin{cases}
\theta^{(t)}, & c_t=c_{t-1}\\
\dfrac{1-\theta^{(t)}}{k-1}, & \text{否则}
\end{cases}
&&\theta^{(t)}=\frac{r^{(t)}+\alpha}{m^{(t)}+2\alpha}\\[4pt]
p_{\mathrm{bigram}}(c_t) &=
\frac{n^{(t)}(c_{t-1},c_t)+\alpha}{n^{(t)}(c_{t-1},\cdot)+\alpha k}
\end{aligned}
$$

每个 choice 的记分为 $-\ln p$。**边界情形**（实现中已处理）：

- $t=1$（无 $c_{t-1}$）：sticky 与 bigram 均回退到 $p_{\mathrm{base}}$；此时
  所有计数为 0，$p_{\mathrm{base}}=1/k$，即三者在首 trial 上都等于 uniform；
- $k=1$（该会话只出现过一个标签）：全部概率取 1，NLL 为 0；
- 平滑常数：会话内在线版取 $\alpha=\tfrac12$（Jeffreys / Krichevsky–Trofimov），
  群体版实现取 $\alpha=1$（Laplace）。在本项目的计数量级下该差异不改变结论
  排序，但跨实现比较时必须显式声明（§9.4.2）。

**群体版**（E2-pop，`run_population_baselines.py`）用同样的公式，但计数**不是
会话内累积**，而是在**不属于 test split 的参与者**上一次性拟合并冻结：
$c(x)$、$n(x,y)$ 取自训练参与者全体，支撑集 $\mathcal L$ 取该任务全表出现过的
选项集；随后对 test 参与者逐 choice 打分（会话首 choice 用 marginal，其后用
transition）。该协议与论文认知模型的拟合协议**逐条对应**（训练参与者拟合、
held-out 参与者评估），因此可直接与之比较。

#### 9.4.1 两种计数来源：会话内在线 vs 跨参与者群体

同样四条基线，**计数从哪里来**决定了它对照的是模型的哪种能力：

| | 会话内在线（E2） | 跨参与者群体（E2-pop） |
|---|---|---|
| 计数来源 | 同一 session 的前 $t-1$ 个 trial | 不在 test split 的其他参与者 |
| 对照的模型能力 | **上下文学习（ICL）** | **行为微调** |
| 可用标签空间 | transcript 的随机按键 | 仅规范编码（见下） |
| 因果性保证 | prequential，严格无泄漏 (Dawid 1984) | 训练/测试划分与 Centaur 一致 |

在线版按 prequential（预测式）原则打分：预测第 $t$ 个 trial 只使用同 session 前
$t-1$ 个 trial 的计数，先打分后更新，因此严格因果、无泄漏，且概念上正好对应
"ICL 能从上下文中薅到的表面统计"。

**群体版必须在规范标签空间上做。** Psych-101 为每个参与者随机分配按键字母，
因此跨参与者在原始标签空间上的计数是在数随机化本身（试点中群体 base rate
$\approx\ln 26$，与瞎猜无异）。会话内计数不受影响，因为映射在 session 内固定。
规范编码需从 HF 上逐任务发布的原始 table 数据集取得（统一的 `choice` 列）。

**群体版的概念地位**：Centaur 的微调正是在**其他参与者**的数据上做的，因此
E2-pop 是**微调的计数版对照**——模型超出 E2-pop 的部分，才是微调学到的、
超出"记住群体选择统计"的东西。这与 E2-online 对照 ICL 恰好构成一对。

#### 9.4.2 平滑与报告纪律

所有计数概率使用 add-$\tfrac12$（Jeffreys / Krichevsky–Trofimov）平滑，避免未
观测到的选项取到零概率而使 NLL 发散；平滑常数的选择在本项目的计数量级下不影响
结论排序，但跨实现比较时必须显式声明（平滑方案对 n-gram 结果的影响见
Chen & Goodman 1999）。

报告时须注意三条边界：

1. **基线看不到逐 trial 的可选项集合**（§12.2 E2 行的既有说明）：纯标签空间的
   计数无法知道某个 trial 只提供了其中两个选项，因此在这类任务上贴 $\ln k$ 而非
   $\ln 2$ 走；而 LLM 读得到选项。这使基线在此类任务上被系统性低估。
2. **选项支撑集不同的两个空间不可直接比较**：例如 `hebart2023things` 在规范空间
   是 1,823 个物体的边际分布，而 transcript 每 trial 只呈现 3 个候选——前者的
   NLL 与后者不在同一尺度上，必须从此类对比中排除。
3. **模型输给基线时须区分失败机制**：NLL 高于 uniform 有两种来源——(a) 有信号但
   过度自信（置信度超出信号支撑）；(b) 无方向信号却仍偏离均匀（偏离本身即纯亏损，
   由 Jensen 不等式）。二者都属校准失败，但含义不同，应通过检查模型赋予人类实际
   选项的概率分布（$p=e^{-\mathrm{NLL}}$）来区分，不可笼统表述为"押错方向"。

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

当前主层展示 Llama-base 与 Minitaur 的 75-experiment task-macro context-window
curve、五个位置 strata 和 participant/task bootstrap uncertainty；session-macro 与
token-micro 作为补充。后续再叠加 history swap、history shuffle、结构化输入和
自然语言输入。位置图必须标注窗口单位是 transcript 段，而非保证原子的 choice。

### Figure D：open-loop 认知表型（第三阶段）

以 effect-size forest plot 或参数分布图比较人类、Centaur、Llama 和认知模型。应避免仅使用雷达图展示均值，因为雷达图不容易表达不确定性和个体差异。

## 12. 第一阶段实验分解与执行计划

### 12.1 范围约束（硬性）

1. **只做推理侧评估**：使用公开的 Centaur / Llama checkpoint 打分，不做任何微调（第二阶段）、不复现认知模型（第三阶段）。认知模型对比使用原论文公开的结果数字，因此只能进入任务级汇总对比；逐 trial、逐条件的对比只在 Centaur、Llama 和简单基线之间进行。
2. **数据划分必须沿用 Centaur 原论文的 held-out split。** 公开 checkpoint 在每个实验约 90% 的参与者上微调过；自建 split 会把 Centaur 的训练参与者混入测试集，造成泄漏，所有对比作废。任何阶段都不得自行划分 split。
3. 所有上下文操作保持自然语言格式（§6 主定义），编码格式对照留给第二阶段。
4. 不估计 noise ceiling（§9.1），归一化只用 null 基线。
5. P0 只作为评估协议控制：必须显式报告 runtime NF4 与原论文 70B/BF16 环境的差异，
   不把它写成论文数值复现或机制证据。

### 12.2 实验列表

| # | 实验 | 验证内容 | 算力 |
|---|---|---|---|
| E0 | 复现论文数字：原始 split + 原始 prompt + 原始打分方式，对齐论文中若干任务的 NLL | 打分管线（tokenization、response token 定位、split）正确；一切后续实验的前置 | 每任务一次 full-context 打分 |
| P0 | paper-like NLL：runtime-NF4 的论文协议控制轨道（36-family、32,768-token、session-mean evaluator 协议） | 检验运行时量化条件下 task allowlist、tokenization、response-token 定位、截断与聚合是否兼容官方 evaluator；不替代 E0 | 可在 cutoff-span 审计通过时从已完成的 runtime full-context cache 派生，否则每个 session 直接截断后打分 |
| E1 | 逐 trial 位置的 NLL 曲线：用 E0 的 full-context 打分结果按 trial 位置分桶，画 Centaur 与 Llama 的曲线 | 优势出现在早期还是晚期 trial → 区分跨参与者泛化与上下文内个体适应；即 §5.1 修正后的适应曲线 | 零（复用 E0 结果重新聚合） |
| E2 | 简单序列基线：uniform、base rate、repeat-last（粘性）、bigram，定义与文献依据见 §9.4。**实现发现**：Psych-101 对每位参与者随机分配按键字母，原始标签空间上的跨参与者群体计数无效（试点中群体 base rate ≈ ln 26 的纯噪音）；因此主版本为**会话内在线（prequential）计数**——预测第 t 个 trial 只用同 session 前 t−1 个 trial，严格因果、无泄漏，恰为"ICL 可从上下文提取的表面统计"的对照。局限：纯标签空间基线看不到逐 trial 的可选项集合（如交替出现的选项对），独立 trial 任务上没有可利用信号 | Centaur 优势中有多少能被局部序列统计解释；同时充当 §9.1 的 null 基线 | 零 GPU（已完成，2026-07：75 实验 × ≤50 人抽样，43.7 万 choice） |
| E2-pop | 规范空间的**群体**计数基线（§9.4.1）：用不在 test split 的参与者拟合 base rate 与 bigram，在 test 参与者上打分。依赖 HF 逐任务原始 table 的规范 `choice` 编码，随机按键问题因此解除 | 微调的计数版对照：模型超出 E2-pop 的部分，才是超出"记住群体选择统计"的能力 | 零 GPU（已完成，2026-07-29：44/59 个有 table 的 experiment、5,088 个 test 参与者、73.8 万 choice） |
| E3 | 上下文窗口截断（§7.1）：instructions + 最近 $w$ 个含 choice transcript 段，$w\in\{0,1,2,5,10,20,\text{full}\}$；E0-informed 五点位置网格 | 有效记忆范围；优势是否依赖近乎无界的长上下文，并保留第 1→2 段的早期 ICL 对比；**Minitaur 与 Llama-base runtime-NF4 全量已完成** | 每模型 6,561 session × 最多 5 位置 × 7 window，459,102 条 response；结果见 §6.1/§7.1 |
| E3a | $w=0\rightarrow1$ 收益分解（§7.4）：合法 label 重归一化解析诊断 + instructions、format-only、matched other participant、own history 四条件阶梯 | 把首段历史收益拆成 response alphabet、格式校准、任务识别和参与者特异适应 | 先在可交换的代表任务上试点，Minitaur 与 Llama base 同跑 |
| E3b | 超过 20 段历史的长度匹配干预（§7.5，**暂缓**） | 区分远端顺序、个体 profile、任务阶段与单纯长度线索 | 暂缓：先过 §7.5 的预分析与功效预算门槛；若重启先做 shuffle-only 试点 |
| E6 | 任务信息消融与 shortcut 诊断（§7.6）：full / instruction-swapped（换成他任务说明，格式分布内）/ instruction-free，与 §7.1 的窗口维度交叉 | 任务信息对预测的边际贡献有多大；全 75 任务上量化 shortcut 的量级而非仅存在性；先验设计分类（Xie & Zhu 三类）与实证分类（E2×E3）的收敛效度 | 复用 E3 网格与 runner（header 可替换/置空），每条件一次打分 |
| E4 | 语言表面扰动（§7.3，保持自然语言格式）：同义改写叙述措辞、交换按键/选项标签、可交换任务上打乱历史顺序 | 是否依赖不改变任务信息的表面语言线索 | 每种扰动一次打分 |
| E5 | 2×2 因子分析（§6）：{Llama, Minitaur} × {full, matched($w$ 固定)}，计算上下文增益与交互项 | 微调是否增强了历史利用；**8B/NF4 已完成：$w\ge1$ 交互近零，微调主要改善 $w=0$ 冷启动** | 零（复用 E0 + E3；结果见 §6.1） |

依赖关系：E0 → E1 / E3 / E4，E3 → E3a / E3b；P0 是与 E0 并列的 runtime-NF4
协议控制，可在其 cutoff-span 审计通过后从已完成的 runtime full-context cache
派生，否则独立直接打分。Minitaur 与 Llama-base 的 E0/E3 均已完成，8B/NF4 的
E5 微调×上下文交互也已计算（§6.1）。E2 完全独立。建议顺序更新为 E3/E5 固定
summary 与 Figure B/C → E1 双模型正式图 → E3a 代表任务试点 → E4；E3b 暂缓，
P0 已可并列报告。

### 12.3 执行步骤

按"单任务验证逻辑 → 小模型跑通代码 → 真 checkpoint 单任务对齐 → 推广全量上服务器"推进，每步有完成标准，不通过不进入下一步：

1. **下载数据与模型**。具体资源：
   - 训练集 [marcelbinz/Psych-101](https://huggingface.co/datasets/marcelbinz/Psych-101)（自然语言 transcript，160 实验 / 60,092 参与者）；
   - 测试集 [marcelbinz/Psych-101-test](https://huggingface.co/datasets/marcelbinz/Psych-101-test)（即原论文 held-out split，JSON，约 92 MB；**gated，需要 HF 账号同意条款后才能下载**）；
   - checkpoint：[Llama-3.1-Centaur-70B](https://huggingface.co/marcelbinz/Llama-3.1-Centaur-70B)（合并权重）、[Llama-3.1-Centaur-70B-adapter](https://huggingface.co/marcelbinz/Llama-3.1-Centaur-70B-adapter)（LoRA adapter）、[Llama-3.1-Minitaur-8B](https://huggingface.co/marcelbinz/Llama-3.1-Minitaur-8B)（同配方 8B 小版本，作者标注适合原型验证，但对分布外实验泛化较弱）；
    - 官方复现代码：[github.com/marcelbinz/Llama-3.1-Centaur-70B](https://github.com/marcelbinz/Llama-3.1-Centaur-70B)（E0 的 prompt 构造与打分方式以此为准；P0 沿用其 evaluator 协议作 runtime-NF4 对照）。
2. **算力预算与实测**（已完成，2026-07）：测试集 6,561 个 session、75 个实验、117.8 万 choice、8,970 万字符；用 Minitaur tokenizer 标定为约 27M token。最长真实 transcript 是 `xiong2023neural/exp1.csv` participant 28 的 53,091 token：在 Minitaur 的 128k context 内，但超过论文 protocol 的 32,768-token 截断；当前 full-context runtime cache 至少有 75 个 session 超过该阈值。该 runtime 打分每 session 一次前向即可，8B 模型上为个位数 GPU 时量级。**E3 不能对每个原始 choice × 每个 $w$ 重构 prompt**（总量会达数十亿 token）：E0-informed 五点网格覆盖第 1、第 2、10%、50% 和末位置，预估约 3.78× E0 的原始字符工作量。实现对早期位置的等价 window prompt 去重；本次 RTX 5060 Ti / runtime-NF4 全量作业实际在一天内完成，得到 459,102 条 response 记录、需评分的 133,034 个 effective prompt cell，且没有 failed/skipped session。分工保持不变：Minitaur-8B 扫全部条件（E3/E4 多条件矩阵及 P0 runtime 对照），Centaur-70B 只跑主结果（E0 复现与 full/matched 两条件）。注意 Minitaur 没有论文主图参考数字，P0 也只能作为 8B runtime-NF4 对照。
3. **单任务 + 本地小模型跑通代码**：选定一个任务（建议 two-step 或某 bandit）的单个参与者 transcript，用本地 0.5B 级小模型搭建并验证完整打分管线——prompt 构造、response token 定位、逐 trial NLL、截断、扰动、逐位置聚合。此阶段数字无意义，只验证代码逻辑，配单元测试。E2 的简单基线也先在这个任务上实现并出数（无 GPU）。
4. **单任务 + 真 checkpoint（服务器）**：E0 在该任务上对齐论文数字（不通过则回到步骤 3 排查管线）；随后在该任务上跑 E1、E3、E3a、E3b、E5、E4。P0 是并列的 runtime-NF4 控制：在代表性官方 family 上验证 head-32k 截断、response mask、session-mean loss 后，再扩展到完整 36-family allowlist；本次数据可在 span audit 通过后从 runtime cache 派生，否则直接截断重打分。
5. **复核后推广全量**：Minitaur runtime-NF4 的 E0/P0/E3 与 Llama-base 的 E0/E3
   已推广到全量并完成；后续需补 E3a 代表任务试点、E4 和 BF16/HPC 主结果，
   产出 Figure A–C。只有试点显示稳定、可解释且计算可承受时，E3a/E3b 才扩展全量。
   P0 与论文 70B/BF16 结果分栏报告，不互相替代。
6. **汇总分析**：按 §9.2 分层聚合，E5 因子分解显式报告交互项，对照 §13 的结论边界撰写结果。

### 12.4 第二、三阶段（占位，届时再设计）

- **第二阶段（动训练）**：做"结构化输入 vs 自然语言输入"的编码对照（§6 操作 2–4 在此阶段才有干净的解释）。结构化输入的构造方式届时再设计——原先设想的 canonical field registry / 数据契约层已于 2026-07-31 移除，可从 git 历史取回。
- **第三阶段（认知模型侧）**：层级贝叶斯认知基线 + prefix 后验更新（§5）、open-loop simulation 与认知表型投影（§4.3、§8、Figure D）。

### 12.5 打分引擎待办（工程，非科学）

HPC 不可用时的本地小规模预览暴露了几处工程改进点，按性价比排序，后续实现：

1. **logits 内存优化（已完成，2026-07）**：`_forward` 改为跑 base model 取 hidden states、只在需要打分的位置上应用 `lm_head`，全词表 `[batch, seq, vocab]` 张量不再 materialize。数值与 dense 路径逐 token 等价（测试 `test_hidden_state_path_matches_dense_logits_path` + Qwen 真模型验证 0 差异）。长会话显存降约一个量级（每位置张量维度 128k→4k）。不暴露 `.model` + output embedding 的模型自动回退到 dense。效果：24GB Mac 上 `--max-chars` 可行阈值从 25000 抬到 100000+；剩余极端长会话压力来自 prefill 激活与注意力工作区，而非 KV cache（打分路径已设置 `use_cache=False`）。
2. **量化加载 `--load {8bit,4bit}`（已完成，2026-07）**：bitsandbytes 量化，`load_model` 用 `BitsAndBytesConfig` + `device_map="auto"`。RTX 5060 Ti 实测 NF4 模型常驻显存 5.68 GiB、短评估峰值 6.27 GiB；从 BF16 checkpoint 现场量化时主内存峰值 15.83 GiB，完成后回落到 1.86 GiB。测试集最长 session（168,968 字符、4,800 choices）的 full-context runtime 与 E3 full-window 探针已通过，整卡峰值约 15.8/16.3 GiB，故 16GB 可跑但余量很窄；随后 E3 五点全量也在该卡无失败完成。Windows CUDA wheel 没有编译 FlashAttention，默认 SDPA 的 GQA 曾回退到平方内存 math 路径；打分核心现按 Flash → cuDNN → memory-efficient → math 选择 CUDA 后端，非 CUDA 路径不变。量化会改变 NLL：P0 必须固定 held-out split、prompt、36-family protocol、metric 与 NF4 配置，并明确标记为 runtime-quantized，不能与 BF16 或 70B 主结果混写。当前 full-context runtime 产物的 `sum(nll) / sum(num_tokens)` 是 token-micro **诊断**，与本设计的 participant/task macro choice NLL 都不能冒充 P0 的 session-mean `official_eval_loss`。CUDA 环境显式安装 `.[centaur-eval]`；Mac 保持 `--load none`，HPC 使用集群原生精度环境。
3. **内存稳定性（已完成，2026-07）**：`_forward` 用 `use_cache=False`(打分不需 KV cache），`empty_device_cache` 每会话 `gc.collect()` + 清 device 缓存。E0/E3 runner 逐会话捕获 OOM，记入 `.failed.csv` 跳过而非中断。**已知限制**：24GB 统一内存的 Mac 跑 bf16 的 8B 仍不稳(模型 16GB，运行时波动冲破物理内存)；本地全量预览应改用 CUDA 卡 + `--load 4bit`，或直接上 HPC。
4. **CPU offload（待实现）**：`device_map="auto"` + `offload_folder`,把溢出层放系统内存,在 16GB 显卡上跑精确 fp16,代价是 PCIe 搬运变慢。
5. 临时的内存闸 `--max-chars`（已实现）按字符跳过极端长尾会话；即使走量化路线也保留作应急守卫，但正式全量结果不应静默丢弃长会话。

## 13. 可以支持与不能支持的结论

本次已完成的双模型 E3 不再只是 full-context 相关性。对于
`Llama-3.1-8B base / Minitaur-8B BF16 checkpoint, runtime NF4`，按总体和跨任务
聚合，删除更多较早历史段时 NLL 单调升高，因此可以支持：

> 在本次 teacher-forced 五点抽样和 prompt 重建协议下，删除较早历史段会干预
> 两个模型的预测表现；大部分可见上下文收益来自最近一个含 choice transcript 段，
> 但 session 中后期及部分任务仍能从超过 20 段的历史获得额外收益。除无历史冷启动
> 外，两条 context-gain 曲线几乎重合。

这是模型输入干预证据，但截断同时改变可见内容、长度和 transcript 连贯性，不能
单独识别抽象的 memory-span 参数，也不涉及人类行为因果机制；窗口单位还不是保证
原子的 trial/choice。Llama-base 配对曲线显示，$w\ge1$ 的微调×上下文交互接近零，
所以现有证据不支持“Psych-101 微调增强了历史利用效率”；微调的主要作用更像是改善
$w=0$ 的任务接口、response alphabet 校准与行为先验。

仅凭 full-context teacher-forced NLL，可以支持：

> Centaur 是一个强大的跨任务条件行为预测器，能够利用完整实验历史预测 held-out participants 的下一次反应。

不能直接支持：

> Centaur 实现了与人类相同或更好的认知机制。

如果 Centaur 在 matched context、$k=0$ 或短 prefix、open-loop simulation、history perturbation 和共同认知表型比较中都接近人类，则可以更有力地支持：

> Centaur 学到的并不只是完整历史中的表面预测规律，而包含能够跨参与者、跨任务结构和生成条件稳定表达的人类行为规律。

如果其主要优势只在 full transcript 或长 prefix 下出现，则更准确的结论是：

> Centaur 的突出能力主要是利用长上下文进行在线行为识别和条件模仿，而其机制层面的人类相似性仍需独立验证。

## 14. 参考资料

- Binz, M. et al. (2025). [A foundation model to predict and capture human cognition](https://www.nature.com/articles/s41586-025-09215-4). *Nature*, 644, 1002–1009.
- Binz, M. et al. (2025). [Supplementary Information: domain-specific models and modelling details](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-025-09215-4/MediaObjects/41586_2025_9215_MOESM1_ESM.pdf).

对 Centaur 的既有评估性短文（§7.6 的直接对话对象。二者均为 2025 年预印本，
未核对正式发表信息，撰写时须补齐 venue/DOI）：

- Xie, H. & Zhu, J.-Q. (2025). *Centaur May Have Learned a Shortcut that
  Explains Away Psychological Tasks.* — 删去任务 token 保留选择历史，2 个任务。
- Liu, W. & Ding, N. (2025). *Can Centaur Truly Simulate Human Cognition? The
  Fundamental Limitation of Instruction Understanding.* — instruction-free /
  context-free / misleading-instruction 三条件，4 个任务；分析脚本
  <https://github.com/y1ny/centaur-evaluation>。

计数基线（§9.4）的文献依据。以下条目的作者、年份、期刊与 DOI 均已通过 Crossref
核对；正式撰写时仍建议核对页码与卷期：

- Chen, S. F. & Goodman, J. (1999). An empirical study of smoothing techniques for language modeling. *Computer Speech & Language*, 13(4). [10.1006/csla.1999.0128](https://doi.org/10.1006/csla.1999.0128) — 平滑方案对 n-gram 结果的影响。
- Dawes, R. M. (1979). The robust beauty of improper linear models in decision making. *American Psychologist*. [10.1037/0003-066X.34.7.571](https://doi.org/10.1037/0003-066X.34.7.571) — 简单基线难以击败。
- Dawid, A. P. (1984). Present position and potential developments: Some personal views. Statistical theory. The prequential approach. *Journal of the Royal Statistical Society A*, 147(2). [10.2307/2981683](https://doi.org/10.2307/2981683) — prequential 原则。
- Gershman, S. J. (2016). Empirical priors for reinforcement learning models. *Journal of Mathematical Psychology*. [10.1016/j.jmp.2016.01.006](https://doi.org/10.1016/j.jmp.2016.01.006) — 选择模型中的 perseveration 项。
- Gigerenzer, G. & Goldstein, D. G. (1996). Reasoning the fast and frugal way: Models of bounded rationality. *Psychological Review*, 103(4).
- Ito, M. & Doya, K. (2009). Validation of decision-making models and analysis of decision variables in the rat basal ganglia. *The Journal of Neuroscience*.
- Katahira, K. (2015). The relation between reinforcement learning parameters and the influence of reinforcement history on choice behavior. *Journal of Mathematical Psychology*.
- Lau, B. & Glimcher, P. W. (2005). Dynamic response-by-response models of matching behavior in rhesus monkeys. *Journal of the Experimental Analysis of Behavior*. [10.1901/jeab.2005.110-04](https://doi.org/10.1901/jeab.2005.110-04) — 选择历史项的经典出处。
- Miller, K. J., Shenhav, A. & Ludvig, E. A. (2019). Habits without values. *Psychological Review*.
- Nowak, M. & Sigmund, K. (1993). A strategy of win-stay, lose-shift that outperforms tit-for-tat in the Prisoner's Dilemma. *Nature*, 364. [10.1038/364056a0](https://doi.org/10.1038/364056a0)
- Shannon, C. E. (1951). Prediction and entropy of printed English. *Bell System Technical Journal*, 30(1) — n-gram 预测与熵基准的原始范式。

