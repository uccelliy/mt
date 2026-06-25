# 今日晚间讨论报告：从“拟合人类行为的研究意义”到行为模型评价框架

## 0. 报告范围

本报告整理的是从“我现在找不到研究意义了……”这一段开始的讨论内容，不包括前面关于高斯分布公式的技术问题。

整理目标：

1. 保留用户提出的问题，按问答模式整理。
2. 提炼当前研究方向的核心矛盾、可行定位和后续工作。
3. 为下周汇报提供一个较清晰的叙述框架。
4. 为后续实现 data contract、baseline、evaluation framework 提供工作清单。

---

# 1. 问题一：只说“理解人类认知”太虚，拟合行为数据还有什么意义？

## 用户问题

你提出的问题大意是：

> 我现在找不到研究意义了。虽然知道要拟合人类行为数据，但想不到这有什么意义。论文里经常说“帮助我们更加理解人类认知”，除此之外还有其他意义吗？

## 回答整理

“理解人类认知”本身太大、太虚，不足以支撑一个博士项目。更合理的做法是把意义压到具体问题：

不是：

```text
拟合人类行为 → 理解人类认知
```

而是：

```text
跨任务 trial-level 行为建模
→ 检验模型是否能学习任务结构、个体结构和动态历史依赖
```

可以分成几类更具体的意义。

### 1.1 方法意义：从 task-specific model 到 task-general model

传统认知建模通常是：

```text
一个任务 → 一个模型
```

例如：

```text
two-step task → model-based/model-free RL
delay discounting → discounting model
go/no-go → inhibition model
n-back → working memory model
flanker/Stroop → conflict/evidence accumulation model
```

问题是：

1. 每个模型只解释一个任务。
2. 不同任务之间参数不一定可比。
3. 很难统一建模一个人在多个任务中的行为。
4. 每个任务都需要特定预处理和特定模型假设。

因此，研究意义可以从“解释某个单一认知过程”转为：

> 构建和评估一种能跨多个认知任务预测 trial-level choice/RT 的统一行为建模框架。

### 1.2 表征意义：学习任务之间的共享结构

你的方向不一定是“解释某个具体心理机制”，而是研究模型能否从多任务行为序列中学到：

```text
任务结构
策略结构
个体差异结构
历史依赖结构
反应时动态
反馈适应过程
```

这比单纯做组水平差异分析更抽象，也更接近 representation learning / meta-learning。

### 1.3 预测意义：把认知模型放进现代 AI 的预测检验框架

传统认知模型常强调解释性，但不一定强调 held-out prediction。

可以转向：

```text
模型能否预测 held-out trials？
模型能否预测 held-out participants？
模型能否预测 held-out tasks？
模型能否预测 held-out task families？
模型能否利用少量 history 进行 few-shot adaptation？
```

这使问题从“参数有没有心理学意义”转为“模型是否具有可泛化预测能力”。

---

# 2. 问题二：用户澄清自己不做儿童阅读障碍，更希望偏算法和计算机

## 用户问题

你指出：

> 我没有儿童阅读障碍的背景，那只是兼职项目。我更想偏向算法和计算机，而不是心理学。

## 回答整理

这个澄清很重要。研究定位不应该靠“儿童阅读障碍/临床评估”支撑，而应该改成：

> 以人类多任务行为数据为测试场，研究通用序列建模、元学习、表征学习和泛化问题。

### 2.1 心理学版本和算法版本的区别

不建议主打：

```text
用模型拟合人类行为，从而理解人类认知。
```

更适合你的版本：

```text
研究大规模序列模型如何在异质、多任务、trial-level 行为序列中进行跨任务泛化、少样本适应和个体化预测。
```

### 2.2 认知任务可以作为 benchmark，而不是最终目的

类似：

```text
ImageNet → 视觉表征学习 benchmark
Atari/MuJoCo → 强化学习 benchmark
GLUE/MMLU → 语言和推理 benchmark
```

你可以把 Psych-101、Choice13K、多任务认知数据看作：

```text
human behavioral sequence modeling benchmark
```

它的特点是：

1. 任务种类多。
2. 每个任务 trial 数有限。
3. 个体差异明显。
4. 数据噪声大。
5. 历史依赖和策略变化存在。
6. 任务结构有时需要从文本或行为历史中推断。
7. 模型需要跨任务泛化。

### 2.3 更合适的方向名称

相比：

```text
computational cognitive modeling
```

更适合你的说法包括：

```text
behavioral sequence modeling
multi-task behavioral prediction
meta-learning from human behavioral sequences
representation learning for behavioral sequences
task-general behavioral prediction
```

---

# 3. 问题三：人类实验数据对机器智能和人类智能到底有什么帮助？

## 用户问题

你进一步提出：

> 我自己用人类实验数据做模型，对于理解机器智能和人类智能有什么帮助？  
> 对机器智能来说，它比不上真正的视觉、强化学习、语言模型主赛道；  
> 对人类智能来说，又很难实现真正突破。  
> 我不想一辈子研究单个过程。

## 回答整理

这个担心是合理的。如果项目只是：

```text
拿一批人类实验数据
→ 训练模型拟合 choice/RT
→ 说帮助理解人类认知
```

那确实会很弱。

真正应该转向的问题是：

> 人类行为序列能否成为研究机器学习模型“任务推断能力”和“上下文适应能力”的测试场？

### 3.1 人类实验数据的价值不在规模，而在结构

它肯定无法在规模上和语言、视觉、RL 主赛道相比。

但它有特殊结构：

```text
许多小任务
每个任务样本有限
trial-level 序列短
行为有噪声
规则常常隐含
反馈历史可能有用
同一个任务不同人策略不同
同一个人跨任务可能有稳定但弱的行为模式
```

这对应一个算法问题：

> 如何从大量小任务中学习可迁移的任务推断能力？

### 3.2 对机器智能的意义

它可以测试模型是否具有：

```text
in-context learning
few-shot task adaptation
latent task inference
history usage
task abstraction
cross-task generalization
```

这些能力不一定能通过普通 supervised learning benchmark 清楚测试。

### 3.3 对人类智能的意义

不必研究单个过程，而是研究：

```text
跨任务结构
行为空间结构
任务之间的可迁移计算结构
个体在多任务中的稳定行为表征
```

也就是从：

```text
single-process modeling
```

转向：

```text
task-general behavioral computation
```

---

# 4. 问题四：这个方向可以算 meta-learning、sequence learning、ICL、representation learning、mechanistic interpretation、world model，但真正贡献在哪里？

## 用户问题

你指出：

> 我做的这个东西可以算 meta-learning、sequence learning、in-context learning、representation learning、mechanistic interpretation、world model，但很难说真跑了这么一个模型对这些领域有什么贡献。可能是论文看得少，也可能确实没有什么算法创新。

## 回答整理

关键不是“能归入哪个领域”，而是：

```text
这些领域已经知道什么？
还不知道什么？
我的数据和模型正好能回答哪一个不知道的问题？
```

### 4.1 不建议主打 world model

world model 在当前 AI 语境里多指：

```text
agent 学习环境动力学
预测未来状态
支持 rollout / planning
model-based RL
embodied AI / robotics
```

如果你的模型只是预测 human next response，而不建模 action-conditioned environment dynamics 和 planning，强行叫 world model 容易被质疑。

### 4.2 更合适的主线

建议主线是：

```text
in-context task inference from behavioral history
```

核心问题：

> 序列模型能否从有限的人类行为历史中推断潜在任务结构和个体策略结构？

这条线可以自然连接：

```text
meta-learning
in-context learning
sequence modeling
representation learning
behavioral modeling
```

但不会显得贴标签太散。

### 4.3 最小可发表核心问题

可以压成一句：

> When predicting human trial-level behavior, how much do large sequence models actually use behavioral history?

对应实验：

```text
no history
short history
long history
shuffled history
wrong participant history
wrong task history
feedback removed
RT removed
task text removed
```

目标不是只看模型是否强，而是看它的性能来自哪里。

---

# 5. 问题五：提出一套评价标准真的算贡献吗？现在没人做 benchmark 吗？

## 用户问题

你问：

> 你说提出一套评价标准，但这些不都是常见 eval 吗？把它们放到一起就算贡献了吗？现在没人做这种 benchmark 吗？

## 回答整理

单纯把常见 eval 拼在一起，不是强贡献。

例如只是说：

```text
held-out trial
held-out participant
held-out task
no-history
shuffled-history
```

这些本身确实常见。

真正可能成为贡献的是：

> 针对 behavioral foundation model 的能力来源诊断框架。

也就是说，不只是测分数，而是分解模型预测能力来自哪里：

```text
task text prior
behavioral history
temporal order
participant-specific signal
feedback learning
RT dynamics
dataset shortcut
```

### 5.1 已有相关工作，不能声称没人做

本轮讨论中查到：

1. Centaur 已经将 Llama 3.1 70B 微调为能够预测自然语言表达实验中的人类行为的模型。
2. Be.FM 也提出了 open foundation models for human behavior，并构建了 benchmark tasks。
3. Human Behavior Atlas 也提供了跨心理和社会行为任务的框架。

因此不能说“没人做 behavioral benchmark”。

更稳的说法是：

> 现有工作已经开始构建 behavioral foundation models 和 benchmark，但对模型预测能力来源的系统诊断还不够统一、可复用和跨模型。

### 5.2 可以主打的贡献

不是：

```text
我提出第一个 behavioral benchmark。
```

而是：

```text
我提出一个 diagnostic evaluation framework，用于分解 behavioral model 的性能来源。
```

例如定义：

```text
History Gain = Score(text + true history) - Score(text only)
Order Sensitivity = Score(true history) - Score(shuffled history)
Participant Specificity = Score(true participant history) - Score(wrong participant history)
Task Prior Dependence = Score(text only) - Score(history only)
RT Utility = Score(choice+RT input) - Score(choice-only input)
```

---

# 6. 问题六：普通模型从头训练会不会完全比不过 Centaur/Be.FM 的大模型微调？

## 用户问题

你问：

> Centaur 和 Be.FM 都是在大模型基础上微调的。大模型算力是很大问题。我如果拿普通模型直接用数据训练，会不会完全比不了大模型微调？

## 回答整理

在 overall prediction accuracy 上，大概率比不过大模型微调。

原因：

```text
LLM 已经有语言理解能力
有任务描述理解能力
有指令跟随能力
有大量预训练语义先验
```

如果你的模型是 GRU、普通 Transformer、Decision Transformer-style model，从头训练，它没有这些语言先验，特别是在任务文本很重要时会处于劣势。

但这不代表普通模型没价值。

### 6.1 普通模型的价值：可控对照

普通模型可以回答：

```text
不依赖语言预训练，仅靠结构化 trial-level 行为历史，模型能学到多少？
```

它适合做：

```text
structured sequence baseline
controlled diagnostic model
history-only model
participant/task embedding model
```

### 6.2 建议两条模型线并行

#### Line A：LLM / Centaur-like line

目标：

```text
对齐现有 SOTA 方向
复现或接近 Centaur 的输入输出形式
分析 LLM 行为预测能力来源
```

模型例子：

```text
Qwen2.5-0.5B/1.5B/7B + LoRA
Llama small + LoRA
Centaur-like prompting
```

#### Line B：structured sequence model line

目标：

```text
不靠语言先验，研究行为序列本身的信息量
```

模型例子：

```text
GRU
LSTM
small Transformer
task embedding + history encoder
participant embedding + history encoder
choice + RT joint head
```

这两条线放在一起的价值是：

> LLM 很强，但需要 structured baselines 才知道它强在哪里。

---

# 7. 问题七：Centaur 现在效果很好，最像人类模型，那还需要做什么？

## 用户问题

你说：

> 我同意先做模型诊断。我现在主要从 Centaur 文章出发，看起来确实是最像人类的模型了，效果很好。

## 回答整理

Centaur 效果好，反而说明它值得被诊断。

需要问：

```text
它为什么好？
它靠什么信息好？
它在哪些任务上好？
它在哪些条件下失效？
它是像 group-level distribution，还是像 individual-level adaptation？
它是像 final choice，还是像 learning curve / RT / error pattern？
```

### 7.1 Centaur 效果好可能有多种原因

```text
1. 它真的学到了人类行为规律。
2. 它靠任务文本理解了实验规则。
3. 它学到了每个任务的平均反应分布。
4. 它利用了 choice history 的局部统计。
5. 它记住了训练数据任务分布。
6. 某些任务简单 baseline 本来就能预测。
7. 它对 seen task family 强，但 OOD task family 弱。
```

### 7.2 诊断实验建议

核心实验：

```text
text-only vs history-only vs text+history
true history vs shuffled history
true history vs wrong participant history
history length scaling
held-out task family
choice-only vs choice+RT
```

---

# 8. 问题八：学校 HPC 是 4×V100 SXM2，能不能部署 Centaur-70B？

## 用户问题

你提供文档信息：

> Each GPU node provided as part of the gpu partition feature 4x Nvidia V100 SXM2 (with either 16G or 32G memory) interconnected by the NVLink 2.0 architecture.

你问能不能继续做。

## 回答整理

可以继续做，但不要把项目成功条件设成：

```text
必须部署 Centaur-70B
```

### 8.1 V100 配置含义

文档说明每个 GPU node 有：

```text
4 × Nvidia V100 SXM2
每张 16GB 或 32GB
NVLink 2.0
```

如果是 4×16GB：

```text
总显存 64GB
70B 4-bit 推理都比较紧
FP16/训练基本不现实
```

如果是 4×32GB：

```text
总显存 128GB
70B 4-bit/8-bit 推理有希望
训练仍然困难
```

### 8.2 当前更现实路线

不是先硬上 70B，而是：

```text
1. 确认 V100 是 16GB 还是 32GB。
2. 建立 Psych-101 / 其他数据的统一 evaluation format。
3. 跑简单 baseline。
4. 跑 Qwen2.5-0.5B/1.5B LoRA。
5. 做 no-history / shuffled-history / text-only 诊断。
6. 再决定是否需要 70B。
```

### 8.3 70B 可以作为外部参照，不是起点

你可以先用：

```text
count baseline
history count baseline
task-specific cognitive models
GRU / Transformer
small LLM LoRA
```

等 pipeline 稳定后，再考虑少量 70B inference 或借助合作资源。

---

# 9. 问题九：你已经在复现 Centaur 里的简单认知模型，这一步是否有价值？

## 用户问题

你说：

> 我已经在复现 Centaur 里面简单的认知模型。对于强化学习任务用传统强化学习模型；对于类似 n-back 的任务，直接算训练集 response 条件分布。

## 回答整理

这一步非常有价值。它是 behavioral model diagnostic suite 的基础层。

因为没有这些 baseline，就无法判断 Centaur 到底强在哪里。

例如：

```text
Centaur NLL = 0.45
count baseline NLL = 0.46
```

和：

```text
Centaur NLL = 0.45
count baseline NLL = 0.80
```

意义完全不同。

### 9.1 baseline 分层建议

#### Level 0：非序列统计 baseline

```text
majority response
task-level response frequency
condition-level response frequency
ground_truth → response count baseline
stimulus-feature → response count baseline
```

#### Level 1：任务特异 cognitive model

```text
RL task → Rescorla-Wagner / Q-learning
delay discounting → hyperbolic discounting
risk choice → prospect theory / expected utility
category learning → rule-based / exemplar model
memory task → SDT / condition response model
```

#### Level 2：通用序列模型

```text
GRU
LSTM
causal Transformer
Decision Transformer-style model
small LLM LoRA
```

---

# 10. 问题十：go/no-go、n-back 也有对应认知模型，一般都是 RNN 吗？

## 用户问题

你说：

> 我前几周整理时发现，go/no-go、n-back 这样的任务也是有对应认知模型的，一般都是 RNN。

## 回答整理

需要稍微纠正：

> RNN 可以作为 cognitive model / neural sequence model，但不是 go/no-go、n-back 的传统认知模型的唯一或主要形式。

### 10.1 n-back 可用模型

n-back 的核心是：

```text
维持过去 n 个刺激
比较当前刺激和 n-back 刺激
决定 target / non-target
```

可用模型包括：

```text
signal detection model
working-memory capacity model
lapse/bias model
DDM / evidence accumulation model
RNN / GRU
condition count baseline
history count baseline
```

如果只有 choice，没有 RT 分布，DDM 的优势会变弱。

### 10.2 go/no-go 可用模型

go/no-go 可以用：

```text
signal detection theory
response inhibition model
DDM / race model
lapse/bias model
RL + inhibition model
RNN / GRU
```

如果有 RT，DDM/race model 更有价值；如果只有按/不按，SDT + response bias + lapse 就是合理 baseline。

### 10.3 Rescorla-Wagner 和 DDM 的关系

不是：

```text
RL task : Rescorla-Wagner
n-back : DDM
```

更准确是：

```text
Rescorla-Wagner/Q-learning 和 DDM/SDT 都是 task-specific cognitive models，
只是适用于不同任务族。
```

---

# 11. 问题十一：Centaur 有哪些批评文章？浙大的文章是哪篇？

## 用户问题

你说：

> Centaur 现在的评价标准只有训练集上的 NLL，没有你说的删掉 history、改 history 这些。哦对，有一个浙大的文章就是对 Centaur 提出了这个批评，是哪个你找一下。

## 回答整理

查到两类相关批评。

### 11.1 浙大相关：instruction understanding 批评

题目：

```text
Can Centaur truly simulate human cognition? The fundamental limitation of instruction understanding
```

作者：

```text
Wei Liu, Nai Ding
Zhejiang University
```

核心观点：

```text
Centaur 可能并没有真正理解任务指令，而是依赖 fine-tuning 数据中的表层统计线索。
```

他们通过修改指令或构造指令理解测试，质疑 Centaur 是否真正理解当前任务指令。

### 11.2 更接近 history/task ablation 的批评

题目：

```text
Centaur May Have Learned a Shortcut that Explains Away Psychological Tasks
```

作者：

```text
Hanbo Xie, Jian-Qiao Zhu
```

核心观点：

```text
Centaur 的优势可能来自 shortcut。
通过隔离 task information 和 choice history，发现 Centaur 在没有心理任务信息时仍能超过 domain-specific cognitive models；
但在某些任务中移除 choice history 后表现变差。
```

这篇更接近你关心的：

```text
history ablation
task information ablation
shortcut diagnosis
```

### 11.3 对你项目的影响

不能再说：

```text
没人做 Centaur 的 ablation / criticism。
```

更稳的定位是：

> 现有批评已经指出 Centaur 可能依赖 shortcut、history dependency 或 instruction-insensitive behavior；你的潜在贡献是把这些零散批评系统化，做成可复用的 behavioral model diagnostic evaluation framework。

---

# 12. 问题十二：condition count baseline 和 history count baseline 有什么区别？

## 用户问题

你问：

> Level 0: condition count baseline  
> Level 1: history count baseline  
> 这两个有什么区别？

## 回答整理

### 12.1 condition count baseline

只看当前 trial 的条件，不看历史。

公式：

```text
P(response_t | condition_t)
```

例子：

n-back：

```text
P(response | target)
P(response | target, lure_type)
```

go/no-go：

```text
P(response | go_or_nogo)
```

flanker：

```text
P(response | congruent_or_incongruent, correct_response)
```

它回答：

> 当前 trial 条件本身能解释多少行为？

### 12.2 history count baseline

在当前条件之外，再看短程历史信息。

公式：

```text
P(response_t | condition_t, history_features_t)
```

常见形式：

```text
P(response_t | condition_t, response_{t-1})
P(response_t | condition_t, correct_{t-1})
P(response_t | condition_t, feedback_{t-1})
P(response_t | condition_t, response_{t-1}, feedback_{t-1})
```

它回答：

> 历史信息是否提供额外预测力？

### 12.3 两者区别一句话

```text
condition count:
只看当前 trial 是什么条件。

history count:
看当前 trial 条件 + 最近历史。
```

---

# 13. 问题十三：history model 怎么拟合？它是不是就变成 RNN 了？

## 用户问题

你问：

> 你说的 history model 是怎么拟合的？conditional model 我复现了，就是个查找表。history model 难道还要用隐空间学习过去信息吗？那不是和 RNN 一样了？

## 回答整理

这里的 history count baseline 不是 RNN，也不学习隐空间。

它仍然是查找表，只是 key 多加了历史特征。

### 13.1 condition count 的查找表

```text
P(y_t | x_t)
```

例如：

```text
target      → press: 0.82, no_press: 0.18
non-target  → press: 0.12, no_press: 0.88
```

### 13.2 history count 的查找表

```text
P(y_t | x_t, y_{t-1})
```

例如：

```text
target, previous_press      → press: 0.86, no_press: 0.14
target, previous_no_press   → press: 0.78, no_press: 0.22

non-target, previous_press     → press: 0.18, no_press: 0.82
non-target, previous_no_press  → press: 0.08, no_press: 0.92
```

它没有 hidden state，没有反向传播，没有神经网络。

它类似：

```text
n-gram model
finite-order Markov empirical baseline
```

### 13.3 拟合方式

训练集统计：

```text
count(condition_t, previous_response, response_t)
```

然后归一化：

```text
P(response_t | condition_t, previous_response)
=
count(condition_t, previous_response, response_t)
/
sum_response count(condition_t, previous_response, response)
```

加 smoothing：

```text
P(y | key)
=
(count(key, y) + α)
/
(sum_y count(key, y) + α * num_choices)
```

### 13.4 backoff

如果 key 太细导致稀疏，可以逐级回退：

```text
P(y_t | condition_t, response_{t-1}, feedback_{t-1})
↓
P(y_t | condition_t, response_{t-1})
↓
P(y_t | condition_t)
↓
P(y_t)
```

这个模型的价值不是替代 RNN，而是判断：

> 简单短程历史依赖能解释多少？

如果 RNN/Transformer 超不过 history count，说明高级序列模型没有学到额外序列结构。

---

# 14. 问题十四：统一认知模型很难，因为一个任务对应多个模型，一个模型也需要任务特定处理

## 用户问题

你指出：

> 对于统一认知模型，包含多种认知任务时，十分困难。  
> 一个任务对应多个模型；同一个模型针对不同任务的数据也需要特定处理。  
> 处理起来很麻烦。  
> 导师让我先做一个 data contract，方便之后使用。

## 回答整理

这个判断非常重要。真正困难的不只是训练模型，而是任务接口。

### 14.1 为什么 data contract 是必要的？

没有 data contract，就会出现：

```text
每个任务写一套预处理
每个模型写一套输入
每次评价都不可复现
最后无法判断模型差异来自算法还是数据处理
```

Data contract 应该规定：

```text
任何任务必须转换成统一字段；
任何模型只能通过这套字段读取数据；
任何评价基于同一 target 和 split。
```

### 14.2 data contract 的三层结构

#### Layer 1：基础 trial table

所有任务都应有：

```text
dataset_id
task_id
experiment_id
participant_id
session_id
trial_index
stimulus
condition
choice
ground_truth
correct
reward
feedback
rt
timestamp
```

没有的字段可以是 null / NaN。

#### Layer 2：structured condition features

例如：

```json
{
  "trial_type": "target",
  "correct_response": 1,
  "difficulty": "high",
  "congruency": "incongruent",
  "n_back": 2,
  "reward_left": 10,
  "reward_right": 5,
  "delay_left": 0,
  "delay_right": 30
}
```

这是给：

```text
count baseline
history count
traditional cognitive models
RNN
structured Transformer
```

使用的。

#### Layer 3：text view

同一个 trial 生成文本：

```text
task instruction
history event text
current trial text
choice text
feedback text
```

这是给：

```text
Centaur-style LLM
LLM prompting/fine-tuning
text-conditioned behavioral model
```

使用的。

关键点：

> text view 应该由 data contract 自动生成，而不是每个模型自己随便写 prompt。

---

# 15. 问题十五：表格数据模型和文本数据模型怎么统一评估？Centaur 的比较是否公平？

## 用户问题

你指出：

> 这些认知模型都是用表格数据的，Centaur 中的比较并不公平。  
> 你上面说的比较也没法在数据一会是表格数据、一会是文本数据时统一评估。

## 回答整理

这个判断是对的。

Centaur 的输入是：

```text
task instruction + trial history + current trial
```

传统认知模型输入通常是：

```text
stimulus features
choice
reward
feedback
rt
```

两者输入空间不同。

因此不能简单说：

```text
Centaur NLL < cognitive model NLL
所以 Centaur 是更好的认知模型。
```

更严谨说法：

```text
Centaur 在 text-conditioned behavioral prediction setting 下更强；
传统认知模型在 structured-variable setting 下比较；
两者输入模态不同，直接比较会混淆模型能力和输入表示能力。
```

### 15.1 解决方案：分 track 评估

#### Track A：Structured-only track

输入：

```text
condition_features
choice history
feedback history
rt history
participant_id
task_id
```

模型：

```text
condition count
history count
logistic regression
cognitive model
RNN
Transformer
```

问题：

```text
没有语言先验时，结构化行为历史能提供多少预测力？
```

#### Track B：Text-conditioned track

输入：

```text
task instruction
trial history as text
current trial as text
```

模型：

```text
LLM prompting
LLM fine-tuning
Centaur-style model
```

问题：

```text
语言预训练和任务描述给行为预测带来多少帮助？
```

#### Track C：Hybrid track

输入：

```text
structured features + text description
```

模型：

```text
LLM + structured tokens
Transformer with task text embedding
hybrid task encoder
```

问题：

```text
语言任务先验和结构化行为历史能否互补？
```

### 15.2 统一的不是输入，而是目标

不同模型不一定吃同一种输入，但必须统一：

```text
same test trial
same prediction target
same train/test split
same metric
```

例如同一个测试样本：

```text
participant_id = 001
task_id = nback
trial = 37
true response = press
```

所有模型都输出：

```text
P(response = press)
```

不同模型可以属于不同 input condition。

---

# 16. 建议定义 input condition，而不是只定义 model

为了避免“Centaur 吃文本，认知模型吃表格”的混乱，建议在框架中明确规定：

```text
InputCondition
```

例如：

```text
C0: current condition only
C1: current condition + local history
C2: full structured history
C3: task instruction only
C4: task instruction + text history
C5: task instruction + structured history
C6: participant history across tasks
```

模型结果必须标注：

```text
condition count under C0
history count under C1
RNN under C2
LLM under C4
hybrid model under C5
```

这样比较就不会伪装成完全公平，而是诚实地说明不同模型使用了不同信息条件。

---

# 17. 当前可形成的核心研究动机

可以总结为：

> 当前 cognitive models 和 LLM-based behavioral models 的比较难以解释，原因有三点。  
> 第一，认知任务和认知模型不是一一对应关系；一个任务可以有多个模型，一个模型也常常需要任务特异预处理。  
> 第二，不同模型使用不同输入模态：传统认知模型使用结构化表格变量，LLM 模型使用自然语言任务描述和 trial history，因此直接比较 NLL 会混淆模型能力与输入表示差异。  
> 第三，需要简单但系统的 empirical baselines，例如 condition count 和 history count，来判断高级模型是否真的利用了有意义的序列结构，还是只是捕捉浅层 response distribution。

因此可以提出：

> 定义一个 behavioral modeling data contract，同时支持 structured view 和 text view；在统一 prediction target、split 和 metric 下，系统比较统计 baseline、任务特异认知模型、神经序列模型和 LLM-based behavioral models，并诊断模型性能来自 condition、history、task text、participant signal、RT 和 feedback 的哪一部分。

---

# 18. 可能的项目标题

以下标题更偏算法/计算机，而不是传统心理学：

```text
A Data Contract and Diagnostic Evaluation Framework for Behavioral Sequence Models
```

```text
Diagnosing Information Sources in Foundation Models of Human Behavior
```

```text
Structured and Text-Conditioned Evaluation of Human Behavioral Sequence Models
```

```text
From Cognitive Models to Behavioral Foundation Models: A Unified Data Contract and Diagnostic Benchmark
```

```text
Do Behavioral Foundation Models Use History? A Diagnostic Framework for Trial-Level Human Behavior Prediction
```

---

# 19. 后续工作建议

## 19.1 短期：实现 data contract MVP

优先字段：

```text
dataset_id
task_id
participant_id
trial_index
condition_features
choice
ground_truth
correct
reward
feedback
rt
text_view.instruction
text_view.current_trial
text_view.history_event
```

目标：

```text
同一个 trial 同时能生成 structured input 和 text input。
```

## 19.2 短期：完成三类 baseline

### Baseline 1：condition count

```text
P(y_t | condition_t)
```

### Baseline 2：history count

```text
P(y_t | condition_t, y_{t-1})
P(y_t | condition_t, y_{t-1}, feedback_{t-1})
```

### Baseline 3：task-specific cognitive model

```text
RL task → Rescorla-Wagner / Q-learning
n-back/go-no-go → SDT / bias-lapse / DDM if RT available
delay/risk → discounting / prospect theory
```

## 19.3 中期：加 structured sequence model

```text
GRU
LSTM
small causal Transformer
task embedding + history encoder
participant embedding + history encoder
```

重点不是追求 SOTA，而是作为 controlled baseline。

## 19.4 中期：加 small LLM LoRA

```text
Qwen2.5-0.5B
Qwen2.5-1.5B
Qwen2.5-7B if compute allows
```

对应 text-conditioned track。

## 19.5 中期：核心诊断实验

优先做：

```text
full history vs no history
true history vs shuffled history
text-only vs text+history
structured-only vs text-conditioned
choice-only vs choice+RT
held-out participant
held-out task / task family
```

## 19.6 长期：形成可复用 evaluation suite

最终目标不是一个模型，而是一个框架：

```text
data contract
model interface
input condition definition
baseline zoo
diagnostic metrics
task-level report
cross-task summary
```

---

# 20. 下周汇报可以这样讲

## 20.1 背景

Centaur/Be.FM 说明大模型可以在 trial-level human behavior prediction 上取得很强表现，但这类模型与传统 cognitive models 的比较存在输入模态和任务接口不一致的问题。

## 20.2 当前问题

1. 认知任务和认知模型不是一一对应。
2. 传统模型用 structured tabular variables，LLM 模型用 text prompt。
3. 直接比较 NLL 不足以说明模型是否真的学到认知机制或行为历史。
4. 现有批评指出 Centaur 可能依赖 shortcut、instruction-insensitive cues 或 choice history。
5. 因此需要一个统一 data contract 和 diagnostic evaluation framework。

## 20.3 我的工作

1. 复现 Centaur 中的简单认知模型 baseline。
2. 对 RL 任务实现 Rescorla-Wagner/Q-learning 类模型。
3. 对 n-back/go-no-go 等任务实现 condition response distribution baseline。
4. 开始设计 data contract，使同一 trial 可以生成 structured view 和 text view。

## 20.4 下一步

1. 明确 data contract 字段。
2. 实现 condition count 和 history count。
3. 定义 input condition：C0-C6。
4. 跑 structured-only track 的 baseline。
5. 再跑 small LLM text-conditioned track。
6. 用 no-history、shuffled-history、text-only 等实验诊断模型能力来源。

---

# 21. 最终总结

本轮讨论后，项目可以从“用模型拟合人类行为”重新定位为：

> 构建一个统一的 behavioral sequence modeling data contract 和 diagnostic evaluation framework，用于比较统计 baseline、任务特异认知模型、神经序列模型和 LLM-based behavioral foundation models，并分解它们在 trial-level human behavior prediction 中依赖的信息来源。

关键不是一开始提出全新算法，也不是硬部署 Centaur-70B，而是先解决：

```text
数据怎么统一？
输入条件怎么定义？
表格模型和文本模型怎么公平/诚实比较？
简单 baseline 已经能解释多少？
高级模型到底利用了 condition、history、text、participant、feedback、RT 中的哪些信息？
```

如果这个框架做出来，它本身就可以成为后续模型创新和论文工作的基础。



# 第一步：把 data contract 服务于 episodic ICL

不要只做 table。

你要让它能自动生成：

support set
query set
episode metadata
task family split
participant split
text view
structured view
history ablation view

这就是 ML 需要的 dataset interface。

第二步：先跑一个最小 ICL 实验

选 5–10 个任务族。

每个 episode：

support length k = 0, 1, 2, 4, 8, 16, 32
query = next trial

跑：

condition count
history count
GRU
small Transformer
Qwen-0.5B LoRA

看：

performance vs k

如果随着 k 增加，模型在 held-out participant / held-out task 上明显变好，这就是 context adaptation。

第三步：做 OOD task-family split

这个最关键。

train task families:
RL / risk / go-no-go / flanker / memory

test task family:
category learning / delay discounting / n-back

换几组。

问：

模型能不能在没见过的任务族上通过 support trials 适应？

这比普通 train/test split 更像 meta-learning。

第四步：比较 structured-only 和 text-conditioned

这一步连接 Centaur。

structured-only:
condition_features + history

text-conditioned:
task instruction + text history

hybrid:
task instruction + structured history

问：

语言先验是否提高 OOD ICL？
还是只是提高 seen-task prior？

这个问题很有意思，因为 Centaur 和 Be.FM 都是 LLM-based behavioral model；Centaur 明确是用 Llama-3.1 70B 在 Psych-101 上微调来预测自然语言可表达实验中的人类行为，Be.FM 也是基于开源 LLM 并构建 behavioral foundation model benchmark。

