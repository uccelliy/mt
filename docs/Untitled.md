可以。这篇 Saxe, Nelli & Summerfield (2021) 如果按照作者自己在摘要里说的目标——**界定概念与方法上的挑战，并由此提出新的 neuroscience research questions**——来读，它其实不是一篇简单讨论“神经网络像不像大脑”的综述，而是在问：

> **如果我们真的把 deep network 当作 brain theory，那么神经科学的研究问题、证据标准和“解释”的含义应该怎样改变？**

作者明确说，他们的目标是为 deep-learning-era 的系统神经科学提供一个 roadmap，重点讨论如何比较**行为、学习动态和神经表征**，并提出由机器学习进展催生的新问题。

---

# 一、文章最核心的概念转变

传统神经科学模型通常是：

\[ \text{researcher proposes mechanism} \rightarrow \text{model} \rightarrow \text{prediction} \]

研究者先假定：

- 某个脑区干什么；
- 某类神经元编码什么；
- 某个 circuit 如何计算。

而 deep learning framework 提出的是：

\[ \text{architecture + learning rule + objective + experience} \rightarrow \text{emergent computation} \]

也就是说，研究者不再直接指定内部 computation，而是指定：

- architecture；
- learning rule；
- objective / cost function；
- training environment；

然后让 representation 和 computation **自己通过学习出现**。这正是 deep learning 对 neuroscience 最根本的吸引力：它第一次让人可以研究复杂 neural computation 如何从 experience 中 **de novo emerge**。

但问题马上就来了：

> 如果机制不是研究者写进去的，而是 optimization 自己产生的，我们究竟还在“解释”什么？

这就是整篇文章真正的中心矛盾。

---

# 二、概念挑战 1：拟合大脑 ≠ 对大脑的理论

这是作者首先要拆掉的一个误区。

现在常见做法是：

\[ X_{\text{DNN}} \xrightarrow{W} X_{\text{brain}} \]

训练一个线性 mapping，发现 DNN activity 可以解释相当比例的 neural variance。

但作者指出：

**这并不足以说明 DNN 和 brain 使用了相同 computation。**

原因包括：

- 分类性能差不少的网络，brain prediction 可能只差一点；
- 随机权重网络和训练网络之间的 brain fit 差距有时并不大；
- 一个灵活的线性 mapping 可以把本来结构差异很大的两个 representation 对齐。

于是文章提出的第一个方法论升级是：

### 不要只问

> 哪个模型的 representation 和 brain correlation 最大？

而应该问：

> **模型和大脑到底在哪种意义上具有相同的 computation？**

这要求更加严格的 evidence。

---

# 三、方法挑战 1：从 representation correlation 转向更强的比较

作者实际上提出了一条 evidence hierarchy。

## Level 1：线性 mapping

\[ X_{\text{NN}}W\approx X_{\text{brain}} \]

只能说明 information spaces 可以被映射。

---

## Level 2：restrict mapping

比如 RSA。

不允许 arbitrarily flexible mapping，而比较：

\[ D_{\text{NN}}(i,j) \]

和

\[ D_{\text{brain}}(i,j) \]

也就是 stimuli 之间的 representational geometry。

但作者仍然警告：

> 即使 RSA 很像，也可能只是因为人脑和网络都知道“长得像的东西比较像”。

因此仍然可能是 superficial resemblance。

---

## Level 3：causal / closed-loop test

这是文章认为更强的方向。

例如：

1. 用 DNN 预测某个 biological neuron；
2. 根据 DNN 生成一个应该最大化这个 neuron activity 的新 stimulus；
3. 真正给动物看；
4. 看 neuron 是否真的被强烈激活。

也就是：

\[ \text{model} \rightarrow \text{novel prediction} \rightarrow \text{biological intervention/test} \]

而不是：

\[ \text{existing brain data} \leftrightarrow \text{existing NN data} \]

作者明确主张，想要超越 correlation，就需要这种 causal assay 和 closed-loop experiment。

这一点和你刚才问的 SAE 问题其实很接近：**模型内部分析本身还不是最终证据，真正强的是它能否产生可干预、可证伪的预测。**

---

# 四、概念挑战 2：不能只比较最终状态，要比较“怎么学出来的”

这可能是全文最重要的观点之一。

假设最终：

\[ R_{\text{DNN}}\approx R_{\text{brain}} \]

传统做法可能就觉得模型成功了。

但作者问：

> 如果最后 representation 一样，但是学习过程完全不一样呢？

例如两个系统最终都形成 face representation：

\[ R_{\text{final}}^{A}\approx R_{\text{final}}^{B} \]

但是：

\[ R_A(t)\neq R_B(t) \]

那么它们是否真的实现了同一种 learning mechanism？

未必。

所以文章反复强调：

\[ \boxed{ \text{learning trajectory} > \text{terminal representation alone} } \]

未来应该比较：

\[ R_{\text{brain}}(t) \]

和

\[ R_{\text{model}}(t) \]

而不是只在训练结束以后做 RSA。作者明确把“比较 representation 如何随 learning 改变”作为判断 brain 和 DNN 是否真正以相似方式学习的重要方向。

---

# 五、新研究问题 1：大脑到底采用什么 learning rule？

Deep learning 把一个传统问题重新变成了可实验检验的问题。

以前问：

> perception 是 innate 还是 learned？

现在可以问得更加具体：

> **什么 learning objective 可以从相对 generic 的 architecture 和 experience 中产生 biological representation？**

候选包括：

- supervised gradient descent；
- Hebbian learning；
- predictive / self-supervised learning；
- reward/value prediction；
- combinations of these。

关键不再只是：

> “哪个 trained model 最像 V1？”

而是不同 learning rule 会预测**不同的学习动态**。

例如文中的 perceptual-learning 例子：

- gradient descent 可能预测 informative neuron 改变最多；
- correlational Hebbian learning 可能预测 active neuron 改变最多；
- predictive coding / contrastive Hebbian learning 可能预测不同 layer 上不同 timing 的变化。

于是：

\[ \text{learning rule} \rightarrow \text{distinct temporal prediction} \]

再拿 neuroimaging / electrophysiology 去区分。

作者认为，通过不断加入实验约束，可以逐渐缩小可能的 learning-rule space。

所以这里真正的新问题不是：

> “大脑是不是用了 backprop？”

而是：

> **哪些 learning principles 能同时解释行为、representational change 和 neural dynamics？**

---

# 六、新研究问题 2：复杂 cognitive architecture 为什么会出现？

视觉分类只是容易开始的地方。

真正困难的问题是：

> 如果 cognition 真的是 optimization 的结果，那么为什么我们的大脑会出现 memory、attention、planning、reasoning 等相对 specialized systems？

作者特别列出了一组问题：

- 人如何形成脱离具体物理特征的 abstract representation？
- 如何形成 tree、ring、grid 等 relational structure？
- 如何把已有行为 component 组合成新行为？
- 如何快速获得并 generalize 新记忆？

注意这个问题的方向发生了变化。

传统 neuroscience：

> hippocampus 是干什么的？

deep-learning-inspired neuroscience：

> **什么 computational pressure 会导致一个类似 hippocampus 的 subsystem 出现？**

也就是：

\[ \text{environmental/learning pressure} \rightarrow \text{architectural specialization} \]

而不是一开始就把 specialization 写进模型。

---

# 七、新研究问题 3：用 human–NN failure difference 反过来研究人类 cognition

作者提出一个非常实用的方法：

> **专门寻找人和网络表现 qualitatively different 的任务。**

例如：

\[ \text{human succeeds} ,\qquad \text{NN fails} \]

然后研究：

1. 人类用了什么 mechanism？
2. 当前 NN 缺什么？
3. 加什么 architecture / learning rule 后 NN 才获得 human-like behaviour？

这实际上把 AI limitation 转换成了 neuroscience hypothesis generator。

因此 deep learning 的价值不只是：

> “模拟脑。”

还包括：

> **通过模型失败来暴露我们需要解释的 biological computation。**

---

# 八、新研究问题 4：为什么人类能够 abstraction 和 systematic generalization？

作者特别强调 human–machine 差异。

神经网络通常：

\[ \text{lots of training} \rightarrow \text{good interpolation} \]

而人类可以：

\[ \text{limited experience} \rightarrow \text{abstract knowledge} \rightarrow \text{transfer} \]

因此新的 neuroscience question 是：

> **brain 如何 encode、compose 和 generalize abstract knowledge？**

但这里又出现了几个非常实际的方法学障碍。

### ① animal model 问题

真正强的 abstraction/generalization 可能是 human-specific 或至少在人类特别突出。

于是：

- rodent / macaque：可以 electrophysiology / optogenetics；
- human：通常只能 fMRI / MEG / EEG。

精细机制和高级 cognition 很难同时获得。

### ② prior 不匹配

人进实验室前已经有几十年的：

\[ P_{\text{human}}(\text{world}) \]

而 NN 的 prior 完全不同。

所以不能简单比较：

\[ \text{same experimental trials} \]

就认为 experience 相同。

### ③ learning timescale 不匹配

网络可能需要百万次 trial，人类几十次就学会。

因此 performance equivalence 也不能自动说明 learning-process equivalence。

---

# 九、新研究问题 5：Continual learning 反过来解释 memory architecture

这是文章里特别有意思的一部分。

神经网络出现 catastrophic forgetting：

\[ A\rightarrow B \]

训练 B 后：

\[ Performance(A)\downarrow \]

但人类通常没有这么严重。

于是一个 AI problem：

> 怎样解决 catastrophic forgetting？

被反过来转换成 neuroscience question：

> **biology 为什么没有严重 catastrophic forgetting？**

这重新赋予 hippocampus–neocortex complementary learning systems 一个 optimization-level explanation：

- hippocampus：快速、稀疏 episodic memory；
- neocortex：慢速 statistical learning；
- replay：把旧 experience 和新 experience interleave。

于是 brain architecture 不只是：

> “这里负责 episodic memory。”

而可以解释成：

> **它可能是 continual-learning problem 的 computational solution。**

---

# 十、新研究问题 6：为什么会有 attention、control 和 capacity limitation？

这部分我觉得特别漂亮。

机器学习告诉我们存在一个 fundamental trade-off：

### Shared representation

优点：

\[ \text{positive transfer}\uparrow \]

但：

\[ \text{interference}\uparrow \]

### separated representation

优点：

\[ \text{interference}\downarrow \]

但：

\[ \text{generalization}\downarrow \]

所以 biological system 必须解决：

\[ \boxed{ \text{maximize transfer} \quad\text{while}\quad \text{minimize interference} } \]

作者讨论的一种解释是：

大脑大量使用 shared representations 来促进 generalization，但同时发展：

- attention；
- task gating；
- cognitive control；

来暂时关闭不相关的 pathways。

因此人类的 control bottleneck，甚至“为什么我们不能同时做很多 cognitive tasks”，可能不是一个纯粹的硬件缺陷，而是：

> **为了在 generalization 和 interference 之间取得 computational trade-off 所产生的结果。**

这就是一个典型的 **deep learning → new neuroscience question**：

> 为什么 cognitive control 有 capacity limit？

不再只是测：

\[ \text{capacity}=4 \]

而是问：

> **为什么 optimization 会产生 capacity limitation？它解决了什么 computational problem？**

---

# 十一、最后一个概念挑战：什么才叫“理解”一个神经网络？

这正好回到你上一条问我的问题。

作者明确承认：

\[ \text{millions of parameters} \rightarrow \text{black-box problem} \]

但他们并不接受：

> “既然 network 很复杂，就不用解释了。”

相反，他们认为要采用 **model idealization**。

也就是不要试图：

\[ \text{interpret every weight of ResNet} \]

而是寻找一个 simplified model：

\[ M_{\text{complex}} \rightarrow M_{\text{idealized}} \rightarrow \text{analytical theory} \]

例如：

- deep linear network；
- infinite-width limit；
- high-dimensional limit；
- teacher–student setting。

在这些特殊条件下，有时可以直接推导：

\[ W(t) \]

也就是整个 learning trajectory，而不只是跑 simulation。

这点特别重要，因为作者对“解释”的理解其实不是：

> 每个 neuron 都起一个名字。

而更接近：

> **找到控制复杂 network behaviour 的低维 mathematical principles，并由它们产生新的 falsifiable prediction。**

---

# 十二、所以这篇文章真正给出的 roadmap 可以浓缩成这样

|旧的研究方式|作者认为的新问题|
|---|---|
|找 DNN–brain correlation|**它们真的实施相同 computation 吗？**|
|比较训练后的 representation|**representation 是以相同 trajectory 学出来的吗？**|
|找最 brain-like architecture|**什么 learning rule / objective / constraint 能产生 biological computation？**|
|看模型能不能做人类任务|**它是否呈现 human-like error、bias、generalization 和 capacity limit？**|
|把 memory/attention 模块写进去|**什么 computational pressure 会使这些 specialized systems 自发出现？**|
|把 human–NN difference 当模型缺陷|**这种差异揭示了什么 biological mechanism？**|
|报告 brain–model association|**模型能否产生新的 causal / closed-loop prediction？**|
|用大型 NN 作为 simulation|**能否把网络 idealize 成一个可理解、可证伪的 theory？**|

---

# 十三、如果只用一句话概括这篇文章

我认为最准确的不是：

> **“Deep neural networks can model the brain.”**

而是：

> **Deep learning 的真正价值，不是给 neuroscience 一个更强的 curve-fitting model，而是把“哪些 computation 是经验和 optimization 的必然产物”变成一组可以被实验检验的问题。**

所以作者最后才明确反对两种极端：

一边是：

\[ \text{DNN fits brain well} \Rightarrow \text{brain theory solved} \]

另一边是：

\[ \text{DNN too complex} \Rightarrow \text{mechanism fundamentally uninterpretable} \]

他们主张的是中间路线：

\[ \boxed{ \text{complex learned model} \rightarrow \text{falsifiable predictions} + \text{model idealization} \rightarrow \text{scientific understanding} } \]

结论部分说得很清楚：不要让越来越复杂的 simulation 超过我们的 conceptual insight，也不要盲目寻找 artificial 和 biological neural codes 的 correspondence；应该让 deep networks 对**行为、learning dynamics 和 neural coding**产生可证伪解释，同时保留 classical neural models 所具有的 interpretability。Saxe et al. - 2021 - If deep learning is the answer, what is the questi.pdfPDF

这也解释了为什么你刚才问“**现在有 SAE 甚至 causal intervention，为什么还说 NN 解释性差？**”时，Saxe 这篇文章的标准其实比“能找到 feature”更高：**他们要求的最终不是 feature interpretability，而是从复杂网络中提炼出能够解释 why、产生新预测、并且能被 biological experiment falsify 的 theory。**