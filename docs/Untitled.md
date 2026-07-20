### 评估轴 + 文献

| 轴                | 指标                          | 关键文献                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | 引用基础       |
| ---------------- | --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| **6 灵活性 / 可证伪性** | 能不能拟合"不可能的被试"               | **Roberts & Pashler (2000), Psych Review**, _How persuasive is a good fit?_  <br>**Pitt & Myung (2002), TiCS**, _When a good fit can be bad_  <br>**Navarro, Pitt & Myung (2004), Cog Psych** — landscaping  <br>**Pitt, Kim, Navarro & Myung (2006), Psych Review** — parameter space partitioning  <br>**Wagenmakers et al. (2004), JMP** — model mimicry / 参数自助法  <br>**Palminteri, Wyart & Koechlin (2017), TiCS** — _The Importance of Falsification in Computational Cognitive Modeling_  <br>_(ML 侧对应)_ Vafa et al. (2025), ICML — inductive bias probing | **★★★ 最硬** |
| **7 参数可恢复性**     | 模拟 → 拟合 → 恢复                | **Wilson & Collins (2019), eLife**, _Ten simple rules for the computational modeling of behavioral data_  <br>Palminteri et al. (2017), TiCS                                                                                                                                                                                                                                                                                                                                                                                                                     | **★★★**    |
| **3 简洁性(MDL)**   | prequential code length(比特) | **Rissanen (1978)** — MDL 原始  <br>**Grünwald (2007), MIT Press** — 教科书  <br>**Blier & Ollivier (2018), NeurIPS**, _The Description Length of Deep Learning Models_  <br>**Voita & Titov (2020), EMNLP** — MDL probing  <br>_(认知建模侧)_ **Pitt, Myung & Zhang (2002), Psych Review** — 模型选择与复杂度                                                                                                                                                                                                                                                                     | **★★★**    |
| **5 可分解性**       | ① 探针 ② 因果 ablation ③ 组合泛化   | ① **Alain & Bengio (2017)** — linear probes  <br>① **Hewitt & Liang (2019), EMNLP** — control tasks / selectivity ⚠️**必读,它是探针的方法学刹车**  <br>① Belinkov (2022), Comp Ling — 综述  <br>② **Li et al. (2023), ICLR** — Othello-GPT,探针 + 因果干预  <br>② Geiger et al. — causal abstraction / interchange intervention  <br>③ **Lake & Baroni (2018), ICML** — SCAN  <br>③ **Hupkes et al. (2020), JAIR** — _Compositionality Decomposed_(五侧面)  <br>③ **Keysers et al. (2020), ICLR** — CFQ / compound divergence  <br>③ **Lake & Baroni (2023), Nature** — MLC,人机头对头       | **★★★**    |
| **8 像人度**        | 剖面/错误模式相关 + noise ceiling   | **Rajalingham et al. (2018), J Neurosci** — 逐图像错误模式一致性(i1/i2n)⚠️**这是"比错误结构而非准确率"的原始出处**  <br>**Nili et al. (2014), PLoS Comp Bio** — RSA 的 noise ceiling  <br>**Schrimpf et al. (2020), Neuron** — Brain-Score 的天花板归一化  <br>**Bear et al. (2021), Physion** — 人机同刺激比较                                                                                                                                                                                                                                                                                            | **★★★**    |
| **9 像脑度**        | encoding model 预测神经         | **Yamins et al. (2014), PNAS**  <br>**Schrimpf et al. (2020), Neuron** — Brain-Score  <br>**Schrimpf et al. (2021), PNAS** — 语言版                                                                                                                                                                                                                                                                                                                                                                                                                                 | **★★★**    |
| **1 预测**         | 留出被试逐 trial NLL             | **Binz et al. (2025), Nature** — Centaur  <br>**Busemeyer & Wang (2000), JMP** — generalization criterion                                                                                                                                                                                                                                                                                                                                                                                                                                                        | **★★**     |
| **2 泛化**         | 新封面故事 / 新域                  | Busemeyer & Wang (2000)  <br>**"LLMs Do Not Simulate Human Psychology" (2508.06950)** — 对 Centaur 的直接 OOD 反驳                                                                                                                                                                                                                                                                                                                                                                                                                                                     | **★★**     |
| **4 学习效率**       | prequential 曲线形状            | **⚠️ 这个轴没有一个公认的独立 metric 文献。** 我只能诚实说:它就是轴 3 的曲线读法,引 Blier & Ollivier。别硬找引用,别硬造。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | **★**      |
- **Bean et al. (2025), NeurIPS D&B** — _Measuring what Matters_,构念效度
- **Hedge, Powell & Sumner (2018), Behav Res Methods** — reliability paradox ⚠️**你的 EF 任务信度问题的必引**
- **Miller (2024)** — Adding Error Bars to Evals


- evaluating capabilities such as grade school mathematics—Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., Hesse, C., and Schulman, J. Training verifiers to solve math word problems, 2021.
- general knowledge—Joshi, M., Choi, E., Weld, D. S., and Zettlemoyer, L. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension, 2017.
- programming—Chen, M., Tworek, J., Jun, H., Yuan, Q., de Oliveira Pinto, H. P., Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman, G., Ray, A., Puri, R., Krueger, G., Petrov, M., Khlaaf, H., Sastry, G., Mishkin, P., Chan, B., Gray, S., Ryder, N., Pavlov, M., Power, A., Kaiser, L., Bavarian, M., Winter, C., Tillet, P., Such, F. P., Cummings, D., Plappert, M., Chantzis, F., Barnes, E., Herbert-Voss, A., Guss, W. H., Nichol, A., Paino, A., Tezak, N., Tang, J., Babuschkin, I., Balaji, S., Jain, S., Saunders, W., Hesse, C., Carr, A. N., Leike, J., Achiam, J., Misra, V., Morikawa, E., Radford, A., Knight, M., Brundage, M., Murati, M., Mayer, K., Welinder, P., McGrew, B., Amodei, D., McCandlish, S., Sutskever, I., and Zaremba, W. Evaluating large language models trained on code, 2021.
- reasoning—Collins, K. M., Wong, C., Feng, J., Wei, M., and Tenenbaum, J. B. Structured, flexible, and robust: benchmarking and improving large language models towards more humanlike behavior in out-of-distribution reasoning tasks. arXiv preprint arXiv:2205.05718, 2022.
- Beyond the Imitation Game Benchmark—Srivastava, A. and authors. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models, 2023


| Benchmark   | 控制的资源                | 比较方式                                      |
| ----------- | -------------------- | ----------------------------------------- |
| EfficientQA | 整个系统的磁盘大小，包括参数、语料、代码 | 设置 6 GiB、500 MiB 等赛道；或者达到指定准确率所需的最小系统大小   |
| BabyLM      | 预训练语言数据量             | 所有模型只能使用约 10M/100M words，再比较语言能力          |
| DataComp    | 训练代码与计算预算            | 在固定训练流程、固定 compute scale 下比较数据选择策略        |
| AlgoPerf    | 硬件和目标 performance    | 比较达到同一目标 performance 所需时间                 |
| DAWNBench   | 目标 accuracy          | 比较 time-to-accuracy、cost-to-accuracy、推理成本 |
Performance–resource Pareto frontier
## 用学习曲线衡量信息/样本效率

如果“信息利用效率”指模型能从多少训练数据中学到多少，那么只比较最终 performance 不够，应当比较完整学习曲线：

Pm(n),n∈{10,50,100,500,…}P_m(n),\quad n\in\{10,50,100,500,\ldots\}Pm​(n),n∈{10,50,100,500,…}

常见总结方式包括：

- 固定 nnn 下的 performance
- 达到目标 performance 所需的数据量
- learning curve 的面积
- 每增加一批数据带来的 performance gain
- prequential/online Minimum Description Length

MDL 特别有意思：它不只看最终预测得多准，还考虑需要多少数据或多复杂的解码器才能提取出信息。Voita 和 Titov 将其用于表征比较，认为它比单一 probe accuracy 更能反映信息是否容易被利用。[MDL probing paper](https://aclanthology.org/2020.emnlp-main.14/)

不过 MDL 更常用于表征和 probe evaluation，还不是通用 benchmark 的标准 leaderboard 指标。


model spec 

model implement

behavior prediction ---- behavior
accuracy -- score
outcome -- env
reference

glanter