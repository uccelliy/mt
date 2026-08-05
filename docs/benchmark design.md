# Behaverse Benchmark 设计（v1）

**日期**：2026-08-04
**性质**：可施工的设计文档。指标定义、协议条款、参照系、判分规则都要能照着执行。
**配套**：`agents/PROJECT.md`（算力预算、结论边界、范围）、
`centaur-eval-design.md`（上一阶段科学设计）、
`centaur-eval-handoff.md`（已有结果与产物）、
`Centaur eval exploration.md`（探索方向）、`Server handoff.md`（集群约束）

**代码立场**（沿用本文件原有声明）：代码仍处于初步阶段，目的是能跑就行，先跑出
第一批结果；当代码量有一定数量时再考虑重构与标准化，因为要考虑代码与实际使用时
的体验。

## 两个课题

- **课题一：evaluation pipeline** —— 更多 metrics、更多 ablation 与 input 改动
- **课题二：不同模型的作用** —— 更多新模型、不同参数量、不同微调（在 P500 上）、
  推理时学习能力

---

## 1. 这份文档在解决什么

上一阶段在 Psych-101-test 上把打分管线做到了与官方数字对齐（r = 1.00000，base
与 adapter 两侧都验过）。管线是可信的，**NLL 这个指标也是有效的**——哪个模型
更准地预测人类反应，它排得出来。

问题只在于**不直观**。teacher-forced NLL 报出来的是一个负对数概率的平均值：

- **它不是"猜中人类选项的概率"**。NLL 是负对数，**本身就不是概率**——逐 choice
  要取 $e^{-\mathrm{NLL}}$ 换算一次才得到概率，而需要换算就说明它不是。聚合之后
  离得更远：对数的平均再取指数，也不还原成任何一个概率。
- **它不是"答对率"**。它不回答"模型会不会做这道题"，也不回答"模型的首选是不是
  人类的选择"。
- **它看不出比别的模型好多少**。一个效应量就是两个 NLL 相减，差以 nat 为单位，
  而 nat 没有自带的意义：脱离参照，"两个模型差 0.01 nat"对任何人都不说明任何
  事情。

**所以不是要换掉 NLL，是要新增指标。** NLL 保留为指标一（§3.1）：它是我们与
官方公布数字对齐的唯一凭据，也是与上一阶段全部结果保持可比的唯一凭据。在它之上
再加两层换算，把同一批数变成能直接读懂的形式：

$$
\underbrace{L_f}_{\text{指标一：NLL}}
\;\xrightarrow{\;e^{-L}\;}\;
\underbrace{p_f}_{\text{指标二：概率}}
\;\xrightarrow{\;\div\,\text{基线}\;}\;
\underbrace{R_f}_{\text{指标三：比值}}
$$

指标二读作"模型给人类实际那个答案多少概率"，指标三读作"比基线好几倍"。三个
指标的计算方式见 §3、模型输出的存储方式见 §2、聚合方式见 §4，三者分开定死。

---

## 2. 要保存的模型输出

**指标不在打分时计算。** 打分时只存"模型输出了什么"，指标全部离线从产物派生
（§3）。这样加一个新指标不需要碰 GPU，改一个指标定义也不需要重跑。Track P 与
Track S 写**同一套 schema**，只是部分列在某一侧为空。

### 2.1 三样东西

| | 存什么 | 名单由谁决定 | 粒度 |
|---|---|---|---|
| **① 模型的 choice** | 模型最终给出的那个答案字符串 | 解码结果 | 一个 choice 一条 |
| **② 模型前 20 的 token 和概率** | 每个 token 位置上模型排名最高的 20 个 token 及其 logprob，从全词表取 | **模型** | **一个 token 位置一条 list** |
| **③ 任务选项的 choice 和概率** | 这道题每个合法选项，及模型给该选项的 logprob | **任务** | 一个选项一条 |

② 和 ③ 都是「候选 + 概率」，区别只在名单谁定：② 是模型自己想说的，③ 是题目
允许说的。两者会重叠，但谁也不包含谁。

### 2.2 多 token 的 choice：② 存多份，③ 只存一份

一个 choice 未必是一个 token。`"yellow"` 在 Llama 下是 1 个，换个分词器可能切成
`"yel"` + `"low"`。这时：

- **②（top-20）按 token 位置存，切成几个 token 就存几份 list。** 位置 0 的
  top-20 是"模型开头想说什么"，位置 1 的 top-20 是"已经说了 `yel` 之后想接
  什么"，两者回答不同问题，不能合并。
- **③（合法选项）只存一份。** 选项是一个**序列**，它的 logprob 是自身各 token
  logprob 之和，天然是 choice 级的量，没有位置维度。

所以 ② 是 token 级、③ 是选项级，**粒度不同，必须分成两张表**（§2.5）。

### 2.3 为什么必须存概率，不能只存 ①

只存 choice 有一个致命盲区。设某类 trial 上人类 60% 选 A、40% 选 B：

- 完美校准的模型输出 $(0.6, 0.4)$ → 预测一致率 **60%**
- 永远说 A 的退化模型 → 预测一致率 **也是 60%**

**光看 choice 分不出"建模了人类的随机性"和"只会押众数"。** 人类反应本来就是
随机的，这是常态不是边缘情况。上一阶段对 wulff / ruggeri 的判别（"有信号但过度
自信" vs "无信号却偏离均匀"）整个建立在概率上；只存 choice 等于把这一整类失败
模式变成不可观测。

**② 单独要的理由。** 一道合法选项是 {yellow, blue} 的 trial，两种模型：

```
正常模型
  ② yellow -0.31 | blue -1.32 | yell -6.2 | Yellow -7.1 | the -8.9
  ③ yellow -0.31 | blue -1.32                     <- ③ 被 ② 完全包住

跑偏的 base 模型
  ② "\n" -0.4 | the -1.9 | You -2.3 | " " -3.1 | I -3.5   <- 一个合法的都没有
  ③ yellow -8.7 | blue -9.1
```

第二种情况只看 ③，会看到两个又低又接近的数，得出"模型在 yellow 和 blue 之间
拿不定主意"——**这个结论是错的**。模型根本没在答题，它想接着写叙述文本。
**② 一眼可见，③ 永远看不出来。**

② 的第二个用途是防返工：§8.5 已写明合法选项集目前取"session 内 `<<>>` 的
并集"，是真实合法集的**下界**。万一漏了 `Yellow`（大写），② 里已经有它，离线
重算即可；没有 ② 就得重跑全部 GPU 作业。

### 2.4 存储量实测

合法选项集大小（session 内 `<<...>>` 的并集，测试集的 6,561 个 session）：

| mean | median | p90 | p99 | max |
|---:|---:|---:|---:|---:|
| 4.49 | 2 | 11 | 49 | **180** |

长尾全部来自多 token response 任务（`wise2019acomputational` 180、
`cox2017information` 122、`popov2023intent` 51）。

⇒ ③ 约 +47 MB（现有 E0 CSV 是 68 MB）；② 按每 token 位置 20 条算约
200 B/行，全 roster 数 GB 量级。**存储不是约束，所以不按成本决定。**

### 2.5 Schema：三张表

**`predictions`** —— 一行一个 choice，日常 join 和聚合的对象（对应 ①）：

```
model, dataset, condition, experiment, participant, choice_index,
pred_choice, human_choice, correct_choice, is_correct, k_options,
nll, num_tokens, top1_prob, legal_mass, pred_entropy,
raw_generation, pred_rt_ms, format_ok          # 后三列仅 Track S
```

**`pred_topk`** —— 一行一个 (token 位置, 排名)，对应 ②。**多 token 的 choice
在这里有多行 `token_index`**：

```
<predictions 的键>, token_index, rank, token, logprob
```

**`pred_options`** —— 一行一个合法选项，对应 ③。选项级，**没有 `token_index`**：

```
<predictions 的键>, option, logprob, n_tokens, is_human, is_correct
```

`condition` 是条件标签（`full` / `w=1` / `openloop` …），保证同一套表能承载所有
消融，不为每个实验另起文件。

### 2.6 两条不可混用

- **Track P 与 Track S 的概率条件不同**：teacher-forced 条件于**人类**的历史，
  open-loop 条件于**模型自己**的历史。两者不得合并计算，也不得相减。
- **`condition` 不同的行不得混入同一次聚合**，除非该分析明确就是在做条件对比。

---

## 3. 指标的计算方式

三个指标，**层层转换，同一批数**：NLL → 概率 → 比值。全部逐任务算，不产出跨
任务总分（§4）。

**三个都不需要新的 GPU 工作**——它们是我们已经在算的 NLL 的变换。§2 存的分布
是给后续指标用的，不是这三个的前提。

### 3.1 指标一：每个任务的 NLL

模型给人类实际反应的负对数概率，按 §4 聚合（choice 内求和，choice 与
participant 逐层平均）：

$$
L_f=\frac{1}{I_f}\sum_{i}\ell_i,
\qquad
\ell_i=\frac{1}{J_i}\sum_{j}c_{j},
\qquad
c_j=-\log P(\text{该 choice 的完整字符串})
$$

#### 与论文口径的关系

Centaur 论文报告的是 session 内 `sum(nll)/sum(tokens)`、再对 session 平均。
**在论文自己的评估集上，这与上式逐位相同。** 原因：单 token 的 choice 上
`sum(tokens)` 就等于 choice 数，token 加权退化为 choice 等权。而实测

> **14 个含多 token choice 的 experiment 全部落在论文 36-family allowlist
> 之外，一个都不在里面。**

也就是说论文那个公式的本意就是逐 choice 算，写成除以 token 数只是因为在它的
数据上没有区别。我们改用 choice 口径：

- **对官方数字的验证不受影响**——r = 1.00000 的对表只跑 36 family，两种口径
  在那里完全一致；
- 只在 36-family 之外的 14 个 experiment 上与论文写法不同，而那正是论文写法
  会引入非本意行为的地方（§4.1）。

`sum(nll)/sum(tokens)` 保留为**管线校验用的中间量**（P0 轨道），不作为 benchmark
指标报告。

### 3.2 指标二：每个任务的概率

直接由 NLL 换算：

$$
p_f=e^{-L_f}
$$

**读法**：模型给人类实际给出的那个答案的**典型概率**。$p_f=0.55$ 就是"这个
任务上，模型平均给人类实际选的那个选项 0.55 的概率"，不需要任何参照就能读懂。

$L_f$ 按 §4 聚合（choice 内求和、choice 与 participant 逐层平均），所以
$e^{-c_j}=P(\text{人类那个答案})$ 在**每个 choice 上是字面成立的**，多 token
的答案也一样（§4.1）。

一条要写清的性质：**$p_f$ 是几何平均，不是算术平均。** $L_f$ 是对数的平均，取
指数得到的是逐 choice 概率的几何平均。它是"典型概率"，不是"平均概率"，两者不
相等——但两者都是概率、都在 $[0,1]$、都可以直接读。

### 3.3 指标三：模型概率 / 基线概率

$$
R_f=\frac{p_f^{\mathrm{model}}}{p_f^{\mathrm{baseline}}}
=\frac{e^{-L_f^{\mathrm{model}}}}{e^{-L_f^{\mathrm{baseline}}}}
=e^{\,L_f^{\mathrm{baseline}}-L_f^{\mathrm{model}}}
$$

**比值等于 NLL 之差取指数。** 这正是 §1 那个问题的解：nat 差读不懂，比值读得
懂。同一个数换个写法——

| NLL 差（nat） | 概率比值 | 读作 |
|---:|---:|---|
| 0.01 | 1.01× | 高 1% |
| 0.10 | 1.11× | 高 11% |
| 0.69 | 2.00× | 翻倍 |
| 1.44 | 4.22× | 4 倍 |

**基线用哪些**：§9 参照阶梯上的计数基线——uniform、base rate、sticky、bigram。
每个基线各出一列 $R_f$，不合并成一个数。

**一条硬性要求**：$L_f^{\mathrm{model}}$ 与 $L_f^{\mathrm{baseline}}$ **必须用
同一个聚合口径**（即都按 §4 算）。口径不同则比值无意义。

计数基线预测的本来就是一个 choice，没有 token，所以它天然落在 §4 的第二层，
第一层（choice 内求和）对它是空操作——**两边因此可以直接相减，不需要为基线
编造 token 划分**。这是 §4 把单位定在 choice 的一个直接好处。现有 E2 基线按
旧口径算过，**要用在这里须按 §4 重算**（A5b），零 GPU。

## 4. 聚合方法

**choice 内求和，choice 之上逐层平均。** 三层：

$$
c_{j}=\sum_{t}\mathrm{NLL}_{t},
\qquad
\ell_i=\frac{1}{J_i}\sum_{j}c_{j},
\qquad
L_f=\frac{1}{I_f}\sum_{i}\ell_i
$$

1. **token → choice：求和。** 一个 choice 若被切成多个 token，把这些 token 的
   NLL **相加**。
2. **choice → participant：平均。** 在每个 session 内，对该人的全部 choice
   取平均，每个 choice 等权。
3. **participant → experiment：平均。** 在每个任务内，对该任务的全部
   participant 取平均，每个人等权。

**到此为止。** 默认不再往上合成跨任务总分——需要总分时逐分析单独决定，并说明
它回答什么问题、附上逐任务表。

### 4.1 第一层为什么是求和

求和不是一种加权选择，**是恒等式**。模型逐 token 打分时每个 token 都条件于它
前面的一切（含同一 choice 内更早的 token），所以由链式法则：

$$
P(\text{“77.37”})=P(\text{“77”})\cdot P(\text{“.”}\mid\text{“77”})\cdot
P(\text{“37”}\mid\text{“77.”})
$$

两边取 $-\log$，乘法变加法，右边正好是三个 token 的 NLL 之和。因此

$$
c_j=-\log P(\text{该 choice 的完整字符串}),
\qquad
e^{-c_j}=P(\text{人类实际给出的那个答案})
$$

**这正是指标二要的读法**（§3.2）：模型给人类所选那个选项多少概率。
若改取平均，$e^{-c_j}$ 变成各 token 概率的几何平均——没有任何一个事件的概率是
这个数，且数值会随分词器把答案切成几段而变。

只有 14/75 个 experiment 的 choice 是多 token 的，其余 61 个（95.2% 的 choice）
求和与平均相同。而这 14 个里装的都是**单个行为决策**——`wise2019acomputational`
是一个连续评分（`77.37`）、`kumar2023disentangling` 是一个坐标（`[3, 4]`）、
`collsiöö2023MCPL` 是一个数值判断（`50`）、`krueger2022identifying` 是一个颜色词
（`turquoise`）——不是多个决策，所以不该被平均掉。

**长答案概率更低不是不公平，是对的**：`77.37` 本就比 `50` 更具体。而指标三是
比值，基线打在同一批 choice、同样的分词上，这部分是共有的，在比值里抵消。

**三个指标同一套聚合。** §3.1 已改用本节口径，论文的 `sum(nll)/sum(tokens)`
只保留为管线校验的中间量，不作为指标报告——因此文档中不再存在"两套口径"。

---

## 5. 数据集

**只用 Psych-101-test，全量，不设抽样层。**

| 文件 | 规模 |
|---|---|
| `data/psych-101-test/prompts_testing_t1.jsonl` | 6,561 session / 75 experiment / 117.8 万 choice |

**必须沿用原论文的 held-out split。** 公开 checkpoint 在每个实验约 90% 的
参与者上微调过，自建 split 会把训练参与者混进测试集，所有对比作废。任何阶段
都不得自行划分。

**没有 RT。** jsonl 只有 `text` / `experiment` / `participant` 三个字段（实测），
部分任务的 transcript 文本里渲染了反应时（如 `enkavi2019gonogo` 的
"press \<\<X\>\> in 555.0ms"），但那是**输入的一部分**，不是打分目标。v1 不做
RT 相关指标。

**一个已知的退化任务**：`enkavi2019gonogo`（59 session）。人类"不按"时不产生
`<<>>` 标记，所以被打分的位置全是按了键的，合法反应集只有一个元素——任何模型
在它上面都必然"猜中"。它是 75 个里唯一全 session 退化的（实测），相关指标须
**排除**该任务并注明。

---

## 6. 模型 roster

### 6.1 选择准则（先定准则再定名单）

(i) 近期开源；(ii) 有 base 权重可做 completion 打分。

### 6.2 70B 的精度：不构成妥协

- **FP16 的 70B 在本集群物理上跑不了**：约 140 GB > 4×32 GB = 128 GB（Server
  handoff §1.2）。int8 约 70 GB 理论可行但未验证，且 volta32 只有 6 个节点。
- **官方公布的 70B 结果本身就是 4-bit**：base 用
  `unsloth/Meta-Llama-3.1-70B-bnb-4bit`，Centaur-70B-adapter 加载其上
  （`scripts/experiments/build_official_comparison_figures.py:43`）。所以
  "4-bit base + adapter" **就是论文的评估配置**。
- **不使用合并权重的 70B**：Minitaur 已证明"合并到 BF16 再重量化"会损伤
  0.182 nat（= 真微调增益的 93%，H2 已判定）。

⇒ 70B 一律走 `4-bit base + 官方 adapter`。这既是官方口径又避开了部署损伤，
"能不量化就不量化"在这里与"跟官方一致"不构成冲突。

---

## 7. 两条轨道

### Track P（预测 / teacher-forced）

把人类真实反应喂回去，逐 token 算 NLL，不生成。复用
`src/mt/evaluation/transcript_scoring.py` 与 `context_windows.py`，另加一次
候选 logprob 的 gather 以产出 §2 的 ② 和 ③。

产出：§2 的三样存储（`predictions` / `pred_topk` / `pred_options`），据此离线
派生 §3 的三个指标（$L_f\to p_f\to R_f$）。

### Track S（模拟 / open-loop）

模型自己做选择、自己接收反馈、跑完整条轨迹，产出一条**行为轨迹**而不是一个
NLL。**直接复用官方 `openloop/` 的代码与逻辑**
（`~/wkspace/Llama-3.1-Centaur-70B/openloop/`）。

#### 7.1 官方已经建好引擎的任务

官方为 5 个任务手写了 `simulate.py`，其中 **4 个在我们的测试集内**：

| 任务 | 范式 | 拟合出的参数 |
|---|---|---|
| `kool2016when` | two-step | $\sigma(\tau)$ = model-based 程度 |
| `kool2017cost` | two-step | 同上 |
| `wilson2014humans` | horizon（5 个 horizon 各一份） | `information_logits.beta` = 定向探索 |
| `jansen2021dunningkruger` | 语法判断 | 见 `simulate.py` |

（`baar2021latent` 也有引擎，但不在测试集内。）

**v1 就做这 4 个，不自己造引擎。** 扩展到更多任务 = 逐任务手写 `simulate.py`，
成本是每个任务一次，不是一次性框架投入。

#### 7.2 引擎结构：复用什么、改什么

**复用**（原样或近乎原样）：

- `<task>/simulate.py` —— 任务环境 + 文本渲染 + 生成循环
- `models.py` —— 认知模型（`DualSystems`、`TabularRescorlaWagner*`、
  `InformationBonus`、`Stickiness`、`Temperature`）
- `trainers.py` —— 逐参与者梯度拟合（AdamWScheduleFree，1000 iter，
  choice 上的 cross-entropy）
- `openloop.py` —— 对人类与模型两边各拟合一遍、导出参数表

**要改的只有一处**：`simulate.py` 用 unsloth 的 `FastLanguageModel` 加载模型，
我们换成本仓库 `_common.py` 的 `load_model`（支持 `--adapter`、与 Track P 同一
套加载路径），以便跑我们自己的 roster。**其余逻辑不动**，改动越少越好对齐。

#### 7.3 环境是 yoked 的，反馈来自人类 CSV

奖励不是现场生成的，而是从人类数据 CSV 里读**预生成的奖励表**——two-step 的
`reward.0.0 / 0.1 / 1.0 / 1.1` 是四个选项在该 trial 的奖励概率，**全部选项都
有**。所以模型选了人类没选的那个，照样知道该给什么反馈。

这正是 design §7.2 所说"预生成反馈"那一类：因果结构可保持，干预合法。**一个
任务能不能做 open-loop，就看它的 CSV 有没有全部选项的结果**——这是逐任务审计
的判据（A6c）。

参与者只取 Psych-101-test 的 held-out 集（官方 `simulate.py` 里已有这段
filter），无泄漏。

#### 7.4 生成协议：采样、单 token、非法即随机

官方设置，照抄：

- `do_sample=True, temperature=1.0` —— **采样不是 argmax**。开环下要的是行为
  分布，取 argmax 会得到退化的确定性轨迹。
- `max_new_tokens=1` —— 每次只生成一个 token；选项标签是单个大写字母。
- **不做约束解码**。生成的 token 若不在合法选项内，均匀随机回退，并计数。
  官方代码在这里打印 `should not happen!`——**我们把它变成一个正式上报的
  指标**（非法率），因为它就是"模型有没有在答题"的直接读数。

#### 7.5 比较方式：拟合参数，不是 NLL

对**人类轨迹**和**模型轨迹**各拟合同一个认知模型，比较：

1. 该模型的**可解释参数**（$\sigma(\tau)$、`beta`）的**分布**，不只是均值；
2. **平均奖励**；
3. 非法率（§7.4）。

这与 §3 的三个指标是不同的东西，**不可互相换算、不可相减**：§3 测的是一步预测，
本节测的是自由运行下的行为形态。

#### 7.6 边界

- **只有 4 个任务**，且都是 RL / 探索类。不能代表 75 个任务。
- **每个任务只有一个参数**，不是完整的表型向量（design §8 的完整版本是更远的
  目标）。
- 环境是 yoked 的：奖励表来自人类那次实验，模型不是在一个全新采样的环境里跑。

---

## 8. 协议条款（预注册，跑之前定死）

1. **raw completion，不套 chat template。** Psych-101 是 completion 式
   transcript，套模板等于换了输入分布。若要评 instruct 模型，另在一个
   experiment 子集上跑 template 版作敏感性附注，不混入主结果。
2. **reasoning / thinking 模型排除出 v1。** teacher-forced 打分在其上没有同一
   定义（答案分布条件于模型自采样的 CoT）。重新纳入需要 n 条 CoT 的蒙特卡洛
   边缘化，成本 ×n、方差大，v1 不做。
3. **数值配置钉死**：`--dtype fp16 --load 4bit --batch-tokens 8192`，
   `-C volta16`。显式写 `--dtype fp16`，**不要用 `auto`**（它在 MPS 上返回
   bf16，是系统性偏差不是零均值噪声）。**要互相比较的运行必须同配置、同硬件
   类别，并随结果记录。**
4. **不与上一阶段的 legacy 结果合并。** 为 §2 的 ② ③ 反正要重跑，且旧结果用的
   是 `--batch-tokens 16384`。重跑后须重新确认 P0 对官方 36-family 的
   r = 1.00000（零 GPU 派生，§3.1）。
5. **合法选项集是下界。** 取 session 内全部 `<<...>>` 的并集，逐任务报覆盖率。
   选项极多的任务（如 `hebart2023things`，规范空间 1,823 个选项）上这个近似基本
   无意义，该类任务不报 §2 的 ③。
6. **Track S 的生成协议按 §7.4 钉死**：`do_sample=True, temperature=1.0`、
   `max_new_tokens=1`、**不做约束解码**、非法选项均匀随机回退并计数。不得改成
   argmax——那会得到退化的确定性轨迹。
7. **`enkavi2019gonogo` 排除**（§5）：合法反应集只有一个元素，任何模型必然
   "猜中"，该任务不进入任何跨模型比较。

---

## 9. 基线

§3.3 的比值 $R_f$ 需要一个分母。基线共四条，按假设强度递增，全部只需数数、
不需要任何任务理解——因此它们共同定义了"**不算认知证据**"的地板：查表就能达到
的水平，不构成模型理解了人类认知的证据。

基线分**两族**：会话内（本节）与群体（§9.1）。

### 9.0 第一族：会话内计数（prequential）

预测第 $t$ 个 choice 只用同一 session 前 $t-1$ 个 choice，不看未来。设该
session 出现过的标签集为 $\mathcal{L}$、大小 $k$：

**四条都是概率预测，不是点预测。** 指标是 NLL，而确定性预测猜错一次就是无穷大
——所以"sticky"不是"一定重复上一个"，而是给重复分配一个拟合出的概率。

$n_\ell$ = 该 session 前 $t-1$ 个 choice 里标签 $\ell$ 的次数，$s$ = 平滑常数：

| 基线 | 对第 $t$ 个 choice 给 $\ell$ 的概率 | 对应的行为规律 |
|---|---|---|
| **uniform** | $1/k$（恒定，NLL 恒为 $\ln k$） | 纯机遇水平 |
| **base rate** | $\dfrac{n_\ell+s}{(t-1)+sk}$ | 反应偏好 / 选择频率 |
| **sticky** | $\theta$ 若 $\ell$ = 上一个，否则 $\dfrac{1-\theta}{k-1}$；$\theta=\dfrac{n_{\text{重复}}+s}{n_{\text{转移}}+2s}$ | 选择惯性、perseveration |
| **bigram** | $\dfrac{n_{(\text{上一个},\,\ell)}+s}{n_{\text{上一个}}+sk}$ | 一阶序列依赖 |

sticky 的 $\theta$ 是从该 session 已发生的转移里在线估出来的，所以它仍然是零
自由参数的计数规则，不需要拟合步骤。

### 9.1 第二族：群体基线（规范空间）

上面四条只看**这个人自己**的历史。第二族看**别人**：用不在 test split 的参与者
拟合 base rate 与 bigram，再给 test 参与者打分。它回答的是另一个问题——模型比
"记住人群的选择统计"强多少。

**为什么不能在 transcript 的标签空间做**：Psych-101 给每位参与者随机分配按键
字母，所以同一个 `A` 在不同人那里指不同选项，跨人计数是噪音（试点中群体
base rate ≈ $\ln 26$）。

**为什么在规范空间可以做**：HF 上有逐任务的原始 table，其中 `choice` 列是
**未随机化的规范编码**，对所有参与者含义一致。`mt.models.baselines.canonical_tables`
管这层访问，本地缓存在 `data/psych101_tables/`。

- **覆盖**：38 个 family 有 table，其余没有——**没有 table 的任务就没有群体
  基线**，不得用会话内基线顶替。
- **无泄漏**：拟合只用非 test 参与者，打分只在 test 参与者上，逐参与者守卫
  （table 行数 == transcript `<<>>` 数）通过才计入。
- **平滑**：Laplace，支撑集为该任务全表的选项集。
- **一个不可比的例子**：`hebart2023things` 规范空间有 1,823 个选项，而 transcript
  每 trial 只给 3 个——边际分布对条件分布，不可比，**排除**。凡是规范空间与
  transcript 选项集不对应的任务都要逐个判，不能一律纳入。

runner：`scripts/experiments/run_population_baselines.py`。

### 9.2 两族基线回答不同问题，分开报

| | 用谁的历史 | 覆盖 | 问的问题 |
|---|---|---|---|
| 会话内（§9 上表） | 这个人自己，前 $t-1$ 个 choice | 75/75 | 模型比"从上下文薅表面统计"强多少 |
| 群体（§9.1） | 其他参与者 | 38 个 family | 模型比"记住人群选择统计"强多少 |

**不合并成一个数**，各出各的 $R_f$ 列。

**平滑与边界情形**（现实现 `mt.models.baselines.sequence.score_sequence_online`，
$s = 0.5$，即 Jeffreys 先验而非 Laplace 的 $s=1$；常数随结果记录）：

- **第一个 choice**：没有前驱，sticky 与 bigram 都退化为 base rate。
- **前驱从未出现过**：bigram 的分子分母都只剩平滑项，退化为 **uniform**（不是
  base rate）。
- **$k=1$**（该 session 只出现过一个标签）：四条基线全部给概率 1、NLL 0。
  `enkavi2019gonogo` 整体如此，已按 §5 排除。

**口径**：基线的 NLL 必须按 §4 聚合，与模型完全一致。基线预测的本来就是一个
choice、没有 token，所以 §4 的第一层（choice 内求和）对它是空操作，两边可以
直接相减（§3.3）。

---

## 10. 登记表

两张表，分工严格：**表一只跑分、不解读；表二只解读、不跑分。**
表一的每一行产出一份逐 choice 的原始记录（§2 的 schema），表二的每一行从这些
记录派生结论。id 前缀 `R` = run，`A` = analysis。

### 10.1 打分运行（只得到初步结果，不做分析）

每行产出一个目录 `outputs/runs/<id>/`，内含 §2 的三张表
`predictions.csv` / `pred_topk.csv` / `pred_options.csv`。
**产出列为空 = 该行还没跑。**

| id | 模型 | 轨道 | 数据 / 任务 | 产出 |
|---|---|---|---|---|
| R1 | `meta-llama/Llama-3.1-8B`（base） | Track P | 75 experiment 全量 | |
| R2 | `meta-llama/Llama-3.1-8B` + `marcelbinz/Llama-3.1-Centaur-8B-adapter` | Track P | 75 experiment 全量 | |
| R3 | `marcelbinz/Llama-3.1-Minitaur-8B` | Track P | 75 experiment 全量 | |
| R4 | uniform | 基线（零 GPU） | 75 experiment 全量 | |
| R5 | base rate | 基线（零 GPU） | 75 experiment 全量 | |
| R6 | sticky | 基线（零 GPU） | 75 experiment 全量 | |
| R7 | bigram | 基线（零 GPU） | 75 experiment 全量 | |
| R8 | 群体 base rate（规范空间） | 基线（零 GPU） | 38 个有 table 的 family | |
| R9 | 群体 bigram（规范空间） | 基线（零 GPU） | 38 个有 table 的 family | |
| R10 | `google/gemma-4-E2B-it` | Track P | 75 experiment 全量 | |
| R11 | `google/gemma-4-E4B-it` | Track P | 75 experiment 全量 | |
| R12 | `google/gemma-4-26B-A4B-it` | Track P | 75 experiment 全量 | |
| R13 | `meta-llama/Llama-3.2-1B` | Track P | 75 experiment 全量 | |
| R14 | `meta-llama/Llama-3.2-1B-Instruct` | Track P | 75 experiment 全量 | |
| R15 | `meta-llama/Llama-3.2-3B` | Track P | 75 experiment 全量 | |
| R16 | `meta-llama/Llama-3.2-3B-Instruct` | Track P | 75 experiment 全量 | |

待补的行：70B（`unsloth/Meta-Llama-3.1-70B-bnb-4bit` + 官方 adapter）、
Track S 的 4 个任务 × roster。

#### R10–R16 的三件事

**规模与卡型**（4-bit 权重实测自 HF 的 safetensors 元数据）：

| 模型 | 参数 | 4-bit 权重 | 卡 |
|---|---:|---:|---|
| `Llama-3.2-1B` / `-Instruct` | 1.24 B | 0.7 GB | volta16 |
| `Llama-3.2-3B` / `-Instruct` | 3.21 B | 1.8 GB | volta16 |
| `gemma-4-E2B-it` | 5.12 B | 2.8 GB | volta16 |
| `gemma-4-E4B-it` | 8.00 B | 4.4 GB | volta16 |
| **`gemma-4-26B-A4B-it`** | 26.54 B | **14.6 GB** | **volta32** |

26B 那个是 MoE，4-bit 权重就占掉 16 GB 卡的 14.6 GB，活化没地方放——只能走
volta32（6 个节点，排队慢）。其余六个在 volta16 上都宽裕。

**Llama-3.2 四个全部是 gated**（`gated=manual`）。下载前必须先在 HF 上用你的
账号同意许可，这一步只能你自己做；没同意的话作业会在 preflight 就失败。

**四个 instruct 模型（三个 `-it` + 三个 `-Instruct`）按 §8 第 1 条处理**：
raw completion，不套 chat template。Llama-3.2 在 1B 和 3B 上各有 base/instruct
配对，正好构成"套不套模板"这条敏感性分析的干净对照——同权重、同规模，只差
指令微调。

**Gemma 4 的架构已验证可用**：`AutoModelForCausalLM` 映射到
`Gemma4ForConditionalGeneration`（多模态壳，文本单独走没问题）；打分器的
`_unwrap_base` 会停在 `Gemma4Model`（它没有 `lm_head`，`get_output_embeddings()`
返回 None），其 `forward` 接受 `input_ids`/`attention_mask`/`use_cache`、
`pixel_values` 可选、输出含 `last_hidden_state`——**显存优化路径照常生效**，
与 Llama 同分支。

#### 全部要重跑

上一阶段的产物（`outputs/scoring/*_e0_full_4bit.csv`）只有五列——
`experiment, participant, choice_index, nll, num_tokens`。**§2 要存的三样一样
都没有**：没有 ①（模型自己给出的 choice），没有 ②（每个 token 位置的 top-20），
也没有 ③（各合法选项的 logprob）。那五列里只有"人类所选那个选项的 NLL"。

所以 R1–R7 全部要按 §2 的 schema 重跑。另外两个理由同向：

1. **协议一致性**。旧产物跑在 `--batch-tokens 16384` 的消费级卡上，不是 §8
   第 3 条钉死的 `8192` + `volta16`。§8 要求互相比较的运行同配置。
2. **两处已知数据脏点**顺带消除：Minitaur CSV 里 `collsiöö2023MCPL` 被写成
   `collsi枚枚2023MCPL`（CP936 残留，分数有效仅名字错）；10 个
   `zorowitz2023data` session 需换成 UTF-8 重打的版本。

**唯一可以先做的事**：三个指标只用到"人类所选选项的 NLL"，而旧 CSV 的 `nll`
列已经是 choice 内各 token NLL 之和（实测：`num_tokens=2` 的行平均 nll 0.8199、
逐 token 0.4099，正好两倍），即 §4 第一层要的 $c_j$。所以可以从旧产物零 GPU
先算出 $L_f\to p_f\to R_f$ 作**预览**。预览数与重跑后的正式数**分开存、不混用**。

### 10.2 分析（在打分结果之上）

| id | 分析名称 | 分析目的 |
|---|---|---|
| | | |
