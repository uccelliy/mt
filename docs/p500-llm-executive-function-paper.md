# Paper C — Unity and Diversity of Executive Control Across Language Models

**日期**：2026-08-13  
**状态**：当前论文设计；正在进入 C0 construct/interface audit  
**共享项目路线图**：[P500 三篇论文路线图](benchmark%20design.md)

## 0. 科学问题

> 跨不同 LLM/checkpoints 的 P500 行为差异，是否呈现与人类执行功能
> Unity–Diversity 理论相似的协方差结构？这些潜在维度能否在模型 lineage 外泛化，并对
> 传统排行榜之外的交互式能力提供增量预测？

本文研究的是**一组模型之间的表现协方差**，不是从因子相关反推出单个模型内部存在人类式
执行控制机制。即使 LLM 与人类出现相似因子，也只先称 human-theory-aligned measurement
structure；只有测量不变性证据足够时，才讨论更强的跨群体可比性。

## 1. 独立贡献与前两篇的边界

论文 A/B 把 P500 人类记录作为训练和评估数据，目标是建立自己的行为模型；论文 C 把现有
LLM/checkpoints 作为被测对象，目标是研究人工系统群体的能力结构。三篇共享 P500 task
ontology、pre-response state 定义、canonical actions 和泄漏测试，但统计单位、数据生成过程
与主结论不同。

论文 C 的独立贡献是：

1. 从 P500 task definitions 构建无泄漏、可重复的 **agent-facing executive-function
   battery**，而不是复用人类反应预测 transcript；
2. 用每个构念多个不同范式，正式比较人类执行功能理论对应的候选测量结构；
3. 在可链接的相同 indicators 上比较人类与 LLM 的 factor congruence，并只在共同题目、
   共同模态或正式 linked-form calibration 成立时检验 measurement invariance；
4. 检验执行功能因子在 model-lineage 外、控制规模和总体能力后的 criterion validity。

排行榜关联属于第 4 项，不单独拆成一篇。可识别的非线性生成式测量模型属于远期方法研究，
不进入当前论文 C 的完成条件。

## 2. 预先登记的理论模型

“三成分理论成立/不成立”不是一个充分的二元检验。正式分析至少比较：

| ID | 测量结构 | 回答的问题 |
|---|---|---|
| C-M0 | 单一 general/control factor | 模型差异是否几乎只有一个能力轴 |
| C-M1 | correlated Inhibition / Updating / Shifting | 原始三成分是否相关但可分离 |
| C-M2 | Common EF + Updating-specific + Shifting-specific | 修订 Unity–Diversity 是否更合适 |
| C-M3 | C-M1/C-M2 + modality/interface/task-method factors | 表面 EF 是否其实由任务形式驱动 |

C-M2 不预设 inhibition-specific factor：修订理论中，抑制任务的共同差异通常由 Common EF
吸收。ESEM/cross-loadings 只作预先规定的 sensitivity analysis；不能看过 test 结果后不断
修改 loading pattern 直到拟合。

## 3. P500 Agent Battery v1

### 3.1 不能直接复用当前 renderer

现有 P500 text conversion 是为“根据记录预测人类 choice”建立的，不是给人工系统施测的
心理测验。部分任务使用了推导后的比较结果，部分视觉/运动信息被摘要或缺失；直接输入会把
执行功能任务变成答案读取、符号解析或语言理解。

论文 C 必须从原 instrument/task generator 或无泄漏的原始 pre-response fields 生成独立
battery：

- 输入不得包含 expected response、correctness、由正确答案反推的 stimulus 描述或未来反馈；
- 视觉任务提供真实可见刺激，或明确限制为一个信息等价的 symbolic 子研究；
- updating/working-memory 任务采用可审计的 stateful streaming：每个 stimulus 只呈现一次，
  后续步骤只输入新事件并保留模型原生状态（如 KV cache），不重新附上完整 transcript，
  不允许外部 scratchpad、retrieval 或工具；
- adaptive task 由该模型自己的反应更新难度，不能重放人类 adaptive path；
- switch、stop、no-go 和 sequence task 保留真正的时序与事件边界；
- 每条 response 保留 model revision、interface、prompt、decoding、seed、item 和 condition IDs。

LLM wall-clock/API latency 主要由硬件、服务负载和解码策略决定，不作为人类 RT 的等价指标。
推理 token 数或计算预算若记录，只能作为系统成本/方法变量另行报告。

这里的 streaming protocol 测量的是模型的 **context-state updating under interference**，不是预先
假定与人类 working-memory capacity 同量尺。无法保存并审计原生状态的模型不能进入 primary
updating 分析；它可以进入不依赖该状态的补充任务。

### 3.2 主模态与模型总体

C2 的 primary population 是具有 immutable revision、能够接受统一 lossless symbolic interface
的 text-capable checkpoints。Primary battery 只纳入可以无损符号化且不改变核心 manipulation
的范式；这避免把“没有视觉输入通道”误当成执行功能较弱。

真实视觉 P500 任务由 VLM 在独立 multi-group sensitivity 中完成。VLM 与 text-only 模型不因
missing-by-design 的不同题目直接进入同一个主因子协方差；只有共同 anchor items 和跨模态
linking 通过后，才报告可比较的跨模态分数。

### 3.3 Construct map gate

每个因子至少需要 3 个、最好 4 个**不同范式**的可靠指标。同一 instrument 的不同负荷、版本
或 condition 不能冒充独立范式；它们用于估计 item difficulty、condition slope 或 method
variance。

初步候选需要逐一审计，而不是直接冻结：

- Inhibition/control：Flanker、Simon、anti-saccade、go/no-go/stop 等不同范式；
- Updating：n-back 之外，还需确认至少两个真正要求持续更新而非单纯 storage 的范式；
- Shifting：card/rule sorting、mixed pro/anti 或明确 switch/repeat 的任务，并确认至少三个
  独立范式和 trial-level switch indicator。

若 Updating 或 Shifting 未达到指标门槛，先缩小理论问题或补建平行任务；不得用同一范式的
多个版本补足 CFA 指标数。

### 3.4 Compact parallel forms 与 interface repeats

不让每个模型重跑全部 777k 人类 trials。先从通过 construct、泄漏、难度和可靠性审计的
items 构建平衡的 compact battery、共同 anchor items，以及至少两个 parallel forms：Form A
用于新 checkpoint 的 latent-score calibration，Form B/held-out items 用于预测检验。正式 item
数由 pilot 的信息量、ceiling/floor 和 Monte Carlo power 决定；题目选择只用开发模型与人类
train/calibration records，不能根据正式 model test roster 的结果筛题。

Item form 与 renderer/interface 是两项不同的重复。C0/C1 另冻结一个 primary interface 和一个
等信息 alternate renderer，在分层代表性 items 上做 form×renderer 交叉，以估计 prompt/interface
method variance。C2 主结果只用冻结的 primary interface，alternate renderer 作为 reliability
与 method-factor sensitivity，不事后挑对理论最有利的模板。

## 4. 模型样本与独立性

因子协方差的统计单位是 **model checkpoint**；大量 trials 只能降低每个 checkpoint 的测量
误差，不能增加独立模型样本量。

- prompt paraphrases、decoding seeds 和重复 runs 是测量重复；
- quantization variants、LoRA 和同一 base 的 post-training checkpoints 嵌套在 base lineage；
- 主要不确定性与验证按 lineage 聚类，不能把近亲 checkpoints 当完全独立；
- 闭源 API 在 C0/C1 保存模型版本与施测日期；无法冻结 exact revision 时不进入 C2 主样本。

当前分阶段预算：

1. **C0 smoke**：6–10 个差异明显的 checkpoints；只检查 stimulus、泄漏、parse、adaptive
   state、prompt sensitivity 和明显 ceiling/floor，不用于因子结构结论；
2. **C1 pilot**：约 30–50 个 checkpoints、至少约 10–15 个 base lineages；只用于
   item reliability、interface sensitivity、ceiling/floor、construct coverage 和 power 输入；
3. **C2 confirmatory roster**：以约 200 个 checkpoints、至少约 25–30 个 base lineages 为
   规划起点；最终 N 由与真实 loading、聚类、missingness 和模型复杂度匹配的 Monte Carlo
   power 决定，不把经验数字当硬阈值。

Roster 要覆盖而非混淆 architecture、scale、base/instruct 与 reasoning/post-training；这些属性
是 covariates/grouping variables，不是新的“被试重复”。Modality 不在 primary roster 内混合，
而在独立 VLM sensitivity 中作为 group/interface 变量。

C1 后冻结 target population、lineage 定义、同一 lineage 的 checkpoint 配额与分析权重。当前
规划默认每个 base lineage 最多 8 个主样本 checkpoint，并让任一 lineage 不超过总分析权重的
5%；最终数值由 within-lineage ICC 的 power simulation 决定。结论必须区分：加权 checkpoint
population 的测量结构，以及 leave-one-lineage-out 对新 lineage 的泛化。

C2 主协方差样本必须能固定 exact checkpoint/revision 和权重 checksum。无法冻结版本的 API
只可作为 supplementary temporal observation，不能进入主 factor covariance、跨群体比较或
正式 roster 的独立样本数。

## 5. 主测量模型

主分析使用 trial/item-level Bayesian confirmatory multidimensional IRT 或 generalized latent
variable model，而不是先把每个任务压成一个平均 accuracy 再做普通 CFA：

- binary/multiclass response 使用相应 likelihood；
- item/task difficulty、discrimination 与 condition manipulation 显式建模；
- 同一范式变体、prompt/interface 和 modality 的 local dependence 用 method/random effects
  处理；
- checkpoint 嵌套在 base lineage；重复 prompt/seed 用于估计 measurement error；
- ceiling/floor 与 primary battery 内的 missing-by-design 保留在 likelihood 中；VLM modality
  sensitivity 单独建 multi-group model。

Task-level CFA/SEM 作为透明、可复核的基线；单因子、三相关因子、nested/bifactor 与
method-factor 模型都要同时比较 in-sample fit、posterior predictive checks 和按 lineage 留出的
item-response predictive density。不能只用一个全局 fit index 决定理论“成立”。

预测验证不能把用于估计同一个 checkpoint latent score 的 responses 再当测试答案。每个
leave-one-lineage-out fold 中，global item/measurement parameters 只由其余 lineages 学习；
held-out checkpoint 只用 Form A/anchor items 做 score calibration，再预测 Form B/held-out
items。结构选择同时要求新 lineage 与新 items/forms 的 predictive fit。

## 6. Human–LLM 测量比较

P500 的人类记录是本文相对普通 LLM benchmark factor analysis 的关键优势，但“同一 task
名称”不自动等于同一 measurement scale。比较遵循以下证据阶梯：

1. 在现有 P500 人类数据上，用相同 construct map 和 behavioral contrasts 估计 reference
   structure，先报告 loading pattern 与 factor congruence；
2. 只有 humans 与 LLM 共享 task-relevant information、anchor items 和 condition definitions
   时，才检验 configural comparability；
3. 只有共同模态/界面，或用独立人类 calibration sample 完成 linked-form calibration 后，
   才检验 metric/partial invariance 与 DIF；
4. 若人类用原 GUI、LLM 用 symbolic renderer 且无法校准，只报告结构对应与差异，不使用
   “相同量尺”或 measurement invariance 表述。

现有人类 N=100，对复杂 multi-group 三因子模型可能偏小。正式方案必须用 trial-level 信息、
合理先验和 Monte Carlo calibration 评估可恢复性；若 power 不足，human structure 只作外部
理论锚点/探索性 comparison，不能宣称完整 measurement invariance。

## 7. 外部效度

预先冻结少量外部 outcome families，不抓取大量排行榜后筛显著相关。角色明确分成：

- **primary criteria**：agentic、interactive、long-horizon、tool-use、dynamic task adaptation；
- **capacity controls/secondary outcomes**：一般知识、数学、语言的冻结 composite，用于估计
  standard general capability，而不是与 primary criteria 混成一个 headline；
- 选择理论上不应由 EF-specific factors 增量预测的 negative-control outcomes。

对 exact checkpoint 使用同一 harness，或冻结来源、日期和协议一致的 leaderboard snapshot。
在联合 latent regression 或传播 factor posterior uncertainty 的预测模型中，检验 EF factors
是否在以下变量之外提供增量：

- P500 common/general factor；
- log parameter count、release date/compute proxy；
- base lineage、architecture、modality；
- base/instruct/reasoning/post-training 类型。

主要证据是 leave-one-lineage-out 的增量 predictive log score/$R^2$；若项目周期允许，保留
未来发布 checkpoints 作 temporal holdout。普通全样本相关只作描述。

## 8. 主要 estimands

1. **结构选择**：C-M0…C-M3 的 lineage-held-out predictive density 与 posterior predictive
   misfit；
2. **Unity/Diversity**：Common EF、specific-factor variance 与 factor correlations 的 posterior；
3. **跨群体对应**：factor congruence；仅在 linked-scale gate 通过后估计
   human–LLM loading/discrimination 差异与 DIF；
4. **可靠性**：parallel form、prompt/seed 和时间重复下的 factor-score stability；
5. **外部效度**：specific factors 相对 common ability 与规模/lineage controls 的增量
   out-of-sample prediction。

所有 factor scores 保留 posterior uncertainty；不把两阶段点估计当无误差 predictor。

## 9. 解释与停止规则

| 结果 | 可支持的解释 |
|---|---|
| C-M2 稳定胜出，且 linked-scale gate 后至少部分 metric invariance | 跨模型表现具有与修订 Unity–Diversity 对齐的结构 |
| C-M1 胜出、三个因子可分 | 原始三相关成分更符合该模型群体 |
| C-M0 足够且 specific variance 接近零 | P500 模型差异主要是单一 general/control axis |
| 加 method factors 后 EF structure 消失 | 原结果主要由 interface/modality/task form 驱动 |
| linked-scale gate 通过但 configural/metric invariance 失败 | LLM 能力结构不复现人类测量结构；仍是主要发现 |
| linked-scale gate 不成立 | 只报告 factor congruence，不把 renderer/modality 差异误作群体 DIF |
| EF-specific factors 在 lineage 外增量预测交互任务 | 得到 criterion validity，不等于识别内部机制 |
| pilot 出现普遍 ceiling/floor、低信度或三域指标不足 | 暂停 confirmatory roster，先重建 battery |

## 10. 执行顺序

1. 完成 task→construct、stimulus leakage、modality 和 temporal-demand audit；
2. 冻结 primary symbolic population、候选 paradigms、Q/loading map、streaming protocol 与
   construct exclusion rules；
3. 构建 agent-facing battery、anchor/parallel forms、primary/alternate interfaces、统一
   harness 和 provenance schema；
4. 用 6–10 checkpoints 通过 C0 smoke，再用 30–50 checkpoints 完成 C1 reliability、难度、
   method variance 和 ceiling/floor pilot；
5. 根据 C1 做含 lineage ICC 的 Monte Carlo power，冻结 lineage cap/weights、C2 immutable
   roster、items、candidate models 和 primary/secondary 外部 outcomes；
6. 运行 C2 confirmatory roster，先锁 P500 measurement model；
7. 再执行 human–LLM comparison、条件允许时的 invariance，以及 lineage-held-out external
   validity；
8. 发布 item/response matrix、精确 model revisions、可复现实验协议与 factor uncertainty。

## 11. 当前排除项与远期方向

当前不做：

- 直接复用预测人类反应的 P500 text JSONL；
- 把同一模型的 prompts/seeds 当独立样本；
- 用 LLM API latency 代替人类 RT；
- 从协方差结构声称 LLM 内部有同样认知机制；
- 事后遍历排行榜寻找最好看的相关；
- 因为 VAE 重构更好就给潜维度命名为 inhibition/updating/shifting。

远期可考虑一个可识别、theory-constrained nonlinear generative measurement model，但它需要
独立的方法学贡献、模拟恢复、强基线和外部数据验证。论文 C 首先使用可解释的 confirmatory
MIRT/GLLVM；只有出现稳定、可重复的系统性非线性失配，才启动远期方法项目。

## 12. 理论与近邻工作锚点

- Miyake et al. (2000), original correlated Inhibition/Updating/Shifting model:
  <https://doi.org/10.1006/cogp.1999.0734>
- Friedman & Miyake, revised Unity–Diversity framework:
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC5104682/>
- LLM general-factor evidence across 591 models:
  <https://doi.org/10.1016/j.intell.2024.101858>
- Recent neuropsychologically grounded LLM evaluation:
  <https://arxiv.org/abs/2603.02540>
- Generalized latent factor models for binary/count observations:
  <https://academic.oup.com/biomet/article/109/3/769/6356503>
