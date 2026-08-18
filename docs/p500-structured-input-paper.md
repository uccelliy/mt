# Paper A — Trial-Native State vs. Language Serialization

**日期**：2026-08-13  
**状态**：论文级设计草案  
**共享协议**：[P500 三篇论文路线图](benchmark%20design.md)

## 0. 科学问题

> 把异质认知任务的原生 trial state 压成自然语言，是否改变多任务行为模型的归纳偏置、
> 数据效率与对未见任务的泛化？

这是一篇 representation/learning-system 论文，不是“换一种输入格式是否涨点”的报告。
它比较两个从同一 canonical state 生成、信息匹配且共享 choice 输出接口的完整系统：

- 自然语言序列化利用语言预训练先验，但增加 token 长度并隐去字段类型；
- trial-native structured view 保留 categorical/numeric/set/sequence roles，但需要学习新的
  typed adapter。

目标不是证明 structured 更像人脑，而是识别两种输入系统在 choice 学习、样本效率和任务
迁移上的可重复差异。

## 1. 论文贡献边界

论文 A 负责首次定义并验证：

1. P500 Trial-State v1、family sufficient-state registry 与 canonical action ontology；
2. 从同一 state 自动生成的等信息 text/structured views；
3. shared candidate-action evaluator 与 participant-safe split；
4. representation 对 task-internal fit、sample/compute efficiency 和 task transfer 的影响。

RT 字段可以保留在共享数据 schema 中，但不作为本论文的训练 target、模型选择依据或主结果。

## 2. 固定条件

- target：choice-only；
- history：预先冻结的 $K_{ref}$；
- 输出：相同 canonical candidate-action head；
- split：相同 participant train/validation/test manifest；
- 数据：相同 target trials、合法 action masks 和 observation fields；
- 训练：相同更新数、choice loss、seed roster，并记录 tokens、FLOPs 与 wall time；
- participant ID 只用于 split/聚合，不进入主模型输入。

Text 不得使用 structured 条件没有的自然语言解释、答案名称或作答后信息；structured 不得通过
family-specific classifier 获得 text 条件没有的输出约束。

## 3. 主实验

先用一个主 backbone、一个规模和 3–5 seeds：

| ID | 输入 | 目的 |
|---|---|---|
| A-T1 | canonical deterministic text renderer | 主要语言化系统 |
| A-T2 | 第二个等信息 renderer | 排除单一模板/措辞效应 |
| A-S | typed structured state | trial-native 系统 |

第二 renderer 必须改变合理的语言实现，例如字段顺序、句式或 compactness，但不能改变信息集合。
正式运行前冻结两个 renderer，不根据 test 表现挑选模板。

至少追加两个定向检验：

- **structured ablation**：去掉 field-role/type encoding 或将字段 flatten，检验收益是否来自
  typed structure，而不只是不同参数化；
- **架构复现**：在第二 backbone、规模或 pretrained/from-scratch 条件上复现主方向，避免
  把某个 adapter 与 backbone 的适配性概括成普遍 representation 结论。

## 4. 泛化与效率

### 4.1 Held-out participants

同一 participant 的全部 sessions、instruments 和 trials 只能属于一个 split。主终点是
held-out participant choice NLL。

### 4.2 Held-out instruments 与 families

- held-out instrument：family 可见，但一个具体 instrument 的人类 targets 完全不进入训练；
- grouped held-out family：预先冻结若干 family folds，state schema/action ontology 可知，
  但该 fold 的人类 targets 不可见。

只有任务层迁移成功时，才使用跨任务/foundation 表述。

### 4.3 Data 与 compute efficiency

在训练前冻结至少三个数据比例，使用同一 participant/family sampling rule。主曲线按 training
examples 匹配；另报 compute/FLOPs-matched 敏感性。Text token 序列更长，因此二者不能合并成
一个“完全公平”的数字。

## 5. 指标与 estimands

主指标：

- participant→instrument→family macro choice NLL；
- held-out-instrument/family choice NLL；
- choice calibration；
- 达到同一 NLL 所需的数据量和 compute。

对每个 renderer 分别计算：

$$
\Delta_{struct}^{(r)}
=L_{choice}(\text{text renderer }r)-L_{choice}(\text{structured}).
$$

不能只汇报对 structured 最有利的 renderer。Family-level paired effects、seed uncertainty 和
预先冻结的等效区间与最小有意义效应在正式 test 前登记。

## 6. 可选历史诊断

减少历史不是主贡献。只有 representation 主效应稳定后才追加：

| 输入 | full history | short history |
|---|---|---|
| text | A-T-full | A-T-short |
| structured | A-S-full | A-S-short |

四个条件保持 target trials、更新数和 task-sufficient current state 相同。若不完成这个交叉，
历史结果只能称某个模型的 robustness，不能支持 representation×history 结论。

## 7. 成功、降级与停止规则

| 结果 | 论文结论 |
|---|---|
| structured 在两个 renderer、迁移层和架构复现上稳定更好 | trial-native interface 改善学习与泛化 |
| 只胜一个 renderer | renderer/template effect；不能概括为 text vs structured |
| 只在 seen tasks 更好 | task-internal fit 改善；没有跨任务证据 |
| 落入预定等效区间 | 语言化没有实质损失；typed interface 不是必要条件 |
| text 稳定更好 | 语言预训练先验超过序列化代价 |

负结果和等效结果可以成文，但等效区间必须事前定义，不能把“不显著”直接解释为相同。

## 8. 排除项

- 不训练或比较 RT head；
- 不运行 text+RT 来补足故事；
- 不把 participant holdout 当作 task transfer；
- 不声称 structured 更接近人脑；
- 不恢复 Psych-101、Centaur、70B roster、prompt 扰动大全或完整 context sweep；
- 不同时展开 ICL/LoRA/full-finetuning 全组合。

## 9. 执行顺序

1. 完成 Trial-State v1、split manifest、action head 与信息泄漏 tests；
2. 冻结 A-T1/A-T2/A-S、$K_{ref}$、数据比例与 family folds；
3. 通过 tiny-set overfit、label shuffle 和简单 baseline；
4. 跑主 participant split 与 data-efficiency；
5. 跑 instrument/family transfer；
6. 做 structured ablation 与第二架构定向复现；
7. 冻结 A-S 接口和配置，交给论文 B 使用；
8. 只有主结论稳定后，决定是否追加 history diagnostic。
