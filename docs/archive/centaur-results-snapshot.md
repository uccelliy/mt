# Centaur 阶段结果快照

**冻结日期**：2026-08-13
**用途**：保存转向 trial-level 自有模型前，Centaur 评估阶段真正值得保留的结果、限制和
产物处置依据。这里的数字是研究记录，不是已经完成复核的论文结果。

## 1. 数据口径

- 模型主对照：official Centaur-8B adapter vs matched Llama-3.1-8B base；
- runtime：NF4/4-bit；不得与 published BF16 放在同一结果列；
- 当前共同覆盖：73 experiments、6,499 sessions、1,144,236 marked choices；
- `enkavi2019gonogo` 因 no-go 无 marker 被排除，`xiong2023neural` 尚未补跑；
- 六个已跑模型在 task/condition 上的 participant 与 choice 数一致。

## 2. 冻结的主结果

| 结果 | Centaur | matched base | 差异/覆盖 |
|---|---:|---:|---:|
| full task-macro NLL | 0.73545 | 0.93579 | 0.20034 nat；Centaur 73/73 更好 |
| 去掉 14 个含 multi-token choice 的任务 | — | — | 0.16488 nat；59/59 更好 |
| 人类 choice 几何平均概率比 | — | — | $e^{0.20034}=1.222\times$ |
| 对 online bigram baseline 的 median ratio | — | — | Centaur 1.470×；71/73 胜 |

E3 必须排除 `frey2017risk`，因为其一行包含一个气球内的大量 choices，marked segment
不是可比较的窗口单位。排除后，在相同 `target>=20` choice 上：

| 历史窗口 | base − Centaur NLL（nat） |
|---:|---:|
| 0 | 2.4326 |
| 1 | 0.3074 |
| 2 | 0.2247 |
| 5 | 0.1955 |
| 10 | 0.2090 |
| 20 | 0.1932 |
| full | 0.1778 |

一个 marked-choice segment 消除了 w=0 差距的 87.4%，但没有消除约 0.18–0.21 nat 的
residual。最稳妥的解释是：行为微调很大程度安装了冷启动/答案接口先验，同时还有一个
较小、跨任务一致的预测增益；现有实验不能把 residual 归因到具体认知机制。

第一、第二 choice 的观测 session-option mass 分别为 Centaur 0.948/0.944，base
0.262/0.903，与“一个示例完成接口发现”方向一致。这只是诊断：合法集为 session 内实际
出现选项的并集下界，现有 A14 又是 micro/session-weighted 聚合。

## 3. 不冻结为主张的分析

- **A12 temperature calibration 未完成**：0.200 nat 不能被完整拆成排序与校准。
- **A13 只能说明 session-level marginal fidelity**：低于 i.i.d. null 不能证明逐 trial
  tracking；需要 trial shuffle/lag 破坏或逐 trial proper-score 对照。
- **A14 denominator/aggregation 不规范**：不能把当前 headline 当 task-macro 主结果。
- **A15 只覆盖 54 个任务且 mixed multi-token 任务采用部分 choice**：不能当 73-task
  accuracy leaderboard。
- **A17 是观察性 position curve**：trial 顺序、难度和参与者流失混杂，不能当作因果
  in-context learning 证据。
- 新批次还没有重新完成 official 36-family `r=1.00000` gate；旧批次完成过，且新旧共享
  任务的分数高度一致，但正式发表仍需重新对拍。

这些缺口正是停止扩展旧分析的原因：补完能让 Centaur audit 更严谨，却不会自动转化为
自己的训练方法或可识别的科学发现。

## 4. 产物处置原则

目标不是永久保存每张探索图，而是保留足够回答三个问题的证据：比较对象是什么、主数字
从哪里来、为什么决定转向。

### 建议保留

- 本快照与冻结的 benchmark 设计；
- 当前 `metrics_by_task.csv`、`probability_ratio.csv`；
- 当前 Centaur-8B 与 matched Llama-3.1-8B 的压缩 raw 表，至少保留到确认不再需要复核；
- 旧批次 `README.md` 与 `analysis_official_vs_ours.csv`，作为官方对拍记录。

### 建议删除

- 当前四个 Llama-3.2 1B/3B base/instruct raw runs；它们不再回答新的主问题；
- A13/A14/A15/A17、A9 和含 Frey 的旧 A4 图表；结论已在上文标明哪些有效、哪些失效；
- 125 MB per-choice online-baseline 表，只留 summary；
- 旧 `outputs/archive/v0-centaur-eval/scoring/` 中的 smoke、probe、failed、Minitaur、Qwen、
  duplicate paper-style 与其他 raw CSV；只留小型对拍摘要；
- 仅服务于上述弃用分析、尚未提交的 figure-builder 脚本，在确认没有新路线复用后删除。

删除前按“tracked 可从 git 恢复 / ignored raw 无法从 git 恢复”分两批执行；不要一次递归清空
整个 `outputs/`。
