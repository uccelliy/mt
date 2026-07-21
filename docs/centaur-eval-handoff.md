# Centaur 评估工作流：交接文档

**日期**：2026-07-21
**配套文档**：[centaur-eval-design.md](centaur-eval-design.md)（科学设计，实验分解见 §12）
**范围**：第一阶段——只做 LLM 推理侧打分（不微调、不复现认知模型）

---

## 0. 一句话现状

打分引擎（E0/E1/E3）与序列基线（E2）已实现；相关评估与 runner 测试 44 个全过。
E2 已全量跑完出结果。E0/E3 尚未用真模型全量跑：HPC 维护中，本地 24GB Mac
跑 bf16 的 8B **内存不稳定**，已加量化支持转投 CUDA 卡（RTX 5060 Ti）。

**当前状态**（2026-07-22）：RTX 5060 Ti 的 CUDA/NF4 环境、E0/E3 真模型 smoke 与
最长真实 session 探针均已跑通；gated `Psych-101-test` 已获授权并下载。按用户要求尚未
启动全量，由用户手动启动。4-bit 结果单列为运行时量化复现，不与 BF16/论文主结果混写。

---

## 1. 实验状态（对应 design §12）

| # | 内容 | 状态 | 产物 / 备注 |
|---|---|---|---|
| E0 | full-context 逐 choice NLL | 引擎就绪，**未全量跑** | 需 CUDA + 4bit 或 HPC |
| E1 | 逐 trial 位置的适应曲线 | 待做（**零算力**） | 从 E0 的 CSV 按 `choice_index` 重聚合即可，无需再打分 |
| E2 | 计数序列基线 | **已全量完成** | `outputs/scoring/e2_all_tasks_s50.csv`（+ `_summary.csv`）|
| E3 | 上下文窗口截断 | 引擎就绪，**未全量跑** | 与 E0 同一次作业跑 |
| E4 | 语言表面扰动 | 未开始 | 每个实验需单独设计变换，最费手工 |
| E5 | 上下文×微调因子分解 | 待做（**纯分析**） | 从 E0+E3 的 CSV 计算，见 design §6 |

---

## 2. 代码地图

四层，依赖严格向下，读代码自底向上一遍即可：

```
scripts/*.slurm            作业层：资源声明 + 流程串接，无逻辑
scripts/experiments/*.py   入口层：argparse、文件读写、续跑/分片/容错编排
src/mt/evaluation/*.py     库层：全部科学逻辑，可单测，无 I/O
mt/data/_llm_supervision   既有代码：唯一衔接点 find_target_spans（定位 <<>>）
tests/evaluation, tests/experiments   平行，只测库层与入口 helper
```

**库层**（`src/mt/evaluation/`）：

- `transcript_scoring.py` — E0/E1 核心。teacher-forced 逐 choice NLL。
  链路：`_prepare_marked_text`（字符 span→token 下标，BOS 偏移、上下文守卫）
  → `score_marked_texts`（按长度打包 batch）→ `_score_batch` + `_forward`
  （**只在打分位置上过 lm_head，不 materialize 全词表 logits**）。
- `context_windows.py` — E3。`segment_transcript` 无损切分（header/逐 choice
  段/tail，重组必须恒等原文否则 RuntimeError）；`build_window_prompt` = header
  + 最近 w 段；`score_window_grid` 批量打分 (target,window) 网格。
  **import 了 transcript_scoring**，是库层内唯一的内部依赖。
- `sequence_baselines.py` — E2。`score_sequence_online`（会话内 prequential
  计数，见下节决策）；`TableBuilder`/`score_sequence` 是群体版备用。

**入口层**（`scripts/experiments/`）：

- `_common.py` — 三个 runner 的共享底座：`load_sessions`（过滤/抽样/分片/
  `max_chars` 跳过 + 逐任务日志）、`load_model`（普通 or bitsandbytes 量化）、
  续跑三件套（`completed_sessions`/`append_records`/`guard_output`）、
  `empty_device_cache`、`resolve_dtype`、`skip_log_for`/`failure_log_for`。
- `run_transcript_scoring.py`（E0）、`run_window_scoring.py`（E3）、
  `run_sequence_baselines.py`（E2）、`preflight.py`（提交前自检）。

**作业层**（`scripts/`）：`smoke_e0_e3.slurm`（30 分钟 1 卡烟测）、
`e0_e3_minitaur.slurm`（主作业，4×V100 数据并行 + 合并分片）。
注：这两个 slurm 是为 HPC 的 Minitaur-8B 写的；本地 CUDA 卡直接用命令行。

---

## 3. 本会话的关键决策（代码里看不出的 why）

1. **必须用 Centaur 原论文的 held-out split**（`Psych-101-test`），不能用本仓库
   splitting 重切——公开 checkpoint 在 ~90% 参与者上微调过，自建 split 会泄漏。
2. **E2 用会话内在线（prequential）计数，不是群体拟合**：Psych-101 给每个参与者
   随机分配按键字母，跨参与者的群体计数在原始标签空间上是噪音（试点 base rate
   ≈ ln 26）。在线计数严格因果、无泄漏，且概念上正好对应"ICL 能从上下文薅到的
   表面统计"。群体版保留为 `--mode population` 备用。
3. **E2 基线看不到逐 trial 的选项集**（如 two-step 每 trial 只给两个选项中的一对），
   所以只能贴 ln(4) 而非 ln(2) 走；对比 LLM 时要记住 LLM 读得到选项。
4. **logits 内存优化**：`_forward` 只在打分位置过 lm_head，避免 `[seq,vocab]`
   全张量（12.8 万词表，长会话仅此就数 GB）。与 dense 路径逐 token 等价
   （测试 + Qwen 真模型 0 差异）。不暴露 `.model`+output embedding 的模型自动回退。
5. **打分只需 `use_cache=False`**（无生成，不需要 KV cache）。
6. **E3 网格抽样**：全量对每 choice×每 w 重构 prompt 会到数十亿 token，按固定
   trial 位置网格抽样（`--num-positions`，默认 8）。

---

## 4. 本地运行的内存问题始末（用户重点关注）

**目标**：HPC 维护期间在本地用真 Minitaur-8B 跑预览。

**踩过的坑与修法（时间顺序）**：

1. 初版打分对整段序列做 float32 log_softmax → 长会话显存爆。
   **修**：只 gather 目标位置（`_score_batch`）。
2. `model(...).logits` 仍 materialize 全词表 `[seq,vocab]` → 长会话数 GB。
   **修**：logits 内存优化（决策 4）。
3. 24GB Mac 上多会话拼 batch 使注意力显存翻倍 + 会话间显存不释放 → OOM。
   **修**：`--chunk-size 1`（不拼会话）、`use_cache=False`、每会话
   `gc.collect()`+清缓存、逐会话捕获 OOM 记入 `.failed.csv` 跳过而非中断。
4. **仍不稳**：用户观察到 Python 内存运行时突然涨到 30–40GB 冲破物理内存。
   根因：**16GB 的 bf16 模型放在 24GB 机器上，运行时波动一冲即破**，非代码可根治。

**结论与出路**：
- **24GB Mac 跑 bf16 8B 判定为不可行**，别再调。单会话探针能跑（NLL 0.44 合理），
  但全量会因累积/波动 OOM。
- 已加 **`--load {8bit,4bit}`**（bitsandbytes，CUDA only）。RTX 5060 Ti 实测
  4-bit 模型常驻显存 5.68 GiB，短评估峰值 6.27 GiB；加载 BF16 checkpoint 并现场
  量化时，Windows 进程主内存峰值 15.83 GiB，加载完成后回落到 1.86 GiB。测试机
  有 32 GiB 主内存，余量够，但加载前仍应关闭占内存较大的程序。
- 当前打分设置 `use_cache=False`，因此长会话的额外显存主要是 prefill 激活与注意力
  工作区，不是 KV cache。Windows PyTorch wheel 未编译 FlashAttention，原先会在 GQA
  上回退到申请约 336 GiB 的平方内存 math 路径；现已在 CUDA 上显式按
  Flash → cuDNN → memory-efficient → math 排序，Mac/CPU 路径不受影响。
- 测试集最长 session（`xiong2023neural/exp1.csv`，participant 28，168,968 字符）现已
  跑通：E0 得到 4,800 个 choice，论文兼容 token NLL 0.4723；E3 的 0/5/full 窗口探针
  也通过。`nvidia-smi` 观察到整卡峰值约 15.8/16.3 GiB，主内存工作集约 7.6 GiB。
  因此 16GB **可跑但余量很窄**；全量时关闭其他 GPU 程序，并使用 `--chunk-size 1`。
- 4-bit 会改变 NLL。允许做一条独立的“量化复现”轨道：固定原始 held-out split、
  prompt、metric 与 NF4 配置，重新计算论文兼容的量化参考数；结果必须标成
  `Minitaur-8B BF16 checkpoint, runtime NF4`，不能替代 BF16 或 70B 主结果。

**⚠️ 未清理的污染**：Mac 上那次失败的运行把一批**本不大**的会话（如 1067 字符）
误记进了 `outputs/scoring/minitaur8b_e0_full.failed.csv`（当时是内存满不是会话大）。
若要在 Mac 上重试，先 `rm -f outputs/scoring/minitaur8b_e0_full.*`，否则 `--resume`
会永久跳过这些好会话。转 CUDA 卡用新文件名则无影响。

---

## 5. 已有结果：E2（全 75 任务，每任务 ≤50 人抽样，seed=0，43.7 万 choice）

macro NLL（trial→参与者→任务→总体）：uniform 1.61 / base_rate 1.44 /
sticky 1.41 / **bigram 1.27**。即纯序列统计能从均匀基线砍掉约 0.34 nat。

分任务两极：`cox2017information`、`popov2023intent`、`garcia2023experiential`
等序列自相关强的任务，bigram 吃掉近 1 nat；`wulff2018description`、
`hebart2023things`、`ruggeri2022globalizability` 等独立 trial 任务增益≈0。
后者是检验 Centaur 真本事的干净战场。这套数字是 Centaur/Llama 结果要超越的底线。

---

## 6. git / 未提交状态

原交接所列内存与量化改动已提交到 `3e65c89`，不是当前未提交状态。2026-07-22
的未提交改动修复运行可靠性与跨设备安装说明：只把真实设备 OOM 记失败、空结果
不再生成坏 CSV、bitsandbytes 作为非 macOS 的可选 extra，并让 E0 同时汇总论文兼容的
token NLL 与本项目的分层 macro choice NLL。另增加精确 `--participant` 探针筛选，并在
CUDA 上优先选择 fused SDPA，避免 Windows 回退到平方内存 math attention。

CUDA wheel 只安装在本机 gitignored `.venv`，没有写入仓库的全平台依赖源。Mac 继续
使用 MPS/`--load none`，HPC 继续使用集群自己的 CUDA PyTorch/`--load none`；需要
量化的 CUDA 环境再显式安装 `.[centaur-eval]`。

安装入口按设备分开，避免互相覆盖：

- Mac：`uv pip install -e ".[dev]"`，运行时用 MPS/`--load none`。
- 5060 Ti 新环境：先从 PyTorch CUDA 13.0 wheel 源安装 `torch`，再执行
  `uv pip install -e ".[dev,centaur-eval]"`。CUDA wheel 源只用于这一步，不写进项目锁文件。
- HPC：保留集群模块提供的 CUDA/PyTorch；环境依赖已由集群准备好时，用
  `uv pip install -e . --no-deps` 安装本项目，运行 `preflight.py --load none` 复核。

---

## 7. 下一步（建议顺序）

1. **由用户手动在 CUDA 卡上启动 E0/E3**：
   - HF CLI 已登录为 `Uccelli`，gated 访问 dry-run 通过；数据已下载到
     `data/psych-101-test/prompts_testing_t1.jsonl`（92 MB，gitignored）。
   - 5060 Ti 已用 PyTorch 2.13.0+cu130、bitsandbytes 0.49.2 验证 sm_120/NF4。
     完整预检与最长 E0/E3 session 探针均已通过；全量使用 `--chunk-size 1`。
   - `run_transcript_scoring.py --model marcelbinz/Llama-3.1-Minitaur-8B
     --data data/psych-101-test/prompts_testing_t1.jsonl --load 4bit
     --device cuda --chunk-size 1
     --output outputs/scoring/minitaur8b_e0_full_4bit.csv
     --summary outputs/scoring/minitaur8b_e0_full_4bit_summary.csv`
     （不用 `--max-chars`）。E3 换 `run_window_scoring.py` + `--windows`。
   - HPC 恢复后：`sbatch scripts/smoke_e0_e3.slurm` 烟测过 → `e0_e3_minitaur.slurm`
     全量（fp16 精确数字，非量化）。
2. **E1**：从 E0 的 CSV 按 trial 位置分桶画适应曲线（零算力）。
3. **E5**：从 E0+E3 算上下文增益与交互项，画 Figure B 瀑布（须显式报告交互项）。
4. **E4**：语言表面扰动（逐任务设计，最费手工），可先只对试点任务做。

---

## 8. 起步要读的文件

1. `docs/centaur-eval-design.md`（§12 实验分解，§12.5 引擎待办，§14 结论边界）
2. 本文件
3. `src/mt/evaluation/transcript_scoring.py`（打分核心，重点 `_forward`/`_score_batch`）
4. `scripts/experiments/_common.py`（`load_sessions` 顺序、`load_model`、续跑）
5. `docs/agents/CONVENTIONS.md`（代码风格：80 列、单空行、填满再折行等）
