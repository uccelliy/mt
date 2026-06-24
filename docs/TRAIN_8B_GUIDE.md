## 8B 模型两天训练 + 断点续训方案

### 快速总结

✅ **支持断点续训** - 可以安全地在两天时间限制内训练，然后继续  
✅ **数据统计** - 自动计算实际训练 token 数  
✅ **参数化配置** - 命令行参数灵活调整  
✅ **4GPU优化** - 使用 LoRA + 梯度累积充分利用显存  

---

### 工作流

#### 第一次提交训练（Day 0）
```bash
sbatch train_8b_two_days.sh
```

监控进度：
```bash
tail -f train_8b_*.out
tail -f logs/train_8b_*/train.log
```

#### 当时间接近时，模型会自动保存检查点
- 检查点保存在 `./checkpoints/8b_model/checkpoint-*`
- 每 500 步保存一个检查点，最新的 5 个会被保留

#### 提交下一个训练任务（Day 2 或之后）
```bash
sbatch train_8b_two_days.sh --resume-latest
```

脚本会自动：
1. 查找最新的检查点
2. 从该检查点恢复训练状态
3. 继续训练直到 `--train-steps` 达到总步数

---

### 关键配置说明

在 [train_8b_two_days.sh](train_8b_two_days.sh) 中修改这些参数：

```bash
MODEL_NAME="meta-llama/Llama-2-7b-hf"    # ← 改成你的8B模型
N_TRAIN=10000                             # ← 训练样本数
SEQ_LEN=2048                              # ← 序列长度(显存<GPU显存，推荐2048-4096)
BATCH_SIZE=1                              # ← 每GPU批大小(通常1就满显存)
GRADIENT_ACC=32                           # ← 梯度累积(4GPU时用32得到等效128)
TOTAL_STEPS=20000                         # ← 总训练步数
SAVE_STEPS=500                            # ← 多少步保存一次
```

### 性能预估（4×V100 SXM2）

假设 8B 参数模型 + LoRA 微调：

| 配置 | 每秒处理 tokens | 40M tokens 耗时 |
|------|-----------------|-----------------|
| SEQ_LEN=2048, Batch=1, Acc=32 | ~1000-1500 | 7-10 小时 |
| SEQ_LEN=4096, Batch=1, Acc=16 | ~500-800 | 14-20 小时 |

**结论**：
- 40M tokens → 约 10-20 小时 → **可在两天内完成**
- 如果数据更少（10-20M tokens）→ **可在一天内完成**

---

### 数据统计

训练开始时会自动输出：
```
Training samples: 10000, Total tokens: 40,000,000
```

这帮助你理解实际的训练数据量。

---

### 调试和监控

**查看最新日志**：
```bash
ls -ltr logs/train_8b_*/
tail -f logs/train_8b_[最新日期]/train.log
```

**查看检查点**：
```bash
ls -ltr checkpoints/8b_model/
du -sh checkpoints/8b_model/checkpoint-*
```

**手动恢复训练**：
```bash
# 从特定检查点恢复
sbatch train_8b_two_days.sh --resume-from-checkpoint ./checkpoints/8b_model/checkpoint-5000
```

---

### 常见问题

**Q: 如何确保断点续训时学习率正确？**  
A: HuggingFace Trainer 会自动从检查点恢复学习率调度，不需要额外配置。

**Q: 时间不够用，能否更改时间限制？**  
A: 修改 `train_8b_two_days.sh` 中的 `#SBATCH --time=2-00:00:00`。例如 3 天改为 `3-00:00:00`。

**Q: 显存不足怎么办？**  
A: 减小 `GRADIENT_ACC` 或 `SEQ_LEN`，或者启用 QLoRA（更省显存）。

**Q: 如何提前终止并保存？**  
A: 发送 SIGTERM（`scancel <jobid>`），模型会保存最后一个检查点。

---

### 修改的代码位置

主要改动在 [mt/src/mt/models/llm/finetuning.py](../mt/src/mt/models/llm/finetuning.py)：
- ✅ 添加 `parse_args()` 函数支持命令行参数
- ✅ 修改 `main()` 函数支持 `--resume-from-checkpoint`
- ✅ 数据统计和日志打印优化

---

### 文件清单

- [train_8b_two_days.sh](train_8b_two_days.sh) - Slurm 提交脚本
- [mt/src/mt/models/llm/finetuning.py](../mt/src/mt/models/llm/finetuning.py) - 改进的训练脚本
- 检查点目录: `./checkpoints/8b_model/`
- 日志目录: `./logs/train_8b_YYYYMMDD/`
