#!/usr/bin/env bash
# 两天内训练8B模型 + 断点续训脚本
# 使用方式:
#   第一次:   sbatch train_8b_two_days.sh
#   继续:     sbatch train_8b_two_days.sh --resume-latest

set -euo pipefail

#SBATCH --job-name=train_8b_lora
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# 4张V100 SXM2，两天时间
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=4
#SBATCH --time=2-00:00:00              # 2天

#SBATCH --partition=gpu
#SBATCH --qos=normal

# 设置基本环境
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

module purge || true
module load numlib/cuDNN || true
module load cuda || true

# 日志目录
LOG_DIR="./logs/train_8b_$(date +%Y%m%d)"
CHECKPOINT_DIR="./checkpoints/8b_model"
mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"

echo "=========================================="
echo "Training 8B Model with LoRA on 4x V100"
echo "=========================================="
echo "Time: $(date)"
echo "GPU Count: $(nvidia-smi -L | wc -l)"
echo "Output Dir: $LOG_DIR"
echo "Checkpoint Dir: $CHECKPOINT_DIR"
echo ""

# 训练参数配置
MODEL_NAME="meta-llama/Llama-2-7b-hf"    # 改成你的8B模型
N_TRAIN=10000                             # 训练样本数
SEQ_LEN=2048                              # 序列长度（相比4000更省显存）
BATCH_SIZE=1                              # 每GPU批大小
GRADIENT_ACC=32                           # 梯度累积步数（4GPU × batch1 × acc32 = 128 effective）
LR=5e-5
TOTAL_STEPS=20000                         # 总步数（可根据需要调整）
SAVE_STEPS=500                            # 每500步保存检查点

# 判断是否需要恢复训练
RESUME_CHECKPOINT=""
if [[ "${1:-}" == "--resume-latest" ]]; then
    echo "Looking for latest checkpoint..."
    if [[ -d "$CHECKPOINT_DIR" && $(ls -la "$CHECKPOINT_DIR" | grep checkpoint | wc -l) -gt 0 ]]; then
        # 找最新的检查点
        LATEST_CKPT=$(ls -t "$CHECKPOINT_DIR"/checkpoint-* 2>/dev/null | head -1)
        if [[ -n "$LATEST_CKPT" ]]; then
            RESUME_CHECKPOINT="--resume-from-checkpoint $LATEST_CKPT"
            echo "Resuming from: $LATEST_CKPT"
        fi
    fi
fi

# 使用 uv 运行训练（如果使用 uv）
echo "Starting training..."
python -m mt.models.llm.finetuning \
    --model "$MODEL_NAME" \
    --output-dir "$CHECKPOINT_DIR" \
    --train-steps "$TOTAL_STEPS" \
    --n-train "$N_TRAIN" \
    --seq-len "$SEQ_LEN" \
    --batch-size "$BATCH_SIZE" \
    --gradient-acc "$GRADIENT_ACC" \
    --learning-rate "$LR" \
    --save-steps "$SAVE_STEPS" \
    $RESUME_CHECKPOINT \
    2>&1 | tee "$LOG_DIR/train.log"

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Training finished with exit code: $EXIT_CODE"
echo "Time: $(date)"
echo "=========================================="

if [[ $EXIT_CODE -eq 0 ]]; then
    echo "✓ Training completed successfully"
    echo "Model saved to: $CHECKPOINT_DIR"
else
    echo "✗ Training failed or interrupted"
    echo "You can resume with: sbatch train_8b_two_days.sh --resume-latest"
fi

exit $EXIT_CODE
