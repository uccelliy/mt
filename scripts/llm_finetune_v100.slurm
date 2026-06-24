#!/usr/bin/bash --login
#SBATCH --job-name=mt-llm-finetune
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu
#SBATCH --qos=normal

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT"

mkdir -p logs/hardware outputs/p500_llama8b_lora
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

module purge || true
module load numlib/cuDNN || true
module load cuda || true

HARDWARE_LOG="logs/hardware/${SLURM_JOB_ID:-local}.csv"
python scripts/monitor_hardware.py --interval 30 --log-file "$HARDWARE_LOG" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT

NPROC="${SLURM_GPUS_ON_NODE:-4}"
NPROC="${NPROC##*:}"

torchrun --standalone --nproc_per_node="$NPROC" scripts/finetune_llm.py \
  --data-source "$HOME/p500-100agents/processed/psych201_prompts_tova_bcs.jsonl" \
  --source-kind file \
  --text-column text \
  --max-samples 0 \
  --output-dir outputs/p500_llama8b_lora \
  --model meta-llama/Meta-Llama-3.1-8B \
  --load-in-4bit \
  --bnb-4bit-compute-dtype float16 \
  --bnb-4bit-quant-type nf4 \
  --bnb-4bit-use-double-quant \
  --seq-len 8192 \
  --max-steps 20 \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --learning-rate 2e-4 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --logging-steps 1 \
  --plot-steps 1 \
  --save-steps 10 \
  --eval-steps 10 \
  --resume-from-checkpoint auto \
  "$@"
