#!/usr/bin/env bash
#SBATCH --job-name=mt-llm-finetune
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT"

mkdir -p logs/hardware logs/slurm outputs/p500_lora
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

HARDWARE_LOG="logs/hardware/${SLURM_JOB_ID:-local}.csv"
python scripts/monitor_hardware.py --interval 30 --log-file "$HARDWARE_LOG" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT

NPROC="${SLURM_GPUS_ON_NODE:-4}"
NPROC="${NPROC##*:}"

torchrun --standalone --nproc_per_node="$NPROC" scripts/finetune_llm.py \
  --data-source /path/to/p500_psych101.csv \
  --source-kind file \
  --text-column text \
  --max-samples 0 \
  --output-dir outputs/p500_llama8b_lora \
  --model Qmeta-llama/Meta-Llama-3.1-8B \
  --max-steps 300 \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --logging-steps 25 \
  --plot-steps 25 \
  --save-steps 100 \
  --eval-steps 100 \
  --resume-from-checkpoint auto \
  "$@"
