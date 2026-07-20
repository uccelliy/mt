#!/usr/bin/bash --login
#SBATCH --job-name=mt-centaur-smoke
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --qos=normal

# 30-minute smoke test before the real job: preflight, then E0 + E3 on one
# experiment (20 sessions) with the real Minitaur-8B on one V100. Verifies
# the CUDA/fp16/memory path and prints throughput for budget estimates.
# If the cluster has a fast debug/short QOS, switch --qos to queue sooner.

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export HF_HUB_OFFLINE=1

module purge || true
module load numlib/cuDNN || true
module load cuda || true

MODEL="marcelbinz/Llama-3.1-Minitaur-8B"
DATA="data/psych-101-test/prompts_testing_t1.jsonl"
EXPERIMENT="kool2016when/exp1.csv"

echo "=== node info ==="
nvidia-smi
python --version

echo "=== preflight ==="
python scripts/experiments/preflight.py --model "$MODEL" --data "$DATA"

rm -f outputs/scoring/smoke_e0.csv outputs/scoring/smoke_e3.csv

echo "=== smoke E0 (one experiment, timed) ==="
time python scripts/experiments/run_transcript_scoring.py \
  --model "$MODEL" \
  --data "$DATA" \
  --experiment "$EXPERIMENT" \
  --chunk-size 8 \
  --batch-tokens 16384 \
  --device cuda \
  --output outputs/scoring/smoke_e0.csv

echo "=== smoke E3 (one experiment, timed) ==="
time python scripts/experiments/run_window_scoring.py \
  --model "$MODEL" \
  --data "$DATA" \
  --experiment "$EXPERIMENT" \
  --windows 0,5,full \
  --num-positions 5 \
  --batch-tokens 16384 \
  --device cuda \
  --output outputs/scoring/smoke_e3.csv

echo "=== smoke passed ==="
