#!/usr/bin/bash --login
#SBATCH --job-name=mt-centaur-smoke
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
### One GPU, and 1/4 of the node's 28 cores to manage it, per ULHPC's
### AI/DL launcher template. iris GPU nodes are 2 sockets x 14 cores.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --partition=gpu
#SBATCH --qos=normal

# L1 + L2 of docs/Server test design.md §5: prove the cluster reproduces
# numbers we already trust, and find the real VRAM ceiling.
#
#   L1  kool2016when/exp1.csv (20 sessions) vs the local baseline
#   L2  the longest session in the whole test set (xiong2023neural p28,
#       53k tokens) — the actual memory ceiling, and far cheaper than
#       scoring a whole experiment
#
# Both arms are compared against outputs/scoring/minitaur8b_e0_full_4bit.csv
# with scripts/experiments/compare_scoring.py.
#
# !!! Set MT_MODULES in scripts/hpc_env.sh before the first submission
# !!! (docs/Server test design.md §1.2, rows C9/C10/C11).
#
# NF4 arm (default) — matches every local baseline, fits a 16 GB volta node:
#     sbatch scripts/smoke_e0_e3.slurm
# FP16 arm (design doc §5.4) — 8B fp16 weights alone are 16 GB, so this
# needs a 32 GB card, i.e. iris-[191-196], selected by the volta32 feature:
#     sbatch --constraint=volta32 --export=ALL,MT_LOAD=none \
#            scripts/smoke_e0_e3.slurm

set -euo pipefail

source scripts/hpc_env.sh

MODEL="marcelbinz/Llama-3.1-Minitaur-8B"
DATA="data/psych-101-test/prompts_testing_t1.jsonl"
BASELINE="outputs/scoring/minitaur8b_e0_full_4bit.csv"
LOAD="${MT_LOAD:-4bit}"

# The criterion is per-experiment mean NLL, not per-row. Measured on
# 2026-08-01: cluster NF4 vs the local NF4 baseline agrees to 6e-4 on the
# aggregate while individual rows differ by ~2e-2, because Volta has
# neither FlashAttention nor cuDNN attention and falls back to a different
# fp16 kernel. FP16-vs-NF4 is a genuinely different quantity, so that arm
# only screens for gross breakage.
if [ "$LOAD" = "none" ]; then
    TOLERANCE=0.05
else
    TOLERANCE=5e-3
fi
ROW_TOLERANCE=0.25
MIN_CORRELATION=0.99

STEM="outputs/scoring/hpc_smoke_${LOAD}"
mkdir -p outputs/scoring

echo "=== load=$LOAD  per-experiment tol=$TOLERANCE  row tol=$ROW_TOLERANCE  r>=$MIN_CORRELATION ==="

echo "=== preflight ==="
python scripts/experiments/preflight.py --model "$MODEL" --data "$DATA" \
    --load "$LOAD"

# A fresh stem every run: --resume permanently skips sessions listed in a
# stale .failed.csv, which is how a bad local run once poisoned a rerun.
rm -f "${STEM}_l1.csv" "${STEM}_l1".*.csv
rm -f "${STEM}_l2.csv" "${STEM}_l2".*.csv

echo "=== L1: kool2016when/exp1.csv, 20 sessions ==="
mt_watch_memory "${STEM}_l1.mem"
time python scripts/experiments/run_transcript_scoring.py \
    --model "$MODEL" \
    --data "$DATA" \
    --experiment "kool2016when/exp1.csv" \
    --load "$LOAD" \
    --chunk-size 8 \
    --batch-tokens 16384 \
    --device cuda \
    --output "${STEM}_l1.csv"
mt_report_memory "${STEM}_l1.mem"

echo "=== L1 compare against $BASELINE ==="
python scripts/experiments/compare_scoring.py \
    --reference "$BASELINE" \
    --candidate "${STEM}_l1.csv" \
    --tolerance "$TOLERANCE" \
    --row-tolerance "$ROW_TOLERANCE" \
    --min-correlation "$MIN_CORRELATION"

echo "=== L2: longest session in the test set (xiong2023neural p28) ==="
mt_watch_memory "${STEM}_l2.mem"
time python scripts/experiments/run_transcript_scoring.py \
    --model "$MODEL" \
    --data "$DATA" \
    --experiment "xiong2023neural/exp1.csv" \
    --participant 28 \
    --load "$LOAD" \
    --chunk-size 1 \
    --batch-tokens 16384 \
    --device cuda \
    --output "${STEM}_l2.csv" \
    --summary "${STEM}_l2_summary.csv"
mt_report_memory "${STEM}_l2.mem"

echo "=== L2 compare (local NF4 probe reported paper_token_nll 0.472271) ==="
python scripts/experiments/compare_scoring.py \
    --reference "$BASELINE" \
    --candidate "${STEM}_l2.csv" \
    --tolerance "$TOLERANCE" \
    --row-tolerance "$ROW_TOLERANCE" \
    --min-correlation "$MIN_CORRELATION"
cat "${STEM}_l2_summary.csv"

echo "=== E3 shape check (3 windows, same experiment) ==="
rm -f "${STEM}_e3.csv" "${STEM}_e3".*.csv
time python scripts/experiments/run_window_scoring.py \
    --model "$MODEL" \
    --data "$DATA" \
    --experiment "kool2016when/exp1.csv" \
    --load "$LOAD" \
    --windows 0,5,full \
    --num-positions 5 \
    --batch-tokens 16384 \
    --device cuda \
    --output "${STEM}_e3.csv"

echo "=== smoke passed ==="
