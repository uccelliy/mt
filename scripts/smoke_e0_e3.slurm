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

# Does this cluster reproduce numbers we already trust? Scores one
# experiment and compares it against a known-good local run.
#
# Everything is overridable, so the same script re-validates the cluster
# after a model or dataset change -- that is the point of it surviving the
# analysis it was first written for:
#
#     sbatch --export=ALL,MT_MODEL=<hf-id>,MT_EXPERIMENT=<task/exp.csv> \
#            scripts/smoke_e0_e3.slurm
#
# With no MT_BASELINE (or a model that has none yet) the comparison is
# skipped and this becomes a plain end-to-end smoke test.
#
# The long-context attention check that used to live here as "L2" now runs
# inside preflight.py, so every job gets it (design doc §5.4b).
#
# FP16 instead of NF4 needs a 32 GB card -- 8B fp16 weights alone are 16 GB
# -- i.e. iris-[191-196], selected by the volta32 feature:
#     sbatch --constraint=volta32 --export=ALL,MT_LOAD=none \
#            scripts/smoke_e0_e3.slurm

set -euo pipefail

source scripts/hpc_env.sh

MODEL="${MT_MODEL:-marcelbinz/Llama-3.1-Minitaur-8B}"
DATA="${MT_DATA:-data/psych-101-test/prompts_testing_t1.jsonl}"
EXPERIMENT="${MT_EXPERIMENT:-kool2016when/exp1.csv}"
BASELINE="${MT_BASELINE:-outputs/scoring/minitaur8b_e0_full_4bit.csv}"
LOAD="${MT_LOAD:-4bit}"

# The criterion is per-experiment mean NLL, not per-row. Measured
# 2026-08-01: cluster NF4 vs the local NF4 baseline agrees to 6e-4 on the
# aggregate while individual rows differ by ~2e-2, because Volta has
# neither FlashAttention nor cuDNN attention and falls back to a different
# fp16 kernel. Those differences are zero-mean and cancel. FP16-vs-NF4 is a
# genuinely different quantity, so that arm only screens for gross breakage.
if [ "$LOAD" = "none" ]; then
    TOLERANCE=0.05
else
    TOLERANCE=5e-3
fi
ROW_TOLERANCE=0.25
MIN_CORRELATION=0.99

STEM="outputs/scoring/hpc_smoke_${LOAD}"
mkdir -p outputs/scoring

echo "=== model=$MODEL experiment=$EXPERIMENT load=$LOAD ==="
echo "=== per-experiment tol=$TOLERANCE row tol=$ROW_TOLERANCE r>=$MIN_CORRELATION ==="

# preflight also verifies that long-context attention stays on a fused
# kernel; on Volta a silent fall back to MATH would ask for tens of GiB
echo "=== preflight ==="
python scripts/experiments/preflight.py --model "$MODEL" --data "$DATA" \
    --load "$LOAD"

# A fresh stem every run: --resume permanently skips sessions listed in a
# stale .failed.csv, which is how a bad local run once poisoned a rerun.
rm -f "${STEM}_e0.csv" "${STEM}_e0".*.csv
rm -f "${STEM}_e3.csv" "${STEM}_e3".*.csv

echo "=== E0: $EXPERIMENT ==="
mt_watch_memory "${STEM}_e0.mem"
time python scripts/experiments/run_transcript_scoring.py \
    --model "$MODEL" \
    --data "$DATA" \
    --experiment "$EXPERIMENT" \
    --load "$LOAD" \
    --chunk-size 8 \
    --batch-tokens 16384 \
    --device cuda \
    --output "${STEM}_e0.csv" \
    --summary "${STEM}_e0_summary.csv"
mt_report_memory "${STEM}_e0.mem"
cat "${STEM}_e0_summary.csv"

if [ -f "$BASELINE" ]; then
    echo "=== compare against $BASELINE ==="
    python scripts/experiments/compare_scoring.py \
        --reference "$BASELINE" \
        --candidate "${STEM}_e0.csv" \
        --tolerance "$TOLERANCE" \
        --row-tolerance "$ROW_TOLERANCE" \
        --min-correlation "$MIN_CORRELATION"
else
    echo "=== no baseline at $BASELINE, skipping comparison ==="
fi

echo "=== E3 shape check (3 windows, same experiment) ==="
time python scripts/experiments/run_window_scoring.py \
    --model "$MODEL" \
    --data "$DATA" \
    --experiment "$EXPERIMENT" \
    --load "$LOAD" \
    --windows 0,5,full \
    --num-positions 5 \
    --batch-tokens 16384 \
    --device cuda \
    --output "${STEM}_e3.csv"

echo "=== smoke passed ==="
