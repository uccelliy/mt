#!/usr/bin/bash --login
#SBATCH --job-name=mt-batching-probe
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --constraint=volta16
#SBATCH --mail-type=END,FAIL

# Does batching change E3 window scores?
#
# On gemma-4-E2B the same session scored twice -- once with the protocol's
# --batch-tokens 8192, once with every prompt alone -- agreed bit-for-bit on
# `full` but differed by 2-4x on w=5/10/20. `full` is the one condition whose
# prompt exceeds the batch budget and therefore never shares a batch, which
# points at padding rather than at the model.
#
# The six completed Llama runs used the same packing, so the question that
# matters is whether Llama shows the same asymmetry. If it does, their E3 arm
# needs re-examining; if it does not, this is Gemma-specific (sliding-window
# attention) and the completed runs stand.
#
#     sbatch scripts/diagnose_window_batching.slurm [model ...]

set -uo pipefail

source scripts/hpc_env.sh

MODELS=("$@")
if [ "${#MODELS[@]}" -eq 0 ]; then
    MODELS=("meta-llama/Llama-3.1-8B")
fi

DATA="data/psych-101-test/prompts_testing_t1.jsonl"
# The session with the most NaN in the gemma smoke, so any effect is loud.
EXPERIMENT="frey2017risk/exp1.csv"
PARTICIPANT="280"
ROOT="/scratch/users/${USER}/diagnostics/batching-${SLURM_JOB_ID:-manual}"
mkdir -p "$ROOT"

score() {
    local model="$1" batch_tokens="$2" out="$3"
    OMP_NUM_THREADS=7 python scripts/experiments/run_window_scoring.py \
        --model "$model" \
        --data "$DATA" \
        --dataset psych-101-test \
        --condition-prefix e3 \
        --choice-readout greedy-unconstrained-1token \
        --experiment "$EXPERIMENT" \
        --participant "$PARTICIPANT" \
        --windows 0,1,2,5,10,20,full \
        --num-positions 5 \
        --position-grid e0-informed \
        --load 4bit \
        --dtype fp16 \
        --batch-tokens "$batch_tokens" \
        --top-k 20 \
        --max-options 256 \
        --device cuda \
        --output-dir "$out"
}

for model in "${MODELS[@]}"; do
    safe="${model//\//_}"
    echo "=== $model: packed (--batch-tokens 8192, the protocol value) ==="
    score "$model" 8192 "${ROOT}/${safe}-packed"
    # Smaller than the shortest prompt, so the packer flushes after every one
    # and no sequence is ever padded to a neighbour's width.
    echo "=== $model: unpadded (--batch-tokens 1, one prompt per forward) ==="
    score "$model" 1 "${ROOT}/${safe}-alone"
done

echo
echo "=== comparison ==="
python - "$ROOT" "${MODELS[@]}" <<'PY'
from pathlib import Path
import sys

import pandas as pd

root = Path(sys.argv[1])

def per_condition(path):
    frame = pd.read_csv(path, low_memory=False)
    grouped = frame.groupby("condition")
    return pd.DataFrame({
        "token_nll": grouped.apply(
            lambda g: g["nll"].sum() / g["num_tokens"].sum(), include_groups=False),
        "nan": grouped["nll"].apply(lambda s: int(s.isna().sum())),
        "rows": grouped.size(),
    })

for model in sys.argv[2:]:
    safe = model.replace("/", "_")
    packed = per_condition(root / f"{safe}-packed" / "predictions.csv")
    alone = per_condition(root / f"{safe}-alone" / "predictions.csv")
    table = packed.join(alone, lsuffix="_packed", rsuffix="_alone")
    table["abs_diff"] = (table["token_nll_packed"]
                         - table["token_nll_alone"]).abs()
    table["ratio"] = (table["token_nll_packed"]
                      / table["token_nll_alone"]).round(3)
    print(f"\n----- {model}")
    print(table[["token_nll_packed", "token_nll_alone", "abs_diff", "ratio",
                 "nan_packed", "nan_alone", "rows_packed"]].to_string())
    worst = table["abs_diff"].max()
    print(f"largest absolute difference across conditions: {worst:.6f} nat")
    print("full is the control: its prompt never shares a batch, so it should "
          "be identical in both runs")
PY

echo "=== artifacts in $ROOT ==="
