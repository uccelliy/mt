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
# `full` but differed by 2-4x on w=5/10/20.
#
# `full` was read as a padding control, but it is confounded: it is also the
# only condition with zero NaN. The conditions that disagreed are exactly the
# conditions that carried NaN, and the per-condition summary those numbers came
# from divides a NaN-skipping numerator by a full denominator -- so a run with
# more NaN reports a lower NLL whether or not padding changed anything. The
# comparison below therefore joins the two runs per choice instead.
#
# The six completed Llama runs used the same packing, so the question that
# matters is whether Llama disagrees per choice on rows that are finite in both
# runs. If it does, their E3 arm needs re-examining; if it does not, the
# completed runs stand and the gemma effect was the NaN, not the packing.
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

# The two runs score exactly the same choices, so the question is per choice.
# A per-condition `nll.sum() / num_tokens.sum()` cannot answer it: pandas drops
# NaN from the numerator but keeps those rows' tokens in the denominator, so a
# run with more NaN reports a lower NLL for that reason alone -- which is the
# shape the gemma smoke showed, and is not evidence about padding.
KEY = ["condition", "experiment", "participant", "target_index", "choice_index"]

def load(path):
    frame = pd.read_csv(path, low_memory=False)
    return frame.set_index(KEY).sort_index()

for model in sys.argv[2:]:
    safe = model.replace("/", "_")
    packed = load(root / f"{safe}-packed" / "predictions.csv")
    alone = load(root / f"{safe}-alone" / "predictions.csv")
    print(f"\n----- {model}")
    if not packed.index.equals(alone.index):
        print("the two runs did not score the same choices; compare manually")
        continue

    joined = pd.DataFrame({
        "packed": packed["nll"],
        "alone": alone["nll"],
        "condition": packed.index.get_level_values("condition"),
    })
    joined["nan_packed"] = joined["packed"].isna()
    joined["nan_alone"] = joined["alone"].isna()
    both = joined[~joined["nan_packed"] & ~joined["nan_alone"]].copy()
    both["abs_diff"] = (both["packed"] - both["alone"]).abs()

    table = both.groupby("condition").agg(
        rows=("abs_diff", "size"),
        max_abs_diff=("abs_diff", "max"),
        mean_abs_diff=("abs_diff", "mean"),
    )
    holes = joined.groupby("condition").agg(
        nan_packed=("nan_packed", "sum"),
        nan_alone=("nan_alone", "sum"),
    )
    # rows NaN in one run but not the other: these, not the arithmetic, are
    # what made the two runs disagree in the gemma smoke
    joined["nan_disagree"] = joined["nan_packed"] != joined["nan_alone"]
    holes["nan_disagree"] = joined.groupby("condition")["nan_disagree"].sum()
    print(table.join(holes).to_string())

    worst = both["abs_diff"].max()
    print(f"\nlargest per-choice |packed - alone|: {worst:.3e} nat")
    print(f"choices NaN in exactly one run: {int(joined['nan_disagree'].sum())}")
    print("verdict: right-padding a causal LM is mathematically a no-op, so a "
          "clean result is ~1e-3 nat or less (fp16 reduction order). Anything "
          "larger on a choice scored in both runs is a real batching effect.")
PY

echo "=== artifacts in $ROOT ==="
