#!/usr/bin/bash
# Submit one scoring job per model in the v1 roster.
#
#     scripts/submit_roster.sh smoke     # 20 sessions each, run this FIRST
#     scripts/submit_roster.sh full      # the whole test set
#     scripts/submit_roster.sh full gemma4_e2b llama32_1b   # named subset
#
# Every model new to this repo gets the smoke pass before the full grid. It
# costs minutes and catches what static inspection cannot: where this
# tokenizer actually lands the `<<...>>` spans, whether any session overruns
# the context window, and the real memory ceiling per card. Gemma 4's module
# structure was already checked offline -- `_unwrap_base` stops at
# `Gemma4Model` and the hidden-state path engages exactly as it does for
# Llama -- so that particular worry does not need the smoke run.
#
# Set the notification address once, in ~/.bashrc on the cluster:
#     export SBATCH_MAIL_USER=<you>@uni.lu
# sbatch reads it for --mail-user and --export=ALL carries it into the job.

set -euo pipefail

MODE="${1:-}"
if [ "$MODE" != "smoke" ] && [ "$MODE" != "full" ]; then
    echo "usage: $0 {smoke|full} [tag ...]" >&2
    exit 2
fi
shift || true

# tag | HF id | constraint
#
# Constraint follows from 4-bit weight size against the card. volta16 (18
# nodes, quick queue) holds everything here except the 26B mixture: its
# weights alone are ~14.6 GB of a 16 GB card, leaving nothing for activations,
# so it goes to volta32 (6 nodes, slow queue) instead.
ROSTER=(
    "gemma4_e2b|google/gemma-4-E2B-it|volta16"
    "gemma4_e4b|google/gemma-4-E4B-it|volta16"
    "gemma4_26b_a4b|google/gemma-4-26B-A4B-it|volta32"
    "llama32_1b|meta-llama/Llama-3.2-1B|volta16"
    "llama32_1b_instruct|meta-llama/Llama-3.2-1B-Instruct|volta16"
    "llama32_3b|meta-llama/Llama-3.2-3B|volta16"
    "llama32_3b_instruct|meta-llama/Llama-3.2-3B-Instruct|volta16"
)

wanted=("$@")

for entry in "${ROSTER[@]}"; do
    IFS='|' read -r tag model constraint <<< "$entry"

    if [ ${#wanted[@]} -gt 0 ]; then
        found=0
        for name in "${wanted[@]}"; do
            [ "$name" = "$tag" ] && found=1
        done
        [ "$found" -eq 1 ] || continue
    fi

    export_vars="ALL,MT_MODEL=$model,MT_TAG=$tag"
    extra=()
    if [ "$MODE" = "smoke" ]; then
        export_vars="$export_vars,MT_SMOKE=${MT_SMOKE:-20}"
        # a smoke pass uses one GPU for minutes, so ask for one hour
        extra=(--time=01:00:00 --job-name="mt-smoke-$tag")
    else
        extra=(--job-name="mt-score-$tag")
    fi

    echo "sbatch --constraint=$constraint ${extra[*]} $model"
    sbatch --constraint="$constraint" \
           "${extra[@]}" \
           --export="$export_vars" \
           scripts/score_model.slurm
done
