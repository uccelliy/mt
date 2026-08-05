#!/usr/bin/bash --login
#SBATCH --job-name=mt-score
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
### A whole iris GPU node: 4 V100 and all 28 cores, per ULHPC's rule of
### dedicating 1/4 of the node's cores to each GPU (4 x 7 = 28).
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --gpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=1-23:00:00
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --signal=B:USR1@600
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90

# Score ONE model over the whole test set, four processes on one node, one
# V100 each, every process taking every 4th session into its own shard CSV.
# Which model is a submit-time argument, so the roster is a list of sbatch
# lines rather than a file per model -- see scripts/submit_roster.sh.
#
#     sbatch --export=ALL,MT_MODEL=<hf-id>,MT_TAG=<short-name> \
#            scripts/score_model.slurm
#
# Each shard resumes from its own CSV, so resubmitting the identical line
# after a timeout continues where it stopped. --time deliberately leaves an
# hour under the gpu partition's 2-day cap, and merging is a SEPARATE job
# (scripts/merge_shards.slurm) so a timeout never costs you the merge.
#
# Variables:
#   MT_MODEL     required, HF id
#   MT_TAG       required, short name used in output paths
#   MT_ADAPTER   optional LoRA adapter, loaded onto the quantized base
#   MT_LOAD      4bit (default) | 8bit | none
#   MT_LIMIT     bound the session count, for the resume test
#   MT_NGPU      default 4
#   MT_SMOKE     if set, score only MT_SMOKE sessions on one GPU and stop --
#                run this ONCE per new model before the full grid
#
# A model too big for a 16 GB card needs the 32 GB nodes, which are only
# iris-[191-196] and therefore queue slowly:
#     sbatch --constraint=volta32 --export=ALL,MT_MODEL=...,MT_TAG=...
#
# The four shards are plain background processes rather than srun steps or
# GNU parallel: ULHPC asks for those above ~10 allocations or ~1000 steps
# per minute, and this is four long-lived processes on one node.

set -euo pipefail

source scripts/hpc_env.sh

MODEL="${MT_MODEL:?set MT_MODEL to an HF model id}"
TAG="${MT_TAG:?set MT_TAG to a short name used in output paths}"
DATA="data/psych-101-test/prompts_testing_t1.jsonl"
LOAD="${MT_LOAD:-4bit}"
NGPU="${MT_NGPU:-4}"

ADAPTER_ARGS=()
if [ -n "${MT_ADAPTER:-}" ]; then
    ADAPTER_ARGS=(--adapter "$MT_ADAPTER")
fi

# Progress emails are opt-in and only the shard-0 process sends them, so a
# data-parallel run does not deliver four copies of every threshold.
NOTIFY_EMAIL="${MT_NOTIFY_EMAIL:-${SBATCH_MAIL_USER:-}}"
NOTIFY_ARGS=()
if [ -n "$NOTIFY_EMAIL" ]; then
    NOTIFY_ARGS=(--notify-email "$NOTIFY_EMAIL"
                 --notify-label "${MT_NOTIFY_LABEL:-$TAG}")
fi

STEM="outputs/runs/${TAG}/predictions"
mkdir -p "outputs/runs/${TAG}"

# load_sessions applies --participants before --shard, so the shards stay
# disjoint and reproducible under a bound.
LIMIT_ARGS=()
if [ -n "${MT_LIMIT:-}" ]; then
    LIMIT_ARGS=(--participants "$MT_LIMIT")
    echo "=== bounded run: --participants $MT_LIMIT ==="
fi

report_progress() {
    echo "=== progress at $(date -Is) ==="
    for path in "${STEM}"_shard*.csv; do
        [ -f "$path" ] || continue
        echo "  $path: $(($(wc -l < "$path") - 1)) rows"
    done
}

# SLURM sends USR1 ten minutes before the wall clock runs out; log where
# each shard got to so the resubmission has something to check against.
# The shards keep running until SLURM actually kills them.
MT_WINDING_DOWN=0
trap 'MT_WINDING_DOWN=1; echo "=== USR1: wall clock nearly up ==="; report_progress' USR1

wait_for_shards() {
    local status=0
    for pid in "$@"; do
        wait "$pid" || status=1
    done
    if [ "$status" -ne 0 ] && [ "$MT_WINDING_DOWN" -eq 0 ]; then
        echo "a shard exited non-zero" >&2
        return 1
    fi
    return 0
}

echo "=== model: $MODEL  (tag $TAG, load $LOAD) ==="
echo "=== preflight ==="
python scripts/experiments/preflight.py --model "$MODEL" --data "$DATA" \
    --load "$LOAD"

# One cheap pass on a handful of sessions. Every model new to this repo gets
# it before the full grid: an architecture the scorer cannot unwrap, or a
# tokenizer that puts the marked spans somewhere unexpected, shows up here in
# minutes instead of after a day of queueing.
if [ -n "${MT_SMOKE:-}" ]; then
    echo "=== SMOKE: $MT_SMOKE sessions on one GPU, then stop ==="
    CUDA_VISIBLE_DEVICES=0 python \
        scripts/experiments/run_transcript_scoring.py \
        --model "$MODEL" \
        ${ADAPTER_ARGS[@]+"${ADAPTER_ARGS[@]}"} \
        --data "$DATA" \
        --participants "$MT_SMOKE" \
        --load "$LOAD" \
        --dtype fp16 \
        --chunk-size 8 \
        --batch-tokens 8192 \
        --device cuda \
        --output "outputs/runs/${TAG}/smoke.csv" \
        --summary "outputs/runs/${TAG}/smoke_summary.csv"
    echo "=== smoke done; inspect outputs/runs/${TAG}/smoke.csv ==="
    exit 0
fi

echo "=== full-context scoring, ${NGPU}-way data parallel ==="
mt_watch_memory "${STEM}.mem"
pids=()
for i in $(seq 0 $((NGPU - 1))); do
    CUDA_VISIBLE_DEVICES="$i" python \
        scripts/experiments/run_transcript_scoring.py \
        --model "$MODEL" \
        ${ADAPTER_ARGS[@]+"${ADAPTER_ARGS[@]}"} \
        --data "$DATA" \
        ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"} \
        --shard "$i/$NGPU" \
        --resume \
        --load "$LOAD" \
        --dtype fp16 \
        --chunk-size 8 \
        --batch-tokens 8192 \
        --device cuda \
        ${NOTIFY_ARGS[@]+"${NOTIFY_ARGS[@]}"} \
        --output "${STEM}_shard$i.csv" &
    pids+=($!)
done
wait_for_shards "${pids[@]}"
mt_report_memory "${STEM}.mem"
report_progress

echo "=== done: merge with scripts/merge_shards.slurm ==="
