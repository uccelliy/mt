#!/usr/bin/bash --login
#SBATCH --job-name=mt-centaur-eval
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

### Slurm native notifications. Verified working on ULHPC 2026-08-03:
### BEGIN/END/FAIL and compute-node mail all deliver to @uni.lu addresses.
### External delivery (Gmail) is accepted by the relay but never arrives, so
### use the university address. TIME_LIMIT_90 fires at 90% of --time, which
### says nothing about how far the analysis got -- that is what the in-process
### progress notifier is for. Supply the address at submit time:
###     sbatch --mail-user=name.surname@uni.lu scripts/<job>.slurm
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90

# E0 (full-context) then E3 (window truncation) over the test set, four
# processes on one node, one V100 each, every process scoring every 4th
# session into its own shard CSV.
#
# Each shard resumes from its own CSV, so resubmitting this exact script
# after a timeout continues where it stopped. --time deliberately leaves an
# hour under the gpu partition's 2-day cap, and merging is a SEPARATE job
# (scripts/merge_shards.slurm) so a timeout never costs you the merge.
#
# The 2-day cap is hard for us — the long QOS (iris-gpu-long) is not
# available on this account. Anything longer has to be completed by
# resubmitting this script until every shard reports no pending sessions.
#
# !!! Set MT_MODULES in scripts/hpc_env.sh before the first submission
# !!! (docs/Server test design.md §1.2, rows C9/C10/C11).
#
# Prerequisites (see design doc §3.3):
#   - data/psych-101-test/prompts_testing_t1.jsonl
#   - Minitaur-8B under $HF_HOME (download on a login node, or rsync -a
#     the models--marcelbinz--Llama-3.1-Minitaur-8B tree from a machine
#     that has it; plain rsync -r breaks the blobs/ symlinks)
#
# L3 of the design doc (bounded resume test, ~1 h) — run this BEFORE any
# real job, cancel it halfway, then resubmit the same line and check that
# the shards keep growing:
#     sbatch --time=01:00:00 --export=ALL,MT_LIMIT=200 \
#            scripts/e0_e3_minitaur.slurm
#
# FP16 instead of NF4 needs 32 GB cards (8B fp16 weights alone are 16 GB),
# i.e. iris-[191-196], selected by the volta32 feature:
#     sbatch --constraint=volta32 --export=ALL,MT_LOAD=none \
#            scripts/e0_e3_minitaur.slurm
#
# The four shards are plain background processes rather than srun steps or
# GNU parallel: ULHPC asks for those above ~10 allocations or ~1000 steps
# per minute, and this is four long-lived processes on one node.

set -euo pipefail

source scripts/hpc_env.sh

MODEL="marcelbinz/Llama-3.1-Minitaur-8B"
DATA="data/psych-101-test/prompts_testing_t1.jsonl"
LOAD="${MT_LOAD:-4bit}"
NGPU="${MT_NGPU:-4}"
# Progress emails are opt-in and only the shard-0 process sends them, so a
# data-parallel run does not deliver four copies of every threshold.
# sbatch reads SBATCH_MAIL_USER at submit time to fill in --mail-user, and
# --export=ALL (its default) also carries that same variable into the job --
# so one exported address drives both channels and nothing has to be typed
# twice. MT_NOTIFY_EMAIL only exists to send progress somewhere else than
# the Slurm state mail, which is rarely what you want.
NOTIFY_EMAIL="${MT_NOTIFY_EMAIL:-${SBATCH_MAIL_USER:-}}"
NOTIFY_ARGS=()
if [ -n "$NOTIFY_EMAIL" ]; then
    NOTIFY_ARGS=(--notify-email "$NOTIFY_EMAIL"
                 --notify-label "${MT_NOTIFY_LABEL:-$SLURM_JOB_NAME}")
fi

E0_STEM="outputs/scoring/hpc_e0_minitaur8b_${LOAD}"
E3_STEM="outputs/scoring/hpc_e3_minitaur8b_e0grid5_${LOAD}"

# MT_LIMIT bounds the session count for the L3 resume test. load_sessions
# applies --participants before --shard, so the four shards stay disjoint
# and reproducible.
LIMIT_ARGS=()
if [ -n "${MT_LIMIT:-}" ]; then
    LIMIT_ARGS=(--participants "$MT_LIMIT")
    echo "=== bounded run: --participants $MT_LIMIT ==="
fi

report_progress() {
    echo "=== progress at $(date -Is) ==="
    for path in "${E0_STEM}"_shard*.csv "${E3_STEM}"_shard*.csv; do
        [ -f "$path" ] || continue
        echo "  $path: $(($(wc -l < "$path") - 1)) rows"
    done
}

# SLURM sends USR1 ten minutes before the wall clock runs out; log where
# each shard got to so the resubmission has something to check against.
# The shards keep running until SLURM actually kills them — the handler
# only records, it must not tear the job down early.
MT_WINDING_DOWN=0
trap 'MT_WINDING_DOWN=1; echo "=== USR1: wall clock nearly up ==="; report_progress' USR1

# Waiting on each PID individually distinguishes a genuinely failed shard
# from `wait` merely being interrupted by the USR1 above.
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

echo "=== preflight ==="
python scripts/experiments/preflight.py --model "$MODEL" --data "$DATA" \
    --load "$LOAD"

mkdir -p outputs/scoring

echo "=== E0: full-context scoring, ${NGPU}-way data parallel ==="
mt_watch_memory "${E0_STEM}.mem"
pids=()
for i in $(seq 0 $((NGPU - 1))); do
    CUDA_VISIBLE_DEVICES="$i" python \
        scripts/experiments/run_transcript_scoring.py \
        --model "$MODEL" \
        --data "$DATA" \
        ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"} \
        --shard "$i/$NGPU" \
        --resume \
        --load "$LOAD" \
        --chunk-size 8 \
        --batch-tokens 16384 \
        --device cuda \
        ${NOTIFY_ARGS[@]+"${NOTIFY_ARGS[@]}"} \
        --output "${E0_STEM}_shard$i.csv" &
    pids+=($!)
done
wait_for_shards "${pids[@]}"
mt_report_memory "${E0_STEM}.mem"
report_progress

echo "=== E3: window truncation, ${NGPU}-way data parallel ==="
# --num-positions must be 5 for the default e0-informed grid, and 5 is what
# every completed E3 run used; the runner rejects any other value.
mt_watch_memory "${E3_STEM}.mem"
pids=()
for i in $(seq 0 $((NGPU - 1))); do
    CUDA_VISIBLE_DEVICES="$i" python \
        scripts/experiments/run_window_scoring.py \
        --model "$MODEL" \
        --data "$DATA" \
        ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"} \
        --shard "$i/$NGPU" \
        --resume \
        --load "$LOAD" \
        --windows 0,1,2,5,10,20,full \
        --num-positions 5 \
        --position-grid e0-informed \
        --batch-tokens 16384 \
        --device cuda \
        ${NOTIFY_ARGS[@]+"${NOTIFY_ARGS[@]}"} \
        --output "${E3_STEM}_shard$i.csv" &
    pids+=($!)
done
wait_for_shards "${pids[@]}"
mt_report_memory "${E3_STEM}.mem"
report_progress

echo "=== done; merge with: ==="
echo "  sbatch --export=ALL,MT_STEMS=\"$E0_STEM $E3_STEM\" scripts/merge_shards.slurm"
