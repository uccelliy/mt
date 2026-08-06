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
# Do not submit this worker directly; use scripts/submit_roster.sh so the
# protocol commit, smoke gate, L3 gate, resources, and output root agree.
#
# Each E3 shard resumes only from atomic per-session commit markers (its CSVs
# are append-only payloads); legacy non-E3 scoring still resumes from its shard
# CSV. Resubmitting the identical line after a timeout therefore continues
# where it stopped. --time deliberately leaves an hour under the gpu
# partition's 2-day cap, and merging is a SEPARATE job
# (scripts/merge_shards.slurm) so a timeout never costs you the merge.
#
# Variables (normally supplied only by scripts/submit_roster.sh):
#   MT_MODEL            required, HF id
#   MT_TAG              required, short name used in output paths
#   MT_ADAPTER          optional LoRA adapter, loaded onto the quantized base
#   MT_LOAD             required formal value: 4bit
#   MT_SUITE            required: full | e3
#   MT_RUN_MODE         required: smoke | l3 | full
#   MT_PROTOCOL_TAG     required, identifies the exact output/readout protocol
#   MT_PROTOCOL_COMMIT  required, commit whose code produced this run
#   MT_RUN_ROOT         required, protocol-isolated directory for this model
#   MT_NGPU             1 for smoke, 4 for l3/full
#   MT_L3_LIMIT         exactly 200 for the bounded resume test
#   MT_L3_ATTEMPT       positive attempt number for l3; written to the run ledger
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
EXPECTED_DATA_SHA256="18b1ab7a61e0288285c888568d449b486f81b7d107397b10fdafb5e9cfc30b60"
LOAD="${MT_LOAD:-4bit}"
RUN_MODE="${MT_RUN_MODE:?submit with scripts/submit_roster.sh}"
SUITE="${MT_SUITE:-full}"
PROTOCOL_TAG="${MT_PROTOCOL_TAG:?set MT_PROTOCOL_TAG}"
PROTOCOL_COMMIT="${MT_PROTOCOL_COMMIT:?set MT_PROTOCOL_COMMIT}"
RUN_ROOT="${MT_RUN_ROOT:?set MT_RUN_ROOT}"
CHOICE_READOUT="greedy-unconstrained-1token"
SMOKE_GATE_VALUE="${PROTOCOL_TAG}|${PROTOCOL_COMMIT}|${CHOICE_READOUT}"

if [ "$LOAD" != "4bit" ]; then
    echo "formal Track P is pinned to 4bit, got $LOAD" >&2
    exit 2
fi
if [ "${MT_CHOICE_READOUT:-$CHOICE_READOUT}" != "$CHOICE_READOUT" ]; then
    echo "unexpected MT_CHOICE_READOUT=${MT_CHOICE_READOUT:-unset}" >&2
    exit 2
fi
if [ "$SUITE" != "full" ] && [ "$SUITE" != "e3" ]; then
    echo "MT_SUITE must be full or e3, got $SUITE" >&2
    exit 2
fi
if [ "$SUITE" = "e3" ]; then
    SMOKE_GATE_VALUE="${SMOKE_GATE_VALUE}|e3"
fi

actual_commit=$(git rev-parse HEAD)
if [ "$actual_commit" != "$PROTOCOL_COMMIT" ]; then
    echo "checkout moved after submission: expected $PROTOCOL_COMMIT, got $actual_commit" >&2
    exit 2
fi
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "compute-node checkout is dirty; refusing an unversioned formal run" >&2
    exit 2
fi
actual_data_sha256=$(sha256sum "$DATA" | awk '{print $1}')
if [ "$actual_data_sha256" != "$EXPECTED_DATA_SHA256" ]; then
    echo "data checksum mismatch: expected $EXPECTED_DATA_SHA256, got $actual_data_sha256" >&2
    exit 2
fi

L3_ATTEMPT=""
case "$RUN_MODE" in
    smoke)
        NGPU=1
        # One deterministic session from every experiment exercises all 75
        # prompt families, including a ~40k-token xiong session.
        EXPECTED_SESSIONS=75
        RUN_DIR="${RUN_ROOT}/smoke-${SLURM_JOB_ID:-manual}"
        RESUME_ARGS=()
        LIMIT_ARGS=(--max-participants 1 --seed 0)
        ;;
    l3)
        NGPU=4
        EXPECTED_SESSIONS="${MT_L3_LIMIT:-200}"
        if [ "$EXPECTED_SESSIONS" -ne 200 ]; then
            echo "L3 is a fixed 200-session protocol, got $EXPECTED_SESSIONS" >&2
            exit 2
        fi
        L3_ATTEMPT="${MT_L3_ATTEMPT:?L3 requires MT_L3_ATTEMPT}"
        if [[ ! "$L3_ATTEMPT" =~ ^[1-9][0-9]*$ ]]; then
            echo "MT_L3_ATTEMPT must be a positive integer, got $L3_ATTEMPT" >&2
            exit 2
        fi
        RUN_DIR="${RUN_ROOT}/l3"
        RESUME_ARGS=(--resume)
        LIMIT_ARGS=(--participants "$EXPECTED_SESSIONS")
        ;;
    full)
        NGPU=4
        EXPECTED_SESSIONS=6561
        RUN_DIR="${RUN_ROOT}/full"
        RESUME_ARGS=(--resume)
        LIMIT_ARGS=()
        ;;
    *)
        echo "MT_RUN_MODE must be smoke, l3, or full; got $RUN_MODE" >&2
        exit 2
        ;;
esac

if [ "${MT_NGPU:-$NGPU}" -ne "$NGPU" ]; then
    echo "$RUN_MODE requires MT_NGPU=$NGPU" >&2
    exit 2
fi

# Four scoring processes share the 28 allocated cores; smoke requests seven
# cores for its single process. Explicitly pin every Python worker to seven
# threads so hpc_env.sh's job-wide OMP value cannot oversubscribe the node.
export OMP_NUM_THREADS=7
export MKL_NUM_THREADS=7

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

PREDICTIONS_STEM="${RUN_DIR}/predictions"
mkdir -p "$RUN_DIR"

LOCK_HOST=$(hostname)
if [ -n "${SLURM_JOB_ID:-}" ]; then
    LOCK_TOKEN="$SLURM_JOB_ID"
else
    LOCK_TOKEN="manual:${LOCK_HOST}:$$"
fi
SCORE_LOCK="${RUN_DIR}/.score.lock"
MERGE_LOCK="${RUN_DIR}/.merge.lock"

query_live_slurm_job() {
    local job_id="$1"
    local queue
    if ! queue=$(squeue -h -o '%A|%T'); then
        return 1
    fi
    awk -F '|' -v wanted="$job_id" '
        {
            id = $1
            state = $2
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", id)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", state)
            if (id == wanted) {
                print state
                exit
            }
        }
    ' <<< "$queue"
}

query_lock_owner_state() {
    local owner="$1"
    if [[ "$owner" =~ ^[0-9]+$ ]]; then
        query_live_slurm_job "$owner"
        return
    fi
    if [[ "$owner" =~ ^manual:([A-Za-z0-9._-]+):([0-9]+)$ ]]; then
        local owner_host="${BASH_REMATCH[1]}"
        local owner_pid="${BASH_REMATCH[2]}"
        if [ "$owner_host" != "$LOCK_HOST" ]; then
            echo "manual lock belongs to host $owner_host, not $LOCK_HOST" >&2
            return 1
        fi
        if kill -0 "$owner_pid" 2>/dev/null; then
            printf 'MANUAL\n'
        fi
        return 0
    fi
    echo "lock owner has an unsupported identity: ${owner:-empty}" >&2
    return 1
}

archive_stale_lock() {
    local lock_dir="$1"
    local label="$2"
    local owner_file="${lock_dir}/owner"
    local owner=""
    if [ -f "$owner_file" ]; then
        owner=$(tr -d '\r\n' < "$owner_file")
    fi
    local state
    if ! state=$(query_lock_owner_state "$owner"); then
        echo "cannot safely query owner $owner for $label lock $lock_dir" >&2
        return 1
    fi
    if [ -n "$state" ]; then
        echo "$label job $owner is still active ($state): $lock_dir" >&2
        return 1
    fi
    local archived="${lock_dir}.stale.${owner}.${LOCK_TOKEN}"
    if mv "$lock_dir" "$archived" 2>/dev/null; then
        echo "archived stale $label lock from job $owner as $archived"
    else
        echo "$label lock changed while it was being checked: $lock_dir" >&2
        return 1
    fi
}

acquire_score_lock() {
    while ! mkdir "$SCORE_LOCK" 2>/dev/null; do
        archive_stale_lock "$SCORE_LOCK" "score" || return 1
    done
    printf '%s\n' "$LOCK_TOKEN" > "${SCORE_LOCK}/owner"
    if [ -d "$MERGE_LOCK" ]; then
        archive_stale_lock "$MERGE_LOCK" "merge" || return 1
    fi
}

release_score_lock() {
    if [ -f "${SCORE_LOCK}/owner" ] && \
       [ "$(tr -d '\r\n' < "${SCORE_LOCK}/owner")" = "$LOCK_TOKEN" ]; then
        unlink "${SCORE_LOCK}/owner" 2>/dev/null || true
        rmdir "$SCORE_LOCK" 2>/dev/null || true
    fi
}

trap release_score_lock EXIT
acquire_score_lock

write_launcher_manifest() {
    python - "$RUN_DIR" "$MODEL" "${MT_ADAPTER:-}" "$TAG" "$LOAD" \
        "$RUN_MODE" "$SUITE" "$PROTOCOL_TAG" "$PROTOCOL_COMMIT" "$CHOICE_READOUT" \
        "$EXPECTED_SESSIONS" "$NGPU" "$DATA" "$actual_data_sha256" <<'PY'
import json
import os
from pathlib import Path
import sys

(run_dir, model, adapter, tag, load, mode, suite, protocol, commit, readout,
 expected_sessions, num_shards, data, data_sha256) = sys.argv[1:]
payload = {
    "model": model,
    "adapter": adapter or None,
    "tag": tag,
    "load": load,
    "run_mode": mode,
    "suite": suite,
    "protocol_tag": protocol,
    "protocol_commit": commit,
    "choice_readout": readout,
    "condition": suite,
    "expected_sessions": int(expected_sessions),
    "num_shards": int(num_shards),
    "data": data,
    "data_sha256": data_sha256,
    "batch_tokens": 8192,
    "dtype": "fp16",
    "top_k": 20,
    "max_options": 256,
}
if suite == "e3":
    payload.update({
        "windows": [0, 1, 2, 5, 10, 20, "full"],
        "num_positions": 5,
        "position_grid": "e0-informed",
        "full_condition": "full",
        "segmentation_protocol": "last-pre-marker-blank-v1",
        "window_unit": "marked-choice-segment",
    })
path = Path(run_dir) / "launcher.json"
if path.exists():
    existing = json.loads(path.read_text(encoding="utf-8"))
    if existing != payload:
        raise SystemExit(
            f"refusing to mix configurations in {run_dir}: "
            f"launcher.json differs")
else:
    temporary = path.with_suffix(f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)
PY
}

record_l3_attempt() {
    [ "$RUN_MODE" = "l3" ] || return 0
    local ledger="${RUN_DIR}/l3-attempts.tsv"
    printf '%s\t%s\t%s\n' "$L3_ATTEMPT" "$LOCK_TOKEN" "$(date -Is)" >> "$ledger"
}

validate_run() {
    python - "$RUN_DIR" "$EXPECTED_SESSIONS" "$NGPU" "$RUN_MODE" \
        "$CHOICE_READOUT" "$SUITE" <<'PY'
import csv
import json
from pathlib import Path
import sys

run_dir = Path(sys.argv[1])
expected_sessions = int(sys.argv[2])
num_shards = int(sys.argv[3])
mode = sys.argv[4]
choice_readout = sys.argv[5]
suite = sys.argv[6]
suffixes = (
    [""]
    if mode == "smoke" and suite != "e3"
    else [f"_shard{i}" for i in range(num_shards)]
)
launcher = json.loads((run_dir / "launcher.json").read_text(encoding="utf-8"))
reference_manifest = None

def has_data(path):
    if not path.exists():
        return False
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return next(reader, None) is not None

sessions = set()
conditions_by_session = {}
for suffix in suffixes:
    prediction = run_dir / f"predictions{suffix}.csv"
    for table in ("predictions", "pred_topk", "pred_options"):
        path = run_dir / f"{table}{suffix}.csv"
        if not has_data(path):
            raise SystemExit(f"acceptance failed: missing/empty {path}")
    for kind in ("failed", "skipped"):
        sidecar = run_dir / f"predictions{suffix}.{kind}.csv"
        if has_data(sidecar):
            raise SystemExit(f"acceptance failed: unresolved {sidecar}")
    with prediction.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            session = (row["experiment"], row["participant"])
            sessions.add(session)
            conditions_by_session.setdefault(session, set()).add(row["condition"])
    manifest = run_dir / (f"run_shard{suffix.removeprefix('_shard')}.json"
                          if suffix else "run.json")
    if not manifest.exists():
        raise SystemExit(f"acceptance failed: missing {manifest}")
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    normalized = {key: value for key, value in manifest_data.items()
                  if key != "shard"}
    if reference_manifest is None:
        reference_manifest = normalized
    elif normalized != reference_manifest:
        raise SystemExit(
            f"acceptance failed: {manifest} differs from other shard manifests")
    if manifest_data.get("choice_readout") != choice_readout:
        raise SystemExit(
            f"acceptance failed: {manifest} does not declare "
            f"choice_readout={choice_readout}")
    expected_model = launcher["model"] + (
        f"+{launcher['adapter']}" if launcher["adapter"] else "")
    expected_fields = {
        "model": expected_model,
        "adapter": launcher["adapter"],
        "data": launcher["data"],
        "condition": launcher["condition"],
        "load": launcher["load"],
        "dtype": launcher["dtype"],
        "batch_tokens": launcher["batch_tokens"],
        "top_k": launcher["top_k"],
        "max_options": launcher["max_options"],
        "choice_readout": launcher["choice_readout"],
        "commit": launcher["protocol_commit"],
    }
    if launcher["suite"] == "e3":
        expected_fields.update({
            "windows": launcher["windows"],
            "num_positions": launcher["num_positions"],
            "position_grid": launcher["position_grid"],
            "full_condition": launcher["full_condition"],
            "segmentation_protocol": launcher["segmentation_protocol"],
            "window_unit": launcher["window_unit"],
        })
    mismatches = {key: (manifest_data.get(key), value)
                  for key, value in expected_fields.items()
                  if manifest_data.get(key) != value}
    if mismatches:
        raise SystemExit(
            f"acceptance failed: {manifest} disagrees with launcher.json: "
            f"{mismatches}")

if len(sessions) != expected_sessions:
    raise SystemExit(
        f"acceptance failed: expected {expected_sessions} sessions, "
        f"found {len(sessions)}")
expected_conditions = ({"full"} if suite == "full" else
                       {"full", "e3:w=0", "e3:w=1", "e3:w=2",
                        "e3:w=5", "e3:w=10", "e3:w=20"})
bad_conditions = {
    session: sorted(values)
    for session, values in conditions_by_session.items()
    if values != expected_conditions
}
if bad_conditions:
    examples = list(bad_conditions.items())[:3]
    raise SystemExit(
        f"acceptance failed: {len(bad_conditions)} sessions have wrong "
        f"condition coverage; expected {sorted(expected_conditions)}, "
        f"examples={examples}")
print(f"accepted {run_dir}: {len(sessions)} sessions, no failed/skipped rows")
PY
}

report_progress() {
    echo "=== progress at $(date -Is) ==="
    for path in "${PREDICTIONS_STEM}"_shard*.csv; do
        [ -f "$path" ] || continue
        echo "  $path: $(($(wc -l < "$path") - 1)) rows"
    done
}

# SLURM sends USR1 ten minutes before the wall clock runs out; log where
# each shard got to so the resubmission has something to check against.
# The shards keep running until SLURM actually kills them.
MT_WINDING_DOWN=0
trap 'MT_WINDING_DOWN=1; echo "=== USR1: wall clock nearly up ==="; report_progress' USR1

wait_for_pid() {
    local pid="$1"
    local status
    while true; do
        if wait "$pid"; then
            return 0
        else
            status=$?
        fi
        # A signal delivered to this shell interrupts wait while leaving the
        # worker alive. Reap the same PID again instead of losing track of it.
        if kill -0 "$pid" 2>/dev/null; then
            continue
        fi
        return "$status"
    done
}

wait_for_shards() {
    local status=0
    for pid in "$@"; do
        wait_for_pid "$pid" || status=1
    done
    if [ "$status" -ne 0 ]; then
        echo "a shard exited non-zero" >&2
        return 1
    fi
    return 0
}

echo "=== model: $MODEL  (tag $TAG, load $LOAD, mode $RUN_MODE) ==="
echo "=== protocol: $PROTOCOL_TAG ($CHOICE_READOUT, $PROTOCOL_COMMIT) ==="
echo "=== output: $RUN_DIR ==="
write_launcher_manifest
record_l3_attempt
echo "=== preflight ==="
OMP_NUM_THREADS=7 python scripts/experiments/preflight.py --model "$MODEL" --data "$DATA" \
    ${ADAPTER_ARGS[@]+"${ADAPTER_ARGS[@]}"} --load "$LOAD" --output-dir "$RUN_DIR"

# One session from each of the 75 experiments. Every model gets this before
# the full grid: architecture, tokenizer, longest-context, or option-output
# failures then show up before a day-long run.
if [ "$RUN_MODE" = "smoke" ]; then
    echo "=== SMOKE: $EXPECTED_SESSIONS sessions on one GPU, then stop ==="
    if [ "$SUITE" = "e3" ]; then
        OMP_NUM_THREADS=7 CUDA_VISIBLE_DEVICES=0 python \
            scripts/experiments/run_window_scoring.py \
            --model "$MODEL" \
            ${ADAPTER_ARGS[@]+"${ADAPTER_ARGS[@]}"} \
            --data "$DATA" \
            --dataset psych-101-test \
            --condition-prefix e3 \
            --choice-readout "$CHOICE_READOUT" \
            ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"} \
            --shard 0/1 \
            --windows 0,1,2,5,10,20,full \
            --num-positions 5 \
            --position-grid e0-informed \
            --load "$LOAD" \
            --dtype fp16 \
            --batch-tokens 8192 \
            --top-k 20 \
            --max-options 256 \
            --device cuda \
            --output-dir "$RUN_DIR"
    else
        OMP_NUM_THREADS=7 CUDA_VISIBLE_DEVICES=0 python \
            scripts/experiments/run_transcript_scoring.py \
            --model "$MODEL" \
            ${ADAPTER_ARGS[@]+"${ADAPTER_ARGS[@]}"} \
            --data "$DATA" \
            --dataset psych-101-test \
            --condition full \
            --choice-readout "$CHOICE_READOUT" \
            ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"} \
            --load "$LOAD" \
            --dtype fp16 \
            --chunk-size 8 \
            --batch-tokens 8192 \
            --top-k 20 \
            --max-options 256 \
            --device cuda \
            --output-dir "$RUN_DIR" \
            --summary "${RUN_DIR}/summary.csv"
    fi
    if [ "$SUITE" = "e3" ]; then
        python scripts/experiments/merge_run_shards.py \
            --num-shards 1 --run-dir "$RUN_DIR"
    fi
    validate_run
    smoke_gate="${RUN_ROOT}/.smoke-passed"
    smoke_temporary="${smoke_gate}.tmp.${SLURM_JOB_ID:-$$}"
    printf '%s\n' "$SMOKE_GATE_VALUE" > "$smoke_temporary"
    mv "$smoke_temporary" "$smoke_gate"
    echo "=== smoke accepted: $RUN_DIR ==="
    echo "=== smoke gate opened: $smoke_gate ==="
    exit 0
fi

echo "=== $RUN_MODE $SUITE scoring, ${NGPU}-way data parallel ==="
mt_watch_memory "${PREDICTIONS_STEM}.mem"
pids=()
for i in $(seq 0 $((NGPU - 1))); do
    if [ "$SUITE" = "e3" ]; then
        OMP_NUM_THREADS=7 CUDA_VISIBLE_DEVICES="$i" python \
            scripts/experiments/run_window_scoring.py \
            --model "$MODEL" \
            ${ADAPTER_ARGS[@]+"${ADAPTER_ARGS[@]}"} \
            --data "$DATA" \
            --dataset psych-101-test \
            --condition-prefix e3 \
            --choice-readout "$CHOICE_READOUT" \
            ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"} \
            --shard "$i/$NGPU" \
            ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} \
            --windows 0,1,2,5,10,20,full \
            --num-positions 5 \
            --position-grid e0-informed \
            --load "$LOAD" \
            --dtype fp16 \
            --batch-tokens 8192 \
            --top-k 20 \
            --max-options 256 \
            --device cuda \
            ${NOTIFY_ARGS[@]+"${NOTIFY_ARGS[@]}"} \
            --output-dir "$RUN_DIR" &
    else
        OMP_NUM_THREADS=7 CUDA_VISIBLE_DEVICES="$i" python \
            scripts/experiments/run_transcript_scoring.py \
            --model "$MODEL" \
            ${ADAPTER_ARGS[@]+"${ADAPTER_ARGS[@]}"} \
            --data "$DATA" \
            --dataset psych-101-test \
            --condition full \
            --choice-readout "$CHOICE_READOUT" \
            ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"} \
            --shard "$i/$NGPU" \
            ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} \
            --load "$LOAD" \
            --dtype fp16 \
            --chunk-size 8 \
            --batch-tokens 8192 \
            --top-k 20 \
            --max-options 256 \
            --device cuda \
            ${NOTIFY_ARGS[@]+"${NOTIFY_ARGS[@]}"} \
            --output-dir "$RUN_DIR" &
    fi
    pids+=($!)
done
wait_for_shards "${pids[@]}"
mt_report_memory "${PREDICTIONS_STEM}.mem"
report_progress
validate_run

echo "=== done: merge $RUN_DIR with scripts/merge_shards.slurm ==="
