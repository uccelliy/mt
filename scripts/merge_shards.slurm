#!/usr/bin/bash --login
#SBATCH --job-name=mt-merge-shards
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
### 2 cores on a 28-core/128 GB batch node buys ~9.1 GB before ULHPC
### re-computes the core equivalent and charges for more, so stay under it.
### The merge streams the top-k table and only retains parent keys in RAM.
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --partition=batch
#SBATCH --qos=normal
### The venv is built against the SKYLAKE EasyBuild tree (.../2023b/skylake/
### software/Python/...), whose binaries carry AVX-512. Half of `batch` is
### broadwell (iris-[001-108]) and would die with SIGILL, so pin skylake --
### iris-[109-168] are the skylake half of the partition.
#SBATCH --constraint=skylake

### Slurm native notifications. Verified working on ULHPC 2026-08-03:
### BEGIN/END/FAIL and compute-node mail all deliver to @uni.lu addresses.
### External delivery (Gmail) is accepted by the relay but never arrives, so
### use the university address. TIME_LIMIT_90 fires at 90% of --time, which
### says nothing about how far the analysis got -- that is what the in-process
### progress notifier is for. Supply the address at submit time:
###     sbatch --mail-user=name.surname@uni.lu scripts/<job>.slurm
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90

# Merge each run directory's three per-GPU tables. This is a separate job
# on purpose: when it lived at the end of the scoring script, a wall-clock
# timeout threw away the merge even though every shard had finished.
#
# No GPU, so it belongs on `batch`, not `gpu` — ULHPC explicitly asks that
# GPU nodes not be used by jobs that do not need accelerators. The same
# goes for the E2 baselines and the figure builders (design doc §8.5).
#
#     sbatch --export=ALL,MT_RUN_DIRS="outputs/runs/centaur8b" \
#            scripts/merge_shards.slurm

set -euo pipefail

source scripts/hpc_env.sh

if [ -z "${MT_RUN_DIRS:-}" ] && [ -z "${MT_STEMS:-}" ]; then
    echo "MT_RUN_DIRS or legacy MT_STEMS is required, e.g." >&2
    echo "  sbatch --export=ALL,MT_RUN_DIRS=\"outputs/runs/centaur8b\" $0" >&2
    exit 1
fi

if [ -n "${MT_RUN_DIRS:-}" ]; then
    read -r -a RUN_DIRS <<< "$MT_RUN_DIRS"
    LOCK_HOST=$(hostname)
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        LOCK_TOKEN="$SLURM_JOB_ID"
    else
        LOCK_TOKEN="manual:${LOCK_HOST}:$$"
    fi
    LOCK_DIRS=()

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

    release_merge_locks() {
        local lock_dir
        for lock_dir in "${LOCK_DIRS[@]}"; do
            if [ -f "${lock_dir}/owner" ] && \
               [ "$(tr -d '\r\n' < "${lock_dir}/owner")" = "$LOCK_TOKEN" ]; then
                unlink "${lock_dir}/owner" 2>/dev/null || true
                rmdir "$lock_dir" 2>/dev/null || true
            fi
        done
    }

    acquire_merge_lock() {
        local run_dir="$1"
        local merge_lock="${run_dir}/.merge.lock"
        local score_lock="${run_dir}/.score.lock"
        if [ ! -d "$run_dir" ]; then
            echo "run directory does not exist: $run_dir" >&2
            return 1
        fi
        while ! mkdir "$merge_lock" 2>/dev/null; do
            archive_stale_lock "$merge_lock" "merge" || return 1
        done
        printf '%s\n' "$LOCK_TOKEN" > "${merge_lock}/owner"
        LOCK_DIRS+=("$merge_lock")
        if [ -d "$score_lock" ]; then
            archive_stale_lock "$score_lock" "score" || return 1
        fi
    }

    trap release_merge_locks EXIT
    for run_dir in "${RUN_DIRS[@]}"; do
        acquire_merge_lock "$run_dir"
    done
    python scripts/experiments/merge_run_shards.py \
        --num-shards "${MT_NGPU:-4}" \
        --run-dir "${RUN_DIRS[@]}"

    # Formal Track-P jobs provide an expected session count. Treat any
    # failed/skipped sidecar or missing session as an acceptance failure,
    # even though the scorer itself logs per-session OOM/context failures and
    # can otherwise exit zero. This validation runs after the atomic merge.
    if [ -n "${MT_EXPECTED_SESSIONS:-}" ]; then
        python - "$MT_EXPECTED_SESSIONS" "${RUN_DIRS[@]}" <<'EOF'
import csv
import json
from pathlib import Path
import sys

expected = int(sys.argv[1])

def has_data(path):
    if not path.exists():
        return False
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return next(reader, None) is not None

for value in sys.argv[2:]:
    run_dir = Path(value)
    launcher_path = run_dir / "launcher.json"
    if not launcher_path.exists():
        raise SystemExit(f"acceptance failed: missing {launcher_path}")
    launcher = json.loads(launcher_path.read_text(encoding="utf-8"))
    manifest_paths = sorted(run_dir.glob("run_shard*.json"))
    if len(manifest_paths) != launcher["num_shards"]:
        raise SystemExit(
            f"acceptance failed: expected {launcher['num_shards']} runner "
            f"manifests, found {len(manifest_paths)}")
    reference = None
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
    if launcher.get("suite") == "e3":
        expected_fields.update({
            "windows": launcher["windows"],
            "num_positions": launcher["num_positions"],
            "position_grid": launcher["position_grid"],
            "full_condition": launcher["full_condition"],
            "segmentation_protocol": launcher["segmentation_protocol"],
            "window_unit": launcher["window_unit"],
        })
    for manifest_path in manifest_paths:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        normalized = {key: value for key, value in manifest.items()
                      if key != "shard"}
        if reference is None:
            reference = normalized
        elif normalized != reference:
            raise SystemExit(
                f"acceptance failed: {manifest_path} differs from other "
                "shard manifests")
        mismatches = {key: (manifest.get(key), wanted)
                      for key, wanted in expected_fields.items()
                      if manifest.get(key) != wanted}
        if mismatches:
            raise SystemExit(
                f"acceptance failed: {manifest_path} disagrees with "
                f"launcher.json: {mismatches}")
    prediction = run_dir / "predictions.csv"
    if not has_data(prediction):
        raise SystemExit(f"acceptance failed: missing/empty {prediction}")
    sidecars = sorted(run_dir.glob("predictions*.failed.csv"))
    sidecars += sorted(run_dir.glob("predictions*.skipped.csv"))
    unresolved = [str(path) for path in sidecars if has_data(path)]
    if unresolved:
        raise SystemExit(
            "acceptance failed: unresolved failed/skipped logs: "
            + ", ".join(unresolved))
    sessions = set()
    conditions_by_session = {}
    with prediction.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            session = (row["experiment"], row["participant"])
            sessions.add(session)
            conditions_by_session.setdefault(session, set()).add(row["condition"])
    if len(sessions) != expected:
        raise SystemExit(
            f"acceptance failed: {run_dir} expected {expected} sessions, "
            f"found {len(sessions)}")
    suite = launcher.get("suite", "full")
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
            f"acceptance failed: {run_dir} has {len(bad_conditions)} "
            f"sessions with wrong condition coverage; expected "
            f"{sorted(expected_conditions)}, examples={examples}")
    print(f"accepted {run_dir}: {len(sessions)} sessions")
EOF
    fi

    if [ -n "${MT_L3_PROOF_FILE:-}" ]; then
        if [ "${#RUN_DIRS[@]}" -ne 1 ]; then
            echo "MT_L3_PROOF_FILE requires exactly one run directory" >&2
            exit 1
        fi
        python - "$MT_L3_PROOF_FILE" "${RUN_DIRS[0]}" <<'PY'
import csv
import json
from pathlib import Path
import sys

proof_path = Path(sys.argv[1])
run_dir = Path(sys.argv[2])
with proof_path.open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
if len(rows) != 4 or {int(row["shard"]) for row in rows} != set(range(4)):
    raise SystemExit(f"invalid four-shard L3 resume proof: {proof_path}")

for row in rows:
    shard = int(row["shard"])
    before_commits = int(row["committed_sessions"])
    before_rows = int(row["prediction_rows"])
    commit_dir = run_dir / f"session_commits_shard{shard}"
    markers = sorted(commit_dir.glob("*.json"))
    for marker in markers:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if payload.get("status") != "success":
            raise SystemExit(f"non-success L3 commit marker: {marker}")
    prediction = run_dir / f"predictions_shard{shard}.csv"
    with prediction.open(encoding="utf-8", newline="") as handle:
        current_rows = sum(1 for _ in handle) - 1
    if len(markers) != 50:
        raise SystemExit(
            f"L3 shard {shard} has {len(markers)} commits after resume; expected 50")
    if len(markers) <= before_commits or current_rows <= before_rows:
        raise SystemExit(
            f"L3 shard {shard} did not grow across resume: "
            f"commits {before_commits}->{len(markers)}, "
            f"prediction rows {before_rows}->{current_rows}")
print(f"accepted four-shard interrupted resume proof: {proof_path}")
PY
    fi
fi

# Retired/legacy window outputs may still use the older one-table stem
# contract. Formal E3 uses MT_RUN_DIRS and the streaming three-table merger
# above; this path remains only for reproducing historical NLL previews.
if [ -n "${MT_STEMS:-}" ]; then
    python - "$MT_STEMS" "${MT_NGPU:-4}" <<'EOF'
import glob
import os
import re
import sys

import pandas as pd

stems = sys.argv[1].split()
num_shards = int(sys.argv[2])
for stem in stems:
    expected = [f"{stem}_shard{i}.csv" for i in range(num_shards)]
    shard_pattern = re.compile(rf"{re.escape(stem)}_shard\d+\.csv")
    actual = sorted(path for path in glob.glob(f"{stem}_shard*.csv")
                    if shard_pattern.fullmatch(path))
    if actual != expected:
        raise SystemExit(f"{stem}: incomplete shard set: {actual}")
    failures = []
    for shard in expected:
        failed = shard.removesuffix(".csv") + ".failed.csv"
        if os.path.exists(failed) and len(pd.read_csv(failed)):
            failures.append(failed)
    if failures:
        raise SystemExit(f"{stem}: unresolved failure logs: {failures}")
    frame = pd.concat([pd.read_csv(path, low_memory=False)
                       for path in expected], ignore_index=True)
    keys = [name for name in ("experiment", "participant", "window",
                              "target_index", "choice_index")
            if name in frame.columns]
    duplicates = frame.duplicated(subset=keys).sum()
    if duplicates:
        raise SystemExit(f"{stem}: {duplicates} duplicate rows on {keys}")
    destination = f"{stem}.csv"
    temporary = f"{destination}.tmp"
    frame.to_csv(temporary, index=False)
    os.replace(temporary, destination)
    sessions = frame[["experiment", "participant"]].drop_duplicates()
    print(f"{stem}: {len(frame)} rows, {len(sessions)} sessions", flush=True)
EOF
fi

# L3 is the only caller allowed to provide a gate file. It is published only
# after merge integrity and session/sidecar acceptance both succeed. The
# launcher compares the exact protocol/readout/commit value before allowing a
# formal full submission.
if [ -n "${MT_GATE_FILE:-}" ]; then
    if [ -z "${MT_GATE_VALUE:-}" ] || [ -z "${MT_EXPECTED_SESSIONS:-}" ]; then
        echo "MT_GATE_FILE requires MT_GATE_VALUE and MT_EXPECTED_SESSIONS" >&2
        exit 1
    fi
    mkdir -p "$(dirname "$MT_GATE_FILE")"
    temporary="${MT_GATE_FILE}.tmp.${SLURM_JOB_ID:-$$}"
    printf '%s\n' "$MT_GATE_VALUE" > "$temporary"
    mv "$temporary" "$MT_GATE_FILE"
    echo "=== L3 gate opened: $MT_GATE_FILE ==="
fi

echo "=== merge done ==="
