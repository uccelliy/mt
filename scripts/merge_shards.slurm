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
#     sbatch --export=ALL,MT_RUN_DIRS="outputs/runs/minitaur8b" \
#            scripts/merge_shards.slurm

set -euo pipefail

source scripts/hpc_env.sh

if [ -z "${MT_RUN_DIRS:-}" ] && [ -z "${MT_STEMS:-}" ]; then
    echo "MT_RUN_DIRS or legacy MT_STEMS is required, e.g." >&2
    echo "  sbatch --export=ALL,MT_RUN_DIRS=\"outputs/runs/minitaur8b\" $0" >&2
    exit 1
fi

if [ -n "${MT_RUN_DIRS:-}" ]; then
    read -r -a RUN_DIRS <<< "$MT_RUN_DIRS"
    python scripts/experiments/merge_run_shards.py \
        --num-shards "${MT_NGPU:-4}" \
        --run-dir "${RUN_DIRS[@]}"
fi

# Window scoring still uses the older one-table stem contract. Keep that
# path separate so Track P's large top-k table always uses the streaming
# three-table merger above.
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

echo "=== merge done ==="
