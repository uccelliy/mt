#!/usr/bin/bash --login
#SBATCH --job-name=mt-merge-shards
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
### 2 cores on a 28-core/128 GB batch node buys ~9.1 GB before ULHPC
### re-computes the core equivalent and charges for more, so stay under it.
### Concatenating four ~40 MB shards needs well under 1 GB.
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --partition=batch
#SBATCH --qos=normal
### The venv is built against the SKYLAKE EasyBuild tree (.../2023b/skylake/
### software/Python/...), whose binaries carry AVX-512. Half of `batch` is
### broadwell (iris-[001-108]) and would die with SIGILL, so pin skylake --
### iris-[109-168] are the skylake half of the partition.
#SBATCH --constraint=skylake

# Merge per-GPU shard CSVs into one file per stem. This is a separate job
# on purpose: when it lived at the end of the scoring script, a wall-clock
# timeout threw away the merge even though every shard had finished.
#
# No GPU, so it belongs on `batch`, not `gpu` — ULHPC explicitly asks that
# GPU nodes not be used by jobs that do not need accelerators. The same
# goes for the E2 baselines and the figure builders (design doc §8.5).
#
#     sbatch --export=ALL,MT_STEMS="outputs/scoring/hpc_e0_minitaur8b_4bit" \
#            scripts/merge_shards.slurm

set -euo pipefail

source scripts/hpc_env.sh

if [ -z "${MT_STEMS:-}" ]; then
    echo "MT_STEMS is required, e.g." >&2
    echo "  sbatch --export=ALL,MT_STEMS=\"outputs/scoring/hpc_e0_minitaur8b_4bit\" $0" >&2
    exit 1
fi

python - "$MT_STEMS" <<'EOF'
import glob
import sys

import pandas as pd

for stem in sys.argv[1].split():
    shards = sorted(glob.glob(f"{stem}_shard*.csv"))
    if not shards:
        print(f"{stem}: no shards found", flush=True)
        continue
    frame = pd.concat([pd.read_csv(s, low_memory=False) for s in shards],
                      ignore_index=True)
    keys = [name
            for name in ('experiment', 'participant', 'window',
                         'target_index', 'choice_index')
            if name in frame.columns]
    duplicates = frame.duplicated(subset=keys).sum()
    sessions = frame[['experiment', 'participant']].drop_duplicates()
    frame.to_csv(f"{stem}.csv", index=False)
    print(f"{stem}: {len(frame)} rows, {len(sessions)} sessions, "
          f"from {len(shards)} shards", flush=True)
    # a resumed run that double-counted work shows up here and nowhere else
    if duplicates:
        print(f"  WARNING: {duplicates} duplicate rows on {keys}",
              flush=True)
EOF

echo "=== merge done ==="
