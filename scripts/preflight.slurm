#!/usr/bin/bash --login
#SBATCH --job-name=mt-centaur-preflight
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --partition=batch
#SBATCH --qos=normal

# CPU-only preflight: checks env, data integrity, segmentation, and model
# cache completeness without requesting a GPU, so it queues fast. If the
# CPU partition is not named "batch" on this cluster, adjust --partition.
# The smoke job (smoke_e0_e3.slurm) repeats these checks before using GPUs,
# so skipping this job costs nothing but queue time.

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT"

module purge || true
module load cuda || true

python --version
python scripts/experiments/preflight.py \
  --model marcelbinz/Llama-3.1-Minitaur-8B \
  --data data/psych-101-test/prompts_testing_t1.jsonl
