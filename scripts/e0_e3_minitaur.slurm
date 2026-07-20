#!/usr/bin/bash --login
#SBATCH --job-name=mt-centaur-eval
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu
#SBATCH --qos=normal

# E0 (full-context scoring, all test sessions) then E3 (window truncation,
# sampled 50 participants/experiment to match the E2 subset). Data-parallel:
# 4 processes, one per V100, each scoring every 4th session into its own
# shard CSV; shards are merged at the end. Every shard resumes from its own
# CSV, so resubmitting this job after a timeout continues where it stopped.
#
# Prerequisites on the cluster:
#   - data/psych-101-test/prompts_testing_t1.jsonl (rsync from local)
#   - Minitaur-8B in the HF cache (download on a login node with network,
#     or rsync ~/.cache/huggingface/hub/models--marcelbinz--Llama-3.1-Minitaur-8B)

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export HF_HUB_OFFLINE=1

module purge || true
module load numlib/cuDNN || true
module load cuda || true

MODEL="marcelbinz/Llama-3.1-Minitaur-8B"
DATA="data/psych-101-test/prompts_testing_t1.jsonl"
NGPU=4

echo "=== node info ==="
nvidia-smi
python --version

echo "=== preflight ==="
python scripts/experiments/preflight.py --model "$MODEL" --data "$DATA"

mkdir -p outputs/scoring

echo "=== E0: full-context scoring, ${NGPU}-way data parallel ==="
for i in $(seq 0 $((NGPU - 1))); do
  CUDA_VISIBLE_DEVICES="$i" python scripts/experiments/run_transcript_scoring.py \
    --model "$MODEL" \
    --data "$DATA" \
    --shard "$i/$NGPU" \
    --resume \
    --chunk-size 8 \
    --batch-tokens 16384 \
    --device cuda \
    --output "outputs/scoring/e0_minitaur8b_full_shard$i.csv" &
done
wait

echo "=== E3: window truncation, ${NGPU}-way data parallel ==="
for i in $(seq 0 $((NGPU - 1))); do
  CUDA_VISIBLE_DEVICES="$i" python scripts/experiments/run_window_scoring.py \
    --model "$MODEL" \
    --data "$DATA" \
    --max-participants 50 \
    --seed 0 \
    --shard "$i/$NGPU" \
    --windows 0,1,2,5,10,20,full \
    --num-positions 8 \
    --resume \
    --batch-tokens 16384 \
    --device cuda \
    --output "outputs/scoring/e3_minitaur8b_windows_s50_shard$i.csv" &
done
wait

echo "=== merging shards ==="
python - <<'EOF'
import glob
import pandas as pd

for stem in ["e0_minitaur8b_full", "e3_minitaur8b_windows_s50"]:
    shards = sorted(glob.glob(f"outputs/scoring/{stem}_shard*.csv"))
    frame = pd.concat([pd.read_csv(s) for s in shards], ignore_index=True)
    frame.to_csv(f"outputs/scoring/{stem}.csv", index=False)
    print(stem, "->", len(frame), "rows from", len(shards), "shards")
EOF

echo "=== done ==="
