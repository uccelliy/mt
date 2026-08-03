#!/usr/bin/bash --login
#SBATCH --job-name=mt-mail-notify-test
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --partition=interactive
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128M
#SBATCH --time=00:02:00
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90

# Confirms --mail-user on a real job, and specifically that TIME_LIMIT_90
# fires -- the one mail type the earlier probe did not exercise. No GPU, no
# module, no venv: 1 CPU and 128 MB for two minutes.
#
#     cd ~/mt && sbatch --mail-user=<you>@uni.lu scripts/mail_notify_test.slurm
#
# The job deliberately outlives its own --time, so expect THREE messages:
#
#   1. BEGIN           immediately
#   2. TIME_LIMIT_90   at ~108 s (90% of the 120 s limit)
#   3. a terminal one  at 120 s, when Slurm kills the job for exceeding
#                      its walltime -- note which type actually arrives
#                      (END vs FAIL), since that decides whether a timed-out
#                      production run would notify you at all
#
# The final echo is never reached; that is the point.

echo "start   $(date -Is)  job=${SLURM_JOB_ID:-none}  node=$(hostname)"
echo "walltime limit is 2 min; sleeping 130 s so the limit is exceeded"
sleep 130
echo "not reached -- if you see this, the job was not killed as expected"
