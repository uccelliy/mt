#!/usr/bin/bash --login

#SBATCH --job-name=gpu_example
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out

### Request one GPU tasks for 4 hours - dedicate 1/4 of available cores for its management
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --time=0-04:00:00

### Submit to the `gpu` partition of Iris
#SBATCH --partition=gpu
#SBATCH --qos=normal

print_error_and_exit() { echo "***ERROR*** $*"; exit 1; }

module purge || print_error_and_exit "No 'module' command available"
module load numlib/cuDNN   # Example using the cuDNN module

[...]
