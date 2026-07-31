# Site-specific environment for this cluster. Sourced by every .slurm script.
#
# EDIT THIS FILE ONCE, after filling in docs/Server test design.md §1.2.
# The C-numbers below point at rows of that checklist. Everything that
# differs between clusters lives here and nowhere else, except the #SBATCH
# headers (which SLURM parses from each script and cannot be sourced).

print_error_and_exit() { echo "***ERROR*** $*"; exit 1; }

# --- modules (C9/C10/C11) -------------------------------------------------
# EasyBuild/LMod names carry a category prefix, e.g. "lang/Python/3.10.8-..."
# or "numlib/cuDNN". Always pin full versions: a floating `module load CUDA`
# silently changes under you and breaks the venv (see design doc §2.3).
MT_MODULES="${MT_MODULES:-}"

# --- storage (C14/C15) ----------------------------------------------------
# $SCRATCH is /scratch/users/$USER (Lustre, no backup) on ULHPC. $HOME is
# GPFS with an SSD cache — faster for small files but tighter on quota, and
# a 70B checkpoint needs ~140 GB on disk even though NF4 only costs ~40 GB
# of VRAM.
export HF_HOME="${HF_HOME:-${SCRATCH:-$HOME/.cache}/hf}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# ULHPC's own guidance; -c is enforced in every launcher so this is set
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# --- repo -----------------------------------------------------------------
MT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$MT_ROOT"

# ULHPC disables `module` on the access/login nodes on purpose, so this
# doubles as the official safeguard against running a launcher there.
module purge || print_error_and_exit "no 'module' command — this must run \
on a compute node, not an access/login node (try: si-gpu, or salloc -p \
interactive)"

if [ -z "$MT_MODULES" ]; then
    echo "WARNING: MT_MODULES is empty; falling back to whatever python is" \
         "on PATH. Fill in docs/Server test design.md §1.2 first." >&2
else
    for name in $MT_MODULES; do
        module load "$name" || print_error_and_exit "module load $name failed"
    done
fi

# Order matters: modules first, venv second. The venv records the absolute
# path of the python that created it, so loading the modules afterwards
# would leave .venv/bin/python pointing at a different interpreter.
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
else
    echo "WARNING: no .venv found at $MT_ROOT; using the module python" >&2
fi

echo "=== environment ==="
echo "node:     $(hostname)"
echo "root:     $MT_ROOT"
echo "HF_HOME:  $HF_HOME"
module list 2>&1 || true
python --version
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('arch list', torch.cuda.get_arch_list())"
nvidia-smi || true

# Poll whole-card memory into a log so the peak survives the job.
# Whole-card is the number that matters against the 16 GB / 32 GB limit;
# torch.cuda.max_memory_allocated() misses the CUDA context and workspaces.
mt_watch_memory() {
    local log="$1"
    nvidia-smi --query-gpu=index,memory.used,memory.total \
        --format=csv,noheader -l 5 > "$log" 2>/dev/null &
    MT_MEMORY_WATCHER=$!
}

mt_report_memory() {
    local log="$1"
    if [ -n "${MT_MEMORY_WATCHER:-}" ]; then
        kill "$MT_MEMORY_WATCHER" 2>/dev/null || true
        wait "$MT_MEMORY_WATCHER" 2>/dev/null || true
        unset MT_MEMORY_WATCHER
    fi
    if [ -s "$log" ]; then
        echo "=== peak memory (MiB, whole card) ==="
        awk -F', *' '{gsub(/ MiB/,"",$2); if ($2+0 > peak[$1]) peak[$1]=$2+0}
                     END {for (i in peak) print "  gpu " i ": " peak[i]}' "$log"
    fi
}
