#!/usr/bin/bash --login
# Read-only probe of the cluster. Every section is labelled with the
# C-number of the row it fills in docs/Server test design.md §1.2. Nothing
# here writes, submits, or loads anything — safe to run repeatedly.
#
# Run it TWICE. ULHPC disables `module` on the access/login nodes on
# purpose, so the module sections only produce output inside a job:
#
#   # on the login node — everything except C9/C10/C11
#   bash --login scripts/hpc_probe.sh 2>&1 | tee hpc_probe_login.txt
#
#   # inside an interactive job — this is where the module rows get filled
#   salloc -p interactive --qos=debug -N 1 --ntasks-per-node=1 -c 1 \
#          --time=0:30:00
#   (the interactive partition allows ONLY the debug QOS -- --qos=normal
#    fails with "Invalid qos specification")
#   bash --login scripts/hpc_probe.sh 2>&1 | tee hpc_probe_compute.txt

section() {
    echo
    echo "################ $* ################"
}

have() {
    command -v "$1" >/dev/null 2>&1
}

# Every probe goes through this. Filesystem and scheduler queries on a busy
# cluster can block in uninterruptible I/O, where Ctrl-C does not help --
# a timeout is the only way to stay recoverable.
run() {
    timeout 20 "$@" 2>/dev/null || echo "  (no output / timed out: $*)"
}

section "C0  host"
hostname
uname -a
have lsb_release && lsb_release -d 2>/dev/null
echo "USER=$USER  HOME=$HOME  PWD=$PWD"

section "C1  partitions"
have sinfo && run sinfo -o "%20P %5a %12l %10s %8D %20N" | head -30

section "C2/C3  gres and node features"
have sinfo && run sinfo -o "%20P %18G %22f %8m %N" | head -30

section "C4  one GPU node in detail"
if have sinfo && have scontrol; then
    # -N lists one line per node, so this never yields a range expression
    node=$(sinfo -h -N -p gpu -o "%N" 2>/dev/null | head -1)
    [ -n "$node" ] || node=$(sinfo -h -N -o "%N" | head -1)
    echo "sampling node: $node"
    run scontrol show node "$node" | head -25
fi

section "C4b  TRESBillingWeights (what each partition actually charges)"
if have scontrol; then
    for part in gpu batch interactive; do
        echo "--- $part ---"
        scontrol show partition "$part" 2>/dev/null \
            | tr ' ' '\n' | grep -iE "TRESBilling|MaxTime|DefaultTime|MaxNodes|QoS" \
            | head -8
    done
fi

section "C5  my associations (account / partition / qos)"
have sacctmgr && run sacctmgr -n -P show assoc user="$USER" \
    format=Account,QOS

section "C6  qos limits"
have sacctmgr && run sacctmgr -n show qos \
    format=Name%20,MaxWall,MaxTRESPU%30,MaxJobsPU,MaxSubmitPU | head -25

section "C6b  my fairshare"
have ulhpcshare && run ulhpcshare | head -15
have sshare && run sshare -U | head -10

section "C7  my queue right now"
have squeue && run squeue -u "$USER" | head -10

section "C8  module system"
if ! type module >/dev/null 2>&1; then
    echo "NOTE: no 'module' command. On ULHPC this is EXPECTED on an"
    echo "      access/login node. Rerun inside an interactive job to"
    echo "      fill in C9/C10/C11."
else
    module --version 2>&1 | head -5
fi

section "C9/C10/C11  python / cuda / cudnn / pytorch modules"
if type module >/dev/null 2>&1; then
    # ULHPC's EasyBuild trees are PER-ARCHITECTURE: the path ends in
    # .../<toolchain>/<broadwell|skylake>/modules/all. A broadwell batch
    # node and a skylake GPU node therefore see DIFFERENT module sets, so
    # this section is only meaningful on the node type you will run on.
    echo "--- MODULEPATH (note the architecture component) ---"
    tr ':' '\n' <<< "$MODULEPATH"
    # spider searches every tree, avail only the currently visible one
    for keyword in Python CUDA cuDNN PyTorch; do
        echo "--- spider $keyword ---"
        module spider "$keyword" 2>&1 | grep -vE "^\s*$|^-{20,}" | head -30
    done
    echo "--- anything with cuda in the name, visible right now ---"
    module avail 2>&1 | grep -iE "cuda|cudnn" | head -20
else
    echo "skipped: no module command here"
fi

section "GPU node only: does a cluster PyTorch actually see the V100?"
if type module >/dev/null 2>&1 && have nvidia-smi; then
    run nvidia-smi --query-gpu=name,memory.total,driver_version \
        --format=csv,noheader
    # The judgement is not "does torch import" but "was it compiled with
    # sm_70": CUDA 13 builds dropped Volta, and a PTX-JIT fallback is not
    # something to discover halfway through a 47-hour job.
    torch_module=$(module -t spider PyTorch 2>&1 | grep -E "^ai/PyTorch/" | tail -1)
    if [ -n "$torch_module" ]; then
        echo "--- trying $torch_module ---"
        module purge 2>/dev/null
        if module load "$torch_module" 2>&1; then
            python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('arch list', torch.cuda.get_arch_list()); print('cuda available', torch.cuda.is_available())" 2>&1 | head -10
        fi
        module purge 2>/dev/null
    else
        echo "no ai/PyTorch module found by spider"
    fi
else
    echo "skipped: not on a GPU node (no nvidia-smi), or no module command"
fi

section "features (ULHPC ships an sfeatures helper)"
have sfeatures && run sfeatures | head -20

section "C12  outbound network from the login node"
if have curl; then
    curl -s -o /dev/null -w "huggingface.co  -> HTTP %{http_code}\n" \
        --max-time 10 https://huggingface.co/ || echo "huggingface.co unreachable"
    curl -s -o /dev/null -w "pypi.org        -> HTTP %{http_code}\n" \
        --max-time 10 https://pypi.org/ || echo "pypi.org unreachable"
else
    echo "no curl"
fi

section "C14  quotas (space AND inodes)"
# NEVER run `du` on $HOME here: on GPFS it walks millions of files and
# blocks in uninterruptible I/O, so Ctrl-C cannot kill it. The quota tools
# below read metadata instead and answer the same question.
echo "--- \$HOME (GPFS -> mmlsquota) ---"
have mmlsquota && run mmlsquota --block-size auto
have quota && run quota -s
echo "--- \$SCRATCH (Lustre -> lfs quota) ---"
have lfs && run lfs quota -h -u "$USER" "${SCRATCH:-/scratch}"
echo "--- ULHPC ships its own helpers on some systems ---"
have df-ulhpc && run df-ulhpc

section "C15  ULHPC filesystems"
echo "SCRATCH=${SCRATCH:-<unset>}"
echo "PROJECTHOME=${PROJECTHOME:-<unset>}"
echo "ULHPC_CLUSTER=${ULHPC_CLUSTER:-<unset>}"
run df -h "$HOME" "${SCRATCH:-$HOME}"
run ls -ld /work/projects/

section "existing toolchain"
for tool in python python3 uv pip git rsync nvidia-smi sbatch seff sfeatures ulhpcshare; do
    printf "%-12s " "$tool"
    have "$tool" && command -v "$tool" || echo "(absent)"
done
python3 --version 2>&1 || true

echo
echo "################ probe complete ################"
echo "Paste this output back, plus the school's example sbatch script."
