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
#   salloc -p interactive --qos=normal -N 1 --ntasks-per-node=1 -c 1 \
#          --time=0:30:00
#   bash --login scripts/hpc_probe.sh 2>&1 | tee hpc_probe_compute.txt

section() {
    echo
    echo "################ $* ################"
}

have() {
    command -v "$1" >/dev/null 2>&1
}

section "C0  host"
hostname
uname -a
have lsb_release && lsb_release -d 2>/dev/null
echo "USER=$USER  HOME=$HOME  PWD=$PWD"

section "C1  partitions"
have sinfo && sinfo -o "%20P %5a %12l %10s %8D %20N" | head -30

section "C2/C3  gres and node features"
have sinfo && sinfo -o "%20P %18G %22f %8m %N" | head -30

section "C4  one GPU node in detail"
if have sinfo && have scontrol; then
    # -N lists one line per node, so this never yields a range expression
    node=$(sinfo -h -N -p gpu -o "%N" 2>/dev/null | head -1)
    [ -n "$node" ] || node=$(sinfo -h -N -o "%N" | head -1)
    echo "sampling node: $node"
    scontrol show node "$node" 2>/dev/null | head -25
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
have sacctmgr && sacctmgr -n show assoc user="$USER" \
    format=Account%20,Partition%20,QOS%60,MaxJobs,GrpTRES%30 2>/dev/null

section "C6  qos limits"
have sacctmgr && sacctmgr -n show qos \
    format=Name%20,MaxWall,MaxTRESPU%30,MaxJobsPU,MaxSubmitPU 2>/dev/null | head -25

section "C6b  my fairshare"
have ulhpcshare && ulhpcshare 2>&1 | head -15
have sshare && sshare -U 2>&1 | head -10

section "C7  my queue right now"
have squeue && squeue -u "$USER" 2>/dev/null | head -10

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
    # EasyBuild names carry a category prefix on ULHPC, e.g. lang/Python
    for keyword in lang/Python system/CUDA numlib/cuDNN PyTorch toolchain/foss; do
        echo "--- $keyword ---"
        module avail "$keyword" 2>&1 | head -14
    done
    echo "--- spider PyTorch (finds hierarchy-hidden modules) ---"
    module spider PyTorch 2>&1 | head -20
else
    echo "skipped: no module command here"
fi

section "features (ULHPC ships an sfeatures helper)"
have sfeatures && sfeatures 2>&1 | head -20

section "C12  outbound network from the login node"
if have curl; then
    curl -s -o /dev/null -w "huggingface.co  -> HTTP %{http_code}\n" \
        --max-time 10 https://huggingface.co/ || echo "huggingface.co unreachable"
    curl -s -o /dev/null -w "pypi.org        -> HTTP %{http_code}\n" \
        --max-time 10 https://pypi.org/ || echo "pypi.org unreachable"
else
    echo "no curl"
fi

section "C14  home quota"
have quota && quota -s 2>/dev/null
have lfs && lfs quota -h -u "$USER" "$HOME" 2>/dev/null
du -sh "$HOME" 2>/dev/null | tail -1

section "C15  ULHPC filesystems"
echo "SCRATCH=${SCRATCH:-<unset>}"
echo "PROJECTHOME=${PROJECTHOME:-<unset>}"
echo "ULHPC_CLUSTER=${ULHPC_CLUSTER:-<unset>}"
df -h "$HOME" "${SCRATCH:-$HOME}" 2>/dev/null
ls -ld /work/projects/* 2>/dev/null | head -10
echo "--- inode usage (space is free up to 500GB home / 10TB scratch;"
echo "    the file-count limit is the one that actually bites) ---"
have lfs && lfs quota -h -u "$USER" "${SCRATCH:-/scratch}" 2>/dev/null
have mmlsquota && mmlsquota --block-size auto 2>/dev/null | head -10

section "existing toolchain"
for tool in python python3 uv pip git rsync nvidia-smi sbatch seff sfeatures ulhpcshare; do
    printf "%-12s " "$tool"
    have "$tool" && command -v "$tool" || echo "(absent)"
done
python3 --version 2>&1 || true

echo
echo "################ probe complete ################"
echo "Paste this output back, plus the school's example sbatch script."
