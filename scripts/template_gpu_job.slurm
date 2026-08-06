#!/usr/bin/bash --login
# 学习用的最小 GPU 作业模板（ULHPC iris）。
# 刻意不含错误检查、进度打印、显存监控——那些在 scripts/hpc_env.sh 和
# 真正的作业脚本里。这里只保留骨架：参数请求、环境准备、调用代码。
#
# 提交（必须先 cd 到仓库根目录）：
#     cd ~/mt && sbatch scripts/template_gpu_job.slurm

# ============================================================
# 一、资源请求
# 每行 #SBATCH 必须在任何可执行语句之前，否则 SLURM 不解析。
# 命令行同名参数会覆盖这里，例如 sbatch --time=02:00:00 ...
# ============================================================

#SBATCH --job-name=mt-example
#SBATCH --output=%x-%j.out          # %x=作业名 %j=作业号，落在提交目录
#SBATCH --error=%x-%j.err

#SBATCH --partition=gpu             # gpu / batch / bigmem / interactive
#SBATCH --qos=normal                # interactive 分区只收 debug

#SBATCH --nodes=1                   # 节点数
#SBATCH --ntasks-per-node=1         # 每节点几个任务（进程）
#SBATCH --cpus-per-task=7           # 每任务几个核；iris GPU 节点 28 核 / 4 卡
#SBATCH --gpus-per-task=1           # 每任务几张卡
#SBATCH --mem=32G                   # 整个作业的内存

#SBATCH --constraint=volta16        # volta16=16GB卡 volta32=32GB卡
#SBATCH --time=00:30:00             # 上限；gpu 分区最长 2-00:00:00

### Slurm native notifications. Verified working on ULHPC 2026-08-03:
### BEGIN/END/FAIL and compute-node mail all deliver to @uni.lu addresses.
### External delivery (Gmail) is accepted by the relay but never arrives, so
### use the university address. TIME_LIMIT_90 fires at 90% of --time, which
### says nothing about how far the analysis got -- that is what the in-process
### progress notifier is for. Supply the address at submit time:
###     sbatch --mail-user=name.surname@uni.lu scripts/<job>.slurm
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90

# ============================================================
# 二、软件环境
# 登录节点没有 module，只有计算节点有——所以这几行只能在作业里跑。
# 顺序不能反：先 purge，再 load，最后 activate。
# venv 里记录着创建它的那个 python 的绝对路径，module 没加载好
# 就会用错解释器。
# ============================================================

module purge
module load env/release/2023b                    # 工具链网关，不加载则后面的模块不可见
module load lang/Python/3.11.5-GCCcore-13.2.0    # 版本写死，不要写裸 lang/Python

# ============================================================
# 三、工作目录与 venv
# sbatch 作业的初始目录就是你敲 sbatch 时所在的目录（SLURM_SUBMIT_DIR）。
# 本项目的脚本用相对路径找 data/ 和 outputs/，所以必须在仓库根目录运行。
# ============================================================

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

# ============================================================
# 四、环境变量
# ============================================================

export HF_HOME=/scratch/users/$USER/hf     # 模型缓存，别放 home
export HF_HUB_OFFLINE=1                    # 禁止联网，缓存不全时立刻失败而不是偷偷去下
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ============================================================
# 五、调用代码
# scripts/experiments/*.py 之间用 `from _common import ...` 互相引用，
# 依赖 sys.path[0] 是脚本自己的目录，所以必须用路径调用，
# 不能写成 python -m mt.xxx
# ============================================================

python scripts/experiments/run_transcript_scoring.py \
    --model meta-llama/Llama-3.1-8B \
    --adapter marcelbinz/Llama-3.1-Centaur-8B-adapter \
    --choice-readout greedy-unconstrained-1token \
    --data data/psych-101-test/prompts_testing_t1.jsonl \
    --experiment kool2016when/exp1.csv \
    --load 4bit \
    --device cuda \
    --output-dir outputs/runs/template_example
