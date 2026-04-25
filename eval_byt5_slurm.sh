#!/bin/bash
#SBATCH --mail-type=none
#SBATCH --mail-user=wagner@lhlt.mpg.de
#SBATCH --output=eval_%j.out
#SBATCH --error=eval_%j.err
#SBATCH --job-name=byt5-eval-salamanca
#SBATCH -D .                   # Initial working directory

#SBATCH --constraint="apu"
#SBATCH --nodes=1

# --- Change the following for testing the workflow/GPU setup ---
#SBATCH --time=23:30:00         # apudev has walltime of 15 min, apu of 24h
#SBATCH --partition=apu         # check actual partition name
# #SBATCH --partition=apudev      # Viper apudev: for testing, 1 node with 2 MI300, 15 min. walltime

# --- VIPER default case: use a single APU on a shared node ---
# #SBATCH --gres=gpu:1            # One node
# #SBATCH --ntasks=1              # One task
# #SBATCH --cpus-per-task=16      # 1/8 of available CPUs
# #SBATCH --mem=110000            # of 128000

# --- VIPER alternative case: two APUs on a shared node ---
#SBATCH --gres=gpu:2            # Two GPUs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=220000

# --- DAIS: H200 on a shared node would be ---
# #SBATCH --partition="gpu1"    # request a shared node.
# #SBATCH --gres=gpu:h200:1     # use 1 H200.
# #SBATCH --cpus-per-task=12    # request 1/8 of available CPUs on a H200 node.
# #SBATCH --mem=250000          # grant the job access to 1/8 of the memory on a H200 node.

# --- DAIS: 2 H200 GPUs on a shared node ---
# #SBATCH --partition="gpu1"    # request a shared node.
# #SBATCH --gres=gpu:h200:2     # use 2 GPU on a shared node.
# #SBATCH --ntasks-per-node=2   # request 2 tasks on that node (1 per gpu).
# #SBATCH --cpus-per-task=12    # request 1/8 of available CPUs on the node *per task*.
# #SBATCH --mem=500000          # grant the job access to 2/8 of the memory on the node.

# ============================================================
# Environment setup
# ============================================================
set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Point cache and checkpoint paths to scratch volume
PTMP_BASE=/ptmp/$USER/byt5-salamanca
OUTPUT_DIR=$PTMP_BASE/output
CHECKPOINTS_DIR=$OUTPUT_DIR/checkpoints
LOG_DIR=$PTMP_BASE/logs

mkdir -p "$OUTPUT_DIR" "$CHECKPOINTS_DIR" "$LOG_DIR"

# Force offline mode for all relevant libraries
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Point HF cache to pre-downloaded location
export HF_HOME=$PTMP_BASE/cache/huggingface
export HF_DATASETS_CACHE=$PTMP_BASE/cache/huggingface/datasets
export HUGGINGFACE_HUB_CACHE=$PTMP_BASE/cache/huggingface/hub

# W&B in offline mode
export WANDB_MODE=offline
export WANDB_DIR=$PTMP_BASE/wandb_offline
export WANDB_ENTITY=mpilhlt
export WANDB_PROJECT=byt5-salamanca-abbr
mkdir -p "$WANDB_DIR"

# CUDA settings
export PYTORCH_ALLOC_CONF=expandable_segments:True

# ============================================================
# Load modules
# ============================================================

# rocm: 6.3, 6.4, 7.0, 7.1, 7.2
# python: condainer/0.1, python-waterboa/2024.06, 2025.06
# tools: amduprof/5.0, 5.2, apptainer/1.4.3, datashare/0.4,
#        git/2.50, git-lfs/3.6, libtool/2.5.3, pandoc/3.1,
#        R/4.5, rclone/1.67.0

module purge
module load gcc/14 rocm/7.2 openmpi/5.0 # Viper: recommended by mpcdf
module load amduprof/5.2 python-waterboa/2025.06

pip install torch --index-url https://download.pytorch.org/whl/rocm7.2
pip install -r requirements.txt

# ============================================================
# Evaluation
# ============================================================
echo "Job $SLURM_JOB_ID started at $(date)"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "GPU:  $(rocm-smi --showproductname 2>/dev/null || echo 'N/A')"

###### Run inside apptainer:
# module load apptainer/1.4.3
# - find suitable apptainer image, install via venv
# CONTAINER="YOUR_CONTAINER"
# srun apptainer exec --nv $CONTAINER python3 train_byt5.py ... 

# - if multi-gpu, instead of `python ...` use either of those:
#   - accelerate launch
#   - torchrun

srun python byt5/train_byt5.py \
    --dataset_local "$PTMP_BASE/datasets/salamanca-abbr/data.jsonl" \
    --eval_model_dir "$PTMP_BASE/output/final_model" \
    --output_dir "$OUTPUT_DIR" \
    --wandb_project byt5-salamanca-abbr \
    --wandb_entity mpilhlt \
    --use_cache \
    --eval_strategy epoch \
    --cap_eval 1000 \
    --save_total_limit 3 \
    --epochs 10 \
    --learning_rate 1e-4 \
    --tokenizer_num_proc 16 \
    --bf16 \
    --oversample_abbr 2.0 \
    --marker_dropout 0.5 \
    --context_lines 1 \
    --train_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --eval_batch_size 64 \
    --max_input_length 512 \
    --max_target_length 384 \
    --seed 42 \
    --eval_only

echo "Eval job $SLURM_JOB_ID finished at $(date)"
