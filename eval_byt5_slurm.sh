#!/bin/bash
#SBATCH --job-name=byt5-eval-salamanca
#SBATCH --partition=gpu          # check actual partition name
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --output=/ptmp/%u/byt5-salamanca/logs/eval_%j.out
#SBATCH --error=/ptmp/%u/byt5-salamanca/logs/eval_%j.err

# ============================================================
# Environment setup
# ============================================================
set -euo pipefail

# Point cache and checkpoint paths to scratch volume
PTMP_BASE=/ptmp/$USER/byt5-salamanca
OUTPUT_DIR=$PTMP_BASE/output_eval

mkdir -p "$OUTPUT_DIR"

# Force offline mode for all relevant libraries
export HF_HUB_OFFLINE=1
export HF_HOME=$PTMP_BASE/cache/huggingface
export HF_DATASETS_CACHE=$PTMP_BASE/cache/huggingface/datasets
export HUGGINGFACE_HUB_CACHE=$PTMP_BASE/cache/huggingface/hub
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export WANDB_MODE=disabled

# CUDA settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# - find suitable apptainer image, instal via venv
#   - `module load rocm` instead of CUDA if on AMD
#   - `pip install torch --index-url https://download.pytorch.org/whl/rocm6.x`
module load cuda anaconda     # or whatever the center provides
conda activate byt5           # your pre-built environment

echo "Job $SLURM_JOB_ID started at $(date)"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# - if multi-gpu, instead of `python ...` use either of those:
#   - accelerate launch
#   - torchrun
python train_byt5.py \
    --dataset_local "$PTMP_BASE/datasets/salamanca-abbr/data.jsonl" \
    --output_dir "$OUTPUT_DIR" \
    --eval_model_dir "$PTMP_BASE/models/byt5-salamanca-abbr-hub" \
    --eval_only \
    --wandb_project byt5-salamanca-abbr \
    --wandb_entity mpilhlt \
    --epochs 10 \
    --learning_rate 1e-4 \
    --oversample_abbr 2.0 \
    --train_batch_size 64 \
    --eval_batch_size 128 \
    --eval_strategy "epoch" \
    --cap_eval         1000 \
    --gradient_accumulation_steps 2 \
    --max_input_length 256 \
    --max_target_length 192 \
    --tokenizer_num_proc 8 \
    --bf16 \
    --use_cache \
    --lang_prefix \
    --seed 42

echo "Eval job $SLURM_JOB_ID finished at $(date)"
