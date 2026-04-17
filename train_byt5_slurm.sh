#!/bin/bash
#SBATCH --job-name=byt5-salamanca
#SBATCH --partition=gpu          # check actual partition name
#SBATCH --gres=gpu:a100:1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16

export WANDB_MODE=offline
export WANDB_DIR=/ptmp/$USER/byt5-salamanca/wandb_offline

# - find suitable apptainer image, instal via venv
#   - `module load rocm` instead of CUDA if on AMD
#   - `pip install torch --index-url https://download.pytorch.org/whl/rocm6.x`
module load cuda anaconda     # or whatever the center provides
conda activate byt5           # your pre-built environment

# - point cache and checkpoint paths to scratch volume
# - handle auth via .env file instead of uv secrets
# - if internet access is restricted:
#   - use WANDB_MODE=offline and sync logs afterward
#   - check HF hub functions?

# - if multi-gpu, instead of `python ...` use either of those:
#   - accelerate launch
#   - torchrun
python train_byt5.py \
    --dataset_repo mpilhlt/salamanca-abbr \
    --output_repo  mpilhlt/byt5-salamanca-abbr \
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
    --tokenizer_num_proc 16 \
    --bf16 \
    --use_cache \
    --lang_prefix
