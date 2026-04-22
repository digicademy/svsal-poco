#!/bin/bash
#SBATCH --mail-type=none
#SBATCH --mail-user=wagner@lhlt.mpg.de
#SBATCH --output=tokenize_%j.out
#SBATCH --error=tokenize_%j.err
#SBATCH --job-name=byt5-tokenize
#SBATCH -D .                    # Initial working directory
#SBATCH --nodes=1               # request 1 node
#SBATCH --ntasks-per-node=1     # only start 1 task via srun because Python multiprocessing starts more tasks internally
#SBATCH --cpus-per-task=128     # assign all the cores to that first task to make room for multithreading
#SBATCH --mem=64000
#SBATCH --time=12:00:00

# ============================================================
# Environment setup
# ============================================================
set -euo pipefail

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
export HF_MODULES_CACHE=$PTMP_BASE/cache/huggingface/modules

# Important:
# Set the number of OMP threads *per process* to avoid overloading of the node!
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Clean up stale lock files from previous killed jobs
find $PTMP_BASE/cache -name "*.lock" -delete 2>/dev/null

# ============================================================
# Load modules
# ============================================================

module purge
module load gcc/14 rocm/7.2 openmpi/5.0 python-waterboa/2025.06


pip install torch --index-url https://download.pytorch.org/whl/rocm7.2
pip install -r requirements.txt
# pip install --no-deps -e $PTMP_BASE/svsal-poco

python byt5/train_byt5.py \
    --dataset_local "$PTMP_BASE/datasets/salamanca-abbr/data.jsonl" \
    --model_name "$PTMP_BASE/models/byt5-base" \
    --output_dir "$PTMP_BASE/output" \
    --epochs 0 \
    --eval_strategy no \
    --marker_dropout 0.5 \
    --context_lines 1 \
    --max_input_length 512 \
    --max_target_length 384 \
    --oversample_abbr 2.0 \
    --tokenizer_num_proc 32 \
    --train_batch_size 32 \
    --use_cache


