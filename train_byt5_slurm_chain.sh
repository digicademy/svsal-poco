#!/bin/bash
#SBATCH --mail-type=none
#SBATCH --mail-user=wagner@lhlt.mpg.de
# #SBATCH --output=/ptmp/%u/byt5-salamanca/logs/train_%j.out
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err
#SBATCH --job-name=byt5-chain
#SBATCH --time=00:12:00          # leave 3min margin
#SBATCH --signal=B:SIGUSR1@900   # send signal 3min before wall time

# Specify chain: indexes (max. 30000) of the job array elements (max. 300 - the default job submit limit per user)
#SBATCH --array=1-20

#SBATCH -D ./                   # Initial working directory

#SBATCH --partition=apudev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=48
#SBATCH --gres=gpu:2             # Request 2 APUs per node.
#SBATCH --mem=64G

# -----------------------------------------------------------------------
# If your HPC supports SLURM job dependencies, you can pre-submit a chain
# Submit a chain of 5 jobs (each depends on the previous one finishing)
JOB1=$(sbatch --parsable train_hpc.slurm)
JOB2=$(sbatch --parsable --dependency=afterany:$JOB1 train_hpc.slurm)
JOB3=$(sbatch --parsable --dependency=afterany:$JOB2 train_hpc.slurm)
JOB4=$(sbatch --parsable --dependency=afterany:$JOB3 train_hpc.slurm)
JOB5=$(sbatch --parsable --dependency=afterany:$JOB4 train_hpc.slurm)
echo "Submitted chain: $JOB1 -> $JOB2 -> $JOB3 -> $JOB4 -> $JOB5"
exit
# -----------------------------------------------------------------------


# ============================================================
# Environment setup
# ============================================================
#  the environment variable $SLURM_ARRAY_TASK_ID holds the index of the job array and
#  can be used to discriminate between individual elements of the job array

set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Point cache and checkpoint paths to scratch volume
PTMP_BASE=/ptmp/$USER/byt5-salamanca
OUTPUT_DIR=$PTMP_BASE/output
CHECKPOINTS_DIR=$OUTPUT_DIR/checkpoints
LOG_DIR=$PTMP_BASE/logs
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

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



# -- Signal handler: on SIGUSR1, save a checkpoint and exit gracefully --
# The Trainer will catch SIGTERM to save, but we set up a trap too
RESUBMIT=true
handle_signal() {
    echo "$(date): Received signal, training will checkpoint and exit."
    # The HF Trainer handles SIGTERM gracefully (saves checkpoint + exits)
    # We just need to resubmit after
    if [ -n "${TRAIN_PID:-}" ]; then
        kill -SIGTERM $TRAIN_PID 2>/dev/null
        wait $TRAIN_PID 2>/dev/null
    fi
}
trap handle_signal SIGUSR1 SIGTERM

# -- Check if training is already complete --
TRAINER_STATE="$OUTPUT_DIR/checkpoints/trainer_state.json"
if [ -f "$TRAINER_STATE" ]; then
    # Check if best_model_checkpoint is set and all epochs are done
    COMPLETED=$(python -c "
import json, sys
state = json.load(open('$TRAINER_STATE'))
# If we've reached the target epochs, training is done
target_epochs = 10
current_epoch = state.get('epoch', 0)
print('yes' if current_epoch >= target_epochs else 'no')
" 2>/dev/null || echo "no")

    if [ "$COMPLETED" = "yes" ]; then
        echo "Training appears complete. Running final evaluation..."
        RESUBMIT=false
    fi
fi

# -- Load modules --
# module load python/3.11 cuda/12.1 cudnn/8.9
# source activate byt5
module purge
module load gcc/14 rocm/7.2 openmpi/5.0 # Viper: recommended by mpcdf
module load amduprof/5.2 python-waterboa/2025.06

pip install torch --index-url https://download.pytorch.org/whl/rocm7.2
pip install -r requirements.txt

echo "=== Job $SLURM_JOB_ID started at $(date) on $(hostname) ==="

# -- Run training (will auto-resume from latest checkpoint) --
srun python byt5/train_byt5.py \
    --dataset_local "$PTMP_BASE/datasets/salamanca-abbr/data.jsonl" \
    --model_name "$PTMP_BASE/models/byt5-base" \
    --output_dir "$OUTPUT_DIR" \
    --wandb_project byt5-salamanca-abbr \
    --wandb_entity mpilhlt \
    --epochs 10 \
    --learning_rate 1e-4 \
    --oversample_abbr 2.0 \
    --train_batch_size 64 \
    --eval_batch_size 128 \
    --eval_strategy "epoch" \
    --cap_eval 1000 \
    --gradient_accumulation_steps 2 \
    --max_input_length 256 \
    --max_target_length 192 \
    --tokenizer_num_proc 16 \
    --bf16 \
    --use_cache \
    --lang_prefix \
    --save_total_limit 3 &

TRAIN_PID=$!
wait $TRAIN_PID
TRAIN_EXIT=$?

echo "=== Training exited with code $TRAIN_EXIT at $(date) ==="

# -- Resubmit if not done --
if [ "$RESUBMIT" = "true" ] && [ $TRAIN_EXIT -ne 0 ]; then
    # Check if we have a checkpoint (i.e., training made progress)
    if ls "$OUTPUT_DIR/checkpoints/checkpoint-"* 1>/dev/null 2>&1; then
        echo "Resubmitting job for continuation..."
        sbatch "$SCRIPT_DIR/$(basename $0)"
    else
        echo "ERROR: No checkpoint found and training failed. Not resubmitting."
        exit 1
    fi
elif [ "$RESUBMIT" = "true" ]; then
    # Exit code 0 but RESUBMIT still true means training finished normally
    # in this segment — check if all epochs done
    if [ -f "$TRAINER_STATE" ]; then
        DONE=$(python -c "
import json
state = json.load(open('$TRAINER_STATE'))
print('yes' if state.get('epoch', 0) >= 10 else 'no')
" 2>/dev/null || echo "no")
        if [ "$DONE" = "no" ]; then
            echo "Not all epochs complete. Resubmitting..."
            sbatch "$SCRIPT_DIR/$(basename $0)"
        else
            echo "All epochs complete! Training finished."
        fi
    fi
fi
