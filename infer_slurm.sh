#!/bin/bash
#
# infer_slurm.sh
# Batch inference over a directory of input files, running on an HPC cluster
# via SLURM.  Discovers all matching files under INPUT_DIR and processes them
# through the full svsal-poco pipeline (boundary detection + ByT5 expansion),
# writing results to OUTPUT_DIR with mirrored relative paths.
#
# Delegates the actual file iteration to infer_local_batch.sh, which in turn
# calls infer_handler.py for each file.
#
# ------------------------------------------------------------------
# Prerequisites
# ------------------------------------------------------------------
# Run prepare_viper.sh first to set up the base environment.  Then also
# download the inference models into PTMP_BASE/models/ if they are not
# already there (e.g. from a previous training run):
#
#   hf download --repo-type model \
#       --local-dir $PTMP_BASE/models/byt5-salamanca-abbr \
#       mpilhlt/byt5-salamanca-abbr
#
#   hf download --repo-type model \
#       --local-dir $PTMP_BASE/models/canine-salamanca-boundary-classifier \
#       mpilhlt/canine-salamanca-boundary-classifier
#
# If you trained your own ByT5 model, point BYT5_MODEL_DIR at the output
# directory of that training run instead.
#
# ------------------------------------------------------------------
# Useful SLURM commands
# ------------------------------------------------------------------
#   sbatch infer_slurm.sh           - submit the job
#   scontrol show job <job>         - job details
#   squeue | grep <username>        - queued/running jobs for this user
#   tail -f infer_<job>.{err,out}   - live log monitoring
#
#SBATCH --mail-type=none
#SBATCH --mail-user=wagner@lhlt.mpg.de
#SBATCH --output=infer_%j.out
#SBATCH --error=infer_%j.err
#SBATCH --job-name=poco-infer
#SBATCH -D .                    # Initial working directory

#SBATCH --constraint="apu"
#SBATCH --nodes=1

# --- Change the following for testing the workflow/GPU setup ---
#SBATCH --time=00:15:00
# #SBATCH --partition=apu
#SBATCH --partition=apudev      # apudev: for testing, 1 node with 2 MI300, 15 min. walltime

# --- VIPER default case: use a single APU on a shared node ---
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110000

# --- VIPER alternative case: two APUs on a shared node ---
# #SBATCH --gres=gpu:2
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=32
# #SBATCH --mem=220000

# --- DAIS: H200 on a shared node ---
# #SBATCH --partition="gpu1"
# #SBATCH --gres=gpu:h200:1
# #SBATCH --cpus-per-task=12
# #SBATCH --mem=250000

# ============================================================
# Configuration — adjust these for each inference run
# ============================================================

PTMP_BASE=/ptmp/$USER/byt5-salamanca

# --- Input / output ----------------------------------------
# Directory of files to process (sub-directories are searched recursively).
INPUT_DIR="$PTMP_BASE/infer/input"
OUTPUT_DIR="$PTMP_BASE/infer/output"

# Processing mode: text | jsonl | xml
MODE="xml"

# File extensions to match (comma-separated, without dots).
# Leave empty to use the mode default:
#   xml   -> xml,tei
#   text  -> txt,text
#   jsonl -> jsonl
EXTENSIONS=""

# --- Models ------------------------------------------------
# Path to the trained ByT5 model directory (local path or HF-loadable path
# within the offline HF cache).
# Use $PTMP_BASE/output/final_model if you want the freshly trained checkpoint.
BYT5_MODEL_DIR="$PTMP_BASE/models/byt5-salamanca-abbr"

# HF model identifier for the abbreviation (BYT5) model (only used when BYT5_MODEL_DIR
# is empty, i.e. the model is loaded from the local HF cache).
BYT5_MODEL_NAME=""

# Local boundary model directory.
# For a canine model it must contain best_model.pt and threshold.json.
# Set to "" and use BOUNDARY_MODEL_NAME instead to load from the HF cache.
BOUNDARY_MODEL_DIR="$PTMP_BASE/models/canine-salamanca-boundary-classifier"

# HF model identifier for the boundary model (only used when BOUNDARY_MODEL_DIR
# is empty, i.e. the model is loaded from the local HF cache).
BOUNDARY_MODEL_NAME=""

# Boundary model type: canine | flair
BOUNDARY_MODEL_TYPE="canine"

# --- Inference settings ------------------------------------
# Per-call batch size forwarded to infer_handler.py.
# A single APU (MI300X) can typically handle 32-64 for ByT5-base.
BATCH_SIZE=32

# Set to 1 to pass --lang-prefix to the inference handler.
LANG_PREFIX=0

# --- Script / package location ----------------------------
# Directory of the local svsal-poco clone used for pip install.
# Adjust if you placed the repo somewhere other than $HOME.
REPO_DIR="$HOME/svsal-poco"

# ============================================================
# Environment setup
# ============================================================
set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mkdir -p "$OUTPUT_DIR"

# Force offline mode — compute nodes typically have no outbound internet access.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Point HF cache to the pre-downloaded location on scratch.
export HF_HOME=$PTMP_BASE/cache/huggingface
export HF_DATASETS_CACHE=$PTMP_BASE/cache/huggingface/datasets
export HUGGINGFACE_HUB_CACHE=$PTMP_BASE/cache/huggingface/hub
export HF_MODULES_CACHE=$PTMP_BASE/cache/huggingface/modules

export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# Clean up stale lock files left by any previously killed jobs.
find "$PTMP_BASE/cache" -name "*.lock" -delete 2>/dev/null || true

# ============================================================
# Load modules
# ============================================================

# rocm: 6.3, 6.4, 7.0, 7.1, 7.2
# python: condainer/0.1, python-waterboa/2024.06, 2025.06
# tools: amduprof/5.0, 5.2, apptainer/1.4.3, datashare/0.4,
#        git/2.50, git-lfs/3.6, libtool/2.5.3, pandoc/3.1,
#        R/4.5, rclone/1.67.0

module purge
module load gcc/14 rocm/7.2 openmpi/5.0   # Viper: recommended by mpcdf
module load amduprof/5.2 python-waterboa/2025.06

pip install torch --index-url https://download.pytorch.org/whl/rocm7.2
pip install -r "$REPO_DIR/requirements.txt"
pip install --no-deps --force-reinstall -e "$REPO_DIR"

# ============================================================
# Inference
# ============================================================
echo "Job $SLURM_JOB_ID started at $(date)"
echo "Node:       $(hostname)"
echo "GPU:        $(rocm-smi --showproductname 2>/dev/null || echo 'N/A')"
echo "---"
echo "Input dir:  $INPUT_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Mode:       $MODE"
echo "Batch size: $BATCH_SIZE"
echo "ByT5 model: $BYT5_MODEL_DIR"
echo "Boundary:   ${BOUNDARY_MODEL_DIR:-$BOUNDARY_MODEL_NAME}"
echo "---"

# Boundary model: --boundary-model-dir and --boundary-model-name are mutually
# exclusive in infer_local_batch.sh, so build the argument here.
# BOUNDARY_ARGS=()
# if [[ -n "$BOUNDARY_MODEL_DIR" ]]; then
#    BOUNDARY_ARGS+=(--boundary-model-dir "$BOUNDARY_MODEL_DIR")
# elif [[ -n "$BOUNDARY_MODEL_NAME" ]]; then
#    BOUNDARY_ARGS+=(--boundary-model-name "$BOUNDARY_MODEL_NAME")
# else
#    echo "ERROR: Set either BOUNDARY_MODEL_DIR or BOUNDARY_MODEL_NAME." >&2
#    exit 1
# fi

# Collect optional flags.
EXTRA_ARGS=()
[[ "$BOUNDARY_MODEL_TYPE" != "canine" ]] && EXTRA_ARGS+=(--boundary-model-type "$BOUNDARY_MODEL_TYPE")
[[ -n "$EXTENSIONS" ]]                   && EXTRA_ARGS+=(--extensions "$EXTENSIONS")
[[ "$LANG_PREFIX" == "1" ]]              && EXTRA_ARGS+=(--lang-prefix)

srun bash "$REPO_DIR/infer_local_batch.sh" \
    --mode          "$MODE" \
    --input-dir     "$INPUT_DIR" \
    --output-dir    "$OUTPUT_DIR" \
    --byt5-model-dir "$BYT5_MODEL_DIR" \
    --boundary-model-dir "$BOUNDARY_MODEL_DIR" \
    --batch-size    "$BATCH_SIZE" \
    "${EXTRA_ARGS[@]}"
#    "${BOUNDARY_ARGS[@]}" \

echo "Inference job $SLURM_JOB_ID finished at $(date)"
