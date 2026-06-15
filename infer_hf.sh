#!/bin/sh
set -euo pipefail

# Hugging Face Jobs inference launcher.
# Runs recursive batch inference via infer_local_batch.sh.

# ------------------------------------------------------------------
# Configuration — adjust these values for your run
# ------------------------------------------------------------------
INPUT_DIR="./infer_input"
OUTPUT_DIR="./infer_output"
MODE="xml"                    # text | xml | jsonl
EXTENSIONS=""                 # optional CSV without dots, e.g. "xml,tei"
BOUNDARY_MODEL_TYPE="flair"   # canine | flair
BATCH_SIZE=32
LANG_PREFIX=0                 # 1 enables --lang-prefix
TEXT_OUTPUT="text"            # text | jsonl (only relevant in mode=text)

# Model repos and local download locations inside the job workspace.
BOUNDARY_MODEL_REPO="mpilhlt/latin-contextual-lb-detector"
BYT5_MODEL_REPO="mpilhlt/byt5-salamanca-abbr"
BOUNDARY_MODEL_DIR="./models/latin-contextual-lb-detector"
BYT5_MODEL_DIR="./models/byt5-salamanca-abbr"

EXTRA_ARGS=""
if [ -n "$EXTENSIONS" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --extensions $EXTENSIONS"
fi
if [ "$BOUNDARY_MODEL_TYPE" != "canine" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --boundary-model-type $BOUNDARY_MODEL_TYPE"
fi
if [ "$LANG_PREFIX" = "1" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --lang-prefix"
fi

hf jobs uv run \
  --flavor a100-large \
  --timeout 2h \
  --label Salamanca \
  --label task=inference \
  --secrets HF_TOKEN \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --with 'transformers>=4.40.0' \
  --with 'datasets>=2.18.0' \
  --with 'evaluate>=0.4.0' \
  --with 'scikit-learn>=1.3.0' \
  --with 'accelerate>=1.1.0' \
  --with 'torch==2.6.0' \
  --with jiwer \
  --with tensorboard \
  --with codecarbon \
  --with 'huggingface-hub>=0.22.0' \
  --with 'git+https://github.com/digicademy/svsal-poco' \
  bash -lc "\
    mkdir -p \"$OUTPUT_DIR\" ./models && \
    hf download --repo-type model \"$BOUNDARY_MODEL_REPO\" --local-dir \"$BOUNDARY_MODEL_DIR\" && \
    hf download --repo-type model \"$BYT5_MODEL_REPO\" --local-dir \"$BYT5_MODEL_DIR\" && \
    bash infer_local_batch.sh \
      --mode \"$MODE\" \
      --input-dir \"$INPUT_DIR\" \
      --output-dir \"$OUTPUT_DIR\" \
      --batch-size \"$BATCH_SIZE\" \
      --text-output \"$TEXT_OUTPUT\" \
      --boundary-model-dir \"$BOUNDARY_MODEL_DIR\" \
      --byt5-model-dir \"$BYT5_MODEL_DIR\" \
      $EXTRA_ARGS"
