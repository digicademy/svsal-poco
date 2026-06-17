#!/usr/bin/env bash
set -euo pipefail

# Hugging Face Jobs inference launcher.
# Runs recursive batch inference via infer_local_batch.sh.

# ------------------------------------------------------------------
# Configuration — adjust these values for your run
# ------------------------------------------------------------------
INPUT_DIR="${INPUT_DIR:-./infer_input}"
OUTPUT_DIR="${OUTPUT_DIR:-./infer_output}"
OUTPUT_BUCKET="${OUTPUT_BUCKET:-mpilhlt/svsal-poco-output}"
MODE="${MODE:-xml}"                         # text | xml | jsonl
EXTENSIONS="${EXTENSIONS:-}"                # optional CSV without dots, e.g. "xml,tei"
INPUT_FILE_URLS="${INPUT_FILE_URLS:-}"      # optional CSV of file URLs to download into INPUT_DIR
BOUNDARY_MODEL_TYPE="${BOUNDARY_MODEL_TYPE:-flair}"   # canine | flair
BATCH_SIZE="${BATCH_SIZE:-32}"
LANG_PREFIX="${LANG_PREFIX:-0}"             # 1 enables --lang-prefix
TEXT_OUTPUT="${TEXT_OUTPUT:-text}"          # text | jsonl (only relevant in mode=text)
FLAVOR="${FLAVOR:-a10g-small}"              # A10G small: 1$/h; runtime for a 15MB work: c. 3.5h
TIMEOUT="${TIMEOUT:-12h}"

# Model repos and local download locations inside the job workspace.
BOUNDARY_MODEL_REPO="${BOUNDARY_MODEL_REPO:-mschonhardt/latin-contextual-lb-detector}"
BYT5_MODEL_REPO="${BYT5_MODEL_REPO:-mpilhlt/byt5-salamanca-abbr}"
BOUNDARY_MODEL_DIR="${BOUNDARY_MODEL_DIR:-./models/latin-contextual-lb-detector}"
BYT5_MODEL_DIR="${BYT5_MODEL_DIR:-./models/byt5-salamanca-abbr}"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >&2
}

usage() {
  cat <<'EOF'
Usage:
  ./infer_hf.sh [options]

Options:
  --input-dir <path>
  --output-dir <path>
  --output-bucket <owner/name>
  --mode <text|xml|jsonl>
  --extensions <csv_no_dot>
  --input-file-urls <csv_urls>
  --input-file-url <url>            # can be repeated
  --boundary-model-type <canine|flair>
  --boundary-model-repo <owner/name>
  --byt5-model-repo <owner/name>
  --boundary-model-dir <path>
  --byt5-model-dir <path>
  --batch-size <int>
  --text-output <text|jsonl>
  --lang-prefix
  --flavor <t4-small|t4-medium|a10g-small|a10g-large|a10g-largex2|a10g-largex4|a100-large>
  --timeout <duration>              # e.g. 12h
  -h, --help
EOF
}

append_csv() {
  value="$1"
  if [ -z "$INPUT_FILE_URLS" ]; then
    INPUT_FILE_URLS="$value"
  else
    INPUT_FILE_URLS="${INPUT_FILE_URLS},$value"
  fi
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --input-dir) INPUT_DIR="${2:-}"; shift 2 ;;
    --output-dir) OUTPUT_DIR="${2:-}"; shift 2 ;;
    --output-bucket) OUTPUT_BUCKET="${2:-}"; shift 2 ;;
    --mode) MODE="${2:-}"; shift 2 ;;
    --extensions) EXTENSIONS="${2:-}"; shift 2 ;;
    --input-file-urls) INPUT_FILE_URLS="${2:-}"; shift 2 ;;
    --input-file-url) append_csv "${2:-}"; shift 2 ;;
    --boundary-model-type) BOUNDARY_MODEL_TYPE="${2:-}"; shift 2 ;;
    --boundary-model-repo) BOUNDARY_MODEL_REPO="${2:-}"; shift 2 ;;
    --byt5-model-repo) BYT5_MODEL_REPO="${2:-}"; shift 2 ;;
    --boundary-model-dir) BOUNDARY_MODEL_DIR="${2:-}"; shift 2 ;;
    --byt5-model-dir) BYT5_MODEL_DIR="${2:-}"; shift 2 ;;
    --batch-size) BATCH_SIZE="${2:-}"; shift 2 ;;
    --text-output) TEXT_OUTPUT="${2:-}"; shift 2 ;;
    --lang-prefix) LANG_PREFIX=1; shift 1 ;;
    --flavor) FLAVOR="${2:-}"; shift 2 ;;
    --timeout) TIMEOUT="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

case "$FLAVOR" in
  t4-small|t4-medium|a10g-small|a10g-large|a10g-largex2|a10g-largex4|a100-large) ;;
  *) echo "ERROR: Unsupported --flavor '$FLAVOR'" >&2; exit 1 ;;
esac
case "$MODE" in
  text|xml|jsonl) ;;
  *) echo "ERROR: Unsupported --mode '$MODE'" >&2; exit 1 ;;
esac
case "$BOUNDARY_MODEL_TYPE" in
  canine|flair) ;;
  *) echo "ERROR: Unsupported --boundary-model-type '$BOUNDARY_MODEL_TYPE'" >&2; exit 1 ;;
esac
case "$TEXT_OUTPUT" in
  text|jsonl) ;;
  *) echo "ERROR: Unsupported --text-output '$TEXT_OUTPUT'" >&2; exit 1 ;;
esac

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

# Function to extract filename from Content-Disposition header (simplified and robust)
extract_filename_from_header() {
  local url="$1"
  
  # Get the Content-Disposition header
  local content_disposition=$(curl -sI "$url" | grep -i "content-disposition" | head -1)
  
  if [ -n "$content_disposition" ]; then
    # Try to extract filename from the header using a more robust method
    # First, try to match the UTF-8 encoded version
    local filename=$(echo "$content_disposition" | grep -o 'filename\*\=[^;]*' | cut -d'=' -f2- | sed "s/^UTF-8''//")
    if [ -n "$filename" ]; then
      echo "$filename"
      return
    fi
    
    # Then try quoted version
    local filename=$(echo "$content_disposition" | grep -o 'filename="[^"]*"' | cut -d'"' -f2)
    if [ -n "$filename" ]; then
      echo "$filename"
      return
    fi
    
    # Then try unquoted version
    local filename=$(echo "$content_disposition" | grep -o 'filename=[^;]*' | cut -d'=' -f2)
    if [ -n "$filename" ]; then
      echo "$filename"
      return
    fi
  fi
  
  # If no filename found in headers, fall back to basename
  basename "$url"
}

# Build the download command for the container
DOWNLOAD_CMD=""
if [ -n "$INPUT_FILE_URLS" ]; then
  log "Build download command for INPUT_FILE_URLS: $INPUT_FILE_URLS"
  IFS=',' read -ra URLS <<< "$INPUT_FILE_URLS"
  for i in "${!URLS[@]}"; do
    url="${URLS[i]}"
    
    # Extract filename properly
    filename=$(extract_filename_from_header "$url")
    
    # Clean up filename
    filename=$(echo "$filename" | sed 's/"//g' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    
    # If filename is still problematic, use basename as fallback
    if [[ "$filename" == *"download"* ]] || [ -z "$filename" ]; then
      filename=$(basename "$url")
      # If basename is still just "download", give it a sensible default
      if [[ "$filename" == "download" ]]; then
        filename="document.xml"  # Change this default as needed
      fi
    fi
    
    log "I will be downloading the file $((i+1))/${#URLS[@]}: $url -> $INPUT_DIR/$filename"
    DOWNLOAD_CMD+="curl -L --retry 3 --retry-delay 5 -o \"$INPUT_DIR/$filename\" \"$url\" && echo \"Downloaded: $url as $filename\" || (echo \"Failed to download: $url\"; exit 1); "
  done
else
  log "No INPUT_FILE_URLS configured; skipping preparatory input download."
fi

# Build bucket mount and upload commands if a bucket was specified
if [ -n "$OUTPUT_BUCKET" ]; then
  BUCKET_MOUNT_CMD="hf mount start bucket $OUTPUT_BUCKET /tmp/data"
  BUCKET_MOUNT_PARA=("-v" "hf://buckets/$OUTPUT_BUCKET:/tmp/data")
  BUCKET_UPLOAD_CMD="install -m 644 $OUTPUT_DIR/* /tmp/data/"
fi

# Create a summary log file
SUMMARY_LOG="/tmp/job_summary_$(date +%s).log"

log "Starting Hugging Face Job with configuration:"
log "  Input Directory: $INPUT_DIR"
log "  Output Directory: $OUTPUT_DIR"
log "  Output Bucket: $OUTPUT_BUCKET"
log "  Mode: $MODE"
log "  Batch Size: $BATCH_SIZE"
log "  Flavor: $FLAVOR"
log "  Boundary Model Repo: $BOUNDARY_MODEL_REPO"
log "  BYT5 Model Repo: $BYT5_MODEL_REPO"

# Write summary to file for later retrieval
{
  echo "Job Summary - $(date)"
  echo "=================="
  echo "Input Directory: $INPUT_DIR"
  echo "Output Directory: $OUTPUT_DIR"
  echo "Output Bucket: $OUTPUT_BUCKET"
  echo "Mode: $MODE"
  echo "Batch Size: $BATCH_SIZE"
  echo "Flavor: $FLAVOR"
  echo "Boundary Model Repo: $BOUNDARY_MODEL_REPO"
  echo "BYT5 Model Repo: $BYT5_MODEL_REPO"
  echo "Input Files: $INPUT_FILE_URLS"
  echo "Extra Args: $EXTRA_ARGS"
  echo ""
} > "$SUMMARY_LOG"

echo "BUCKET_MOUNT_PARA: ${BUCKET_MOUNT_PARA[@]}"
echo "BUCKET_UPLOAD_CMD: ${BUCKET_UPLOAD_CMD}"

# Main execution - everything happens in the container
log "Executing Hugging Face Job..."
hf jobs uv run \
  --flavor "$FLAVOR" \
  --timeout "$TIMEOUT" \
  --label Salamanca \
  --label task=inference \
  --secrets HF_TOKEN \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env INPUT_DIR="$INPUT_DIR" \
  --env INPUT_FILE_URLS="$INPUT_FILE_URLS" \
  "${BUCKET_MOUNT_PARA[@]}" \
  --with 'transformers>=4.40.0' \
  --with 'datasets>=2.18.0' \
  --with 'evaluate>=0.4.0' \
  --with 'scikit-learn>=1.3.0' \
  --with 'accelerate>=1.1.0' \
  --with 'torch==2.8.0' \
  --with jiwer \
  --with tensorboard \
  --with codecarbon \
  --with 'huggingface-hub>=0.22.0' \
  --with 'git+https://github.com/digicademy/svsal-poco' \
  bash -c "
    set -euxo pipefail \\
    
    # Define logging function inline
    log() { echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] \$*\" >&2; } \\
    
    # Create directories
    mkdir -p \"$INPUT_DIR\" \"$OUTPUT_DIR\" ./models && \\
    log 'Directories created' \\

    # Test bucket
    echo "test" > "${OUTPUT_DIR}/test" && \\
    $BUCKET_UPLOAD_CMD && \\
    cat /tmp/data/test && \\
    rm /tmp/data/test && \\
    rm "${OUTPUT_DIR}/test" && \\

    # Log start of download phase
    log 'Starting download phase...' \\
    
    # Download files if specified
    $DOWNLOAD_CMD \\
    
    # Get inferencing scripts
    git clone https://github.com/digicademy/svsal-poco.git && \\
    cp svsal-poco/infer_*.* . \\

    # Log model download start
    log 'Downloading boundary model...' && \\
    hf download --repo-type model \"$BOUNDARY_MODEL_REPO\" --local-dir \"$BOUNDARY_MODEL_DIR\" \\
    
    log 'Downloading BYT5 model...' && \\
    hf download --repo-type model \"$BYT5_MODEL_REPO\" --local-dir \"$BYT5_MODEL_DIR\" \\
    
    # Log completion of setup
    log 'Setup complete. Starting inference...' \\
    
    # Run inference
    bash infer_local_batch.sh \\
      --mode \"$MODE\" \\
      --input-dir \"$INPUT_DIR\" \\
      --output-dir \"$OUTPUT_DIR\" \\
      --batch-size \"$BATCH_SIZE\" \\
      --text-output \"$TEXT_OUTPUT\" \\
      --boundary-model-dir \"$BOUNDARY_MODEL_DIR\" \\
      --byt5-model-dir \"$BYT5_MODEL_DIR\" \\
      $EXTRA_ARGS \\
    
    # Log completion
    log 'Inference completed successfully!' \\
    
    # Copy summary log to output directory for easy retrieval
    cp \"$SUMMARY_LOG\" \"$OUTPUT_DIR/job_summary.log\" 2>/dev/null || true \\

    # Upload to bucket if specified
    ls -la "${OUTPUT_DIR}" && \\
    $BUCKET_UPLOAD_CMD && \\
    ls -la /tmp/data
    echo "ALL DONE"
"
