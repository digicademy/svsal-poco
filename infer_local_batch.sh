#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------
# infer_local_batch.sh
# Batch runner over directories for svsal-poco local inference.
#
# Calls infer_local.sh once per matching file and writes outputs with
# mirrored relative paths under --output-dir.
# ------------------------------------------------------------------

MODE=""
INPUT_DIR=""
OUTPUT_DIR=""
BOUNDARY_MODEL_DIR=""
BOUNDARY_MODEL_NAME=""
BOUNDARY_MODEL_TYPE="canine"
BYT5_MODEL_DIR=""
BYT5_MODEL_NAME=""
WRAPPER_SCRIPT=""
BATCH_SIZE="16"
LANG_PREFIX="0"
TEXT_OUTPUT="text"
PYTHON_BIN="python"

# Comma-separated extension overrides (without dots), optional.
# If omitted, defaults by mode:
#   text  -> txt,text
#   xml   -> xml,tei
#   jsonl -> jsonl
EXTENSIONS=""

usage() {
  cat <<'EOF'
Usage:
  ./infer_local_batch.sh \
    --mode <text|xml|jsonl> \
    --input-dir <path> \
    --output-dir <path> \
    --byt5-model-dir <path> \
    --boundary-model-dir <path> \
    --boundary-model-name <string> \
    [--boundary-model-type <canine|flair>] \
    [--wrapper <path_to_infer_local.sh>] \
    [--batch-size <int>] \
    [--lang-prefix] \
    [--text-output <text|jsonl>] \
    [--python <python_bin>] \
    [--extensions <csv_no_dot>] \
    [--dry-run]

Required:
  --mode
  --input-dir
  --output-dir
  --byt5-model-dir

The boundary model must be specified, but it can be 
specified either with a reference to a local directory:
  --boundary-model-dir
or with a hugging face model identifier:
  --boundary-model-name
If it is a flair model, that must be specified with:
  --boundary-model-type flair
Otherwise the default boundary model type is "canine"

Optional:
  --wrapper       Path to infer_local.sh (default: same directory as this script)
  --batch-size    Default 16
  --lang-prefix   Forwarded to infer_local.sh
  --text-output   For mode=text: text (default) or jsonl
  --python        Forwarded to infer_local.sh
  --extensions    Override extensions, e.g. "txt,md" or "xml"
  --dry-run       Print planned operations without executing

Examples:
  ./infer_local_batch.sh \
    --mode text \
    --input-dir ./in_txt \
    --output-dir ./out_txt \
    --boundary-model-dir ./canine-salamanca-boundary-classifier \
    --byt5-model-dir ./byt5-salamanca-abbr

  ./infer_local_batch.sh \
    --mode xml \
    --input-dir ./in_xml \
    --output-dir ./out_xml \
    --boundary-model-dir ./canine-salamanca-boundary-classifier \
    --byt5-model-dir ./byt5-salamanca-abbr \
    --extensions xml,tei
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

DRY_RUN="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="${2:-}"; shift 2 ;;
    --input-dir) INPUT_DIR="${2:-}"; shift 2 ;;
    --output-dir) OUTPUT_DIR="${2:-}"; shift 2 ;;
    --boundary-model-dir) BOUNDARY_MODEL_DIR="${2:-}"; shift 2 ;;
    --boundary-model-name) BOUNDARY_MODEL_NAME="${2:-}"; shift 2 ;;
    --boundary-model-type) BOUNDARY_MODEL_TYPE="${2:-}"; shift 2 ;;
    --byt5-model-dir) BYT5_MODEL_DIR="${2:-}"; shift 2 ;;
    --byt5-model-name) BYT5_MODEL_NAME="${2:-}"; shift 2 ;;
    --wrapper) WRAPPER_SCRIPT="${2:-}"; shift 2 ;;
    --batch-size) BATCH_SIZE="${2:-}"; shift 2 ;;
    --lang-prefix) LANG_PREFIX="1"; shift 1 ;;
    --text-output) TEXT_OUTPUT="${2:-}"; shift 2 ;;
    --python) PYTHON_BIN="${2:-}"; shift 2 ;;
    --extensions) EXTENSIONS="${2:-}"; shift 2 ;;
    --dry-run) DRY_RUN="1"; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown argument: $1 (use --help)" ;;
  esac
done

[[ -n "$MODE" ]] || die "--mode is required"
[[ -n "$INPUT_DIR" ]] || die "--input-dir is required"
[[ -n "$OUTPUT_DIR" ]] || die "--output-dir is required"
# [[ -n "$BYT5_MODEL_DIR" ]] || die "--byt5-model-dir is required"

[[ -n "$BOUNDARY_MODEL_DIR" && -n "$BOUNDARY_MODEL_NAME" ]] \
  && die "boundary model must be specified with --boundary-model-dir or --boundary-model-name"
[[ "$BOUNDARY_MODEL_DIR" != "" && "$BOUNDARY_MODEL_NAME" != "" ]] \
  && die "boundary model must be specified with either --boundary-model-dir or --boundary-model-name, not both"

[[ -n "$BYT5_MODEL_DIR" && -n "$BYT5_MODEL_NAME" ]] \
  && die "abbrev model must be specified with --byt5-model-dir or --byt5-model-name"
[[ "$BYT5_MODEL_DIR" != "" && "$BYT5_MODEL_NAME" != "" ]] \
  && die "abbrev model must be specified with either --byt5-model-dir or --byt5-model-name, not both"

[[ "$MODE" == "text" || "$MODE" == "xml" || "$MODE" == "jsonl" ]] \
  || die "--mode must be text|xml|jsonl"

[[ "$TEXT_OUTPUT" == "text" || "$TEXT_OUTPUT" == "jsonl" ]] \
  || die "--text-output must be text|jsonl"

[[ -d "$INPUT_DIR" ]] || die "Input dir not found: $INPUT_DIR"
mkdir -p "$OUTPUT_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "$WRAPPER_SCRIPT" ]]; then
  WRAPPER_SCRIPT="${SCRIPT_DIR}/infer_local.sh"
fi
[[ -f "$WRAPPER_SCRIPT" ]] || die "Wrapper script not found: $WRAPPER_SCRIPT"

# strict model checks
# ls -la $BOUNDARY_MODEL_DIR

                                      [[ "$BOUNDARY_MODEL_DIR" != "" && ! -d "${BOUNDARY_MODEL_DIR}" ]]                   && die "Boundary model dir not found: ${BOUNDARY_MODEL_DIR}"
[[ "$BOUNDARY_MODEL_TYPE" == "canine" && "$BOUNDARY_MODEL_DIR" != "" && ! -f "${BOUNDARY_MODEL_DIR}/best_model.pt" ]]     && die "Missing ${BOUNDARY_MODEL_DIR}/best_model.pt"
[[ "$BOUNDARY_MODEL_TYPE" == "canine" && "$BOUNDARY_MODEL_DIR" != "" && ! -f "${BOUNDARY_MODEL_DIR}/threshold.json" ]]    && die "Missing ${BOUNDARY_MODEL_DIR}/threshold.json"
[[ "$BOUNDARY_MODEL_TYPE" == "flair"  && "$BOUNDARY_MODEL_DIR" != "" && ! -f "${BOUNDARY_MODEL_DIR}/pytorch_model.bin" ]] && die "Missing ${BOUNDARY_MODEL_DIR}/pytorch_model.bin"

[[ "$BYT5_MODEL_DIR" != "" && ! -d "$BYT5_MODEL_DIR" ]]                                     && die "Abbrev/BYT5 model dir not found: ${BYT5_MODEL_DIR}"
[[ "$BYT5_MODEL_DIR" != "" && ! -f "${BYT5_MODEL_DIR}/final_model/model.safetensors" ]]     && die "Missing ${BYT5_MODEL_DIR}/final_model/model.safetensors"

# Resolve extensions
if [[ -z "$EXTENSIONS" ]]; then
  case "$MODE" in
    text)  EXTENSIONS="txt,text" ;;
    xml)   EXTENSIONS="xml,tei" ;;
    jsonl) EXTENSIONS="jsonl" ;;
  esac
fi

IFS=',' read -r -a EXT_ARR <<< "$EXTENSIONS"

# Build find expression
FIND_EXPR=()
for ext in "${EXT_ARR[@]}"; do
  ext="$(echo "$ext" | xargs)"  # trim
  [[ -n "$ext" ]] || continue
  if [[ ${#FIND_EXPR[@]} -gt 0 ]]; then
    FIND_EXPR+=(-o)
  fi
  FIND_EXPR+=(-iname "*.${ext}")
done
[[ ${#FIND_EXPR[@]} -gt 0 ]] || die "No valid extensions resolved"

mapfile -t FILES < <(find "$INPUT_DIR" -type f \( "${FIND_EXPR[@]}" \) | sort)

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "No matching files found in $INPUT_DIR for mode=$MODE extensions=$EXTENSIONS"
  exit 0
fi

echo "Found ${#FILES[@]} file(s)."
echo "Mode: $MODE"
echo "Input dir: $INPUT_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Extensions: $EXTENSIONS"
echo

ok=0
fail=0

for in_file in "${FILES[@]}"; do
  rel_path="${in_file#"$INPUT_DIR"/}"
  out_file="${OUTPUT_DIR}/${rel_path}"

  # Output suffix handling:
  # - keep extension for jsonl/xml
  # - for text mode with text output: append .expanded before extension
  if [[ "$MODE" == "text" && "$TEXT_OUTPUT" == "text" ]]; then
    base="${out_file%.*}"
    ext="${out_file##*.}"
    if [[ "$base" == "$out_file" ]]; then
      out_file="${out_file}.expanded"
    else
      out_file="${base}.expanded.${ext}"
    fi
  elif [[ "$MODE" == "text" && "$TEXT_OUTPUT" == "jsonl" ]]; then
    out_file="${out_file%.*}.jsonl"
  fi

  if [[ "$BOUNDARY_MODEL_DIR" != "" ]]; then
	  BOUNDARY_MODEL=(--boundary-model-dir "$BOUNDARY_MODEL_DIR")
  else
	  BOUNDARY_MODEL=(--boundary-model-name "$BOUNDARY_MODEL_NAME")
  fi
 
  if [[ "$BYT5_MODEL_DIR" != "" ]]; then
	  BYT5_MODEL=(--byt5-model-dir "$BYT5_MODEL_DIR")
  else
	  BYT5_MODEL=(--byt5-model-name "$BYT5_MODEL_NAME")
  fi
 
  mkdir -p "$(dirname "$out_file")"

  cmd=(
    $WRAPPER_SCRIPT
    --mode "$MODE"
    --input "$in_file"
    --output "$out_file"
    --boundary-model-type "$BOUNDARY_MODEL_TYPE"
    "${BOUNDARY_MODEL[@]}"
    "${BYT5_MODEL[@]}"
    --batch-size "$BATCH_SIZE"
    --text-output "$TEXT_OUTPUT"
    --python "$PYTHON_BIN"
  )

  if [[ "$LANG_PREFIX" == "1" ]]; then
    cmd+=(--lang-prefix)
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY-RUN] ${cmd[*]}"
    ok=$((ok + 1))
    continue
  fi

  echo ">> Processing: $in_file"
  if "${cmd[@]}"; then
    ok=$((ok + 1))
  else
    echo "!! Failed: $in_file" >&2
    fail=$((fail + 1))
  fi
done

echo
echo "Batch complete. Success: $ok | Failed: $fail | Total: ${#FILES[@]}"

if [[ "$fail" -gt 0 ]]; then
  exit 1
fi
