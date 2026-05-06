#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------
# infer_local.sh
# Batch-friendly wrapper for infer_handler.py
#
# Supports:
#   - text mode (plaintext in -> expanded text out, app-like)
#   - jsonl mode (jsonl in -> jsonl out)
#   - xml mode (xml in -> xml out, with TEI roundtrip)
#
# Requires:
#   - python script: infer_handler.py (same directory or set --script)
#   - local model dirs downloaded beforehand
# ------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/infer_handler.py"

MODE=""
INPUT_FILE=""
OUTPUT_FILE=""
BOUNDARY_MODEL_DIR=""
BOUNDARY_MODEL_NAME=""
BOUNDARY_MODEL_TYPE="canine"
BYT5_MODEL_DIR=""
BATCH_SIZE="16"
LANG_PREFIX="0"
TEXT_OUTPUT="text"   # for --mode text: text|jsonl
PYTHON_BIN="python"

usage() {
  cat <<'EOF'
Usage:
  ./infer_local.sh --mode <text|jsonl|xml> \
    --input <path> \
    --output <path> \
    --byt5-model-dir <path> \
    --boundary-model-dir <path> \
    --boundary-model-name <string> \
    [--boundary-model-type <canine|flair>] \
    [--batch-size <int>] \
    [--lang-prefix] \
    [--text-output <text|jsonl>] \
    [--python <python_bin>] \
    [--script <path_to_infer_handler.py>]

Required:
  --mode                Input mode: text, jsonl, or xml
  --input               Input file path
  --output              Output file path (required for batch processing)
  --byt5-model-dir      Directory or HF-loadable path for ByT5 model

The boundary model must be specified, but it can be 
specified either with a reference to a local directory:
  --boundary-model-dir
or with a hugging face model identifier:
  --boundary-model-name
If it is a flair model, that must be specified with:
  --boundary-model-type flair
Otherwise the default boundary model type is "canine"

Optional:
  --batch-size          Default: 16
  --lang-prefix         Passes --lang-prefix to python handler
  --text-output         Only for mode=text; text (default) or jsonl
  --python              Python executable (default: python)
  --script              Path to infer_handler.py

Examples:
  # Plaintext file -> expanded plaintext file
  ./infer_local.sh \
    --mode text \
    --input ./in/snippet.txt \
    --output ./out/snippet.expanded.txt \
    --boundary-model-dir ./canine-salamanca-boundary-classifier \
    --byt5-model-dir ./byt5-salamanca-abbr

  # XML file -> processed XML file
  ./infer_local.sh \
    --mode xml \
    --input ./in/input.xml \
    --output ./out/output.xml \
    --boundary-model-dir ./canine-salamanca-boundary-classifier \
    --byt5-model-dir ./byt5-salamanca-abbr

  # JSONL file -> JSONL file
  ./infer_local.sh \
    --mode jsonl \
    --input ./in/input.jsonl \
    --output ./out/output.jsonl \
    --boundary-model-dir ./canine-salamanca-boundary-classifier \
    --byt5-model-dir ./byt5-salamanca-abbr
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

# ------------------------
# Parse args
# ------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="${2:-}"; shift 2 ;;
    --input) INPUT_FILE="${2:-}"; shift 2 ;;
    --output) OUTPUT_FILE="${2:-}"; shift 2 ;;
    --boundary-model-dir) BOUNDARY_MODEL_DIR="${2:-}"; shift 2 ;;
    --boundary-model-name) BOUNDARY_MODEL_NAME="${2:-}"; shift 2 ;;
    --boundary-model-type) BOUNDARY_MODEL_TYPE="${2:-}"; shift 2 ;;
    --byt5-model-dir) BYT5_MODEL_DIR="${2:-}"; shift 2 ;;
    --batch-size) BATCH_SIZE="${2:-}"; shift 2 ;;
    --lang-prefix) LANG_PREFIX="1"; shift 1 ;;
    --text-output) TEXT_OUTPUT="${2:-}"; shift 2 ;;
    --python) PYTHON_BIN="${2:-}"; shift 2 ;;
    --script) PY_SCRIPT="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown argument: $1 (use --help)" ;;
  esac
done

# ------------------------
# Validate
# ------------------------
[[ -n "$MODE" ]] || die "--mode is required"
[[ -n "$INPUT_FILE" ]] || die "--input is required"
[[ -n "$OUTPUT_FILE" ]] || die "--output is required"
[[ -n "$BYT5_MODEL_DIR" ]] || die "--byt5-model-dir is required"

[[ -n "$BOUNDARY_MODEL_DIR" && -n "$BOUNDARY_MODEL_NAME" ]] \
  && die "boundary model must be specified with --boundary-model-dir or --boundary-model-name"
[[ "$BOUNDARY_MODEL_DIR" != "" && "$BOUNDARY_MODEL_NAME" != "" ]] \
  && die "boundary model must be specified with either --boundary-model-dir or --boundary-model-name, not both"

[[ "$MODE" == "text" || "$MODE" == "jsonl" || "$MODE" == "xml" ]] \
  || die "--mode must be one of: text, jsonl, xml"

[[ "$TEXT_OUTPUT" == "text" || "$TEXT_OUTPUT" == "jsonl" ]] \
  || die "--text-output must be text or jsonl"

[[ -f "$INPUT_FILE" ]] || die "Input file does not exist: $INPUT_FILE"
[[ -f "$PY_SCRIPT" ]] || die "Python handler not found: $PY_SCRIPT"

# boundary dir checks (strict)
[[ "$BOUNDARY_MODEL_DIR" != "" && ! -d "$BOUNDARY_MODEL_DIR" ]] && die "Boundary model dir not found: $BOUNDARY_MODEL_DIR"
[[ "$BOUNDARY_MODEL_TYPE" == "canine" && "$BOUNDARY_MODEL_DIR" != "" && ! -f "${BOUNDARY_MODEL_DIR}/best_model.pt" ]]     && die "Missing ${BOUNDARY_MODEL_DIR}/best_model.pt"
[[ "$BOUNDARY_MODEL_TYPE" == "canine" && "$BOUNDARY_MODEL_DIR" != "" && ! -f "${BOUNDARY_MODEL_DIR}/threshold.json" ]]    && die "Missing ${BOUNDARY_MODEL_DIR}/threshold.json"
[[ "$BOUNDARY_MODEL_TYPE" == "flair"  && "$BOUNDARY_MODEL_DIR" != "" && ! -f "${BOUNDARY_MODEL_DIR}/pytorch_model.bin" ]] && die "Missing ${BOUNDARY_MODEL_DIR}/pytorch_model.bin"

# output directory
OUT_DIR="$(dirname "$OUTPUT_FILE")"
mkdir -p "$OUT_DIR"

# ------------------------
# Build command
# ------------------------
CMD=(
  "$PYTHON_BIN" "$PY_SCRIPT"
  --mode "$MODE"
  --input-file "$INPUT_FILE"
  --output-file "$OUTPUT_FILE"
  --boundary-model-dir "$BOUNDARY_MODEL_DIR"
  --boundary-model-name "$BOUNDARY_MODEL_NAME"
  --boundary-model-type "$BOUNDARY_MODEL_TYPE"
  --byt5-model-dir "$BYT5_MODEL_DIR"
  --batch-size "$BATCH_SIZE"
)

if [[ "$LANG_PREFIX" == "1" ]]; then
  CMD+=(--lang-prefix)
fi

if [[ "$MODE" == "text" ]]; then
  CMD+=(--text-output "$TEXT_OUTPUT")
fi

echo "Running:"
printf '  %q' "${CMD[@]}"
echo

"${CMD[@]}"

echo "Done."
echo "Output written to: $OUTPUT_FILE"
