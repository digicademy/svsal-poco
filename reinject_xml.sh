#!/usr/bin/env bash
# =============================================================================
# reinject_xml.sh — re-run TEI XML re-injection without models / GPU
#
# Uses the *.inferred.jsonl intermediate file that was written during a
# previous `infer_handler.py --mode xml --save-intermediate` run.
# The models are NOT loaded; the script is fast and CPU-only.
#
# Usage:
#   ./reinject_xml.sh \
#       --input   path/to/original_input.xml \
#       --infer-jsonl path/to/original.inferred.jsonl \
#       --output  path/to/reinjected_output.xml
#
# The original input XML must be the *same file* (or identical content) as
# was used for the inference run that produced the JSONL, because
# process_tei_xml re-extracts row IDs from the XML and matches them against
# the JSONL.  A mismatch will produce a warning but will not abort.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/infer_handler.py"

INPUT=""
INFER_JSONL=""
OUTPUT=""
PYTHON_BIN="python"

usage() {
    cat <<'EOF'
Usage:
  ./reinject_xml.sh \
      --input       <original_input.xml> \
      --infer-jsonl <*.inferred.jsonl>   \
      --output      <output.xml>

Options:
  --input       -i   Path to the original TEI XML input file
  --infer-jsonl      Path to the *.inferred.jsonl intermediate file
  --output      -o   Path for the re-injected TEI XML output
  --python           Python executable to use (default: python)
  --script           Path to infer_handler.py (default: same dir as this script)
  -h, --help         Show this message
EOF
}

die() { echo "ERROR: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input|-i)      INPUT="${2:-}";       shift 2 ;;
        --infer-jsonl)   INFER_JSONL="${2:-}"; shift 2 ;;
        --output|-o)     OUTPUT="${2:-}";      shift 2 ;;
        --python)        PYTHON_BIN="${2:-}";  shift 2 ;;
        --script)        PY_SCRIPT="${2:-}";   shift 2 ;;
        -h|--help)       usage; exit 0 ;;
        *) die "Unknown argument: $1  (use --help)" ;;
    esac
done

[[ -n "$INPUT"      ]] || die "--input is required"
[[ -n "$INFER_JSONL" ]] || die "--infer-jsonl is required"
[[ -n "$OUTPUT"     ]] || die "--output is required"
[[ -f "$INPUT"      ]] || die "Input file not found: $INPUT"
[[ -f "$INFER_JSONL" ]] || die "Inferred JSONL not found: $INFER_JSONL"
[[ -f "$PY_SCRIPT"  ]] || die "Python handler not found: $PY_SCRIPT"

mkdir -p "$(dirname "$OUTPUT")"

echo "[reinject_xml] input XML    : $INPUT"
echo "[reinject_xml] infer JSONL  : $INFER_JSONL"
echo "[reinject_xml] output XML   : $OUTPUT"
echo "[reinject_xml] no models loaded; CPU only"
echo

"$PYTHON_BIN" "$PY_SCRIPT" \
    --mode       xml        \
    --input-file  "$INPUT"  \
    --output-file "$OUTPUT" \
    --infer-jsonl "$INFER_JSONL"

echo
echo "[reinject_xml] Done. Output: $OUTPUT"
