#!/usr/bin/env bash
set -euo pipefail

# Hugging Face Jobs inference launcher.
# Runs recursive batch inference via infer_local_batch.sh.

# ------------------------------------------------------------------
# Configuration — adjust these values for your run
# ------------------------------------------------------------------
INPUT_DIR="${INPUT_DIR:-./infer_input}"
OUTPUT_DIR="${OUTPUT_DIR:-./infer_output}"
MODE="${MODE:-xml}"                         # text | xml | jsonl
EXTENSIONS="${EXTENSIONS:-}"                # optional CSV without dots, e.g. "xml,tei"
INPUT_FILE_URLS="${INPUT_FILE_URLS:-}"      # optional CSV of file URLs to download into INPUT_DIR
BOUNDARY_MODEL_TYPE="${BOUNDARY_MODEL_TYPE:-flair}"   # canine | flair
BATCH_SIZE="${BATCH_SIZE:-32}"
LANG_PREFIX="${LANG_PREFIX:-0}"             # 1 enables --lang-prefix
TEXT_OUTPUT="${TEXT_OUTPUT:-text}"          # text | jsonl (only relevant in mode=text)
FLAVOR="${FLAVOR:-a10g-small}"
TIMEOUT="${TIMEOUT:-12h}"

# Model repos and local download locations inside the job workspace.
BOUNDARY_MODEL_REPO="${BOUNDARY_MODEL_REPO:-mschonhardt/latin-contextual-lb-detector}"
BYT5_MODEL_REPO="${BYT5_MODEL_REPO:-mpilhlt/byt5-salamanca-abbr}"
BOUNDARY_MODEL_DIR="${BOUNDARY_MODEL_DIR:-./models/latin-contextual-lb-detector}"
BYT5_MODEL_DIR="${BYT5_MODEL_DIR:-./models/byt5-salamanca-abbr}"

usage() {
  cat <<'EOF'
Usage:
  ./infer_hf.sh [options]

Options:
  --input-dir <path>
  --output-dir <path>
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

hf jobs uv run \
  --flavor "$FLAVOR" \
  --timeout "$TIMEOUT" \
  --label Salamanca \
  --label task=inference \
  --secrets HF_TOKEN \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env INPUT_DIR="$INPUT_DIR" \
  --env INPUT_FILE_URLS="$INPUT_FILE_URLS" \
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
  bash -lc "\
    mkdir -p \"$INPUT_DIR\" \"$OUTPUT_DIR\" ./models && \
    python - <<'PY' && \
import os, re
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

raw_urls = os.getenv('INPUT_FILE_URLS', '').strip()
if raw_urls:
    input_dir = Path(os.getenv('INPUT_DIR', './infer_input'))
    input_dir.mkdir(parents=True, exist_ok=True)
    urls = [u.strip() for u in raw_urls.split(',') if u.strip()]
    for i, url in enumerate(urls, 1):
        with urlopen(url) as resp:
            cd = resp.headers.get('Content-Disposition', '')
            m = re.search(r'filename\*?=["\']?(?:UTF-8\'\')?([^"\';\r\n]+)', cd, re.IGNORECASE)
            if m:
                file_name = Path(m.group(1).strip()).name
            else:
                file_name = Path(urlparse(url).path).name or f'download_{i}'
            data = resp.read()
        target = input_dir / file_name
        print(f'Downloading [{i}/{len(urls)}]: {url} -> {target}')
        target.write_bytes(data)
else:
    print('No INPUT_FILE_URLS configured; skipping preparatory input download.')
PY
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
      $EXTRA_ARGS
"
