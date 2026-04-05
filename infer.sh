#!/bin/sh

huggingface-cli download mpilhlt/byt5-salamanca-abbr \
  --repo-type model \
  --local-dir ./byt5-salamanca-abbr

huggingface-cli download mpilhlt/canine-salamanca-boundary-classifier \
  --repo-type model \
  --local-dir ./canine-salamanca-boundary-classifier

python infer.py \
  --input             new_texts.jsonl \
  --output            expanded.jsonl \
  --boundary_model_dir ./canine-salamanca-boundary-classifier \
  --byt5_model_dir    ./byt5-salamanca-abbr \
  --batch_size        32
