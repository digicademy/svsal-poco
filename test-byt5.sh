#!/bin/sh

head -1000 data/data.jsonl > data/testdata.jsonl

CUDA_VISIBLE_DEVICES="" python byt5/train_byt5.py \
  --dataset_local data/testdata.jsonl \
  --output_dir    ./test-output-byt5 \
  --epochs        2 \
  --batch_size    4 \
  --learning_rate 1e-4 \
  --oversample_abbr 2.0 \
  --lang_prefix \
