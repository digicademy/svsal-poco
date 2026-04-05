#!/bin/sh

head -1000 data/data.jsonl > data/testdata.jsonl

CUDA_VISIBLE_DEVICES="" python boundary_classifier/boundary_classifier.py \
  --dataset_local data/testdata.jsonl \
  --output_dir    ./test-output \
  --epochs        2 \
  --batch_size    8 \
  --learning_rate 2e-5 \
  --threshold     0.6 \
  --min_precision 0.90
