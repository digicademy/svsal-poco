#!/bin/sh

# head -1000 data/data.jsonl > data/testdata.jsonl
head -20000 data/data.jsonl > data/evaltest.jsonl

CUDA_VISIBLE_DEVICES="" WANDB_MODE=disabled python byt5/train_byt5.py \
  --dataset_local data/evaltest.jsonl
  --output_dir       ./test_output_byt5 \
  --epochs           1 \
  --batch_size       4 \
  --eval_batch_size  8 \
  --eval_strategy    "steps" \
  --eval_steps       50 \
  --cap_eval         10000 \
  --gradient_accumulation_steps 1 \
  --learning_rate    1e-4 \
  --oversample_abbr  2.0 \
  --max_input_length 256 \
  --lang_prefix \
  --no_carbon
#  --use_cache
