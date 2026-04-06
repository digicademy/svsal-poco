#!/bin/sh

hf jobs uv run \
  --flavor a100-large \
  --timeout 2d \
  --label Salamanca \
  --label model=byt5 \
  --secrets HF_TOKEN \
  --secrets-file .env \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --with 'transformers>=4.40.0' \
  --with 'torch>=2.1.0' \
  --with 'datasets>=2.18.0' \
  --with 'evaluate>=0.4.0' \
  --with 'scikit-learn>=1.3.0' \
  --with 'accelerate>=1.1.0' \
  --with jiwer \
  --with tensorboard \
  --with wandb \
  --with codecarbon \
  --with 'git+https://github.com/digicademy/svsal-poco' \
  train-byt5 \
    --dataset_repo mpilhlt/salamanca-abbr \
    --output_repo  mpilhlt/byt5-salamanca-abbr \
    --epochs 10 \
    --batch_size 128 \
    --learning_rate 1e-4 \
    --oversample_abbr 2.0 \
    --lang_prefix

# If OOM: reduce batch_size to from 16 to 8 or 4 — effective batch size is preserved
# by adding gradient_accumulation_steps=2 (or 4) in Seq2SeqTrainingArguments.
