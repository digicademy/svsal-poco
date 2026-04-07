#!/bin/sh

hf jobs uv run \
  --flavor a100-large \
  --timeout 2d \
  --label Salamanca \
  --label model=byt5 \
  --secrets HF_TOKEN \
  --secrets-file .env \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env WANDB_WATCH=all \
  --with 'transformers>=4.40.0' \
  --with 'datasets>=2.18.0' \
  --with 'evaluate>=0.4.0' \
  --with 'scikit-learn>=1.3.0' \
  --with 'accelerate>=1.1.0' \
  --with 'torch==2.6.0' \
  --with jiwer \
  --with tensorboard \
  --with wandb \
  --with codecarbon \
  --with 'git+https://github.com/digicademy/svsal-poco' \
  train-byt5 \
      --dataset_repo mpilhlt/salamanca-abbr \
      --output_repo  mpilhlt/byt5-salamanca-abbr \
      --wandb_project byt5-salamanca-abbr \
      --wandb_entity mpilhlt \
      --epochs 10 \
      --learning_rate 1e-4 \
      --oversample_abbr 2.0 \
      --batch_size 64 \
      --gradient_accumulation_steps 2 \
      --max_input_length 256 \
      --max_target_length 256 \
      --tokenizer_num_proc 16 \
      --bf16 \
      --use_cache \
      --lang_prefix

# If OOM: reduce batch_size to from — and preserve effective batch size
# by increasing gradient_accumulation_steps.
