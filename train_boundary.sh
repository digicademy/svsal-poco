#!/bin/sh

hf jobs uv run \
  --flavor t4-small \
  --timeout 8h \
  --label Salamanca \
  --label model=canine \
  --secrets HF_TOKEN \
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
  train-boundary \
    --dataset_repo mpilhlt/salamanca-abbr \
    --output_repo  mpilhlt/canine-salamanca-boundary-classifier \
    --epochs 5 \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --threshold     0.6 \
    --min_precision 0.90
