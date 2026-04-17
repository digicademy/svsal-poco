# Push final model to Hub
huggingface-cli upload mpilhlt/byt5-salamanca-abbr \
  $PTMP_BASE/output/final_model \
  --repo-type model \
  --commit-message "Final model from HPC training"

# Push test breakdown
huggingface-cli upload mpilhlt/byt5-salamanca-abbr \
  $PTMP_BASE/output/test_breakdown.json \
  test_breakdown.json \
  --repo-type model

# Upload wandb offline data (see section below)
cd $PTMP_BASE/wandb_offline
wandb sync --sync-all
