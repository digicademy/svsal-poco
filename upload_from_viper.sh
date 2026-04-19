# Push final model to Hub
hf upload \
  --repo-type model \
  --commit-message "Final model from HPC training" \
  mpilhlt/byt5-salamanca-abbr \
  $PTMP_BASE/output/final_model \
  /final_model

# Push test breakdown
hf upload \
  --repo-type model \
  mpilhlt/byt5-salamanca-abbr \
  $PTMP_BASE/output/test_breakdown.json \
  test_breakdown.json

# Upload wandb offline data (see section below)
cd $PTMP_BASE/wandb_offline
wandb sync --sync-all
