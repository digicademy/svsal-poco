module purge
module load gcc/14 rocm/7.2 openmpi/5.0 # Viper: recommended by mpcdf
module load python-waterboa/2025.06

# Set up base directory
export PTMP_BASE=/ptmp/$USER/byt5-salamanca
mkdir -p $PTMP_BASE/{models,datasets,output,cache,wandb_offline}

# Install huggingface_hub CLI if not already available
pip install --user huggingface-hub wandb

# Authenticate
hf auth login --token $HF_TOKEN
wandb login

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
