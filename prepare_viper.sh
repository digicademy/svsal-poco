module purge
module load gcc/14 rocm/6.3 openmpi/5.0 # Viper: recommended by mpcdf
module load python-waterboa/2025.06

# Set up base directory
export PTMP_BASE=/ptmp/$USER/byt5-salamanca
mkdir -p $PTMP_BASE/{models,datasets,output,cache,wandb_offline}

# Install huggingface_hub CLI if not already available
pip install --user huggingface-hub wandb

# Authenticate
hf auth login --token $HF_TOKEN

# 1. Download the base model
hf download google/byt5-base \
  --local-dir $PTMP_BASE/models/byt5-base \
  --local-dir-use-symlinks False

# 2. Download the dataset
hf download mpilhlt/salamanca-abbr \
  --repo-type dataset \
  --local-dir $PTMP_BASE/datasets/salamanca-abbr \
  --local-dir-use-symlinks False

# 3. Download the svsal-poco package (since pip install from git won't work offline)
cd $PTMP_BASE
git clone https://github.com/digicademy/svsal-poco.git

# 4. Download your existing checkpoint/model repo (for resuming the failed run)
hf download mpilhlt/byt5-salamanca-abbr \
  --repo-type model \
  --local-dir $PTMP_BASE/models/byt5-salamanca-abbr-hub \
  --local-dir-use-symlinks False

# 5. Download the evaluate metric (CER) for offline use
# This pre-caches the metric module
python -c "
import evaluate
m = evaluate.load('cer')
print('CER metric cached at:', m.module_path if hasattr(m, 'module_path') else 'default cache')
"

# Copy the HF evaluate cache into ptmp so it's accessible offline
cp -r ~/.cache/huggingface $PTMP_BASE/cache/huggingface

# 6. Pre-download any pip packages you'll need into a wheelhouse (optional but recommended)
mkdir -p $PTMP_BASE/wheels
pip download -d $PTMP_BASE/wheels \
  'transformers>=4.40.0' 'datasets>=2.18.0' 'evaluate>=0.4.0' \
  'scikit-learn>=1.3.0' 'accelerate>=1.1.0' 'torch==2.6.0' \
  jiwer tensorboard wandb codecarbon

