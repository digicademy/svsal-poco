# School of Salamanca Post-Correction Pipeline

<!-- [![DOI](https://zenodo.org/badge/1201861271.svg)](https://zenodo.org/badge/latestdoi/1201861271) [![HuggingFace](https://img.shields.io/badge/huggingface-%23FFD21E.svg?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/mpilhlt) ![MIT License](https://img.shields.io/github/license/digicademy/svsal-poco) -->

Machine learning models for correcting early modern Spanish/Latin printed text
from the [School of Salamanca](https://salamanca.school/) digital edition project.
Features nonbreaking line boundary detection and abbreviation expansion using
Canine and ByT5-base.

## Repository structure

```
svsal-poco/
├── boundary_classifier/
│   └── boundary_classifier.py # Canine classifier: train, evaluate, infer
├── byt5/
│   └── train_byt5.py          # ByT5-base Seq2seq: train and evaluate
├── data/
│   ├── prepare_data/          # Scripts to prepare training data from SvSal corpus
│   │   ├── scripts/
│   │   │   └── *              # Various scripts to transform SvSal TEI XML to jsonl
│   │   └── runme.sh           # Commands and explanation to prepare training data
│   ├── check_data.py          # Profile dataset for length of lines
│   └── data_utils.py          # Shared: loading, sorting, example construction
├── evaluation/
│   └── evaluation.py          # Span-level CER, exact match, type breakdown
├── infer/
│   └── __init__.py            # Full inference pipeline (both models chained)
├── tei/
│   └── tei_roundtrip.py       # XML handling for inference (strip and re-inject XML)
├── README.md                  # This file with documentation
├── README-ByT5.md             # Model card of resulting ByT5 model (backup from HF)
├── README-Canine.md           # Model card of resulting Canine model (backup from HF)
├── README-dataset.md          # Dataset card of training dataset (backup from HF)
├── env.template               # A template for you to create your own .env file with
│                              # secrets.
├── eval_byt5_slurm.sh         # Script that runs just the eval part, for resuming runs
│                              # that have been killed (due to timeout) after training
│                              # has completed
├── prepare_viper.sh           # Script to download all online resources for HPC nodes
│                              # that have no access to the net
├── pyproject.toml             # Project metadata for uv and other package maintenance
├── requirements.txt           # Some environments use this to handle dependencies
│                              # (others use pyproject.toml)
├── test_boundary.sh           # Run smoke test for boundary classifier
├── test_byt5.sj               # Run smoke test for byt5 abbreviation expansion model
├── tokenize_byt5_slurm.sh     # Script to just tokenize the dataset. This is a task
│                              # better suited to CPU nodes and can be run separately
├── train_boundary.sh          # Run boundary classifier training job on HuggingFace
├── train_byt5.sh              # Run ByT5 expansion model training job on HuggingFace
├── train_byt5_slurm.sh        # Run ByT5 expansion model training job on HPC
├── upload_from_viper.sh       # Script to upload model and assets after training and
│                              # evaluation in offline-mode HPC have completed
└── uv.lock                    # Dependencies and their versions for uv package mgmt
```

## Data format

The training data has been created by the scripts in the [data/prepare_data](./data/prepare_data/) folder. Most importantly, the [01_create_jsonl.xsl](./data/prepare_data/scripts/01_create_jsonl.xsl) and [02_adjust_shifted_lbs.py](./data/prepare_data/scripts/02_adjust_shifted_lbs.py) scripts. The dataset used for our training pipeline is online
at Hugging Face: [mpilhlt/salamanca-abbr](https://huggingface.co/datasets/mpilhlt/salamanca-abbr).

If you want to run it with your own dataset, your JSONL export should have at minimum these fields per line:

```json
{
  "id":                    "W0011-00-0006-lb-2027",
  "doc_id":                "W0011",
  "source_sic":            "lib. Lex est communis ciuitatis ⦃cōsensus⦄ qui",
  "target_corr":           "lib. Lex est communis ciuitatis consensus qui",
  "contains_abbr":         "true",
  "nonbreaking_next_line": "W0011-00-0006-lb-2028"
}
```

**Key point**: `source_sic` must have abbreviation spans wrapped in `⦃⦄`
delimiters (U+2983, U+2984). Insert these during TEI export by wrapping
each `<abbr>` element's text content. `target_corr` is plain expanded text
with no delimiters.

Lines where `contains_abbr` is `"false"` are used as-is (copy-through
training signal for ByT5). The `nonbreaking_next_line` field is used by
both the boundary classifier (as positive labels) and ByT5 preprocessing
(for line pair concatenation).

## Training on HuggingFace Jobs

### Boundary classifier (Canine)

```bash
./train_boundary.sh
```

This job will do the following on HuggingFace infrastructure:
- Download `data.jsonl` from the dataset repo (configured as `mpilhlt/salamanca-abbr`)
- Train for 5 epochs, selecting best checkpoint by validation precision
- Run threshold selection on the PR curve targeting ≥0.90 precision
- Upload `boundary_eval.json`, `best_model.pt`, and `threshold.json`
  to `mpilhlt/canine-salamanca-boundary-classifier`

### Abbreviation expansion (ByT5)

```bash
./train_byt5.sh
```

This job will do the following on HuggingFace infrastructure:
- Train for up to 10 epochs with early stopping (patience 3)
- Select best checkpoint by span CER on the validation set
- Push each checkpoint to Hub as it is saved
- Upload final model, test results and `test_breakdown.json`
  with per-abbreviation-type analysis

## Training on Slurm-based HPC

There are several scripts tailored to Slurm-based environments
(in our case, the HPC cluster was called *"viper"*, hence the
name appears in several places):
We have used this for the abbreviation expansion (ByT5) model
only, so if you want to train the boundary classifier, you'll
have to adjust the scripts as needed.

1. `./prepare_viper.sh`: As our HPC nodes have no access to the
   internat and cannot download dataset or models, nor install
   python packages from PyPI, this script downloads all such
   resources to the login node before you launch the Slurm job
   on the compute node.
2. `././tokenize_byt5_slurm.sh`: The loading, splitting and
   tokenizing of the dataset is a compute-intensive task that
   is better suited to a CPU node than to a GPU one. Therefore,
   we have broken out these steps to a separate preprocessing
   step creating `tokenized_cache` data that the main training
   script can resume from.
3. `./train_byt5_slurm.sh`: This is the main training script. It
   can resume from the tokenizing cache or from checkpoints
   saved in earlier runs. Checkpoints are saved after each
   epoch. This should enable the script to just be repeatedly
   run when the walltime for the job expires and the job is
   being killed.
4. `././eval_byt5_slurm.sh`: If the training job is killed when the
   training has actually completed (i.e. during test set
   evaluation), this script makes the process resume with just
   the evaluation.
5. `./upload_from_viper.sh`: Once all is finished, this script
   uploads model and evaluation results to HuggingFace and
   optionally syncs with your weights and bits repository for
   analytics.

## Inference on new texts

### Local inference

#### Setup

Install dependencies

```bash
pip install -r requirements.txt
# or: uv sync
```

Download models locally

```bash
huggingface-cli download mpilhlt/byt5-salamanca-abbr \
  --repo-type model \
  --local-dir ./byt5-salamanca-abbr

huggingface-cli download mpilhlt/canine-salamanca-boundary-classifier \
  --repo-type model \
  --local-dir ./canine-salamanca-boundary-classifier
```

#### Standard inferencing

Run the standard inference like this:

```bash
python -m infer \
  --input              new_texts.jsonl \
  --output             expanded.jsonl \
  --boundary_model_dir ./canine-salamanca-boundary-classifier \
  --byt5_model_dir     ./byt5-salamanca-abbr \
  --batch_size         32
```

If the package is installed, you can run the last command also directly
with the `infer` command.

Input JSONL needs: `id`, `doc_id`, `source_sic`. No abbreviation
markup is expected — the pipeline handles detection via the boundary
classifier and ByT5's learned span associations.

Output JSONL adds an `expanded_text` field to each input row.

#### Convenience wrappers

Besides `python -m infer` (JSONL input), this repo also supports local wrappers that offer more convenient workflows:

- `infer_handler.py`: Python CLI for `text`, `jsonl`, and `xml` modes
- `infer_local.sh`: file-to-file wrapper for single inputs
- `infer_local_batch.sh`: recursive directory batch processing

Notes:
- `text` mode: one input line per line; output includes `¬` for predicted nonbreaking boundaries.
- `xml` mode: runs TEI/XML roundtripping and writes processed XML.
- `jsonl` mode: preserves "standard" JSONL pipeline behavior.

**Single files**

```bash
./infer_local.sh \
  --mode <text|jsonl|xml> \
  --input <input-file> \
  --output <output-file> \
  --boundary-model-dir ./canine-salamanca-boundary-classifier \
  --byt5-model-dir ./byt5-salamanca-abbr
```

Examples:

```bash
# plaintext -> expanded plaintext
./infer_local.sh --mode text --input in.txt --output out.txt \
  --boundary-model-dir ./canine-salamanca-boundary-classifier \
  --byt5-model-dir ./byt5-salamanca-abbr

# XML -> processed XML
./infer_local.sh --mode xml --input in.xml --output out.xml \
  --boundary-model-dir ./canine-salamanca-boundary-classifier \
  --byt5-model-dir ./byt5-salamanca-abbr
```

**Batch usage (directories)**

```bash
./infer_local_batch.sh \
  --mode <text|xml|jsonl> \
  --input-dir <input-dir> \
  --output-dir <output-dir> \
  --boundary-model-dir ./canine-salamanca-boundary-classifier \
  --byt5-model-dir ./byt5-salamanca-abbr
```

### (Hugging Face) Gradio Space

This repository's [`space` branch](https://github.com/digicademy/svsal-poco/tree/space)
has a gradio web application that can drive, e.g. a
[Hugging Face Space](https://huggingface.co/spaces).

It offers forms for running the combined pipeline in both plaintext and
XML processing mode as well as a standalone boundary detection demo tab.

## Training decisions and rationale

**Why document-level splitting?**
Prevents data leakage from shared compositor conventions and sliding window
context. Lines from the same document share orthographic patterns that would
inflate test metrics if mixed into train.

**Why span-infilling framing for ByT5?**
Abbreviations are full tokens wrapped in ⦃⦄ delimiters. ByT5 sees these
as distinct bytes and learns to replace marked spans while copying the rest.
This aligns with ByT5's pretraining objective and gives the clearest
learning signal.

**Why high-precision threshold for the boundary classifier?**
False positives (spuriously concatenating lines) corrupt the ByT5 input
by joining text that should remain separate. False negatives (missing a
nonbreaking boundary) mean the abbreviation model sees a split token,
which is a recoverable error. Precision is therefore more important than
recall for this upstream component.

**Why Canine-s for boundary classification?**
Canine operates on Unicode codepoints (better than byte-level for this
task), uses local attention to downsample before the main transformer
(efficient for short inputs), and is an encoder model — well suited to
the binary classification framing. ByT5 would require generative framing
for a classification task, which is unnecessarily complex here.

## License

The code of the present repository is published under the [MIT license](./LICENSE).
