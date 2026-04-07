# School of Salamanca Post-Correction Pipeline

ML models for correcting early modern Spanish/Latin printed text from the
[School of Salamanca](https://salamanca.school/) digital edition project.
Features nonbreaking line boundary detection and abbreviation expansion using
Canine and ByT5-base.

## Repository structure

```
abbr-expansion/
├── data/
│   ├── prepare_data/          # Scripts to prepare training data from SvSal corpus
│   │   ├── scripts/
│   │   │   └── *              # Various scripts to transform TEI XML to jsonl
│   │   └── runme.sh           # Commands and explanation to prepare training data
│   ├── check_data.py          # Profile length of lines in dataset
│   └── data_utils.py          # Shared: loading, sorting, example construction
├── evaluation/
│   └── evaluation.py          # Span-level CER, exact match, type breakdown
├── boundary_classifier/
│   └── boundary_classifier.py # Canine classifier: train, evaluate, infer
├── byt5/
│   └── train_byt5.py          # ByT5-base: train and evaluate on HF Jobs
├── infer/
│   └── __init__.py            # Full inference pipeline (both models chained)
├── env.template               # A template for you to create your own .env file with
│                              # secrets. Currently used only for WandB monitoring (optional)
├── pyproject.toml             # Project metadata for uv and other python package maintenance
└── README.md                  # This file with documentation
├── requirements.txt
├── test_boundary.sh             # Run smoke test for boundary classifier
├── test_byt5.sj               # Run smoke test for byt5 abbreviation expansion model
├── train_boundary.sh          # Run boundary classifier training job on HuggingFace
├── train_byt5.sh              # Run ByT5 expansion model training job on HuggingFace
├── uv.lock2                    # Dependencies and their versions for uv package mgmt
```

## Data format

The training data has been created by the scripts in the [data/prepare_data](./data/prepare_data/) folder. Most importantly, the [01_create_jsonl.xsl](./data/prepare_data/scripts/01_create_jsonl.xsl) and [02_adjust_shifted_lbs.py](./data/prepare_data/scripts/02_adjust_shifted_lbs.py) scripts.

Your JSONL export should have at minimum these fields per line:

```json
{
  "id":                    "W0011-00-0006-lb-2027",
  "doc_id":                "W0011",
  "facs_id":               "W0011-0006",
  "ancestor_id":           "W0011-00-0006-pa-03f6",
  "lang":                  ["la"],
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

## Setup

```bash
pip install -r requirements.txt
# or with uv:
uv sync
```

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

### ByT5 abbreviation expansion

```bash
./train_byt5.sh
```

This job will do the following on HuggingFace infrastructure:
- Train for up to 10 epochs with early stopping (patience 3)
- Select best checkpoint by span CER on the validation set
- Push each checkpoint to Hub as it is saved (`hub_strategy="every_save"`)
- Upload `test_breakdown.json` with per-abbreviation-type analysis

## Inference on new texts

```bash
huggingface-cli download mpilhlt/byt5-salamanca-abbr \
  --repo-type model \
  --local-dir ./byt5-salamanca-abbr

huggingface-cli download mpilhlt/canine-salamanca-boundary-classifier \
  --repo-type model \
  --local-dir ./canine-salamanca-boundary-classifier

python -m infer \
  --input              new_texts.jsonl \
  --output             expanded.jsonl \
  --boundary_model_dir ./canine-salamanca-boundary-classifier \
  --byt5_model_dir     ./byt5-salamanca-abbr \
  --batch_size         32
```

If the package is installed, you can run the last command also directly
with the `infer` command.

Input JSONL needs: `id`, `doc_id`, `source_sic`, `lang`. No abbreviation
markup is expected — the pipeline handles detection via the boundary
classifier and ByT5's learned span associations.

Output JSONL adds an `expanded_text` field to each input row.

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

**Why span CER for model selection, not full-line CER?**
Full-line CER rewards correct copying of non-abbreviated text, which the
model learns trivially. Span CER focuses checkpoint selection on the
quality of expansions — the actual task.

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

## What is not yet implemented

- Canine-based abbreviation expansion (encoder + character decoder head):
  implement after establishing ByT5-base results as a baseline
- Language-conditioned training ablation (--lang_prefix flag exists but
  effect is not yet measured)
- Multi-GPU training support for span metrics in compute_metrics callback
