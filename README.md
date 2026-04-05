# Abbreviation Expansion Pipeline

Early modern Spanish/Latin printed text — nonbreaking line boundary detection
and abbreviation expansion using Canine and ByT5-base.

## Repository structure

```
abbr-expansion/
├── data/
│   └── data_utils.py          # Shared: loading, sorting, example construction
├── evaluation/
│   └── evaluation.py          # Span-level CER, exact match, type breakdown
├── boundary_classifier/
│   └── boundary_classifier.py # Canine classifier: train, evaluate, infer
├── byt5/
│   └── train_byt5.py          # ByT5-base: train and evaluate on HF Jobs
├── infer.py                   # Full inference pipeline (both models chained)
├── requirements.txt
├── job_configs.yaml           # HuggingFace Jobs launch configs
└── README.md
```

## Data format

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
```

## Pushing your dataset to the Hub

```python
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("mpilhlt/salamanca-abbr", repo_type="dataset", private=True)
api.upload_file(
    path_or_fileobj="data.jsonl",
    path_in_repo="data.jsonl",
    repo_id="mpilhlt/salamanca-abbr",
    repo_type="dataset",
)
```

## Training on HuggingFace Jobs

### Boundary classifier (Canine)

```bash
huggingface-cli jobs run --config job_configs.yaml  # boundary section
```

Or point at `job_boundary.yaml` specifically. The job will:
- Download `data.jsonl` from your dataset repo
- Train for 5 epochs, selecting best checkpoint by validation precision
- Run threshold selection on the PR curve targeting ≥0.90 precision
- Upload `boundary_eval.json`, `best_model.pt`, and `threshold.json`
  to `mpilhlt/canine-salamanca-boundary-classifier`

### ByT5 abbreviation expansion

```bash
huggingface-cli jobs run --config job_configs.yaml  # byt5 section
```

The job will:
- Train for up to 10 epochs with early stopping (patience 3)
- Select best checkpoint by span CER on the validation set
- Push each checkpoint to Hub as it is saved (`hub_strategy="every_save"`)
- Upload `test_breakdown.json` with per-abbreviation-type analysis

## Inference on new texts

```bash
python infer.py \
  --input             new_texts.jsonl \
  --output            expanded.jsonl \
  --boundary_model_dir ./canine-salamanca-boundary-classifier \
  --byt5_model_dir    mpilhlt/byt5-salamanca-abbr \
  --lexicon_data      data.jsonl \
  --batch_size        32
```

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
