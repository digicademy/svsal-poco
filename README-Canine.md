---
license: cc-by-4.0
datasets:
- mpilhlt/salamanca-abbr
language:
- la
- es
base_model:
- google/canine-s
library_name: transformers
pipeline_tag: token-classification
doi: 10.57967/hf/8275
tags:
- history
- humanities
- text-classification
- sequence-classification
- early-modern
- historical-text
- latin
- spanish
- canine
- boundary-detection
metrics:
- precision
- recall
- f1
model-index:
- name: canine-salamanca-boundary-classifier
  results:
  - task:
      type: text-classification
      name: Binary Sequence Classification
    dataset:
      name: salamanca-abbr
      type: mpilhlt/salamanca-abbr
    metrics:
    - type: precision
      value: 0.96
      name: Precision (nonbreaking)
    - type: recall
      value: 0.92
      name: Recall (nonbreaking)
    - type: f1
      value: 0.94
      name: F1 (nonbreaking)
    - type: accuracy
      value: 0.95
      name: Accuracy
---

# Canine Nonbreaking Line Boundary Classifier

This model identifies nonbreaking line boundaries in early modern Spanish and
Latin printed texts — that is, line breaks where a word continues on the next
line without a hyphen. It is the first stage in a two-model pipeline for
automatic abbreviation expansion in historical texts from the
[Salamanca School corpus](https://salamanca.school/).

## Model description

The model is a fine-tuned [CANINE-s](https://huggingface.co/google/canine-s)
encoder with a binary classification head. CANINE operates directly on Unicode
codepoints without tokenization, making it well suited to historical text where
orthographic variation, diacritics, and special characters are meaningful
signals. The local attention downsampling in CANINE-s makes it efficient for
the short input sequences used here.

**Input:** the last 40 characters of line N concatenated with the first 40
characters of line N+1, separated by the boundary marker `↵` (U+21B5).

**Output:** a binary label — `1` (nonbreaking, word continues across the
boundary) or `0` (genuine line break).

## Training

The model was trained on approximately 1.6 million line boundary examples
extracted from TEI XML transcriptions of early modern legal and theological
texts in Latin and Spanish, published as part of the
[School of Salamanca Digital Collection](https://salamanca.school/).

- **Positive examples** (nonbreaking): ~684,000 boundaries identified by
  editors as word-continuations across line breaks
- **Negative examples** (breaking): ~918,000 genuine line boundaries within
  the same paragraphs
- **Base model:** `google/canine-s`
- **Training hardware:** NVIDIA T4 (HuggingFace Jobs)
- **Training duration:** approximately 5.5 hours
- **Optimizer:** AdamW, learning rate 2e-5, weight decay 0.01
- **Batch size:** 32
- **Epochs:** 5
- **Class imbalance handling:** BCEWithLogitsLoss with pos_weight capped at 8.0

### Training results

| Epoch | Loss   | Precision | Recall | F1    |
|-------|--------|-----------|--------|-------|
| 1     | 0.4343 | 0.949     | 0.838  | 0.890 |
| 2     | 0.1771 | 0.932     | 0.934  | 0.933 |
| 3     | 0.1135 | 0.925     | 0.939  | 0.932 |
| 4     | 0.0761 | 0.957     | 0.916  | 0.936 |
| 5     | 0.0536 | 0.949     | 0.925  | 0.937 |

### Test set evaluation

Evaluated on a held-out document-level test set (documents not seen during
training):

|              | Precision | Recall | F1   | Support   |
|--------------|-----------|--------|------|-----------|
| breaking     | 0.94      | 0.97   | 0.96 | 918,084   |
| nonbreaking  | 0.96      | 0.92   | 0.94 | 684,090   |
| **accuracy** |           |        | **0.95** | **1,602,174** |
| macro avg    | 0.95      | 0.95   | 0.95 | 1,602,174 |
| weighted avg | 0.95      | 0.95   | 0.95 | 1,602,174 |

### Threshold selection

A classification threshold of **0.0017** was selected from the
precision-recall curve to achieve ≥0.90 precision on the validation set,
yielding precision 0.900 and recall 0.965 on nonbreaking boundaries. This
threshold is saved as `threshold.json` in the model repository and loaded
automatically by the inference code below.

The low raw threshold value (well below 0.5) reflects the class imbalance
handling via `pos_weight` in training — the model's raw logits are shifted,
so the sigmoid output should be interpreted relative to this calibrated
threshold rather than the naive 0.5 default.

## Intended use

This model is intended as a preprocessing step for abbreviation expansion
in early modern Latin and Spanish printed texts. It identifies which
consecutive line pairs should be concatenated before being passed to a
downstream abbreviation expansion model (such as
[mpilhlt/byt5-salamanca-abbr](https://huggingface.co/mpilhlt/byt5-salamanca-abbr)).

It is **not** intended as a general-purpose line boundary detector and has
not been evaluated on other corpora or languages.

## How to use

### Installation

```bash
pip install transformers torch
# or, to install the full pipeline:
pip install git+https://github.com/digicademy/svsal-poco
```

### Direct use with transformers

```python
import json
import torch
from transformers import CanineTokenizer
from huggingface_hub import hf_hub_download

# If using the svsal-poco package:
from boundary_classifier.boundary_classifier import BoundaryClassifier

# Load model and threshold
model_repo = "mpilhlt/canine-salamanca-boundary-classifier"
tokenizer  = CanineTokenizer.from_pretrained("google/canine-s")

model = BoundaryClassifier(use_lexicon=False)
weights_path = hf_hub_download(model_repo, "best_model.pt")
model.load_state_dict(torch.load(weights_path, map_location="cpu"))
model.eval()

threshold_path = hf_hub_download(model_repo, "threshold.json")
threshold = json.loads(open(threshold_path).read())["threshold"]

# Classify a single boundary
line_end   = "rum prudentium consultum. &c̃. legi tan"   # end of line N
line_start = "tùm adscribitur humanę. Sicuti & illa"    # start of line N+1

text = line_end[-40:] + "↵" + line_start[:40]
enc  = tokenizer(text, return_tensors="pt", max_length=128,
                 truncation=True, padding="max_length")

with torch.no_grad():
    out  = model(input_ids=enc["input_ids"],
                 attention_mask=enc["attention_mask"])
    prob = torch.sigmoid(out["logits"]).item()

is_nonbreaking = prob >= threshold
print(f"Probability: {prob:.4f} | Nonbreaking: {is_nonbreaking}")
```

### Use in the full pipeline

When using the full `svsal-poco` pipeline, the boundary classifier runs
automatically as the first stage of `infer.py`. See the
[svsal-poco repository](https://github.com/digicademy/svsal-poco) for the
complete pipeline documentation.

```bash
python infer.py \
  --input             texts.jsonl \
  --output            expanded.jsonl \
  --boundary_model_dir ./canine-salamanca-boundary-classifier \
  --byt5_model_dir    mpilhlt/byt5-salamanca-abbr \
  --batch_size        32
```

## Limitations

- Trained exclusively on texts from the School of Salamanca corpus; performance
  on other early modern corpora is unknown and may be lower
- Cross-page boundaries (where a word straddles a page break) are included in
  training data but may show lower performance than within-page boundaries
- The model has no explicit language identification; Latin and Spanish
  boundaries are handled jointly
- Marginal notes and main text are treated identically; boundaries between
  different text zones are excluded from training

## Citation

If you use this model, please cite the School of Salamanca project and this
repository:

```bibtex
@misc{svsal-poco,
  author    = {Wagner, Andreas and others},
  title     = {svsal-poco: Abbreviation Expansion Pipeline for Early Modern
               Spanish and Latin Printed Texts},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/digicademy/svsal-poco}
}
```