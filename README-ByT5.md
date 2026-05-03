---
license: cc-by-4.0
datasets:
- mpilhlt/salamanca-abbr
language:
- la
- es
base_model:
- google/byt5-base
library_name: transformers
pipeline_tag: text-generation
doi: 10.57967/hf/8641
tags:
- history
- humanities
- text-generation
- abbreviation-expansion
- early-modern
- historical-text
- latin
- spanish
- byt5
- seq2seq
metrics:
- cer
model-index:
- name: byt5-salamanca-abbr
  results:
  - task:
      type: text2text-generation
      name: Abbreviation Expansion
    dataset:
      name: salamanca-abbr
      type: mpilhlt/salamanca-abbr
    metrics:
    - type: cer
      value: 0.00005
      name: Full-line CER
    - type: cer
      value: 0.008
      name: Span CER
    - type: accuracy
      value: 0.9554
      name: Span exact match
---

# ByT5 Abbreviation Expansion for Early Modern Texts

This model expands abbreviations in early modern Spanish and Latin printed
texts — brevigraphs, macrons, tildes, and other typographic shorthand
conventions used in 16th–17th century printing. Ideally, it should even be
capable of expanding some abbreviations without typographical signals.
It is the second stage in a two-model pipeline, following a boundary classifier
([mpilhlt/canine-salamanca-boundary-classifier](https://huggingface.co/mpilhlt/canine-salamanca-boundary-classifier))
that identifies nonbreaking line boundaries.

## Model description

The model is a fine-tuned [ByT5-base](https://huggingface.co/google/byt5-base),
a byte-level sequence-to-sequence Transformer that operates directly on UTF-8
bytes without tokenization. This makes it well suited to historical text where
abbreviation markers (macrons, tildes, superscripts) are individual Unicode
codepoints that subword tokenizers would split unpredictably.

**Input:** one or more lines of transcribed text, joined with the line separator
character `¬` (U+00AC) for nonbreaking boundaries or `↵` (U+21B5) for normal
line boundaries. Context lines (preceding and following) improve disambiguation.

**Output:** the same text with abbreviations expanded. Unchanged characters are
copied through verbatim; only abbreviated spans are modified.

### Marker dropout training

The [training data](https://huggingface.co/datasets/mpilhlt/salamanca-abbr) have
abbreviation delimiters indicating beginning and end of abbreviation tokens (`⦃⦄`).
During training, 50% of examples have their abbreviation delimiters randomly
stripped. This teaches the model to both *expand pre-marked abbreviations* (when
markers are present) and *detect and expand abbreviations from orthographic cues
alone* (when markers are absent). At inference time, input is always unmarked —
the model relies on the distinctive Unicode characters of abbreviation markers
(macrons, tildes, brevigraphs) or typical character sequences to identify spans
requiring expansion.

## Training

The model was trained on approximately 1.5 million multi-line examples
constructed from TEI XML transcriptions of early modern legal and theological
texts in Latin and Spanish, published as part of the
[School of Salamanca Digital Collection](https://salamanca.school/).

### Training data

- **Source:** 2.4 million transcribed lines from the Salamanca corpus
- **Examples:** ~1.54 million training examples (sliding windows of 3–5 lines)
- **Context:** 1 line of context on each side of the central line
- **Abbreviation oversampling:** 2× oversampling of examples containing abbreviations
- **Document-level splits:** no document appears in more than one split
- **Dataset:** <https://huggingface.co/datasets/mpilhlt/salamanca-abbr>

### Training configuration

- **Base model:** `google/byt5-base` (582M parameters)
- **Training hardware:** 2× AMD Instinct MI300A (MPCDF Viper HPC)
- **Training duration:** approximately 45 hours (6 epochs, early stopping at epoch 6, without final test eval that took another 17 hours)
- **Optimizer:** AdamW, learning rate 1e-4 with linear warmup and cosine decay
- **Effective batch size:** 128 (batch 32 per GPU × 2 GPUs × 2 gradient accumulation steps)
- **Max input length:** 512 bytes
- **Max target length:** 384 bytes
- **Precision:** bf16
- **Marker dropout:** 0.5

### Training results

Loss and learning rate over training (early stopping triggered at epoch 6
with patience 3):

| Epoch | Train Loss | Eval Loss | Eval Span CER | Eval Span Exact Match | Eval Full-line CER |
|-------|------------|-----------|---------------|-----------------------|--------------------|
| 1     | 0.0059     | 0.0029    | 0.0312        | 87.89%                | 0.0013             |
| 2     | 0.0035     | 0.0032    | 0.0404        | 85.23%                | 0.0016             |
| 3     | 0.0022     | 0.0043    | 0.0404        | 84.99%                | 0.0017             |
| 4     | 0.0015     | 0.0039    | 0.0348        | 86.92%                | 0.0014             |
| 5     | 0.0010     | 0.0048    | 0.0340        | 86.92%                | 0.0014             |
| 6     | 0.0006     | 0.0058    | 0.0369        | 86.68%                | 0.0015             |

Note: span-level eval metrics are measured on unmarked input (abbreviation
delimiters stripped), so the model must both detect and expand abbreviations.
The eval set was capped at 1,000 examples for efficiency during training.

### Test set evaluation

Evaluated on a held-out document-level test set (20,000 examples from
documents not seen during training):

| Metric               | Value   |
|----------------------|---------|
| **Full-line CER**    | 0.00005 |
| **Span CER**         | 0.008   |
| **Span exact match** | 95.54%  |
| **Spans evaluated**  | 1,459   |

**Interpretation:**
- Full-line CER of 0.00005 means approximately one wrong character per 21,000
  characters
- Span exact match of 95.5% means about 1 in 22 abbreviations has any error
  (assuming the average length of abbreviations is 7 characters), but most
  errors are false negatives (abbreviation left unchanged) rather than
  incorrect expansions
- The rate of *incorrect* expansions (false positives) is approximately
  1 in 36 abbreviations (same assumption about abbreviation length)

### Common error patterns

The most frequent error types in the test set:

- **m/n confusion:** the tilde (~) over vowels is genuinely ambiguous between
  *m* and *n* in Latin abbreviation conventions (e.g., *ã* → *am* vs *an*)
- **False negatives:** the model leaves some abbreviations unexpanded,
  particularly less frequent forms
- **OCR/transcription artifacts:** the model cannot correct upstream
  transcription errors (e.g., *ſ/f* confusion: *plufquã* instead of *pluſquã*)

For details, see the [breakdown by abbreviation](./test_breakdown.json), which
contains counts and erroneous predictions of each abbreviation. (Or the
[test_breakdown_errors file](./test_breakdown_errors.json) that has been
filtered to include only those abbreviations where errors happened in test set
evaluation.)

## Intended use

This model is intended for post-processing transcribed early modern Latin
and Spanish printed texts. It works best as part of the full
[svsal-poco pipeline](https://github.com/digicademy/svsal-poco), which
chains boundary detection with abbreviation expansion.

It is **not** intended as a general-purpose text normalizer and has not
been evaluated on other corpora, time periods, or languages.

## How to use

### Installation

```bash
pip install transformers torch
# or, to install the full pipeline:
pip install git+https://github.com/digicademy/svsal-poco
```

### Direct use with transformers

```python
from transformers import AutoTokenizer, T5ForConditionalGeneration

model_repo = "mpilhlt/byt5-salamanca-abbr"
tokenizer  = AutoTokenizer.from_pretrained("google/byt5-base")
model      = T5ForConditionalGeneration.from_pretrained(
    model_repo, tie_word_embeddings=False
)
model.eval()

# Single line with abbreviation
text = "Lex est communis ciuitatis cōsensus qui"

enc = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
out = model.generate(**enc, max_length=384)
result = tokenizer.decode(out[0], skip_special_tokens=True)
print(result)
# → "Lex est communis ciuitatis consensus qui"
```

### Multi-line input with context

The model was trained on multi-line windows using two separator characters:

- `¬` (U+00AC): nonbreaking boundary — word continues across the line break
- `↵` (U+21B5): normal line boundary

```python
# Two lines with a nonbreaking boundary (word split across lines)
text = "ex epicheia poſſet occultè, absq́; paro¬cho, & teſtibus celebrari."

enc = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
out = model.generate(**enc, max_length=384)
result = tokenizer.decode(out[0], skip_special_tokens=True)
print(result)
# → "ex epicheia poſſet occultè, absque paro¬cho, & teſtibus celebrari."
```

### Use in the full pipeline

When using the full `svsal-poco` pipeline, the ByT5 model runs automatically
as the second stage after boundary detection. See the
[svsal-poco repository](https://github.com/digicademy/svsal-poco) for the
complete pipeline documentation.

```bash
python -m infer \
  --input             texts.jsonl \
  --output            expanded.jsonl \
  --boundary_model_dir ./canine-salamanca-boundary-classifier \
  --byt5_model_dir    mpilhlt/byt5-salamanca-abbr \
  --batch_size        32
```

### Interactive demo

Try the model in the
[SvSal PoCo HuggingFace Space](https://huggingface.co/spaces/awagner-mainz/svsal-poco),
which provides a web interface for both plain text and TEI XML input.

## Limitations

- Trained exclusively on texts from the School of Salamanca corpus; performance
  on other early modern corpora is unknown and may be lower
- The m/n ambiguity in tilde abbreviations is inherent to the notation system
  and cannot be fully resolved without semantic understanding
- The model cannot correct upstream OCR or transcription errors
- Cross-line abbreviations (where the abbreviated word straddles a line break)
  require correct boundary detection as a prerequisite
- Very rare abbreviation forms seen fewer than ~5 times in training may not
  be recognized
- The model has no explicit language identification; Latin and Spanish
  abbreviations are handled jointly

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
