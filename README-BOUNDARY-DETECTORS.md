# Boundary Detector Model Swapping

This document describes the boundary detector abstraction layer that allows easy swapping between different boundary detection models.

## Overview

The svsal-poco pipeline now supports multiple boundary detection models through a common interface. This allows you to:

1. Use the original Canine-based boundary classifier
2. Use the Flair-based Latin line break detector
3. Easily add new boundary detection models in the future

## Supported Models

### 1. Canine Boundary Detector (Default)

The original implementation using a custom PyTorch model based on `google/canine-s`.

**Model Details:**
- Framework: PyTorch + Transformers
- Base model: `google/canine-s`
- Input format: Line end + "↵" + line start
- Output: Binary classification (nonbreaking=1, breaking=0)
- Storage: `best_model.pt` + `threshold.json`
- Hugging Face: `mpilhlt/canine-salamanca-boundary-classifier`

**Usage with local directory:**
```bash
python infer_handler.py \
    --mode text \
    --input-file sample.txt \
    --output-file expanded.txt \
    --boundary-model-type canine \
    --boundary-model-dir ./canine-salamanca-boundary-classifier \
    --byt5-model-dir ./byt5-salamanca-abbr
```

**Usage with Hugging Face identifier:**
```bash
python infer_handler.py \
    --mode text \
    --input-file sample.txt \
    --output-file expanded.txt \
    --boundary-model-type canine \
    --boundary-model-name mpilhlt/canine-salamanca-boundary-classifier \
    --byt5-model-dir ./byt5-salamanca-abbr
```

### 2. Flair Boundary Detector

Uses the Flair sequence tagger model from Hugging Face.

**Model Details:**
- Framework: Flair
- Model: `mschonhardt/latin-contextual-lb-detector`
- Input format: Line end + `<lb/>` + line start
- Output: Sequence tags (WB = Split Word/Join, NB = Separate Words/Space)
- Storage: Hugging Face Hub

**Installation:**
```bash
pip install flair
```

**Usage:**
```bash
python infer_handler.py \
    --mode text \
    --input-file sample.txt \
    --output-file expanded.txt \
    --boundary-model-type flair \
    --boundary-model-name mschonhardt/latin-contextual-lb-detector \
    --byt5-model-dir ./byt5-salamanca-abbr
```

## Command-Line Arguments

### New Arguments

- `--boundary-model-type`: Type of boundary detector to use
  - Options: `canine`, `flair`
  - Default: `canine`

- `--boundary-model-dir`: Directory containing `best_model.pt` and `threshold.json` (for canine) OR Hugging Face model identifier
  - Works for: Both `canine` and `flair` detectors
  - For `canine`: Local directory with model files
  - For `flair`: Hugging Face model identifier or local path

- `--boundary-model-name`: Hugging Face model name (alias for `--boundary-model-dir`)
  - Works for: Both `canine` and `flair` detectors
  - For `canine`: Local directory with model files
  - For `flair`: Hugging Face model identifier or local path
  - Default for `flair`: `mschonhardt/latin-contextual-lb-detector`

**Note:** Both `--boundary-model-dir` and `--boundary-model-name` are interchangeable and work for both model types. You can use either parameter to specify the model path/identifier.

### Example Commands

#### Using Canine Detector (Default)

**With local directory:**
```bash
# Plaintext input
python infer_handler.py \
    --mode text \
    --input-file sample.txt \
    --output-file expanded.txt \
    --boundary-model-dir ./canine-salamanca-boundary-classifier \
    --byt5-model-dir ./byt5-salamanca-abbr

# JSONL input
python infer_handler.py \
    --mode jsonl \
    --input-file input.jsonl \
    --output-file out.jsonl \
    --boundary-model-dir ./canine-salamanca-boundary-classifier \
    --byt5-model-dir ./byt5-salamanca-abbr

# XML input
python infer_handler.py \
    --mode xml \
    --input-file input.xml \
    --output-file output.xml \
    --boundary-model-dir ./canine-salamanca-boundary-classifier \
    --byt5-model-dir ./byt5-salamanca-abbr
```

**With Hugging Face identifier:**
```bash
# Plaintext input (downloads model from Hugging Face)
python infer_handler.py \
    --mode text \
    --input-file sample.txt \
    --output-file expanded.txt \
    --boundary-model-name mpilhlt/canine-salamanca-boundary-classifier \
    --byt5-model-dir ./byt5-salamanca-abbr

# JSONL input
python infer_handler.py \
    --mode jsonl \
    --input-file input.jsonl \
    --output-file out.jsonl \
    --boundary-model-name mpilhlt/canine-salamanca-boundary-classifier \
    --byt5-model-dir ./byt5-salamanca-abbr

# XML input
python infer_handler.py \
    --mode xml \
    --input-file input.xml \
    --output-file output.xml \
    --boundary-model-name mpilhlt/canine-salamanca-boundary-classifier \
    --byt5-model-dir ./byt5-salamanca-abbr
```

#### Using Flair Detector
```bash
# Plaintext input with default Flair model
python infer_handler.py \
    --mode text \
    --input-file sample.txt \
    --output-file expanded.txt \
    --boundary-model-type flair \
    --byt5-model-dir ./byt5-salamanca-abbr

# JSONL input with custom Flair model
python infer_handler.py \
    --mode jsonl \
    --input-file input.jsonl \
    --output-file out.jsonl \
    --boundary-model-type flair \
    --boundary-model-name custom/latin-lb-detector \
    --byt5-model-dir ./byt5-salamanca-abbr
```

## Programmatic Usage

### Using the Detector Interface

```python
from boundary_detector import create_boundary_detector
from infer import run_pipeline

# Create a Canine detector
detector = create_boundary_detector(
    model_type="canine",
    model_dir="./canine-salamanca-boundary-classifier",
    use_lexicon=False,
)

# Or create a Flair detector
detector = create_boundary_detector(
    model_type="flair",
    model_name="mschonhardt/latin-contextual-lb-detector",
)

# Use the detector in the pipeline
run_pipeline(
    input_path="input.jsonl",
    output_path="output.jsonl",
    boundary_detector=detector,
    byt5_model_dir="./byt5-salamanca-abbr",
)
```

### Direct Detector Usage

```python
from boundary_detector import create_boundary_detector

# Create detector
detector = create_boundary_detector(
    model_type="canine",
    model_dir="./canine-salamanca-boundary-classifier",
)

# Prepare lines
lines = [
    {"id": "1", "doc_id": "doc1", "source_sic": "first line"},
    {"id": "2", "doc_id": "doc1", "source_sic": "second line"},
]

# Predict boundaries
lines_with_boundaries = detector.predict_boundaries(
    lines=lines,
    threshold=0.6,
    context_chars=40,
)

# Result includes 'predicted_nonbreaking_next_line' field
for line in lines_with_boundaries:
    print(f"{line['id']}: {line.get('predicted_nonbreaking_next_line', 'none')}")
```

## Model Comparison

| Feature | Canine | Flair |
|---------|--------|-------|
| Framework | PyTorch | Flair |
| Base Model | google/canine-s | Custom sequence tagger |
| Input Format | "↵" separator | `<lb/>` token |
| Output | Binary probability | Sequence tags |
| Threshold | Configurable (default 0.6) | Confidence score (default 0.5) |
| Storage | Local files | Hugging Face Hub |
| Dependencies | transformers | flair |
| Training Data | Salamanca corpus | Latin texts |

## Adding a New Boundary Detector

To add a new boundary detection model:

1. Create a new class in `boundary_detector.py` that inherits from `BoundaryDetector`:

```python
class MyBoundaryDetector(BoundaryDetector):
    def __init__(self, model_path: str):
        # Initialize your model
        self.model = load_my_model(model_path)
        self.default_threshold = 0.5
    
    def predict_boundaries(
        self,
        lines: list[dict],
        threshold: float = 0.5,
        context_chars: int = 40,
    ) -> list[dict]:
        # Implement prediction logic
        predictions = {}
        for i in range(len(lines) - 1):
            # Your prediction code here
            if should_join(lines[i], lines[i+1]):
                predictions[lines[i]["id"]] = lines[i+1]["id"]
        
        # Add predictions to lines
        result = []
        for row in lines:
            row = dict(row)
            row["predicted_nonbreaking_next_line"] = predictions.get(row["id"], "")
            result.append(row)
        return result
    
    def get_default_threshold(self) -> float:
        return self.default_threshold
```

2. Update the `create_boundary_detector` factory function:

```python
def create_boundary_detector(
    model_type: str,
    model_dir: Optional[str] = None,
    model_name: Optional[str] = None,
    use_lexicon: bool = False,
    device: Optional[torch.device] = None,
) -> BoundaryDetector:
    model_type = model_type.lower()
    
    if model_type == "canine":
        # ... existing code ...
    elif model_type == "flair":
        # ... existing code ...
    elif model_type == "mydetector":
        return MyBoundaryDetector(model_path=model_dir)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
```

3. Update command-line arguments in `infer_handler.py`:

```python
parser.add_argument(
    "--boundary-model-type",
    choices=["canine", "flair", "mydetector"],
    default="canine",
    help="Type of boundary detector (default: canine)"
)
```

## Backward Compatibility

The implementation maintains full backward compatibility:

- Existing scripts using `--boundary-model-dir` continue to work without changes
- The default behavior is unchanged (uses Canine detector)
- Legacy parameters (`boundary_model`, `boundary_tokenizer`) are still supported in `run_pipeline()`

## Troubleshooting

### Flair Import Error

If you get an error about missing Flair:

```bash
pip install flair
```

### Canine Model Not Found

If you get an error about missing `best_model.pt`:

- Ensure `--boundary-model-dir` points to the correct directory
- Verify the directory contains `best_model.pt` and `threshold.json`

### Threshold Issues

If you want to adjust the detection threshold:

```bash
# For Canine detector
python infer_handler.py \
    --mode text \
    --input-file sample.txt \
    --output-file expanded.txt \
    --boundary-model-type canine \
    --boundary-model-dir ./canine-salamanca-boundary-classifier \
    --byt5-model-dir ./byt5-salamanca-abbr \
    --threshold 0.7  # Higher threshold = more conservative

# For Flair detector
python infer_handler.py \
    --mode text \
    --input-file sample.txt \
    --output-file expanded.txt \
    --boundary-model-type flair \
    --byt5-model-dir ./byt5-salamanca-abbr \
    --threshold 0.6  # Higher threshold = more conservative
```

## References

- Original Canine boundary classifier: `boundary_classifier/boundary_classifier.py`
- Flair model: https://huggingface.co/mschonhardt/latin-contextual-lb-detector
- Jupyter notebook example: https://github.com/michaelscho/ml-notebooks/blob/main/latin_lb_detector.ipynb
