# infer.py
#
# Inference pipeline: chains the boundary classifier and ByT5 expansion model.
#
# Given a JSONL file of unseen lines (source_sic only, no markup),
# this script:
#   1. Sorts lines into document order
#   2. Runs the boundary classifier to identify nonbreaking line pairs
#   3. Concatenates identified pairs with ↵
#   4. Runs ByT5 to expand abbreviations
#   5. Splits output on ↵ to restore line structure
#   6. Writes a JSONL output file with expanded text per line

import json
import argparse
import torch
from pathlib import Path
from collections import defaultdict
from transformers import AutoTokenizer, T5ForConditionalGeneration, CanineTokenizer

from data.data_utils import load_and_sort_lines, CorpusLexicon, ABBR_OPEN, ABBR_CLOSE, LINE_SEP
from boundary_classifier.boundary_classifier import BoundaryClassifier, predict_boundaries


# ---------------------------------------------------------------------------
# ByT5 batched inference
# ---------------------------------------------------------------------------

def expand_abbreviations(
    examples:          list[dict],   # list of {source, line_ids} dicts
    model:             T5ForConditionalGeneration,
    tokenizer:         AutoTokenizer,
    max_input_length:  int = 512,
    max_target_length: int = 512,
    batch_size:        int = 32,
    device:            torch.device = None,
) -> list[str]:
    """
    Run ByT5 on a list of source strings, returning expanded output strings.
    Handles batching explicitly for inference — Trainer is not used here.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    sources  = [e["source"] for e in examples]
    outputs  = []

    for i in range(0, len(sources), batch_size):
        batch_sources = sources[i:i + batch_size]
        enc = tokenizer(
            batch_sources,
            max_length=max_input_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            generated = model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_new_tokens=max_target_length,
            )

        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        outputs.extend(decoded)

    return outputs


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    input_path:          str,
    output_path:         str,
    boundary_model_dir:  str,
    byt5_model_dir:      str,
    lexicon_data_path:   str = None,
    boundary_threshold:  float = None,   # None = load from threshold.json
    batch_size:          int   = 32,
    context_chars:       int   = 40,
):
    """
    Full inference pipeline for unseen texts.

    input_path:         JSONL file with at minimum: id, doc_id, source_sic,
                        lang fields. No abbreviation markup expected.
    output_path:        JSONL output file with expanded_text field added.
    boundary_model_dir: directory containing best_model.pt and threshold.json
    byt5_model_dir:     HF model directory or Hub repo id for ByT5
    lexicon_data_path:  optional path to training JSONL for lexicon construction
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load boundary classifier ---
    print("Loading boundary classifier...")
    canine_tokenizer = CanineTokenizer.from_pretrained("google/canine-s")
    boundary_model   = BoundaryClassifier(use_lexicon=(lexicon_data_path is not None))
    boundary_model.load_state_dict(
        torch.load(f"{boundary_model_dir}/best_model.pt", map_location=device)
    )

    # Load threshold selected during training
    if boundary_threshold is None:
        threshold_path = Path(boundary_model_dir) / "threshold.json"
        if threshold_path.exists():
            boundary_threshold = json.loads(threshold_path.read_text())["threshold"]
        else:
            boundary_threshold = 0.6
            print(f"No threshold.json found; using default {boundary_threshold}")
    print(f"Boundary threshold: {boundary_threshold}")

    # Optional lexicon
    lexicon = None
    if lexicon_data_path:
        lexicon = CorpusLexicon()
        lexicon.build_from_jsonl(lexicon_data_path)

    # --- Load ByT5 ---
    print("Loading ByT5...")
    byt5_tokenizer = AutoTokenizer.from_pretrained(byt5_model_dir)
    byt5_model     = T5ForConditionalGeneration.from_pretrained(byt5_model_dir)

    # --- Load and sort input lines ---
    print("Loading input lines...")
    lines = load_and_sort_lines(input_path)

    # --- Stage 1: boundary classification ---
    print("Running boundary classifier...")
    lines_with_boundaries = predict_boundaries(
        lines=lines,
        model=boundary_model,
        tokenizer=canine_tokenizer,
        lexicon=lexicon,
        threshold=boundary_threshold,
        context_chars=context_chars,
    )

    # --- Stage 2: build ByT5 input examples (concatenate pairs) ---
    print("Building ByT5 input examples...")
    index    = {row["id"]: row for row in lines_with_boundaries}
    consumed = set()
    byt5_examples = []   # each entry: {source, line_ids}

    for row in lines_with_boundaries:
        if row["id"] in consumed:
            continue

        next_id  = row.get("predicted_nonbreaking_next_line", "")
        next_row = index.get(next_id) if next_id else None

        if next_row:
            source   = row["source_sic"] + LINE_SEP + next_row["source_sic"]
            line_ids = [row["id"], next_row["id"]]
            consumed.add(next_row["id"])
        else:
            source   = row["source_sic"]
            line_ids = [row["id"]]

        byt5_examples.append({"source": source, "line_ids": line_ids})

    # --- Stage 3: ByT5 expansion ---
    print(f"Running ByT5 on {len(byt5_examples)} examples...")
    expanded_outputs = expand_abbreviations(
        examples=byt5_examples,
        model=byt5_model,
        tokenizer=byt5_tokenizer,
        batch_size=batch_size,
        device=device,
    )

    # --- Stage 4: restore line structure ---
    # Split on LINE_SEP to recover per-line expanded text
    output_by_line_id = {}
    for example, output in zip(byt5_examples, expanded_outputs):
        parts = output.split(LINE_SEP)
        for line_id, part in zip(example["line_ids"], parts):
            output_by_line_id[line_id] = part

    # --- Stage 5: write output ---
    print(f"Writing output to {output_path}...")
    with open(output_path, "w") as f:
        for row in lines:
            out_row = dict(row)
            out_row["expanded_text"] = output_by_line_id.get(row["id"], row["source_sic"])
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    print("Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",              required=True)
    p.add_argument("--output",             required=True)
    p.add_argument("--boundary_model_dir", required=True)
    p.add_argument("--byt5_model_dir",     required=True)
    p.add_argument("--lexicon_data",       default=None)
    p.add_argument("--threshold",          type=float, default=None)
    p.add_argument("--batch_size",         type=int,   default=32)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        input_path=args.input,
        output_path=args.output,
        boundary_model_dir=args.boundary_model_dir,
        byt5_model_dir=args.byt5_model_dir,
        lexicon_data_path=args.lexicon_data,
        boundary_threshold=args.threshold,
        batch_size=args.batch_size,
    )
