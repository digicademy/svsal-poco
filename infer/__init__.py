# infer/__init__.py
#
# Inference pipeline: chains the boundary classifier and ByT5 expansion model.
#
# Given a JSONL file of unseen lines (source_sic only, no markup),
# this script:
#   1. Sorts lines into document order
#   2. Runs the boundary classifier to identify nonbreaking line pairs
#   3. Builds sliding windows of lines (matching training format)
#   4. Runs ByT5 to expand abbreviations
#   5. Splits output on ↵/¬ to restore line structure, keeping only
#      "owned" lines from each window
#   6. Writes a JSONL output file with expanded text per line

import json
import re
import argparse
import torch
from pathlib import Path
from collections import defaultdict
from typing import Optional
from transformers import AutoTokenizer, T5ForConditionalGeneration, CanineTokenizer

from data.data_utils import load_and_sort_lines, CorpusLexicon, ABBR_OPEN, ABBR_CLOSE, LINE_BREAK, LINE_SEP
from boundary_classifier.boundary_detector import BoundaryDetector, create_boundary_detector
from boundary_classifier.boundary_classifier import BoundaryClassifier, predict_boundaries


def load_byt5_tokenizer(model_id_or_path: str, **kwargs):
    try:
        return AutoTokenizer.from_pretrained(model_id_or_path, **kwargs)
    except AttributeError as exc:
        if "'list' object has no attribute 'keys'" not in str(exc):
            raise
        print("Retrying ByT5 tokenizer load with normalized extra_special_tokens...")
        return AutoTokenizer.from_pretrained(
            model_id_or_path,
            extra_special_tokens={},
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Sliding window construction
# ---------------------------------------------------------------------------

def build_nonbreaking_chains(
    lines: list[dict],
) -> list[list[int]]:
    """
    Group line indices into nonbreaking chains based on boundary
    predictions. Each chain is a list of consecutive line indices
    that must be concatenated (joined with LINE_SEP) because words
    continue across the line breaks.

    A standalone line is a chain of length 1.
    """
    index = {row["id"]: i for i, row in enumerate(lines)}
    consumed: set[int] = set()
    chains: list[list[int]] = []

    for i, row in enumerate(lines):
        if i in consumed:
            continue

        chain = [i]
        consumed.add(i)
        current = row

        while True:
            next_id = current.get("predicted_nonbreaking_next_line", "")
            next_idx = index.get(next_id) if next_id else None
            if next_idx is not None and next_idx not in consumed:
                chain.append(next_idx)
                consumed.add(next_idx)
                current = lines[next_idx]
            else:
                break

        chains.append(chain)

    return chains


def build_sliding_windows(
    lines:          list[dict],
    chains:         list[list[int]],
    context_lines:  int = 2,
    max_bytes:      int = 480,
) -> list[dict]:
    """
    Build sliding windows for ByT5 inference, mirroring the training
    format from build_byt5_examples.

    Each window contains:
    - A central chain (the "owned" lines whose output we keep)
    - Up to context_lines lines before and after for context

    The window is bounded by max_bytes to avoid exceeding the model's
    input length. Context is trimmed (not owned lines) if the window
    would be too long.

    Returns a list of window dicts:
        {
            "line_indices":  [int, ...],   # all line indices in window
            "owned_start":   int,          # offset into line_indices
            "owned_end":     int,          # offset into line_indices
        }
    """
    windows = []

    for chain in chains:
        chain_start = chain[0]
        chain_end = chain[-1]

        # Find valid context range (don't cross document boundaries)
        doc_id = lines[chain_start]["doc_id"]

        # Context before: walk backwards, same document
        ctx_before = []
        idx = chain_start - 1
        while idx >= 0 and len(ctx_before) < context_lines:
            if lines[idx]["doc_id"] != doc_id:
                break
            ctx_before.insert(0, idx)
            idx -= 1

        # Context after: walk forwards, same document
        ctx_after = []
        idx = chain_end + 1
        while idx < len(lines) and len(ctx_after) < context_lines:
            if lines[idx]["doc_id"] != doc_id:
                break
            ctx_after.append(idx)
            idx += 1

        # Check byte budget — trim context if needed
        all_indices = ctx_before + chain + ctx_after
        while _window_byte_len(lines, all_indices) > max_bytes and ctx_after:
            ctx_after.pop()
            all_indices = ctx_before + chain + ctx_after
        while _window_byte_len(lines, all_indices) > max_bytes and ctx_before:
            ctx_before.pop(0)
            all_indices = ctx_before + chain + ctx_after

        owned_start = len(ctx_before)
        owned_end = owned_start + len(chain)

        windows.append({
            "line_indices": all_indices,
            "owned_start":  owned_start,
            "owned_end":    owned_end,
        })

    return windows


def _window_byte_len(lines: list[dict], indices: list[int]) -> int:
    """Estimate byte length of a window when lines are joined with LINE_SEP."""
    if not indices:
        return 0
    total = sum(len(lines[i]["source_sic"].encode("utf-8")) for i in indices)
    total += len(indices) - 1  # LINE_SEP characters
    return total


# ---------------------------------------------------------------------------
# ByT5 batched inference
# ---------------------------------------------------------------------------

def expand_abbreviations(
    examples:          list[dict],   # list of {source, line_ids, ...} dicts
    model:             T5ForConditionalGeneration,
    tokenizer:         AutoTokenizer,
    max_input_length:  int = 512,
    max_target_length: int = 384,
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

    sources = [e["source"] for e in examples]
    outputs = []

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


def _split_window_output_parts(
    output: str,
    expected_parts: int,
    split_pattern: re.Pattern,
) -> tuple[list[str], Optional[str]]:
    """
    Split a window output into line parts.

    Returns (parts, warning_suffix). warning_suffix is None on exact split.
    """
    if expected_parts <= 1:
        return [output], None

    raw_parts = split_pattern.split(output)
    if len(raw_parts) == expected_parts:
        return raw_parts, None

    limited_parts = split_pattern.split(output, maxsplit=expected_parts - 1)
    if len(limited_parts) == expected_parts:
        return limited_parts, (
            f"recovered via maxsplit (raw split produced {len(raw_parts)} parts)"
        )

    return raw_parts, None


def _recover_window_parts(
    parts: list[str],
    expected_parts: int,
    source_lines: list[str],
) -> tuple[list[str], list[str]]:
    """
    Recover window parts to expected_parts using conservative best effort.

    Returns (recovered_parts, recovery_notes).
    """
    recovered = list(parts)
    notes: list[str] = []

    if len(recovered) > expected_parts:
        overflow = len(recovered) - expected_parts + 1
        recovered = recovered[:expected_parts - 1] + ["".join(recovered[expected_parts - 1:])]
        notes.append(f"collapsed {overflow} overflow part(s) into final line")

    if len(recovered) < expected_parts:
        missing_count = expected_parts - len(recovered)
        recovered.extend(source_lines[len(recovered):expected_parts])
        notes.append(f"filled {missing_count} missing part(s) from original source lines")

    return recovered, notes


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    input_path:          str,
    output_path:         str,
    boundary_model_dir:  str   | None = None,
    byt5_model_dir:      str   | None = None,
    lexicon_data_path:   str   | None = None,
    boundary_threshold:  float | None = None,
    batch_size:          int   = 32,
    lang_prefix:         bool  = False,
    context_chars:       int   = 40,
    context_lines:       int   = 2,
    # Pre-loaded objects — skip loading when provided
    pre_annotated_boundaries: dict = None,
    boundary_model:      BoundaryClassifier         | None = None,
    boundary_tokenizer:  CanineTokenizer            | None = None,
    byt5_model:          T5ForConditionalGeneration | None = None,
    byt5_tokenizer:      AutoTokenizer              | None = None,
    # New: boundary detector interface
    boundary_detector:   BoundaryDetector           | None = None,
):
    """
    Full inference pipeline for unseen texts.

    input_path:         JSONL file with at minimum: id, doc_id, source_sic,
                        lang fields. No abbreviation markup expected.
    output_path:        JSONL output file with expanded_text field added.
    boundary_model_dir: directory containing best_model.pt and threshold.json
    byt5_model_dir:     HF model directory or Hub repo id for ByT5
    lexicon_data_path:  optional path to training JSONL for lexicon construction
    context_lines:      number of context lines on each side of owned lines
    pre_annotated_boundaries: optional dict mapping line id to predicted nonbreaking next line id,
                        to inject known boundary classification information.
    boundary_detector:  Optional pre-loaded BoundaryDetector instance. If provided,
                        boundary_model and boundary_tokenizer are ignored.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load boundary detector (new interface) ---
    if boundary_detector is not None:
        # Use the provided detector
        print(f"Using pre-loaded boundary detector")
        if boundary_threshold is None:
            boundary_threshold = boundary_detector.get_default_threshold()
        print(f"Boundary threshold: {boundary_threshold}")
        
        # Optional lexicon
        lexicon = None
        if lexicon_data_path:
            lexicon = CorpusLexicon()
            lexicon.build_from_jsonl(lexicon_data_path)
            # Set lexicon on detector if it supports it
            if hasattr(boundary_detector, 'set_lexicon'):
                boundary_detector.set_lexicon(lexicon)
    else:
        # --- Load boundary classifier (legacy interface, for backward compatibility) ---
        if boundary_model is None:
            if boundary_model_dir is None:
                raise ValueError("Provide boundary_detector, boundary_model, or boundary_model_dir")
            print("Loading boundary classifier...")
            boundary_tokenizer = CanineTokenizer.from_pretrained("google/canine-s")
            boundary_model = BoundaryClassifier(
                use_lexicon=(lexicon_data_path is not None),
            )
            boundary_model.load_state_dict(
                torch.load(f"{boundary_model_dir}/best_model.pt",
                            map_location=device)
            )
            print(f"Boundary model loaded from {boundary_model_dir}")

        if boundary_threshold is None:
            if boundary_model_dir:
                threshold_path = Path(boundary_model_dir) / "threshold.json"
                if threshold_path.exists():
                    boundary_threshold = json.loads(
                        threshold_path.read_text()
                    )["threshold"]
            if boundary_threshold is None:
                boundary_threshold = 0.6
                print(f"No threshold found; using default {boundary_threshold}")
        print(f"Boundary threshold: {boundary_threshold}")

        # Optional lexicon
        lexicon = None
        if lexicon_data_path:
            lexicon = CorpusLexicon()
            lexicon.build_from_jsonl(lexicon_data_path)

    # --- Load ByT5 (only if not provided) ---
    if byt5_model is None:
        if byt5_model_dir is None:
            raise ValueError("Provide byt5_model or byt5_model_dir")
        print("Loading ByT5...")
        byt5_tokenizer = load_byt5_tokenizer(byt5_model_dir)
        byt5_model = T5ForConditionalGeneration.from_pretrained(byt5_model_dir)
        print(f"ByT5 model loaded from {byt5_model_dir}")

    # --- Load and sort input lines ---
    print("Loading input lines...")
    lines = load_and_sort_lines(input_path)

    # --- Stage 1: boundary classification ---
    print("Running boundary classifier...")
    if boundary_detector is not None:
        # Use new detector interface
        lines_with_boundaries = boundary_detector.predict_boundaries(
            lines=lines,
            threshold=boundary_threshold,
            context_chars=context_chars,
        )
    else:
        # Use legacy interface
        lines_with_boundaries = predict_boundaries(
            lines=lines,
            model=boundary_model,
            tokenizer=boundary_tokenizer,
            lexicon=lexicon,
            threshold=boundary_threshold,
            context_chars=context_chars,
        )

    # --- Stage 1b: override with pre-annotated boundaries ---
    if pre_annotated_boundaries:
        for row in lines_with_boundaries:
            if row["id"] in pre_annotated_boundaries:
                row["predicted_nonbreaking_next_line"] = (
                    pre_annotated_boundaries[row["id"]]
                )

    # --- Stage 2: build sliding windows ---
    print("Building nonbreaking chains...")
    chains = build_nonbreaking_chains(lines_with_boundaries)
    print(f"  {len(chains)} chains from {len(lines_with_boundaries)} lines "
          f"(longest chain: {max(len(c) for c in chains)} lines)")

    print(f"Building sliding windows (context_lines={context_lines})...")
    windows = build_sliding_windows(
        lines_with_boundaries, chains,
        context_lines=context_lines,
    )
    print(f"  {len(windows)} windows")

    # Build ByT5 examples from windows
    byt5_examples = []
    for window in windows:
        indices = window["line_indices"]
        parts = []
        for j, idx in enumerate(indices):
            parts.append(lines_with_boundaries[idx]["source_sic"])
            if j < len(indices) - 1:
                current = lines_with_boundaries[idx]
                next_id = lines_with_boundaries[indices[j + 1]]["id"]
                if current.get("predicted_nonbreaking_next_line", "") == next_id:
                    parts.append(LINE_SEP)
                else:
                    parts.append(LINE_BREAK)
        source = "".join(parts)

        byt5_examples.append({
            "source":   source,
            "window":   window,
        })

    # --- Stage 3: ByT5 expansion ---
    print(f"Running ByT5 on {len(byt5_examples)} windows...")
    expanded_outputs = expand_abbreviations(
        examples=byt5_examples,
        model=byt5_model,
        tokenizer=byt5_tokenizer,
        batch_size=batch_size,
        device=device,
    )

    # --- Stage 4: restore line structure ---
    # Split each output on LINE_SEP, keep only owned lines
    output_by_line_id = {}

    LINE_SPLIT_PATTERN = re.compile(f"[{re.escape(LINE_SEP)}{re.escape(LINE_BREAK)}]")
    for example, output in zip(byt5_examples, expanded_outputs):
        window = example["window"]
        indices = window["line_indices"]
        owned_start = window["owned_start"]
        owned_end = window["owned_end"]
        expected_parts = len(indices)
        parts, split_note = _split_window_output_parts(
            output, expected_parts, LINE_SPLIT_PATTERN,
        )

        source_lines = [lines_with_boundaries[idx]["source_sic"] for idx in indices]
        recovered, warning_notes = _recover_window_parts(
            parts, expected_parts, source_lines,
        )
        if split_note:
            warning_notes.append(split_note)

        if warning_notes:
            print(
                f"  Warning: window expected {expected_parts} parts, "
                f"got {len(parts)}. Applying best-effort recovery: "
                + "; ".join(warning_notes)
                + "."
            )

        # Keep only owned lines
        for offset in range(owned_start, owned_end):
            line_idx = indices[offset]
            line_id = lines_with_boundaries[line_idx]["id"]
            output_by_line_id[line_id] = recovered[offset]

    # --- Stage 5: write output ---
    print(f"Writing output to {output_path}...")
    with open(output_path, "w") as f:
        for row in lines_with_boundaries:
            out_row = dict(row)
            out_row["expanded_text"] = output_by_line_id.get(
                row["id"], row["source_sic"]
            )
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
    p.add_argument("--context_lines",      type=int,   default=2)
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
        context_lines=args.context_lines,
    )
