#!/usr/bin/env python3
"""
Local inference handler for svsal-poco.

Replicates the space app behavior for:
1) Plaintext input (one line per line) -> full pipeline
2) JSONL input (existing README format) -> full pipeline
3) TEI XML input -> XML roundtripping with expansions + break="no"

Usage examples:
  # Plaintext from file
  python infer_handler.py --mode text --input-file sample.txt --output-file expanded.txt \
      --boundary-model-dir ./canine-salamanca-boundary-classifier \
      --byt5-model-dir ./byt5-salamanca-abbr

  # Plaintext from stdin
  cat sample.txt | python infer_handler.py --mode text --stdin \
      --boundary-model-dir ./canine-salamanca-boundary-classifier \
      --byt5-model-dir ./byt5-salamanca-abbr

  # JSONL passthrough mode
  python infer_handler.py --mode jsonl --input-file input.jsonl --output-file out.jsonl \
      --boundary-model-dir ./canine-salamanca-boundary-classifier \
      --byt5-model-dir ./byt5-salamanca-abbr

  # XML from file
  python infer_handler.py --mode xml --input-file input.xml --output-file output.xml \
      --boundary-model-dir ./canine-salamanca-boundary-classifier \
      --byt5-model-dir ./byt5-salamanca-abbr
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration, CanineTokenizer

from boundary_classifier.boundary_classifier import BoundaryClassifier
from infer import run_pipeline
from tei.tei_roundtrip import process_tei_xml


NONBREAKING_MARKER = "¬"


def load_models(boundary_model_dir: str, byt5_model_dir: str):
    # Boundary model
    boundary_tokenizer = CanineTokenizer.from_pretrained("google/canine-s")
    boundary_model = BoundaryClassifier(use_lexicon=False)

    weights_path = os.path.join(boundary_model_dir, "best_model.pt")
    threshold_path = os.path.join(boundary_model_dir, "threshold.json")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Missing boundary weights: {weights_path}")
    if not os.path.exists(threshold_path):
        raise FileNotFoundError(f"Missing boundary threshold: {threshold_path}")

    boundary_model.load_state_dict(
        torch.load(weights_path, map_location="cpu", weights_only=True)
    )
    boundary_model.eval()

    boundary_threshold = json.loads(Path(threshold_path).read_text(encoding="utf-8"))["threshold"]

    # ByT5 model (same tokenizer choice as app.py)
    byt5_tokenizer = AutoTokenizer.from_pretrained("google/byt5-base")
    byt5_model = T5ForConditionalGeneration.from_pretrained(
        byt5_model_dir,
        tie_word_embeddings=False,
    )
    byt5_model.eval()

    return boundary_model, boundary_tokenizer, boundary_threshold, byt5_model, byt5_tokenizer


def read_input(args) -> str:
    if args.stdin:
        return sys.stdin.read()
    if args.input_file:
        return Path(args.input_file).read_text(encoding="utf-8")
    raise ValueError("Provide --stdin or --input-file")


def write_output(s: str, output_file: str | None):
    if output_file:
        Path(output_file).write_text(s, encoding="utf-8")
    else:
        sys.stdout.write(s)


def plaintext_to_rows(text: str) -> tuple[list[dict], dict[str, str]]:
    """Returns (rows, pre_annotated_boundaries)."""
    rows = []
    pre_annotated = {}
    raw_lines = text.strip().splitlines()

    for i, line in enumerate(raw_lines):
        line = line.strip()
        if not line:
            continue
        line_id = f"demo-00-0001-lb-{i:04d}"
        is_nonbreaking = line.endswith(NONBREAKING_MARKER)
        clean_line = line.rstrip(NONBREAKING_MARKER) if is_nonbreaking else line

        rows.append({
            "id": line_id,
            "doc_id": "demo",
            "facs_id": "demo-0001",
            "ancestor_id": "demo-00-0001-pa-0001",
            "lang": ["la"],
            "source_sic": clean_line,
            "target_corr": clean_line,
            "contains_abbr": "false",
            "nonbreaking_next_line": "",
        })

        if is_nonbreaking and i + 1 < len(raw_lines):
            next_id = f"demo-00-0001-lb-{i+1:04d}"
            pre_annotated[line_id] = next_id

    return rows, pre_annotated


def run_pipeline_on_rows(rows, output_format, models, batch_size=16,
                         lang_prefix=False, pre_annotated_boundaries=None):
    boundary_model, boundary_tokenizer, boundary_threshold, byt5_model, byt5_tokenizer = models

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        input_path = f.name
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    output_path = input_path.replace(".jsonl", "_out.jsonl")

    try:
        run_pipeline(
            input_path=input_path,
            output_path=output_path,
            boundary_model=boundary_model,
            boundary_tokenizer=boundary_tokenizer,
            boundary_threshold=boundary_threshold,
            byt5_model=byt5_model,
            byt5_tokenizer=byt5_tokenizer,
            pre_annotated_boundaries=pre_annotated_boundaries,
            batch_size=batch_size,
            lang_prefix=lang_prefix,
        )

        out_rows = []
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out_rows.append(json.loads(line))

        if output_format == "jsonl":
            return "\n".join(json.dumps(r, ensure_ascii=False) for r in out_rows) + "\n"

        # "text" output: mimic app.py (expanded text with ¬ on nonbreaking lines)
        nonbreaking_ids = {
            r["id"] for r in out_rows
            if r.get("predicted_nonbreaking_next_line")
        }
        lines = []
        for r in out_rows:
            line = r.get("expanded_text", r["source_sic"])
            if r["id"] in nonbreaking_ids:
                line += NONBREAKING_MARKER
            lines.append(line)
        return "\n".join(lines) + "\n"

    finally:
        if os.path.exists(input_path):
            os.unlink(input_path)
        if os.path.exists(output_path):
            os.unlink(output_path)


def run_xml(xml_text, models, batch_size=16):
    boundary_model, boundary_tokenizer, boundary_threshold, byt5_model, byt5_tokenizer = models

    def pipeline_fn(line_rows, pre_annotated=None):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as tmp:
            input_path = tmp.name
            for row in line_rows:
                tmp.write(json.dumps(row, ensure_ascii=False) + "\n")
        output_path = input_path.replace(".jsonl", "_out.jsonl")

        try:
            run_pipeline(
                input_path=input_path,
                output_path=output_path,
                boundary_model=boundary_model,
                boundary_tokenizer=boundary_tokenizer,
                boundary_threshold=boundary_threshold,
                byt5_model=byt5_model,
                byt5_tokenizer=byt5_tokenizer,
                pre_annotated_boundaries=pre_annotated,
                batch_size=batch_size,
            )

            out_rows = []
            with open(output_path, encoding="utf-8") as f:
                for l in f:
                    l = l.strip()
                    if l:
                        out_rows.append(json.loads(l))

            expanded_dict = {r["id"]: r.get("expanded_text", r["source_sic"]) for r in out_rows}
            boundary_dict = {
                r["id"]: r["predicted_nonbreaking_next_line"]
                for r in out_rows if r.get("predicted_nonbreaking_next_line")
            }
            return expanded_dict, boundary_dict
        finally:
            if os.path.exists(input_path):
                os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)

    return process_tei_xml(xml_text, pipeline_fn)


def main():
    parser = argparse.ArgumentParser(description="Local SvSal PoCo inference handler")
    parser.add_argument("--mode", choices=["text", "jsonl", "xml"], required=True)
    parser.add_argument("--stdin", action="store_true", help="Read input from stdin")
    parser.add_argument("--input-file", help="Path to input file")
    parser.add_argument("--output-file", help="Path to output file (defaults to stdout)")
    parser.add_argument("--boundary-model-dir", required=True, help="Dir with best_model.pt and threshold.json")
    parser.add_argument("--byt5-model-dir", required=True, help="Dir/loadable HF path for ByT5 model")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lang-prefix", action="store_true", default=False)

    # for --mode text only
    parser.add_argument(
        "--text-output",
        choices=["text", "jsonl"],
        default="text",
        help="For plaintext input: output expanded plaintext (default) or output JSONL rows",
    )

    args = parser.parse_args()

    raw = read_input(args)
    models = load_models(args.boundary_model_dir, args.byt5_model_dir)

    if args.mode == "jsonl":
        # passthrough: run pipeline directly on provided JSONL
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
            f.write(raw)
            input_path = f.name
        output_path = input_path.replace(".jsonl", "_out.jsonl")
        try:
            boundary_model, boundary_tokenizer, boundary_threshold, byt5_model, byt5_tokenizer = models
            run_pipeline(
                input_path=input_path,
                output_path=output_path,
                boundary_model=boundary_model,
                boundary_tokenizer=boundary_tokenizer,
                boundary_threshold=boundary_threshold,
                byt5_model=byt5_model,
                byt5_tokenizer=byt5_tokenizer,
                batch_size=args.batch_size,
                lang_prefix=args.lang_prefix,
            )
            out = Path(output_path).read_text(encoding="utf-8")
            write_output(out, args.output_file)
        finally:
            if os.path.exists(input_path):
                os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)

    elif args.mode == "text":
        rows, pre_annotated = plaintext_to_rows(raw)
        out = run_pipeline_on_rows(
            rows,
            output_format=args.text_output,
            models=models,
            batch_size=args.batch_size,
            lang_prefix=args.lang_prefix,
            pre_annotated_boundaries=pre_annotated,
        )
        write_output(out, args.output_file)

    elif args.mode == "xml":
        out_xml = run_xml(raw, models=models, batch_size=args.batch_size)
        write_output(out_xml, args.output_file)


if __name__ == "__main__":
    main()
