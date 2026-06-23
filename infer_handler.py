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

  # XML from file — also save intermediate JSONL for later re-injection debugging
  python infer_handler.py --mode xml --input-file input.xml --output-file output.xml \
      --boundary-model-dir ./canine-salamanca-boundary-classifier \
      --byt5-model-dir ./byt5-salamanca-abbr \
      --save-intermediate
  # writes output.extracted.jsonl (rows from XML) and output.inferred.jsonl (ML results)

  # Re-inject from saved JSONL only — no models / GPU needed; iterates on tei_roundtrip.py
  python infer_handler.py --mode xml --input-file input.xml --output-file output.xml \
      --infer-jsonl output.inferred.jsonl
  # or use the convenience wrapper:
  #   ./reinject_xml.sh --input input.xml --infer-jsonl output.inferred.jsonl --output output.xml
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import torch
from transformers import T5ForConditionalGeneration

from boundary_classifier.boundary_detector import create_boundary_detector
from infer import load_byt5_tokenizer, run_pipeline
from tei.tei_roundtrip import process_tei_xml


NONBREAKING_MARKER = "¬"


def load_models(
    boundary_model_dir: str | None = None,
    byt5_model_dir: str = "",
    boundary_model_type: str = "canine",
    boundary_model_name: str | None = None,
):
    """
    Load models for inference.
    
    Args:
        boundary_model_dir: Directory with best_model.pt (for canine) or HF model identifier
        byt5_model_dir: Directory or HF path for ByT5 model
        boundary_model_type: Type of boundary detector ('canine' or 'flair')
        boundary_model_name: HF model name (alias for boundary_model_dir)
        
    Returns:
        Tuple of (boundary_detector, boundary_threshold, byt5_model, byt5_tokenizer)
    """

    # Load boundary detector
    if boundary_model_type == "canine":
        # Use boundary_model_dir or boundary_model_name (both work for canine)
        model_path = boundary_model_dir or boundary_model_name
        if not model_path:
            raise ValueError("boundary_model_dir or boundary_model_name is required for canine detector")
        boundary_detector = create_boundary_detector(
            model_type="canine",
            model_path=model_path,
            use_lexicon=False,
        )
        boundary_threshold = boundary_detector.get_default_threshold()
    elif boundary_model_type == "flair":
        # Use boundary_model_name or boundary_model_dir (both work for flair)
        model_name = boundary_model_name or (boundary_model_dir + "/pytorch_model.bin" if boundary_model_dir else None)
        boundary_detector = create_boundary_detector(
            model_type="flair",
            model_name=model_name or "mschonhardt/latin-contextual-lb-detector",
        )
        boundary_threshold = boundary_detector.get_default_threshold()
    else:
        raise ValueError(f"Unknown boundary_model_type: {boundary_model_type}")

    # ByT5 model/tokenizer from local path (offline-safe)
    byt5_tokenizer = load_byt5_tokenizer(
        byt5_model_dir,
        local_files_only=True,
    )
    byt5_model = T5ForConditionalGeneration.from_pretrained(
        byt5_model_dir,
        tie_word_embeddings=False,
        local_files_only=True,
    )
    byt5_model.eval()

    return boundary_detector, boundary_threshold, byt5_model, byt5_tokenizer


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
    boundary_detector, boundary_threshold, byt5_model, byt5_tokenizer = models

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        input_path = f.name
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    output_path = input_path.replace(".jsonl", "_out.jsonl")

    try:
        run_pipeline(
            input_path=input_path,
            output_path=output_path,
            boundary_detector=boundary_detector,
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


def _write_reconstruction_report(report: list, path: str | None) -> None:
    """Write the sentinel-repair audit log: one JSON object per repaired line
    ({line_id, from_line_id, ratio, kind}). No-op when *path* is falsy or the
    report is empty, so clean documents leave no stray sidecar."""
    if not path or not report:
        return
    with open(path, "w", encoding="utf-8") as fh:
        for entry in report:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"[reconstruction] {len(report)} line(s) repaired \u2192 {path}", file=sys.stderr)


def run_xml(xml_text, models, batch_size=16, save_intermediate_path: str | None = None,
            reconstruction_report_path: str | None = None):
    """Run XML roundtrip inference, optionally persisting intermediate data.

    Args:
        xml_text:               Input TEI XML string.
        models:                 Tuple returned by :func:`load_models`.
        batch_size:             Forwarded to :func:`run_pipeline`.
        save_intermediate_path: Optional path *stem* (no extension).
                                When given, two sibling files are written:
                                ``<stem>.extracted.jsonl`` — rows extracted
                                from the XML before inference, and
                                ``<stem>.inferred.jsonl``  — rows produced
                                by the pipeline (contains ``expanded_text``
                                and ``predicted_nonbreaking_next_line``).
                                The ``.inferred.jsonl`` file is what
                                :func:`run_xml_from_inferred` needs for a
                                re-injection-only debug run.
    """
    boundary_detector, boundary_threshold, byt5_model, byt5_tokenizer = models

    def pipeline_fn(line_rows, pre_annotated=None):
        # Persist rows extracted from XML (before inference)
        if save_intermediate_path:
            extracted_path = save_intermediate_path + ".extracted.jsonl"
            with open(extracted_path, "w", encoding="utf-8") as fex:
                for row in line_rows:
                    fex.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[intermediate] extracted rows → {extracted_path}", file=sys.stderr)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as tmp:
            input_path = tmp.name
            for row in line_rows:
                tmp.write(json.dumps(row, ensure_ascii=False) + "\n")
        output_path = input_path.replace(".jsonl", "_out.jsonl")

        try:
            run_pipeline(
                input_path=input_path,
                output_path=output_path,
                boundary_detector=boundary_detector,
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

            # Persist full inference output rows
            if save_intermediate_path:
                inferred_path = save_intermediate_path + ".inferred.jsonl"
                with open(inferred_path, "w", encoding="utf-8") as fin:
                    for row in out_rows:
                        fin.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"[intermediate] inferred rows  → {inferred_path}", file=sys.stderr)

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

    report: list = []
    result = process_tei_xml(xml_text, pipeline_fn, reconstruction_report=report)
    _write_reconstruction_report(report, reconstruction_report_path)
    return result


def run_xml_from_inferred(xml_text: str, inferred_jsonl: str,
                          reconstruction_report_path: str | None = None) -> str:
    """Re-injection only: rebuild XML from a pre-saved inference JSONL.

    Reads the ``*.inferred.jsonl`` file written by :func:`run_xml` when
    ``save_intermediate_path`` is set, reconstructs *expanded_dict* and
    *boundary_dict*, and calls :func:`process_tei_xml` with a stub
    ``pipeline_fn`` that ignores the live rows and returns the cached
    results.  No models are loaded; no GPU is required.

    Args:
        xml_text:       Original TEI XML string (same file used for inference).
        inferred_jsonl: Path to the ``*.inferred.jsonl`` intermediate file.

    Returns:
        Re-injected TEI XML string.
    """
    import warnings

    out_rows: list[dict] = []
    with open(inferred_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out_rows.append(json.loads(line))

    if not out_rows:
        raise ValueError(f"No rows found in {inferred_jsonl}")

    expanded_dict = {r["id"]: r.get("expanded_text", r["source_sic"]) for r in out_rows}
    boundary_dict = {
        r["id"]: r["predicted_nonbreaking_next_line"]
        for r in out_rows if r.get("predicted_nonbreaking_next_line")
    }

    def pipeline_fn(line_rows, pre_annotated=None):
        # Sanity-check: IDs in the re-parsed XML must match the stored JSONL.
        # A mismatch means the XML and JSONL are out of sync.
        live_ids   = {r["id"] for r in line_rows}
        stored_ids = set(expanded_dict)
        if live_ids != stored_ids:
            missing = sorted(live_ids   - stored_ids)[:5]
            extra   = sorted(stored_ids - live_ids)[:5]
            warnings.warn(
                "[reinject] ID mismatch between XML rows and stored JSONL.\n"
                f"  IDs in XML but not in JSONL : {missing}{'…' if len(missing) == 5 else ''}\n"
                f"  IDs in JSONL but not in XML : {extra  }{'…' if len(extra)   == 5 else ''}",
                stacklevel=2,
            )
        return expanded_dict, boundary_dict

    report: list = []
    result = process_tei_xml(xml_text, pipeline_fn, reconstruction_report=report)
    _write_reconstruction_report(report, reconstruction_report_path)
    return result


def main():
    parser = argparse.ArgumentParser(description="Local SvSal PoCo inference handler")
    parser.add_argument("--mode", choices=["text", "jsonl", "xml"], required=True)
    parser.add_argument("--stdin", action="store_true", help="Read input from stdin")
    parser.add_argument("--input-file", help="Path to input file")
    parser.add_argument("--output-file", help="Path to output file (defaults to stdout)")
    parser.add_argument("--boundary-model-dir", help="Dir with best_model.pt and threshold.json (for canine)")
    parser.add_argument("--boundary-model-type", choices=["canine", "flair"], default="canine",
                        help="Type of boundary detector (default: canine)")
    parser.add_argument("--boundary-model-name", help="HF model name for flair detector")
    parser.add_argument("--byt5-model-dir", default=None, help="Dir/loadable HF path for ByT5 model")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lang-prefix", action="store_true", default=False)

    # for --mode text only
    parser.add_argument(
        "--text-output",
        choices=["text", "jsonl"],
        default="text",
        help="For plaintext input: output expanded plaintext (default) or output JSONL rows",
    )

    # for --mode xml: intermediate checkpointing
    parser.add_argument(
        "--save-intermediate",
        metavar="PATH_STEM",
        nargs="?",
        const="",
        default=None,
        help=(
            "xml mode only: persist intermediate JSONL files alongside the output. "
            "Writes <stem>.extracted.jsonl (rows stripped from XML before inference) "
            "and <stem>.inferred.jsonl (full pipeline output with expanded_text and "
            "predicted boundaries). When given without a value, the stem is derived "
            "from --output-file. The .inferred.jsonl is what --infer-jsonl expects."
        ),
    )
    parser.add_argument(
        "--infer-jsonl",
        metavar="PATH",
        default=None,
        help=(
            "xml mode only: skip model loading and inference entirely; re-inject "
            "results from this pre-saved *.inferred.jsonl file. Useful for "
            "iterating on tei_roundtrip.py without re-running the 3.5h GPU job. "
            "When set, --boundary-model-dir / --byt5-model-dir are not needed."
        ),
    )
    parser.add_argument(
        "--reconstruction-report",
        metavar="PATH",
        nargs="?",
        const="",
        default=None,
        help=(
            "xml mode only: write a JSONL audit of every line whose expansion was "
            "repaired from a leaked model line-break sentinel (U+21AC). Each row: "
            "{line_id, from_line_id, ratio, kind}. Given without a value, the path "
            "is derived from --output-file as <stem>.reconstruction.jsonl. Also "
            "emitted automatically when --save-intermediate is active."
        ),
    )

    args = parser.parse_args()

    # Re-injection mode bypasses all model loading
    reinject_only = args.mode == "xml" and bool(args.infer_jsonl)

    if not reinject_only:
        if not args.byt5_model_dir:
            parser.error("--byt5-model-dir is required (omit only when using --infer-jsonl)")
        # Validate boundary model arguments
        # Both --boundary-model-dir and --boundary-model-name work for both model types
        if args.boundary_model_type == "canine":
            if not args.boundary_model_dir and not args.boundary_model_name:
                parser.error("--boundary-model-dir or --boundary-model-name is required for canine detector")
        elif args.boundary_model_type == "flair":
            if not args.boundary_model_dir and not args.boundary_model_name:
                # Use default for flair if neither is specified
                args.boundary_model_name = "mschonhardt/latin-contextual-lb-detector"

        models = load_models(
            args.boundary_model_dir,
            args.byt5_model_dir,
            args.boundary_model_type,
            args.boundary_model_name,
        )
    else:
        models = None  # not needed for re-injection

    raw = read_input(args)

    if args.mode == "jsonl":
        # passthrough: run pipeline directly on provided JSONL
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
            f.write(raw)
            input_path = f.name
        output_path = input_path.replace(".jsonl", "_out.jsonl")
        try:
            boundary_detector, boundary_threshold, byt5_model, byt5_tokenizer = models
            run_pipeline(
                input_path=input_path,
                output_path=output_path,
                boundary_detector=boundary_detector,
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
        # Derive intermediate stem when --save-intermediate is present
        save_stem: str | None = None
        if args.save_intermediate is not None:
            if args.save_intermediate:              # explicit path stem provided
                save_stem = args.save_intermediate
            elif args.output_file:                  # derive from output path
                save_stem = str(Path(args.output_file).with_suffix(""))
            else:                                   # stdout mode: use cwd
                save_stem = "infer_intermediate"

        # Resolve the reconstruction-audit output path (xml mode only).
        recon_path: str | None = None
        if args.reconstruction_report is not None:
            if args.reconstruction_report:          # explicit path provided
                recon_path = args.reconstruction_report
            elif args.output_file:                  # derive from output path
                recon_path = str(Path(args.output_file).with_suffix("")) + ".reconstruction.jsonl"
            else:
                recon_path = "infer_intermediate.reconstruction.jsonl"
        elif save_stem:                             # parity with other intermediates
            recon_path = save_stem + ".reconstruction.jsonl"

        if reinject_only:
            out_xml = run_xml_from_inferred(
                raw, inferred_jsonl=args.infer_jsonl,
                reconstruction_report_path=recon_path,
            )
        else:
            out_xml = run_xml(raw, models=models, batch_size=args.batch_size,
                              save_intermediate_path=save_stem,
                              reconstruction_report_path=recon_path)
        write_output(out_xml, args.output_file)


if __name__ == "__main__":
    main()
