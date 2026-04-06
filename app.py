"""
app.py — SvSal PoCo Gradio Space
Hyphenation detection and abbreviation expansion for early modern
Spanish and Latin printed texts.
"""

import subprocess
import sys
subprocess.run([sys.executable, "-m", "pip", "install", "sentencepiece"], check=True)

import json
import re
import tempfile
import os
from pathlib import Path
from typing import Optional

import gradio as gr
import spaces
import torch
from huggingface_hub import hf_hub_download, login
from transformers import AutoTokenizer, T5ForConditionalGeneration, CanineTokenizer
from codecarbon import EmissionsTracker

# ---------------------------------------------------------------------------
# These imports resolve at runtime on the Space because the package is
# installed via requirements.txt. Editor warnings about unresolved imports
# in the space branch can be ignored.
# ---------------------------------------------------------------------------
from boundary_classifier.boundary_classifier import BoundaryClassifier
from data.data_utils import (
    ABBR_OPEN, ABBR_CLOSE, LINE_SEP,
    CorpusLexicon,
)
from infer import run_pipeline


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BOUNDARY_REPO = "mpilhlt/canine-salamanca-boundary-classifier"
BYT5_REPO     = "mpilhlt/byt5-salamanca-abbr"

# Displayed in the UI to show model status
_models_loaded = False
_load_error    = None


# ---------------------------------------------------------------------------
# Authenticate using the Space secret — allows access to private repos
# in organisations the token owner belongs to
# ---------------------------------------------------------------------------

hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)


# ---------------------------------------------------------------------------
# Model loading — happens once at Space startup, not per request
# ---------------------------------------------------------------------------

def load_models():
    """
    Load both models from the Hub at Space startup.
    Called once; results are module-level globals used by all handlers.
    """
    global canine_tokenizer, boundary_model, boundary_threshold
    global byt5_tokenizer, byt5_model
    global _models_loaded, _load_error

    try:
        # --- Boundary classifier ---
        canine_tokenizer = CanineTokenizer.from_pretrained("google/canine-s")
        boundary_model   = BoundaryClassifier(use_lexicon=False)
        weights_path     = hf_hub_download(BOUNDARY_REPO, "best_model.pt")
        boundary_model.load_state_dict(
            torch.load(weights_path, map_location="cpu")
        )
        boundary_model.eval()

        threshold_path   = hf_hub_download(BOUNDARY_REPO, "threshold.json")
        boundary_threshold = json.loads(
            Path(threshold_path).read_text()
        )["threshold"]

        # --- ByT5 ---
        # The tokenizer can be loaded directly from the "google/byt5-base"
        # checkpoint since it uses the same byte-level tokenization.
        # The model weights are loaded from our custom checkpoint, which
        # has the same architecture but different fine-tuned weights. We set
        # tie_word_embeddings=False to avoid an error about mismatched vocab sizes,
        # since the custom checkpoint doesn't include the tied embeddings that the
        # original ByT5 uses.
        # byt5_tokenizer = AutoTokenizer.from_pretrained(BYT5_REPO)
        byt5_tokenizer = AutoTokenizer.from_pretrained("google/byt5-base")
        try:
            byt5_model = T5ForConditionalGeneration.from_pretrained(
                BYT5_REPO, tie_word_embeddings=False
            )
            byt5_model.eval()
        except Exception as e:
            print(f"ByT5 model not available: {e}")
            byt5_model = None

        _models_loaded = True

    except Exception as e:
        _load_error = str(e)
        raise


load_models()


# ---------------------------------------------------------------------------
# Core processing helpers
# ---------------------------------------------------------------------------

def lines_to_jsonl_rows(text: str) -> list[dict]:
    """
    Convert plain text (one line per newline) into minimal row dicts
    compatible with the pipeline. Assigns synthetic ids and a single
    doc_id so document-level logic works correctly.
    """
    rows = []
    for i, line in enumerate(text.strip().splitlines()):
        line = line.strip()
        if not line:
            continue
        rows.append({
            "id":                    f"demo-00-0001-lb-{i:04d}",
            "doc_id":               "demo",
            "facs_id":              "demo-0001",
            "ancestor_id":          "demo-00-0001-pa-0001",
            "lang":                 ["la"],
            "source_sic":           line,
            "target_corr":          line,   # placeholder; overwritten by model
            "contains_abbr":        "false",
            "nonbreaking_next_line": "",
        })
    return rows


def classify_boundaries_only(text: str) -> str:
    """
    Run only the boundary classifier and return annotated text showing
    which line boundaries were detected as nonbreaking.
    """
    from boundary_classifier.boundary_classifier import predict_boundaries

    rows   = lines_to_jsonl_rows(text)
    result = predict_boundaries(
        lines=rows,
        model=boundary_model,
        tokenizer=canine_tokenizer,
        threshold=boundary_threshold,
        context_chars=40,
    )

    lines      = [r["source_sic"] for r in result]
    nonbreaking = {r["id"] for r in result if r["predicted_nonbreaking_next_line"]}
    id_list    = [r["id"] for r in result]

    output_lines = []
    for i, (line, row_id) in enumerate(zip(lines, id_list)):
        output_lines.append(line)
        if row_id in nonbreaking:
            # Mark the boundary visually — word continues on next line
            output_lines.append("    ↪ [continues →]")

    return "\n".join(output_lines)


def expand_text(text: str) -> tuple[str, str, str]:
    """
    Run the full pipeline (boundary detection + abbreviation expansion)
    on plain text input.

    Returns:
        expanded:   fully processed text
        boundaries: boundary detection annotation only
        diff:       side-by-side original vs expanded for changed lines
    """
    rows = lines_to_jsonl_rows(text)

    # Write to temp file for run_pipeline
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        input_path = f.name

    output_path = input_path.replace(".jsonl", "_out.jsonl")

    try:
        run_pipeline(
            input_path=input_path,
            output_path=output_path,
            boundary_model_dir=BOUNDARY_REPO,
            byt5_model_dir=BYT5_REPO,
            boundary_threshold=boundary_threshold,
            batch_size=16,
        )

        with open(output_path, encoding="utf-8") as f:
            out_rows = [json.loads(l) for l in f if l.strip()]

    finally:
        os.unlink(input_path)
        if os.path.exists(output_path):
            os.unlink(output_path)

    # --- Expanded text ---
    expanded = "\n".join(r.get("expanded_text", r["source_sic"]) for r in out_rows)

    # --- Boundary annotation ---
    boundaries = classify_boundaries_only(text)

    # --- Diff: lines that changed ---
    diff_lines = []
    for orig, out in zip(rows, out_rows):
        src = orig["source_sic"]
        exp = out.get("expanded_text", src)
        if src != exp:
            diff_lines.append(f"  IN:  {src}")
            diff_lines.append(f"  OUT: {exp}")
            diff_lines.append("")
    diff = "\n".join(diff_lines) if diff_lines else "(no changes)"

    return expanded, boundaries, diff


# ---------------------------------------------------------------------------
# ZeroGPU-decorated inference entry points
# One decorator per function — @spaces.GPU cannot wrap a shared helper
# ---------------------------------------------------------------------------

@spaces.GPU(duration=300)
def run_boundary_only(text: str) -> str:
    if not text.strip():
        return "Please enter some text."
    return classify_boundaries_only(text)


@spaces.GPU(duration=300)
def run_full_pipeline(text: str) -> tuple[str, str, str, str]:
    if byt5_model is None:
        return "ByT5 model not yet trained — abbreviation expansion unavailable.", "", "", ""
    if not text.strip():
        return "Please enter some text.", "", "", ""

    tracker = EmissionsTracker(log_level="error", save_to_file=False)
    tracker.start()
    expanded, boundaries, diff = expand_text(text)
    emissions = tracker.stop()

    carbon_info = (
        f"Estimated CO2 for this request: {emissions*1000:.4f} g CO2eq.\n"
        # f"Equivalent to {emissions/0.000404:.1f}m of driving a petrol car"
    )

    return expanded, boundaries, diff, carbon_info

# ---------------------------------------------------------------------------
# XML tab placeholder
# ---------------------------------------------------------------------------

def run_xml_pipeline(xml_file) -> Optional[str]:
    """
    Placeholder for TEI XML processing.
    Will accept a TEI XML file, strip to plaintext, run the pipeline,
    and reinsert expansions into the XML structure.
    Not yet implemented.
    """
    return (
        "XML processing is not yet available. "
        "Please use the plain text tab for now."
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

DESCRIPTION = """
## SvSal PoCo — Early Modern Text Tools

Processes early modern Spanish and Latin printed texts from the
[School of Salamanca](https://salamanca.school/) corpus.

**Two tasks:**
- **Hyphenation detection**: identifies line breaks where a word continues
  on the next line without a hyphen marker
- **Abbreviation expansion**: expands brevigraphs, macrons, and other
  abbreviation conventions to their full forms

Input one line of text per line. The text should already be transcribed
(e.g. from HTR or manual transcription); the tools handle post-processing.
"""

EXAMPLE_TEXT = """rum prudentium consultum. &c̃. legi tan
tùm adscribitur humanę. Sicuti & illa Arist.
in Rhetoricis ad Alexandrum. c. de gen. de
lib. Lex est communis ciuitatis cōsensus qui
scriptis præceperit quomodò vnumquodq́;
agendum sit."""

with gr.Blocks(
    title="SvSal PoCo — Early Modern Text Tools",
    theme=gr.themes.Base(
        primary_hue="stone",
        secondary_hue="amber",
        font=[gr.themes.GoogleFont("IM Fell English"), "Georgia", "serif"],
        font_mono=[gr.themes.GoogleFont("Inconsolata"), "monospace"],
    ),
) as demo:

    gr.Markdown(DESCRIPTION)

    if not _models_loaded:
        gr.Markdown(f"⚠️ **Model loading failed:** {_load_error}")

    with gr.Tabs():

        # ---------------------------------------------------------------
        # Tab 1: Full pipeline (boundary detection + expansion combined)
        # ---------------------------------------------------------------
        with gr.Tab("Full pipeline"):
            gr.Markdown(
                "Runs hyphenation detection and abbreviation expansion together. "
                "The **Expanded text** tab shows the final output; the other tabs "
                "show intermediate results."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    full_input = gr.Textbox(
                        lines=12,
                        label="Input text (one line per line)",
                        placeholder="Paste transcribed early modern text here...",
                        value=EXAMPLE_TEXT,
                    )
                    full_btn = gr.Button(
                        "Process", variant="primary", size="lg"
                    )

                with gr.Column(scale=1):
                    with gr.Tabs():
                        with gr.Tab("Expanded text"):
                            full_output_expanded = gr.Textbox(
                                lines=12,
                                label="Expanded output",
                            )
                        with gr.Tab("Boundary detection"):
                            full_output_boundaries = gr.Textbox(
                                lines=12,
                                label="Detected nonbreaking boundaries (↪ marks continuation)",
                            )
                        with gr.Tab("Changes only"):
                            full_output_diff = gr.Textbox(
                                lines=12,
                                label="Lines where expansions were made",
                            )
                        with gr.Tab("CO2eq estimate"):
                            full_output_carbon = gr.Textbox(
                                lines=2,
                                label="Environmental cost",
                                interactive=False,
                            )

            full_btn.click(
                fn=run_full_pipeline,
                inputs=full_input,
                outputs=[
                    full_output_expanded,
                    full_output_boundaries,
                    full_output_diff,
                    full_output_carbon,
                ],
            )

        # ---------------------------------------------------------------
        # Tab 2: Boundary detection only
        # ---------------------------------------------------------------
        with gr.Tab("Boundary detection only"):
            gr.Markdown(
                "Runs only the Canine boundary classifier. "
                "Lines marked with ↪ are identified as nonbreaking — "
                "the word at the end of that line continues on the next line."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    boundary_input = gr.Textbox(
                        lines=12,
                        label="Input text",
                        placeholder="Paste transcribed text here...",
                        value=EXAMPLE_TEXT,
                    )
                    boundary_btn = gr.Button(
                        "Detect boundaries", variant="primary"
                    )

                with gr.Column(scale=1):
                    boundary_output = gr.Textbox(
                        lines=12,
                        label="Annotated output",
                    )

            boundary_btn.click(
                fn=run_boundary_only,
                inputs=boundary_input,
                outputs=boundary_output,
            )

        # ---------------------------------------------------------------
        # Tab 3: XML processing (placeholder)
        # ---------------------------------------------------------------
        with gr.Tab("TEI XML processing"):
            gr.Markdown(
                "### Coming soon\n\n"
                "This tab will accept a TEI XML file, run the full pipeline, "
                "and return a corrected TEI XML file with abbreviations expanded "
                "and `<expan>` elements properly filled in.\n\n"
                "The plain text tabs above are available in the meantime."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    xml_input = gr.File(
                        label="Upload TEI XML file",
                        file_types=[".xml"],
                        interactive=False,   # disabled until implemented
                    )
                    xml_btn = gr.Button(
                        "Process XML (not yet available)",
                        variant="secondary",
                        interactive=False,
                    )

                with gr.Column(scale=1):
                    xml_output = gr.File(
                        label="Processed TEI XML",
                        interactive=False,
                    )
                    xml_status = gr.Textbox(
                        label="Status",
                        value="XML processing is not yet implemented.",
                        interactive=False,
                    )

            xml_btn.click(
                fn=run_xml_pipeline,
                inputs=xml_input,
                outputs=xml_status,
            )

    # -------------------------------------------------------------------
    # Footer
    # -------------------------------------------------------------------
    gr.Markdown("""
---
**Models:**
[Boundary classifier](https://huggingface.co/mpilhlt/canine-salamanca-boundary-classifier) •
[ByT5 expansion](https://huggingface.co/mpilhlt/byt5-salamanca-abbr)

**Code:** [digicademy/svsal-poco](https://github.com/digicademy/svsal-poco)

**Corpus:** [School of Salamanca Digital Collection](https://salamanca.school/)
""")


demo.launch(
    ssr_mode=False,
    auth=None,        # explicitly disable auth
)
