"""
app.py — SvSal PoCo Gradio Space
Hyphenation detection and abbreviation expansion for early modern
Spanish and Latin printed texts.
"""

import json
from datetime import datetime, timezone
import re
import tempfile
import os
import threading
from pathlib import Path

import gradio as gr
import spaces
import torch
from huggingface_hub import HfApi, hf_hub_download, login, RepoFolder, snapshot_download
from transformers import AutoTokenizer, T5ForConditionalGeneration, CanineTokenizer
from codecarbon import EmissionsTracker

# ---------------------------------------------------------------------------
# These imports resolve at runtime on the Space because the package is
# installed via requirements.txt. Editor warnings about unresolved imports
# in the space branch can be ignored.
# ---------------------------------------------------------------------------
from boundary_classifier.boundary_classifier import BoundaryClassifier, predict_boundaries
from data.data_utils import (
    ABBR_OPEN, ABBR_CLOSE, LINE_SEP,
    CorpusLexicon,
)
from infer import run_pipeline


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BOUNDARY_REPO       = "mpilhlt/canine-salamanca-boundary-classifier"
BYT5_REPO           = "mpilhlt/byt5-salamanca-abbr"
EMISSIONS_REPO      = os.environ.get("SPACE_ID", "awagner-mainz/svsal-poco")
EMISSIONS_REPO_TYPE = "space"
EMISSIONS_FILE      = "inference_emissions.json"
_UPLOAD_EVERY_N     = 10

_models_loaded = False
_load_error    = None
_emissions_lock   = threading.Lock()           # thread-safe updates
_session_emissions = {"kg": 0.0, "n": 0}
_api              = HfApi()

boundary_model = None
boundary_tokenizer = None
boundary_threshold = None
byt5_model = None
byt5_tokenizer = None


# ---------------------------------------------------------------------------
# Authenticate using the Space secret — allows access to private repos
# in organisations the token owner belongs to
# ---------------------------------------------------------------------------

hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)


# ---------------------------------------------------------------------------
# Emissions tracking — backed by a JSON file in a Hub dataset repo
# ---------------------------------------------------------------------------

def make_emissions_tracker() -> tuple:
    """
    Create an EmissionsTracker appropriate for the current hardware.

    On ZeroGPU, calls to nvml for enumerating GPUs/CPUs do not work
    and we try workarounds.
    On standard GPU hardware, full GPU tracking is used.
    Returns (tracker, mode) where mode is 'full', 'cpu_only', or 'unavailable'.
    """

    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    is_restricted = cuda_devices and not cuda_devices.replace(",", "").isdigit()

    if is_restricted:
        # ZeroGPU environment — NVML access is restricted
        # Try CPU-only tracking by hiding GPU entirely
        saved = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        try:
            tracker = EmissionsTracker(
                log_level="error",
                save_to_file=False,
                allow_multiple_runs=True,
            )
            print("Carbon tracking: CPU-only mode (ZeroGPU)")
            return tracker, "cpu_only"
        except Exception as e:
            print(f"Carbon tracking unavailable on this hardware: {e}")
            return None, "unavailable"
        finally:
            if saved is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = saved
    else:
        try:
            tracker = EmissionsTracker(
                log_level="error",
                save_to_file=False,
                allow_multiple_runs=True,
            )
            return tracker, "full"
        except Exception as e:
            print(f"Carbon tracking unavailable: {e}")
            return None, "unavailable"


def load_accumulated_emissions() -> dict:
    """Download and parse the persistent emissions file from the Hub."""
    try:
        path = hf_hub_download(
            repo_id=EMISSIONS_REPO,
            filename=EMISSIONS_FILE,
            repo_type=EMISSIONS_REPO_TYPE,
            force_download=True,   # always get fresh copy
        )
        return json.loads(Path(path).read_text())
    except Exception:
        # File doesn't exist yet — first run
        return {
            "kg_co2eq_total": 0.0,
            "n_requests":     0,
            "first_request":  None,
            "last_request":   None,
        }


def save_accumulated_emissions(accumulated: dict):
    """Upload updated emissions file to Hub dataset repo."""
    try:
        content = json.dumps(accumulated, indent=2, ensure_ascii=False)
        _api.upload_file(
            path_or_fileobj=content.encode(),
            path_in_repo=EMISSIONS_FILE,
            repo_id=EMISSIONS_REPO,
            repo_type=EMISSIONS_REPO_TYPE,
            commit_message="Update inference emissions",
        )
    except Exception as e:
        print(f"Warning: could not persist emissions: {e}")


def record_request_emissions(kg_co2eq: float) -> dict:
    """
    Thread-safe update of the persistent emissions accumulator.
    Returns the updated accumulated dict for display.
    """
    with _emissions_lock:
        _session_emissions["kg"] += kg_co2eq
        _session_emissions["n"]  += 1

        accumulated = load_accumulated_emissions()
        accumulated["kg_co2eq_total"] += kg_co2eq
        accumulated["n_requests"]     += 1
        accumulated["last_request"]    = datetime.now(timezone.utc).isoformat()
        if accumulated["first_request"] is None:
            accumulated["first_request"] = accumulated["last_request"]

        # Only persist every N requests to reduce Hub commit noise
        if _session_emissions["n"] % _UPLOAD_EVERY_N == 0:
            save_accumulated_emissions(accumulated)

        return accumulated


def format_carbon_info(
    request_kg: float,
    accumulated: dict,
    tracking_mode: str = "full",
) -> str:
    kg_total = accumulated["kg_co2eq_total"]
    n        = accumulated["n_requests"]
    km_total = kg_total / 0.000404

    if tracking_mode == "unavailable":
        lines = [
            f"This request: CO2 not measurable on this hardware",
            f"Cumulative ({n} requests): {kg_total*1000:.2f} g CO2eq tracked so far",
        ]
    else:
        g_this  = request_kg * 1000
        km_this = request_kg / 0.000404
        note    = " (CPU estimate only)" if tracking_mode == "cpu_only" else ""
        lines   = [
            f"This request: {g_this:.4f} g CO2eq (≈ {km_this:.1f} m driving){note}",
            f"Cumulative ({n} requests): {kg_total*1000:.2f} g CO2eq (≈ {km_total:.2f} m driving)",
        ]

    if accumulated["first_request"]:
        lines.append(f"Tracking since {accumulated['first_request'][:10]}")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Model loading — happens once at Space startup, not per request
# ---------------------------------------------------------------------------

def load_models():
    """
    Load both models from the Hub at Space startup.
    Called once; results are module-level globals used by all handlers.
    """
    global boundary_model, boundary_tokenizer, boundary_threshold
    global byt5_model, byt5_tokenizer
    global _models_loaded, _load_error

    try:

        # --- Boundary classifier ---

        boundary_tokenizer = CanineTokenizer.from_pretrained("google/canine-s")
        boundary_model   = BoundaryClassifier(use_lexicon=False)
        weights_path     = hf_hub_download(BOUNDARY_REPO, "best_model.pt")
        boundary_model.load_state_dict(
            torch.load(weights_path, map_location="cpu", weights_only=True)
        )
        boundary_model.eval()

        threshold_path   = hf_hub_download(BOUNDARY_REPO, "threshold.json")
        boundary_threshold = json.loads(
            Path(threshold_path).read_text()
        )["threshold"]

        # --- ByT5 ---

        # The tokenizer can be loaded directly from the "google/byt5-base"
        # checkpoint since it uses the same byte-level tokenization.
        # We set tie_word_embeddings=False to avoid an error about mismatched
        # vocab sizes, since the custom checkpoint doesn't include the tied
        # embeddings that the original ByT5 uses.
        byt5_tokenizer = AutoTokenizer.from_pretrained("google/byt5-base")

        try:
            # Try root model first
            # byt5_model = T5ForConditionalGeneration.from_pretrained(
            #     BYT5_REPO, tie_word_embeddings=False,
            # )
            # print(f"ByT5 loaded from {BYT5_REPO} (root)")
            # Temporarily switch to "last-checkpoint" subfolder to work around
            # an issue with the root model loading stemming from aborted training runs
            # leaving inconsistent files in the root.
            byt5_model = T5ForConditionalGeneration.from_pretrained(
                BYT5_REPO, subfolder="last-checkpoint", tie_word_embeddings=False,
            )
            print(f"ByT5 loaded from {BYT5_REPO} (last checkpoint)")
 
        except Exception:
            # Fall back to latest checkpoint
            try:
                entries = list(_api.list_repo_tree(
                    BYT5_REPO, path_in_repo="checkpoints", repo_type="model",
                ))
                checkpoint_dirs = sorted(
                    [e.path for e in entries
                    if isinstance(e, RepoFolder)
                    and e.path.startswith("checkpoints/checkpoint-")],
                    key=lambda x: int(x.rsplit("-", 1)[-1]),
                )
                if not checkpoint_dirs:
                    raise FileNotFoundError("No ByT5 model checkpoints found")
                latest = checkpoint_dirs[-1]
                print(f"Loading ByT5 from checkpoint: {latest}")
                byt5_model = T5ForConditionalGeneration.from_pretrained(
                    BYT5_REPO, subfolder=latest, tie_word_embeddings=False,
                )
            except Exception as e2:
                print(f"ByT5 model not available: {e2}")
                byt5_model = None

        if byt5_model is not None:
            byt5_model.eval()

        if boundary_model and byt5_model:
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
    rows   = lines_to_jsonl_rows(text)
    result = predict_boundaries(
        lines=rows,
        model=boundary_model,
        tokenizer=boundary_tokenizer,
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
            boundary_model=boundary_model,
            boundary_tokenizer=boundary_tokenizer,
            boundary_threshold=boundary_threshold,
            byt5_model=byt5_model,
            byt5_tokenizer=byt5_tokenizer,
            batch_size=16,
            lang_prefix=True,
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
def run_boundary_only(text: str) -> tuple[str, str]:
    if boundary_model is None:
        return "Boundary model not available — cannot run boundary detection.", ""
    if not text.strip():
        return "Please enter some text.", ""

    tracker, mode = make_emissions_tracker()
    if tracker:
        tracker.start()

    result     = classify_boundaries_only(text)
    request_kg = tracker.stop() if tracker else 0.0
    if request_kg is None:
        request_kg = 0.0

    accumulated = record_request_emissions(request_kg)
    carbon_info = format_carbon_info(request_kg, accumulated, tracking_mode=mode)

    return result, carbon_info


@spaces.GPU(duration=300)
def run_full_pipeline(text: str) -> tuple[str, str, str, str]:
    if byt5_model is None:
        return "ByT5 model not yet trained — abbreviation expansion unavailable.", "", "", ""
    if not text.strip():
        return "Please enter some text.", "", "", ""

    tracker, mode = make_emissions_tracker()
    if tracker:
        tracker.start()

    expanded, boundaries, diff = expand_text(text)
    request_kg = tracker.stop() if tracker else 0.0
    if request_kg is None:
        request_kg = 0.0

    accumulated = record_request_emissions(request_kg)
    carbon_info = format_carbon_info(request_kg, accumulated, tracking_mode=mode)

    return expanded, boundaries, diff, carbon_info

# ---------------------------------------------------------------------------
# XML tab placeholder
# ---------------------------------------------------------------------------

def run_xml_pipeline(xml_file) -> str | None:
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
                        with gr.Tab("Environmental cost"):
                            full_output_carbon = gr.Textbox(
                                lines=4,
                                label="CO2 emissions",
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
                    boundary_carbon = gr.Textbox(
                        lines=4,
                        label="CO2 emissions",
                        interactive=False,
                    )

            boundary_btn.click(
                fn=run_boundary_only,
                inputs=boundary_input,
                outputs=[boundary_output, boundary_carbon],
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
