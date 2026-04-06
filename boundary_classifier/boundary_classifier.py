# boundary_classifier.py
#
# Canine-based nonbreaking line boundary classifier.
#
# Task: given the end of line N and the start of line N+1, predict whether
# the two lines share a word (nonbreaking=1) or represent a genuine line
# break (nonbreaking=0).
#
# This classifier runs as the first stage in the inference pipeline,
# identifying which line pairs should be concatenated before being
# passed to the ByT5 abbreviation expansion model.

import os
import json
import random
import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import CanineTokenizer, CanineModel
from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report,
    precision_recall_curve,
)
from huggingface_hub import login, HfApi, hf_hub_download

from codecarbon import EmissionsTracker
from transformers import TrainerCallback

from data.data_utils import (
    BoundaryExample,
    CorpusLexicon,
    build_boundary_examples,
    document_split_boundary,
    load_and_sort_lines,
)


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class BoundaryDataset(Dataset):
    """
    Tokenizes BoundaryExample instances for Canine.

    Input format: [line_end] ↵ [line_start]

    The ↵ character marks the boundary position explicitly in the
    character stream. Canine operates on Unicode codepoints so sees
    this naturally without any special token handling.
    """
    def __init__(
        self,
        examples:   list[BoundaryExample],
        tokenizer:  CanineTokenizer,
        max_length: int = 128,
        lexicon:    Optional[CorpusLexicon] = None,
    ):
        self.examples   = examples
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.lexicon    = lexicon

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex   = self.examples[idx]
        text = ex.line_end + "↵" + ex.line_start

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        item = {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label":          torch.tensor(ex.label, dtype=torch.float),
        }

        if self.lexicon is not None:
            item["lexicon_hit"] = torch.tensor(
                float(self.lexicon.concatenation_is_known(
                    ex.line_end, ex.line_start
                )),
                dtype=torch.float,
            )

        return item


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BoundaryClassifier(nn.Module):
    """
    Canine-s encoder with a binary classification head.

    Uses the [CLS] token representation (position 0) as the sequence
    summary, consistent with Canine's design for classification tasks.

    If a lexicon is provided, its single-bit feature is concatenated
    to the CLS representation before classification.
    """
    def __init__(
        self,
        use_lexicon: bool  = False,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.canine      = CanineModel.from_pretrained("google/canine-s")
        hidden           = self.canine.config.hidden_size   # 768
        classifier_input = hidden + 1 if use_lexicon else hidden
        self.use_lexicon = use_lexicon
        self.dropout     = nn.Dropout(dropout)
        self.classifier  = nn.Linear(classifier_input, 1)

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        label:          Optional[torch.Tensor] = None,
        lexicon_hit:    Optional[torch.Tensor] = None,
        pos_weight:     Optional[torch.Tensor] = None,
    ) -> dict:
        outputs  = self.canine(
            input_ids=input_ids, attention_mask=attention_mask
        )
        cls_repr = outputs.last_hidden_state[:, 0, :]
        cls_repr = self.dropout(cls_repr)

        if self.use_lexicon and lexicon_hit is not None:
            cls_repr = torch.cat([cls_repr, lexicon_hit.unsqueeze(1)], dim=1)

        logits = self.classifier(cls_repr).squeeze(1)

        loss = None
        if label is not None:
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss    = loss_fn(logits, label)

        return {"loss": loss, "logits": logits}


# ---------------------------------------------------------------------------
# Class imbalance
# ---------------------------------------------------------------------------

def compute_pos_weight(
    examples: list[BoundaryExample],
    cap:      float = 8.0,
) -> torch.Tensor:
    """
    Compute pos_weight for BCEWithLogitsLoss from training set distribution.

    pos_weight = n_negatives / n_positives, capped to avoid pushing
    recall too aggressively given the high-precision preference.
    With 5-10x more negatives, uncapped this would be 5-10; cap at 8.
    """
    n_pos  = sum(1 for e in examples if e.label == 1)
    n_neg  = sum(1 for e in examples if e.label == 0)
    weight = min(n_neg / n_pos, cap) if n_pos > 0 else 1.0
    return torch.tensor([weight], dtype=torch.float)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_classifier(
    model:     BoundaryClassifier,
    loader:    DataLoader,
    device:    torch.device,
    threshold: float = 0.6,
) -> dict:
    model.eval()
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for batch in loader:
            out = model(
                input_ids      = batch["input_ids"].to(device),
                attention_mask = batch["attention_mask"].to(device),
                lexicon_hit    = batch.get("lexicon_hit"),
            )
            all_logits.extend(out["logits"].cpu().tolist())
            all_labels.extend(batch["label"].tolist())

    probs  = torch.sigmoid(torch.tensor(all_logits)).numpy()
    preds  = (probs >= threshold).astype(int)
    labels = np.array(all_labels)

    p, r, f, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    return {"precision": p, "recall": r, "f1": f,
            "probs": probs, "preds": preds, "labels": labels}


def evaluate_by_page_break(
    examples: list[BoundaryExample],
    preds:    np.ndarray,
    labels:   np.ndarray,
    probs:    np.ndarray,
) -> dict:
    """
    Separate precision/recall/F1 for within-page vs. cross-page boundaries.
    A large gap signals that cross-page cases may need a higher threshold.
    """
    results = {}
    for crosses in [False, True]:
        mask = np.array([e.crosses_page_break == crosses for e in examples])
        if mask.sum() == 0:
            continue
        p, r, f, _ = precision_recall_fscore_support(
            labels[mask], preds[mask], average="binary", zero_division=0
        )
        key = "cross_page" if crosses else "within_page"
        results[key] = {
            "precision":  p,
            "recall":     r,
            "f1":         f,
            "n":          int(mask.sum()),
            "n_positive": int(labels[mask].sum()),
        }
    return results


def select_threshold(
    pr_curve:      dict,
    min_precision: float = 0.90,
) -> float:
    """
    Select the lowest threshold achieving at least min_precision.
    0.90 is recommended given the high-precision preference: spurious
    concatenations in the abbreviation pipeline are more harmful than
    missed boundaries.
    """
    for precision, recall, threshold in zip(
        pr_curve["precisions"],
        pr_curve["recalls"],
        pr_curve["thresholds"],
    ):
        if precision >= min_precision:
            print(f"Threshold {threshold:.3f} → "
                  f"precision {precision:.3f}, recall {recall:.3f}")
            return float(threshold)
    print("Warning: min_precision not achievable; returning 0.9")
    return 0.9


def full_evaluation(
    model:         BoundaryClassifier,
    test_examples: list[BoundaryExample],
    tokenizer:     CanineTokenizer,
    lexicon:       Optional[CorpusLexicon] = None,
    threshold:     float = 0.6,
) -> dict:
    """
    Full test-set evaluation including per-doc, per-lang,
    cross-page breakdowns, PR curve, and error samples.
    """
    from collections import defaultdict
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = BoundaryDataset(test_examples, tokenizer, lexicon=lexicon)
    loader  = DataLoader(dataset, batch_size=32)
    model   = model.to(device)

    metrics = evaluate_classifier(model, loader, device, threshold)
    probs, preds, labels = metrics["probs"], metrics["preds"], metrics["labels"]

    pr_p, pr_r, pr_t = precision_recall_curve(labels, probs)

    # Per-document
    doc_results: dict = defaultdict(lambda: {"labels": [], "preds": []})
    for ex, pred, label in zip(test_examples, preds, labels):
        doc_results[ex.doc_id]["labels"].append(label)
        doc_results[ex.doc_id]["preds"].append(pred)

    per_doc = {}
    for doc_id, dr in doc_results.items():
        p, r, f, _ = precision_recall_fscore_support(
            dr["labels"], dr["preds"], average="binary", zero_division=0
        )
        per_doc[doc_id] = {
            "precision": p, "recall": r, "f1": f,
            "n": len(dr["labels"]),
            "n_positive": sum(dr["labels"]),
        }

    # Per-language
    lang_results: dict = defaultdict(lambda: {"labels": [], "preds": []})
    for ex, pred, label in zip(test_examples, preds, labels):
        for lang in ex.lang:
            lang_results[lang]["labels"].append(label)
            lang_results[lang]["preds"].append(pred)

    per_lang = {}
    for lang, lr in lang_results.items():
        p, r, f, _ = precision_recall_fscore_support(
            lr["labels"], lr["preds"], average="binary", zero_division=0
        )
        per_lang[lang] = {"precision": p, "recall": r, "f1": f,
                          "n": len(lr["labels"])}

    print(classification_report(labels, preds,
                                target_names=["breaking", "nonbreaking"]))

    return {
        "precision":       metrics["precision"],
        "recall":          metrics["recall"],
        "f1":              metrics["f1"],
        "threshold":       threshold,
        "pr_curve":        {"precisions": pr_p.tolist(),
                            "recalls":    pr_r.tolist(),
                            "thresholds": pr_t.tolist()},
        "per_doc":         per_doc,
        "per_lang":        per_lang,
        "by_page_break":   evaluate_by_page_break(
                               test_examples, preds, labels, probs),
        "false_positives": [
            {"line_end": ex.line_end, "line_start": ex.line_start,
             "prob": float(prob), "doc_id": ex.doc_id}
            for ex, pred, label, prob
            in zip(test_examples, preds, labels, probs)
            if pred == 1 and label == 0
        ][:20],
        "false_negatives": [
            {"line_end": ex.line_end, "line_start": ex.line_start,
             "prob": float(prob), "doc_id": ex.doc_id}
            for ex, pred, label, prob
            in zip(test_examples, preds, labels, probs)
            if pred == 0 and label == 1
        ][:20],
        "n_test": len(test_examples),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    train_examples: list[BoundaryExample],
    val_examples:   list[BoundaryExample],
    tokenizer:      CanineTokenizer,
    lexicon:        Optional[CorpusLexicon] = None,
    output_dir:     str   = "./boundary-classifier",
    epochs:         int   = 5,
    batch_size:     int   = 32,
    learning_rate:  float = 2e-5,
    threshold:      float = 0.6,
) -> BoundaryClassifier:

    # Guard against empty val set (happens with small test data slices)
    if not val_examples:
        print("Warning: empty val set, using train set for validation")
        val_examples = train_examples

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = BoundaryDataset(train_examples, tokenizer, lexicon=lexicon)
    val_ds   = BoundaryDataset(val_examples,   tokenizer, lexicon=lexicon)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    model      = BoundaryClassifier(use_lexicon=(lexicon is not None)).to(device)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    pos_weight = compute_pos_weight(train_examples).to(device)

    # --- Start carbon tracking ---
    from codecarbon import EmissionsTracker
    tracker = EmissionsTracker(
        project_name="canine-salamanca-boundary-classifier",
        output_dir=output_dir,
        output_file="emissions.csv",
        log_level="warning",
    )
    tracker.start()

    best_precision = 0.0
    best_state     = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(
                input_ids      = batch["input_ids"].to(device),
                attention_mask = batch["attention_mask"].to(device),
                label          = batch["label"].to(device),
                lexicon_hit    = batch.get("lexicon_hit"),
                pos_weight     = pos_weight,
            )
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += out["loss"].item()

        val_metrics = evaluate_classifier(model, val_loader, device, threshold)
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"loss: {total_loss/len(train_loader):.4f} | "
            f"precision: {val_metrics['precision']:.3f} | "
            f"recall: {val_metrics['recall']:.3f} | "
            f"F1: {val_metrics['f1']:.3f}"
        )

        if val_metrics["precision"] >= best_precision:
            best_precision = val_metrics["precision"]
            best_state     = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}

    # --- Stop tracking and save ---
    emissions = tracker.stop()
    print(f"Training CO2: {emissions:.4f} kg CO2eq")

    emissions_path = Path(output_dir) / "emissions.json"
    emissions_path.write_text(json.dumps({
        "kg_co2eq":   emissions,
        "model":      "google/canine-s",
        "epochs":     epochs,
        "hardware":   "t4-small",
    }, indent=2, ensure_ascii=False))

    model.load_state_dict(best_state)
    Path(output_dir).mkdir(exist_ok=True)
    torch.save(model.state_dict(), f"{output_dir}/best_model.pt")
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_boundaries(
    lines:         list[dict],
    model:         BoundaryClassifier,
    tokenizer:     CanineTokenizer,
    lexicon:       Optional[CorpusLexicon] = None,
    threshold:     float = 0.6,
    context_chars: int   = 40,
) -> list[dict]:
    """
    Run boundary classifier over unseen lines. Returns lines annotated
    with 'predicted_nonbreaking_next_line', ready for the concatenation
    step in the ByT5 abbreviation expansion pipeline.
    """
    from collections import defaultdict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    model.eval()

    by_doc: dict = defaultdict(list)
    for row in lines:
        by_doc[row["doc_id"]].append(row)

    predictions: dict = {}

    for doc_lines in by_doc.values():
        for i in range(len(doc_lines) - 1):
            row      = doc_lines[i]
            next_row = doc_lines[i + 1]
            text     = (row["source_sic"][-context_chars:]
                        + "↵"
                        + next_row["source_sic"][:context_chars])

            enc = tokenizer(
                text, max_length=128, truncation=True,
                padding="max_length", return_tensors="pt",
            )

            lexicon_hit = None
            if lexicon is not None:
                lexicon_hit = torch.tensor([[
                    float(lexicon.concatenation_is_known(
                        row["source_sic"], next_row["source_sic"]
                    ))
                ]])

            with torch.no_grad():
                out = model(
                    input_ids      = enc["input_ids"].to(device),
                    attention_mask = enc["attention_mask"].to(device),
                    lexicon_hit    = lexicon_hit,
                )
            prob = torch.sigmoid(out["logits"]).item()
            if prob >= threshold:
                predictions[row["id"]] = next_row["id"]

    result = []
    for row in lines:
        row = dict(row)
        row["predicted_nonbreaking_next_line"] = predictions.get(row["id"], "")
        result.append(row)
    return result


# ---------------------------------------------------------------------------
# Entry point for HuggingFace Jobs
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_repo",  default=None, help="HF dataset repo id (for Hub/Jobs mode)")
    p.add_argument("--dataset_local", default=None, help="Local JSONL path (for local testing mode)")
    p.add_argument("--output_repo",   default=None, help="HF model repo (omit for local testing)")
    p.add_argument("--output_dir",    default="./boundary-classifier-output", help="Local output directory (for local testing mode)")
    p.add_argument("--epochs",        type=int,   default=5)
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--threshold",     type=float, default=0.6)
    p.add_argument("--min_precision", type=float, default=0.90)
    p.add_argument("--use_lexicon",   action="store_true")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--hf_token",      default=os.environ.get("HF_TOKEN"))
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    # --- Resolve data path ---
    if args.dataset_local:
        data_path = args.dataset_local
        print(f"Local mode: reading from {data_path}")
    elif args.dataset_repo:
        login(token=args.hf_token)
        data_path = hf_hub_download(
            repo_id=args.dataset_repo,
            filename="data.jsonl",
            repo_type="dataset",
        )
    else:
        raise ValueError("Provide either --dataset_local or --dataset_repo")

    use_hub = args.output_repo is not None
    if use_hub:
        login(token=args.hf_token)
        api = HfApi()

    # output_dir is where best_model.pt, threshold.json, and
    # boundary_eval.json are all written locally before optionally
    # being uploaded to the Hub
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Data ---
    lines    = load_and_sort_lines(data_path)
    examples = build_boundary_examples(lines)

    lexicon = None
    if args.use_lexicon:
        lexicon = CorpusLexicon()
        lexicon.build_from_jsonl(data_path)

    train_ex, val_ex, test_ex = document_split_boundary(examples, seed=args.seed)
    print(f"Train: {len(train_ex)} | Val: {len(val_ex)} | Test: {len(test_ex)}")
    print(f"Train positives: {sum(e.label for e in train_ex)}")

    # Guard for small test slices where splits may be empty
    if not val_ex:
        print("Warning: empty val set, using train set for validation")
        val_ex = train_ex
    if not test_ex:
        print("Warning: empty test set, using train set for evaluation")
        test_ex = train_ex

    tokenizer = CanineTokenizer.from_pretrained("google/canine-s")

    # --- Train ---
    # pass output_dir so best_model.pt lands in the right place
    model = train(
        train_ex, val_ex, tokenizer,
        lexicon=lexicon,
        output_dir=str(output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        threshold=args.threshold,
    )

    # --- Evaluate on test set ---
    results = full_evaluation(
        model, test_ex, tokenizer, lexicon=lexicon, threshold=args.threshold
    )

    # --- Threshold selection from val PR curve ---
    val_ds     = BoundaryDataset(val_ex, tokenizer, lexicon=lexicon)
    val_loader = DataLoader(val_ds, batch_size=32)
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_m      = evaluate_classifier(model, val_loader, device, threshold=0.5)
    pr_p, pr_r, pr_t = precision_recall_curve(val_m["labels"], val_m["probs"])
    best_threshold = select_threshold(
        {"precisions": pr_p, "recalls": pr_r, "thresholds": pr_t},
        min_precision=args.min_precision,
    )
    print(f"Selected threshold: {best_threshold}")

    # --- Write all outputs to output_dir ---
    eval_path      = output_dir / "boundary_eval.json"
    threshold_path = output_dir / "threshold.json"

    eval_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=float)
    )
    threshold_path.write_text(
        json.dumps({"threshold": best_threshold})
    )
    # best_model.pt was already written by train() into output_dir

    print(f"Wrote boundary_eval.json, threshold.json, best_model.pt to {output_dir}/")

    # --- Optionally upload to Hub ---
    if use_hub:
        api.upload_folder(
            folder_path=str(output_dir),
            repo_id=args.output_repo,
            repo_type="model",
        )
        print(f"Uploaded to {args.output_repo}")


if __name__ == "__main__":
    main()
