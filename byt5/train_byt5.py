# train_byt5.py
#
# ByT5-base abbreviation expansion — full training script for HuggingFace Jobs.
#
# Workflow:
#   1. Push your JSONL dataset to the Hub as a private dataset repository
#   2. Launch a job pointing at this script (see job_config.yaml)
#   3. The job trains, saves checkpoints to Hub after each epoch,
#      and uploads a test breakdown JSON on completion
#   4. Pull the best checkpoint for inference
#
# Requirements: see requirements.txt

# Make the project root importable regardless of where Python is invoked from
import sys
from pathlib import Path
_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "data"))
sys.path.insert(0, str(_root / "evaluation"))

import os
import json
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import evaluate as hf_evaluate

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)
from huggingface_hub import login, HfApi, hf_hub_download

from data.data_utils import load_and_sort_lines, build_byt5_examples, document_split
from evaluation.evaluation import compute_span_cer

cer_metric = hf_evaluate.load("cer")

ABBR_OPEN = "⦃"


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name",        default="google/byt5-base")
    p.add_argument("--dataset_repo",      default=None, help="HF dataset repo id (for Hub/Jobs mode)")
    p.add_argument("--dataset_local",     default=None, help="Local JSONL path (for local testing mode)")
    p.add_argument("--output_repo",       default=None, help="HF model repo (omit for local testing)")
    p.add_argument("--output_dir",        default="./boundary-classifier-output", help="Local output directory (for local testing mode)")
    p.add_argument("--epochs",            type=int,   default=10)
    p.add_argument("--batch_size",        type=int,   default=16)
    p.add_argument("--learning_rate",     type=float, default=1e-4)
    p.add_argument("--max_input_length",  type=int,   default=512)
    p.add_argument("--max_target_length", type=int,   default=512)
    p.add_argument("--oversample_abbr",   type=float, default=2.0)
    p.add_argument("--lang_prefix",       action="store_true")
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--hf_token",          default=os.environ.get("HF_TOKEN"))
    # p.add_argument("--threshold",     type=float, default=0.6)
    # p.add_argument("--min_precision", type=float, default=0.90)
    # p.add_argument("--use_lexicon",   action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def make_tokenize_fn(tokenizer, max_input_length, max_target_length):
    """
    Returns a batched tokenization function for Dataset.map().

    ByT5's tokenizer encodes directly to UTF-8 bytes, so Unicode
    combining characters (macrons, brevigraphs, etc.) and the
    ⦃⦄ delimiters are all handled correctly without special casing.

    Padding is deferred to the DataCollatorForSeq2Seq, which pads
    per-batch rather than globally — important for ByT5's long sequences.
    """
    def tokenize(batch):
        model_inputs = tokenizer(
            batch["source"],
            max_length=max_input_length,
            truncation=True,
            padding=False,
        )
        labels = tokenizer(
            text_target=batch["target"],
            max_length=max_target_length,
            truncation=True,
            padding=False,
        )
        # Replace pad token id in labels with -100 so padding is
        # excluded from the cross-entropy loss computation
        model_inputs["labels"] = [
            [(t if t != tokenizer.pad_token_id else -100) for t in ids]
            for ids in labels["input_ids"]
        ]
        return model_inputs
    return tokenize


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

def decode_predictions(tokenizer, predictions, labels):
    """
    Decode token id arrays from the Trainer into strings.
    Replaces -100 (loss-ignored positions) with pad token id first.
    """
    predictions = np.where(
        predictions != -100, predictions, tokenizer.pad_token_id
    )
    labels = np.where(
        labels != -100, labels, tokenizer.pad_token_id
    )
    decoded_preds  = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels,      skip_special_tokens=True)
    return decoded_preds, decoded_labels


# ---------------------------------------------------------------------------
# compute_metrics callback
# ---------------------------------------------------------------------------

def make_compute_metrics(tokenizer, val_sources):
    """
    Returns a compute_metrics function for Seq2SeqTrainer.

    val_sources: the marked input strings for the val set, captured in
    a closure. Used to compute span-level CER during validation without
    threading extra state through the Trainer API.

    Note: assumes single-GPU training (Trainer preserves val set order).
    For multi-GPU, disable span metrics during training and run post-hoc.
    """
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds, decoded_labels = decode_predictions(
            tokenizer, preds, labels
        )

        full_line_cer = cer_metric.compute(
            predictions=decoded_preds, references=decoded_labels
        )

        span_results = compute_span_cer(
            marked_inputs=val_sources,
            model_outputs=decoded_preds,
            target_corrs=decoded_labels,
        )

        return {
            "full_line_cer":    round(full_line_cer, 4),
            "span_cer":         round(span_results["span_cer"] or 0.0, 4),
            "span_exact_match": round(span_results["span_exact_match"] or 0.0, 4),
            "n_spans":          span_results["n_spans"],
        }

    return compute_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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

    use_hub    = args.output_repo is not None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if use_hub:
        login(token=args.hf_token)
        api = HfApi()

    # --- Data, tokenizer, model (unchanged) ---
    lines    = load_and_sort_lines(data_path)
    examples = build_byt5_examples(
        lines,
        oversample_abbr=args.oversample_abbr,
        lang_prefix=args.lang_prefix,
        seed=args.seed,
    )
    train_ex, val_ex, test_ex = document_split(examples, seed=args.seed)
    print(f"Train: {len(train_ex)} | Val: {len(val_ex)} | Test: {len(test_ex)}")

    if not train_ex:
        print("Warning: empty train set, using full dataset for training")
        train_ex = examples
    if not val_ex:
        print("Warning: empty val set, using full dataset for validation")
        val_ex = examples
    if not test_ex:
        print("Warning: empty test set, using full dataset for evaluation")
        test_ex = examples

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model     = T5ForConditionalGeneration.from_pretrained(args.model_name)
    model.config.tie_word_embeddings = False

    # HuggingFace Datasets
    def to_hf_dataset(exs):
        return Dataset.from_dict({
            "source":   [e["source"]   for e in exs],
            "target":   [e["target"]   for e in exs],
            "has_abbr": [e["has_abbr"] for e in exs],
            "doc_id":   [e["doc_id"]   for e in exs],
            "lang":     [e["lang"]     for e in exs],
        })

    raw = DatasetDict({
        "train": to_hf_dataset(train_ex),
        "val":   to_hf_dataset(val_ex),
        "test":  to_hf_dataset(test_ex),
    })

    tokenize_fn = make_tokenize_fn(
        tokenizer, args.max_input_length, args.max_target_length
    )
    tokenized = raw.map(
        tokenize_fn,
        batched=True,
        remove_columns=["source", "target", "has_abbr", "doc_id", "lang"],
    )

    # Data collator — pads per batch rather than globally
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    # Metrics closure over val sources
    val_sources     = [e["source"] for e in val_ex]
    compute_metrics = make_compute_metrics(tokenizer, val_sources)

    # --- Training arguments: checkpoints go into output_dir/checkpoints ---
    checkpoints_dir = output_dir / "checkpoints"
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(checkpoints_dir),

        # Hub integration only when use_hub is True
        push_to_hub=use_hub,
        hub_model_id=args.output_repo if use_hub else None,
        **({"hub_strategy": "every_save"} if use_hub else {}),

        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=500,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        weight_decay=0.01,

        predict_with_generate=True,
        generation_max_length=args.max_target_length,

        # Use span CER as the model selection criterion —
        # this focuses early stopping on expansion quality,
        # not copying fidelity
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="span_cer",
        greater_is_better=False,

        seed=args.seed,
        logging_steps=100,
        report_to="tensorboard",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["val"],
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    # --- Test evaluation ---
    test_sources = [e["source"] for e in test_ex]
    test_targets = [e["target"] for e in test_ex]

    test_output = trainer.predict(tokenized["test"])
    test_preds, test_labels = decode_predictions(
        tokenizer,
        test_output.predictions,
        test_output.label_ids,
    )

    test_metrics = compute_span_cer(
        marked_inputs=test_sources,
        model_outputs=test_preds,
        target_corrs=test_targets,
    )

    print(f"\nTest span CER:         {test_metrics['span_cer']:.4f}")
    print(f"Test span exact match: {test_metrics['span_exact_match']:.4f}")
    print(f"Test full-line CER:    {test_metrics['full_line_cer']:.4f}")
    print(f"Spans evaluated:       {test_metrics['n_spans']}")

    # --- Write breakdown locally ---
    breakdown_path = output_dir / "test_breakdown.json"
    breakdown_path.write_text(
        json.dumps(test_metrics["by_abbr_type"], ensure_ascii=False, indent=2)
    )
    print(f"Wrote test_breakdown.json to {output_dir}/")

    # --- Optionally push to Hub ---
    if use_hub:
        api.upload_file(
            path_or_fileobj=str(breakdown_path),
            path_in_repo="test_breakdown.json",
            repo_id=args.output_repo,
            repo_type="model",
        )
        trainer.push_to_hub(commit_message="Training complete — best checkpoint")
    else:
        print(f"Results saved locally to {output_dir}/")


if __name__ == "__main__":
    main()
