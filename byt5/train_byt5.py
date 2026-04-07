# train_byt5.py
#
# ByT5-base abbreviation expansion — full training script for HuggingFace Jobs.
#
# Task: sequence-to-sequence expansion of marked input lines, using the same
# examples as the boundary classifier but with the full line as input and the
# fully expanded line as output. ByT5's byte-level tokenization allows it to
# handle the full variety of Unicode characters in the source and target text
# without special-casing, and to learn character-level expansion patterns.
#
# The script is designed to run in two modes:
# 1) Local testing mode: specify a local JSONL path with --dataset_local and
#    an output directory with --output_dir. No Hub integration.
# 2) HuggingFace Hub/Jobs mode: specify a dataset repo with --dataset_repo and
#    a model repo with --output_repo. The script will read the data
#    from the dataset repo, train the model, and push the best checkpoint and
#    test breakdown to the model repo at the end of training.
#
# The model runs as the second stage of a pipeline, following the boundary
# classifier. The boundary classifier identifies which line pairs should be
# concatenated before being passed to the ByT5 abbreviation expansion
# model. The model presupposes the availability of this concatenation
# information in inferencing.

import os
import json
import random
import argparse
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import evaluate as hf_evaluate
import wandb

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

from codecarbon import EmissionsTracker
from transformers import TrainerCallback

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
    p.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps to accumulate gradients for (to achieve larger effective batch size on limited VRAM)")
    p.add_argument("--learning_rate",     type=float, default=1e-4)
    p.add_argument("--max_input_length",  type=int,   default=512)
    p.add_argument("--max_target_length", type=int,   default=512)
    p.add_argument("--oversample_abbr",   type=float, default=2.0)
    p.add_argument("--lang_prefix",       action="store_true")
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--hf_token",          default=os.environ.get("HF_TOKEN"))
    p.add_argument("--wandb_key",         default=os.environ.get("WANDB_API_KEY"))
    p.add_argument("--wandb_project",     default=os.environ.get("WANDB_PROJECT", "byt5-salamanca-abbr"))
    p.add_argument("--wandb_entity",      default=os.environ.get("WANDB_ENTITY", None))
    p.add_argument("--use_cache",         action="store_true", help="Load tokenized dataset from cache if available")
    p.add_argument("--bf16",              action="store_true", help="Use bf16 mixed precision (A10G support; requires compatible GPU and drivers)")
    p.add_argument("--fp16",              action="store_true", help="Use fp16 mixed precision (for GPUs without bf16 support; requires compatible GPU and drivers)")
    p.add_argument("--tokenizer_num_proc", type=int, default=1, help="Number of processes for tokenization. Use >1 to speed up on multi-core machines.")
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
# Carbon tracking callback for HuggingFace Trainer --- optional but
# recommended for monitoring training emissions, especially on GPU.
# Logs total CO2 emissions at the end of training.
# ---------------------------------------------------------------------------

class CarbonTrackerCallback(TrainerCallback):
    def __init__(self, output_dir: str):
        self.tracker = EmissionsTracker(
            project_name="byt5-salamanca-abbr",
            output_dir=output_dir,
            log_level="warning",
        )
        self.output_dir = Path(output_dir)

    def on_train_begin(self, args, state, control, **kwargs):
        self.tracker.start()

    def on_train_end(self, args, state, control, **kwargs):
        emissions = self.tracker.stop()
        print(f"Training CO2: {emissions:.4f} kg CO2eq")

        # Write structured summary alongside other outputs
        emissions_path = self.output_dir / "emissions.json"
        emissions_path.write_text(json.dumps({
            "kg_co2eq":     emissions,
            "model":        "google/byt5-base",
            "hardware":     "a100-large",
        }, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    random.seed(args.seed)

    # Login to HugingFace Hub
    use_hub = args.output_repo is not None
    if args.hf_token and (args.dataset_repo or use_hub):
        login(token=args.hf_token)
        api = HfApi()

    # --- Fail fast if GPU requested but unavailable ---
    if args.bf16 or args.fp16:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "bf16/fp16 requested but no CUDA GPU is available. "
                "Check driver compatibility or remove --bf16/--fp16 flag."
            )
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # --- Resolve data path ---
    if args.dataset_local:
        data_path = args.dataset_local
        print(f"Local mode: reading from {data_path}")
    elif args.dataset_repo:
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
    model     = T5ForConditionalGeneration.from_pretrained(
        args.model_name,
        tie_word_embeddings=False,
    )

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

    cache_path = output_dir / "tokenized_cache"
    if args.use_cache and not cache_path.exists() and use_hub:
        try:
            print("Downloading tokenized cache from Hub...")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=args.output_repo,
                local_dir=str(output_dir),
                allow_patterns="tokenized_cache/**",
                repo_type="model",
            )
            if cache_path.exists():
                print("Cache downloaded successfully.")
            else:
                print("No cached tokenization found on Hub, will tokenize from scratch.")
        except Exception as e:
            print(f"Cache download failed ({e}), will tokenize from scratch.")
    if args.use_cache and cache_path.exists():
        print("Loading tokenized dataset from cache...")
        tokenized = DatasetDict.load_from_disk(str(cache_path))
    else:
        print("Tokenizing dataset...")
        tokenize_fn = make_tokenize_fn(
            tokenizer, args.max_input_length, args.max_target_length
        )
        tokenized = raw.map(
            tokenize_fn,
            batched=True,
            num_proc=args.tokenizer_num_proc,
            remove_columns=["source", "target", "has_abbr", "doc_id", "lang"],
        )
        if args.use_cache:
            tokenized.save_to_disk(str(cache_path))
            print(f"Tokenized dataset cached to {cache_path}")
            if use_hub:
                print("Uploading tokenized cache to Hub...")
                try:
                    api.upload_folder(
                        folder_path=str(cache_path),
                        path_in_repo="tokenized_cache",
                        repo_id=args.output_repo,
                        repo_type="model",
                    )
                    print("Cache uploaded to Hub.")
                except Exception as e:
                    print(f"Warning: cache upload failed ({e}). Training will continue.")
        else:
            print("Tokenized dataset not cached (use --use_cache to enable caching)")
    print(f"Tokenized dataset: {tokenized}")

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
    training_kwargs = dict(
        output_dir=str(checkpoints_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=500,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        weight_decay=0.01,

        bf16=args.bf16, # Mixed precision to save memory, as configured in command-line
        fp16=args.fp16, # A10G support bf16; use fp16=True if not available
        # If we have to use smaller batches to avoid OOM,
        # we have to compensate with gradient accumulation to reach
        # an effective batch size of at least 16
        # (gradient_accumulation_steps * batch_size = effective batch size)
        gradient_accumulation_steps=args.gradient_accumulation_steps,

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
        report_to="all", # tensorboard + WandB (if installed and logged in)
        dataloader_pin_memory=False,   # suppress pin_memory warning on CPU

        # Hub integration only when use_hub is True
        push_to_hub=use_hub,
        hub_model_id=args.output_repo if use_hub else None,
    )
    if use_hub:
        training_kwargs["hub_strategy"] = "every_save"
    training_args = Seq2SeqTrainingArguments(**training_kwargs)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["val"],
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3),
            CarbonTrackerCallback(str(output_dir)),
        ],
    )

    # --- Initialize WandB run (optional, but recommended for rich logging and visualization) ---
    wandb.login(key=args.wandb_key)
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,   # None means wandb uses your default entity
        name=f"byt5-base-{args.epochs}ep-bs{args.batch_size}",
        config=vars(args),
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
