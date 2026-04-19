# train_byt5.py
#
# ByT5-base abbreviation expansion — full training script.
# Adapted for HPC/SLURM offline execution with checkpoint-resume support.

import os
import json
import random
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

# HPC: conditionally import wandb (may run in offline mode)
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

import evaluate as hf_evaluate

if torch.cuda.is_available() and "CUDA" in torch.cuda.get_device_name(0).upper():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
    TrainerCallback,
)

# HPC: only import hub utilities when needed
from huggingface_hub import login, HfApi, hf_hub_download, RepoFolder, snapshot_download

from codecarbon import EmissionsTracker

from data.data_utils import load_and_sort_lines, build_byt5_examples, document_split
from evaluation.evaluation import compute_span_cer, extract_cer

cer_metric = hf_evaluate.load("cer")

ABBR_OPEN = "⦃"


def _ts():
    return datetime.now().strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name",        default="google/byt5-base")
    p.add_argument("--output_repo",       default=None, help="HF model repo (omit for local/HPC mode)")
    p.add_argument("--output_dir",        default="./byt5_classifier_output", help="Local output directory")
    p.add_argument("--epochs",            type=int,   default=10)
    p.add_argument("--eval_strategy",     default="steps", choices=["no", "epoch", "steps"])
    p.add_argument("--eval_steps",        type=int,   default=5000)
    p.add_argument("--train_batch_size",  type=int,   default=16)
    p.add_argument("--eval_batch_size",   type=int,   default=128)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps to accumulate gradients for (to achieve larger effective batch size on limited VRAM)")
    p.add_argument("--learning_rate",     type=float, default=1e-4)
    p.add_argument("--max_input_length",  type=int,   default=512)
    p.add_argument("--max_target_length", type=int,   default=512)
    p.add_argument("--cap_eval",          type=int,   default=None, help="Max eval samples for generation and metrics during training")
    p.add_argument("--oversample_abbr",   type=float, default=2.0)
    p.add_argument("--lang_prefix",       action="store_true")
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--hf_token",          default=os.environ.get("HF_TOKEN"))
    p.add_argument("--wandb_key",         default=os.environ.get("WANDB_API_KEY"))
    p.add_argument("--wandb_project",     default=os.environ.get("WANDB_PROJECT", "byt5-salamanca-abbr"))
    p.add_argument("--wandb_entity",      default=os.environ.get("WANDB_ENTITY", None))
    p.add_argument("--use_cache",         action="store_true", help="Load tokenized dataset from cache if available")
    p.add_argument("--tokenizer_num_proc", type=int, default=1, help="Number of processes for tokenization. Use >1 to speed up on multi-core machines.")
    p.add_argument("--attn_implementation", default=None, help="Attention backend: 'sdpa', 'flash_attention_2', or None for default")

    # HPC: save_total_limit for checkpoint disk management
    p.add_argument("--save_total_limit",  type=int, default=3, help="Max checkpoints to keep on disk (oldest deleted)")
    # HPC: eval_model_dir to point at a local model directory for eval_only
    p.add_argument("--eval_model_dir", default=None, help="Local directory containing model to evaluate (for eval_only without Hub access)")

    data_group = p.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--dataset_repo",  default=None, help="HF dataset repo id (for Hub/Jobs mode)")
    data_group.add_argument("--dataset_local", default=None, help="Local JSONL path (for local testing mode)")

    precision_group = p.add_mutually_exclusive_group()
    precision_group.add_argument("--bf16", action="store_true", help="Use bf16 mixed precision (A10G support; requires compatible GPU and drivers)")
    precision_group.add_argument("--fp16", action="store_true", help="Use fp16 mixed precision (for GPUs without bf16 support; requires compatible GPU and drivers)")
    resume_group = p.add_mutually_exclusive_group()
    resume_group.add_argument("--eval_only", action="store_true", help="Skip training, run test eval only")
    resume_group.add_argument("--no_resume", action="store_true", help="Train from scratch ignoring existing checkpoints")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def make_tokenize_fn(tokenizer, max_input_length, max_target_length):
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
# Checkpoint management — HPC local version
# ---------------------------------------------------------------------------

def find_local_resume_checkpoint(checkpoints_dir: Path) -> str | None:
    """
    HPC: Find the latest checkpoint in the local checkpoints directory.
    Returns the path to the latest checkpoint, or None if none found.
    """
    if not checkpoints_dir.exists():
        print(f"No checkpoints directory at {checkpoints_dir}")
        return None

    checkpoint_dirs = sorted(
        [d for d in checkpoints_dir.iterdir()
         if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.rsplit("-", 1)[-1]),
    )

    if not checkpoint_dirs:
        print("No local checkpoints found, training from scratch.")
        return None

    latest = checkpoint_dirs[-1]
    # Validate: must contain at minimum model weights and optimizer state
    required_files = ["trainer_state.json"]
    has_weights = (latest / "model.safetensors").exists() or \
                  (latest / "pytorch_model.bin").exists() or \
                  (latest / "adapter_model.safetensors").exists()

    if not has_weights or not all((latest / f).exists() for f in required_files):
        print(f"Warning: checkpoint {latest} appears incomplete, skipping.")
        # Try second-to-last
        if len(checkpoint_dirs) >= 2:
            latest = checkpoint_dirs[-2]
            has_weights = (latest / "model.safetensors").exists() or \
                          (latest / "pytorch_model.bin").exists()
            if has_weights and all((latest / f).exists() for f in required_files):
                print(f"Falling back to {latest}")
            else:
                print("No valid checkpoint found.")
                return None
        else:
            return None

    print(f"Found local checkpoint: {latest}")
    return str(latest)


def find_resume_checkpoint_hub(api: HfApi, repo_id: str, local_dir: Path) -> str | None:
    """Original Hub-based checkpoint finder (kept for non-HPC use)."""
    try:
        entries = list(api.list_repo_tree(
            repo_id=repo_id,
            path_in_repo="checkpoints",
            repo_type="model",
        ))
        checkpoint_dirs = sorted(
            [e.path for e in entries
             if isinstance(e, RepoFolder) and e.path.startswith("checkpoints/checkpoint-")],
            key=lambda x: int(x.rsplit("-", 1)[-1]),
        )
        if not checkpoint_dirs:
            print("No remote checkpoints found, training from scratch.")
            return None

        latest = checkpoint_dirs[-1]
        print(f"Found remote checkpoint: {latest}, downloading...")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            allow_patterns=f"{latest}/**",
            repo_type="model",
        )
        local_path = str(local_dir / latest)
        print(f"Will resume from {local_path}")
        return local_path

    except Exception as e:
        print(f"Checkpoint lookup failed ({e}), training from scratch.")
        return None


# ---------------------------------------------------------------------------
# compute_metrics callback
# ---------------------------------------------------------------------------

def make_compute_metrics(tokenizer, val_sources, cap_eval=None):
    if cap_eval is None:
        cap_eval = len(val_sources)

    def compute_metrics(eval_preds):

        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        print(f"[{_ts()}] compute_metrics: decoding {len(preds)} predictions ...")
        decoded_preds, decoded_labels = decode_predictions(
            tokenizer, preds, labels
        )

        print(f"[{_ts()}] compute_metrics: decoding done. Starting full-line CER ...")
        try:
            full_line_cer_result = cer_metric.compute(
                predictions=decoded_preds[:cap_eval],
                references=decoded_labels[:cap_eval],
            )
            full_line_cer = extract_cer(full_line_cer_result) if full_line_cer_result else 0.0
        except ZeroDivisionError:
            full_line_cer = 0.0

        capped_val_sources = val_sources[:len(decoded_preds)]
        print(f"[{_ts()}] compute_metrics: full-line CER done ({full_line_cer:.4f}). Starting span CER over {len(capped_val_sources)} sources ...")

        span_results = compute_span_cer(
            marked_inputs=capped_val_sources,
            model_outputs=decoded_preds,
            target_corrs=decoded_labels,
            include_breakdown=False,   # skip expensive breakdown during training
            max_source_lines=cap_eval,
        )
        print(f"[{_ts()}] compute_metrics: span CER done, returning metrics")

        return {
            "full_line_cer":    round(full_line_cer, 4),
            "span_cer":         round(span_results["span_cer"] or 0.0, 4),
            "span_exact_match": round(span_results["span_exact_match"] or 0.0, 4),
            "n_spans":          span_results["n_spans"],
        }

    return compute_metrics


# ---------------------------------------------------------------------------
# Carbon tracking callback
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
        emissions_path = self.output_dir / "emissions.json"
        emissions_path.write_text(json.dumps({
            "kg_co2eq":     emissions,
            "model":        "google/byt5-base",
            "hardware":     "hpc-gpu",
        }, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    random.seed(args.seed)
    use_hub = args.output_repo is not None

    if args.eval_only and not use_hub and not args.eval_model_dir:
        raise ValueError("--eval_only requires either --output_repo or --eval_model_dir")

    if args.eval_strategy == "no" and args.cap_eval:
        print("Warning: --cap_eval has no effect with --eval_strategy no")

    # HPC: Hub login only when explicitly using Hub mode
    api = None
    if use_hub:
        if not args.hf_token:
            raise ValueError("--output_repo requires HF_TOKEN to be set")
        api = HfApi()
        login(token=args.hf_token)

    # --- Fail fast if GPU requested but unavailable ---
    if args.bf16 or args.fp16:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "bf16/fp16 requested but no CUDA GPU is available."
            )
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # --- Resolve data path ---
    if args.dataset_local:
        data_path = args.dataset_local
        print(f"Local mode: reading from {data_path}")
    elif args.dataset_repo:
        # Try local path first (for HPC pre-downloaded data)
        data_path = hf_hub_download(
            repo_id=args.dataset_repo,
            filename="data.jsonl",
            repo_type="dataset",
        )
    else:
        raise ValueError("Provide either --dataset_local or --dataset_repo")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    print("Loading lines...")
    lines = load_and_sort_lines(data_path)
    print(f"Loaded {len(lines)} lines from {data_path}")
    print("Building examples...")
    examples = build_byt5_examples(
        lines,
        oversample_abbr=args.oversample_abbr,
        lang_prefix=args.lang_prefix,
        seed=args.seed,
    )
    print(f"Built {len(examples)} examples")
    print("Splitting dataset...")
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

    # --- Load model, tokenizer and collator ---
    # HPC: Determine which model to load
    if args.eval_only:
        if args.eval_model_dir:
            model_id = args.eval_model_dir
        elif use_hub:
            model_id = args.output_repo
        else:
            raise ValueError("eval_only needs --eval_model_dir or --output_repo")
    else:
        model_id = args.model_name

    print(f"Loading tokenizer from {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(f"Loading model from {model_id} ...")
    kwargs = dict(tie_word_embeddings=False)
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    model = T5ForConditionalGeneration.from_pretrained(model_id, **kwargs)
    print(f"Model loaded from: {model_id}")

    print("Initializing data collator...")
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    # --- Tokenize (with caching) ---

    tokenize_fn = make_tokenize_fn(
        tokenizer, args.max_input_length, args.max_target_length,
    )

    def tokenize_examples(exs: list[dict]) -> Dataset:
        raw = Dataset.from_dict({
            "source": [e["source"] for e in exs],
            "target": [e["target"] for e in exs],
        })
        return raw.map(
            tokenize_fn,
            batched=True,
            num_proc=args.tokenizer_num_proc,
            remove_columns=["source", "target"],
        )

    if args.eval_only:
        print("Eval-only mode: tokenizing just the test split...")
        tokenized_test = tokenize_examples(test_ex)
    else:
        cache_path = output_dir / "tokenized_cache"
        cache_valid = False

        if args.use_cache and cache_path.exists():
            print("Loading tokenized dataset from cache...")
            tokenized = DatasetDict.load_from_disk(str(cache_path))
            expected = {
                "train": len(train_ex),
                "val": len(val_ex),
                "test": len(test_ex),
            }
            actual = { split: len(tokenized[split]) for split in tokenized }
            if actual == expected:
                print(f"Cache valid with splits: {actual}")
                cache_valid = True
            else:
                print(f"Cache invalid: expected {expected}, got {actual}.")

        if not cache_valid:
            print("Tokenizing all splits from scratch ...")
            tokenized = DatasetDict({
                "train": tokenize_examples(train_ex),
                "val":   tokenize_examples(val_ex),
                "test":  tokenize_examples(test_ex),
            })
            if args.use_cache:
                tokenized.save_to_disk(str(cache_path))
                print(f"Tokenized dataset cached to {cache_path}")

        print(f"Tokenized dataset: {tokenized}")
        tokenized_test = tokenized["test"]

    # --- Training ---
    if not args.eval_only:

        val_sources     = [e["source"] for e in val_ex]
        compute_metrics = make_compute_metrics(
            tokenizer, val_sources, args.cap_eval
        )

        checkpoints_dir = output_dir / "checkpoints"
        training_kwargs = dict(
            output_dir=str(checkpoints_dir),

            warmup_steps=500,
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            predict_with_generate=True,

            generation_max_length=args.max_target_length,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.train_batch_size,
            learning_rate=args.learning_rate,

            bf16=args.bf16, # Mixed precision to save memory, as configured in command-line
            fp16=args.fp16, # A10G support bf16; use fp16=True if not available
            gradient_accumulation_steps=args.gradient_accumulation_steps,

            metric_for_best_model="eval_span_cer",
            per_device_eval_batch_size=args.eval_batch_size,
            eval_strategy=args.eval_strategy,
            eval_steps=args.eval_steps,
            save_strategy=args.eval_strategy,  # checkpoint at same frequency as eval
            save_steps=args.eval_steps,
            load_best_model_at_end=True,
            greater_is_better=False,

            # HPC: save_total_limit to manage disk space
            save_total_limit=args.save_total_limit,

            seed=args.seed,
            logging_steps=100,
            report_to="all",
            dataloader_pin_memory=False,

            # HPC: disable Hub push during training (no internet)
            push_to_hub=False,
            hub_model_id=None,
        )

        training_args = Seq2SeqTrainingArguments(**training_kwargs)

        print("Initializing Trainer...")
        eval_size = args.cap_eval or min(10000, len(tokenized["val"]))
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["val"].select(
                range(min(eval_size, len(tokenized["val"])))
            ),
            data_collator=collator,
            compute_metrics=compute_metrics,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=3),
                CarbonTrackerCallback(str(output_dir)),
            ],
        )

        # HPC: Initialize WandB in offline mode
        if HAS_WANDB:
            if args.wandb_key:
                wandb.login(key=args.wandb_key)
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,   # None means wandb uses your default entity
                name=f"byt5-base-{args.epochs}ep-bs{args.train_batch_size}",
                config=vars(args),
                mode=os.environ.get("WANDB_MODE", "online"),
            )

        # HPC: find checkpoint locally instead of from Hub
        resume_checkpoint = None
        if not args.no_resume:
            resume_checkpoint = find_local_resume_checkpoint(checkpoints_dir)

        # --- Do it! ---
        print("Starting training...")
        trainer.train(resume_from_checkpoint=resume_checkpoint)
        print("Training complete.")

        # HPC: Save the best model to a clean directory for later upload
        final_model_dir = output_dir / "final_model"
        print(f"Saving best model to {final_model_dir} ...")
        trainer.save_model(str(final_model_dir))
        tokenizer.save_pretrained(str(final_model_dir))
        print("Best model saved.")

    # --- Test evaluation (in both training and eval_only mode) ---
    test_sources = [e["source"] for e in test_ex]

    if args.eval_only:
        eval_args = Seq2SeqTrainingArguments(
            output_dir=str(output_dir / "eval_tmp"),
            per_device_eval_batch_size=args.eval_batch_size,
            predict_with_generate=True,
            generation_max_length=args.max_target_length,
            bf16=args.bf16,
            fp16=args.fp16,
        )
        trainer = Seq2SeqTrainer(
            model=model, args=eval_args,
            data_collator=collator,
        )

    print("Running test evaluation. Predicting...")
    test_output = trainer.predict(tokenized_test)
    test_preds, test_labels = decode_predictions(
        tokenizer,
        test_output.predictions,
        test_output.label_ids,
    )
    print("Test evaluation complete. Computing test metrics...")
    test_metrics = compute_span_cer(
        marked_inputs=test_sources,
        model_outputs=test_preds,
        target_corrs=test_labels,
        include_breakdown=True,   # include breakdown for final test evaluation
        max_source_lines=None,    # process everything for the final evaluation
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

    # HPC: Also save a summary metrics file
    metrics_path = output_dir / "test_metrics.json"
    metrics_path.write_text(json.dumps({
        "span_cer": test_metrics["span_cer"],
        "span_exact_match": test_metrics["span_exact_match"],
        "full_line_cer": test_metrics["full_line_cer"],
        "n_spans": test_metrics["n_spans"],
    }, indent=2))
    print(f"Wrote test_metrics.json to {output_dir}/")

    # HPC: In eval_only mode, also save the model cleanly if loaded from
    # a checkpoint (to create a consistent final_model directory)
    if args.eval_only:
        final_model_dir = output_dir / "final_model"
        if not final_model_dir.exists():
            print(f"Saving evaluated model to {final_model_dir} ...")
            model.save_pretrained(str(final_model_dir))
            tokenizer.save_pretrained(str(final_model_dir))
            print("Model saved for upload.")

    print(f"All results saved to {output_dir}/")


if __name__ == "__main__":
    main()
