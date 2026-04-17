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
from datetime import datetime

import numpy as np
import torch
import evaluate as hf_evaluate

# HPC: conditionally import wandb (may run in offline mode)
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# HPC: Maybe we have CUDA, maybe ROC, Triton or sth.

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
    p.add_argument("--output_repo",       default=None, help="HF model repo (omit for local testing)")
    p.add_argument("--output_dir",        default="./byt5_classifier_output", help="Local output directory (for local testing mode)")
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
    p.add_argument("--gradient_checkpointing", action="store_true")

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
    resume_group.add_argument("--eval_only", action="store_true", help="Skip training, load best model from Hub, run test eval only")
    resume_group.add_argument("--no_resume", action="store_true", help="Train from scratch even if a checkpoint exists on the Hub")
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
# Checkpoint management for HuggingFace Hub integration
# ---------------------------------------------------------------------------

def find_resume_checkpoint(api: HfApi, repo_id: str, local_dir: Path) -> str | None:
    """
    Check the Hub model repo for existing checkpoints.
    If found, download the latest one and return its local path.
    Returns None if no checkpoint exists or download fails.
    """
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

        latest = checkpoint_dirs[-1]  # e.g. "checkpoints/checkpoint-14979"
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
    """
    Returns a compute_metrics function for Seq2SeqTrainer.

    val_sources: the marked input strings for the val set, captured in
    a closure. Used to compute span-level CER during validation without
    threading extra state through the Trainer API.

    Note: assumes single-GPU training (Trainer preserves val set order).
    For multi-GPU, disable span metrics during training and run post-hoc.
    """

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
        # Sample to avoid jiwer hanging on large val sets
        try:
            full_line_cer_result = cer_metric.compute(
                predictions=decoded_preds[:cap_eval],
                references=decoded_labels[:cap_eval],
            )
            full_line_cer = extract_cer(full_line_cer_result) if full_line_cer_result else 0.0
        except ZeroDivisionError:
            full_line_cer = 0.0

        # Cap val_sources to match decoded_preds length
        # to avoid passing 314k sources when only 10k were evaluated
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
    use_hub = args.output_repo is not None

    if args.eval_only and not use_hub:
        raise ValueError("--eval_only requires --output_repo (model to evaluate)")

    if args.eval_strategy == "no":
        if args.cap_eval:
            print("Warning: --cap_eval has no effect with --eval_strategy no")

    if use_hub and not args.hf_token:
        raise ValueError("--output_repo requires HF_TOKEN to be set")

    # Login to HugingFace Hub
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
    # If eval_only is set, we want to load the best model from the Hub (if available)
    # otherwise we'll train from scratch and load the base model.
    model_id = args.output_repo if args.eval_only and use_hub else args.model_name

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print("Loading model...")
    kwargs = dict(tie_word_embeddings=False)
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    model = T5ForConditionalGeneration.from_pretrained(model_id, **kwargs)
    print(f"Model loaded: {args.model_name}")

    print("Initializing data collator...")
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    # Optional: torch.compile for fused operations across ByT5's deep encoder
    if not args.eval_only and torch.__version__ >= "2.0":
        try:
            print("Compiling model with torch.compile...")
            model = torch.compile(model)
            print("Model compiled with torch.compile")
        except Exception as e:
            print(f"torch.compile failed ({e}), continuing without compilation")
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # --- Tokenize (with caching) ---

    tokenize_fn = make_tokenize_fn(
        tokenizer, args.max_input_length, args.max_target_length,
    )

    def tokenize_examples(exs: list[dict]) -> Dataset:
        """Build a HF Dataset from raw examples and tokenize it."""
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

    # In eval-only mode, we only need the test split — no caching, no training data
    if args.eval_only:
        print("Eval-only mode: tokenizing just the test split...")
        tokenized_test = tokenize_examples(test_ex)

    else:
        cache_path = output_dir / "tokenized_cache"
        cache_valid = False

        if args.use_cache:
            # Try to download cache from Hub if not present locally
            if not cache_path.exists() and use_hub:
                try:
                    print("Downloading tokenized cache from Hub...")
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

            # Check if cache is (now) available locally, validate it, and load if valid
            if cache_path.exists():
                print("Loading tokenized dataset from cache...")
                tokenized = DatasetDict.load_from_disk(str(cache_path))
                expected = { "train": len(train_ex), "val": len(val_ex), "test": len(test_ex) }
                actual = { split: len(tokenized[split]) for split in tokenized }
                if actual == expected:
                    print(f"Cache valid with splits: {actual}")
                    cache_valid = True
                else:
                    print(f"Cache invalid: expected {expected}, got {actual}. Will tokenize from scratch.")

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
                        print(f"Warning: cache upload failed ({e}). "
                              "Training will continue.")

        print(f"Tokenized dataset: {tokenized}")
        tokenized_test = tokenized["test"]

    # --- Training ---
    if not args.eval_only:

        # Metrics closure over val sources
        val_sources     = [e["source"] for e in val_ex]
        compute_metrics = make_compute_metrics(tokenizer, val_sources, args.cap_eval)

        # Training arguments: checkpoints go into output_dir/checkpoints
        checkpoints_dir = output_dir / "checkpoints"
        training_kwargs = dict(
            output_dir=str(checkpoints_dir),

            dataloader_num_workers=4,  # or args.cpus_per_task // 2
            dataloader_pin_memory=True,  # enable on HPC (real GPU)
            # dataloader_pin_memory=False,   # suppress pin_memory warning on CPU
            dataloader_prefetch_factor=2,

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
            # If we have to use smaller batches to avoid OOM,
            # we have to compensate with gradient accumulation to reach
            # an effective batch size of at least 16
            # (gradient_accumulation_steps * batch_size = effective batch size)
            gradient_accumulation_steps=args.gradient_accumulation_steps,

            # Use span CER as the model selection criterion —
            # this focuses early stopping on expansion quality,
            # not copying fidelity
            metric_for_best_model="eval_span_cer",
            per_device_eval_batch_size=args.eval_batch_size,
            eval_strategy=args.eval_strategy,
            eval_steps=args.eval_steps,
            save_strategy=args.eval_strategy,  # checkpoint at same frequency as eval
            save_steps=args.eval_steps,
            load_best_model_at_end=True,
            greater_is_better=False,

            seed=args.seed,
            logging_steps=100,
            report_to="all",  # log to both console and WandB (if initialized)

            # Hub integration only when use_hub is True
            push_to_hub=use_hub,
            hub_model_id=args.output_repo if use_hub else None,
        )
        if use_hub:
            training_kwargs["hub_strategy"] = "checkpoint"
        training_args = Seq2SeqTrainingArguments(**training_kwargs)

        # Use SDPA (flash-like attention) — this is likely already the default
        # but being explicit ensures it
        kwargs["attn_implementation"] = args.attn_implementation or "sdpa"



        # Optional: gradient checkpointing for memory savings
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled")

        print("Initializing Trainer...")
        eval_size = args.cap_eval or min(10000, len(tokenized["val"]))
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["val"].select(range(min(eval_size, len(tokenized["val"])))),
            data_collator=collator,
            # val_sources=val_sources,
            # cap_eval=args.cap_eval,
            # tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=3),
                CarbonTrackerCallback(str(output_dir)),
            ],
        )

        print("Initializing WandB...")
        wandb.login(key=args.wandb_key)
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,   # None means wandb uses your default entity
            name=f"byt5-base-{args.epochs}ep-bs{args.train_batch_size}",
            config=vars(args),
        )

        resume_checkpoint = (
            find_resume_checkpoint(api, args.output_repo, output_dir)
            if use_hub and not args.no_resume else None
        )

        # --- Do it! ---
        print("Starting training...")
        trainer.train(resume_from_checkpoint=resume_checkpoint)
        print("Training complete.")

        # Push the best model to the Hub even before evaluation to be on the safe side
        # in case something goes wrong during evaluation and prevents the final push
        if use_hub:
            print("Pushing best model to Hub...")
            trainer.push_to_hub(commit_message="Best checkpoint after training")

    # --- Test evaluation (in both training and eval_only mode) ---
    test_sources = [e["source"] for e in test_ex]

    # eval-only mode: we can do with light-weight trainer just for prediction
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

    # --- Optionally push to Hub ---
    if use_hub:
        print("Uploading test breakdown to Hub...")
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
