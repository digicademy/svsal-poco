# streaming_eval_trainer.py
#
# Drop-in Seq2SeqTrainer subclass that replaces the default evaluate()
# with a streaming version: generation + metric accumulation happen
# batch-by-batch, so no giant prediction array is ever materialized.
#
# This fixes the multi-hour stall observed when the Trainer's internal
# prediction-gathering (pad-to-max + numpy concatenation) causes memory
# pressure and swap thrashing on HF Jobs.

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
import torch
from datetime import datetime
from transformers import Seq2SeqTrainer, PreTrainedTokenizerBase
from torch.utils.data import Dataset
from evaluation.evaluation import (
    extract_span_alignments,
    extract_predicted_spans,
    extract_cer,
    ABBR_OPEN,
)
import evaluate as hf_evaluate

cer_metric = hf_evaluate.load("cer")


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


class StreamingEvalTrainer(Seq2SeqTrainer):
    """
    Seq2SeqTrainer that computes span-level CER incrementally during
    evaluation, avoiding the default evaluate() path that materializes
    all predictions into a single padded numpy array.

    Usage:
        trainer = StreamingEvalTrainer(
            ...,                        # same args as Seq2SeqTrainer
            val_sources=val_sources,    # list[str] of marked source lines
            cap_eval=10000,             # optional cap
        )

    The standard compute_metrics callback is NOT used; instead, metrics
    are computed inline.  predict_with_generate is still honoured for
    the final trainer.predict() call on the test set.
    """

    def __init__(
        self,
        *args: Any,
        val_sources: list[str],
        cap_eval: Optional[int] = None,
        **kwargs: Any,
    ):
        # Remove compute_metrics — we handle it ourselves during evaluate()
        kwargs.pop("compute_metrics", None)
        super().__init__(*args, **kwargs)
        self._val_sources = val_sources
        self._cap_eval = cap_eval

    # ------------------------------------------------------------------
    def evaluate(                                       # type: ignore[override]
        self,
        eval_dataset: Optional[Union[Dataset, Any]] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
        **gen_kwargs: Any,
    ) -> dict[str, float]:
        """
        Streaming evaluation: iterate over the eval DataLoader, generate
        predictions batch-by-batch, decode immediately, and accumulate
        span-level and full-line CER metrics without storing all
        predictions in memory.
        """
        ds: Any = eval_dataset if eval_dataset is not None else self.eval_dataset
        dataloader = self.get_eval_dataloader(ds)

        model: Any = self.model
        model.eval()

        tokenizer: PreTrainedTokenizerBase = self.processing_class  # type: ignore[assignment]

        # Accumulators
        all_pred_strs: list[str] = []
        all_label_strs: list[str] = []
        sample_idx = 0
        total_batches = len(dataloader)

        print(f"[{_ts()}] streaming_evaluate: {total_batches} batches, "
              f"cap_eval={self._cap_eval}")

        cap = self._cap_eval or len(self._val_sources)

        for batch_i, inputs in enumerate(dataloader):
            if sample_idx >= cap:
                break

            inputs = self._prepare_inputs(inputs)
            # Pop labels before generate() — it doesn't accept them
            labels = inputs.pop("labels", None)

            with torch.no_grad():
                generated: torch.Tensor = model.generate(
                    **inputs,
                    max_length=self.args.generation_max_length,
                )

            # Decode this batch immediately — no accumulation of token arrays
            gen_np: np.ndarray[Any, Any] = generated.cpu().numpy()
            if labels is not None:
                lab_np: np.ndarray[Any, Any] = labels.cpu().numpy()
                lab_np = np.where(lab_np != -100, lab_np, tokenizer.pad_token_id)
                batch_labels = tokenizer.batch_decode(
                    lab_np, skip_special_tokens=True,
                )
            else:
                batch_labels = []

            batch_preds = tokenizer.batch_decode(gen_np, skip_special_tokens=True)

            all_pred_strs.extend(batch_preds)
            all_label_strs.extend(batch_labels)
            sample_idx += len(batch_preds)

            if (batch_i + 1) % 20 == 0 or batch_i == total_batches - 1:
                print(f"[{_ts()}] streaming_evaluate: batch {batch_i+1}/{total_batches}, "
                      f"{sample_idx} samples decoded")

        # Trim to cap
        all_pred_strs = all_pred_strs[:cap]
        all_label_strs = all_label_strs[:cap]
        sources = self._val_sources[:len(all_pred_strs)]

        # ---- Compute metrics ----
        print(f"[{_ts()}] streaming_evaluate: computing metrics over "
              f"{len(all_pred_strs)} samples ...")

        # Full-line CER
        try:
            fl_cer = extract_cer(
                cer_metric.compute(predictions=all_pred_strs, references=all_label_strs)
            )
        except ZeroDivisionError:
            fl_cer = 0.0

        # Span-level CER (inline, no breakdown during training)
        gold_spans: list[str] = []
        pred_spans: list[str] = []
        for marked, output, target in zip(sources, all_pred_strs, all_label_strs):
            if ABBR_OPEN not in marked:
                continue
            aligns = extract_span_alignments(marked, target)
            preds = extract_predicted_spans(output, aligns)
            for a, p in zip(aligns, preds):
                gold_spans.append(a.expanded_text)
                pred_spans.append(p)

        n_spans = len(gold_spans)
        if n_spans > 0:
            n_exact = sum(p == g for p, g in zip(pred_spans, gold_spans))
            try:
                span_cer = extract_cer(
                    cer_metric.compute(predictions=pred_spans, references=gold_spans)
                )
            except ZeroDivisionError:
                span_cer = 0.0
        else:
            n_exact, span_cer = 0, 0.0

        metrics: dict[str, float] = {
            f"{metric_key_prefix}_full_line_cer": round(fl_cer, 4),
            f"{metric_key_prefix}_span_cer": round(span_cer, 4),
            f"{metric_key_prefix}_span_exact_match": round(n_exact / n_spans, 4) if n_spans else 0.0,
            f"{metric_key_prefix}_n_spans": n_spans,
            f"{metric_key_prefix}_loss": 0.0,
        }

        print(f"[{_ts()}] streaming_evaluate: done — "
              f"span_cer={span_cer:.4f}, exact={n_exact}/{n_spans}, "
              f"full_line_cer={fl_cer:.4f}")

        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )
        return metrics
