# evaluation.py
#
# Span-level CER and exact match evaluation for abbreviation expansion.
# Used by both the ByT5 training loop and post-hoc analysis.

import evaluate as hf_evaluate
from dataclasses import dataclass
from collections import defaultdict, Counter
from typing import Optional

cer_metric = hf_evaluate.load("cer")

ABBR_OPEN  = "⦃"
ABBR_CLOSE = "⦄"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SpanAlignment:
    """
    Holds the abbreviated source span, its expected expansion,
    and the character offset in the unmarked output line where
    the expansion begins.
    """
    abbr_text:     str
    expanded_text: str
    output_offset: int


@dataclass
class SpanResult:
    """Result of a single span prediction, for breakdown analysis."""
    abbr_text: str
    gold:      str
    predicted: str
    exact:     bool


# ---------------------------------------------------------------------------
# Span alignment
# ---------------------------------------------------------------------------

def extract_span_alignments(
    marked_input: str,
    target_corr:  str,
) -> list[SpanAlignment]:
    """
    Parse a marked input line to recover (abbr_text, expansion, offset) triples.

    marked_input: source line with ⦃...⦄ delimiters around abbreviated spans,
                  e.g. "lib. Lex est communis ciuitatis ⦃cōsensus⦄ qui"
    target_corr:  fully expanded gold line, e.g.
                  "lib. Lex est communis ciuitatis consensus qui"

    Walks both strings in parallel, tracking character offsets. When a
    ⦃...⦄ span is encountered, uses a short lookahead into the context
    after the span to locate the corresponding position in target_corr,
    handling variable-length expansions correctly.
    """
    alignments = []
    i          = 0
    out_offset = 0

    while i < len(marked_input):
        if marked_input[i] == ABBR_OPEN:
            close     = marked_input.index(ABBR_CLOSE, i)
            abbr_text = marked_input[i+1:close]
            after     = marked_input[close+1:close+6]   # 5-char lookahead

            exp_end = out_offset
            while exp_end < len(target_corr):
                if target_corr[exp_end:exp_end+len(after)] == after:
                    break
                exp_end += 1

            alignments.append(SpanAlignment(
                abbr_text=abbr_text,
                expanded_text=target_corr[out_offset:exp_end],
                output_offset=out_offset,
            ))
            out_offset = exp_end
            i          = close + 1

        elif marked_input[i] not in (ABBR_OPEN, ABBR_CLOSE):
            i          += 1
            out_offset += 1
        else:
            i += 1

    return alignments


def extract_predicted_spans(
    model_output: str,
    alignments:   list[SpanAlignment],
) -> list[str]:
    """
    Extract what the model produced at each span position,
    using stored offsets from extract_span_alignments.
    """
    spans = []
    for a in alignments:
        end  = a.output_offset + len(a.expanded_text)
        pred = (model_output[a.output_offset:end]
                if end <= len(model_output)
                else model_output[a.output_offset:])
        spans.append(pred)
    return spans


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_span_cer(
    marked_inputs: list[str],
    model_outputs: list[str],
    target_corrs:  list[str],
) -> dict:
    """
    Compute evaluation metrics over abbreviated spans and full lines.

    Returns:
        span_cer:          CER over abbreviated spans only
        full_line_cer:     CER over full output lines
        span_exact_match:  proportion of spans predicted exactly correctly
        n_spans:           total number of spans evaluated
        n_exact:           raw count of exactly correct spans
        by_abbr_type:      per-abbreviation-type breakdown dict
    """
    gold_all  = []
    pred_all  = []
    results   = []

    for marked, output, target in zip(marked_inputs, model_outputs, target_corrs):
        if ABBR_OPEN not in marked:
            continue
        alignments = extract_span_alignments(marked, target)
        predicted  = extract_predicted_spans(output, alignments)
        for a, p in zip(alignments, predicted):
            gold_all.append(a.expanded_text)
            pred_all.append(p)
            results.append(SpanResult(
                abbr_text=a.abbr_text,
                gold=a.expanded_text,
                predicted=p,
                exact=(p == a.expanded_text),
            ))

    n_spans = len(gold_all)
    if n_spans == 0:
        return {
            "span_cer":         None,
            "full_line_cer":    None,
            "span_exact_match": None,
            "n_spans":          0,
            "n_exact":          0,
            "by_abbr_type":     {},
        }

    n_exact = sum(r.exact for r in results)

    try:
        span_cer_result = cer_metric.compute(predictions=pred_all, references=gold_all)
        span_cer: float = float(span_cer_result["cer"]) if span_cer_result else 0.0
    except ZeroDivisionError:
        span_cer = 0.0

    # Full-line CER is computed on a sample to avoid hanging on large val sets
    max_cer_samples = 10000
    try:
        full_line_cer_result = cer_metric.compute(
            predictions=model_outputs[:max_cer_samples],
            references=target_corrs[:max_cer_samples],
        )
        full_line_cer: float = float(full_line_cer_result["cer"]) if full_line_cer_result else 0.0
    except ZeroDivisionError:
        full_line_cer = 0.0

    return {
        "span_cer":         span_cer,
        "full_line_cer":    full_line_cer,
        "span_exact_match": n_exact / n_spans,
        "n_spans":          n_spans,
        "n_exact":          n_exact,
        "by_abbr_type":     build_type_breakdown(results),
    }

def build_type_breakdown(results: list[SpanResult]) -> dict:
    """
    Group SpanResults by abbr_text and compute per-type statistics.

    For each abbreviation type returns:
        n:           total occurrences in the evaluated set
        n_exact:     exactly correct predictions
        exact_match: exact match rate
        cer:         CER over this type's predictions
        errors:      up to 20 (predicted, gold) error pairs
        expansions:  Counter of gold expansions seen for this abbr_text,
                     useful for spotting genuine ambiguity in the corpus
    """
    grouped: dict = defaultdict(list)
    for r in results:
        grouped[r.abbr_text].append(r)

    breakdown = {}
    for abbr_text, type_results in sorted(
        grouped.items(), key=lambda kv: len(kv[1]), reverse=True
    ):
        n       = len(type_results)
        n_exact = sum(r.exact for r in type_results)
        preds   = [r.predicted for r in type_results]
        golds   = [r.gold      for r in type_results]

        # Guard against empty strings causing ZeroDivisionError in CER
        try:
            type_cer_result = cer_metric.compute(predictions=preds, references=golds)
            type_cer: float = float(type_cer_result["cer"]) if type_cer_result else 0.0
        except ZeroDivisionError:
            type_cer = 0.0

        breakdown[abbr_text] = {
            "n":           n,
            "n_exact":     n_exact,
            "exact_match": n_exact / n,
            "cer":         type_cer,
            "errors":      [
                {"predicted": r.predicted, "gold": r.gold}
                for r in type_results if not r.exact
            ][:20],
            "expansions":  dict(Counter(r.gold for r in type_results)),
        }

    return breakdown
