# evaluation.py
#
# Span-level CER and exact match evaluation for abbreviation expansion.
# Used by both the ByT5 training loop and post-hoc analysis.

import evaluate as hf_evaluate
from difflib import SequenceMatcher
from dataclasses import dataclass
from collections import defaultdict, Counter
from datetime import datetime

_cer_metric = None
def get_cer_metric():
    global _cer_metric
    if _cer_metric is None:
        _cer_metric = hf_evaluate.load("cer")
    return _cer_metric

ABBR_OPEN  = "⦃"
ABBR_CLOSE = "⦄"

def _ts():
    return datetime.now().strftime("%H:%M:%S")

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SpanResult:
    """Result of a single span prediction, for breakdown analysis."""
    abbr_text: str
    gold:      str
    predicted: str
    exact:     bool


# ---------------------------------------------------------------------------
# Diff-based span alignment
# ---------------------------------------------------------------------------

def _strip_markers(marked_input: str) -> tuple[str, list[tuple[int, int, str]]]:
    """
    Remove ⦃⦄ delimiters from marked input.

    Returns:
        plain:  the source string with delimiters removed (abbreviated
                forms kept as-is)
        spans:  list of (start, end, abbr_text) tuples giving the
                position of each abbreviated span in the plain string
    """
    parts: list[str] = []
    spans: list[tuple[int, int, str]] = []
    pos = 0
    i   = 0

    while i < len(marked_input):
        if marked_input[i] == ABBR_OPEN:
            close = marked_input.find(ABBR_CLOSE, i + 1)
            if close == -1:
                # Malformed: unclosed delimiter — treat rest as literal
                parts.append(marked_input[i:])
                pos += len(marked_input) - i
                break
            abbr_text = marked_input[i + 1 : close]
            start = pos
            parts.append(abbr_text)
            pos += len(abbr_text)
            spans.append((start, pos, abbr_text))
            i = close + 1
        elif marked_input[i] == ABBR_CLOSE:
            # Stray close delimiter — skip silently
            i += 1
        else:
            parts.append(marked_input[i])
            pos += 1
            i += 1

    return "".join(parts), spans


def _build_char_alignment(source: str, target: str) -> list[int]:
    """
    Build a boundary-position map from source to target using
    SequenceMatcher.

    Returns a list of len(source)+1 integers.  pos_map[i] is the
    target position corresponding to the boundary before source[i].
    pos_map[len(source)] is the target position after the last
    aligned character.

    For 'equal' regions the mapping is 1:1.
    For 'replace' regions the start maps to start and end to end.
    For 'delete' regions all positions map to the single target
    position where the deletion occurs.
    """
    sm = SequenceMatcher(None, source, target, autojunk=False)
    pos_map = [0] * (len(source) + 1)

    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            for k in range(i2 - i1 + 1):
                if i1 + k <= len(source):
                    pos_map[i1 + k] = j1 + k

        elif op == "replace":
            pos_map[i1] = j1
            if i2 <= len(source):
                pos_map[i2] = j2
            # Interior positions — proportional (rarely needed,
            # since span boundaries normally fall at opcode edges)
            src_len = i2 - i1
            tgt_len = j2 - j1
            for k in range(1, src_len):
                if i1 + k <= len(source):
                    pos_map[i1 + k] = j1 + int(k * tgt_len / src_len)

        elif op == "delete":
            for k in range(i2 - i1 + 1):
                if i1 + k <= len(source):
                    pos_map[i1 + k] = j1

        # 'insert': no source positions consumed; adjacent opcodes
        # already cover the boundary.

    return pos_map


def _map_spans(
    spans:   list[tuple[int, int, str]],
    pos_map: list[int],
    text:    str,
) -> list[str]:
    """
    Extract substrings from *text* corresponding to each span,
    using the source→text position map.
    """
    result: list[str] = []
    for s_start, s_end, _ in spans:
        t_start = pos_map[s_start]
        t_end   = pos_map[min(s_end, len(pos_map) - 1)]
        result.append(text[t_start:t_end])
    return result


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def extract_cer(result) -> float:
    """Extract CER float from get_cer_metric().compute() result regardless of return type."""
    if result is None:
        return 0.0
    elif isinstance(result, float):
        return result
    elif isinstance(result, dict) and "cer" in result:
        return float(result["cer"])
    else:
        return 0.0


def compute_span_cer(
    marked_inputs:     list[str],
    model_outputs:     list[str],
    target_corrs:      list[str],
    include_breakdown: bool = True,
    max_source_lines:  int | None = None,
) -> dict:
    """
    Compute evaluation metrics over abbreviated spans and full lines.

    Uses diff-based alignment to robustly map abbreviation spans
    (delimited by ⦃⦄ in marked_inputs) to the corresponding regions
    in target_corrs (gold) and model_outputs (predicted).

    Returns:
        span_cer:          CER over abbreviated spans only
        full_line_cer:     CER over full output lines
        span_exact_match:  proportion of spans predicted exactly correctly
        n_spans:           total number of spans evaluated
        n_exact:           raw count of exactly correct spans
        by_abbr_type:      per-abbreviation-type breakdown dict
        source_cap:        max_source_lines value used (for reporting)
    """

    # Apply cap if specified
    if max_source_lines is not None:
        marked_inputs = marked_inputs[:max_source_lines]
        model_outputs = model_outputs[:max_source_lines]
        target_corrs  = target_corrs[:max_source_lines]

    # Align all three lists to the shortest length
    n = min(len(marked_inputs), len(model_outputs), len(target_corrs))
    if n < len(marked_inputs) or n < len(target_corrs):
        print(f"[{_ts()}] compute_span_cer: aligning lists to shortest "
              f"length ({n}) — inputs={len(marked_inputs)}, "
              f"outputs={len(model_outputs)}, targets={len(target_corrs)}")
        marked_inputs = marked_inputs[:n]
        model_outputs = model_outputs[:n]
        target_corrs  = target_corrs[:n]

    gold_all:  list[str] = []
    pred_all:  list[str] = []
    results:   list[SpanResult] = []

    print(f"[{_ts()}] compute_span_cer: starting span alignment loop "
          f"over {len(marked_inputs)} inputs ...")

    for marked, output, target in zip(marked_inputs, model_outputs, target_corrs):
        if ABBR_OPEN not in marked:
            continue

        plain, spans = _strip_markers(marked)
        if not spans:
            continue

        gold_map = _build_char_alignment(plain, target)
        pred_map = _build_char_alignment(plain, output)

        golds = _map_spans(spans, gold_map, target)
        preds = _map_spans(spans, pred_map, output)

        for (_, _, abbr_text), g, p in zip(spans, golds, preds):
            gold_all.append(g)
            pred_all.append(p)
            results.append(SpanResult(
                abbr_text=abbr_text,
                gold=g,
                predicted=p,
                exact=(p == g),
            ))

    print(f"[{_ts()}] compute_span_cer: span alignment done, "
          f"{len(pred_all)} spans found.")

    n_spans = len(gold_all)
    if n_spans == 0:
        print(f"[{_ts()}] compute_span_cer: no spans found, returning early")
        return {
            "span_cer":         None,
            "full_line_cer":    None,
            "span_exact_match": None,
            "n_spans":          0,
            "n_exact":          0,
            "by_abbr_type":     {},
        }

    n_exact = sum(r.exact for r in results)

    print(f"[{_ts()}] compute_span_cer: exact match computed, "
          f"starting span CER over {n_spans} spans ...")
    try:
        span_cer = extract_cer(
            get_cer_metric().compute(predictions=pred_all, references=gold_all)
        )
    except ZeroDivisionError:
        span_cer = 0.0

    print(f"[{_ts()}] compute_span_cer: span CER done ({span_cer:.4f}). "
          f"Starting full-line CER over {len(model_outputs)} lines...")
    try:
        full_line_cer = extract_cer(
            get_cer_metric().compute(predictions=model_outputs, references=target_corrs)
        )
    except ZeroDivisionError:
        full_line_cer = 0.0

    print(f"get_cer_metric().compute() done. Span CER: {span_cer:.4f}, "
          f"Full Line CER: {full_line_cer:.4f}, "
          f"Exact Match: {n_exact}/{n_spans} ({n_exact/n_spans:.2%})")

    return {
        "span_cer":         span_cer,
        "full_line_cer":    full_line_cer,
        "span_exact_match": n_exact / n_spans,
        "n_spans":          n_spans,
        "n_exact":          n_exact,
        "by_abbr_type":     build_type_breakdown(results) if include_breakdown else {},
        "source_cap":       max_source_lines if max_source_lines is not None else "None",
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
        expansions:  Counter of gold expansions seen for this abbr_text
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

        try:
            type_cer: float = extract_cer(
                get_cer_metric().compute(predictions=preds, references=golds)
            )
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
