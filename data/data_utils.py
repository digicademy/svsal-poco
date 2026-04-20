# data_utils.py
#
# Shared data loading, sorting, and example construction utilities
# used by both the boundary classifier and ByT5 abbreviation expansion pipelines.

import json
import random
from collections import defaultdict
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ABBR_OPEN  = "⦃"   # U+2983 — wraps abbreviated span in source_sic
ABBR_CLOSE = "⦄"   # U+2984
LINE_SEP   = "↵"   # U+21B5 — marks concatenated nonbreaking line boundary
LANG_TOKENS = {"la": "[LA]", "es": "[ES]", "default": "[LA]"}


# ---------------------------------------------------------------------------
# Line id parsing and sorting
# ---------------------------------------------------------------------------

def parse_line_id(line_id: str) -> tuple:
    """
    Parse a full line id into a sort key for correct document ordering.

    Syntax: {work_id}-{volume}-{page}-lb-{sequence}
    e.g.:   W0011-00-0006-lb-2027  → ("W0011", 0, 6, False, 2027)
            W0011-00-0006-lb-m016  → ("W0011", 0, 6, True, 16)

    Volume and page are parsed as integers for correct numeric ordering.
    Marginal note lines (m-prefixed sequence) sort after main text lines
    within the same page.
    """
    prefix, sequence = line_id.split("-lb-")
    parts  = prefix.split("-")
    page   = int(parts[-1])
    volume = int(parts[-2])
    work   = "-".join(parts[:-2])
    is_marginal = sequence.startswith("m")
    seq_num     = int(sequence[1:]) if is_marginal else int(sequence)
    return (work, volume, page, is_marginal, seq_num)


def crosses_page_break(id_a: str, id_b: str) -> bool:
    """
    Returns True if two line ids are on different pages or volumes.
    Used to flag cross-page nonbreaking boundaries in the classifier.
    """
    key_a = parse_line_id(id_a)
    key_b = parse_line_id(id_b)
    return key_a[:3] != key_b[:3]


def load_and_sort_lines(path: str) -> list[dict]:
    """
    Load a JSONL export file and sort rows into correct document order
    using the structured line id field.
    """
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: parse_line_id(r["id"]))
    return rows


# ---------------------------------------------------------------------------
# Document-level train/val/test split
# ---------------------------------------------------------------------------

def document_split(
    examples:  list[dict],
    test_frac: float = 0.1,
    val_frac:  float = 0.1,
    seed:      int   = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Split examples by document id, so no document appears in more than
    one split. This is mandatory to avoid data leakage from sliding
    window concatenation and shared compositor conventions.

    Returns (train, val, test).
    """
    doc_ids = list({e["doc_id"] for e in examples})
    random.seed(seed)
    random.shuffle(doc_ids)

    n_test = max(1, int(len(doc_ids) * test_frac))
    n_val  = max(1, int(len(doc_ids) * val_frac))

    test_docs  = set(doc_ids[:n_test])
    val_docs   = set(doc_ids[n_test:n_test + n_val])
    train_docs = set(doc_ids[n_test + n_val:])

    return (
        [e for e in examples if e["doc_id"] in train_docs],
        [e for e in examples if e["doc_id"] in val_docs],
        [e for e in examples if e["doc_id"] in test_docs],
    )


# ---------------------------------------------------------------------------
# ByT5 example construction
# ---------------------------------------------------------------------------

def build_byt5_examples(
    lines:           list[dict],
    oversample_abbr: float = 2.0,
    lang_prefix:     bool  = False,
    seed:            int   = 42,
    marker_dropout: float = 0.5,
) -> list[dict]:
    """
    Construct (source, target) training pairs for ByT5.

    - Concatenates nonbreaking line pairs with LINE_SEP
    - Oversamples lines containing abbreviations by oversample_abbr factor
    - Optionally prepends a language token ([LA] or [ES]) to the source
    - Retains doc_id and lang for splitting and stratified evaluation

    Source lines must already have abbreviation spans wrapped in
    ABBR_OPEN / ABBR_CLOSE delimiters (inserted during TEI export).
    Target lines (target_corr) are used as-is, without delimiters.
    """
    index    = {row["id"]: row for row in lines}
    consumed = set()
    examples = []

    for row in lines:
        if row["id"] in consumed:
            continue

        next_id  = row.get("nonbreaking_next_line", "")
        next_row = index.get(next_id) if next_id else None

        if next_row:
            source   = row["source_sic"] + LINE_SEP + next_row["source_sic"]
            target   = row["target_corr"] + LINE_SEP + next_row["target_corr"]
            has_abbr = (row["contains_abbr"] == "true"
                        or next_row["contains_abbr"] == "true")
            lang     = row["lang"]
            consumed.add(next_row["id"])
        else:
            source   = row["source_sic"]
            target   = row["target_corr"]
            has_abbr = row["contains_abbr"] == "true"
            lang     = row["lang"]

        # Randomly strip markers to teach marker-free detection
        if has_abbr and marker_dropout > 0 and random.random() < marker_dropout:
            source = source.replace(ABBR_OPEN, "").replace(ABBR_CLOSE, "")

        if lang_prefix:
            lang_tag = LANG_TOKENS.get(
                lang[0] if lang else "la", LANG_TOKENS["default"]
            )
            source = lang_tag + " " + source

        examples.append({
            "source":   source,
            "target":   target,
            "has_abbr": has_abbr,
            "doc_id":   row["doc_id"],
            "lang":     lang[0] if lang else "la",
        })

    # Oversample abbreviation-containing examples
    if oversample_abbr > 1.0:
        rng     = random.Random(seed)
        abbr_ex = [e for e in examples if e["has_abbr"]]
        n_extra = int(len(abbr_ex) * (oversample_abbr - 1.0))
        examples += rng.choices(abbr_ex, k=n_extra)
        rng.shuffle(examples)

    return examples


# ---------------------------------------------------------------------------
# Boundary classifier example construction
# ---------------------------------------------------------------------------

from dataclasses import dataclass


@dataclass
class BoundaryExample:
    """
    A single line boundary instance for the boundary classifier.

    line_end:           last ~40 characters of line N (source_sic)
    line_start:         first ~40 characters of line N+1 (source_sic)
    label:              1 = nonbreaking (word continues), 0 = genuine break
    doc_id:             for document-level splitting
    lang:               language tag list
    boundary_id:        id of line N, for traceability
    crosses_page_break: True if the boundary spans a page boundary
    """
    line_end:           str
    line_start:         str
    label:              int
    doc_id:             str
    lang:               list
    boundary_id:        str
    crosses_page_break: bool


def build_boundary_examples(
    lines:         list[dict],
    context_chars: int = 40,
) -> list[BoundaryExample]:
    """
    Construct BoundaryExample instances from sorted line rows.

    Positive examples: wherever nonbreaking_next_line is set.
    Negative examples: consecutive line pairs within the same ancestor
                       paragraph that are NOT nonbreaking boundaries.

    Cross-paragraph and cross-document boundaries are excluded from
    negatives to avoid trivially easy negative examples.
    """
    index              = {row["id"]: row for row in lines}
    examples           = []
    consumed_positives = set()

    # --- Positive examples ---
    for row in lines:
        next_id = row.get("nonbreaking_next_line", "")
        if not next_id:
            continue
        next_row = index.get(next_id)
        if not next_row:
            continue

        examples.append(BoundaryExample(
            line_end            = row["source_sic"][-context_chars:],
            line_start          = next_row["source_sic"][:context_chars],
            label               = 1,
            doc_id              = row["doc_id"],
            lang                = row["lang"],
            boundary_id         = row["id"],
            crosses_page_break  = crosses_page_break(row["id"], next_id),
        ))
        consumed_positives.add((row["id"], next_id))

    # --- Negative examples ---
    by_para: dict = defaultdict(list)
    for row in lines:
        key = (row["doc_id"], row.get("ancestor_id", ""))
        by_para[key].append(row)

    for para_lines in by_para.values():
        for i in range(len(para_lines) - 1):
            row      = para_lines[i]
            next_row = para_lines[i + 1]
            pair_key = (row["id"], next_row["id"])

            if pair_key in consumed_positives:
                continue
            if row.get("nonbreaking_next_line", ""):
                continue

            examples.append(BoundaryExample(
                line_end            = row["source_sic"][-context_chars:],
                line_start          = next_row["source_sic"][:context_chars],
                label               = 0,
                doc_id              = row["doc_id"],
                lang                = row["lang"],
                boundary_id         = row["id"],
                crosses_page_break  = crosses_page_break(row["id"], next_row["id"]),
            ))

    return examples


def document_split_boundary(
    examples:  list[BoundaryExample],
    test_frac: float = 0.1,
    val_frac:  float = 0.1,
    seed:      int   = 42,
) -> tuple[list[BoundaryExample], list[BoundaryExample], list[BoundaryExample]]:
    doc_ids = list({e.doc_id for e in examples})
    random.seed(seed)
    random.shuffle(doc_ids)

    n_test = max(1, int(len(doc_ids) * test_frac))
    n_val  = max(1, int(len(doc_ids) * val_frac))

    test_docs  = set(doc_ids[:n_test])
    val_docs   = set(doc_ids[n_test:n_test + n_val])
    train_docs = set(doc_ids[n_test + n_val:])

    return (
        [e for e in examples if e.doc_id in test_docs],
        [e for e in examples if e.doc_id in val_docs],
        [e for e in examples if e.doc_id in train_docs],
    )

# ---------------------------------------------------------------------------
# Optional: corpus lexicon for boundary classifier feature
# ---------------------------------------------------------------------------

from collections import Counter


class CorpusLexicon:
    """
    Word list constructed from target_corr fields of the training data.
    Used as an optional single-bit feature for the boundary classifier:
    if concatenating the last token of line N with the first token of
    line N+1 produces a known word, that is evidence for a nonbreaking
    boundary.
    """
    def __init__(self):
        self.words: set = set()

    def build_from_jsonl(self, path: str, min_count: int = 3):
        counts: Counter = Counter()
        with open(path) as f:
            for line in f:
                row  = json.loads(line)
                text = row.get("target_corr", "")
                for word in text.split():
                    word = word.strip(".,;:!?()[]\"'")
                    if len(word) > 2:
                        counts[word.lower()] += 1
        self.words = {w for w, c in counts.items() if c >= min_count}

    def is_known_word(self, candidate: str) -> bool:
        return candidate.lower().strip(".,;:!?()[]\"'") in self.words

    def concatenation_is_known(self, line_end: str, line_start: str) -> bool:
        last_token  = line_end.split()[-1]  if line_end.split()  else ""
        first_token = line_start.split()[0] if line_start.split() else ""
        return self.is_known_word(last_token + first_token)
