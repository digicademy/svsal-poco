# tei_roundtrip.py
#
# TEI XML ↔ plaintext roundtripping for the abbreviation expansion pipeline.
#
# Handles:
# - Extracting plain text lines from TEI XML (stripping inline tags, separating notes)
# - Feeding lines through the boundary + expansion pipeline
# - Diffing original vs expanded text to locate abbreviation changes
# - Wrapping changes in <choice><abbr>...</abbr><expan>...</expan></choice>
# - Moving notes that fall inside expanded abbreviations to after </choice>
# - Updating <lb/> with @break="no" from boundary classifier predictions
# - Preserving all existing inline markup and existing <choice> elements
#
# Design:
#   Uses lxml for XML parsing and tree manipulation. Text extraction walks the
#   tree depth-first, recording a "text run" list that maps plain-text offsets
#   back to tree positions. After expansion, diffs identify changed ranges,
#   which are applied back to the tree via direct node manipulation.

from __future__ import annotations

import copy
import re
import unicodedata
from datetime import date
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Optional

from lxml import etree

TEI_NS = "http://www.tei-c.org/ns/1.0"
XML_NS = "http://www.w3.org/XML/1998/namespace"
NSMAP  = {"tei": TEI_NS, "xml": XML_NS}
LINE_SEP = "¬"   # U+00AC — nonbreaking line separator in concatenated text

# Model identifiers for resp attributes on auto-generated elements.
# Set these before calling process_tei_xml, or pass model names to it.
EXPANSION_MODEL = "byt5-salamanca-abbr"
BOUNDARY_MODEL  = "flair-lb-detector"


def _expansion_resp() -> str:
    """Build resp attribute value for expansion-generated elements."""
    return f"#auto #{EXPANSION_MODEL}"


def _boundary_resp() -> str:
    """Build resp attribute value for boundary-generated elements."""
    return f"#auto #{BOUNDARY_MODEL}"

# Punctuation that marks word boundaries (not part of abbreviation tokens)
WORD_BREAK_PUNCT = set('.,:!?()[]')


def _is_punctuation_only_change(orig: str, expanded: str) -> bool:
    """
    Return True if the only difference between orig and expanded is
    whitespace or punctuation — not a real abbreviation expansion.
    Filters out model artifacts like 'quia\\n' → 'quia,'.
    """
    # Extract just the letters from both
    orig_letters = "".join(c for c in orig if c.isalpha())
    exp_letters = "".join(c for c in expanded if c.isalpha())
    return orig_letters == exp_letters


def _is_glyph_variant_only_change(orig: str, expanded: str) -> bool:
    """
    Return True if the only differences between orig and expanded are
    glyph variants (ſ/s, u/v, è/e, etc.) or combining-mark stripping —
    i.e. NOT real abbreviation expansions.

    This prevents the model's character normalization (e.g. long-s → s,
    accented vowels → plain) from being treated as abbreviation changes.
    """
    import unicodedata as _ud

    # Canonical mapping: for each equivalence pair, pick one representative
    _CANON = {}
    for a, b in EQUIV.items():
        # Use the one that sorts lower as canonical
        canon = min(a, b)
        _CANON[a] = canon
        _CANON[b] = canon

    def _normalize(s: str) -> str:
        """Normalize to canonical form: strip combining marks, map equivalents."""
        out: list[str] = []
        for ch in _ud.normalize("NFD", s):
            if _ud.category(ch).startswith("M"):
                continue  # skip combining marks
            out.append(_CANON.get(ch, ch))
        return "".join(out)

    return _normalize(orig) == _normalize(expanded)
EQUIV = {
    'u': 'v', 'v': 'u',
    'U': 'V', 'V': 'U',
    's': 'ſ', 'ſ': 's',
    'y': '⁊', '⁊': 'y',
    'z': 'ʒ', 'ʒ': 'z',
#    'ꝗ': 'q', 'q': 'ꝗ',
}


def chars_match(a, b):
    return a == b or EQUIV.get(a) == b


def _detect_namespace(root: etree._Element) -> str:
    """
    Detect whether the document uses the TEI namespace.
    Returns the namespace prefix for Clark notation, or "" if no namespace.
    """
    tag = root.tag
    if "{" in tag:
        ns = tag[1:tag.index("}")]
        if "tei-c.org" in ns:
            return f"{{{ns}}}"
    # Check children too — fragment might start with <p> but contain
    # namespaced children
    for el in root.iter():
        if "{" in el.tag and "tei-c.org" in el.tag:
            ns = el.tag[1:el.tag.index("}")]
            return f"{{{ns}}}"
    return ""


def _make_tag_fn(ns_prefix: str):
    """Return a function that builds tag names with the detected namespace."""
    def tag(name: str) -> str:
        return f"{ns_prefix}{name}"
    return tag


def _xml_id(el: etree._Element) -> Optional[str]:
    """Get the xml:id attribute of an element."""
    return el.get(f"{{{XML_NS}}}id")

# Module-level tag function — set by process_tei_xml before use.
# Default assumes TEI namespace; overridden for non-namespaced XML.
_tag = _make_tag_fn(f"{{{TEI_NS}}}")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TextRun:
    """
    A contiguous piece of text extracted from the XML tree.

    Records enough context to map plain-text offsets back to exact tree
    positions for modification.
    """
    text:         str                # the text content
    node:         etree._Element     # the element this text belongs to
    is_tail:      bool               # True = node.tail; False = node.text
    plain_start:  int = 0            # offset in the stitched plain text
    plain_end:    int = 0
    from_choice:  bool = False


@dataclass
class NoteInfo:
    """
    A <note> element that was extracted from a main-text line.

    The note's content is processed as separate lines. During reinsertion,
    if the note falls inside an abbreviation that gets wrapped in <choice>,
    the note is moved to after </choice>.
    """
    element:        etree._Element   # the <note> element
    plain_offset:   int              # offset in the stitched main-text line
    parent:         etree._Element   # original parent of the note
    parent_index:   int              # index within parent's children


@dataclass
class ExtractedLine:
    """
    A line extracted from TEI XML, ready for pipeline processing.

    Represents the text between two consecutive <lb/> elements (or between
    an <lb/> and the end of its containing block).
    """
    line_id:       str               # xml:id of the <lb/>
    lb_element:    etree._Element    # the <lb/> element itself
    plain_text:    str               # plain text (tags stripped, notes removed)
    text_runs:     list[TextRun]     # ordered text fragments with tree positions
    notes:         list[NoteInfo]    # notes extracted from this line
    is_in_note:    bool              # whether this lb is inside a <note>
    lang:          list[str]         # language(s) from @xml:lang ancestry


# ---------------------------------------------------------------------------
# Line extraction
# ---------------------------------------------------------------------------

def extract_lines(tree: etree._ElementTree) -> tuple[list[ExtractedLine], dict[str, str]]:
    """
    Extract all lines and pre-annotated boundaries from a TEI XML tree.

    Walks the tree to find all <lb/> elements. For each lb, collects
    the text content between it and the next lb, handling:
    - Notes: extracted as separate entries, main text stitched around them
    - Inline tags: stripped for plain text, but tree positions recorded
    - Existing <choice> elements: their <abbr> text is used as source
    - Note-initial text: text inside a <note> before its first <lb/>

    Returns a tuple of a flat list of ExtractedLine objects and
    pre_annotated boundaries. Main-text lines and note lines are
    interleaved in document order, distinguished by is_in_note.
    """
    root = tree.getroot()
    pre_annotated = {}

    # Find all <lb/> elements in document order
    all_lbs = root.iter(_tag("lb"))
    lb_list = [lb for lb in all_lbs if lb.get("sameAs") is None]

    if not lb_list:
        return [], pre_annotated

    # Find notes with text before their first internal <lb/>.
    # These need synthetic line entries since no lb triggers extraction.
    note_initial_lines = _extract_note_initial_lines(root, lb_list)

    # Build lb-based lines
    lb_lines: list[ExtractedLine] = []

    for i, lb in enumerate(lb_list):
        next_lb = lb_list[i + 1] if i + 1 < len(lb_list) else None
        is_in_note = _is_inside_note(lb)
        line_id = _xml_id(lb) or f"__lb_{i}"

        # Record pre-annotated boundary if this lb has break="no".
        # break="no" on THIS lb means the PREVIOUS line continues here.
        if lb.get("break") == "no" and i > 0:
            prev_lb = lb_list[i - 1]
            prev_id = _xml_id(prev_lb) or f"__lb_{i - 1}"
            pre_annotated[prev_id] = line_id

        # Collect text runs between this lb and the next
        text_runs: list[TextRun] = []
        notes: list[NoteInfo] = []

        _collect_text_after_lb(
            lb, next_lb, is_in_note,
            text_runs, notes,
        )

        # Build plain text from runs, recording offsets
        plain_parts: list[str] = []
        offset = 0
        for run in text_runs:
            run.plain_start = offset
            run.plain_end = offset + len(run.text)
            plain_parts.append(run.text)
            offset = run.plain_end

        plain_text = "".join(plain_parts)

        # Determine language from xml:lang ancestry
        lang = _get_languages(lb, text_runs)

        lb_lines.append(ExtractedLine(
            line_id=line_id,
            lb_element=lb,
            plain_text=plain_text,
            text_runs=text_runs,
            notes=notes,
            is_in_note=is_in_note,
            lang=lang,
        ))

    # Merge note-initial lines into the lb-based list in document order.
    # Each note-initial line should appear just before the first lb line
    # inside that note.
    lines = _merge_note_initial_lines(lb_lines, note_initial_lines, lb_list)

    return lines, pre_annotated


def _extract_note_initial_lines(
    root:    etree._Element,
    lb_list: list[etree._Element],
) -> list[ExtractedLine]:
    """
    Find <note> elements that have text content before their first
    internal <lb/>. Creates synthetic ExtractedLine entries for this text.
    """
    lb_set = set(id(lb) for lb in lb_list)
    result: list[ExtractedLine] = []

    for note in root.iter(_tag("note")):
        # Find the first <lb/> inside this note
        first_lb = None
        for child_lb in note.iter(_tag("lb")):
            if id(child_lb) in lb_set:
                first_lb = child_lb
                break

        # Collect text before the first lb (or all text if no lb)
        text_runs: list[TextRun] = []
        _collect_note_initial_text(note, first_lb, text_runs)

        if not text_runs:
            continue

        # Build plain text
        plain_parts: list[str] = []
        offset = 0
        for run in text_runs:
            run.plain_start = offset
            run.plain_end = offset + len(run.text)
            plain_parts.append(run.text)
            offset = run.plain_end

        plain_text = "".join(plain_parts)
        if not plain_text.strip():
            continue

        note_id = _xml_id(note) or f"__note_{id(note)}"
        lang = _get_languages(note, text_runs)

        result.append(ExtractedLine(
            line_id=f"{note_id}_initial",
            lb_element=note,   # use note element as anchor (no real lb)
            plain_text=plain_text,
            text_runs=text_runs,
            notes=[],
            is_in_note=True,
            lang=lang,
        ))

    return result


def _collect_note_initial_text(
    note:     etree._Element,
    first_lb: Optional[etree._Element],
    text_runs: list[TextRun],
) -> None:
    """
    Collect text inside a <note> element that appears before first_lb.
    If first_lb is None, collects all text in the note.
    """
    # Note's own text
    if note.text:
        text_runs.append(TextRun(
            text=note.text, node=note, is_tail=False,
        ))

    # Walk children until we hit first_lb
    for child in note:
        if first_lb is not None and child is first_lb:
            return
        if first_lb is not None and _is_descendant_of(first_lb, child):
            # first_lb is inside this child — descend partially
            _collect_note_initial_text_recursive(child, first_lb, text_runs)
            return

        # Collect all text from this child
        for text in child.itertext():
            if text:
                text_runs.append(TextRun(
                    text=text, node=child, is_tail=False,
                ))
        if child.tail:
            text_runs.append(TextRun(
                text=child.tail, node=child, is_tail=True,
            ))


def _collect_note_initial_text_recursive(
    el:        etree._Element,
    first_lb:  etree._Element,
    text_runs: list[TextRun],
) -> None:
    """Recurse into an element, stopping at first_lb."""
    if el.text:
        text_runs.append(TextRun(
            text=el.text, node=el, is_tail=False,
        ))
    for child in el:
        if child is first_lb:
            return
        if _is_descendant_of(first_lb, child):
            _collect_note_initial_text_recursive(child, first_lb, text_runs)
            return
        for text in child.itertext():
            if text:
                text_runs.append(TextRun(
                    text=text, node=child, is_tail=False,
                ))
        if child.tail:
            text_runs.append(TextRun(
                text=child.tail, node=child, is_tail=True,
            ))


def _merge_note_initial_lines(
    lb_lines:           list[ExtractedLine],
    note_initial_lines: list[ExtractedLine],
    lb_list:            list[etree._Element],
) -> list[ExtractedLine]:
    """
    Merge note-initial lines into the lb-based line list.

    Each note-initial line is inserted just before the first lb line
    that is inside the same note. If the note has no internal lb,
    the note-initial line is inserted at the position where the note
    appears in document order relative to the lb lines.
    """
    if not note_initial_lines:
        return lb_lines

    # Build a map: note element → its initial line
    note_to_initial: dict[int, ExtractedLine] = {}
    for nil in note_initial_lines:
        note_el = nil.lb_element  # this is actually the note element
        note_to_initial[id(note_el)] = nil

    result: list[ExtractedLine] = []
    used_notes: set[int] = set()

    for lb_line in lb_lines:
        # Check if this lb is inside a note that has an initial line
        if lb_line.is_in_note:
            note_el = lb_line.lb_element
            # Walk up to find the note ancestor
            parent = note_el.getparent()
            while parent is not None:
                if parent.tag == _tag("note") and id(parent) in note_to_initial:
                    if id(parent) not in used_notes:
                        result.append(note_to_initial[id(parent)])
                        used_notes.add(id(parent))
                    break
                parent = parent.getparent()

        result.append(lb_line)

    # Append any note-initial lines whose notes had no internal lbs
    for nid, nil in note_to_initial.items():
        if nid not in used_notes:
            result.append(nil)

    return result


def _is_inside_note(el: etree._Element) -> bool:
    """Check whether an element is inside a <note>."""
    parent = el.getparent()
    while parent is not None:
        if parent.tag == _tag("note"):
            return True
        parent = parent.getparent()
    return False


def _is_descendant_of(el: etree._Element, ancestor: etree._Element) -> bool:
    """Check whether el is a descendant of ancestor."""
    parent = el.getparent()
    while parent is not None:
        if parent is ancestor:
            return True
        parent = parent.getparent()
    return False


def _find_enclosing_choice_abbr(
    lb: etree._Element,
) -> Optional[etree._Element]:
    """
    Return the enclosing <choice> element if lb is a descendant of a
    <choice>/<abbr> or <choice>/<sic> branch, else None.

    Used to detect lb elements that live inside a pre-existing cross-line
    choice so that text collection skips the <expan>/<corr> sibling.
    """
    parent = lb.getparent()
    while parent is not None:
        local = parent.tag.split("}")[-1] if "}" in parent.tag else parent.tag
        if local in ("abbr", "sic"):
            grandparent = parent.getparent()
            if grandparent is not None:
                gp_local = grandparent.tag.split("}")[-1] if "}" in grandparent.tag else grandparent.tag
                if gp_local == "choice":
                    return grandparent
        # Stop climbing at block-level boundaries
        if local in ("p", "ab", "div", "body", "text", "TEI", "item", "list"):
            break
        parent = parent.getparent()
    return None


def _collect_until_lb(
    el:    etree._Element,
    lb:    etree._Element,
    parts: list[str],
) -> bool:
    """
    Depth-first collect text from el's subtree until lb is encountered.
    Returns True if lb was found (caller should stop collecting).
    """
    if el.text:
        parts.append(el.text)
    for child in el:
        if child is lb:
            return True
        if _collect_until_lb(child, lb, parts):
            return True
        if child.tail:
            parts.append(child.tail)
    return False


def _text_before_lb_in_el(el: etree._Element, lb: etree._Element) -> str:
    """
    Return text content of el (and its descendants) that appears before lb
    in document order.  lb must be a descendant of el.
    """
    parts: list[str] = []
    _collect_until_lb(el, lb, parts)
    return "".join(parts)


def _get_languages(lb: etree._Element, text_runs: list[TextRun]) -> list[str]:
    """Collect distinct xml:lang values from lb ancestry and text runs."""
    langs: list[str] = []
    seen: set[str] = set()

    # From lb's own ancestry
    el = lb
    while el is not None:
        lang = el.get(f"{{{XML_NS}}}lang")
        if lang and lang not in seen:
            langs.append(lang)
            seen.add(lang)
            break
        el = el.getparent()

    # From text run nodes' ancestry
    for run in text_runs:
        el = run.node
        while el is not None:
            lang = el.get(f"{{{XML_NS}}}lang")
            if lang and lang not in seen:
                langs.append(lang)
                seen.add(lang)
                break
            el = el.getparent()

    return langs or ["la"]


def _element_precedes(a: etree._Element, b: etree._Element) -> bool:
    """
    Check if element a comes before element b in document order.
    Uses the tree's iter() ordering.
    """
    root = a.getroottree().getroot()
    for el in root.iter():
        if el is a:
            return True
        if el is b:
            return False
    return False


def _collect_text_after_lb(
    lb:           etree._Element,
    next_lb:      Optional[etree._Element],
    is_in_note:   bool,
    text_runs:    list[TextRun],
    notes:        list[NoteInfo],
) -> None:
    """
    Collect text content between lb and next_lb.

    For main-text lines (not in a note):
      - Skips text inside <note> elements (records them in notes list)
      - Skips text inside existing <choice> elements' <expan>/<corr> branches
        (uses <abbr>/<sic> text instead — we want the original form)
    For note lines:
      - Scopes to the note's own content

    Special case: if lb sits inside a pre-existing <choice>/<abbr> (a cross-line
    abbreviation that was already annotated by the rule-based system), only the
    text after lb within the same <abbr>/<sic> is collected (marked from_choice),
    then the walk resumes after the enclosing <choice> element.  This prevents
    accidentally collecting the <expan>/<corr> branch as plain text.
    """
    # --- Special case: lb is inside an existing <choice>/<abbr> ---
    choice_el = _find_enclosing_choice_abbr(lb)
    if choice_el is not None:
        # Collect lb.tail and its siblings within <abbr>/<sic>, marking them
        # from_choice so that _apply_cross_line_choice can detect the overlap
        # and merge model resp with the existing <expan> instead of building
        # a new nested <choice>.
        if lb.tail:
            text_runs.append(TextRun(
                text=lb.tail, node=choice_el, is_tail=False, from_choice=True,
            ))
        for sib in lb.itersiblings():
            if next_lb is not None and sib is next_lb:
                return
            sib_text = _inner_text(sib)
            if sib_text:
                text_runs.append(TextRun(
                    text=sib_text, node=choice_el, is_tail=False, from_choice=True,
                ))
            if sib.tail:
                text_runs.append(TextRun(
                    text=sib.tail, node=choice_el, is_tail=False, from_choice=True,
                ))
        # Jump past the <choice> element: collect its tail, then continue
        if choice_el.tail:
            text_runs.append(TextRun(
                text=choice_el.tail, node=choice_el, is_tail=True,
            ))
        _walk_after(choice_el, next_lb, is_in_note, text_runs, notes, 0)
        return

    # --- Normal case ---
    # Start from lb's tail text
    if lb.tail:
        text = lb.tail
        if text.strip() or text:
            text_runs.append(TextRun(
                text=text, node=lb, is_tail=True,
            ))

    # Walk siblings and their subtrees after lb
    _walk_after(lb, next_lb, is_in_note, text_runs, notes, 0)


def _walk_after(
    start:        etree._Element,
    next_lb:      Optional[etree._Element],
    is_in_note:   bool,
    text_runs:    list[TextRun],
    notes:        list[NoteInfo],
    plain_offset: int,
) -> None:
    """
    Walk the tree after start element, collecting text runs.

    Stops when next_lb is encountered. Handles note extraction
    and existing choice elements.
    """
    # Process siblings after start
    sibling = start.getnext()
    while sibling is not None:
        if next_lb is not None and sibling is next_lb:
            return
        if next_lb is not None and _is_descendant_of(next_lb, sibling):
            # next_lb is inside this sibling — descend but stop at next_lb
            _walk_into(sibling, next_lb, is_in_note, text_runs, notes)
            return

        _walk_into(sibling, next_lb, is_in_note, text_runs, notes)

        sibling = sibling.getnext()

    # If we haven't found next_lb among siblings, go up to parent
    # and continue with parent's next siblings
    parent = start.getparent()
    if parent is not None and parent.tag != _tag("body"):
        # Add parent's tail if it exists (text after </parent>)
        # Actually no — parent's tail belongs to the parent's parent context
        # We need to continue walking at the parent level
        _walk_after(parent, next_lb, is_in_note, text_runs, notes, 0)


def _walk_into(
    el:           etree._Element,
    next_lb:      Optional[etree._Element],
    is_in_note:   bool,
    text_runs:    list[TextRun],
    notes:        list[NoteInfo],
) -> None:
    """
    Walk into an element, collecting text. Handles special elements:
    - <note>: extract for separate processing (main-text lines only)
    - <choice>: use <abbr>/<sic> text
    - <lb/>: stop signal (handled by caller)
    - Other inline elements: recurse, collecting text
    """
    tag = el.tag

    # --- Note handling ---
    if tag == _tag("note") and not is_in_note:
        # Record the note's position and skip its content
        current_offset = sum(len(r.text) for r in text_runs)
        parent = el.getparent()
        idx = list(parent).index(el)
        notes.append(NoteInfo(
            element=el,
            plain_offset=current_offset,
            parent=parent,
            parent_index=idx,
        ))
        # The note's tail text belongs to the main line
        if el.tail:
            text_runs.append(TextRun(
                text=el.tail, node=el, is_tail=True,
            ))
        return

    # --- Existing <choice> handling ---
    if tag == _tag("choice"):
        # Use <abbr> or <sic> text (the original form)
        abbr = el.find(_tag("abbr"))
        sic = el.find(_tag("sic"))
        source_el = abbr if abbr is not None else sic
        if source_el is not None:
            if next_lb is not None and _is_descendant_of(next_lb, source_el):
                # The next line boundary is inside this choice's abbr/sic.
                # Collect only the text that precedes next_lb — the content
                # after next_lb belongs to the next line.  choice.tail is
                # also omitted: it comes after next_lb in document order.
                pre_lb = _text_before_lb_in_el(source_el, next_lb)
                if pre_lb:
                    text_runs.append(TextRun(
                        text=pre_lb, node=el, is_tail=False,
                        from_choice=True,
                    ))
                return
            # Collect text from the source branch
            source_text = _inner_text(source_el)
            if source_text:
                text_runs.append(TextRun(
                    text=source_text, node=el, is_tail=False,
                    from_choice=True,
                ))
        # Tail text after </choice>
        if el.tail:
            text_runs.append(TextRun(
                text=el.tail, node=el, is_tail=True,
            ))
        return

    # --- Self-closing elements (lb, pb, cb, etc.) ---
    if tag in (_tag("lb"), _tag("pb"), _tag("cb"), _tag("fw")):
        # These don't contribute text content
        if el.tail:
            text_runs.append(TextRun(
                text=el.tail, node=el, is_tail=True,
            ))
        return

    # --- Regular inline elements (hi, foreign, ref, term, etc.) ---
    # Recurse into children, collecting text
    if el.text:
        text_runs.append(TextRun(
            text=el.text, node=el, is_tail=False,
        ))

    for child in el:
        if next_lb is not None and child is next_lb:
            return
        if next_lb is not None and _is_descendant_of(next_lb, child):
            # next_lb lives inside this child's subtree.  Descend to collect the
            # text that precedes it, then STOP — everything in document order
            # after next_lb (including later sibling subtrees such as the next
            # <item>) belongs to the next line.  Without this return the walk
            # fell through to following siblings and bled the next item's text
            # into this line.
            _walk_into(child, next_lb, is_in_note, text_runs, notes)
            return
        _walk_into(child, next_lb, is_in_note, text_runs, notes)

    # Tail text after the closing tag
    if el.tail:
        text_runs.append(TextRun(
            text=el.tail, node=el, is_tail=True,
        ))


def _inner_text(el: etree._Element) -> str:
    """Get all text content of an element (like XPath string())."""
    return "".join(el.itertext())


# ---------------------------------------------------------------------------
# Applying expansion results back to the XML tree
# ---------------------------------------------------------------------------

def apply_expansions(
    tree:                 etree._ElementTree,
    lines:                list[ExtractedLine],
    expanded_texts:       dict[str, str],     # line_id → expanded plain text
    boundary_predictions: dict[str, str],     # line_id → next_line_id (nonbreaking)
    pre_annotated:        dict[str, str] | None = None,  # boundaries already in source XML
) -> etree._ElementTree:
    """
    Apply abbreviation expansions and boundary predictions back to the XML tree.

    For each line where the expanded text differs from the original:
    1. Diff to find changed character ranges
    2. For each change, wrap in <choice><abbr>orig</abbr><expan>expanded</expan></choice>
    3. If a note falls inside a changed range, move it to after </choice>
    4. Update <lb/> with @break="no" where boundary classifier detected nonbreaking
    5. Handle cross-line abbreviations with <lb sameAs> in <expan>

    Returns the modified tree (modified in-place).
    """
    pre_annotated = pre_annotated or {}

    # --- Apply boundary predictions ---
    line_by_id = {line.line_id: line for line in lines}
    for line in lines:
        if line.line_id in boundary_predictions:
            # Skip if this boundary already existed in the source XML
            if line.line_id in pre_annotated:
                continue
            next_line_id = boundary_predictions[line.line_id]
            next_line = line_by_id.get(next_line_id)
            if next_line is not None:
                next_line.lb_element.set("break", "no")
                next_line.lb_element.set("resp", _boundary_resp())

    # --- Build nonbreaking chains ---
    chains = _build_line_chains(lines, boundary_predictions)

    # --- Apply expansions per chain ---
    for chain in chains:
        if len(chain) == 1:
            expanded = expanded_texts.get(chain[0].line_id)
            if expanded is not None and expanded != chain[0].plain_text:
                _apply_line_expansion(chain[0], expanded)
        else:
            _apply_chain_expansion(chain, expanded_texts)

    return tree


def _build_line_chains(
    lines:                list[ExtractedLine],
    boundary_predictions: dict[str, str],
) -> list[list[ExtractedLine]]:
    """
    Group lines into nonbreaking chains based on boundary predictions.
    Each chain is a list of consecutive lines connected by nonbreaking boundaries.
    """
    line_by_id = {line.line_id: line for line in lines}
    consumed: set[str] = set()
    chains: list[list[ExtractedLine]] = []

    for line in lines:
        if line.line_id in consumed:
            continue

        chain = [line]
        consumed.add(line.line_id)
        current = line

        while True:
            next_id = boundary_predictions.get(current.line_id, "")
            if not next_id or next_id in consumed or next_id not in line_by_id:
                break
            next_line = line_by_id[next_id]
            chain.append(next_line)
            consumed.add(next_id)
            current = next_line

        chains.append(chain)

    return chains


def _content_end_before(text: str, pos: int) -> int:
    """Walk left from ``pos`` past trailing whitespace.

    Returns the index just after the last non-whitespace character before
    ``pos``.  Used to locate the real end of a line's content when the
    LINE_SEP that replaced an <lb/> is preceded by pretty-print whitespace.
    """
    while pos > 0 and text[pos - 1].isspace():
        pos -= 1
    return pos


def _content_start_after(text: str, pos: int) -> int:
    """Walk right from ``pos`` past leading whitespace.

    Returns the index of the first non-whitespace character at or after
    ``pos`` (clamped to ``len(text)``).  Mirror of :func:`_content_end_before`
    for the start of the line following a LINE_SEP.
    """
    n = len(text)
    while pos < n and text[pos].isspace():
        pos += 1
    return pos


def _merge_sep_adjacent_changes(
    changes:       list[tuple[int, int, int, int]],
    sep_positions: list[int],
    orig_concat:   str,
    exp_concat:    str,
) -> list[tuple[int, int, int, int]]:
    """
    Merge or extend changes that border a LINE_SEP so genuine cross-line
    abbreviations are routed through _apply_cross_line_choice.

    _expand_left/_expand_right (fixed in PR #10) stop at LINE_SEP to prevent
    spurious cross-line choices from glyph variants.  But a real cross-line
    abbreviation — e.g. pr⟨æ⟩[lb]st⟨ã⟩tes → praestantes — can produce two
    single-line changes that each touch the separator:

        L1: orig_end == sep_pos    (word ending at the line boundary)
        L2: orig_start == sep_pos+1 (word starting right after the boundary)

    This function detects such pairs (and lone L2-boundary changes) *after*
    glyph-variant filtering has already removed noise, then stitches them
    into a single cross-line change whose range spans the LINE_SEP so the
    caller dispatches it via _apply_cross_line_choice.
    """
    if not changes or not sep_positions:
        return changes

    # Locate LINE_SEP characters in exp_concat (one per chain junction,
    # matching sep_positions[k] in orig_concat one-to-one).
    exp_sep_positions: list[int] = [i for i, c in enumerate(exp_concat) if c == LINE_SEP]

    result = list(changes)
    removed: set[int] = set()

    for k, sep_pos in enumerate(sep_positions):
        if k >= len(exp_sep_positions):
            continue
        exp_sep_pos = exp_sep_positions[k]

        # An <lb/> in pretty-printed TEI is almost always surrounded by
        # whitespace (newlines, indentation) — e.g. "ſucce\n  <lb/>dãt" — so
        # the LINE_SEP that replaces it in the concatenated text rarely sits
        # flush against L1's last word or L2's first word.  Compute the
        # *content* boundaries (sep_pos minus L1 trailing whitespace, and
        # sep_pos+len(SEP) plus L2 leading whitespace) so that word-boundary
        # change edges still line up with the junction.  Without this, a
        # single trailing newline on L1 made the leftward extension below a
        # no-op and the cross-line abbreviation collapsed into an L2-only
        # single-line <choice>.
        l1_content_end = _content_end_before(orig_concat, sep_pos)
        l2_content_start = _content_start_after(orig_concat, sep_pos + len(LINE_SEP))

        # Change whose right edge is at L1's content boundary (last word).
        l1_idx = next(
            (i for i, (o1, o2, e1, e2) in enumerate(result)
             if i not in removed and o2 == l1_content_end),
            None,
        )
        # Change whose left edge starts at L2's content boundary (first word).
        l2_idx = next(
            (i for i, (o1, o2, e1, e2) in enumerate(result)
             if i not in removed and o1 == l2_content_start),
            None,
        )

        if l1_idx is not None and l2_idx is not None:
            # Both halves differ from the model — merge into one cross-line
            # change whose range includes the LINE_SEP.
            l1_o1, l1_o2, l1_e1, l1_e2 = result[l1_idx]
            _l2_o1, l2_o2, _l2_e1, l2_e2 = result[l2_idx]
            result[l1_idx] = (l1_o1, l2_o2, l1_e1, l2_e2)
            removed.add(l2_idx)

        elif l2_idx is not None:
            # Only the L2 part changed.  Extend the change leftward in both
            # orig_concat and exp_concat to include L1's last word, so
            # _apply_cross_line_choice can wrap the full cross-line token.
            _l2_o1, l2_o2, _l2_e1, l2_e2 = result[l2_idx]

            # Walk left from L1's content end (past any trailing whitespace)
            # to the start of its last word.
            o_word_start = l1_content_end
            while (o_word_start > 0
                   and not orig_concat[o_word_start - 1].isspace()
                   and orig_concat[o_word_start - 1] != LINE_SEP):
                o_word_start -= 1

            # Same on the exp side: skip trailing whitespace, then walk to the
            # matching word start (L1 expansions may shift absolute positions
            # in exp_concat).
            exp_l1_content_end = _content_end_before(exp_concat, exp_sep_pos)
            e_word_start = exp_l1_content_end
            while (e_word_start > 0
                   and not exp_concat[e_word_start - 1].isspace()
                   and exp_concat[e_word_start - 1] != LINE_SEP):
                e_word_start -= 1

            if o_word_start < l1_content_end:
                result[l2_idx] = (o_word_start, l2_o2, e_word_start, l2_e2)
        # If only L1 exists, leave it as a single-line change (correct as-is).

    return [c for i, c in enumerate(result) if i not in removed]


def _apply_chain_expansion(
    chain:          list[ExtractedLine],
    expanded_texts: dict[str, str],
) -> None:
    """
    Apply expansions to a nonbreaking chain of lines.

    Concatenates original and expanded texts with LINE_SEP, diffs the
    concatenated strings, and dispatches each change:
    - Changes within a single line → _apply_change_preserving_markup
    - Changes crossing a LINE_SEP → _apply_cross_line_choice
    """
    # Concatenate originals and expanded with LINE_SEP
    orig_parts = [line.plain_text for line in chain]
    exp_parts = [expanded_texts.get(line.line_id, line.plain_text) for line in chain]

    orig_concat = LINE_SEP.join(orig_parts)
    exp_concat = LINE_SEP.join(exp_parts)

    if orig_concat == exp_concat:
        return

    changes = _find_changes(orig_concat, exp_concat)
    if not changes:
        return

    # Compute line boundaries and separator positions in concatenated string
    line_boundaries: list[tuple[int, int]] = []
    sep_positions: list[int] = []  # positions of LINE_SEP characters
    offset = 0
    for i, part in enumerate(orig_parts):
        line_boundaries.append((offset, offset + len(part)))
        offset += len(part)
        if i < len(orig_parts) - 1:
            sep_positions.append(offset)
            offset += len(LINE_SEP)

    # Filter out punctuation-only and glyph-variant-only changes, then
    # merge boundary-adjacent pairs into cross-line changes so that
    # abbreviations spanning a non-breaking line break are dispatched as
    # a single _apply_cross_line_choice call rather than two single-line
    # changes.  This must happen *after* individual glyph-variant filtering
    # so that e.g. "cleſia"→"clesia" at the start of L2 (glyph-variant for
    # ſ→s) is removed before any merging step considers it.
    real_changes: list[tuple[int, int, int, int]] = []
    for o1, o2, e1, e2 in changes:
        ot = orig_concat[o1:o2]
        et = exp_concat[e1:e2]
        if not ot or not et:
            continue
        if _is_punctuation_only_change(ot, et):
            continue
        if _is_glyph_variant_only_change(ot, et):
            continue
        real_changes.append((o1, o2, e1, e2))

    if sep_positions:
        real_changes = _merge_sep_adjacent_changes(
            real_changes, sep_positions, orig_concat, exp_concat,
        )

    # Process changes in reverse order (right-to-left) so tree offsets
    # remain valid as earlier modifications are applied.
    for orig_start, orig_end, exp_start, exp_end in reversed(real_changes):
        orig_text = orig_concat[orig_start:orig_end]
        exp_text = exp_concat[exp_start:exp_end]

        # A TEI <abbr> wraps a single token.  Refuse multi-token abbr spans
        # (cross-line LINE_SEP spans are allowed — see _abbr_is_single_token).
        if not _abbr_is_single_token(orig_text):
            continue

        # Check if change spans a separator → cross-line
        crossed_seps = [s for s in sep_positions if orig_start <= s < orig_end]

        if not crossed_seps:
            # Change is within a single line — find which one
            line_idx = _find_line_for_offset(line_boundaries, orig_start)
            line = chain[line_idx]
            lb_start, _ = line_boundaries[line_idx]
            local_start = orig_start - lb_start
            local_end = orig_end - lb_start

            # Check notes
            affected_notes = [
                n for n in line.notes
                if local_start <= n.plain_offset < local_end
            ]

            choice = _apply_change_preserving_markup(
                line, local_start, local_end, exp_text,
            )

            if choice is not None:
                for note_info in affected_notes:
                    _move_note_after(note_info, choice)

        else:
            # Change crosses line boundaries — cross-line abbreviation
            # Find which lines are involved
            first_sep = crossed_seps[0]
            start_line_idx = _find_line_for_offset(line_boundaries, first_sep - 1)
            end_line_idx = _find_line_for_offset(line_boundaries, orig_end - 1)
            # Clamp: if orig_end lands on a separator, use the line before
            if end_line_idx < 0:
                end_line_idx = len(chain) - 1

            _apply_cross_line_choice(
                chain, line_boundaries,
                start_line_idx, end_line_idx,
                orig_start, orig_end,
                orig_text, exp_text,
            )


def _find_line_for_offset(
    line_boundaries: list[tuple[int, int]],
    offset: int,
) -> int:
    """Find which line index contains the given offset in concatenated text."""
    for i, (start, end) in enumerate(line_boundaries):
        if start <= offset < end:
            return i
    # If offset is at the very end, return last line
    return len(line_boundaries) - 1


def _find_tgt_break(full_src, full_tgt, src_break):
    def _is_abbr_or_combining(c):
        cat = unicodedata.category(c)
        if cat.startswith('M'):
            return True
        if cat == 'Co':
            return True
        if not c.isascii() and cat.startswith('L'):
            return True
        return False

    def _is_plain(c):
        return not _is_abbr_or_combining(c)

    def _align(src, tgt, brk):
        if len(src) == 0:
            return 0
        if brk <= 0:
            return 0
        if brk >= len(src):
            return len(tgt)

        prefix = 0
        for i in range(min(len(src), len(tgt))):
            if chars_match(src[i], tgt[i]):
                prefix = i + 1
            else:
                break

        if brk <= prefix:
            return brk

        suffix = 0
        for i in range(1, min(len(src), len(tgt)) + 1):
            if chars_match(src[-i], tgt[-i]):
                suffix = i
            else:
                break

        if suffix > 0 and brk >= len(src) - suffix:
            return len(tgt) - (len(src) - brk)

        s_end = len(src) - suffix if suffix else len(src)
        t_end = len(tgt) - suffix if suffix else len(tgt)
        src_gap = src[prefix:s_end]
        tgt_gap = tgt[prefix:t_end]
        gap_brk = brk - prefix

        if not src_gap or not tgt_gap:
            return prefix

        for si in range(len(src_gap)):
            if not _is_plain(src_gap[si]):
                continue
            for ti in range(len(tgt_gap)):
                if not chars_match(src_gap[si], tgt_gap[ti]):
                    continue
                next_si = si + 1
                while next_si < len(src_gap) and not _is_plain(src_gap[next_si]):
                    next_si += 1
                if next_si < len(src_gap) and ti + 1 + (next_si - si - 1) < len(tgt_gap):
                    confirmed = False
                    for ti2 in range(ti + 1, min(ti + 1 + (next_si - si) * 3, len(tgt_gap))):
                        if chars_match(src_gap[next_si], tgt_gap[ti2]):
                            confirmed = True
                            break
                    if not confirmed:
                        continue
                if gap_brk <= si:
                    sub = _align(src_gap[:si], tgt_gap[:ti], gap_brk)
                    if sub is None:
                        return None
                    return prefix + sub
                else:
                    sub = _align(src_gap[si:], tgt_gap[ti:], gap_brk - si)
                    if sub is None:
                        return None
                    return prefix + ti + sub
        return None

    return _align(full_src, full_tgt, src_break)


def _find_tgt_break_with_fallback(full_src, full_tgt, src_break):
    tgt_break = _find_tgt_break(full_src, full_tgt, src_break)

    if tgt_break is not None:
        return tgt_break

    suffix_len = 0
    for sc, tc in zip(reversed(full_src), reversed(full_tgt)):
        if sc == tc and sc.isascii() and sc.isalpha():
            suffix_len += 1
        else:
            break

    if suffix_len == 0:
        return None

    chars_after_break = len(full_src) - src_break
    tgt_break = len(full_tgt) - chars_after_break

    if 0 < tgt_break < len(full_tgt):
        return tgt_break

    return None


def _apply_cross_line_choice(
    chain:           list[ExtractedLine],
    line_boundaries: list[tuple[int, int]],
    start_line_idx:  int,
    end_line_idx:    int,
    orig_start:      int,
    orig_end:        int,
    orig_text:       str,
    exp_text:        str,
) -> None:
    """
    Build a cross-line <choice> element for an abbreviation spanning
    a nonbreaking line boundary.

    Produces:
      <choice>
        <abbr>part1<lb xml:id="X" break="no"/>part2</abbr>
        <expan>exp1<lb sameAs="#X"/>exp2</expan>
      </choice>
    """
    if end_line_idx - start_line_idx != 1:
        return  # only handle 2-line spans for now

    line1 = chain[start_line_idx]
    line2 = chain[end_line_idx]
    lb1_start, _ = line_boundaries[start_line_idx]
    lb2_start, _ = line_boundaries[end_line_idx]

    # Check if the change overlaps with existing <choice> elements.
    # If so, merge with the existing choice rather than creating a new one.
    local_start_l1 = orig_start - lb1_start
    local_end_l1 = min(orig_end, lb1_start + len(line1.plain_text)) - lb1_start
    local_start_l2 = max(0, orig_start - lb2_start)
    local_end_l2 = orig_end - lb2_start

    choice_runs_l1 = [
        r for r in line1.text_runs
        if r.from_choice and r.plain_end > local_start_l1 and r.plain_start < local_end_l1
    ]
    choice_runs_l2 = [
        r for r in line2.text_runs
        if r.from_choice and r.plain_end > local_start_l2 and r.plain_start < local_end_l2
    ]

    if choice_runs_l1 or choice_runs_l2:
        # The change overlaps with pre-existing choice elements.
        # Merge resp with existing choice rather than creating a new wrapper.
        for cr in choice_runs_l1 + choice_runs_l2:
            choice_el = cr.node
            if choice_el.tag == _tag("choice"):
                existing_expan = choice_el.find(_tag("expan"))
                if existing_expan is not None:
                    existing_text = _inner_text(existing_expan)
                    # Strip LINE_SEP before comparing: for a cross-line
                    # merged change, exp_text carries "repe¬riatur" whereas
                    # existing_text is the flattened "reperiatur".
                    exp_text_clean = exp_text.replace(LINE_SEP, "")
                    if exp_text_clean == existing_text or _texts_equivalent(exp_text_clean, existing_text):
                        _add_resp_to_element(existing_expan, f"#{EXPANSION_MODEL}")
                    else:
                        # Disagreement: flag for manual inspection
                        existing_expan.set("cert", "low")
        return

    l1_text = line1.plain_text
    l2_text = line2.plain_text

    # Punctuation that should NOT be part of abbreviation tokens
    # (semicolon excluded — it's part of abbreviations like atq;)
    WORD_BREAK_PUNCT = set('.,:!?()[]')

    # Find the last word in L1 (cross-line word's first part)
    l1_stripped = l1_text.rstrip()
    pos = len(l1_stripped)
    while pos > 0 and not l1_stripped[pos - 1].isspace() and l1_stripped[pos - 1] not in WORD_BREAK_PUNCT:
        pos -= 1
    word_start_in_l1 = pos
    abbr_part1 = l1_stripped[word_start_in_l1:]

    # Find the first word in L2 (cross-line word's second part)
    l2_content_start = 0
    while l2_content_start < len(l2_text) and l2_text[l2_content_start].isspace():
        l2_content_start += 1
    pos = l2_content_start
    while pos < len(l2_text) and not l2_text[pos].isspace() and l2_text[pos] not in WORD_BREAK_PUNCT:
        pos += 1
    word_end_in_l2 = pos
    abbr_part2 = l2_text[l2_content_start:word_end_in_l2]

    # Build the expanded word parts.
    # When exp_text contains LINE_SEP, the model placed the break at the
    # correct position in the expanded form — use it directly.  This avoids
    # fragile character-count alignment when, for example, æ expands to ae
    # right at the boundary.
    diff_local_start_in_l1 = orig_start - lb1_start
    l1_prefix = l1_text[word_start_in_l1:min(diff_local_start_in_l1, len(l1_text))]
    l1_prefix = l1_prefix.rstrip()

    trailing_punct = ""

    if LINE_SEP in exp_text:
        sep_idx = exp_text.index(LINE_SEP)
        # The LINE_SEP stands in for an <lb/> that, in pretty-printed TEI, is
        # usually flanked by whitespace.  That whitespace is part of the
        # layout, not the reading text, so trim it off the L1 tail and the L2
        # head of the expansion — otherwise it surfaces inside <expan> as
        # "ſucce\n<lb/>dant".
        exp_part1 = (l1_prefix + exp_text[:sep_idx]).rstrip()
        exp_part2_raw = exp_text[sep_idx + len(LINE_SEP):].lstrip()
        while exp_part2_raw and exp_part2_raw[-1] in WORD_BREAK_PUNCT:
            trailing_punct = exp_part2_raw[-1] + trailing_punct
            exp_part2_raw = exp_part2_raw[:-1]
        exp_part2 = exp_part2_raw
    else:
        # Fallback: rebuild from cleaned exp_text and align proportionally.
        exp_clean = exp_text.replace(LINE_SEP, "")
        full_exp = l1_prefix + exp_clean
        while full_exp and full_exp[-1] in WORD_BREAK_PUNCT:
            trailing_punct = full_exp[-1] + trailing_punct
            full_exp = full_exp[:-1]
        total_abbr_len = len(abbr_part1) + len(abbr_part2)
        if total_abbr_len > 0:
            src_break = len(abbr_part1)
            full_abbr = abbr_part1 + abbr_part2
            split = _find_tgt_break_with_fallback(full_abbr, full_exp, src_break)
            if split is None:
                split = round(len(full_exp) * len(abbr_part1) / max(total_abbr_len, 1))
        else:
            split = len(full_exp) // 2
        exp_part1 = full_exp[:split]
        exp_part2 = full_exp[split:]

    # --- Build <choice> element ---
    choice = etree.Element(_tag("choice"))
    choice.set("resp", _expansion_resp())
    abbr_el = etree.SubElement(choice, _tag("abbr"))
    expan_el = etree.SubElement(choice, _tag("expan"))
    expan_el.set("resp", _expansion_resp())

    # <abbr>: part1 from L1 (with inline elements) + <lb/> + part2 from L2
    l1_end_stripped = len(l1_text.rstrip())
    _populate_abbr_from_runs(
        abbr_el, line1.text_runs, word_start_in_l1, l1_end_stripped,
    )

    # Clone L2's <lb/> into <abbr>
    lb2_el = line2.lb_element
    lb2_id = _xml_id(lb2_el)
    lb2_clone = copy.deepcopy(lb2_el)
    lb2_clone.tail = None
    abbr_el.append(lb2_clone)

    # Add part2 from L2 as content after the <lb/> in <abbr>.  Start at
    # l2_content_start (not 0) so any whitespace that followed the <lb/> in
    # the source — i.e. leading layout whitespace on L2 — is not pulled into
    # the abbreviation token.
    _populate_abbr_from_runs_as_tail(
        lb2_clone, abbr_el, line2.text_runs, l2_content_start, word_end_in_l2,
    )

    # <expan>: exp_part1 + <lb sameAs="#id" break="no" xml:id="…"/> + exp_part2.
    # Re-wrap any special characters (e.g. a surviving long-s) in the same <g>
    # markup they carry inside <abbr>, so <expan> mirrors the diplomatic glyphs
    # rather than emitting bare fallback characters.
    glyph_map = _build_glyph_map(abbr_el)
    _emit_expansion_with_glyphs(expan_el, None, exp_part1, glyph_map)
    lb_same = etree.SubElement(expan_el, _tag("lb"))
    if lb2_id:
        lb_same.set("sameAs", f"#{lb2_id}")
        # Derive a unique xml:id for the sameAs lb by inserting "s" before
        # the leading digit of the last hyphen-separated segment, matching
        # the corpus convention (e.g. "…-lb-0035" → "…-lb-s0035").
        lb_same_id = re.sub(r'(lb-)(\d)', r'\1s\2', lb2_id)
        if lb_same_id != lb2_id:
            _set_xml_id(lb_same, lb_same_id)
    lb_same.set("break", "no")
    _emit_expansion_with_glyphs(expan_el, lb_same, exp_part2, glyph_map)

    # --- Tree surgery ---
    _truncate_line_end(line1, word_start_in_l1)

    # Compute choice.tail: only the un-consumed portion of the last
    # partially-consumed run. Remaining tree nodes stay in place.
    choice_tail = ""
    for run in line2.text_runs:
        if run.plain_start >= word_end_in_l2:
            break  # remaining runs stay in the tree
        if run.plain_end > word_end_in_l2:
            # Partially consumed — remaining text becomes choice.tail
            local_cut = word_end_in_l2 - run.plain_start
            choice_tail = run.text[local_cut:]

    _remove_line_start(line2, word_end_in_l2, lb2_el)
    _insert_choice_at_line_end(line1, word_start_in_l1, choice)

    choice.tail = choice_tail if choice_tail else (trailing_punct if trailing_punct else None)

    # Evict any notes that fell inside the cross-line abbreviation token to after
    # </choice>. A note is not part of the token's reading text, so it must not
    # be pulled into <abbr> (nor reproduced in <expan>); it is re-anchored just
    # after the whole <choice>. This mirrors the single-line path in
    # _apply_line_expansion / _apply_chain_expansion, which the cross-line branch
    # previously lacked.
    affected_notes = [n for n in line1.notes if n.plain_offset >= word_start_in_l1]
    affected_notes += [n for n in line2.notes if n.plain_offset < word_end_in_l2]
    for note_info in affected_notes:
        _move_note_after(note_info, choice)


def _populate_abbr_from_runs(
    abbr_el:    etree._Element,
    text_runs:  list[TextRun],
    start:      int,
    end:        int,
) -> None:
    """Populate <abbr> with text/inline elements from runs within [start, end)."""
    last_child = None
    for run in text_runs:
        if run.plain_end <= start or run.plain_start >= end:
            continue
        clip_start = max(start, run.plain_start) - run.plain_start
        clip_end = min(end, run.plain_end) - run.plain_start
        portion = run.text[clip_start:clip_end]
        if not portion:
            continue

        if not run.is_tail and _is_intoken_element(run.node):
            cloned = copy.deepcopy(run.node)
            cloned.text = portion
            cloned.tail = None
            for child in list(cloned):
                cloned.remove(child)
            abbr_el.append(cloned)
            last_child = cloned
        else:
            if last_child is not None:
                last_child.tail = (last_child.tail or "") + portion
            else:
                abbr_el.text = (abbr_el.text or "") + portion


def _populate_abbr_from_runs_as_tail(
    after_el:   etree._Element,
    abbr_el:    etree._Element,
    text_runs:  list[TextRun],
    start:      int,
    end:        int,
) -> None:
    """Populate <abbr> with runs from [start, end), placing first text as after_el.tail."""
    last_child = after_el
    for run in text_runs:
        if run.plain_end <= start or run.plain_start >= end:
            continue
        clip_start = max(start, run.plain_start) - run.plain_start
        clip_end = min(end, run.plain_end) - run.plain_start
        portion = run.text[clip_start:clip_end]
        if not portion:
            continue

        if not run.is_tail and _is_intoken_element(run.node):
            cloned = copy.deepcopy(run.node)
            cloned.text = portion
            cloned.tail = None
            for child in list(cloned):
                cloned.remove(child)
            abbr_el.append(cloned)
            last_child = cloned
        else:
            last_child.tail = (last_child.tail or "") + portion


def _build_glyph_map(source_el: etree._Element) -> dict[str, etree._Element]:
    """Harvest a character → <g> template map from an element's <g> glyphs.

    Used to re-wrap special characters in <expan> the same way they appear in
    <abbr>: e.g. a long-s that survives an expansion (ſucce[lb]dãt → ſuccedant)
    should be emitted as <g ref="#char017f">ſ</g>, not as a bare U+017F, mirror-
    ing the diplomatic markup of the token.

    Only single-character glyphs are mapped (the corpus convention — ſ, ã, &,
    …); the first occurrence of each character wins, so all instances of that
    character in the expansion are wrapped with the same @ref.
    """
    glyph_map: dict[str, etree._Element] = {}
    for g in source_el.iter(_tag("g")):
        if g.text and len(g.text) == 1:
            glyph_map.setdefault(g.text, g)
    return glyph_map


def _emit_expansion_with_glyphs(
    parent:    etree._Element,
    anchor:    Optional[etree._Element],
    text:      str,
    glyph_map: dict[str, etree._Element],
) -> Optional[etree._Element]:
    """Emit ``text`` into ``parent`` as mixed content, wrapping every character
    present in ``glyph_map`` in a clone of its <g> element.

    ``anchor is None``: the text/elements become ``parent``'s leading content
    (``parent.text`` plus appended <g> children).
    ``anchor`` given:   the text/elements are placed *after* ``anchor`` (its
    ``.tail`` plus following siblings) — used to emit the post-<lb/> part of a
    cross-line <expan>.

    Returns the last element emitted, or ``anchor`` if the text was plain.
    """
    if not glyph_map:
        # Fast path: no special characters to wrap — keep it as flat text.
        if anchor is None:
            parent.text = (parent.text or "") + text
        else:
            anchor.tail = (anchor.tail or "") + text
        return anchor

    last = anchor
    buf: list[str] = []

    def _flush() -> None:
        nonlocal last
        s = "".join(buf)
        buf.clear()
        if not s:
            return
        if last is None:
            parent.text = (parent.text or "") + s
        else:
            last.tail = (last.tail or "") + s

    for ch in text:
        tmpl = glyph_map.get(ch)
        if tmpl is None:
            buf.append(ch)
            continue
        _flush()
        g = copy.deepcopy(tmpl)
        g.text = ch
        g.tail = None
        for child in list(g):
            g.remove(child)
        if last is None:
            parent.append(g)
        else:
            parent.insert(parent.index(last) + 1, g)
        last = g

    _flush()
    return last


def _truncate_line_end(line: ExtractedLine, at_offset: int) -> None:
    """Remove text from at_offset onwards in a line, including inline elements."""
    for run in reversed(line.text_runs):
        if run.plain_start >= at_offset:
            # Entire run is after the cut — clear it
            if run.is_tail:
                run.node.tail = None
            elif _is_intoken_element(run.node):
                parent = run.node.getparent()
                if parent is not None:
                    # Transfer tail before removal
                    if run.node.tail:
                        prev = run.node.getprevious()
                        if prev is not None:
                            prev.tail = (prev.tail or "") + run.node.tail
                        else:
                            parent.text = (parent.text or "") + run.node.tail
                    parent.remove(run.node)
            else:
                run.node.text = None
        elif run.plain_end > at_offset:
            # Run partially overlaps — truncate
            local_cut = at_offset - run.plain_start
            truncated = run.text[:local_cut]
            if run.is_tail:
                run.node.tail = truncated if truncated else None
            else:
                run.node.text = truncated if truncated else None


def _remove_line_start(
    line:       ExtractedLine,
    up_to:      int,
    lb_el:      etree._Element,
) -> None:
    """Remove text from start up to up_to in a line, and remove the lb element."""
    # First pass: clear/truncate text content
    # Track which inline elements had their text fully consumed
    # but might still have surviving tail text
    elements_to_maybe_remove: list[etree._Element] = []

    for run in line.text_runs:
        if run.plain_end <= up_to:
            # Entire run before cut — clear it
            if run.is_tail:
                run.node.tail = None
            elif _is_intoken_element(run.node):
                run.node.text = None
                elements_to_maybe_remove.append(run.node)
            else:
                run.node.text = None
        elif run.plain_start < up_to:
            # Run partially overlaps — truncate from start
            local_cut = up_to - run.plain_start
            remaining = run.text[local_cut:]
            if run.is_tail:
                run.node.tail = remaining if remaining else None
            else:
                run.node.text = remaining if remaining else None

    # Second pass: remove inline elements that have no text AND no tail
    for el in elements_to_maybe_remove:
        if not el.text and not el.tail:
            parent = el.getparent()
            if parent is not None:
                parent.remove(el)
        elif not el.text and el.tail:
            # Element has no text but surviving tail — keep it but
            # it's now an empty wrapper. Convert to just tail text
            # on the previous sibling or parent.
            parent = el.getparent()
            if parent is not None:
                prev = el.getprevious()
                if prev is not None:
                    prev.tail = (prev.tail or "") + el.tail
                else:
                    parent.text = (parent.text or "") + el.tail
                parent.remove(el)

    # Remove the <lb/> element from the tree (it's now inside <abbr>)
    parent = lb_el.getparent()
    if parent is not None:
        lb_el.tail = None
        parent.remove(lb_el)


def _insert_choice_at_line_end(
    line:       ExtractedLine,
    at_offset:  int,
    choice:     etree._Element,
) -> None:
    """Insert <choice> at the position where line1's text was truncated."""
    # Find the last text run before/at the cut point
    for run in reversed(line.text_runs):
        if run.plain_start < at_offset:
            if run.is_tail:
                parent = run.node.getparent()
                if parent is None:
                    continue
                idx = list(parent).index(run.node)
                parent.insert(idx + 1, choice)
            else:
                run.node.insert(0, choice)
            return

    # Fallback: insert after the lb element
    lb = line.lb_element
    parent = lb.getparent()
    if parent is not None:
        idx = list(parent).index(lb)
        parent.insert(idx + 1, choice)


def _apply_line_expansion(
    line:     ExtractedLine,
    expanded: str,
) -> None:
    """
    Apply expansion to a single line by diffing original vs expanded text
    and wrapping changes in <choice><abbr>...<expan>...</choice>.
    """
    original = line.plain_text
    changes = _find_changes(original, expanded)
    changes_with_exp_text = _merge_changes_by_shared_runs(
        changes, line.text_runs, original, expanded,
    )

    if not changes_with_exp_text:
        return

    # Process changes in reverse order (right to left) so that
    # earlier offsets remain valid as we modify the tree
    for orig_start, orig_end, exp_text in reversed(changes_with_exp_text):
        orig_text = original[orig_start:orig_end]

        if not orig_text or not exp_text:
            continue

        # Skip changes that are only whitespace/punctuation differences
        if _is_punctuation_only_change(orig_text, exp_text):
            continue

        # Skip changes that are only glyph variants (ſ→s, è→e, etc.)
        if _is_glyph_variant_only_change(orig_text, exp_text):
            continue

        # A TEI <abbr> wraps a single token; refuse multi-token abbr spans.
        if not _abbr_is_single_token(orig_text):
            continue

        # Check if any notes fall inside this change range
        affected_notes = [
            n for n in line.notes
            if orig_start <= n.plain_offset < orig_end
        ]

        # Apply the change, preserving inline markup in <abbr>
        choice = _apply_change_preserving_markup(
            line, orig_start, orig_end, exp_text,
        )

        if choice is None:
            continue

        # Move affected notes to after the <choice>
        for note_info in affected_notes:
            _move_note_after(note_info, choice)


def _merge_changes_by_shared_runs(
    changes: list[tuple[int, int, int, int]],
    text_runs: list[TextRun],
    original: str,
    expanded: str,
) -> list[tuple[int, int, str]]:
    """
    Merge adjacent changes when their affected run sets overlap.

    This avoids applying two independent tree surgeries to ranges that share
    the same underlying XML node, which can orphan the insertion anchor for
    the second change when processing right-to-left.
    """
    if not changes:
        return []

    merged: list[tuple[int, int, str]] = []
    cur_o1, cur_o2, cur_e1, cur_e2 = changes[0]
    cur_exp_text = expanded[cur_e1:cur_e2]
    cur_run_nodes = {
        id(run.node)
        for run in text_runs
        if run.plain_end > cur_o1 and run.plain_start < cur_o2
    }

    for o1, o2, e1, e2 in changes[1:]:
        next_run_nodes = {
            id(run.node)
            for run in text_runs
            if run.plain_end > o1 and run.plain_start < o2
        }

        unchanged_between = original[cur_o2:o1] if o1 >= cur_o2 else ""
        # Only merge two changes when they share a tree node AND belong to the
        # same whitespace-delimited token.  Two *separate* abbreviations can
        # share a node — e.g. a <g>'s tail that runs from one word into the
        # next ("…itat<g>ẽ</g> fidelium bonor<g>ũ</g>") makes the "…ẽ" element
        # and the " fidelium bonor" tail the same node — but they must stay
        # separate <choice> elements (a multi-token <abbr> spanning whitespace
        # is not TEI-compliant).  Merging is still required within a single
        # token so the right-to-left tree surgery does not orphan a shared
        # anchor, so we gate the merge on the gap being whitespace-free.
        if (cur_run_nodes.intersection(next_run_nodes)
                and not _contains_whitespace(unchanged_between)):
            cur_exp_text = cur_exp_text + unchanged_between + expanded[e1:e2]
            cur_o2 = max(cur_o2, o2)
            cur_run_nodes.update(next_run_nodes)
        else:
            merged.append((cur_o1, cur_o2, cur_exp_text))
            cur_o1, cur_o2, cur_e1, cur_e2 = o1, o2, e1, e2
            cur_exp_text = expanded[cur_e1:cur_e2]
            cur_run_nodes = next_run_nodes

    merged.append((cur_o1, cur_o2, cur_exp_text))
    return merged


def _contains_whitespace(text: str) -> bool:
    """True if ``text`` contains any whitespace or the LINE_SEP separator."""
    return any(c.isspace() or c == LINE_SEP for c in text)


def _abbr_is_single_token(orig_text: str) -> bool:
    """True if ``orig_text`` is a single abbreviation token (TEI-compliant).

    A TEI <abbr> wraps one token, not a run of words.  A cross-line token may
    legitimately span a LINE_SEP (with layout whitespace around the break), so
    we split on LINE_SEP and require each segment — stripped of its surrounding
    layout whitespace — to contain no *internal* whitespace.

    This rejects multi-token <abbr> spans that the upstream diff/merge can
    produce, e.g. a model that replaces a whole literature reference
    ("476. col. 2. nu. 1.") with unrelated text: the abbr side would carry
    internal whitespace and is refused, so no spurious multi-token <choice> is
    emitted.  Multi-*word* expansions of a single token ("&c" → "et cetera")
    are unaffected: only the abbr (original) side is checked, never <expan>.
    """
    for segment in orig_text.split(LINE_SEP):
        if any(c.isspace() for c in segment.strip()):
            return False
    return True


def _find_changes(
    original: str,
    expanded: str,
) -> list[tuple[int, int, int, int]]:
    """
    Find character ranges that differ between original and expanded text,
    expanded to word boundaries.

    Uses SequenceMatcher to find character-level diffs, then expands each
    diff range outward to the nearest whitespace/punctuation boundary in
    both strings. This ensures whole abbreviated tokens are wrapped in
    <choice>, matching the Salamanca corpus convention.

    Returns list of (orig_start, orig_end, exp_start, exp_end) tuples.
    """
    sm = SequenceMatcher(None, original, expanded, autojunk=False)
    raw_changes: list[tuple[int, int, int, int]] = []

    # Collect every non-equal opcode, not just "replace".  When an expansion
    # keeps the special characters and only *adds* plain ones around them
    # (e.g. ſcm → ſanctum, or a cross-line token whose surviving letters anchor
    # the inserted ones), SequenceMatcher emits "insert"/"delete" opcodes
    # rather than a single "replace".  Collecting only "replace" silently
    # dropped those expansions, leaving the abbreviation unwrapped.
    #
    # Insertions/deletions are only kept when they are *intra-token*: the added
    # (or removed) text must not contain whitespace or a LINE_SEP.  A whole-word
    # insertion/deletion always carries the separating space, so this guard
    # stops an inserted word from being mis-attributed as the expansion of an
    # adjacent token (e.g. "alpha" → "alpha beta").  Multi-word expansions of a
    # single abbreviation (e.g. "&c" → "et cetera") arrive as "replace" opcodes
    # and are unaffected.
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "replace":
            raw_changes.append((i1, i2, j1, j2))
        elif op == "insert":
            if not _contains_word_break(expanded[j1:j2]):
                raw_changes.append((i1, i2, j1, j2))
        elif op == "delete":
            if not _contains_word_break(original[i1:i2]):
                raw_changes.append((i1, i2, j1, j2))

    if not raw_changes:
        return []

    # Expand each change to word boundaries
    changes: list[tuple[int, int, int, int]] = []
    for i1, i2, j1, j2 in raw_changes:
        # Expand in original
        oi1 = _expand_left(original, i1)
        oi2 = _expand_right(original, i2)
        # Expand in expanded — use the same word context
        oj1 = _expand_left(expanded, j1)
        oj2 = _expand_right(expanded, j2)
        # A pure insertion or deletion that sits between words (flanked by
        # whitespace/LINE_SEP) expands to an empty range on one side and cannot
        # be represented as <choice><abbr>…</abbr><expan>…</expan></choice> —
        # both branches need a non-empty token — so skip it.  In-token
        # insertions/deletions expand to the full word on both sides.
        if oi1 >= oi2 or oj1 >= oj2:
            continue
        changes.append((oi1, oi2, oj1, oj2))

    if not changes:
        return []

    # Merge overlapping/adjacent expanded ranges
    return _merge_changes(changes)


def _contains_word_break(text: str) -> bool:
    """True if ``text`` contains any whitespace or the LINE_SEP separator.

    Used to tell an *intra-token* insertion/deletion (kept — it edits a single
    word) from a *whole-word* one (skipped — it adds or removes entire words and
    must not be folded into an adjacent abbreviation token)."""
    return any(c.isspace() or c == LINE_SEP for c in text)


def _expand_left(text: str, pos: int) -> int:
    """Expand pos leftward to the start of the current word.

    Treats whitespace and LINE_SEP as word boundaries so that
    word-boundary expansion in concatenated chain text never crosses
    the artificial line separator.
    """
    while pos > 0 and not text[pos - 1].isspace() and text[pos - 1] != LINE_SEP:
        pos -= 1
    return pos


def _expand_right(text: str, pos: int) -> int:
    """Expand pos rightward to the end of the current word.

    Treats whitespace and LINE_SEP as word boundaries so that
    word-boundary expansion in concatenated chain text never crosses
    the artificial line separator.
    """
    while pos < len(text) and not text[pos].isspace() and text[pos] != LINE_SEP:
        pos += 1
    return pos


def _merge_changes(
    changes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """Merge overlapping or adjacent change ranges."""
    if not changes:
        return []
    merged: list[tuple[int, int, int, int]] = [changes[0]]
    for i1, i2, j1, j2 in changes[1:]:
        pi1, pi2, pj1, pj2 = merged[-1]
        if i1 <= pi2 and j1 <= pj2:
            # Overlapping — extend
            merged[-1] = (pi1, max(pi2, i2), pj1, max(pj2, j2))
        else:
            merged.append((i1, i2, j1, j2))
    return merged


def _is_intoken_element(node: etree._Element) -> bool:
    """True only for elements that are *part of* an abbreviation token and so
    belong INSIDE <abbr> (cloned in, original removed): glyph elements <g>.

    Span containers that merely *wrap* the token as text (<hi>, <foreign>,
    <ref>, <term>, <mentioned>, <title>, <emph>, ...) are deliberately NOT
    included: for those, <choice> nests inside the container and the container
    is left in place, e.g.

        <hi><choice><abbr>dño</abbr><expan>domino</expan></choice></hi>

    rather than cloning <hi> into <abbr>. A <g> stays a single character/glyph
    even when it carries text content (an ASCII/Unicode fallback plus an @ref),
    so it is atomic and moves into <abbr> with its attributes intact.

    Milestones (<lb>, <cb>, <pb>, <milestone>) are also conceptually in-token,
    but they are handled separately on the cross-line path and do not surface as
    text runs here, so they are not listed."""
    tag = node.tag
    if not isinstance(tag, str):
        return False
    local = tag.split("}")[-1] if "}" in tag else tag
    return local == "g"


def _merge_with_existing_choice(
    line:          ExtractedLine,
    affected_runs: list[TextRun],
    choice_runs:   list[TextRun],
    orig_start:    int,
    orig_end:      int,
    expan_text:    str,
) -> Optional[etree._Element]:
    """
    Merge a new expansion with an existing <choice> element rather than nesting.

    When word-boundary expansion includes text from a pre-existing <choice>
    along with surrounding punctuation/text, we:
    1. Strip punctuation/tail text from the expansion to isolate the word
    2. Compare with the existing <expan> text
    3. If they agree: add the new model's resp to the existing <expan>
    4. If they disagree: flag with @cert="low" for manual inspection
       (do NOT add the model identifier to @resp)

    Returns the existing <choice> element, or None if skipped.
    """
    # The choice_run's node IS the <choice> element itself
    choice_el = choice_runs[0].node
    if choice_el.tag != _tag("choice"):
        return None

    # Get existing expan text
    existing_expan = choice_el.find(_tag("expan"))
    if existing_expan is None:
        return None
    existing_expan_text = _inner_text(existing_expan)

    # Determine what portion of expan_text corresponds to the choice part
    # vs. surrounding punctuation/text that was pulled in by word boundary expansion.
    # The choice_run covers [choice_run.plain_start, choice_run.plain_end) in the
    # original text. The non-choice affected runs are punctuation/surrounding text.
    #
    # In the original: "politicorũ," — choice covers "politicorũ", comma is tail
    # In the expanded: "politicorum," — we need to extract "politicorum"
    #
    # Strategy: identify leading/trailing non-choice text in orig, then strip the
    # same from expan_text.
    choice_orig_start = choice_runs[0].plain_start
    choice_orig_end = choice_runs[-1].plain_end

    # Text before the choice portion in the affected range
    leading_non_choice = line.plain_text[orig_start:max(orig_start, choice_orig_start)]
    # Text after the choice portion in the affected range
    trailing_non_choice = line.plain_text[min(orig_end, choice_orig_end):orig_end]

    # The expansion should have the same leading/trailing non-choice text
    # (since the model typically doesn't change punctuation)
    exp_choice_text = expan_text
    if leading_non_choice and exp_choice_text.startswith(leading_non_choice):
        exp_choice_text = exp_choice_text[len(leading_non_choice):]
    if trailing_non_choice and exp_choice_text.endswith(trailing_non_choice):
        exp_choice_text = exp_choice_text[:-len(trailing_non_choice)]

    # Compare the isolated expansion with the existing expan text
    if exp_choice_text == existing_expan_text:
        # Expansions agree — just add the new model's resp to existing expan
        _add_resp_to_element(existing_expan, f"#{EXPANSION_MODEL}")
        return choice_el
    else:
        # Check if they agree ignoring glyph differences (e.g. long-s vs s)
        if _texts_equivalent(exp_choice_text, existing_expan_text):
            _add_resp_to_element(existing_expan, f"#{EXPANSION_MODEL}")
            return choice_el

        # Expansions disagree — flag for manual inspection without adding
        # the model's resp.  The existing (rule-based) expansion is kept
        # unchanged; @cert="low" signals that the model produced a different
        # reading and the element should be reviewed by a human editor.
        existing_expan.set("cert", "low")
        return choice_el


def _add_resp_to_element(el: etree._Element, new_resp: str) -> None:
    """Add a resp token to an element's @resp attribute if not already present."""
    current = el.get("resp", "")
    tokens = current.split() if current else []
    if new_resp not in tokens:
        tokens.append(new_resp)
        el.set("resp", " ".join(tokens))


def _texts_equivalent(a: str, b: str) -> bool:
    """
    Check if two texts are equivalent ignoring glyph variants
    (long-s/s, u/v, etc.).
    """
    if len(a) != len(b):
        return False
    for ca, cb in zip(a, b):
        if ca != cb and not chars_match(ca, cb):
            return False
    return True


def _apply_change_preserving_markup(
    line:       ExtractedLine,
    orig_start: int,
    orig_end:   int,
    expan_text: str,
) -> Optional[etree._Element]:
    """
    Apply a single abbreviation expansion to the XML tree,
    preserving inline elements (like <g>) inside <abbr>.

    Returns the created <choice> element, or None if skipped.
    """
    affected_runs = [
        r for r in line.text_runs
        if r.plain_end > orig_start and r.plain_start < orig_end
    ]

    if not affected_runs:
        return None

    # --- Handle changes that overlap with pre-existing <choice> elements ---
    # When any affected run comes from an existing <choice>, merge with it
    # rather than creating nested choices or wrapping around them.
    choice_runs = [r for r in affected_runs if r.from_choice]
    if choice_runs:
        return _merge_with_existing_choice(
            line, affected_runs, choice_runs, orig_start, orig_end, expan_text,
        )

    first_run = affected_runs[0]
    last_run = affected_runs[-1]

    # --- Build <choice> element ---
    choice = etree.Element(_tag("choice"))
    choice.set("resp", _expansion_resp())
    abbr_el = etree.SubElement(choice, _tag("abbr"))
    expan_el = etree.SubElement(choice, _tag("expan"))
    expan_el.set("resp", _expansion_resp())

    # --- Build <abbr> content preserving inline elements ---
    abbr_last_child = None  # tracks last element appended to abbr

    for run in affected_runs:
        clip_start = max(orig_start, run.plain_start) - run.plain_start
        clip_end = min(orig_end, run.plain_end) - run.plain_start
        text_portion = run.text[clip_start:clip_end]

        if not text_portion:
            continue

        if not run.is_tail and _is_intoken_element(run.node):
            # Clone the inline element into <abbr>
            cloned = copy.deepcopy(run.node)
            cloned.text = text_portion
            cloned.tail = None
            for child in list(cloned):
                cloned.remove(child)
            abbr_el.append(cloned)
            abbr_last_child = cloned
        else:
            # Plain text
            if abbr_last_child is not None:
                abbr_last_child.tail = (abbr_last_child.tail or "") + text_portion
            else:
                abbr_el.text = (abbr_el.text or "") + text_portion

    # --- Build <expan> content, re-wrapping any special characters in the same
    # <g> markup they carry inside <abbr> (deferred until <abbr> is built so the
    # glyph map can be harvested from it).
    _emit_expansion_with_glyphs(
        expan_el, None, expan_text, _build_glyph_map(abbr_el),
    )

    # --- Determine insertion point ---
    if first_run.is_tail:
        insert_parent = first_run.node.getparent()
        insert_after = first_run.node
    else:
        if _is_intoken_element(first_run.node):
            insert_parent = first_run.node.getparent()
            insert_after = first_run.node.getprevious()
        else:
            insert_parent = first_run.node
            insert_after = None  # insert as first child

    # --- Capture tail text that survives after </choice> ---
    local_end_in_last = orig_end - last_run.plain_start

    if last_run.is_tail:
        after_text = last_run.text[local_end_in_last:]
    elif _is_intoken_element(last_run.node):
        # the in-token element (e.g. <g>) is moved into <abbr> and removed;
        # what survives after </choice> is that element's tail
        after_text = last_run.node.tail or ""
    else:
        # the token lived in a wrapper/block .text (e.g. <hi>) which we keep in
        # place: the surviving text is the remainder of that same text node,
        # which becomes the choice's tail inside the preserved wrapper
        after_text = last_run.text[local_end_in_last:]

    # --- Tree surgery ---

    # Truncate first run (keep text before the change)
    local_start_in_first = orig_start - first_run.plain_start
    before_text = first_run.text[:local_start_in_first]

    if first_run.is_tail:
        first_run.node.tail = before_text if before_text else None
    elif not _is_intoken_element(first_run.node):
        first_run.node.text = before_text if before_text else None

    # Remove inline elements that were cloned into <abbr>
    nodes_to_remove: list[etree._Element] = []
    consumed_tails: set[int] = set()
    for run in affected_runs:
        if not run.is_tail and _is_intoken_element(run.node):
            nodes_to_remove.append(run.node)
        if run.is_tail:
            consumed_tails.add(id(run.node))

    for node in nodes_to_remove:
        parent = node.getparent()
        if parent is not None:
            if node.tail and node is last_run.node:
                pass  # will become choice.tail
            elif node.tail and id(node) not in consumed_tails:
                prev = node.getprevious()
                if prev is not None:
                    prev.tail = (prev.tail or "") + node.tail
                else:
                    parent.text = (parent.text or "") + node.tail
            parent.remove(node)

    # Clear last run's consumed text
    if last_run is not first_run and not last_run.is_tail and not _is_intoken_element(last_run.node):
        last_run.node.text = None
    if last_run is not first_run and last_run.is_tail:
        last_run.node.tail = None

    # --- Insert <choice> ---
    choice.tail = after_text if after_text else None

    if insert_parent is None:
        # Anchor node was detached by an earlier change in this pass; skip safely.
        return None

    if insert_after is not None:
        idx = list(insert_parent).index(insert_after) + 1
    else:
        idx = 0

    insert_parent.insert(idx, choice)

    return choice


def _move_note_after(
    note_info: NoteInfo,
    choice:    etree._Element,
) -> None:
    """
    Move a <note> element to immediately after a <choice> element.

    Used when a note falls inside an abbreviation that gets wrapped
    in <choice>. The note is detached from its current position and
    re-inserted as a sibling after <choice>.

    The choice's tail text (text between </choice> and the next tag)
    is transferred to the note's tail, so the note sits directly
    after </choice> with no intervening text.
    """
    note_el = note_info.element
    parent = note_el.getparent()

    if parent is None:
        return

    # Preserve note's original tail text
    old_note_tail = note_el.tail
    note_el.tail = None

    # Remove note from current position
    parent.remove(note_el)

    # Splice note between choice and choice's tail text:
    # Before: <choice>...</choice>TAIL_TEXT
    # After:  <choice>...</choice><note>...</note>TAIL_TEXT
    choice_parent = choice.getparent()
    if choice_parent is not None:
        choice_idx = list(choice_parent).index(choice)

        # Transfer choice's tail to note's tail
        note_el.tail = choice.tail
        choice.tail = None

        choice_parent.insert(choice_idx + 1, note_el)


# ---------------------------------------------------------------------------
# Application information (teiHeader/encodingDesc/appInfo)
# ---------------------------------------------------------------------------

# Default Hugging Face org used to build a URL for any model identifier not
# found in _MODEL_INFO below.
_DEFAULT_HF_ORG = "mpilhlt"

# Hugging Face URLs and human-readable labels for the models that may appear in
# @resp, keyed by the model identifier exactly as written in @resp (i.e. the
# value of EXPANSION_MODEL / BOUNDARY_MODEL). <application>/@version must be a
# teidata.versionNumber (dotted integers), so the precise revision is NOT put
# there; instead each <application> is stamped with @notAfter (the processing
# date) and the repo URL is given without a commit hash.
#
# NB: the boundary step may be run with either of two interchangeable models —
# a Flair model and a CANINE model. Whichever one BOUNDARY_MODEL names is the
# one declared. If your identifier strings differ from the keys below, add/adjust
# the entries (an unknown id falls back to https://huggingface.co/<org>/<id>).
_MODEL_INFO = {
    "byt5-salamanca-abbr": {
        "url":   "https://huggingface.co/mpilhlt/byt5-salamanca-abbr",
        "label": "ByT5 Salamanca abbreviation-expansion model",
        "desc":  "Sequence-to-sequence model that expands abbreviations; "
                 "supplies the content of tei:expan and the wrapping tei:choice.",
    },
    "flair-lb-detector": {
        "url":   "https://huggingface.co/mschonhardt/latin-contextual-lb-detector",
        "label": "Flair contextual line-boundary detector",
        "desc":  "Token-level classifier predicting whether a line break falls "
                 "inside a word; supplies tei:lb/@break=\"no\".",
    },
    "canine-salamanca-boundary-classifier": {
        "url":   "https://huggingface.co/mpilhlt/canine-salamanca-boundary-classifier",
        "label": "CANINE Salamanca boundary classifier",
        "desc":  "Token-level classifier predicting whether a line break falls "
                 "inside a word; supplies tei:lb/@break=\"no\".",
    },
}


def _model_info(model_id: str) -> dict:
    """Return {url, label, desc} for a model id, with a sensible fallback."""
    info = _MODEL_INFO.get(model_id)
    if info is not None:
        return info
    return {
        "url":   f"https://huggingface.co/{_DEFAULT_HF_ORG}/{model_id}",
        "label": model_id,
        "desc":  f"Model {model_id}.",
    }


def _set_xml_id(el: etree._Element, value: str) -> None:
    """Set xml:id in the XML namespace."""
    el.set(f"{{{XML_NS}}}id", value)


def _ensure_app_info(
    root:            etree._Element,
    processing_date: Optional[str] = None,
) -> None:
    """
    Declare, in teiHeader/encodingDesc/appInfo, the applications that the @resp
    pointers refer to: the svsal-poco pipeline (given xml:id="auto" so the
    literal "#auto" token keeps resolving) plus the active expansion and
    boundary models (xml:id == the model identifier, matching the "#<model>"
    token in @resp).

    Each <application> carries the required @ident and @version, plus @notAfter
    set to the processing date (the model revision is therefore recorded by date
    rather than by commit hash) and a <ref> to its repository URL.

    Idempotent (no duplicate <application> elements on re-runs) and a no-op when
    there is no <teiHeader> (e.g. a bare fragment).
    """
    if processing_date is None:
        processing_date = date.today().isoformat()

    # Locate the teiHeader. root may be the TEI element, the teiHeader itself,
    # or a header-less fragment.
    if root.tag == _tag("teiHeader"):
        header = root
    else:
        header = next(iter(root.iter(_tag("teiHeader"))), None)
    if header is None:
        return  # fragment without a header — nothing to attach to

    # Find or create <encodingDesc>, keeping teiHeader child order valid
    # (after <fileDesc>, before <revisionDesc>).
    enc = header.find(_tag("encodingDesc"))
    if enc is None:
        enc = etree.Element(_tag("encodingDesc"))
        file_desc = header.find(_tag("fileDesc"))
        rev_desc = header.find(_tag("revisionDesc"))
        if file_desc is not None:
            file_desc.addnext(enc)
        elif rev_desc is not None:
            rev_desc.addprevious(enc)
        else:
            header.append(enc)

    # Find or create <appInfo>.
    app_info = enc.find(_tag("appInfo"))
    if app_info is None:
        app_info = etree.SubElement(enc, _tag("appInfo"))

    existing_ids = {_xml_id(a) for a in app_info.findall(_tag("application"))}

    exp_info = _model_info(EXPANSION_MODEL)
    bnd_info = _model_info(BOUNDARY_MODEL)

    # (xml:id, ident, label, desc, url, [ptr targets])
    apps = [
        ("auto", "svsal-poco",
         "svsal-poco transcription post-correction pipeline",
         "Automated post-correction of transcriptions: abbreviation expansion "
         "(tei:choice with tei:abbr/tei:expan) and line-boundary detection "
         "(tei:lb/@break), applied via the svsal-poco toolchain.",
         "https://github.com/digicademy/svsal-poco",
         [EXPANSION_MODEL, BOUNDARY_MODEL]),
        (EXPANSION_MODEL, EXPANSION_MODEL,
         exp_info["label"], exp_info["desc"], exp_info["url"], []),
        (BOUNDARY_MODEL, BOUNDARY_MODEL,
         bnd_info["label"], bnd_info["desc"], bnd_info["url"], []),
    ]

    for xml_id, ident, label, desc, url, ptr_targets in apps:
        if xml_id in existing_ids:
            continue
        app = etree.SubElement(app_info, _tag("application"))
        _set_xml_id(app, xml_id)
        app.set("ident", ident)
        app.set("version", "1.0")
        app.set("notAfter", processing_date)
        # content model: model.labelLike+ , (model.ptrLike* | model.pLike*)
        etree.SubElement(app, _tag("label")).text = label
        desc_el = etree.SubElement(app, _tag("desc"))
        desc_el.set(f"{{{XML_NS}}}lang", "en")
        desc_el.text = desc
        etree.SubElement(app, _tag("ref")).set("target", url)
        for target_id in ptr_targets:
            etree.SubElement(app, _tag("ptr")).set("target", f"#{target_id}")


# ---------------------------------------------------------------------------
# High-level pipeline integration
# ---------------------------------------------------------------------------

def process_tei_xml(
    xml_string:       str,
    run_pipeline_fn,  # callable: (lines_jsonl, pre_annotated) → (expanded_dict, boundary_dict)
    expansion_model:  Optional[str] = None,
    boundary_model:   Optional[str] = None,
) -> str:
    """
    Full roundtrip: TEI XML string → expand abbreviations → TEI XML string.

    1. Parse XML
    2. Extract lines (main text + notes separately)
    3. Run pipeline on plain text lines
    4. Apply expansions and boundary predictions back to XML
    5. Serialize back to string

    run_pipeline_fn should accept (line_dicts, pre_annotated_boundaries)
    and return (expanded_dict, boundary_dict).

    expansion_model / boundary_model: model names for resp attributes
    on auto-generated elements. Defaults to module-level EXPANSION_MODEL
    and BOUNDARY_MODEL.
    """
    # Set model names for resp attributes
    global EXPANSION_MODEL, BOUNDARY_MODEL
    if expansion_model is not None:
        EXPANSION_MODEL = expansion_model
    if boundary_model is not None:
        BOUNDARY_MODEL = boundary_model
    # Parse
    parser = etree.XMLParser(remove_blank_text=False)
    tree = etree.ElementTree(etree.fromstring(xml_string.encode("utf-8"), parser))

    # Detect namespace and configure tag builder
    global _tag
    ns_prefix = _detect_namespace(tree.getroot())
    _tag = _make_tag_fn(ns_prefix)

    # Extract lines
    lines, pre_annotated = extract_lines(tree)

    if not lines:
        return xml_string

    # Build pipeline input
    pipeline_rows = []
    for line in lines:
        pipeline_rows.append({
            "id":         line.line_id,
            "doc_id":     _xml_id(tree.getroot()) or "doc",
            "source_sic": line.plain_text,
            "lang":       line.lang,
        })

    # Run pipeline
    expanded_dict, boundary_dict = run_pipeline_fn(
        pipeline_rows, pre_annotated
    )

    # Apply results
    apply_expansions(tree, lines, expanded_dict, boundary_dict, pre_annotated)

    # Declare the applications referenced by the @resp pointers (the "#auto"
    # pipeline marker and the model ids) in teiHeader/encodingDesc/appInfo.
    _ensure_app_info(tree.getroot())

    # Clean up namespace declarations: ensure they appear only on the root
    # element, and remove any empty xmlns or xml:id attributes.
    etree.cleanup_namespaces(tree.getroot())
    _remove_empty_attrs(tree.getroot())

    # Serialize
    raw = etree.tostring(
        tree.getroot(),
        encoding="unicode",
        pretty_print=False,
    )

    # Post-process: format <lb break="no"/> elements
    return _format_lb_break_no(raw)


def _remove_empty_attrs(root: etree._Element) -> None:
    """
    Remove empty xmlns and xml:id attributes from all elements in the tree.
    These can appear as artifacts of tree manipulation.
    """
    xml_id_key = f"{{{XML_NS}}}id"
    for el in root.iter():
        # Remove empty xml:id
        if el.get(xml_id_key) == "":
            del el.attrib[xml_id_key]
        # Remove explicit xmlns="" (lxml represents this differently,
        # but check for safety)
        if "xmlns" in el.attrib and el.attrib["xmlns"] == "":
            del el.attrib["xmlns"]


def _format_lb_break_no(xml_string: str) -> str:
    """
    Post-process serialized XML to format <lb break="no"/> elements:
    1. Remove whitespace (including newlines) before <lb ... break="no" .../>
    2. Reorder attributes: break="no" first, then resp, then others,
       then xml:id on a new indented line
    """
    # Match <lb .../> or <cb .../> elements that have break="no"
    # Also capture any preceding whitespace
    pattern = re.compile(
        r'(\s*)'                     # preceding whitespace (group 1)
        r'(<(?:lb|cb)\s)'            # tag opening (group 2)
        r'([^>]*?)'                  # attributes (group 3)
        r'(/>)',                     # self-close (group 4)
    )

    def _reformat_lb(match):
        pre_ws = match.group(1)
        tag_open = match.group(2)     # "<lb " or "<cb "
        attrs_str = match.group(3)
        close = match.group(4)

        # Parse attributes from the string
        attr_pattern = re.compile(r'(\S+)="([^"]*)"')
        attrs = attr_pattern.findall(attrs_str)
        attr_dict = {k: v for k, v in attrs}

        # Only reformat if break="no" is present
        if attr_dict.get("break") != "no":
            return match.group(0)  # leave unchanged

        # Remove preceding whitespace (concatenate lines)
        pre_ws = ""

        # Build attribute lines:
        # Line 1: all attributes except xml:id
        # Line 2 (indented): xml:id only
        line1_attrs = ['break="no"']
        for k, v in attrs:
            if k in ("break", "xml:id"):
                continue
            line1_attrs.append(f'{k}="{v}"')

        xml_id = attr_dict.get("xml:id")

        # Build the formatted element
        result = pre_ws + tag_open.rstrip() + " " + " ".join(line1_attrs)

        if xml_id:
            result += f'\n    xml:id="{xml_id}"'

        return result + close

    return pattern.sub(_reformat_lb, xml_string)
