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
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Optional

from lxml import etree

TEI_NS = "http://www.tei-c.org/ns/1.0"
XML_NS = "http://www.w3.org/XML/1998/namespace"
NSMAP  = {"tei": TEI_NS, "xml": XML_NS}

def _tei(tag: str) -> str:
    """Build a Clark-notation tag name in the TEI namespace."""
    return f"{{{TEI_NS}}}{tag}"

def _xml_id(el: etree._Element) -> Optional[str]:
    """Get the xml:id attribute of an element."""
    return el.get(f"{{{XML_NS}}}id")


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

def extract_lines(tree: etree._ElementTree) -> list[ExtractedLine]:
    """
    Extract all lines from a TEI XML tree.

    Walks the tree to find all <lb/> elements. For each lb, collects
    the text content between it and the next lb, handling:
    - Notes: extracted as separate entries, main text stitched around them
    - Inline tags: stripped for plain text, but tree positions recorded
    - Existing <choice> elements: their <abbr> text is used as source
    - Note-initial text: text inside a <note> before its first <lb/>

    Returns a flat list of ExtractedLine objects. Main-text lines and
    note lines are interleaved in document order, distinguished by
    is_in_note.
    """
    root = tree.getroot()

    # Find all <lb/> elements in document order
    all_lbs = root.iter(_tei("lb"))
    lb_list = [lb for lb in all_lbs if lb.get("sameAs") is None]

    if not lb_list:
        return []

    # Find notes with text before their first internal <lb/>.
    # These need synthetic line entries since no lb triggers extraction.
    note_initial_lines = _extract_note_initial_lines(root, lb_list)

    # Build lb-based lines
    lb_lines: list[ExtractedLine] = []

    for i, lb in enumerate(lb_list):
        next_lb = lb_list[i + 1] if i + 1 < len(lb_list) else None
        is_in_note = _is_inside_note(lb)
        line_id = _xml_id(lb) or f"__lb_{i}"

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

    return lines


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

    for note in root.iter(_tei("note")):
        # Find the first <lb/> inside this note
        first_lb = None
        for child_lb in note.iter(_tei("lb")):
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
                if parent.tag == _tei("note") and id(parent) in note_to_initial:
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
        if parent.tag == _tei("note"):
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
    """
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
    if parent is not None and parent.tag != _tei("body"):
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
    if tag == _tei("note") and not is_in_note:
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
    if tag == _tei("choice"):
        # Use <abbr> or <sic> text (the original form)
        abbr = el.find(_tei("abbr"))
        sic = el.find(_tei("sic"))
        source_el = abbr if abbr is not None else sic
        if source_el is not None:
            # Collect text from the source branch
            source_text = _inner_text(source_el)
            if source_text:
                text_runs.append(TextRun(
                    text=source_text, node=el, is_tail=False,
                ))
        # Tail text after </choice>
        if el.tail:
            text_runs.append(TextRun(
                text=el.tail, node=el, is_tail=True,
            ))
        return

    # --- Self-closing elements (lb, pb, cb, etc.) ---
    if tag in (_tei("lb"), _tei("pb"), _tei("cb"), _tei("fw")):
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
) -> etree._ElementTree:
    """
    Apply abbreviation expansions and boundary predictions back to the XML tree.

    For each line where the expanded text differs from the original:
    1. Diff to find changed character ranges
    2. For each change, wrap in <choice><abbr>orig</abbr><expan>expanded</expan></choice>
    3. If a note falls inside a changed range, move it to after </choice>
    4. Update <lb/> with @break="no" where boundary classifier detected nonbreaking

    Returns the modified tree (modified in-place).
    """
    # --- Apply boundary predictions ---
    for line in lines:
        if line.line_id in boundary_predictions:
            lb = line.lb_element
            lb.set("break", "no")

    # --- Apply expansions ---
    for line in lines:
        expanded = expanded_texts.get(line.line_id)
        if expanded is None or expanded == line.plain_text:
            continue

        _apply_line_expansion(line, expanded)

    return tree


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

    if not changes:
        return

    # Process changes in reverse order (right to left) so that
    # earlier offsets remain valid as we modify the tree
    for orig_start, orig_end, exp_start, exp_end in reversed(changes):
        orig_text = original[orig_start:orig_end]
        exp_text = expanded[exp_start:exp_end]

        if not orig_text or not exp_text:
            continue

        # Check if any notes fall inside this change range
        affected_notes = [
            n for n in line.notes
            if orig_start <= n.plain_offset < orig_end
        ]

        # Find which text runs are affected
        affected_runs = [
            r for r in line.text_runs
            if r.plain_end > orig_start and r.plain_start < orig_end
        ]

        if not affected_runs:
            continue

        # Build the <choice> element
        choice = _build_choice_element(orig_text, exp_text)

        # Insert the <choice> into the tree, replacing the affected text
        _replace_text_range(
            line.text_runs, affected_runs,
            orig_start, orig_end,
            choice,
        )

        # Move affected notes to after the <choice>
        for note_info in affected_notes:
            _move_note_after(note_info, choice)


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

    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "replace":
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
        changes.append((oi1, oi2, oj1, oj2))

    # Merge overlapping/adjacent expanded ranges
    return _merge_changes(changes)


def _expand_left(text: str, pos: int) -> int:
    """Expand pos leftward to the start of the current word."""
    while pos > 0 and not text[pos - 1].isspace():
        pos -= 1
    return pos


def _expand_right(text: str, pos: int) -> int:
    """Expand pos rightward to the end of the current word."""
    while pos < len(text) and not text[pos].isspace():
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


def _build_choice_element(
    abbr_text: str,
    expan_text: str,
) -> etree._Element:
    """
    Build a <choice><abbr>...</abbr><expan>...</expan></choice> element.
    """
    choice = etree.Element(_tei("choice"))
    abbr = etree.SubElement(choice, _tei("abbr"))
    abbr.text = abbr_text
    expan = etree.SubElement(choice, _tei("expan"))
    expan.text = expan_text
    return choice


def _replace_text_range(
    all_runs:      list[TextRun],
    affected_runs: list[TextRun],
    orig_start:    int,
    orig_end:      int,
    choice:        etree._Element,
) -> None:
    """
    Replace a character range in the tree with a <choice> element.

    This is the most delicate operation. It needs to:
    1. Split text nodes at the change boundaries
    2. Remove the old text
    3. Insert the <choice> element at the correct position

    For simplicity, handles the common case where the change falls
    within a single text run. Multi-run changes (spanning inline tags)
    are handled by collapsing the affected range.
    """
    if len(affected_runs) == 1:
        run = affected_runs[0]
        _replace_in_single_run(run, orig_start, orig_end, choice)
    else:
        # Multi-run change: replace from first run's start to last run's end
        _replace_across_runs(affected_runs, orig_start, orig_end, choice)


def _replace_in_single_run(
    run:        TextRun,
    orig_start: int,
    orig_end:   int,
    choice:     etree._Element,
) -> None:
    """
    Replace a range within a single TextRun with a <choice> element.

    Splits the text node into: before + <choice> + after.
    """
    # Offsets within this run's text
    local_start = orig_start - run.plain_start
    local_end = orig_end - run.plain_start

    before_text = run.text[:local_start]
    after_text = run.text[local_end:]

    node = run.node
    if run.is_tail:
        # This is tail text — text after a closing tag
        # Set tail to the 'before' part, insert <choice> after node,
        # then set choice's tail to the 'after' part
        node.tail = before_text or None
        parent = node.getparent()
        idx = list(parent).index(node)
        parent.insert(idx + 1, choice)
        choice.tail = after_text or None
    else:
        # This is element text — text before first child
        node.text = before_text or None
        node.insert(0, choice)
        choice.tail = after_text or None


def _replace_across_runs(
    affected_runs: list[TextRun],
    orig_start:    int,
    orig_end:      int,
    choice:        etree._Element,
) -> None:
    """
    Replace a range spanning multiple TextRuns.

    This happens when an abbreviation spans across an inline element
    boundary, e.g. <hi>cōsen</hi>sus where the expansion is "consensus".

    Strategy: truncate the first run, remove intermediate content,
    truncate the last run, and insert <choice> at the first run's position.
    """
    first_run = affected_runs[0]
    last_run = affected_runs[-1]

    # Truncate first run
    local_start = orig_start - first_run.plain_start
    before_text = first_run.text[:local_start]

    # Truncate last run
    local_end = orig_end - last_run.plain_start
    after_text = last_run.text[local_end:]

    # Set up the first run's text
    if first_run.is_tail:
        first_run.node.tail = before_text or None
        parent = first_run.node.getparent()
        idx = list(parent).index(first_run.node)
        parent.insert(idx + 1, choice)
    else:
        first_run.node.text = before_text or None
        first_run.node.insert(0, choice)

    choice.tail = after_text or None

    # Remove intermediate runs' text content
    for run in affected_runs[1:-1]:
        if run.is_tail:
            run.node.tail = None
        else:
            run.node.text = None

    # Clear the last run's consumed portion
    if last_run is not first_run:
        if last_run.is_tail:
            last_run.node.tail = None
        else:
            last_run.node.text = None


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
# High-level pipeline integration
# ---------------------------------------------------------------------------

def process_tei_xml(
    xml_string:  str,
    run_pipeline_fn,   # callable: (lines_jsonl) → (expanded_dict, boundary_dict)
) -> str:
    """
    Full roundtrip: TEI XML string → expand abbreviations → TEI XML string.

    1. Parse XML
    2. Extract lines (main text + notes separately)
    3. Run pipeline on plain text lines
    4. Apply expansions and boundary predictions back to XML
    5. Serialize back to string

    run_pipeline_fn should accept a list of line dicts
    [{id, doc_id, source_sic, lang}] and return
    (expanded_dict, boundary_dict) where:
      - expanded_dict: {line_id: expanded_text}
      - boundary_dict: {line_id: next_line_id} for nonbreaking boundaries
    """
    # Parse
    parser = etree.XMLParser(remove_blank_text=False)
    tree = etree.ElementTree(etree.fromstring(xml_string.encode("utf-8"), parser))

    # Extract lines
    lines = extract_lines(tree)

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
    expanded_dict, boundary_dict = run_pipeline_fn(pipeline_rows)

    # Apply results
    apply_expansions(tree, lines, expanded_dict, boundary_dict)

    # Serialize
    return etree.tostring(
        tree.getroot(),
        encoding="unicode",
        pretty_print=False,
    )
