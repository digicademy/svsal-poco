"""
Analyse a differences JSONL produced by 05_check_compare.py.

For every "differences_found" entry it checks, field by field, whether the
only disagreement between A and B is whitespace.  Entries where *all* fields
are whitespace-only are counted separately.  For the remaining entries the
script collects the concrete non-whitespace character pairs that differ and
prints a frequency table.

Usage:
    python 06_analyze_diffs.py <diffs.jsonl>
                               [-r real_diffs.jsonl]
                               [-w ws_only.jsonl]

Optional output files:
  -r / --real-diffs FILE   Write entries with non-whitespace differences to FILE.
  -w / --ws-only FILE      Write whitespace-only-difference entries to FILE.
"""

import argparse
import collections
import contextlib
import difflib
import html
import json
import sys
import re
import unicodedata

# ── Same EQUIV map as 05_check_compare.py ────────────────────────────────────
EQUIV = {
    'u': 'v', 'v': 'u',
    'U': 'V', 'V': 'U',
    's': 'ſ', 'ſ': 's',
    'y': '⁊', '⁊': 'y',
    'z': 'ʒ', 'ʒ': 'z',
    'ꝗ': 'q', 'q': 'ꝗ',
}

DELIMITERS = {'⦃', '⦄'}


def nfc(s):
    if s is None:
        return ''
    return unicodedata.normalize('NFC', html.unescape(s))


def strip_delimiters(s):
    """Remove ⦃ and ⦄ bracket characters (keep their content)."""
    return ''.join(c for c in s if c not in DELIMITERS)


def strip_whitespace(s):
    return re.sub(r'\s+', '', s)


def chars_equiv(a, b):
    return a == b or EQUIV.get(a) == b


def whitespace_only_diff(a_raw, b_raw):
    """Return True when A and B differ only in whitespace (after delimiter removal)."""
    a = strip_whitespace(strip_delimiters(nfc(a_raw)))
    b = strip_whitespace(strip_delimiters(nfc(b_raw)))

    if len(a) != len(b):
        return False
    return all(chars_equiv(ca, cb) for ca, cb in zip(a, b))


def collect_diff_chars(a_raw, b_raw):
    """
    Yield (char_from_A, char_from_B) pairs for every position where A and B
    disagree, using a character-level SequenceMatcher.  Delimiter characters
    and whitespace are stripped before alignment so we look at content only.
    """
    a = strip_whitespace(strip_delimiters(nfc(a_raw)))
    b = strip_whitespace(strip_delimiters(nfc(b_raw)))

    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            continue
        a_chunk = a[i1:i2]
        b_chunk = b[j1:j2]
        # Pair up as many chars as possible; pad the shorter side with ''
        for ca, cb in zip(a_chunk.ljust(max(len(a_chunk), len(b_chunk))),
                          b_chunk.ljust(max(len(a_chunk), len(b_chunk)))):
            ca = ca if ca != ' ' else ''   # ljust pads with spaces → use ''
            cb = cb if cb != ' ' else ''
            if not chars_equiv(ca or '', cb or ''):
                yield (ca, cb)


def char_repr(c):
    if c == '':
        return '<absent>'
    cp = ord(c)
    name = unicodedata.name(c, '?')
    return f"U+{cp:04X} '{c}' ({name})"


def main(path, real_diffs_path=None, ws_only_path=None):
    total = 0
    ws_only_entries = 0
    real_diff_entries = 0
    diff_char_counter = collections.Counter()   # (repr_A, repr_B) → count

    @contextlib.contextmanager
    def open_optional(p):
        if p:
            with open(p, 'w', encoding='utf-8') as fh:
                yield fh
        else:
            yield None

    def write_line(fh, entry):
        if fh is not None:
            fh.write(json.dumps(entry, ensure_ascii=False) + '\n')

    with open(path, encoding='utf-8') as fh_in, \
         open_optional(real_diffs_path) as fh_real, \
         open_optional(ws_only_path) as fh_ws:

        for raw in fh_in:
            raw = raw.strip()
            if not raw:
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if entry.get('status') != 'differences_found':
                continue

            total += 1
            details = entry.get('details') or {}

            entry_ws_only = True
            for field, pair in details.items():
                a_val = pair.get('A') or ''
                b_val = pair.get('B') or ''
                if not whitespace_only_diff(a_val, b_val):
                    entry_ws_only = False
                    for ca, cb in collect_diff_chars(a_val, b_val):
                        diff_char_counter[(char_repr(ca), char_repr(cb))] += 1

            if entry_ws_only:
                ws_only_entries += 1
                write_line(fh_ws, entry)
            else:
                real_diff_entries += 1
                write_line(fh_real, entry)

    print(f"Total 'differences_found' entries : {total}")
    print(f"  Whitespace-only differences     : {ws_only_entries}")
    print(f"  Entries with real differences   : {real_diff_entries}")
    if real_diffs_path:
        print(f"  Real-diffs written to           : {real_diffs_path}")
    if ws_only_path:
        print(f"  WS-only written to              : {ws_only_path}")

    if diff_char_counter:
        print("\nNon-whitespace differing character pairs (A → B), by frequency:")
        print(f"  {'Count':>6}  {'A':45}  {'B'}")
        print(f"  {'-'*6}  {'-'*45}  {'-'*45}")
        for (ra, rb), count in diff_char_counter.most_common():
            print(f"  {count:>6}  {ra:45}  {rb}")
    else:
        print("\nNo non-whitespace differences found.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyse differences JSONL from 05_check_compare.py.')
    parser.add_argument('diffs_jsonl', help='Input diffs JSONL file')
    parser.add_argument('-r', '--real-diffs', metavar='FILE',
                        help='Write entries with non-whitespace differences to FILE')
    parser.add_argument('-w', '--ws-only', metavar='FILE',
                        help='Write whitespace-only-difference entries to FILE')
    args = parser.parse_args()
    main(args.diffs_jsonl, args.real_diffs, args.ws_only)
