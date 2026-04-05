import json
import sqlite3
import sys
import unicodedata


NFC = unicodedata.normalize


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


def strings_match(s, t):
    return len(s) == len(t) and all(chars_match(a, b) for a, b in zip(s, t))


def _contains_abbr_chars(s):
    """Return True if s contains any abbreviation-related characters
    (non-ASCII letters, combining marks, private-use)."""
    for c in s:
        cat = unicodedata.category(c)
        if cat.startswith('M') or cat == 'Co':
            return True
        if not c.isascii() and cat.startswith('L'):
            return True
    return False


def last_plain_char(s):
    for c in reversed(s):
        cat = unicodedata.category(c)
        if cat.startswith('M') or cat == 'Co':
            continue
        if not c.isascii() and cat.startswith('L'):
            continue
        return c
    return None


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


def _apply_shift(l0_field, l1_field, shift):
    """Move *shift* characters between the end of l0_field and the start of
    l1_field.  Positive shift: l0 is too long, move chars to l1.  Negative:
    l1 is too long, move chars to l0.  Returns (new_l0, new_l1) or None if
    the shift is out of range."""
    if shift > 0 and shift <= len(l0_field or ''):
        return (l0_field[:-shift],
                l0_field[-shift:] + (l1_field or ''))
    elif shift < 0 and (-shift) <= len(l1_field or ''):
        mv = -shift
        return ((l0_field or '') + l1_field[:mv],
                l1_field[mv:])
    return None


def _try_fix_corr_independently(l0_src_sic, l0_src_corr, l0_tgt_sic, l0_tgt_corr,
                                 l1_src_sic, l1_src_corr, l1_tgt_sic, l1_tgt_corr):
    """Fix _corr break position to match the physical line break defined by _sic.

    The physical line break is defined by the _sic fields.  When _corr has
    a different word at the boundary (due to an editorial sic/corr correction),
    the break must be re-mapped to the corresponding position in the corrected
    word.  This uses the _sic break as the authoritative reference:

      1. Map the _sic break position into the _corr source word.
      2. If _corr source and _corr target also differ (abbreviation expansion),
         map again into the _corr target word.
      3. Compare the computed breaks with the current breaks in source_corr
         and target_corr; if either differs, return the corrected values.

    Returns (new_l0_src_corr, new_l1_src_corr,
             new_l0_tgt_corr, new_l1_tgt_corr) or None.
    """
    if not l0_src_sic or not l1_src_sic:
        return None
    if not l0_src_corr or not l1_src_corr:
        return None

    l0_src_sic_last   = l0_src_sic.rsplit(None, 1)[-1]   if l0_src_sic.split()   else l0_src_sic
    l1_src_sic_first  = l1_src_sic.split(None, 1)[0]     if l1_src_sic.split()   else l1_src_sic
    l0_src_corr_last  = l0_src_corr.rsplit(None, 1)[-1]   if l0_src_corr.split()  else l0_src_corr
    l1_src_corr_first = l1_src_corr.split(None, 1)[0]     if l1_src_corr.split()  else l1_src_corr

    full_src_sic  = l0_src_sic_last  + l1_src_sic_first
    full_src_corr = l0_src_corr_last + l1_src_corr_first

    # If _sic and _corr boundary words are identical, any fix was
    # already handled by Sub-types A/B/C — nothing to do here.
    if full_src_sic == full_src_corr:
        return None

    # Step 1: map the _sic break into the _corr source word.
    sic_break = len(l0_src_sic_last)
    corr_break = _find_tgt_break_with_fallback(
        full_src_sic, full_src_corr, sic_break
    )
    if corr_break is None:
        return None

    # --- Fix source_corr ---
    src_corr_shift = len(l0_src_corr_last) - corr_break
    if src_corr_shift != 0:
        src_fix = _apply_shift(l0_src_corr, l1_src_corr, src_corr_shift)
    else:
        src_fix = None

    # --- Fix target_corr ---
    tgt_fix = None
    if l0_tgt_corr and l1_tgt_corr:
        l0_tgt_corr_last  = l0_tgt_corr.rsplit(None, 1)[-1]  if l0_tgt_corr.split()  else l0_tgt_corr
        l1_tgt_corr_first = l1_tgt_corr.split(None, 1)[0]    if l1_tgt_corr.split()  else l1_tgt_corr
        full_tgt_corr = l0_tgt_corr_last + l1_tgt_corr_first

        # Step 2: map _corr source break into _corr target word.
        if full_src_corr == full_tgt_corr:
            tgt_corr_break = corr_break
        else:
            tgt_corr_break = _find_tgt_break_with_fallback(
                full_src_corr, full_tgt_corr, corr_break
            )

        if tgt_corr_break is not None:
            tgt_corr_shift = len(l0_tgt_corr_last) - tgt_corr_break
            if tgt_corr_shift != 0:
                tgt_fix = _apply_shift(l0_tgt_corr, l1_tgt_corr, tgt_corr_shift)

    # If neither field needs fixing, return None.
    if src_fix is None and tgt_fix is None:
        return None

    new_l0_src_corr = src_fix[0] if src_fix else l0_src_corr
    new_l1_src_corr = src_fix[1] if src_fix else l1_src_corr
    new_l0_tgt_corr = tgt_fix[0] if tgt_fix else l0_tgt_corr
    new_l1_tgt_corr = tgt_fix[1] if tgt_fix else l1_tgt_corr

    return (new_l0_src_corr, new_l1_src_corr,
            new_l0_tgt_corr, new_l1_tgt_corr)


def normalize_record(d):
    return {k: NFC('NFC', v) if isinstance(v, str) else v for k, v in d.items()}


def fix_jsonl_file(input_path, output_path):
    BATCH_SIZE = 50_000

    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE lines (
            id                    TEXT PRIMARY KEY,
            source_sic            TEXT,
            source_corr           TEXT,
            target_sic            TEXT,
            target_corr           TEXT,
            nonbreaking_next_line TEXT,
            raw_json              TEXT
        )
    """)

    # ------------------------------------------------------------------ #
    # Pass 1                                                             #
    # ------------------------------------------------------------------ #
    print(f"Pass 1 - indexing {input_path} ...")
    batch = []
    total = 0

    with open(input_path, 'r', encoding='utf-8') as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                d = normalize_record(json.loads(raw))
            except json.JSONDecodeError:
                continue

            batch.append((
                d.get('id', ''),
                d.get('source_sic', ''),
                d.get('source_corr', ''),
                d.get('target_sic', ''),
                d.get('target_corr', ''),
                d.get('nonbreaking_next_line', ''),
                raw,
            ))
            total += 1

            if len(batch) >= BATCH_SIZE:
                cur.executemany(
                    "INSERT OR REPLACE INTO lines VALUES (?,?,?,?,?,?,?)", batch
                )
                conn.commit()
                batch = []
                print(f"  ... {total:,} lines indexed", end='\r')

    if batch:
        cur.executemany(
            "INSERT OR REPLACE INTO lines VALUES (?,?,?,?,?,?,?)", batch
        )
        conn.commit()

    print(f"  ... {total:,} lines indexed. Done.")

    # ------------------------------------------------------------------ #
    # Pass 2                                                             #
    # ------------------------------------------------------------------ #
    print("Pass 2 - detecting shifted prefixes ...")

    cur.execute("""
        CREATE TABLE corrections (
            id               TEXT PRIMARY KEY,
            new_source_corr  TEXT,
            new_target_sic   TEXT,
            new_target_corr  TEXT
        )
    """)

    cur.execute("""
        SELECT l0.id,
               l0.source_sic,
               l0.source_corr,
               l0.target_sic,
               l0.target_corr,
               l0.nonbreaking_next_line,
               l1.source_sic,
               l1.source_corr,
               l1.target_sic,
               l1.target_corr
        FROM   lines l0
        JOIN   lines l1 ON l1.id = l0.nonbreaking_next_line
        WHERE  l0.nonbreaking_next_line != ''
    """)

    fixes_a = 0
    fixes_b = 0
    fixes_c = 0
    fixes_d = 0
    corrections_batch = []

    # pending: id -> (new_source_corr, new_target_sic, new_target_corr)
    pending = {}

    for (l0_id, l0_src_sic, l0_src_corr, l0_tgt_sic, l0_tgt_corr,
         l1_id, l1_src_sic, l1_src_corr, l1_tgt_sic, l1_tgt_corr) in cur.fetchall():

        if l0_id in pending:
            l0_src_corr, l0_tgt_sic, l0_tgt_corr = pending[l0_id]
        if l1_id in pending:
            l1_src_corr, l1_tgt_sic, l1_tgt_corr = pending[l1_id]

        if not l1_src_sic or not l1_tgt_sic:
            continue

        x = l1_src_sic[0]
        x_is_ascii = x.isascii() and x.isprintable()

        sic_was_fixed = False

        # ------------------------------------------------------------------
        # Sub-type A: Actual target prefix was mistakenly shifted to previous line
        # ------------------------------------------------------------------
        if x_is_ascii and not chars_match(l1_tgt_sic[0], x):
            prefix_len_a = None
            l0_src_lpc = last_plain_char(l0_src_sic)
            for plen in range(1, min(6, len(l1_tgt_sic))):
                if chars_match(l1_tgt_sic[plen], x):
                    if l0_src_lpc is None or chars_match(l1_tgt_sic[plen - 1], l0_src_lpc):
                        src_suffix = l0_src_sic[-plen:] if len(l0_src_sic) >= plen else l0_src_sic
                        tgt_prefix = l1_tgt_sic[:plen]
                        if len(src_suffix) == plen and strings_match(src_suffix, tgt_prefix):
                            prefix_len_a = plen
                            break

            # Validate: check that the proposed move brings the target
            # break closer to where _find_tgt_break says it should be.
            if prefix_len_a is not None:
                l0_src_last_w = l0_src_sic.rsplit(None, 1)[-1] if l0_src_sic.split() else l0_src_sic
                l1_src_first_w = l1_src_sic.split(None, 1)[0] if l1_src_sic.split() else l1_src_sic
                l0_tgt_last_w = l0_tgt_sic.rsplit(None, 1)[-1] if l0_tgt_sic.split() else l0_tgt_sic
                l1_tgt_first_w = l1_tgt_sic.split(None, 1)[0] if l1_tgt_sic.split() else l1_tgt_sic

                full_src_w = l0_src_last_w + l1_src_first_w
                full_tgt_w = l0_tgt_last_w + l1_tgt_first_w

                if full_src_w != full_tgt_w:
                    ideal_break = _find_tgt_break_with_fallback(
                        full_src_w, full_tgt_w, len(l0_src_last_w)
                    )
                    if ideal_break is not None:
                        current_break = len(l0_tgt_last_w)
                        proposed_break = current_break + prefix_len_a
                        # Only accept if the proposed break is closer to
                        # or equal to the ideal break
                        if abs(proposed_break - ideal_break) > abs(current_break - ideal_break):
                            prefix_len_a = None

            # Apply the fix if validation passed
            if prefix_len_a is not None:
                new_l0_tgt_sic = (l0_tgt_sic or '') + l1_tgt_sic[:prefix_len_a]
                new_l1_tgt_sic = l1_tgt_sic[prefix_len_a:]

                if l1_tgt_corr and len(l1_tgt_corr) >= prefix_len_a:
                    new_l0_tgt_corr = (l0_tgt_corr or '') + l1_tgt_corr[:prefix_len_a]
                    new_l1_tgt_corr = l1_tgt_corr[prefix_len_a:]
                else:
                    new_l0_tgt_corr = l0_tgt_corr
                    new_l1_tgt_corr = l1_tgt_corr

                l0_tgt_sic, l0_tgt_corr = new_l0_tgt_sic, new_l0_tgt_corr
                l1_tgt_sic, l1_tgt_corr = new_l1_tgt_sic, new_l1_tgt_corr
                corrections_batch.append((l0_id, l0_src_corr, new_l0_tgt_sic, new_l0_tgt_corr))
                corrections_batch.append((l1_id, l1_src_corr, new_l1_tgt_sic, new_l1_tgt_corr))
                pending[l0_id] = (l0_src_corr, new_l0_tgt_sic, new_l0_tgt_corr)
                pending[l1_id] = (l1_src_corr, new_l1_tgt_sic, new_l1_tgt_corr)
                fixes_a += 1
                sic_was_fixed = True

        # ------------------------------------------------------------------
        # Sub-type B: Actual target suffix of the previous line was shifted to the present line
        # ------------------------------------------------------------------
        if not sic_was_fixed:
            shifted = None
            if x_is_ascii and not chars_match(l1_tgt_sic[0], x):
                for plen in range(5, 0, -1):
                    if (l0_tgt_sic and len(l0_tgt_sic) >= plen
                            and strings_match(l1_src_sic[:plen], l0_tgt_sic[-plen:])):
                        shifted = plen
                        break

            # Validate: check direction against _find_tgt_break
            if shifted is not None:
                l0_src_last_w = l0_src_sic.rsplit(None, 1)[-1] if l0_src_sic.split() else l0_src_sic
                l1_src_first_w = l1_src_sic.split(None, 1)[0] if l1_src_sic.split() else l1_src_sic
                l0_tgt_last_w = l0_tgt_sic.rsplit(None, 1)[-1] if l0_tgt_sic.split() else l0_tgt_sic
                l1_tgt_first_w = l1_tgt_sic.split(None, 1)[0] if l1_tgt_sic.split() else l1_tgt_sic

                full_src_w = l0_src_last_w + l1_src_first_w
                full_tgt_w = l0_tgt_last_w + l1_tgt_first_w

                if full_src_w != full_tgt_w:
                    ideal_break = _find_tgt_break_with_fallback(
                        full_src_w, full_tgt_w, len(l0_src_last_w)
                    )
                    if ideal_break is not None:
                        current_break = len(l0_tgt_last_w)
                        proposed_break = current_break - shifted
                        if abs(proposed_break - ideal_break) > abs(current_break - ideal_break):
                            shifted = None

            # Apply the fix if validation passed
            if shifted is not None:
                new_l0_tgt_sic = l0_tgt_sic[:-shifted]
                new_l1_tgt_sic = l0_tgt_sic[-shifted:] + l1_tgt_sic

                if l0_tgt_corr and len(l0_tgt_corr) >= shifted:
                    new_l0_tgt_corr = l0_tgt_corr[:-shifted]
                    new_l1_tgt_corr = l0_tgt_corr[-shifted:] + (l1_tgt_corr or '')
                else:
                    new_l0_tgt_corr = l0_tgt_corr
                    new_l1_tgt_corr = l1_tgt_corr

                l0_tgt_sic, l0_tgt_corr = new_l0_tgt_sic, new_l0_tgt_corr
                l1_tgt_sic, l1_tgt_corr = new_l1_tgt_sic, new_l1_tgt_corr
                corrections_batch.append((l0_id, l0_src_corr, new_l0_tgt_sic, new_l0_tgt_corr))
                corrections_batch.append((l1_id, l1_src_corr, new_l1_tgt_sic, new_l1_tgt_corr))
                pending[l0_id] = (l0_src_corr, new_l0_tgt_sic, new_l0_tgt_corr)
                pending[l1_id] = (l1_src_corr, new_l1_tgt_sic, new_l1_tgt_corr)
                fixes_b += 1
                sic_was_fixed = True

        # ------------------------------------------------------------------
        # Sub-type C: there is not no identical suffix/prefix (an expansion is involved),
        #             that means we must calculate the correct break position based on the
        #             full word and see if we have to move chars accordingly.
        # ------------------------------------------------------------------
        if not sic_was_fixed and l0_src_sic and l0_tgt_sic:
            # identify two semi-tokens around break in source
            l0_src_last  = l0_src_sic.rsplit(None, 1)[-1] if l0_src_sic.split() else l0_src_sic
            l1_src_first = l1_src_sic.split(None, 1)[0]   if l1_src_sic.split() else l1_src_sic
            # identify two semi-tokens around break in target
            l0_tgt_last  = l0_tgt_sic.rsplit(None, 1)[-1] if l0_tgt_sic.split() else l0_tgt_sic
            l1_tgt_first = l1_tgt_sic.split(None, 1)[0]   if l1_tgt_sic.split() else l1_tgt_sic

            # Guard: if l1's source and target first-words start with
            # different plain characters AND neither is a suffix/prefix
            # of the other, this is likely a sic/corr correction rather
            # than shifted expansion — skip Sub-type C.
            l1_src_initial = l1_src_first[0] if l1_src_first else None
            l1_tgt_initial = l1_tgt_first[0] if l1_tgt_first else None

            boundary_is_corrected = False
            if (l1_src_initial and l1_tgt_initial
                    and l1_src_initial.isascii() and l1_tgt_initial.isascii()
                    and not chars_match(l1_src_initial, l1_tgt_initial)):
                # Check if l1_tgt_first looks like l1_src_first with
                # characters prepended (spillover from expansion)
                spillover_prefix = False
                for k in range(1, min(4, len(l1_tgt_first))):
                    if chars_match(l1_tgt_first[k], l1_src_initial):
                        spillover_prefix = True
                        break

                is_expansion = (
                    _contains_abbr_chars(l1_src_first)
                    or (_contains_abbr_chars(l0_src_last)
                        and len(l1_tgt_first) > len(l1_src_first))
                    or l1_tgt_first.endswith(l1_src_first)
                    or l1_src_first.endswith(l1_tgt_first)
                    or spillover_prefix
                )
                if not is_expansion:
                    boundary_is_corrected = True

            if not boundary_is_corrected:

                # get full words (abbreviated/source and expanded/target)
                full_src_word = l0_src_last + l1_src_first
                full_tgt_word = l0_tgt_last + l1_tgt_first
                # get relative position of break in source (for reference)
                src_break     = len(l0_src_last)

                # if the words are identical, any mistake in break positioning
                # would necessarily have triggered type A or B above, so we
                # proceed only if the words are not identical. (This is a safety gate.)
                if full_src_word != full_tgt_word:
                    # call function to determine break position
                    tgt_break = _find_tgt_break_with_fallback(
                        full_src_word, full_tgt_word, src_break
                    )

                    # we proceed only if the current break in target is not
                    # where our calculation says it should be (and also not out of range);
                    if tgt_break is not None and tgt_break != len(l0_tgt_last):

                        current_break = len(l0_tgt_last)

                        # Safety gate: only suppress the fix if the current break
                        # already aligns with the source line start AND the
                        # computed break does not.
                        computed_initial_matches_src = (
                            tgt_break < len(full_tgt_word)
                            and chars_match(full_tgt_word[tgt_break], l1_src_first[0])
                        )
                        current_initial_matches_src = (
                            current_break < len(full_tgt_word)
                            and chars_match(full_tgt_word[current_break], l1_src_first[0])
                        )
                        suppress = (current_initial_matches_src
                                    and not computed_initial_matches_src)

                        if not suppress:

                        # if not initials_match:

                            # how is the break currently off?
                            shift = len(l0_tgt_last) - tgt_break

                            # if the current break is off to the left
                            # and we have chars to move from l0 to l1, do it
                            if shift > 0 and shift <= len(l0_tgt_sic):
                                new_l0_tgt_sic = l0_tgt_sic[:-shift]
                                new_l1_tgt_sic = l0_tgt_sic[-shift:] + l1_tgt_sic

                                # if we also have target_corr to move, do it;
                                # otherwise just adjust target_sic
                                if l0_tgt_corr and len(l0_tgt_corr) >= shift:
                                    new_l0_tgt_corr = l0_tgt_corr[:-shift]
                                    new_l1_tgt_corr = l0_tgt_corr[-shift:] + (l1_tgt_corr or '')
                                else:
                                    new_l0_tgt_corr = l0_tgt_corr
                                    new_l1_tgt_corr = l1_tgt_corr

                            # if the current break is off to the right
                            # and we have chars to move from l1 to l0, do it
                            elif shift < 0 and (-shift) <= len(l1_tgt_sic):
                                mv = -shift
                                new_l0_tgt_sic = (l0_tgt_sic or '') + l1_tgt_sic[:mv]
                                new_l1_tgt_sic = l1_tgt_sic[mv:]

                                # if we also have target_corr to move, do it;
                                # otherwise just adjust target_sic
                                if l1_tgt_corr and len(l1_tgt_corr) >= mv:
                                    new_l0_tgt_corr = (l0_tgt_corr or '') + l1_tgt_corr[:mv]
                                    new_l1_tgt_corr = l1_tgt_corr[mv:]
                                else:
                                    new_l0_tgt_corr = l0_tgt_corr
                                    new_l1_tgt_corr = l1_tgt_corr

                            # if the shift is out of range, we give up on this line pair
                            # (for type C) and move on to type D
                            else:
                                shift = 0

                            if shift != 0:
                                l0_tgt_sic, l0_tgt_corr = new_l0_tgt_sic, new_l0_tgt_corr
                                l1_tgt_sic, l1_tgt_corr = new_l1_tgt_sic, new_l1_tgt_corr
                                corrections_batch.append((l0_id, l0_src_corr, new_l0_tgt_sic, new_l0_tgt_corr))
                                corrections_batch.append((l1_id, l1_src_corr, new_l1_tgt_sic, new_l1_tgt_corr))
                                pending[l0_id] = (l0_src_corr, new_l0_tgt_sic, new_l0_tgt_corr)
                                pending[l1_id] = (l1_src_corr, new_l1_tgt_sic, new_l1_tgt_corr)
                                fixes_c += 1
                                sic_was_fixed = True

        # ------------------------------------------------------------------
        # Sub-type D: _corr-only misalignment.
        # ------------------------------------------------------------------
        corr_fix = _try_fix_corr_independently(
            l0_src_sic, l0_src_corr, l0_tgt_sic, l0_tgt_corr,
            l1_src_sic, l1_src_corr, l1_tgt_sic, l1_tgt_corr
        )
        if corr_fix is not None:
            new_l0_src_corr, new_l1_src_corr, new_l0_tgt_corr, new_l1_tgt_corr = corr_fix

            # Preserve _sic values (possibly already corrected by A/B/C).
            new_l0_tgt_sic = l0_tgt_sic
            new_l1_tgt_sic = l1_tgt_sic

            l0_src_corr, l0_tgt_corr = new_l0_src_corr, new_l0_tgt_corr
            l1_src_corr, l1_tgt_corr = new_l1_src_corr, new_l1_tgt_corr

            corrections_batch.append((l0_id, new_l0_src_corr, new_l0_tgt_sic, new_l0_tgt_corr))
            corrections_batch.append((l1_id, new_l1_src_corr, new_l1_tgt_sic, new_l1_tgt_corr))
            pending[l0_id] = (new_l0_src_corr, new_l0_tgt_sic, new_l0_tgt_corr)
            pending[l1_id] = (new_l1_src_corr, new_l1_tgt_sic, new_l1_tgt_corr)
            fixes_d += 1

        if len(corrections_batch) >= BATCH_SIZE * 2:
            cur.executemany(
                "INSERT OR REPLACE INTO corrections VALUES (?,?,?,?)",
                corrections_batch
            )
            conn.commit()
            corrections_batch = []

    if corrections_batch:
        cur.executemany(
            "INSERT OR REPLACE INTO corrections VALUES (?,?,?,?)",
            corrections_batch
        )
        conn.commit()

    fixes_total = fixes_a + fixes_b + fixes_c + fixes_d
    print(f"  ... {fixes_total:,} line-pair fixes identified"
          f" (A={fixes_a:,}, B={fixes_b:,}, C={fixes_c:,}, D={fixes_d:,}).")

    # ------------------------------------------------------------------ #
    # Pass 3                                                               #
    # ------------------------------------------------------------------ #
    print(f"Pass 3 - writing corrected output to {output_path} ...")

    written = 0
    with open(input_path, 'r', encoding='utf-8') as fh_in, \
         open(output_path, 'w', encoding='utf-8') as fh_out:

        for raw in fh_in:
            raw = raw.strip()
            if not raw:
                fh_out.write('\n')
                continue
            try:
                d = normalize_record(json.loads(raw))
            except json.JSONDecodeError:
                fh_out.write(raw + '\n')
                continue

            lid = d.get('id', '')
            if lid:
                cur.execute(
                    "SELECT new_source_corr, new_target_sic, new_target_corr "
                    "FROM corrections WHERE id=?",
                    (lid,)
                )
                row = cur.fetchone()
                if row:
                    d['source_corr'] = row[0]
                    d['target_sic']  = row[1]
                    d['target_corr'] = row[2]

            fh_out.write(json.dumps(d, ensure_ascii=False) + '\n')
            written += 1

            if written % 100_000 == 0:
                print(f"  ... {written:,} lines written", end='\r')

    print(f"  ... {written:,} lines written. Done.")

    conn.close()
    print(f"\nFinished. {fixes_total:,} pairs corrected"
          f" (A={fixes_a:,}, B={fixes_b:,}, C={fixes_c:,}, D={fixes_d:,})."
          f" Output: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python 02_adjust_shifted_lbs.py <input.jsonl> <output.jsonl>")
        sys.exit(1)

    fix_jsonl_file(sys.argv[1], sys.argv[2])

