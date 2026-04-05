import json
import difflib
import unicodedata
from typing import Optional

NFC = unicodedata.normalize

# Grapheme normalization map: source special char → its plain-text expansion.
# Used both to normalize source for alignment and to detect restorable positions.
NORM_MAP: dict[str, str] = {
    'ſ': 's',   # long s
    'æ': 'ae',  # ae ligature
    'Æ': 'AE',  # AE ligature
    'œ': 'oe',  # oe ligature
    'Œ': 'OE',  # OE ligature
}


def process_file(input_path: str, output_path: str) -> None:
    with open(input_path, 'r', encoding='utf-8') as infile, \
      open(output_path, 'w', encoding='utf-8') as outfile:

      for raw in infile:
          raw = raw.strip()
          if not raw:
              continue

          try:
              data = json.loads(raw)

          except json.JSONDecodeError:
              continue

          # For "target_sic" and "target_corr", restore ſ, æ, Æ, œ, and Œ
          # based on alignment with "source_sic" and "source_corr" respectively.
          source_sic = data.get('source_sic', '')
          target_sic = data.get('target_sic', '')
          if source_sic and target_sic:
              data['target_sic'] = align_and_restore(source_sic, target_sic)

          source_corr = data.get('source_corr', '')
          target_corr = data.get('target_corr', '')
          if source_corr and target_corr:
              data['target_corr'] = align_and_restore(source_corr, target_corr)

          # NFC-normalize the entire record
          for key in data:
              if isinstance(data[key], str):
                  data[key] = NFC('NFC', data[key])

          # Write the modified record to the output file
          outfile.write(json.dumps(data, ensure_ascii=False) + '\n')


def align_and_restore(source: str, target: str) -> str:
    """
    Given a source string (which may contain ſ, æ, Æ, œ, Œ) and a target
    string where those characters have been normalized (ſ→s, æ→ae, Æ→AE,
    œ→oe, Œ→OE), restore the original source graphemes in the target
    wherever the alignment supports it.

    Handles both 1-to-1 (ſ→s) and 1-to-2 (æ→ae) source-to-target mappings.
    Restoration is source-driven: only graphemes present as special characters
    in the source are candidates for restoration in the target.
    """
    if not source or not target:
        return target

    # Build a normalized source string and record, for each position in that
    # normalized string, which original source index it came from.
    source_norm_chars: list[str] = []
    norm_to_orig: list[int] = []
    for orig_idx, ch in enumerate(source):
        expansion = NORM_MAP.get(ch, ch)
        for ec in expansion:
            source_norm_chars.append(ec)
            norm_to_orig.append(orig_idx)
    source_norm = "".join(source_norm_chars)

    # Normalize target for alignment (ſ→s; ligatures already expanded in target)
    target_norm = "".join('s' if c == 'ſ' else c for c in target)

    # Build a character-level alignment using SequenceMatcher
    matcher = difflib.SequenceMatcher(None, source_norm, target_norm, autojunk=False)

    # For each target position, find the corresponding source_norm position
    # (or None if the target character was inserted with no source counterpart).
    target_norm_map: list[Optional[int]] = [None] * len(target_norm)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op in ("equal", "replace"):
            for si, ti in zip(range(i1, i2), range(j1, j2)):
                target_norm_map[ti] = si
        # "insert": no source counterpart → stays None
        # "delete": source chars with no target counterpart → nothing to do

    # Rebuild target, restoring original source graphemes where supported.
    result: list[str] = []
    ti = 0
    while ti < len(target):
        norm_pos = target_norm_map[ti]
        if norm_pos is not None:
            orig_pos = norm_to_orig[norm_pos]
            src_ch = source[orig_pos]
            expansion = NORM_MAP.get(src_ch)
            if expansion is not None:
                n = len(expansion)
                if target[ti : ti + n] == expansion:
                    # Source had a special grapheme whose expansion matches
                    # the next n characters in target → restore the original.
                    result.append(src_ch)
                    ti += n
                    continue
        result.append(target[ti])
        ti += 1

    return "".join(result)


# Keep the old name as an alias for backwards compatibility.
align_and_restore_long_s = align_and_restore


def restore_long_s_in_record(record: dict) -> dict:
    """
    For a single JSONL record, restore ſ, æ, Æ, œ, and Œ in target_sic and
    target_corr using source_sic and source_corr respectively as the reference.
    """
    record = record.copy()

    pairs = [
        ("source_sic", "target_sic"),
        ("source_corr", "target_corr"),
    ]

    for source_key, target_key in pairs:
        source_text = record.get(source_key, "")
        target_text = record.get(target_key, "")
        if source_text and target_text:
            record[target_key] = align_and_restore(source_text, target_text)

    return record


def restore_long_s_in_jsonl(input_path: str, output_path: str) -> None:
    """
    Process a JSONL file, restoring ſ, æ, Æ, œ, and Œ in target fields of
    every record.
    """
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                fout.write("\n")
                continue
            record = json.loads(line)
            restored = restore_long_s_in_record(record)
            fout.write(json.dumps(restored, ensure_ascii=False) + "\n")


# ── quick smoke test ──────────────────────────────────────────────────────────
def smoke_test():
    samples = [
        {
            "source_sic":  "ego autem videns eius ſalutarem ⦃intentionẽ",
            "source_corr": "ego autem videns eius ſalutarem intentionen",
            "target_sic":  "ego autem videns eius salutarem intentionen",
            "target_corr": "ego autem videns eius salutarem intentionen",
        },
        {
            "source_sic":  "do⦄ illi ſumptus neceſſarios ad literis ⦃incumbẽ",
            "source_corr": "do⦄ illi ſumptus neceſſarios ad literis ⦃incumbẽ",
            "target_sic":  "do illi sumptus necessarios ad literis incum",
            "target_corr": "do illi sumptus necessarios ad literis incum",
        },
        {
            "source_sic":  "dum⦄, adiecto pacto de repetendo, ſi poſtea ⦃nõ⦄",
            "source_corr": "dum⦄, adiecto pacto de repetendo, ſi poſtea ⦃nõ⦄",
            "target_sic":  "bendum, adiecto pacto de repetendo, si postea non",
            "target_corr": "bendum, adiecto pacto de repetendo, si postea non",
        },
        {
            "source_sic":  "ingretiatur ⦃religionẽ⦄: An id fieri licitè poſsit.",
            "source_corr": "ingretiatur ⦃religionẽ⦄: An id fieri licitè poſsit.",
            "target_sic":  "ingretiatur religionem: An id fieri licitè possit.",
            "target_corr": "ingretiatur religionem: An id fieri licitè possit.",
        },
        # ── ligature tests ──────────────────────────────────────────────────
        {   # source_sic has æ → restored; source_corr has plain ae → not restored
            "source_sic":  "cælum",
            "source_corr": "caelum",
            "target_sic":  "caelum",
            "target_corr": "caelum",
        },
        {   # source_sic has œ → restored; source_corr has plain oe → not restored
            "source_sic":  "pœna",
            "source_corr": "poena",
            "target_sic":  "poena",
            "target_corr": "poena",
        },
        {   # both ſ and æ present → both restored
            "source_sic":  "ſæculo",
            "source_corr": "ſæculo",
            "target_sic":  "saeculo",
            "target_corr": "saeculo",
        },
        {   # uppercase Æ → restored
            "source_sic":  "Æneas",
            "source_corr": "Æneas",
            "target_sic":  "AEneas",
            "target_corr": "AEneas",
        },
    ]

    for rec in samples:
        restored = restore_long_s_in_record(rec)
        print("target_sic  before:", rec["target_sic"])
        print("target_sic  after :", restored["target_sic"])
        print("target_corr before:", rec["target_corr"])
        print("target_corr after :", restored["target_corr"])
        print()

if __name__ == "__main__":
    # if first argument is "test", run the smoke test instead of processing files
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        sys.argv.pop(1)  # remove "test" from arguments
        print("Running smoke test...")
        smoke_test()
        sys.exit(0)

    if len(sys.argv) < 3:
        print("Usage: python 03_denormalize_long_s.py <input.jsonl> <output.jsonl>")
        print("  input.jsonl: The input file")
        print("  output.jsonl: The file to write results to")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    process_file(input_file, output_file)
