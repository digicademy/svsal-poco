import json
import sys
import unicodedata
from collections import deque


NFC = unicodedata.normalize

# Characters treated as interchangeable when comparing source and target.
# Kept in sync with adjust_shifted_lbs.py.
EQUIV = {
    'u': 'v', 'v': 'u',
    'U': 'V', 'V': 'U',
    's': 'ſ', 'ſ': 's',
    'y': '⁊', '⁊': 'y',
    'z': 'ʒ', 'ʒ': 'z',
    'ꝗ': 'q', 'q': 'ꝗ',
}


def chars_match(a, b):
    return a == b or EQUIV.get(a) == b


def process_jsonl(input_path, output_path):
    """
    Reads a JSONL file and reports lines where source_sic and target_sic begin
    with characters that differ (even under EQUIV equivalences), and where a
    preceding nonbreaking line can be found in a recent window of the input.

    Lines where the first-character mismatch is explained by an EQUIV pair
    (e.g. u/v) are not reported, since those are handled by adjust_shifted_lbs.py.

    For each reported line the output record includes the id, diagnostic fields,
    the source_sic and target_sic of the current line, and the id plus last
    space-separated tokens of source_sic and target_sic of the preceding line.

    Lines with a genuine first-character mismatch but no preceding nonbreaking
    line found in the window are also written, with "preceding_nonbreaking_line"
    set to null, so they are not silently dropped from the report.
    """

    # A deque with maxlen handles buffer trimming automatically and
    # offers O(1) appends and pops at both ends.
    BUFFER_SIZE = 25
    buffer = deque(maxlen=BUFFER_SIZE)

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

            curr_id = data.get('id')
            source_sic = NFC('NFC', data.get('source_sic', '') or '')
            target_sic = NFC('NFC', data.get('target_sic', '') or '')
            prec_line = data.get('preceding_nonbreaking_line')

            # Report lines where the first characters differ and are not
            # explained by an EQUIV equivalence.
            if (source_sic and target_sic
                    and not chars_match(source_sic[0], target_sic[0])):

                # Search the buffer for the closest preceding line whose
                # nonbreaking_next_line points to the current line.
                found_link = None
                for prev_entry in reversed(buffer):
                    if prev_entry['nonbreaking_next_line'] == curr_id:
                        found_link = prev_entry
                        break

                if found_link is not None:
                    preceding = {
                        "id": found_link['id'],
                        "last_src_token": found_link['source_sic'].split()[-1]
                            if found_link['source_sic'].split() else "",
                        "last_trg_token": found_link['target_sic'].split()[-1]
                            if found_link['target_sic'].split() else "",
                    }
                else:
                    # No preceding link found in the window. Don't include the line in the report.
                    preceding = None

                output_entry = {
                    "id": curr_id,
                    "contains_abbr": data.get("contains_abbr"),
                    "contains_sic": data.get("contains_sic"),
                    "is_in_note": data.get("is_in_note"),
                    "source_sic": source_sic,
                    "target_sic": target_sic,
                    "preceding_nonbreaking_line": preceding,
                }
                if preceding:
                    outfile.write(json.dumps(output_entry, ensure_ascii=False) + '\n')

            buffer.append({
                "id": curr_id,
                "nonbreaking_next_line": data.get("nonbreaking_next_line", ""),
                "source_sic": source_sic,
                "target_sic": target_sic,
            })


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python check_lb_positions.py <input_file.jsonl> <output_file.jsonl>")
        sys.exit(1)

    process_jsonl(sys.argv[1], sys.argv[2])
