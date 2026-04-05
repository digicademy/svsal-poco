import json
import sqlite3
import sys
import unicodedata
import html

# Characters treated as interchangeable when comparing source and target.
# Kept in sync with adjust_shifted_lbs.py.
EQUIV = {
    'u': 'v', 'v': 'u',
    'U': 'V', 'V': 'U',
    's': 'ſ', 'ſ': 's',
    'y': '⁊', '⁊': 'y',
    'z': 'ʒ', 'ʒ': 'z',
    'ꝗ': 'q', 'q': 'ꝗ',
    '': 'ꝺ̇', 'ꝺ̇': '',
}


def normalize(s):
    """Normalize a string: HTML-unescape, then NFC unicode form. Returns None if input is None."""
    if s is None:
        return None
    return unicodedata.normalize('NFC', html.unescape(s))


def chars_match(a, b):
    return a == b or EQUIV.get(a) == b


def strings_match(a, b):
    """Compare two strings using character equivalence, skipping ⦃ and ⦄ delimiters."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    
    # Characters to skip during comparison
    skip_chars = {'⦃', '⦄'}
    
    i, j = 0, 0
    len_a, len_b = len(a), len(b)
    
    while i < len_a and j < len_b:
        # Skip delimiter characters in string a
        while i < len_a and a[i] in skip_chars:
            i += 1
        # Skip delimiter characters in string b
        while j < len_b and b[j] in skip_chars:
            j += 1
        
        # If we've reached the end of either string, check if the rest is only delimiters
        if i >= len_a or j >= len_b:
            # Check remaining characters in a
            while i < len_a:
                if a[i] not in skip_chars:
                    return False
                i += 1
            # Check remaining characters in b
            while j < len_b:
                if b[j] not in skip_chars:
                    return False
                j += 1
            return True
        
        # Compare the current characters
        if not chars_match(a[i], b[j]):
            return False
        
        i += 1
        j += 1
    
    # Check if any remaining characters are non-delimiters
    while i < len_a:
        if a[i] not in skip_chars:
            return False
        i += 1
    while j < len_b:
        if b[j] not in skip_chars:
            return False
        j += 1
    
    return True


def compare_jsonl_files(file_a_path, file_b_path, output_path):
    """
    Compares B.jsonl against A.jsonl and outputs results to a JSONL file.
    """
    
    # 1. Setup SQLite Database
    try:
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        
        # Create table to store the necessary fields from A.jsonl
        cursor.execute("""
            CREATE TABLE data_a (
                id TEXT PRIMARY KEY,
                source_sic TEXT,
                target_sic TEXT,
                target_corr TEXT
            )
        """)
        
        # 2. Index A.jsonl
        print(f"Indexing {file_a_path}...")
        batch = []
        batch_size = 50000
        
        with open(file_a_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    batch.append((
                        data.get('id'),
                        normalize(data.get('source_sic')),
                        normalize(data.get('target_sic')),
                        normalize(data.get('target_corr'))
                    ))
                    
                    if len(batch) >= batch_size:
                        cursor.executemany("INSERT OR REPLACE INTO data_a VALUES (?,?,?,?)", batch)
                        conn.commit()
                        batch = []
                except json.JSONDecodeError:
                    continue
            
            if batch:
                cursor.executemany("INSERT OR REPLACE INTO data_a VALUES (?,?,?,?)", batch)
                conn.commit()
                
        print(f"Indexing complete. Comparing with {file_b_path}...")
        
        # 3. Iterate B.jsonl, Compare, and Write to Output
        with open(file_b_path, 'r', encoding='utf-8') as f_in, \
             open(output_path, 'w', encoding='utf-8') as f_out:
            
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                try:
                    data_b = json.loads(line)
                    bid = data_b.get('id')
                    
                    if not bid:
                        continue
                        
                    # Look up the ID in the A-index
                    cursor.execute(
                        "SELECT source_sic, target_sic, target_corr FROM data_a WHERE id=?", 
                        (bid,)
                    )
                    result = cursor.fetchone()
                    
                    output_entry = {
                        "id": bid,
                        "status": "match",
                        "details": None
                    }
                    
                    if not result:
                        output_entry["status"] = "missing_in_A"
                        output_entry["details"] = f"ID '{bid}' found in B but not in A"
                    else:
                        a_source_sic, a_target_sic, a_target_corr = result
                        
                        b_source = normalize(data_b.get('source'))
                        b_target = normalize(data_b.get('target'))
                        b_target_corr = normalize(data_b.get('target_corr'))
                        
                        differences = {}
                        if not strings_match(a_target_sic, b_target):
                            differences["target/target_sic"] = {
                                "A": a_target_sic,
                                "B": b_target
                            }
                        if not strings_match(a_target_corr, b_target_corr):
                            differences["target_corr"] = {
                                "A": a_target_corr,
                                "B": b_target_corr
                            }
                        
                        # Collect source differences if they exist, but also if there are target differences,
                        # since source differences might be relevant in that context.
                        if differences or not strings_match(a_source_sic, b_source):
                            differences["source/source_sic"] = {
                                "A": a_source_sic,
                                "B": b_source
                            }
                            output_entry["status"] = "differences_found"
                            output_entry["details"] = differences
                    
                    # Only write if there is an issue (missing or differences)
                    # If you want to write all lines (including matches), remove the 'if' check below.
                    if output_entry["status"] != "match":
                        f_out.write(json.dumps(output_entry, ensure_ascii=False) + '\n')
                            
                except json.JSONDecodeError:
                    continue

    except FileNotFoundError as e:
        print(f"Error: {e}")
    finally:
        conn.close()
    
    print(f"Comparison finished. Results saved to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python check_compare.py <A.jsonl> <B.jsonl> <output.jsonl>")
        print("  A.jsonl: The reference superset file")
        print("  B.jsonl: The file to check against A")
        print("  output.jsonl: The file to write results to")
        sys.exit(1)

    file_a = sys.argv[1]
    file_b = sys.argv[2]
    output_file = sys.argv[3]

    compare_jsonl_files(file_a, file_b, output_file)
