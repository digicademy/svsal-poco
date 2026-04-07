import json
from collections import Counter

lines = []
with open("data/data.jsonl") as f:
    for line in f:
        lines.append(json.loads(line))

index = {row["id"]: row for row in lines}
lengths = []

successor_ids = {row["nonbreaking_next_line"] 
                 for row in lines 
                 if row.get("nonbreaking_next_line")}

index = {row["id"]: row for row in lines}
lengths = []

for row in lines:
    # Skip lines that are continuations — they'll be counted as part of a chain
    if row["id"] in successor_ids:
        continue

    # Follow the chain from this starting line
    chain = [row["source_sic"]]
    current = row
    while current.get("nonbreaking_next_line"):
        next_id = current["nonbreaking_next_line"]
        next_row = index.get(next_id)
        if not next_row:
            break
        chain.append(next_row["source_sic"])
        current = next_row

    combined = "↵".join(chain)
    lengths.append(len(combined.encode("utf-8")))

print(f"Total examples: {len(lengths)}")
print(f"Max length (bytes): {max(lengths)}")
print(f"Mean length: {sum(lengths)/len(lengths):.1f}")
print(f"95th percentile: {sorted(lengths)[int(len(lengths)*0.95)]}")
print(f"99th percentile: {sorted(lengths)[int(len(lengths)*0.99)]}")
print()

# Also show chain length distribution
chain_lengths = []
for row in lines:
    if row["id"] in successor_ids:
        continue
    length = 1
    current = row
    while current.get("nonbreaking_next_line"):
        next_row = index.get(current["nonbreaking_next_line"])
        if not next_row:
            break
        length += 1
        current = next_row
    chain_lengths.append(length)

from collections import Counter
chain_dist = Counter(chain_lengths)
print("Chain length distribution:")
for length, count in sorted(chain_dist.items()):
    print(f"  {length} line(s): {count:,} examples ({count/len(chain_lengths):.2%})")
