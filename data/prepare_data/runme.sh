#!/bin/sh

# === Dataset preparation ===

# First, run the XSLT transformation 'scripts/01_create_jsonl.xsl' on the Salamanca source files, e.g. as an oXygen transformation scenario
# probably there is also a way to do it via saxon on the commandline ...

# Second, concatenate all resulting files: `cat W*.jsonl > salamanca_edited_works.jsonl`

# Fix erroneously places linebreaks in expansions:
python scripts/02_adjust_shifted_lbs.py salamanca_edited_works.jsonl output/salamanca_edited_works.fixed.jsonl

# Revert normalization of 'ſ', 'æ', 'œ' etc.:
python scripts/03_denormalize.py output/salamanca_edited_works.fixed.jsonl output/salamanca_edited_works.longs.fixed.jsonl

# === The rest is diagnostics ===

# Check weird break position - the most important heuristic is when lines begin differently for source and target
# (i.e. abbreviated and expanded) text versions, but in many cases, that is because the first character in the abbreviated
# version is a special character like 'ꝙ' and the first character in the expanded version is a regular ASCII character ('q' from 'quod').
# In other cases, it is because there is a mistake in the abbreviation where a subword is missing or duplicated, whereas
# the abbreviation is corrected, leading to another difference in the line beginning. 
python scripts/04_check_lb_positions.py output/salamanca_edited_works.longs.fixed.jsonl output/problems_lb.fixed.jsonl
wc -l output/problems_lb.fixed.jsonl

# Check against a version exported by Michael Schonhardt. There are many differences, they will be analyzed below.
python scripts/05_check_compare.py output/salamanca_edited_works.longs.fixed.jsonl bak/line_expansion.jsonl output/problems_compare.fixed.jsonl
wc -l output/problems_compare.fixed.jsonl

# Separate those lines that come from works that my export does not include but Michael's does
grep "found in B but not in A" output/problems_compare.fixed.jsonl > output/not-in-a.jsonl
wc -l output/not-in-a.jsonl
grep -v "found in B but not in A" output/problems_compare.fixed.jsonl > output/ohne_not-in-a.jsonl
wc -l output/ohne_not-in-a.jsonl

# Of those that should be in both exports, separate those differences that are caused by a difference in whitespace
# (Michael's export has some blanks before punctuation characters, which my export does not)
python scripts/06_analyze_diffs.py output/ohne_not-in-a.jsonl -r output/ohne_not-in-a.r.jsonl -w output/ohne_not-in-a.w.jsonl > output/rw-analysis.txt
wc -l output/ohne_not-in-a.r.jsonl
wc -l output/ohne_not-in-a.w.jsonl
