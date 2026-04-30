---
title: SvSal Post-Correction Tools
emoji: 📜
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
hardware: zero-gpu
python_version: '3.10'
license: mit
short_description: Abbreviations and unmarked hyphenations for Latin/Spanish
---

This space uses the
[mpilhlt/canine-salamanca-boundary-classifier](https://huggingface.co/mpilhlt/canine-salamanca-boundary-classifier)
and
[mpilhlt/byt5-salamanca-abbr](https://huggingface.co/mpilhlt/byt5-salamanca-abbr)
models to correct latin and spanish text from early modern prints. The issues the
models are trained to fix are:

- detecting unmarked word hyphenations at the end of lines
- expanding abbreviations
