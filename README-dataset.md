---
license: cc-by-4.0
doi: 10.57967/hf/8278
language:
- la
- es
tags:
- history
- humanities
- early-modern
- historical-text
task_categories:
- text-generation
- token-classification
pretty_name: Salamanca Abbreviation and Hyphenation Dataset
size_categories:
- 1M<n<10M
---

# Salamanca Abbreviation and Hyphenation Dataset

This is a dataset created from manually edited and curated digital
edition texts of the so-called School of Salamanca, a group of
16th- and 17th-century theologians and jurists. The digital editions
can be studied at the
[School of Salamanca Website](https://salamanca.school/), together
with a dictionary of the political-juridical language these authors
were using and contributing to shape.

The corpus contains printed texts of various genres (academic summae,
in some cases an author's collected works, as well as pragmatic
booklets for merchants or confessors) in Latin and Spanish, but all
the texts are concerned with law, politics, and ethics.

The pipeline extracting the dataset from the TEI XML sources as they
have been prepared in the project is documented in the
[SvSal-PoCo repository](https://github.com/digicademy/svsal-poco) at
GitHub, more specifically in the
[data/prepare_data subfolder](https://github.com/digicademy/svsal-poco/tree/main/data/prepare_data).

The creation of the dataset happened in the course of an experiment
aiming to establish machine learning tools to aid the project's
editors in their work, i.e. detecting cases where a word has been
broken to straddle two lines without this being indicated by a
hyphenation dash, and expanding abbreviations (also at times
straddling two or even three lines - yes, these exist). The
experiment's pipeline code and tools can be accessed at the GitHub
repository, too.