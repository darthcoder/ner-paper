# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Named Entity Recognition (NER) recovery** research project. The goal is to train a model to infer redacted named entities from text. The training corpus is drawn from World War I Wikipedia articles.

## Setup

Uses `uv` and Python 3.11 (see `.python-version`). Install dependencies:

```bash
uv sync
```

For spaCy NER (required by `scripts/redactor.py`):

```bash
python -m spacy download en_core_web_sm
```

## Running the Pipeline

The data pipeline runs in three sequential stages:

```bash
# Stage 1: Fetch raw Wikipedia articles → wwi_corpus.jsonl
python fetcher.py

# Stage 2: Strip wikitext markup → data/wwi_clean.jsonl
python scripts/janitor.py

# Stage 3: Redact named entities → data/wwi_redacted.jsonl
python scripts/redactor.py
```

The CLI entry points (`build-corpus`, `train`, `evaluate`) defined in `pyproject.toml` map to `src/ner_recovery/` modules that are not yet implemented.

## Architecture

### Data Pipeline

```
Wikipedia API → wwi_corpus.jsonl (raw wikitext, 145 articles)
             → data/wwi_clean.jsonl (plaintext, mwparserfromhell strips markup)
             → data/wwi_redacted.jsonl (text with [REDACTED] tokens + annotations)
```

### JSONL Schemas

**wwi_corpus.jsonl**: `{pageid, title, wikitext}`

**data/wwi_clean.jsonl**: `{pageid, title, text}`

**data/wwi_redacted.jsonl**:
```json
{
  "pageid": 4764461,
  "title": "World War I",
  "text": "...Austria-Hungary blamed [REDACTED], and declared war...",
  "redactions": [
    {"start": 1107, "end": 1117, "label": "GPE", "original": "Serbia"}
  ]
}
```
Offsets (`start`/`end`) are character positions in the redacted text. Redaction sampling is seeded at 42 (2–5% of entities per article, minimum 1).

### Package Structure (planned)

`src/ner_recovery/` is the installable package. Entry points to implement:
- `ner_recovery.corpus:main` — corpus loading from Wikipedia XML dumps
- `ner_recovery.train:main` — transformer model training
- `ner_recovery.eval:main` — evaluation

### Corpus Loader Spec

The corpus loader (`featherweight-corpus-load.md`) specifies:
- Use the `mediawiki` package (not `mwparserfromhell`) for XML dump processing
- Token count via whitespace split only (`len(text.split())`)
- Filter articles: 500–5000 tokens
- doc_id format: `"wikipedia_" + slugified_title` (lowercase, underscores)
- Use `polars` (not `pandas`)
- Generator-based (lazy evaluation, one article at a time)

## Key Constraints

- No `pandas` — use `polars`
- No async, no multiprocessing in corpus loading stage
- Redaction seed is fixed at 42 for reproducibility
