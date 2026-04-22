# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Engrammatic Named Entity Inference** — a research project probing a model's latent entity knowledge by training it to infer redacted named entities from context. Given a passage with typed mask tokens (e.g. `[REDACTED:GPE]`, `[REDACTED:PERSON]`), the model predicts what was removed.

The broader research goal is using this as a hallucination probe for frontier model pretraining (target: Mythos / Anthropic). The training corpus is drawn from World War I Wikipedia articles.

## Setup

Uses `uv` and Python 3.11 (see `.python-version`). Install dependencies:

```bash
uv sync
```

For spaCy NER (required by `scripts/redactor.py`):

```bash
uv run python -m spacy download en_core_web_sm
```

Git LFS is required — `.jsonl` and `.json` files are LFS-tracked. Install and pull:

```bash
git lfs install
git lfs pull
```

## Data Pipeline

Runs in sequential stages per source corpus. The main working corpus is `wwi_extended.jsonl` (3,230 articles). The training split uses `all_clean.jsonl` (540 articles, 80/20 train/test).

```bash
# Fetch Wikipedia articles (recursive BFS over category tree)
uv run python fetcher.py --category "Category:World_War_I" --output data/wwi_corpus.jsonl --max-depth 3

# Extended multi-category fetch (47 WWI categories, append mode)
bash scripts/fetch_wwi_extended.sh

# Strip wikitext markup → cleaned plaintext
uv run python scripts/janitor.py --input data/wwi_corpus.jsonl --output data/wwi_clean.jsonl

# Combine all *_clean.jsonl sources, deduplicate by pageid
uv run python scripts/combiner.py

# Split 80/20 train/test (seed=42)
uv run python scripts/splitter.py

# Redact named entities → training data
uv run python scripts/redactor.py
```

## Model

**DistilBERT MLM** (`distilbert-base-uncased`). Fine-tuned on the redaction task: given a passage with `[REDACTED:LABEL]` tokens replaced by `[MASK]` tokens (N masks matching the original entity's subword length), predict the entity.

Training:

```bash
uv run train   # maps to src/ner_recovery/train.py
```

Default: 7 epochs. Model saves to `models/final/`.

## Evaluation

Two inference modes, both in `src/ner_recovery/eval.py`:

**Free** — argmax over full vocabulary at each `[MASK]` position.

**Constrained** (default, better) — score each candidate from the per-label training set against the MLM logits. Inspired by Engram (arXiv:2601.07372). Collapses the search space from ~30k vocab tokens to ~50–300 per-label candidates.

```bash
uv run evaluate                          # constrained mode (default)
uv run evaluate --mode free              # unconstrained
uv run evaluate --data data/test_redacted.jsonl --train data/train_redacted.jsonl
```

### Latest Results (April 2026, models/final, constrained)

| Mode | Accuracy | Notes |
|---|---|---|
| Constrained | 5.58% | 46/825; EVENT 38.5%, LOC 19.1% |
| Free | 3.64% | 30/825 |

OOV rate: 51.9% of test entities not seen in training candidates.

## Corpus Index

Positional inverted index for ghost term discovery across the full corpus:

```bash
# Build once (~30s for 3,230 articles)
uv run python ignore/wikipedia-institutional-fetishism/analysis/corpus_index.py build --corpus data/wwi_extended.jsonl

# Direct lookup
uv run python ignore/wikipedia-institutional-fetishism/analysis/corpus_index.py query Sazonov

# Proximity: semantic neighbourhood within 50 words
uv run python ignore/wikipedia-institutional-fetishism/analysis/corpus_index.py query Sazonov --radius 50
```

## Sub-projects

`ignore/wikipedia-institutional-fetishism/` — a separate analysis of Wikipedia's empty category structure as institutional fetishism. Has its own `.git` and `CLAUDE.md`. Uses the same corpus (`data/wwi_extended.jsonl`) and `uv` environment.

## Architecture

### JSONL Schemas

**Raw corpus** (`wwi_corpus.jsonl`, `wwi_extended.jsonl`): `{pageid, title, wikitext}`

**Cleaned** (`*_clean.jsonl`): `{pageid, title, text}`

**Redacted** (`*_redacted.jsonl`):
```json
{
  "pageid": 4764461,
  "title": "World War I",
  "text": "...Austria-Hungary blamed [REDACTED:GPE], and declared war...",
  "redactions": [
    {"start": 1107, "end": 1117, "label": "GPE", "original": "Serbia"}
  ]
}
```
Offsets are character positions in the redacted text. Redaction sampling seeded at 42 (2–5% of entities per article, minimum 1).

### Package Structure

`src/ner_recovery/` — installable package:
- `train.py` — DistilBERT MLM fine-tuning with batched windowed inputs
- `eval.py` — free and constrained evaluation

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
- All commands via `uv run` (not bare `python`)
