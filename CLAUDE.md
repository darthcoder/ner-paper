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

Runs in sequential stages. The main working corpus is `wwi_extended.jsonl` (3,230 articles). The pipeline produces intermediate files at each stage (documented below).

### Fetching

Fetch Wikipedia articles from category trees:

```bash
# Wikipedia WWI articles (main corpus)
uv run python scripts/fetch_wikipedia_portal.py

# Optional: other sources
uv run python scripts/fetch_wikisource.py
uv run python scripts/fetch_gutenberg.py
```

Outputs: `data/wwi_corpus.jsonl` (or other source-specific names), containing `{pageid, title, wikitext}`.

### Cleaning

Strip wikitext markup to plaintext:

```bash
uv run python scripts/janitor.py --input data/wwi_corpus.jsonl --output data/wwi_clean.jsonl
```

Outputs: `data/*_clean.jsonl`, containing `{pageid, title, text}`.

### Combining & Splitting

Combine all cleaned sources and split 80/20:

```bash
# Combine all *_clean.jsonl sources, deduplicate by pageid
uv run python scripts/combiner.py

# Outputs: data/all_clean.jsonl

# Split 80/20 train/test (seed=42)
uv run python scripts/splitter.py

# Outputs: data/train_clean.jsonl, data/test_clean.jsonl
```

### Redaction

Redact named entities using spaCy NER:

```bash
# Training set: 10–20% redaction rate (weighted by inverse accuracy)
uv run python scripts/redactor.py --input data/train_clean.jsonl --output data/train_redacted.jsonl --mode train

# Test set: 2–5% redaction rate (uniform random)
uv run python scripts/redactor.py --input data/test_clean.jsonl --output data/test_redacted.jsonl --mode test

# Outputs: data/train_redacted.jsonl, data/test_redacted.jsonl
# Schema: {pageid, title, text, redactions: [{original, label, start, end}]}
```

### Curation (Strongly Recommended)

Filter noisy redactions before training. This is a **critical step** — uncurated data significantly degrades model performance.

```bash
# Filter noise: numbers, malformed tokens, low-confidence entity types
uv run python src/ner_recovery/curator.py --input data/train_redacted.jsonl --output data/train_redacted_curated.jsonl
uv run python src/ner_recovery/curator.py --input data/test_redacted.jsonl --output data/test_redacted_curated.jsonl

# Outputs: data/train_redacted_curated.jsonl, data/test_redacted_curated.jsonl
# Keeps only: PERSON, ORG, GPE, EVENT, LOC, FAC, NORP
```

## Training

**DistilBERT MLM** (`distilbert-base-uncased`). Fine-tuned on the redaction task: given a passage with `[REDACTED:LABEL]` tokens replaced by `[MASK]` tokens (N masks matching the original entity's subword length), predict the entity.

Train on curated data (recommended):

```bash
uv run train --epochs 7 --output-dir models/current
```

The `--epochs` flag is required — the internal default is 3. **7 epochs** is the current standard. Model saves to `models/current/` with per-epoch checkpoints and `final/` directory containing the final model weights and tokenizer.

**Output directory conventions**:
- `models/current/` — latest training run (overwritten with each train)
- `models/curated/` — production model trained on curated data
- `models/baseline/` — baseline model for comparison
- Each directory contains subdirectories for each epoch plus `final/` (the final checkpoint)

## Evaluation

Two inference modes (both in `src/ner_recovery/eval.py`, exposed via `uv run evaluate`):

**Constrained** (default, recommended) — score each candidate from the per-label training set against the MLM logits. Inspired by Engram (arXiv:2601.07372). Collapses the search space from ~30k vocab tokens to ~50–300 per-label candidates.

**Free** — argmax over full vocabulary at each `[MASK]` position. Shows true learning without candidate constraints.

### Evaluation Commands

```bash
# Constrained mode — use test_zero_oov.jsonl for cleanest signal (OOV = 0% by construction)
uv run evaluate --model-dir models/current/final --data data/test_zero_oov.jsonl --train data/train_redacted_curated.jsonl

# Free mode
uv run evaluate --mode free --model-dir models/current/final --data data/test_zero_oov.jsonl --train data/train_redacted_curated.jsonl

# Use test_redacted_curated.jsonl instead to measure raw OOV impact
```

**Output**: JSON summary with per-label accuracy, total accuracy, OOV counts, and timestamped results file in `evals/`.

### Frontier Model Topline

```bash
# Single model (batch API — async, 50% cheaper, no RPM concerns)
uv run python scripts/eval_claude.py --data data/test_zero_oov.jsonl
uv run python scripts/eval_claude.py --model claude-sonnet-4-6

# Sequential fallback (~27 RPM, rate-limited)
uv run python scripts/eval_claude.py --no-batch

# Multiple models in parallel (one batch submission per model, all simultaneous)
uv run python scripts/benchmark.py
uv run python scripts/benchmark.py --models claude-haiku-4-5-20251001 claude-sonnet-4-6 claude-opus-4-6

# Print comparison table from all saved JSON results in evals/
uv run python scripts/benchmark_table.py --latest
```

All eval runs save both `.txt` and `.json` to `evals/`. The JSON files are machine-readable and aggregated by `benchmark_table.py`.

### Analysis Tools

```bash
# Analyze OOV entities: identify which test entities are missing from training corpus
uv run python src/ner_recovery/oov_analysis.py --train data/train_redacted_curated.jsonl --test data/test_redacted_curated.jsonl --corpus data/wwi_extended.jsonl

# Fetch Wikipedia articles for OOV entities (appends directly to wwi_extended.jsonl)
uv run python scripts/fetch_oov_entities.py \
    --train data/train_redacted_curated.jsonl \
    --test data/test_redacted_curated.jsonl \
    --output data/wwi_extended.jsonl
```

### Interpreting Results

**Accuracy tiers**:
- **0–10%**: Baseline. Model struggles; likely indicates data quality issues or insufficient corpus coverage.
- **10–20%**: Modest improvement. Model is learning context + entity type hints.
- **20%+**: Strong signal. Model reliably infers entities from context (target for publication).

**OOV (Out-of-Vocabulary) entities**:
- Entities present in test but absent from training candidates.
- OOV % = (# OOV entities) / (# total test entities).
- OOV > 50% indicates corpus coverage gap — accuracy ceiling is limited until corpus expands.

**Constrained vs. Free**:
- Constrained is 2–3× higher than free (by design — smaller search space).
- If free accuracy is very low, model may not be learning meaningful representations.
- Both should improve with more curated training data.

**Per-label breakdown**:
- EVENT and LOC typically perform best (more context-dependent).
- PERSON worse (requires specific knowledge).
- Rare entity types (FAC, NORP) often below 5%.

### Latest Results (April 2026)

**Frontier model topline** (zero-OOV test set, 316 redactions, April 28):
| Model | Overall | NORP | GPE | EVENT | PERSON | ORG |
|---|---|---|---|---|---|---|
| Claude Sonnet 4.6 | 56.3% | 79.5% | 57.7% | 58.1% | 56.8% | 34.7% |
| Claude Haiku 4.5 | 44.0% | 68.0% | 52.6% | 35.5% | 29.5% | 25.3% |
| DistilBERT constrained | 6.6% | 12.7% | — | 10.3% | — | — |

**`models/zero_oov`** (expanded corpus, curated, 9 epochs, April 24):
| Mode | Accuracy | OOV | Notes |
|---|---|---|---|
| Constrained | 6.61% | 37% (3,131/8,468) | 560/8,468; best: NORP 12.68%, EVENT 10.27% |

### Zero-OOV Test Filtering

After curation, filter the test set so every redaction's entity appears in the training candidate set (OOV = 0% by construction). OOV redactions are stripped from each article's redactions list — the entity text remains visible as plain text in the passage but is not scored. Articles with no remaining in-train redactions are dropped.

```bash
uv run python scripts/filter_zero_oov.py \
    --train data/train_redacted_curated.jsonl \
    --test data/test_redacted_curated.jsonl \
    --output data/test_zero_oov.jsonl
```

This produces `data/test_zero_oov.jsonl` (~1,066 articles, ~5,326 evaluable redactions). Use this as `--data` for evaluation. The `run_pipeline.sh` script includes this step automatically.

**Why this framing**: measures engrammatic recall — can the model recover entities it was trained on from context? This is the cleanest signal for the hallucination-probe thesis.

## Current Phase: Accuracy Improvement

**Goal**: Reach 20%+ constrained accuracy before publishing. The zero-OOV test set (`data/test_zero_oov.jsonl`) is built — evaluation now measures engrammatic recall directly. Best constrained result to date: 7.16% (April 22, 4,299-article corpus).

**Next lever**: More training data / more epochs. OOV elimination via corpus expansion is the prior strategy; `scripts/fetch_oov_entities.py` appends articles directly to `wwi_extended.jsonl` if re-running that phase.

See `notes/2026-04-22_next_steps.md` for detailed breakdown.

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

**Cleaned** (`*_clean.jsonl`, including `wwi_extended_clean.jsonl`): `{pageid, title, text}`

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
- `train.py` — DistilBERT MLM fine-tuning with batched windowed inputs (`uv run train`)
- `eval.py` — free and constrained evaluation (`uv run evaluate`)
- `curator.py` — multi-pass noise filter; keeps only PERSON, ORG, GPE, EVENT, LOC, FAC, NORP
- `oov_analysis.py` — identifies test entities absent from training candidates

`pyproject.toml` also exposes `uv run build-corpus` (`ner_recovery.corpus:main`) for XML dump processing.

### Corpus Loader Spec

The corpus loader (`featherweight-corpus-load.md`) specifies:
- Use the `mediawiki` package (not `mwparserfromhell`) for XML dump processing
- Token count via whitespace split only (`len(text.split())`)
- Filter articles: 500–5000 tokens
- doc_id format: `"wikipedia_" + slugified_title` (lowercase, underscores)
- Use `polars` (not `pandas`)
- Generator-based (lazy evaluation, one article at a time)

## Tests

```bash
uv run pytest
```

Tests live in `tests/` and cover masking logic, janitor, redactor, and splitter. Fast — no GPU required.

## Common Development Tasks

### Running a Full Pipeline

Use `run_pipeline.sh` for a one-command run (combines, splits, redacts, curates, trains at 7 epochs, filters zero-OOV, evaluates):

```bash
bash run_pipeline.sh
```

Or step by step when starting fresh or after expanding the corpus:

```bash
# 1. Clean sources
uv run python scripts/janitor.py --input data/wwi_corpus.jsonl --output data/wwi_clean.jsonl

# 2. Combine and split
uv run python scripts/combiner.py
uv run python scripts/splitter.py

# 3. Redact with proper modes
uv run python scripts/redactor.py --input data/train_clean.jsonl --output data/train_redacted.jsonl --mode train
uv run python scripts/redactor.py --input data/test_clean.jsonl --output data/test_redacted.jsonl --mode test

# 4. Curate before training
uv run python src/ner_recovery/curator.py --input data/train_redacted.jsonl --output data/train_redacted_curated.jsonl
uv run python src/ner_recovery/curator.py --input data/test_redacted.jsonl --output data/test_redacted_curated.jsonl

# 5. Train and evaluate
uv run train --epochs 7 --output-dir models/current
uv run evaluate --model-dir models/current/final --data data/test_redacted_curated.jsonl --train data/train_redacted_curated.jsonl

# 6. Analyze bottlenecks
uv run python src/ner_recovery/oov_analysis.py --train data/train_redacted_curated.jsonl --test data/test_redacted_curated.jsonl --corpus data/wwi_extended.jsonl
```

### Iterative Improvement Loop

When refining an existing model:

```bash
# Run OOV analysis to identify missing entities
uv run python src/ner_recovery/oov_analysis.py --train data/train_redacted_curated.jsonl --test data/test_redacted_curated.jsonl --corpus data/wwi_extended.jsonl

# Fetch/add articles for OOV entities
uv run python scripts/fetch_wikipedia_portal.py  # or other fetch scripts

# Clean, combine, split, redact, curate (full pipeline from combining step)
uv run python scripts/combiner.py
uv run python scripts/splitter.py
uv run python scripts/redactor.py --input data/train_clean.jsonl --output data/train_redacted.jsonl --mode train
uv run python scripts/redactor.py --input data/test_clean.jsonl --output data/test_redacted.jsonl --mode test
uv run python src/ner_recovery/curator.py --input data/train_redacted.jsonl --output data/train_redacted_curated.jsonl
uv run python src/ner_recovery/curator.py --input data/test_redacted.jsonl --output data/test_redacted_curated.jsonl

# Train with a named checkpoint
uv run train --epochs 7 --output-dir models/iteration_2

# Compare to baseline
uv run evaluate --model-dir models/iteration_2/final --data data/test_redacted_curated.jsonl --train data/train_redacted_curated.jsonl
uv run evaluate --model-dir models/current/final --data data/test_redacted_curated.jsonl --train data/train_redacted_curated.jsonl
```

### Diagnosing Pipeline Issues

- **Low accuracy on curated data**: Check OOV %. If OOV > 50%, expand corpus first.
- **OOV entities missing from corpus**: Use corpus index to search (`ignore/wikipedia-institutional-fetishism/analysis/corpus_index.py query ENTITY`).
- **Evaluation crashes**: Ensure `--train` file uses same label set as model was trained on. Curator may have filtered labels.

## Key Constraints

- No `pandas` — use `polars`
- No async, no multiprocessing in corpus loading stage
- Redaction seed is fixed at 42 for reproducibility
- All commands via `uv run` (not bare `python`)
- Curation is essential for production runs — uncurated data significantly degrades results
