# Engrammatic Named Entity Inference

A research project probing a model's latent entity knowledge by training it to infer redacted named entities from context. Given a passage with typed mask tokens (e.g. `[REDACTED:GPE]`, `[REDACTED:PERSON]`), the model predicts what was removed.

Embedding the entity type in the mask token — suggested by Meesum — gives the model a strong prior: predicting a country name, a person name, and a date are very different tasks.

---

## Task

Each article is processed by a spaCy NER tagger. A subset of detected entities are replaced with `[REDACTED:{LABEL}]` tokens. The model is trained to recover the original entity text from surrounding context.

---

## Pipeline

### Corpus fetching

```bash
# Wikipedia WWI portal
python scripts/fetch_wikipedia_portal.py

# Wikisource WW1 primary sources
python scripts/fetch_wikisource.py --output data/wikisource_corpus.jsonl

# Project Gutenberg WW1 books
python scripts/fetch_gutenberg.py --output data/gutenberg_corpus.jsonl
```

### Cleaning, splitting, redacting

```bash
# Combine all *_clean.jsonl into all_clean.jsonl (deduplicates by pageid)
python scripts/combiner.py

# Strip wikitext markup (run per source corpus)
python scripts/janitor.py --input <corpus>.jsonl --output data/<corpus>_clean.jsonl

# Split 80/20 train/test (seed=42)
python scripts/splitter.py

# Redact
python scripts/redactor.py --input data/train_clean.jsonl --output data/train_redacted.jsonl --mode train
python scripts/redactor.py --input data/test_clean.jsonl  --output data/test_redacted.jsonl  --mode test
```

### Train and evaluate

```bash
uv run train
uv run evaluate
```

---

## Setup

Requires Python 3.11 and [uv](https://github.com/astral-sh/uv).

```bash
uv sync
python -m spacy download en_core_web_sm
```

---

## Data

| File | Description |
|---|---|
| `data/wwi_portal_corpus.jsonl` | Raw wikitext, ~2093 Wikipedia WWI articles |
| `data/wikisource_corpus.jsonl` | Raw wikitext, 1043 Wikisource WW1 documents |
| `data/gutenberg_corpus.jsonl` | Raw text, 317 Project Gutenberg WW1 books |
| `data/all_clean.jsonl` | Combined, deduplicated plaintext (540 articles) |
| `data/train_clean.jsonl` | 432 articles (80% split) |
| `data/test_clean.jsonl` | 108 articles (20% split) |
| `data/train_redacted.jsonl` | Training set — `[REDACTED:LABEL]` tokens + annotations |
| `data/test_redacted.jsonl` | Test set — `[REDACTED:LABEL]` tokens + annotations |

### JSONL schema (`*_redacted.jsonl`)

```json
{
  "pageid": 4764461,
  "title": "World War I",
  "text": "...Austria-Hungary blamed [REDACTED:GPE], and declared war...",
  "redactions": [
    {"start": 1107, "end": 1119, "label": "GPE", "original": "Serbia"}
  ]
}
```

`start`/`end` are character offsets in the redacted text.

---

## Redaction Modes

| Mode | Rate | Sampling |
|---|---|---|
| `train` (weighted) | 10–20% of entities, min 1 | Weighted by inverse accuracy — hard labels sampled more |
| `test` (uniform) | 2–5% of entities, min 1 | Uniform random |

Seed fixed at 42 for reproducibility.

---

## Model

- **Architecture**: `distilbert-base-uncased` (~66M params), Masked Language Model
- **Task framing**: `[REDACTED:LABEL]` tokens replaced by N `[MASK]` tokens (N = subword count of original entity); model predicts original tokens
- **Training examples**: grouped redactions in 2000-char windows, up to 512 tokens per example
- **Hyperparameters**: 3 epochs, batch size 8, lr 5e-5, AdamW + linear warmup (50 steps)

---

## Reference Papers

Stored in `refs/`:

- BERT: `bert_1810.04805.pdf`
- Blank language models: `blank_lm_2002.03079.pdf`
- Fill-in-the-blank: `fill_blanks_2005.05339.pdf`
- MLM inductive bias: `mlm_inductive_bias_2104.05694.pdf`
- Redaction & privacy: `redaction_privacy_2410.07772.pdf`
- Hallucination surveys: `hallucination_*.pdf`
- UniFact: `unifact_2512.02772.pdf`
