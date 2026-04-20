# NER Redaction Recovery

A research project training a model to infer redacted named entities from text. Given a passage with `[REDACTED]` tokens, the model predicts what was removed.

---

## Task

Each article is processed by a spaCy NER tagger. A subset of detected entities are replaced with typed mask tokens (e.g. `[REDACTED:GPE]`, `[REDACTED:PERSON]`). The model is trained to recover the original entity text from surrounding context. Embedding the entity type in the mask token — suggested by Meesum — gives the model a strong prior: predicting a country name, a person name, and a date are very different tasks.

---

## Pipeline

### Corpus fetching

```bash
# Wikipedia WWI portal (2093 raw articles)
python scripts/fetch_wikipedia_portal.py

# Wikisource WW1 primary sources (1043 articles)
python scripts/fetch_wikisource.py --output data/wikisource_corpus.jsonl

# Project Gutenberg WW1 books (317 entries)
python scripts/fetch_gutenberg.py --output data/gutenberg_corpus.jsonl
```

### Cleaning, splitting, redacting

```bash
# Strip wikitext markup
python scripts/janitor.py --input <corpus>.jsonl --output data/<corpus>_clean.jsonl

# Split 80/20 train/test
python scripts/splitter.py

# Redact train split (weighted)
python scripts/redactor.py --input data/train_clean.jsonl --output data/train_redacted.jsonl --mode train

# Redact test split (uniform)
python scripts/redactor.py --input data/test_clean.jsonl --output data/test_redacted.jsonl --mode test
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
| `data/wwi_clean.jsonl` | Plaintext WWI Wikipedia articles |
| `data/napoleonic_clean.jsonl` | Plaintext Napoleonic Wars articles |
| `data/wikisource_clean.jsonl` | Plaintext Wikisource documents (276 after filtering) |
| `data/train_clean.jsonl` | Training split (80%, seed=42) |
| `data/test_clean.jsonl` | Test split (20%, seed=42) |
| `data/train_redacted.jsonl` | Training set with `[REDACTED]` tokens + annotations |
| `data/test_redacted.jsonl` | Held-out test set with `[REDACTED]` tokens + annotations |

### JSONL schema (`*_redacted.jsonl`)

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

`start`/`end` are character offsets in the redacted text.

---

## Model

- **Architecture**: `t5-small` — encoder-decoder (~60M params), span reconstruction
- **Task framing**: Input is text with `[REDACTED]` tokens; model generates the original entity spans
- **Previous**: `bert-base-uncased` MLM (~110M params), masked token prediction
- **Hyperparameters**: 7 epochs, batch size 8, lr 5e-5, AdamW + linear warmup (50 steps)

---

## Redaction Modes

`scripts/redactor.py` supports two modes:

| Mode | Rate | Sampling |
|---|---|---|
| `test` (uniform) | 2–5% of entities, min 1 | Uniform random |
| `train` (weighted) | 10–20% of entities, min 1 | Weighted by inverse accuracy — weak labels sampled more |

The test split always uses uniform mode to reflect natural entity distribution. Redaction seed is fixed at 42.

---

## Results

### Eval history

All eval reports are saved in `evals/` with timestamps.

| Run | Accuracy | Notes |
|---|---|---|
| `baseline_eval_20260412_144546` | 37.95% | DistilBERT, train=eval, **single `[MASK]` bug** |
| `fix1_eval_20260412_144946` | 67.17% | Multi-token mask fix, still train=eval (inflated) |
| `fix2_eval_20260412_151920` | 39.27% | Weighted redaction, train=eval |
| `fix2b_eval_20260412_160328` | 18.21% | Weighted train, 80/20 split, first honest held-out eval |
| `clean_baseline_eval_20260412_162911` | 17.34% | Uniform train, 80/20 split — **true clean baseline** |
| `7epoch_eval_20260412_173053` | 17.63% | DistilBERT, 7 epochs |
| `bert_base_eval_20260412_190501` | 20.52% | bert-base-uncased, 7 epochs, WWI only |
| `napoleonic_eval_20260413_175211` | **21.94%** | bert-base-uncased, 7 epochs, WWI + Napoleonic corpus — **current best** |

### Best result (21.94%)

```
Train : 154 articles / 2254 redactions (WWI + Napoleonic Wars)
Test  :  39 articles / 629 redactions
```

| Label | Correct | Total | Accuracy |
|---|---|---|---|
| NORP | 39 | 85 | 45.88% |
| GPE | 33 | 91 | 36.26% |
| ORDINAL | 4 | 10 | 40.00% |
| PERSON | 19 | 88 | 21.59% |
| EVENT | 4 | 21 | 19.05% |
| CARDINAL | 12 | 80 | 15.00% |
| ORG | 12 | 79 | 15.19% |
| DATE | 13 | 112 | 11.61% |
| LOC | 1 | 10 | 10.00% |

---

## Lessons Learned

1. **Always separate train and eval data.** Early evals (37.95%, 67.17%) measured memorization, not generalization. The honest number is **17.34%**.
2. **Multi-token entity evaluation matters.** A single `[MASK]` per entity kills accuracy for multi-word entities. Always match mask count to token count.
3. **Corpus breadth helps.** Adding the Napoleonic Wars corpus (+80 articles) improved accuracy from 20.52% → 21.94% on a harder test set.
4. **7 epochs > 3.** On this corpus size, 3 epochs undertrained — 7 is the working default.

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
