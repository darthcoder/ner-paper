# NER Redaction Recovery

A research project training a model to infer redacted named entities from text. Given a passage with `[REDACTED]` tokens, the model predicts what was removed.

Training corpus: 145 World War I Wikipedia articles.

---

## Task

Each article is processed by a spaCy NER tagger. A subset of detected entities are replaced with `[REDACTED]`. The model — fine-tuned DistilBERT as a Masked Language Model — is trained to recover the original entity text from surrounding context.

---

## Pipeline

Three sequential data stages:

```bash
# Stage 1: Fetch raw Wikipedia articles → wwi_corpus.jsonl
python fetcher.py

# Stage 2: Strip wikitext markup → data/wwi_clean.jsonl
python scripts/janitor.py

# Stage 3: Split 80/20 train/test
python scripts/splitter.py

# Stage 4a: Redact train split (weighted or uniform)
python scripts/redactor.py --input data/train_clean.jsonl --output data/train_redacted.jsonl --mode train

# Stage 4b: Redact test split (always uniform)
python scripts/redactor.py --input data/test_clean.jsonl --output data/test_redacted.jsonl --mode test
```

Then train and evaluate:

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

All processed data is committed to this repo and serves as a benchmark artifact.

| File | Description |
|---|---|
| `wwi_corpus.jsonl` | Raw wikitext, 145 articles |
| `data/wwi_clean.jsonl` | Plaintext after markup stripping |
| `data/train_clean.jsonl` | 116 articles (80% split, seed=42) |
| `data/test_clean.jsonl` | 29 articles (20% split, seed=42) |
| `data/train_redacted.jsonl` | Training set with `[REDACTED]` tokens + annotations |
| `data/test_redacted.jsonl` | Held-out test set with `[REDACTED]` tokens + annotations |

### JSONL schema (`*_redacted.jsonl`)

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

`start`/`end` are character offsets in the redacted text.

---

## Model

- **Architecture**: `distilbert-base-uncased` — encoder-only Masked Language Model (~66M params)
- **Task framing**: Entity token positions are masked in `input_ids`; model predicts original tokens from bidirectional context
- **Sliding window**: 512-token window, 256-token stride for long articles
- **Hyperparameters**: 3 epochs, batch size 8, lr 5e-5, AdamW + linear warmup (50 steps)

---

## Redaction Modes

`scripts/redactor.py` supports two modes:

| Mode | Rate | Sampling |
|---|---|---|
| `test` (uniform) | 2–5% of entities, min 1 | Uniform random |
| `train` (weighted) | 10–20% of entities, min 1 | Weighted by inverse accuracy — weak labels sampled more |

The test split always uses uniform mode to reflect natural entity distribution.

---

## Results

### Eval history

All eval reports are saved in `evals/` with timestamps.

| Run | Accuracy | Notes |
|---|---|---|
| `baseline_eval_20260412_144546` | 37.95% | Original model, train=eval, **single `[MASK]` bug** |
| `fix1_eval_20260412_144946` | 67.17% | Multi-token mask fix applied, still train=eval (inflated) |
| `fix2_eval_20260412_151920` | 39.27% | Weighted redaction, train=eval, larger test set |
| `fix2b_eval_20260412_160328` | 18.21% | Weighted train, 80/20 split, **first honest held-out eval** |
| `clean_baseline_eval_20260412_162911` | **17.34%** | Uniform train, 80/20 split — **true clean baseline** |

### True baseline (held-out test set, 346 redactions across 29 articles)

```
Accuracy : 17.34%  (60/346)
```

| Label | Correct | Total | Accuracy |
|---|---|---|---|
| ORDINAL | 2 | 2 | 100.00% |
| NORP | 19 | 36 | 52.78% |
| GPE | 19 | 56 | 33.93% |
| CARDINAL | 3 | 25 | 12.00% |
| EVENT | 3 | 24 | 12.50% |
| ORG | 6 | 48 | 12.50% |
| WORK_OF_ART | 1 | 8 | 12.50% |
| DATE | 4 | 73 | 5.48% |
| PERSON | 3 | 56 | 5.36% |
| FAC | 0 | 4 | 0.00% |
| LAW | 0 | 2 | 0.00% |
| LOC | 0 | 7 | 0.00% |
| MONEY | 0 | 1 | 0.00% |
| PRODUCT | 0 | 3 | 0.00% |
| TIME | 0 | 1 | 0.00% |

### Key finding: fix #1 (multi-token mask correction)

The original `eval.py` replaced every `[REDACTED]` with a single `[MASK]` token, making exact-match impossible for any multi-token entity (e.g. "Franz Ferdinand", "Western Front"). The fix tokenizes each original entity to determine N, inserts N `[MASK]` tokens, predicts each position, and joins the decoded tokens. This was responsible for the apparent jump from 37.95% → 67.17% — though that comparison was on training data.

---

## Lessons Learned

1. **Always separate train and eval data.** Early evals (37.95%, 67.17%) were on the same articles used for training — measuring memorization, not generalization. The honest number is **17.34%**.
2. **Multi-token entity evaluation matters.** A single `[MASK]` per entity silently kills accuracy for all multi-word entities. Always match mask count to token count.
3. **Error-driven weighted redaction** (oversampling weak labels during training) showed marginal improvement (18.21% vs 17.34%) but not statistically significant on this corpus size.

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
