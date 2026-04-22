# scripts/

Data pipeline scripts for corpus fetching, cleaning, splitting, and redaction.

## Data Pipeline (Sequential)

Run in this order:

### 1. Fetching

- **`fetch_wikipedia_portal.py`** — Fetch WWI articles from Wikipedia
- **`fetch_wikisource.py`** — Fetch WWI primary sources from Wikisource
- **`fetch_gutenberg.py`** — Fetch WWI books from Project Gutenberg
- **`fetch_category_map.py`** — Map category membership (no text, metadata only)

### 2. Cleaning

- **`janitor.py`** — Strip wikitext markup, convert to plaintext
  - Input: raw corpus (`.jsonl` with `wikitext` field)
  - Output: cleaned corpus (`.jsonl` with `text` field)

### 3. Combining & Splitting

- **`combiner.py`** — Combine all `*_clean.jsonl` sources into `all_clean.jsonl`
  - Deduplicates by `pageid`
  - No arguments (reads from `data/` automatically)

- **`splitter.py`** — Split into 80/20 train/test (seed=42)
  - Input: `all_clean.jsonl`
  - Output: `train_clean.jsonl`, `test_clean.jsonl`

### 4. Redaction

- **`redactor.py`** — Redact named entities using spaCy NER
  - Usage: `python scripts/redactor.py --input FILE --output FILE --mode {train,test}`
  - `--mode train`: 10–20% redaction rate, weighted by inverse accuracy
  - `--mode test`: 2–5% redaction rate, uniform random
  - Output: `{pageid, title, text, redactions: [{original, label, start, end}]}`

## Quick Reference

```bash
# Full pipeline from scratch
uv run python scripts/combiner.py
uv run python scripts/splitter.py
uv run python scripts/redactor.py --input data/train_clean.jsonl --output data/train_redacted.jsonl --mode train
uv run python scripts/redactor.py --input data/test_clean.jsonl --output data/test_redacted.jsonl --mode test
```

## Notes

- All paths are relative to project root
- Requires spaCy: `python -m spacy download en_core_web_sm`
- Redaction seed is fixed at 42 for reproducibility
