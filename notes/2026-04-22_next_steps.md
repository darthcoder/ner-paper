# Next Steps: OOV Elimination & Scaling

Date: April 22, 2026

## Current State

- **Constrained accuracy**: 7.16% (curated model) — improved from 5.58%
- **OOV rate**: 67% (272/406 test entities not in training)
- **Root cause**: All 272 OOV entities have zero corpus mentions — they don't exist in the 4,299-article corpus

## Bottleneck Identified

Can't improve accuracy further without expanding corpus to include missing entities.

**Missing entity breakdown:**
- PERSON: 78/93 (83.9%)
- ORG: 115/147 (78.2%)
- FAC: 5/6 (83.3%)
- LOC: 10/14 (71.4%)
- EVENT: 12/20 (60.0%)
- GPE: 34/73 (46.6%) — best coverage
- NORP: 18/53 (34.0%) — best coverage

## Strategic Fetch Plan

1. **Target missing entity types** (not random expansion):
   - Countries & regions: Afghanistan, Armenia, Asia, etc.
   - Military units: B.E.F. Headquarters, Bernadotte's Army, etc.
   - Geographic features: Baghdad, Belgrad, etc.
   - Key people: Bethmann-Hollweg, Alexander Pope Krosnovitch, etc.

2. **Use corpus index** to find articles mentioning these entities
   - `ignore/wikipedia-institutional-fetishism/analysis/corpus_index.py`
   - Query each missing entity to see if any mention exists
   - If not, fetch articles from Wikipedia categories

3. **Parallel benefit**: IP rate limits cool while doing this
   - Heavy API hitting earlier has likely triggered rate limits
   - Local work (curation, analysis, redaction) doesn't need API
   - Fetch strategically once limits settle

## Retrain & Evaluate

Once corpus expanded with zero-OOV data:

```bash
# Recombine + split + redact + curate
uv run python scripts/combiner.py
uv run python scripts/splitter.py
uv run python scripts/redactor.py --input data/train_clean.jsonl --output data/train_redacted.jsonl --mode train
uv run python scripts/redactor.py --input data/test_clean.jsonl --output data/test_redacted.jsonl --mode test
uv run python src/ner_recovery/curator.py --input data/train_redacted.jsonl --output data/train_redacted_curated.jsonl
uv run python src/ner_recovery/curator.py --input data/test_redacted.jsonl --output data/test_redacted_curated.jsonl

# Train (with adaptive epochs if needed)
uv run train --epochs 5 --output-dir models/zero_oov

# Evaluate
uv run evaluate --model-dir models/zero_oov/final --data data/test_redacted_curated.jsonl --train data/train_redacted_curated.jsonl
```

## Expected Impact

If we achieve **0% OOV**:
- Constrained accuracy should jump to **20%+ range** (from current 7.16%)
- More meaningful benchmark (no "can't predict what you haven't seen" excuses)
- Validates whether model limitation is data or architecture

## Timeline

- Fetch phase: 1–2 days (strategic, not bulk)
- Rate limit cooldown: natural overlap
- Retrain + eval: 2–3 hours
- Analysis: 1 hour

Total: 2–4 days before next major checkpoint.

## Research Philosophy

Still chasing 40%+ accuracy. This 0% OOV milestone is a prerequisite, not a finish line.
