# findings/

Analysis results and evaluation reports.

## Files

- `2026-04-22_expanded_corpus_eval.md` — Evaluation on 4,299-article corpus (5 epochs)
  - Constrained accuracy: 5.82% (46/825)
  - Free accuracy: 3.64% (30/825)
  - OOV rate: 51.9% (428 entities never seen in training)
  - Key insight: Larger corpus alone didn't help; OOV is the bottleneck

## Interpreting Results

### Accuracy Scores

- **Constrained mode** — model picks from known entities (candidate list from training)
  - More realistic, gives better numbers
  - Shows what the model learned about ranking entities by context

- **Free mode** — model generates any token
  - Harder, shows raw learning
  - Usually 30–50% lower than constrained

### OOV (Out-of-Vocabulary)

- Entities in test set that never appear in training candidates
- If OOV is high (>50%), no model improvement is possible
- Must expand training data to include more entity instances

### Per-Label Breakdown

- **EVENT**, **LOC** — high accuracy (30%+), well-represented in training
- **PERSON**, **ORG** — low accuracy (<5%), rare or diverse entities
- **CARDINAL**, **PERCENT**, **ORDINAL** — filtered out in curated sets (noise)

## Next Steps

If accuracy is still <20% after curation:
1. Implement adaptive epochs (train until loss plateau)
2. Expand corpus (finish politician fetch, add Monroe Doctrine)
3. Improve redaction strategy
4. Consider architecture changes

Target: 40%+ accuracy before publishing findings.
