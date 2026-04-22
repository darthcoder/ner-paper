# evals/

Evaluation results and reports.

## File Naming

`eval_{mode}_{timestamp}.txt` — evaluation mode + timestamp

- `eval_constrained_*.txt` — constrained mode (rank from known candidates)
- `eval_free_*.txt` — free mode (generate any token)

## Report Format

Each report contains:

```
Device: mps or cuda or cpu
Model: path/to/model
Data: path/to/test_redacted.jsonl
Mode: constrained or free

RESULTS
  Total redactions: N
  Exact matches: M
  Accuracy: M/N %
  OOV (not in train candidates): X (Y%)  [constrained mode only]

PER-LABEL BREAKDOWN
  PERSON: A/B %
  ORG: C/D %
  ...
```

## Reading Results

### Accuracy

- **Constrained** — typical range 5–15% (entity ranking from known set)
- **Free** — typical range 2–8% (open-ended generation)

### OOV

- **Out-of-Vocabulary**: test entities never seen in training candidates
- High OOV (>50%) means model can't be improved without more training data
- Curation targets reducing OOV by expanding training entity coverage

### Per-Label Performance

- **EVENT**, **LOC** — usually higher (20–40%)
- **PERSON**, **ORG** — usually lower (<5%) due to diversity and rarity
- **CARDINAL**, etc. — often 0% (filtered in curated sets)

## Comparing Runs

To track progress, compare same mode across runs:

```
Baseline (3230 articles, 7 epochs):
  Constrained: 5.58% (46/825)

Expanded (4299 articles, 5 epochs):
  Constrained: 5.82% (48/825)  ← marginal improvement

Curated (filtered noise):
  Constrained: ? (check latest eval)  ← expected improvement
```

Look for patterns: does accuracy climb with more training data? Does curation help?
