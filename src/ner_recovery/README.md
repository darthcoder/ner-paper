# src/ner_recovery/

Core training and evaluation code for the Engrammatic Named Entity Inference project.

## Files

- **`train.py`** — Fine-tune DistilBERT on the redaction task
  - Usage: `uv run train [--epochs N] [--output-dir PATH]`
  - Saves checkpoints per epoch + final model

- **`eval.py`** — Evaluate trained models
  - Usage: `uv run evaluate [--model-dir PATH] [--data FILE] [--train FILE] [--mode {constrained,free}]`
  - Constrained: rank candidates from training set (better numbers)
  - Free: generate any token (shows true learning)

- **`curator.py`** — Curate redactions for clean training data
  - Filters out noise: numbers, malformed tokens, low-confidence entity types
  - Usage: `uv run python src/ner_recovery/curator.py [--input FILE] [--output FILE]`
  - Keeps only: PERSON, ORG, GPE, EVENT, LOC, FAC, NORP

- **`oov_analysis.py`** — Analyze out-of-vocabulary entities
  - Reports which test entities are missing from training candidates
  - Shows corpus coverage (can we find the OOV entities in articles?)
  - Usage: `uv run python src/ner_recovery/oov_analysis.py [--train FILE] [--test FILE] [--corpus FILE]`

## Workflow

```
Train → Evaluate → Analyze OOV → Curate (if needed) → Retrain
```

1. Train on raw redacted data
2. Evaluate to see baseline accuracy
3. Run OOV analysis to identify bottlenecks
4. Curate if noisy redactions found
5. Retrain on clean data
