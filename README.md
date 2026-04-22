# Engrammatic Named Entity Inference

## What's This About?

Imagine you're reading a history book where every person's name has been covered up with a sticky note that says `[REDACTED:PERSON]`. Your job is to guess who it was based on everything else written around it. That's essentially what this project teaches an AI to do — but with more sophistication.

The core question we're asking: **Can a language model learn to reconstruct missing named entities (people, places, organizations) just from context?** And more importantly, what does it tell us about how well the model actually understands the world?

### Why This Matters

Language models like GPT, Claude, and others are deployed everywhere — answering questions, writing code, summarizing documents. But we don't fully understand what they actually "know" versus what they memorized or hallucinate. If we can train a model to successfully fill in redacted names, we can probe its real knowledge of historical actors, geography, and relationships. This is a window into the model's internal understanding and a potential detector for hallucinations.

Our specific angle: **using World War I history as a test bed.** WWI is rich with named entities (generals, diplomats, nations, battles, treaties), well-documented, and challenging enough that it reveals gaps in a model's knowledge.

**A key insight from Meesum**: Instead of just masking entity names with a generic `[MASK]` token, we embed the entity *type* in the mask itself — `[REDACTED:PERSON]`, `[REDACTED:GPE]`, etc. This gives the model a strong prior: predicting a country name, a person name, and a date are very different tasks. It's a small detail with big implications for what the model can learn.

---

## The Idea

Here's the approach:

1. **Collect historical texts** about World War I from Wikipedia, archives, and historical documents
2. **Automatically identify named entities** (using a standard NER tagger) — people, places, organizations, events
3. **Redact some of them** by replacing the name with a token that says the type: `[REDACTED:PERSON]`, `[REDACTED:GPE]` (geographic), etc.
4. **Train a language model** to predict the original name from the surrounding context
5. **Test the model** and see how well it recovers the redacted information

The trick: by telling the model "this is definitely a person's name, not a country," we give it a strong hint, which makes the task more realistic. It's not just guessing missing words — it's understanding what kind of entity fits the context.

### Current Status (Honestly)

We're in the middle of this experiment. Our best results so far show the model can recover ~6% of redacted names correctly when we let it choose from a list of known candidates. When we let it generate freely, it's down to ~3–4%. These are modest numbers, and that's actually useful data — it tells us that even with context, reconstructing specific historical names is genuinely hard.

The bigger insight: **over half of the entities the model misses have never appeared in its training data at all.** This is a major bottleneck. It's not that the model can't learn to recognize patterns — it's that some entities are simply absent from its training corpus.

---

## What If We Succeed?

If we could push this to, say, 40–50% accuracy (still far from perfect, but meaningful), it would mean:

- **Better hallucination detection**: We'd have a probe for when models confabulate names or facts
- **Understanding knowledge gaps**: We'd know which historical actors and events a model genuinely doesn't know about
- **Smarter language models**: Techniques learned here could improve how models handle factual entities in general
- **A better measure of "understanding"**: Instead of just asking "does the model pass a test?", we'd ask "does it actually understand context and relationships?"

This could become a standard benchmark for evaluating frontier models — a way to ask not just "is this model smart?" but "does this model actually know what it claims to know?"

---

## The Technical Side (For Those Who Want It)

**The model**: We use DistilBERT, a lighter version of BERT (~66 million parameters). It's fast to train and good at understanding context without being enormous.

**The data**: ~4,300 World War I articles from Wikipedia, with careful curation to remove noisy redactions (numbers, malformed tokens, etc.). We split 80/20 into training and test sets.

**The training**: We show the model passages where some named entities are hidden behind `[MASK]` tokens. It learns to predict what was hidden by looking at the surrounding words and the hint (e.g., "this is a person").

**Two evaluation modes**:
- **Constrained**: The model picks from a list of known entities. More realistic, gives better numbers.
- **Free**: The model generates any name. Shows what it actually learned, usually harder.

---

## How to Use This

### Quick Start

```bash
# Install dependencies
uv sync
python -m spacy download en_core_web_sm

# Data pipeline (if starting from scratch)
uv run python scripts/combiner.py          # Combine sources
uv run python scripts/splitter.py          # 80/20 split
uv run python scripts/redactor.py --input data/train_clean.jsonl --output data/train_redacted.jsonl --mode train
uv run python scripts/redactor.py --input data/test_clean.jsonl --output data/test_redacted.jsonl --mode test

# Data curation (removes noisy redactions — strongly recommended)
uv run python src/ner_recovery/curator.py --input data/train_redacted.jsonl --output data/train_redacted_curated.jsonl
uv run python src/ner_recovery/curator.py --input data/test_redacted.jsonl --output data/test_redacted_curated.jsonl

# Train the model (with curated data)
uv run train --epochs 5 --output-dir models/current

# Evaluate (constrained and free modes)
uv run evaluate --model-dir models/current/final --data data/test_redacted_curated.jsonl --train data/train_redacted_curated.jsonl
uv run evaluate --model-dir models/current/final --data data/test_redacted_curated.jsonl --train data/train_redacted_curated.jsonl --mode free

# Analyze OOV (out-of-vocabulary entities)
uv run python src/ner_recovery/oov_analysis.py --train data/train_redacted_curated.jsonl --test data/test_redacted_curated.jsonl
```

### Data

Raw corpora live in `data/`:
- `wwi_extended.jsonl` — ~4,300 Wikipedia articles, cleaned plaintext
- `train_redacted.jsonl` — training set with redacted entities
- `test_redacted.jsonl` — test set for evaluation

Each redaction record contains:
- The article title and text
- A list of redactions: `{original: "Serbia", label: "GPE", start: 1107, end: 1119}`

### Model Outputs

Trained models save to `models/`:
- `models/current/` — latest training run
- `models/final/` — stable checkpoint

Each contains the full model weights and tokenizer.

---

## Key Files

| File | What It Does |
|---|---|
| `src/ner_recovery/train.py` | Fine-tunes DistilBERT on redaction task; supports `--epochs N` and `--output-dir PATH` |
| `src/ner_recovery/eval.py` | Evaluates model in constrained and free modes; accepts `--model-dir`, `--data`, `--train`, `--mode` |
| `src/ner_recovery/curator.py` | **[NEW]** Filters noisy redactions (numbers, malformed tokens, low-confidence labels) for clean training |
| `src/ner_recovery/oov_analysis.py` | **[NEW]** Reports which test entities are missing from training candidates; identifies corpus coverage gaps |
| `scripts/redactor.py` | Uses spaCy to redact entities in text; supports `--mode train` (weighted) or `--mode test` (uniform) |
| `scripts/combiner.py` | Combines all `*_clean.jsonl` sources into `all_clean.jsonl` (deduplicates by pageid) |
| `scripts/splitter.py` | Splits `all_clean.jsonl` into 80/20 train/test with seed=42 |

---

## The Summit: What We're Aiming For

If this work succeeds, it becomes a tool for understanding what language models really know. Instead of asking "is this model intelligent?", we ask "does this model understand the structure of history, geography, and human relationships?" That's a much harder question, and a much more useful one.

Imagine a future where every large language model comes with an "understanding score" — not just accuracy on a benchmark, but a measure of how well it genuinely knows entities, facts, and their relationships. This project is a step toward that.

We're not there yet. But the road is clear, and the question is worth asking.

---

## References

Key papers we're building on:

- **Engram**: Conditional lookup + constrained decoding for factual generation
- **BERT**: The foundation for masked language model pretraining
- **Hallucination research**: Understanding when and why models make things up
- **Entity understanding**: How well do models actually track named entities?

Full references in `refs/`.

---

## Questions?

This is research in progress. If you want to understand more or contribute, reach out.
