# Session Notes: April 22, 2026 — Curation Strategy & OOV Analysis

## Problem Identified

After expanding corpus to 4,299 articles and retraining with 5 epochs:
- **Constrained accuracy**: 5.82% (marginal +0.24% vs baseline)
- **Free accuracy**: 3.52% (slight decline)
- **OOV rate**: 51.9% (unchanged)

The larger corpus didn't help because **over half of test entities never appear in training data**. This is the bottleneck.

## Root Cause Analysis

Ran OOV analysis on unique test entities (637 total):
- **64.4% OOV rate** (410 entities missing from training candidates)
- **Zero corpus mentions** for all 410 OOV entities

Examining the garbage:
- Numbers: `1,846`, `107–144`, `124`
- Malformed tokens: `10(1/2`, `10(3`
- Percentages: `10%`
- Nonsense: `10 January 1920` parsed as separate entities

**Realization**: The problem isn't missing articles. It's **redaction noise**. The redactor over-redacts garbage that shouldn't be entities.

## Solution: Strict Curation

Built `curator.py` with multi-pass filtering:

1. **Label filtering** — keep only high-signal labels: PERSON, ORG, GPE, EVENT, LOC, FAC, NORP
   - Drop: CARDINAL, PERCENT, ORDINAL, MONEY, LANGUAGE, LAW, PRODUCT, WORK_OF_ART

2. **Length filtering** — drop entities < 3 characters

3. **Pattern filtering** — regex reject:
   - Pure digits
   - Formatted numbers
   - Percentages
   - Malformed tokens
   - Roman numerals
   - Date fragments

### Results

**Training data**:
- 14,028 redactions → 8,188 kept (58.4%)
- Removed 5,840 junk redactions

**Test data**:
- 825 redactions → 559 kept (67.8%)
- Removed 266 junk redactions

Clean curated files ready for retraining.

## Philosophical Note: When to Stop Training

While waiting for epochs, discussed convergence philosophy from CFD background:

**CFD question**: "How long to run the simulation?"
- Answer: Until residuals plateau — you accept some tolerance
- You never reach perfect zero; you ask: "What error margin is acceptable?"

**ML parallel**: "How many epochs?"
- Train until loss plateau — further epochs don't meaningfully improve
- Accept that you'll never reach zero loss
- Ask: "What accuracy threshold matters for the research question?"

**The cost-benefit curve**: 
- Cost of one more epoch = compute time
- Value = maybe +0.2% accuracy (or nothing)
- At some point, diminishing returns exceed the investment

## Research Philosophy: Chasing Perfection

**Decision made**: Won't publish at 15–20% accuracy without exhausting alternatives.

If curation yields modest gains (<20%), next steps:
1. Implement adaptive epochs (stop at loss plateau, not fixed epoch count)
2. Complete politician fetch + index to ensure zero OOV
3. Refine redaction strategy further
4. Try architecture changes if needed

**Target**: 40%+ accuracy before publishing findings. The work deserves that rigor.

## Next Steps (In Order)

1. **Run curated training** (5 epochs, models/curated/)
2. **Evaluate curated model** (both modes)
3. **Re-run OOV analysis** on curated data
4. **Compare accuracy vs baseline** — did curation help enough?
5. **If results still modest** — implement adaptive epochs + continue corpus expansion

## Key Files Created This Session

- `src/ner_recovery/curator.py` — strict redaction filtering
- `src/ner_recovery/oov_analysis.py` — OOV breakdown by label + corpus coverage
- `data/train_redacted_curated.jsonl` — cleaned training data
- `data/test_redacted_curated.jsonl` — cleaned test data
- Updated `README.md` — accessible essay format + Meesum attribution

## Insights

1. **Data quality matters more than data quantity** — expanding corpus didn't help; cleaning it does
2. **Redaction artifacts are real** — numbers and malformed tokens masquerade as entities
3. **OOV is a hard constraint** — you can't predict what you've never seen, no matter how good the model
4. **Convergence is philosophical** — you decide when "good enough" is enough; there's no objective truth

## README Rewrite

Rewrote README as accessible essay (instead of technical spec):
- Sticky-note analogy upfront (what we're doing)
- Why it matters (hallucination detection, understanding probes)
- Honest about current modest results (6% accuracy)
- Vision of what success means (40%+ would be meaningful)
- Included Meesum's attribution for typed redaction tuple idea

Goal: Your wife should understand it now.
