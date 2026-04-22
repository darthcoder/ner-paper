# Expanded Corpus Evaluation (April 22, 2026)

## Experiment Setup
- **Corpus**: 4,299 articles (expanded from 3,230)
- **Training epochs**: 5 (compromise for larger dataset)
- **Model**: DistilBERT MLM (`models/current/final`)
- **Data split**: 80/20 train/test (seed=42)

## Results

### Constrained Mode
- **Accuracy**: 5.82% (48/825 correct)
- **Change**: +0.24% vs baseline (5.58%)
- **OOV**: 428/825 (51.9%) — unchanged from baseline

### Free Mode
- **Accuracy**: 3.52% (29/825 correct)
- **Change**: -0.12% vs baseline (3.64%)

### Per-Label Breakdown (Constrained)
| Label | Accuracy | vs Baseline | Count |
|-------|----------|-----------|-------|
| EVENT | 38.46% | — | 15/39 |
| LOC | 19.05% | — | 4/21 |
| LANGUAGE | 50.00% | — | 1/2 |
| NORP | 6.25% | ↑ | 7/112 |
| GPE | 6.11% | ↓ | 8/131 |
| CARDINAL | 3.90% | — | 3/77 |
| ORG | 3.14% | ↑ | 5/159 |
| DATE | 2.88% | — | 4/139 |
| PERSON | 0.93% | — | 1/107 |
| Other | 0% | — | 0/66 |

## Key Insights

1. **Corpus expansion effect**: +1,069 articles → marginal +0.24% in constrained mode
   - Data scaling alone is not sufficient
   
2. **OOV is the primary bottleneck**: 51.9% of test entities have no candidates
   - Over half of predictions have an empty search space
   - No amount of model tuning can fix this
   
3. **Epoch count**: 5 epochs vs prior 7 may have contributed to free-mode decline
   - But constrained mode still improved slightly

## Next Steps

**Critical**: Fix OOV by ensuring 100% of test entities appear in training candidates
- Approach: Merge train/test candidate sets or ensure test redactions are subsets of training entities
- This would collapse the effective test space but would validate whether OOV is truly the limiting factor
