#!/bin/bash
set -e  # Exit on any error

echo "=== NER Recovery Pipeline ==="
echo ""

echo "1. Combining corpus..."
uv run python scripts/combiner.py
echo "✓ Corpus combined"
echo ""

echo "2. Splitting 80/20..."
uv run python scripts/splitter.py
echo "✓ Data split"
echo ""

echo "3. Redacting entities..."
uv run python scripts/redactor.py --input data/train_clean.jsonl --output data/train_redacted.jsonl --mode train
uv run python scripts/redactor.py --input data/test_clean.jsonl --output data/test_redacted.jsonl --mode test
echo "✓ Redactions complete"
echo ""

echo "4. Curating redactions..."
uv run python src/ner_recovery/curator.py --input data/train_redacted.jsonl --output data/train_redacted_curated.jsonl
uv run python src/ner_recovery/curator.py --input data/test_redacted.jsonl --output data/test_redacted_curated.jsonl
echo "✓ Curation complete"
echo ""

echo "5. Filtering test set to zero OOV..."
uv run python scripts/filter_zero_oov.py \
    --train data/train_redacted_curated.jsonl \
    --test data/test_redacted_curated.jsonl \
    --output data/test_zero_oov.jsonl
echo "✓ Zero-OOV test set ready"
echo ""

echo "6. Training model (7 epochs)..."
uv run train --epochs 7 --output-dir models/current
echo "✓ Training complete"
echo ""

echo "7. Evaluating..."
uv run evaluate --model-dir models/current/final --data data/test_zero_oov.jsonl --train data/train_redacted_curated.jsonl
echo "✓ Evaluation complete"
echo ""

echo "=== Pipeline finished ==="
