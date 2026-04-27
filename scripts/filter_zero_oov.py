"""Filter test set to zero OOV: keep only redactions whose entity appears in training candidates.

For each test article, strips OOV redactions from the redactions list (the entity text
remains visible in the passage as plain text — just not scored). Articles with no
remaining in-train redactions are dropped entirely.

Usage:
    uv run python scripts/filter_zero_oov.py \
        --train data/train_redacted_curated.jsonl \
        --test data/test_redacted_curated.jsonl \
        --output data/test_zero_oov.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_train_vocab(train_path: Path) -> set[tuple[str, str]]:
    vocab: set[tuple[str, str]] = set()
    with open(train_path) as f:
        for line in f:
            for red in json.loads(line).get("redactions", []):
                vocab.add((red["label"], red["original"].lower()))
    return vocab


def filter_test(train_path: Path, test_path: Path, output_path: Path) -> None:
    vocab = build_train_vocab(train_path)

    kept_articles = 0
    dropped_articles = 0
    kept_redactions = 0
    dropped_redactions = 0

    with open(test_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            record = json.loads(line)
            filtered = [
                r for r in record.get("redactions", [])
                if (r["label"], r["original"].lower()) in vocab
            ]
            dropped_redactions += len(record.get("redactions", [])) - len(filtered)
            if not filtered:
                dropped_articles += 1
                continue
            record["redactions"] = filtered
            fout.write(json.dumps(record) + "\n")
            kept_articles += 1
            kept_redactions += len(filtered)

    total_articles = kept_articles + dropped_articles
    total_redactions = kept_redactions + dropped_redactions
    print(f"Articles : {kept_articles}/{total_articles} kept ({dropped_articles} dropped — no in-train redactions)")
    print(f"Redactions: {kept_redactions}/{total_redactions} kept ({dropped_redactions} OOV stripped)")
    print(f"OOV      : 0.0% (by construction)")
    print(f"Output   : {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    filter_test(args.train, args.test, args.output)


if __name__ == "__main__":
    main()
