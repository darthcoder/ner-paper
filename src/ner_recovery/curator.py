"""Redaction Curator — strict multi-pass filtering for high-confidence NER.

Removes low-quality redactions (numbers, short tokens, malformed entities)
before training. Outputs clean training data with curated redactions only.

Filters:
  1. Length: drop entities < 3 characters
  2. Labels: keep only high-signal labels (PERSON, ORG, GPE, EVENT, LOC, FAC)
  3. Patterns: drop pure numbers, percentages, dates, malformed tokens

Usage:
    uv run python src/ner_recovery/curator.py \
        --input data/train_redacted.jsonl \
        --output data/train_redacted_curated.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


# High-signal NER labels (drop CARDINAL, PERCENT, ORDINAL, MONEY, LANGUAGE, LAW, PRODUCT, WORK_OF_ART)
KEEP_LABELS = frozenset(["PERSON", "ORG", "GPE", "EVENT", "LOC", "FAC", "NORP"])

# Patterns to reject
REJECT_PATTERNS = [
    r"^\d+$",                    # Pure digits (1, 123, 1846)
    r"^\d{1,3}(?:,\d{3})*$",     # Formatted numbers (1,846, 127,000)
    r"^\d+%$",                   # Percentages (10%)
    r"^\d+(?:\.\d+)?(?:\s*(?:million|billion|thousand))?$",  # Numeric + unit
    r"^\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)",  # Dates
    r"^\d+\s*\(\d",              # Malformed (10(1/2, 10(3)
    r"^\d{4}s$",                 # Decades (1550s)
    r"^[IVX]+$",                 # Roman numerals
    r"^\d+–\d+$",                # Ranges (107–144)
    r"^\d+\s+million$|^\d+\s+thousand$",  # "18 million"
    r"^(a|an|the|and|or|of|in|to|for|by)$",  # Stray articles (should not happen, but safety)
]

REJECT_RE = re.compile("|".join(f"(?:{p})" for p in REJECT_PATTERNS), re.IGNORECASE)


def is_high_quality_redaction(entity: str, label: str) -> bool:
    """Pass through strict quality checks."""
    # Check label
    if label not in KEEP_LABELS:
        return False

    # Check length
    if len(entity.strip()) < 3:
        return False

    # Check patterns
    if REJECT_RE.match(entity.strip()):
        return False

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate redactions for high-confidence NER only")
    parser.add_argument("--input", type=Path, default=Path("data/train_redacted.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/train_redacted_curated.jsonl"))
    args = parser.parse_args()

    print("=" * 80)
    print("REDACTION CURATOR")
    print("=" * 80)
    print(f"\nInput: {args.input}")
    print(f"Output: {args.output}")
    print(f"Keep labels: {', '.join(sorted(KEEP_LABELS))}")
    print("\nFilters:")
    print("  1. Length >= 3 characters")
    print("  2. Label in high-signal set (not CARDINAL, PERCENT, ORDINAL, etc.)")
    print("  3. No numeric/date/malformed patterns")

    stats = {
        "articles_in": 0,
        "redactions_in": 0,
        "redactions_kept": 0,
        "redactions_rejected": 0,
        "articles_out": 0,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.input.open() as fin, args.output.open("w") as fout:
        for line in fin:
            article = json.loads(line)
            stats["articles_in"] += 1

            redactions = article.get("redactions", [])
            stats["redactions_in"] += len(redactions)

            # Filter redactions
            curated = []
            for r in redactions:
                entity = r["original"]
                label = r["label"]
                if is_high_quality_redaction(entity, label):
                    curated.append(r)
                    stats["redactions_kept"] += 1
                else:
                    stats["redactions_rejected"] += 1

            # Only write if article has curated redactions
            if curated:
                article["redactions"] = curated
                fout.write(json.dumps(article) + "\n")
                stats["articles_out"] += 1

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Articles in:           {stats['articles_in']:,}")
    print(f"Articles out:          {stats['articles_out']:,}")
    print(f"  (articles with ≥1 curated redaction)")
    print(f"\nRedactions in:         {stats['redactions_in']:,}")
    print(f"Redactions kept:       {stats['redactions_kept']:,}")
    print(f"Redactions rejected:   {stats['redactions_rejected']:,}")
    print(f"Retention rate:        {100 * stats['redactions_kept'] / stats['redactions_in']:.1f}%")
    print(f"\nOutput: {args.output}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
