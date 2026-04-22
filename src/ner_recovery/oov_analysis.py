"""OOV Analysis — identify test entities missing from training candidates.

Builds an inverted index of training entities per label, then compares against
test entities to report gaps. Suggests which corpus articles to fetch to
eliminate OOV.

Usage:
    uv run python src/ner_recovery/oov_analysis.py \
        --train data/train_redacted.jsonl \
        --test data/test_redacted.jsonl \
        --corpus data/wwi_extended.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_redactions(path: Path) -> tuple[dict[str, set[str]], list[str]]:
    """Load redactions from JSONL, return (entities_by_label, all_entities)."""
    entities_by_label: dict[str, set[str]] = defaultdict(set)
    all_entities = []

    with open(path) as f:
        for line in f:
            record = json.loads(line)
            for r in record.get("redactions", []):
                label = r["label"]
                entity = r["original"].strip()
                entities_by_label[label].add(entity)
                all_entities.append(entity)

    return dict(entities_by_label), all_entities


def build_corpus_mentions(corpus_path: Path, test_entities: set[str]) -> dict[str, list[str]]:
    """Which corpus articles mention each test entity?"""
    mentions: dict[str, list[str]] = defaultdict(list)

    with open(corpus_path) as f:
        for line in f:
            article = json.loads(line)
            title = article.get("title", "")
            text = article.get("text", "").lower()

            for entity in test_entities:
                if entity.lower() in text:
                    mentions[entity].append(title)

    return dict(mentions)


def main() -> None:
    parser = argparse.ArgumentParser(description="OOV analysis for NER evaluation")
    parser.add_argument("--train", type=Path, default=Path("data/train_redacted.jsonl"))
    parser.add_argument("--test", type=Path, default=Path("data/test_redacted.jsonl"))
    parser.add_argument("--corpus", type=Path, default=Path("data/wwi_extended.jsonl"))
    args = parser.parse_args()

    print("=" * 80)
    print("OOV ANALYSIS — NER Entity Coverage")
    print("=" * 80)

    # Load training and test entities
    print(f"\nLoading training entities from {args.train}...")
    train_by_label, train_all = load_redactions(args.train)
    train_set = set(train_all)
    print(f"  {len(train_set)} unique training entities across {len(train_by_label)} labels")

    print(f"\nLoading test entities from {args.test}...")
    test_by_label, test_all = load_redactions(args.test)
    test_set = set(test_all)
    print(f"  {len(test_set)} unique test entities across {len(test_by_label)} labels")

    # OOV analysis
    print("\n" + "=" * 80)
    print("OOV BREAKDOWN BY LABEL")
    print("=" * 80)

    total_oov = 0
    oov_entities_all: set[str] = set()

    print(f"\n{'Label':<15} {'Test':<6} {'In Train':<8} {'OOV':<6} {'OOV %':<8}")
    print("-" * 50)

    for label in sorted(test_by_label.keys()):
        test_ents = test_by_label[label]
        train_ents = train_by_label.get(label, set())
        oov_ents = test_ents - train_ents
        oov_count = len(oov_ents)
        oov_pct = 100 * oov_count / len(test_ents) if test_ents else 0

        total_oov += oov_count
        oov_entities_all.update(oov_ents)

        in_train = len(test_ents) - oov_count
        print(f"{label:<15} {len(test_ents):<6} {in_train:<8} {oov_count:<6} {oov_pct:>6.1f}%")

    print("-" * 50)
    total_test = sum(len(v) for v in test_by_label.values())
    oov_pct_total = 100 * total_oov / total_test if total_test else 0
    print(f"{'TOTAL':<15} {total_test:<6} {total_test - total_oov:<8} {total_oov:<6} {oov_pct_total:>6.1f}%")

    # Find corpus articles mentioning OOV entities
    if oov_entities_all:
        print(f"\n" + "=" * 80)
        print(f"CORPUS COVERAGE FOR {len(oov_entities_all)} OOV ENTITIES")
        print("=" * 80)

        print(f"\nSearching corpus for OOV entities...")
        mentions = build_corpus_mentions(args.corpus, oov_entities_all)

        # Categorize OOV entities
        has_corpus_mention = [e for e in oov_entities_all if e in mentions]
        no_mention = oov_entities_all - set(has_corpus_mention)

        print(f"\n  OOV entities WITH corpus mentions: {len(has_corpus_mention)}")
        print(f"  OOV entities WITH NO corpus mention: {len(no_mention)}")

        if no_mention:
            print(f"\nEntities with ZERO corpus mentions (need external fetch):")
            for entity in sorted(no_mention)[:20]:
                print(f"    - {entity}")
            if len(no_mention) > 20:
                print(f"    ... and {len(no_mention) - 20} more")

        if has_corpus_mention:
            print(f"\nEntities WITH corpus mentions (add to training redactions):")
            # Show top 10 by mention count
            top = sorted(
                has_corpus_mention,
                key=lambda e: len(mentions[e]),
                reverse=True
            )[:10]
            for entity in top:
                print(f"    {entity:<30} ({len(mentions[entity])} articles)")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Current OOV rate: {oov_pct_total:.1f}% ({total_oov}/{total_test})")
    print(f"Target: 0.0% OOV (all test entities in training)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
