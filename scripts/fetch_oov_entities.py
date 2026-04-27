#!/usr/bin/env python3
"""Fetch Wikipedia articles for OOV entities missing from training candidates.

Computes OOV = test entities - train entities, then fetches a Wikipedia article
for each missing entity and appends it to wwi_extended.jsonl.

Usage:
    uv run python scripts/fetch_oov_entities.py \
        --train data/train_redacted_curated.jsonl \
        --test data/test_redacted_curated.jsonl \
        --output data/wwi_extended.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import requests
from tqdm import tqdm

HEADERS = {"User-Agent": "ner-recovery/0.2 (research; github.com/ner-paper)"}
WIKITEXT_API = "https://en.wikipedia.org/w/api.php"
SLEEP = 0.5
MAX_RETRIES = 6

NOISE_PATTERNS = re.compile(
    r'\|'           # wiki markup artifact (pipe)
    r'|\d+px'       # image size reference
    r'|[()[\]]'     # parentheses/brackets suggest extraction artifact
    r'|\d{4}\)'     # year followed by closing paren
)


def is_noise(entity: str) -> bool:
    return bool(NOISE_PATTERNS.search(entity)) or len(entity) < 2


def load_entities_by_label(path: Path) -> dict[str, set[str]]:
    entities: dict[str, set[str]] = defaultdict(set)
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            for r in record.get("redactions", []):
                entities[r["label"]].add(r["original"].strip())
    return dict(entities)


def compute_oov(
    train_path: Path, test_path: Path
) -> set[str]:
    train = load_entities_by_label(train_path)
    test = load_entities_by_label(test_path)

    oov: set[str] = set()
    for label, test_ents in test.items():
        train_ents = train.get(label, set())
        oov.update(test_ents - train_ents)

    return oov


def load_existing_pageids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    seen: set[int] = set()
    with open(path) as f:
        for line in f:
            try:
                seen.add(json.loads(line)["pageid"])
            except (json.JSONDecodeError, KeyError):
                pass
    return seen


def fetch_article(title: str) -> dict | None:
    """Fetch wikitext and real pageid for a title via the MediaWiki API."""
    params = {
        "action": "query",
        "titles": title,
        "prop": "revisions|info",
        "rvprop": "content",
        "rvslots": "main",
        "inprop": "url",
        "format": "json",
        "redirects": 1,
    }
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(WIKITEXT_API, params=params, headers=HEADERS, timeout=30)
            if r.status_code == 429:
                wait = 2 ** (attempt + 3)
                print(f"  429 — waiting {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            pages = r.json()["query"]["pages"]
            page = next(iter(pages.values()))
            if "missing" in page or int(next(iter(pages))) < 0:
                return None
            wikitext = (
                page.get("revisions", [{}])[0]
                .get("slots", {})
                .get("main", {})
                .get("*", "") or None
            )
            if not wikitext:
                return None
            return {
                "pageid": page["pageid"],
                "title": page["title"],
                "wikitext": wikitext,
            }
        except (requests.exceptions.RequestException, KeyError, StopIteration) as e:
            wait = 2 ** (attempt + 2)
            print(f"  error ({e}) — waiting {wait}s")
            time.sleep(wait)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Wikipedia articles for OOV entities")
    parser.add_argument("--train", type=Path, default=Path("data/train_redacted_curated.jsonl"))
    parser.add_argument("--test", type=Path, default=Path("data/test_redacted_curated.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/wwi_extended.jsonl"))
    args = parser.parse_args()

    print("Computing OOV entities...")
    oov_raw = compute_oov(args.train, args.test)
    oov = {e for e in oov_raw if not is_noise(e)}
    filtered = len(oov_raw) - len(oov)
    print(f"  {len(oov_raw)} OOV entities → {len(oov)} after filtering {filtered} noise entries")

    existing_ids = load_existing_pageids(args.output)
    print(f"  {len(existing_ids)} articles already in {args.output.name}")
    print()

    fetched = skipped = failed = duplicate = 0

    with open(args.output, "a") as f:
        for entity in tqdm(sorted(oov), desc="Fetching"):
            article = fetch_article(entity)
            if article is None:
                failed += 1
            elif article["pageid"] in existing_ids:
                duplicate += 1
            else:
                existing_ids.add(article["pageid"])
                f.write(json.dumps(article) + "\n")
                fetched += 1
            time.sleep(SLEEP)

    print()
    print(f"  Fetched  : {fetched}")
    print(f"  Duplicate: {duplicate}")
    print(f"  Failed   : {failed}")
    print(f"  Output   : {args.output}")


if __name__ == "__main__":
    main()
