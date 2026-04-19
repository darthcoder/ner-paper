"""Recursively fetch all articles under a Wikipedia category tree.

Uses the wikipedia-api library which handles rate limiting correctly.
Traverses subcategories up to a configurable depth, deduplicates across
branches, and writes raw wikitext to JSONL.

Usage:
    uv run python scripts/fetch_wikipedia_portal.py \
        --root "World War I" \
        --output data/wwi_portal_corpus.jsonl \
        --depth 2
"""
from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import requests
import wikipediaapi

HEADERS = {"User-Agent": "ner-recovery/0.2 (research; github.com/ner-paper)"}
WIKITEXT_API = "https://en.wikipedia.org/w/api.php"
SLEEP = 1.0
MAX_RETRIES = 6


def fetch_wikitext(title: str) -> str | None:
    """Fetch raw wikitext for a page title via the MediaWiki action API."""
    params = {
        "action": "query",
        "titles": title,
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "format": "json",
    }
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(WIKITEXT_API, params=params, headers=HEADERS, timeout=30)
            if r.status_code == 429:
                wait = 2 ** (attempt + 3)
                print(f"    429 — waiting {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            pages = r.json()["query"]["pages"]
            page = next(iter(pages.values()))
            if "missing" in page:
                return None
            return (
                page.get("revisions", [{}])[0]
                .get("slots", {})
                .get("main", {})
                .get("*", "") or None
            )
        except requests.exceptions.RequestException as e:
            wait = 2 ** (attempt + 2)
            print(f"    request error ({e}) — waiting {wait}s")
            time.sleep(wait)
    return None


def crawl_category_tree(
    wiki: wikipediaapi.Wikipedia,
    root_category: str,
    max_depth: int,
) -> set[str]:
    """BFS over category tree using wikipedia-api; return all article titles."""
    article_titles: set[str] = set()
    seen_cats: set[str] = {root_category}
    queue: deque[tuple[str, int]] = deque([(root_category, 0)])

    while queue:
        cat_name, depth = queue.popleft()
        print(f"  [d={depth}] Category:{cat_name}")

        cat_page = wiki.page(f"Category:{cat_name}")
        if not cat_page.exists():
            continue

        for title, member in cat_page.categorymembers.items():
            if member.ns == wikipediaapi.Namespace.MAIN:
                article_titles.add(title)
            elif member.ns == wikipediaapi.Namespace.CATEGORY and depth < max_depth:
                # Strip "Category:" prefix for the queue
                sub = title.removeprefix("Category:")
                if sub not in seen_cats:
                    seen_cats.add(sub)
                    queue.append((sub, depth + 1))

        time.sleep(SLEEP)

    return article_titles


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recursively fetch Wikipedia category tree to JSONL"
    )
    parser.add_argument("--root", default="World War I",
                        help="Root category name without 'Category:' prefix")
    parser.add_argument("--output", type=Path, default=Path("data/wwi_portal_corpus.jsonl"))
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    seen_pageids: set[int] = set()
    if args.resume and args.output.exists():
        with args.output.open() as f:
            for line in f:
                seen_pageids.add(json.loads(line)["pageid"])
        print(f"Resuming: {len(seen_pageids)} articles already on disk")

    wiki = wikipediaapi.Wikipedia(
        language="en",
        user_agent="ner-recovery/0.2 (research; github.com/ner-paper)",
    )

    titles_cache = args.output.with_suffix(".titles.json")
    if args.resume and titles_cache.exists():
        titles = set(json.loads(titles_cache.read_text()))
        print(f"Loaded {len(titles)} titles from cache (skipping crawl)")
    else:
        print(f"Crawling category tree from 'Category:{args.root}' (depth={args.depth})...")
        titles = crawl_category_tree(wiki, args.root, args.depth)
        titles_cache.write_text(json.dumps(sorted(titles)))
        print(f"\nFound {len(titles)} unique article titles")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    written = skipped = errors = 0

    with args.output.open(mode) as f:
        for i, title in enumerate(sorted(titles)):
            try:
                # Get pageid via wikipedia-api (it's already cached from crawl).
                page = wiki.page(title)
                if not page.exists():
                    skipped += 1
                    continue

                pageid = page.pageid
                if pageid in seen_pageids:
                    skipped += 1
                    continue

                wikitext = fetch_wikitext(title)
                if not wikitext:
                    skipped += 1
                    continue

                seen_pageids.add(pageid)
                f.write(json.dumps({
                    "pageid": pageid,
                    "title": title,
                    "wikitext": wikitext,
                }) + "\n")
                written += 1

                if written % 100 == 0:
                    print(f"  {written} written ({i+1}/{len(titles)} processed, {errors} errors)")
                time.sleep(SLEEP)

            except Exception as e:
                errors += 1
                print(f"  ERROR {title}: {e}")
                time.sleep(SLEEP)

    print(f"\nDone → {args.output}")
    print(f"  Written : {written}")
    print(f"  Skipped : {skipped}")
    print(f"  Errors  : {errors}")


if __name__ == "__main__":
    main()
