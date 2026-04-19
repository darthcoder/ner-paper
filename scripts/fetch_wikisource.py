"""Fetch WW1-related documents from English Wikisource.

Traverses the Wikisource category tree for World War I primary sources:
treaties, dispatches, memoirs, official histories, parliamentary records.
Writes plain wikitext to JSONL in the same schema as the Wikipedia fetcher.

Usage:
    uv run python scripts/fetch_wikisource.py \
        --output data/wikisource_corpus.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import requests

HEADERS = {"User-Agent": "ner-recovery/0.2 (research; github.com/ner-paper)"}
API = "https://en.wikisource.org/w/api.php"
SLEEP = 5.0
MAX_RETRIES = 6

# Root categories on Wikisource covering WW1 primary sources.
ROOT_CATEGORIES = [
    "Category:World War I",
    "Category:World War I documents",
]


def api_get(params: dict) -> dict:
    for attempt in range(MAX_RETRIES):
        r = requests.get(API, params=params, headers=HEADERS, timeout=30)
        if r.status_code == 429:
            wait = 2 ** (attempt + 2)
            print(f"    429 rate limit — waiting {wait}s")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries: {params}")


def get_category_members(category: str, cmtype: str) -> list[dict]:
    members: list[dict] = []
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": category,
        "cmtype": cmtype,
        "cmlimit": 500,
        "format": "json",
    }
    while True:
        data = api_get(params)
        members.extend(data["query"]["categorymembers"])
        cont = data.get("continue", {}).get("cmcontinue")
        if not cont:
            break
        params["cmcontinue"] = cont
        time.sleep(SLEEP)
    return members


def crawl_category_tree(roots: list[str], max_depth: int) -> set[str]:
    article_titles: set[str] = set()
    seen_cats: set[str] = set(roots)
    queue: deque[tuple[str, int]] = deque((r, 0) for r in roots)

    while queue:
        cat, depth = queue.popleft()
        print(f"  [d={depth}] {cat}")

        for m in get_category_members(cat, "page"):
            # Skip talk pages, author pages, help pages, etc.
            ns_prefixes = ("Talk:", "Author:", "Help:", "Wikisource:", "File:", "Template:")
            if not any(m["title"].startswith(p) for p in ns_prefixes):
                article_titles.add(m["title"])
        time.sleep(SLEEP)

        if depth < max_depth:
            for sub in get_category_members(cat, "subcat"):
                title = sub["title"]
                if title not in seen_cats:
                    seen_cats.add(title)
                    queue.append((title, depth + 1))
            time.sleep(SLEEP)

    return article_titles


def fetch_page(title: str) -> dict | None:
    data = api_get({
        "action": "query",
        "titles": title,
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "format": "json",
    })
    pages = data["query"]["pages"]
    page = next(iter(pages.values()))
    if "missing" in page:
        return None
    wikitext = (
        page.get("revisions", [{}])[0]
        .get("slots", {})
        .get("main", {})
        .get("*", "")
    )
    if not wikitext or len(wikitext) < 200:
        return None
    return {
        "pageid": page["pageid"],
        "title": page["title"],
        "wikitext": wikitext,
        "source": "wikisource",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch WW1 Wikisource documents to JSONL"
    )
    parser.add_argument("--output", type=Path, default=Path("data/wikisource_corpus.jsonl"))
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    seen_pageids: set[int] = set()
    if args.resume and args.output.exists():
        with args.output.open() as f:
            for line in f:
                seen_pageids.add(json.loads(line)["pageid"])
        print(f"Resuming: {len(seen_pageids)} pages already on disk")

    titles_cache = args.output.with_suffix(".titles.json")
    if args.resume and titles_cache.exists():
        titles = set(json.loads(titles_cache.read_text()))
        print(f"Loaded {len(titles)} titles from cache (skipping crawl)")
    else:
        print(f"Crawling Wikisource WW1 categories (depth={args.depth})...")
        titles = crawl_category_tree(ROOT_CATEGORIES, args.depth)
        titles_cache.write_text(json.dumps(sorted(titles)))
        print(f"\nFound {len(titles)} unique page titles")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    written = skipped = errors = 0

    with args.output.open(mode) as f:
        for i, title in enumerate(sorted(titles)):
            try:
                page = fetch_page(title)
                if page is None or page["pageid"] in seen_pageids:
                    skipped += 1
                    continue
                seen_pageids.add(page["pageid"])
                f.write(json.dumps(page) + "\n")
                written += 1
                if written % 50 == 0:
                    print(f"  {written} written ({i+1}/{len(titles)} processed)")
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
