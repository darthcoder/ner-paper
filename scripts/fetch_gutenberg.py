"""Fetch WW1-related books from Project Gutenberg.

Uses the Gutenberg search API to find books with WW1-related subjects,
downloads plain text, and writes to JSONL. Gutenberg plain text is already
clean — no wikitext stripping needed — so the janitor should pass it through
with minimal processing.

Output schema matches other fetchers: {pageid, title, wikitext, source}
where 'wikitext' contains the plain text (field name kept for pipeline compat).

Usage:
    uv run python scripts/fetch_gutenberg.py \
        --output data/gutenberg_corpus.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

HEADERS = {"User-Agent": "ner-recovery/0.2 (research; github.com/ner-paper)"}
GUTENDEX_API = "https://gutendex.com/books"
SLEEP = 1.0  # Gutenberg asks for polite crawling

# Search terms that reliably surface WW1 histories and memoirs.
SEARCH_QUERIES = [
    "world war",
    "great war",
    "western front",
    "trench warfare",
    "gallipoli",
    "somme",
    "verdun",
    "armistice",
]

# Minimum plain-text length to bother keeping (chars).
MIN_LENGTH = 5_000


def search_books(query: str) -> list[dict]:
    """Return all Gutenberg books matching a subject/title query."""
    books: list[dict] = []
    url = GUTENDEX_API
    params = {"search": query, "languages": "en"}
    while url:
        r = requests.get(url, params=params, headers=HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()
        books.extend(data["results"])
        url = data.get("next")
        params = {}  # next URL already has params encoded
        time.sleep(SLEEP)
    return books


def best_text_url(book: dict) -> str | None:
    """Return the plain text download URL for a book, preferring UTF-8."""
    formats = book.get("formats", {})
    for fmt in ("text/plain; charset=utf-8", "text/plain; charset=us-ascii", "text/plain"):
        if fmt in formats:
            return formats[fmt]
    return None


def fetch_text(url: str) -> str | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=60)
        r.raise_for_status()
        return r.text
    except Exception:
        return None


def strip_gutenberg_header_footer(text: str) -> str:
    """Remove the standard Gutenberg boilerplate header and footer."""
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG",
        "***START OF THE PROJECT GUTENBERG",
        "*** START OF THIS PROJECT GUTENBERG",
    ]
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG",
        "***END OF THE PROJECT GUTENBERG",
        "*** END OF THIS PROJECT GUTENBERG",
    ]

    start_idx = 0
    for marker in start_markers:
        pos = text.find(marker)
        if pos != -1:
            # Skip to end of this line.
            start_idx = text.find("\n", pos) + 1
            break

    end_idx = len(text)
    for marker in end_markers:
        pos = text.find(marker)
        if pos != -1:
            end_idx = pos
            break

    return text[start_idx:end_idx].strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch WW1 Project Gutenberg books to JSONL"
    )
    parser.add_argument("--output", type=Path, default=Path("data/gutenberg_corpus.jsonl"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    seen_ids: set[int] = set()
    if args.resume and args.output.exists():
        with args.output.open() as f:
            for line in f:
                seen_ids.add(json.loads(line)["pageid"])
        print(f"Resuming: {len(seen_ids)} books already on disk")

    # Collect candidate books across all queries, deduplicate by Gutenberg ID.
    candidates: dict[int, dict] = {}
    for query in SEARCH_QUERIES:
        print(f"Searching: '{query}'")
        for book in search_books(query):
            candidates[book["id"]] = book
    print(f"\nFound {len(candidates)} unique candidate books")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    written = skipped = errors = 0

    with args.output.open(mode) as f:
        for i, (book_id, book) in enumerate(candidates.items()):
            if book_id in seen_ids:
                skipped += 1
                continue

            url = best_text_url(book)
            if url is None:
                skipped += 1
                continue

            try:
                raw = fetch_text(url)
                if raw is None:
                    skipped += 1
                    continue

                text = strip_gutenberg_header_footer(raw)
                if len(text) < MIN_LENGTH:
                    skipped += 1
                    continue

                authors = ", ".join(
                    a.get("name", "") for a in book.get("authors", [])
                )
                record = {
                    "pageid": book_id,
                    "title": book.get("title", f"gutenberg_{book_id}"),
                    "wikitext": text,  # plain text, not wikitext — pipeline compat
                    "source": "gutenberg",
                    "authors": authors,
                }
                f.write(json.dumps(record) + "\n")
                written += 1
                seen_ids.add(book_id)

                if written % 20 == 0:
                    print(f"  {written} written ({i+1}/{len(candidates)} processed)")
                time.sleep(SLEEP)
            except Exception as e:
                errors += 1
                print(f"  ERROR {book['title']}: {e}")

    print(f"\nDone → {args.output}")
    print(f"  Written : {written}")
    print(f"  Skipped : {skipped}")
    print(f"  Errors  : {errors}")


if __name__ == "__main__":
    main()
