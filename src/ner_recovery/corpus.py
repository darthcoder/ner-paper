"""Stream a full Wikipedia XML dump and filter to token-count-bounded articles.

Implements the loader described in featherweight-corpus-load.md. That spec calls
for the `mediawiki` package, but `mediawiki` (pymediawiki) is an online API client
with no dump-reading support at all — `mediawiki.MediaWikiDump()` does not exist.
`mwxml` is the actual purpose-built library for streaming MediaWiki XML dumps
(used by Wikimedia research tooling), so this implementation uses that instead.

Output preserves this project's raw-corpus schema ({pageid, title, wikitext}) so
scripts/janitor.py can consume it directly, same as the other fetch_*.py scripts.
"""
from __future__ import annotations

import argparse
import bz2
import gzip
import json
import re
from pathlib import Path
from typing import Iterator, TextIO

import mwxml

ARTICLE_NAMESPACE = 0
MIN_TOKENS = 500
MAX_TOKENS = 5000


def _open_dump(dump_path: Path) -> TextIO:
    """Open a Wikipedia XML dump, auto-detecting .bz2 / .gz / plain text."""
    if dump_path.suffix == ".bz2":
        return bz2.open(dump_path, "rt", encoding="utf-8")
    if dump_path.suffix == ".gz":
        return gzip.open(dump_path, "rt", encoding="utf-8")
    return dump_path.open("rt", encoding="utf-8")


def load_wikipedia_dump(dump_path: Path) -> Iterator[mwxml.Page]:
    """Open a dump and yield article (namespace 0) pages, one at a time."""
    with _open_dump(dump_path) as f:
        dump = mwxml.Dump.from_file(f)
        for page in dump:
            if page.namespace != ARTICLE_NAMESPACE:
                continue
            yield page


def extract_page_content(page: mwxml.Page) -> tuple[int, str, str] | None:
    """From a page object, extract (pageid, title, wikitext) of its latest revision."""
    revision = next(iter(page), None)
    if revision is None or not revision.text:
        return None
    return page.id, page.title, revision.text


def create_doc_id(title: str) -> str:
    """Convert an article title to a stable, filesystem-safe identifier."""
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    return f"wikipedia_{slug}"


def count_tokens_approximate(raw_text: str) -> int:
    """Whitespace-split token count — O(1), deterministic, no NLP model."""
    return len(raw_text.split())


def should_keep_article(raw_text: str, min_tokens: int = MIN_TOKENS, max_tokens: int = MAX_TOKENS) -> bool:
    token_count = count_tokens_approximate(raw_text)
    return min_tokens <= token_count <= max_tokens


def corpus_loader(
    dump_path: Path, min_tokens: int = MIN_TOKENS, max_tokens: int = MAX_TOKENS
) -> Iterator[tuple[int, str, str, str]]:
    """Main generator: open dump, extract, filter, yield (pageid, doc_id, title, wikitext)."""
    for page in load_wikipedia_dump(dump_path):
        extracted = extract_page_content(page)
        if extracted is None:
            continue
        pageid, title, wikitext = extracted

        if not should_keep_article(wikitext, min_tokens, max_tokens):
            continue

        yield pageid, create_doc_id(title), title, wikitext


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream a Wikipedia XML dump to JSONL, filtered to 500-5000 token articles"
    )
    parser.add_argument("--dump", type=Path, required=True,
                        help="Path to enwiki-*-pages-articles*.xml[.bz2|.gz]")
    parser.add_argument("--output", type=Path, default=Path("data/wikipedia_corpus.jsonl"))
    parser.add_argument("--min-tokens", type=int, default=MIN_TOKENS)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with args.output.open("w") as fout:
        for pageid, doc_id, title, wikitext in corpus_loader(args.dump, args.min_tokens, args.max_tokens):
            fout.write(json.dumps({
                "pageid": pageid,
                "doc_id": doc_id,
                "title": title,
                "wikitext": wikitext,
            }) + "\n")
            written += 1
            if written % 1000 == 0:
                print(f"  {written} articles written...")

    print(f"Done → {args.output} ({written} articles, {args.min_tokens}-{args.max_tokens} tokens)")


if __name__ == "__main__":
    main()
