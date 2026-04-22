"""Fetch Wikipedia articles from a category (recursively) into a JSONL corpus file."""
import argparse
import json
import time
from collections import deque
from pathlib import Path

import requests

HEADERS = {"User-Agent": "ner-recovery/0.1 (research; https://github.com/darthcoder/ner-paper)"}
API = "https://en.wikipedia.org/w/api.php"
MAX_RETRIES = 3


def _api_get(params: dict) -> dict:
    """GET from MediaWiki API with exponential-backoff retry on errors."""
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(API, params=params, headers=HEADERS, timeout=30)
            if r.status_code == 429:
                wait = 2 ** (attempt + 2)
                print(f"    rate limited — waiting {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            if not r.text.strip():
                wait = 2 ** (attempt + 1)
                print(f"    empty body (attempt {attempt + 1}, status {r.status_code}) — waiting {wait}s")
                time.sleep(wait)
                continue
            return r.json()
        except (requests.exceptions.JSONDecodeError, ValueError) as e:
            wait = 2 ** (attempt + 1)
            print(f"    bad response (attempt {attempt + 1}): {e} — waiting {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"API failed after {MAX_RETRIES} retries: {params}")


def _category_members(category: str, cmtype: str) -> list[str]:
    """Return direct members of *category* filtered by cmtype ('page' or 'subcat')."""
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": category,
        "cmtype": cmtype,
        "cmlimit": 500,
        "format": "json",
    }
    results = []
    while True:
        data = _api_get(params)
        results.extend(p["title"] for p in data["query"]["categorymembers"])
        cont = data.get("continue", {}).get("cmcontinue")
        if not cont:
            break
        params["cmcontinue"] = cont
        time.sleep(0.1)
    return results


def get_all_members(
    root_category: str, max_depth: int = 5, seen_pageids: set[int] | None = None
) -> list[str]:
    """BFS over the category tree; return deduplicated article titles."""
    seen_pageids = seen_pageids or set()
    seen_cats: set[str] = set()
    article_titles: list[str] = []

    queue: deque[tuple[str, int]] = deque([(root_category, 0)])

    while queue:
        cat, depth = queue.popleft()
        if cat in seen_cats:
            continue
        seen_cats.add(cat)
        print(f"  {'  ' * depth}↳ {cat}")

        for title in _category_members(cat, "page"):
            if title not in {t for t in article_titles}:  # cheap title dedup
                article_titles.append(title)
            time.sleep(0.05)

        if depth < max_depth:
            for subcat in _category_members(cat, "subcat"):
                if subcat not in seen_cats:
                    queue.append((subcat, depth + 1))
            time.sleep(0.05)

    return article_titles


def fetch_article(title: str) -> dict:
    pages = _api_get({
        "action": "query",
        "titles": title,
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "format": "json",
    })["query"]["pages"]
    page = next(iter(pages.values()))
    text = (
        page.get("revisions", [{}])[0]
        .get("slots", {})
        .get("main", {})
        .get("*", "")
    )
    return {"pageid": page["pageid"], "title": page["title"], "wikitext": text}


def load_existing_titles(path: Path) -> set[str]:
    """Return the set of titles already present in an existing JSONL file."""
    if not path.exists():
        return set()
    titles = set()
    with path.open() as f:
        for line in f:
            try:
                titles.add(json.loads(line)["title"])
            except Exception:
                pass
    return titles


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch Wikipedia category articles (recursively) to JSONL"
    )
    parser.add_argument(
        "--category",
        required=True,
        help="Wikipedia category title, e.g. 'Category:World_War_I'",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL file path")
    parser.add_argument(
        "--max-depth",
        type=int,
        default=5,
        help="Max subcategory recursion depth (default 5)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing output file, skipping already-fetched titles",
    )
    parser.add_argument(
        "--user-agent",
        default=None,
        help="Override the User-Agent header for this run",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.1,
        help="Sleep between article fetches in seconds (default 0.1)",
    )
    args = parser.parse_args()

    if args.user_agent:
        HEADERS["User-Agent"] = args.user_agent

    existing_titles = load_existing_titles(args.output) if args.append else set()
    if existing_titles:
        print(f"Skipping {len(existing_titles)} already-fetched titles")

    print(f"Walking category tree from: {args.category}")
    titles = get_all_members(args.category, max_depth=args.max_depth)
    titles = [t for t in titles if t not in existing_titles]
    print(f"\nFound {len(titles)} new articles to fetch")

    mode = "a" if args.append else "w"
    with args.output.open(mode) as f:
        for i, title in enumerate(titles):
            try:
                article = fetch_article(title)
                f.write(json.dumps(article) + "\n")
                print(f"  [{i+1}/{len(titles)}] {title}", flush=True)
                time.sleep(args.sleep)
            except Exception as e:
                print(f"  SKIP {title}: {e}")

    print(f"Done → {args.output}")


if __name__ == "__main__":
    main()
