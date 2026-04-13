"""Fetch Wikipedia articles from a category into a JSONL corpus file."""
import argparse
import json
import time
from pathlib import Path

import requests

HEADERS = {"User-Agent": "ner-recovery/0.1 (research project)"}


def get_members(category: str, limit: int = 500) -> list[str]:
    r = requests.get("https://en.wikipedia.org/w/api.php", params={
        "action": "query", "list": "categorymembers",
        "cmtitle": category, "cmlimit": limit, "format": "json"
    }, headers=HEADERS)
    return [p["title"] for p in r.json()["query"]["categorymembers"]]


def fetch_article(title: str) -> dict:
    r = requests.get("https://en.wikipedia.org/w/api.php", params={
        "action": "query", "titles": title, "prop": "revisions",
        "rvprop": "content", "rvslots": "main", "format": "json"
    }, headers=HEADERS)
    pages = r.json()["query"]["pages"]
    page = next(iter(pages.values()))
    text = page.get("revisions", [{}])[0].get("slots", {}).get("main", {}).get("*", "")
    return {"pageid": page["pageid"], "title": page["title"], "wikitext": text}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Wikipedia category articles to JSONL")
    parser.add_argument("--category", required=True,
                        help="Wikipedia category title, e.g. 'Category:World_War_I'")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output JSONL file path")
    parser.add_argument("--limit", type=int, default=500,
                        help="Max articles to fetch (default 500)")
    args = parser.parse_args()

    titles = get_members(args.category, args.limit)
    print(f"Found {len(titles)} articles in {args.category}")

    with args.output.open("w") as f:
        for i, title in enumerate(titles):
            try:
                article = fetch_article(title)
                f.write(json.dumps(article) + "\n")
                if i % 50 == 0:
                    print(f"  {i}/{len(titles)}")
                time.sleep(0.1)
            except Exception as e:
                print(f"  SKIP {title}: {e}")

    print(f"Done → {args.output}")


if __name__ == "__main__":
    main()
