"""Check Wikipedia category sizes (pages + subcats) without fetching articles."""
import json
import sys
import time

import requests

HEADERS = {"User-Agent": "ner-recovery/0.1 (research; scope-check)"}
API = "https://en.wikipedia.org/w/api.php"

CATEGORIES = [
    "Category:American Revolutionary War",
    "Category:American Revolution",
    "Category:Continental Army",
    "Category:Founding Fathers of the United States",
    "Category:American Revolutionary War battles",
    "Category:Monroe Doctrine",
    "Category:Manifest Destiny",
    "Category:Mexican-American War",
    "Category:Spanish-American War",
]


def category_info(titles: list[str]) -> dict:
    r = requests.get(API, headers=HEADERS, params={
        "action": "query",
        "prop": "categoryinfo",
        "titles": "|".join(titles),
        "format": "json",
    }, timeout=30)
    r.raise_for_status()
    return r.json()["query"]["pages"]


def main():
    print(f"{'Category':<50} {'pages':>7} {'subcats':>8} {'total':>7}")
    print("-" * 76)

    # batch in groups of 5 to stay polite
    batch_size = 5
    for i in range(0, len(CATEGORIES), batch_size):
        batch = CATEGORIES[i:i + batch_size]
        try:
            pages = category_info(batch)
            for page in pages.values():
                title = page.get("title", "?")
                info = page.get("categoryinfo", {})
                npages = info.get("pages", 0)
                nsubcats = info.get("subcats", 0)
                print(f"{title:<50} {npages:>7,} {nsubcats:>8,} {npages+nsubcats:>7,}")
            time.sleep(1.0)
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            time.sleep(5.0)


if __name__ == "__main__":
    main()
