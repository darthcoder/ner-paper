"""Fetch category membership for all 75 WWI categories (no article text).

Maps each article → which categories it appears in. Outputs:
  - data/wwi_articles_by_category.jsonl: {pageid, title, categories, category_count}
  - data/wwi_category_overlap_stats.json: overlap distribution, contested nodes

This reconnaissance pass shows article/category density *before* committing 
to 4.5 GB text download. High-overlap nodes are the contested actors where 
Wikipedia's narrative frames diverge most.

Usage:
    uv run python scripts/fetch_category_map.py \
        --output data/wwi_articles_by_category.jsonl \
        --stats data/wwi_category_overlap_stats.json \
        --sleep 1.0

Output schema:
    {
      "pageid": 12345,
      "title": "Friedrich Engels",
      "categories": ["Category:German communist writers", "Category:Marxist theorists", ...],
      "category_count": 12
    }
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import requests

HEADERS = {"User-Agent": "ner-recovery/0.3 (research; github.com/ner-paper)"}
API = "https://en.wikipedia.org/w/api.php"
MAX_RETRIES = 6

CATEGORIES = [
    # Baseline (34)
    "Category:Assassination of Archduke Franz Ferdinand",
    "Category:July Crisis",
    "Category:Black Hand (Serbia)",
    "Category:Balkan Wars",
    "Category:Pan-Slavism",
    "Category:Serbian nationalism",
    "Category:Politicians of the German Empire",
    "Category:Austria-Hungary politicians",
    "Category:Young Turks",
    "Category:Committee of Union and Progress",
    "Category:Bulgarian World War I military personnel",
    "Category:Russian Empire politicians",
    "Category:French Third Republic politicians",
    "Category:British politicians 1910–1945",
    "Category:American World War I political figures",
    "Category:Wilson administration",
    "Category:Australian World War I military personnel",
    "Category:Canadian World War I military personnel",
    "Category:New Zealand World War I military personnel",
    "Category:South African World War I military personnel",
    "Category:Arab Revolt",
    "Category:Gallipoli campaign",
    "Category:Ottoman Empire in World War I",
    "Category:Middle East World War I",
    "Category:Turkish World War I",
    "Category:Battle of the Somme",
    "Category:Paris Peace Conference, 1919",
    "Category:Treaty of Versailles",
    "Category:Partitioning of the Ottoman Empire",
    "Category:League of Nations",
    "Category:World War I",
    "Category:Western Front",
    "Category:Eastern Front (World War I)",
    "Category:Home front World War I",
    # High-signal additions (24)
    "Category:Bolshevism",
    "Category:Spartacist uprising",
    "Category:Irish independence movement",
    "Category:Anarchism and World War I",
    "Category:Syndicalism",
    "Category:Krupp family",
    "Category:Munitions manufacturing World War I",
    "Category:War profiteering",
    "Category:Submarine warfare",
    "Category:World War I espionage",
    "Category:Code-breaking World War I",
    "Category:Mata Hari",
    "Category:China in World War I",
    "Category:Japan in World War I",
    "Category:African World War I campaigns",
    "Category:Mesopotamian campaign",
    "Category:East African campaign (World War I)",
    "Category:Chemical weapons World War I",
    "Category:Aviation World War I",
    "Category:Tank development World War I",
    "Category:Submarine technology",
    "Category:Women World War I",
    "Category:Nurses World War I",
    "Category:Suffragism during World War I",
    # India (5)
    "Category:India in World War I",
    "Category:Indian Army during World War I",
    "Category:Indian independence movement",
    "Category:Indian soldiers World War I",
    "Category:British Raj",
    # Communist & Socialist Theory & Movement (12)
    "Category:Communism",
    "Category:Communist movement",
    "Category:Marxism",
    "Category:Karl Marx",
    "Category:Socialist International",
    "Category:First International",
    "Category:Labor movement World War I",
    "Category:Trade unions World War I",
    "Category:Anarcho-communism",
    "Category:Bolshevik Revolution",
    "Category:Lenin",
    "Category:Rosa Luxemburg",
    "Category:Frederich Engels",
    # Extended — from fetch_wwi_extended.sh (12)
    "Category:Aftermath of World War I",
    "Category:Conservative Party (UK) politicians 1910–1945",
    "Category:February Revolution",
    "Category:Liberal Party (UK) politicians",
    "Category:October Revolution",
    "Category:Russian Revolution",
    "Category:Siege of Tsingtao",
    "Category:United States in World War I",
    "Category:West African campaign (World War I)",
    "Category:World War I commanders",
    "Category:World War I military personnel by country",
    "Category:Young Bosnia",
]


def api_get(params: dict) -> dict:
    """Fetch from MediaWiki API with retries."""
    for attempt in range(MAX_RETRIES):
        r = requests.get(API, params=params, headers=HEADERS, timeout=30)
        if r.status_code == 429:
            wait = 2 ** (attempt + 2)
            print(f"    429 rate limit — waiting {wait}s")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries")


def get_category_members(category: str) -> list[dict]:
    """Fetch all members of a category (pages only, not subcats)."""
    members: list[dict] = []
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": category,
        "cmtype": "page",
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
    return members


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch category membership mapping (no text) for 75 WWI categories"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/wwi_articles_by_category.jsonl"),
        help="Output JSONL: pageid, title, categories, category_count",
    )
    parser.add_argument(
        "--stats",
        type=Path,
        default=Path("data/wwi_category_overlap_stats.json"),
        help="Output JSON: overlap distribution and stats",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Sleep between category fetches (seconds)",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.stats.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"fetch_category_map.py started")
    print(f"Output: {args.output}")
    print(f"Stats:  {args.stats}")
    print(f"Categories: {len(CATEGORIES)}")
    print("=" * 70)

    # Aggregate: pageid → {title, set of categories}
    articles: dict[int, dict] = {}

    for i, category in enumerate(CATEGORIES, start=1):
        print(f"\n[{i:2d}/{len(CATEGORIES)}] {category}")
        try:
            members = get_category_members(category)
            print(f"    → {len(members)} members")

            for member in members:
                pageid = member["pageid"]
                title = member["title"]

                if pageid not in articles:
                    articles[pageid] = {"title": title, "categories": set()}

                articles[pageid]["categories"].add(category)
        except Exception as e:
            print(f"    ERROR: {e}")

        time.sleep(args.sleep)

    # Write output JSONL
    print(f"\nWriting {len(articles)} unique articles to {args.output}")
    with args.output.open("w") as f:
        for pageid, data in sorted(articles.items()):
            record = {
                "pageid": pageid,
                "title": data["title"],
                "categories": sorted(data["categories"]),
                "category_count": len(data["categories"]),
            }
            f.write(json.dumps(record) + "\n")

    # Compute overlap statistics
    overlap_dist: dict[int, int] = defaultdict(int)
    high_overlap_nodes: list[tuple[int, str, int]] = []

    for pageid, data in articles.items():
        count = len(data["categories"])
        overlap_dist[count] += 1
        if count >= 5:  # Contested actors: appear in 5+ categories
            high_overlap_nodes.append((pageid, data["title"], count))

    # Sort high-overlap nodes by count descending
    high_overlap_nodes.sort(key=lambda x: x[2], reverse=True)

    # Write stats
    stats = {
        "total_unique_articles": len(articles),
        "total_categories": len(CATEGORIES),
        "overlap_distribution": dict(sorted(overlap_dist.items())),
        "high_overlap_nodes": [
            {"pageid": pid, "title": title, "category_count": count}
            for pid, title, count in high_overlap_nodes[:50]  # Top 50
        ],
        "high_overlap_summary": {
            "articles_in_5plus_categories": len(
                [x for x in high_overlap_nodes if x[2] >= 5]
            ),
            "articles_in_10plus_categories": len(
                [x for x in high_overlap_nodes if x[2] >= 10]
            ),
            "articles_in_15plus_categories": len(
                [x for x in high_overlap_nodes if x[2] >= 15]
            ),
        },
    }

    with args.stats.open("w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nStats written to {args.stats}")
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Total unique articles: {stats['total_unique_articles']}")
    print(
        f"Articles in 5+ categories: {stats['high_overlap_summary']['articles_in_5plus_categories']}"
    )
    print(
        f"Articles in 10+ categories: {stats['high_overlap_summary']['articles_in_10plus_categories']}"
    )
    print(
        f"Articles in 15+ categories: {stats['high_overlap_summary']['articles_in_15plus_categories']}"
    )
    print("\nOverlap distribution:")
    for count in sorted(stats["overlap_distribution"].keys()):
        freq = stats["overlap_distribution"][count]
        print(f"  {count:2d} categories: {freq:5d} articles")
    print("\nTop 10 contested nodes (most category overlap):")
    for pid, title, count in high_overlap_nodes[:10]:
        print(f"  [{count:2d}] {title}")


if __name__ == "__main__":
    main()
