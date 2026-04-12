import requests, json, time
from pathlib import Path

HEADERS = {"User-Agent": "ner-recovery/0.1 (research project)"}

def get_members(limit=500):
    r = requests.get("https://en.wikipedia.org/w/api.php", params={
        "action": "query", "list": "categorymembers",
        "cmtitle": "Category:World_War_I", "cmlimit": limit, "format": "json"
    }, headers=HEADERS)
    return [p["title"] for p in r.json()["query"]["categorymembers"]]

def fetch_article(title):
    r = requests.get("https://en.wikipedia.org/w/api.php", params={
        "action": "query", "titles": title, "prop": "revisions",
        "rvprop": "content", "rvslots": "main", "format": "json"
    }, headers=HEADERS)
    pages = r.json()["query"]["pages"]
    page = next(iter(pages.values()))
    text = page.get("revisions", [{}])[0].get("slots", {}).get("main", {}).get("*", "")
    return {"pageid": page["pageid"], "title": page["title"], "wikitext": text}

out = Path("wwi_corpus.jsonl")
titles = get_members()
print(f"Found {len(titles)} articles")

with out.open("w") as f:
    for i, title in enumerate(titles):
        try:
            article = fetch_article(title)
            f.write(json.dumps(article) + "\n")
            if i % 50 == 0: print(f"{i}/{len(titles)}")
            time.sleep(0.1)
        except Exception as e:
            print(f"SKIP {title}: {e}")

print("Done")