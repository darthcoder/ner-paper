import json
from pathlib import Path

import mwparserfromhell

IN = Path("wwi_corpus.jsonl")
OUT = Path("data/wwi_clean.jsonl")


def clean(wikitext: str) -> str:
    parsed = mwparserfromhell.parse(wikitext)
    return parsed.strip_code(
        normalize=True,
        collapse=True,
        keep_template_params=False,
    ).strip()


NON_ARTICLE_PREFIXES = (
    "Category:",
    "Template:",
    "Wikipedia:",
    "Help:",
    "Portal:",
    "File:",
    "Talk:",
)

OUT.parent.mkdir(parents=True, exist_ok=True)

skipped = 0
written = 0

with IN.open() as fin, OUT.open("w") as fout:
    for line in fin:
        record = json.loads(line)
        title = record["title"]
        if title.startswith(NON_ARTICLE_PREFIXES):
            skipped += 1
            continue
        text = clean(record.get("wikitext", ""))
        if not text or len(text) < 100:
            skipped += 1
            continue
        fout.write(json.dumps({"pageid": record["pageid"], "title": title, "text": text}) + "\n")
        written += 1

print(f"Written {written} articles, skipped {skipped} non-articles/empty.")
