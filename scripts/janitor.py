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


OUT.parent.mkdir(parents=True, exist_ok=True)

skipped = 0
written = 0

with IN.open() as fin, OUT.open("w") as fout:
    for line in fin:
        record = json.loads(line)
        text = clean(record.get("wikitext", ""))
        if not text:
            skipped += 1
            continue
        fout.write(json.dumps({"pageid": record["pageid"], "title": record["title"], "text": text}) + "\n")
        written += 1

print(f"Written {written} articles, skipped {skipped} empty.")
