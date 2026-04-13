"""Strip wikitext markup from a raw corpus JSONL file."""
import argparse
import json
from pathlib import Path

import mwparserfromhell

NON_ARTICLE_PREFIXES = (
    "Category:",
    "Template:",
    "Wikipedia:",
    "Help:",
    "Portal:",
    "File:",
    "Talk:",
)


def clean(wikitext: str) -> str:
    parsed = mwparserfromhell.parse(wikitext)
    return parsed.strip_code(
        normalize=True,
        collapse=True,
        keep_template_params=False,
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Strip wikitext markup from a raw corpus JSONL")
    parser.add_argument("--input", type=Path, default=Path("wwi_corpus.jsonl"),
                        help="Input raw JSONL (default: wwi_corpus.jsonl)")
    parser.add_argument("--output", type=Path, default=Path("data/wwi_clean.jsonl"),
                        help="Output clean JSONL (default: data/wwi_clean.jsonl)")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    skipped = 0
    written = 0

    with args.input.open() as fin, args.output.open("w") as fout:
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
            fout.write(json.dumps({
                "pageid": record["pageid"],
                "title": title,
                "text": text,
            }) + "\n")
            written += 1

    print(f"Written {written} articles, skipped {skipped} non-articles/empty → {args.output}")


if __name__ == "__main__":
    main()
