import json
import random
from pathlib import Path

import spacy

IN = Path("data/wwi_clean.jsonl")
OUT = Path("data/wwi_redacted.jsonl")

MASK = "[REDACTED]"
MASK_LEN = len(MASK)

nlp = spacy.load("en_core_web_sm")
rng = random.Random(42)


def redact(text: str):
    doc = nlp(text)
    ents = list(doc.ents)
    if not ents:
        return text, []

    # 2–5% of entities, minimum 1
    k = max(1, round(len(ents) * rng.uniform(0.02, 0.05)))
    k = min(k, len(ents))
    chosen = sorted(rng.sample(ents, k), key=lambda e: e.start_char)

    parts = []
    redactions = []
    prev_end = 0
    offset = 0  # cumulative char-length delta from prior replacements

    for ent in chosen:
        parts.append(text[prev_end : ent.start_char])
        new_start = ent.start_char + offset
        parts.append(MASK)
        new_end = new_start + MASK_LEN
        offset += MASK_LEN - (ent.end_char - ent.start_char)
        redactions.append({
            "start": new_start,
            "end": new_end,
            "label": ent.label_,
            "original": ent.text,
        })
        prev_end = ent.end_char

    parts.append(text[prev_end:])
    return "".join(parts), redactions


OUT.parent.mkdir(parents=True, exist_ok=True)

written = 0
total_redactions = 0

with IN.open() as fin, OUT.open("w") as fout:
    for line in fin:
        record = json.loads(line)
        redacted_text, redactions = redact(record["text"])
        fout.write(json.dumps({
            "pageid": record["pageid"],
            "title": record["title"],
            "text": redacted_text,
            "redactions": redactions,
        }) + "\n")
        written += 1
        total_redactions += len(redactions)

print(f"Written {written} articles, {total_redactions} total redactions "
      f"({total_redactions / written:.1f} avg per article).")
