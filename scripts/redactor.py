"""Redact named entities from a clean JSONL corpus.

Modes:
  --mode train   10–20% of entities, error-driven label weights (default)
  --mode test    2–5%  of entities, uniform weights
"""
import argparse
import json
import random
from pathlib import Path

import spacy
from tqdm import tqdm

def mask(label: str) -> str:
    return f"[REDACTED:{label}]"

nlp = spacy.load("en_core_web_sm")

# Error-driven label weights: derived from fix1_eval (2026-04-12).
# Weight = 1 - accuracy, clipped to [0.05, 1.0].
_RAW_WEIGHTS: dict[str, float] = {
    "CARDINAL":    1 - 0.4954,
    "DATE":        1 - 0.5794,
    "EVENT":       1 - 0.4896,
    "FAC":         1 - 0.5263,
    "GPE":         1 - 0.9347,
    "LANGUAGE":    1 - 1.0000,
    "LAW":         1 - 0.7500,
    "LOC":         1 - 0.8621,
    "MONEY":       1 - 0.1111,
    "NORP":        1 - 0.9043,
    "ORDINAL":     1 - 0.9545,
    "ORG":         1 - 0.6132,
    "PERCENT":     1 - 0.3333,
    "PERSON":      1 - 0.5591,
    "PRODUCT":     1 - 0.6667,
    "QUANTITY":    1 - 0.1250,
    "TIME":        1 - 0.5000,
    "WORK_OF_ART": 1 - 0.2857,
}
LABEL_WEIGHTS: dict[str, float] = {
    k: max(0.05, min(1.0, v)) for k, v in _RAW_WEIGHTS.items()
}
_DEFAULT_WEIGHT = 0.5


def redact(text: str, mode: str, rng: random.Random) -> tuple[str, list[dict]]:
    doc = nlp(text)
    ents = list(doc.ents)
    if not ents:
        return text, []

    if mode == "train":
        weights = [LABEL_WEIGHTS.get(e.label_, _DEFAULT_WEIGHT) for e in ents]
        k = max(1, round(len(ents) * rng.uniform(0.10, 0.20)))
        k = min(k, len(ents))
        chosen = sorted(rng.choices(ents, weights=weights, k=k), key=lambda e: e.start_char)
        # Deduplicate (choices samples with replacement).
        seen: set[int] = set()
        unique: list = []
        for e in chosen:
            if e.start_char not in seen:
                seen.add(e.start_char)
                unique.append(e)
        chosen = unique
    else:
        k = max(1, round(len(ents) * rng.uniform(0.02, 0.05)))
        k = min(k, len(ents))
        chosen = sorted(rng.sample(ents, k), key=lambda e: e.start_char)

    parts: list[str] = []
    redactions: list[dict] = []
    prev_end = 0
    offset = 0

    for ent in chosen:
        parts.append(text[prev_end:ent.start_char])
        new_start = ent.start_char + offset
        m = mask(ent.label_)
        parts.append(m)
        new_end = new_start + len(m)
        offset += len(m) - (ent.end_char - ent.start_char)
        redactions.append({
            "start": new_start,
            "end": new_end,
            "label": ent.label_,
            "original": ent.text,
        })
        prev_end = ent.end_char

    parts.append(text[prev_end:])
    return "".join(parts), redactions


def main() -> None:
    parser = argparse.ArgumentParser(description="Redact named entities from a clean JSONL corpus")
    parser.add_argument("--input", type=Path, required=True, help="Input clean JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Output redacted JSONL")
    parser.add_argument("--mode", choices=["train", "test"], default="train",
                        help="train: weighted 10-20%%; test: uniform 2-5%%")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    total_redactions = 0

    n_lines = sum(1 for _ in args.input.open())

    with args.input.open() as fin, args.output.open("w") as fout:
        for line in tqdm(fin, total=n_lines, desc=f"[{args.mode}] Redacting"):
            record = json.loads(line)
            redacted_text, redactions = redact(record["text"], args.mode, rng)
            fout.write(json.dumps({
                "pageid": record["pageid"],
                "title": record["title"],
                "text": redacted_text,
                "redactions": redactions,
            }) + "\n")
            written += 1
            total_redactions += len(redactions)

    print(f"[{args.mode}] Written {written} articles, {total_redactions} redactions "
          f"({total_redactions / written:.1f} avg per article) → {args.output}")


if __name__ == "__main__":
    main()
