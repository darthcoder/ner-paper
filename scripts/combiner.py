"""Combine all data/*_clean.jsonl files into data/all_clean.jsonl.

Deduplicates by pageid — first occurrence wins.
"""
import json
from pathlib import Path

DATA = Path("data")
OUT = DATA / "all_clean.jsonl"

sources = sorted(DATA.glob("*_clean.jsonl"))
# Exclude the output file itself and the split files
sources = [s for s in sources if s.name not in ("all_clean.jsonl", "train_clean.jsonl", "test_clean.jsonl")]

seen: set[int] = set()
written = 0
skipped_dupes = 0

with OUT.open("w") as fout:
    for src in sources:
        src_written = 0
        with src.open() as fin:
            for line in fin:
                record = json.loads(line)
                pid = record["pageid"]
                if pid in seen:
                    skipped_dupes += 1
                    continue
                seen.add(pid)
                fout.write(json.dumps(record) + "\n")
                written += 1
                src_written += 1
        print(f"  {src.name}: {src_written} articles")

print(f"Combined {written} unique articles ({skipped_dupes} dupes removed) → {OUT}")
