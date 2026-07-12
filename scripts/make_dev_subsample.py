"""Materialize a fixed, seeded subsample of a redacted jsonl for fast dev iteration.

Dev runs should use the materialized file (e.g. data/train_redacted_dev.jsonl)
rather than re-sampling on every invocation, so tokenization caching (see
train.py) stays valid across runs. Metrics from dev-subsample runs are
directional only — never report them as paper numbers.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    total = kept = 0
    with open(args.input) as f_in, open(args.output, "w") as f_out:
        for line in f_in:
            total += 1
            if rng.random() < args.fraction:
                f_out.write(line)
                kept += 1

    print(f"Wrote {kept}/{total} articles ({args.fraction:.0%} target) → {args.output}")


if __name__ == "__main__":
    main()
