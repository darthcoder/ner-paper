"""Print a comparison table from all saved JSON eval results in evals/.

Reads every *.json file in evals/ and prints a ranked comparison table.
Includes both DistilBERT results (from uv run evaluate) and Claude results
(from eval_claude.py and benchmark.py).

Usage:
    uv run python scripts/benchmark_table.py
    uv run python scripts/benchmark_table.py --latest   # one result per model
    uv run python scripts/benchmark_table.py --evals-dir evals/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_EVALS_DIR = Path("evals")
LABELS = ["EVENT", "FAC", "GPE", "LOC", "NORP", "ORG", "PERSON"]

# Display name overrides for known model IDs
MODEL_NAMES = {
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    "claude-haiku-4-5": "Claude Haiku 4.5",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "claude-opus-4-6": "Claude Opus 4.6",
    "claude-opus-4-7": "Claude Opus 4.7",
}


def display_name(model: str) -> str:
    if model in MODEL_NAMES:
        return MODEL_NAMES[model]
    # DistilBERT: shorten path to just the last two parts
    p = Path(model)
    if len(p.parts) >= 2:
        return f"DistilBERT ({p.parts[-2]}/{p.parts[-1]})"
    return model


def load_results(evals_dir: Path, latest_only: bool) -> list[dict]:
    results = []
    for path in sorted(evals_dir.glob("*.json")):
        try:
            with path.open() as f:
                r = json.load(f)
            if "model" not in r or "accuracy" not in r or "by_label" not in r:
                continue
            r["_path"] = path
            r["_timestamp"] = r.get("timestamp", path.stem)
            results.append(r)
        except (json.JSONDecodeError, KeyError):
            continue

    if latest_only:
        # Keep the most recent result per (model, mode) pair
        seen: dict[tuple, dict] = {}
        for r in sorted(results, key=lambda x: x["_timestamp"]):
            key = (r["model"], r.get("mode", ""))
            seen[key] = r
        results = list(seen.values())

    return results


def print_table(results: list[dict]) -> None:
    if not results:
        print("No JSON results found in evals/. Run an evaluation first.")
        return

    all_labels = sorted({lbl for r in results for lbl in r["by_label"]})
    col_w = 10
    model_w = 34
    mode_w = 22
    n_cols = 2 + len(all_labels)
    sep = "-" * (model_w + mode_w + n_cols * (col_w + 1) + 1)

    print(f"\n{'='*len(sep)}")
    print("ENGRAMMATIC NEI — BENCHMARK COMPARISON")
    print(f"{'='*len(sep)}")
    header = f"{'Model':<{model_w}} {'Mode':<{mode_w}} {'Overall':>{col_w}}"
    for lbl in all_labels:
        header += f" {lbl:>{col_w}}"
    print(header)
    print(sep)

    for r in sorted(results, key=lambda x: -x["accuracy"]):
        name = display_name(r["model"])[:model_w]
        mode = r.get("mode", "")[:mode_w]
        row = f"{name:<{model_w}} {mode:<{mode_w}} {r['accuracy']:>{col_w}.1%}"
        for lbl in all_labels:
            stat = r["by_label"].get(lbl)
            if stat and stat["total"] > 0:
                acc = stat["correct"] / stat["total"]
                row += f" {acc:>{col_w}.1%}"
            else:
                row += f" {'—':>{col_w}}"
        print(row)

    print(sep)
    print(f"\n{len(results)} result(s) shown. JSON files in {DEFAULT_EVALS_DIR}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Print NEI benchmark comparison table")
    parser.add_argument("--evals-dir", type=Path, default=DEFAULT_EVALS_DIR)
    parser.add_argument(
        "--latest", action="store_true",
        help="Show only the most recent result per (model, mode) pair",
    )
    args = parser.parse_args()

    results = load_results(args.evals_dir, args.latest)
    print_table(results)


if __name__ == "__main__":
    main()
