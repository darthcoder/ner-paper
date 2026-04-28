"""Benchmark multiple Claude models simultaneously on the NEI test set.

Submits one Message Batch per model in parallel, then polls until all complete.
Prints a side-by-side comparison table and saves per-model JSON results to evals/.

Usage:
    uv run python scripts/benchmark.py
    uv run python scripts/benchmark.py --models claude-haiku-4-5-20251001 claude-sonnet-4-6
    uv run python scripts/benchmark.py --data data/test_zero_oov.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

DEFAULT_MODELS = [
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-6",
]
DEFAULT_DATA = Path("data/test_zero_oov.jsonl")
POLL_INTERVAL = 60  # seconds

SYSTEM_PROMPT = """\
You are given a passage from a historical encyclopaedia. Some named entities have \
been replaced with tokens of the form [REDACTED:LABEL], where LABEL is the entity \
type (e.g. PERSON, GPE, ORG, LOC, EVENT, NORP, FAC).

Your task: for each [REDACTED:LABEL] token, in the order it appears in the passage, \
predict the original named entity that was removed.

Rules:
- Return ONLY a JSON object: {"predictions": ["entity1", "entity2", ...]}
- One prediction per [REDACTED] token, in document order.
- Each prediction must be a short noun phrase — the entity name only, no explanation.
- If you are uncertain, give your best guess; never omit an entry.\
"""

LABELS = ["EVENT", "FAC", "GPE", "LOC", "NORP", "ORG", "PERSON"]


# ---------------------------------------------------------------------------
# Helpers (shared with eval_claude.py)
# ---------------------------------------------------------------------------

def count_redactions(text: str) -> int:
    return len(re.findall(r"\[REDACTED:[A-Z]+\]", text))


def parse_predictions(raw: str, n_expected: int) -> list[str]:
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        preds = json.loads(cleaned).get("predictions", [])
        if len(preds) < n_expected:
            preds.extend([""] * (n_expected - len(preds)))
        return [str(p).strip() for p in preds[:n_expected]]
    except (json.JSONDecodeError, AttributeError):
        pass
    match = re.search(r"\{[^{}]*\"predictions\"[^{}]*\[.*?\][^{}]*\}", raw, re.DOTALL)
    if match:
        try:
            preds = json.loads(match.group()).get("predictions", [])
            if len(preds) < n_expected:
                preds.extend([""] * (n_expected - len(preds)))
            return [str(p).strip() for p in preds[:n_expected]]
        except (json.JSONDecodeError, AttributeError):
            pass
    return [""] * n_expected


def load_records(data_path: Path) -> list[dict]:
    records = []
    with open(data_path) as f:
        for line in f:
            r = json.loads(line)
            redactions = sorted(r.get("redactions", []), key=lambda x: x["start"])
            if not redactions:
                continue
            r["redactions"] = redactions
            positions = [m.start() for m in re.finditer(r"\[REDACTED:[A-Z]+\]", r["text"])]
            pos_to_idx = {pos: i for i, pos in enumerate(positions)}
            r["_pred_indices"] = [
                pos_to_idx[red["start"]] for red in redactions if red["start"] in pos_to_idx
            ]
            r["_n_tokens"] = len(positions)
            records.append(r)
    return records


def score(
    records: list[dict],
    predictions_by_id: dict[str, list[str]],
) -> tuple[int, int, dict[str, dict]]:
    total = correct = 0
    by_label: dict[str, list[bool]] = defaultdict(list)
    for record in records:
        preds = predictions_by_id.get(str(record["pageid"]), [])
        pred_indices = record.get("_pred_indices", list(range(len(record["redactions"]))))
        for r, idx in zip(record["redactions"], pred_indices):
            pred = preds[idx] if idx < len(preds) else ""
            total += 1
            match = pred.lower() == r["original"].strip().lower()
            if match:
                correct += 1
            by_label[r["label"]].append(match)
    by_label_summary = {
        label: {"correct": sum(res), "total": len(res)}
        for label, res in sorted(by_label.items())
    }
    return total, correct, by_label_summary


# ---------------------------------------------------------------------------
# Batch lifecycle
# ---------------------------------------------------------------------------

def submit_batch(
    client: anthropic.Anthropic,
    records: list[dict],
    model: str,
) -> str:
    requests = [
        Request(
            custom_id=str(r["pageid"]),
            params=MessageCreateParamsNonStreaming(
                model=model,
                max_tokens=max(512, r["_n_tokens"] * 20),
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": r["text"]}],
            ),
        )
        for r in records
    ]
    batch = client.messages.batches.create(requests=requests)
    return batch.id


def collect_batch(
    client: anthropic.Anthropic,
    batch_id: str,
    records: list[dict],
) -> dict[str, list[str]]:
    predictions: dict[str, list[str]] = {}
    record_by_id = {str(r["pageid"]): r for r in records}
    for result in client.messages.batches.results(batch_id):
        pageid = result.custom_id
        record = record_by_id.get(pageid)
        if record is None:
            continue
        n_tokens = record["_n_tokens"]
        if result.result.type == "succeeded":
            raw = next(
                (b.text for b in result.result.message.content if b.type == "text"), ""
            )
            predictions[pageid] = parse_predictions(raw, n_tokens)
        else:
            predictions[pageid] = [""] * n_tokens
    return predictions


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def save_result(
    model: str,
    data_path: Path,
    total: int,
    correct: int,
    by_label: dict[str, dict],
    timestamp: str,
) -> Path:
    accuracy = correct / total if total else 0.0
    model_slug = model.replace("-", "_")
    json_path = Path("evals") / f"benchmark_{model_slug}_{timestamp}.json"
    json_path.parent.mkdir(exist_ok=True)
    result = {
        "model": model,
        "mode": "claude-batch",
        "data": str(data_path),
        "timestamp": datetime.now().isoformat(),
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "by_label": by_label,
    }
    with json_path.open("w") as f:
        json.dump(result, f, indent=2)
    return json_path


def print_table(results: list[dict]) -> None:
    """Print a comparison table across all benchmarked models."""
    all_labels = sorted({lbl for r in results for lbl in r["by_label"]})

    # Header
    col_w = 10
    model_w = 36
    sep = "-" * (model_w + 1 + col_w + 1 + len(all_labels) * (col_w + 1))
    print(f"\n{'='*len(sep)}")
    print("BENCHMARK COMPARISON TABLE")
    print(f"{'='*len(sep)}")
    header = f"{'Model':<{model_w}} {'Overall':>{col_w}}"
    for lbl in all_labels:
        header += f" {lbl:>{col_w}}"
    print(header)
    print(sep)

    for r in sorted(results, key=lambda x: -x["accuracy"]):
        name = r["model"][:model_w]
        row = f"{name:<{model_w}} {r['accuracy']:>{col_w}.1%}"
        for lbl in all_labels:
            stat = r["by_label"].get(lbl)
            if stat and stat["total"] > 0:
                acc = stat["correct"] / stat["total"]
                row += f" {acc:>{col_w}.1%}"
            else:
                row += f" {'—':>{col_w}}"
        print(row)
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel Claude benchmark — Engrammatic NEI")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS,
        help="Claude model IDs to benchmark (space-separated)",
    )
    args = parser.parse_args()

    client = anthropic.Anthropic()
    records = load_records(args.data)
    print(f"Loaded {len(records)} articles from {args.data}")
    print(f"Models: {', '.join(args.models)}\n")

    # Submit all batches simultaneously
    batch_ids: dict[str, str] = {}
    for model in args.models:
        bid = submit_batch(client, records, model)
        batch_ids[model] = bid
        print(f"  {model}: batch {bid} submitted")

    # Poll until all complete
    print("\nWaiting for batches to complete...")
    pending = set(args.models)
    while pending:
        time.sleep(POLL_INTERVAL)
        still_pending = set()
        for model in pending:
            batch = client.messages.batches.retrieve(batch_ids[model])
            if batch.processing_status == "ended":
                print(f"  ✓ {model} done")
            else:
                counts = batch.request_counts
                print(
                    f"  … {model}: processing={counts.processing} "
                    f"succeeded={counts.succeeded}"
                )
                still_pending.add(model)
        pending = still_pending

    # Collect and score
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []
    for model in args.models:
        preds = collect_batch(client, batch_ids[model], records)
        total, correct, by_label = score(records, preds)
        json_path = save_result(model, args.data, total, correct, by_label, timestamp)
        accuracy = correct / total if total else 0.0
        all_results.append({
            "model": model,
            "accuracy": accuracy,
            "total": total,
            "correct": correct,
            "by_label": by_label,
        })
        print(f"  {model}: {accuracy:.1%}  → {json_path}")

    print_table(all_results)


if __name__ == "__main__":
    main()
