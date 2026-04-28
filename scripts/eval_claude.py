"""Topline evaluation: use a Claude model to recover redacted named entities.

Reads the same test file format as eval.py. For each article, sends the passage
(with [REDACTED:LABEL] tokens intact) to Claude and asks it to predict each
entity in order.

Default mode uses the Message Batches API — one batch submission, async
completion, 50% cost reduction, no per-minute rate-limit concerns.

Sequential mode (--no-batch) calls the API one article at a time, throttled to
~27 requests/minute to stay within the 30 RPM limit.

Usage:
    uv run python scripts/eval_claude.py                          # batch mode
    uv run python scripts/eval_claude.py --no-batch               # sequential
    uv run python scripts/eval_claude.py --model claude-sonnet-4-6
    uv run python scripts/eval_claude.py --data data/test_zero_oov.jsonl
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
from tqdm import tqdm

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

DEFAULT_DATA = Path("data/test_zero_oov.jsonl")
DEFAULT_MODEL = "claude-haiku-4-5-20251001"
# Sequential mode: 60s / 27 req = ~2.22s per request → ~27 RPM
SEQUENTIAL_DELAY = 60.0 / 27
MAX_RETRIES = 3
RETRY_DELAY = 15  # seconds, for sequential rate-limit back-off
BATCH_POLL_INTERVAL = 60  # seconds between batch status checks


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def count_redactions(text: str) -> int:
    return len(re.findall(r"\[REDACTED:[A-Z]+\]", text))


def build_message(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


def parse_predictions(raw: str, n_expected: int) -> list[str]:
    """Parse Claude's JSON response into a fixed-length prediction list."""
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
        preds = data.get("predictions", [])
    except (json.JSONDecodeError, AttributeError):
        preds = []
    if len(preds) < n_expected:
        preds.extend([""] * (n_expected - len(preds)))
    return [str(p).strip() for p in preds[:n_expected]]


def load_records(data_path: Path) -> list[dict]:
    with open(data_path) as f:
        records = []
        for line in f:
            r = json.loads(line)
            redactions = sorted(r.get("redactions", []), key=lambda x: x["start"])
            n = count_redactions(r["text"])
            if not redactions or n != len(redactions):
                continue
            r["redactions"] = redactions
            records.append(r)
    return records


def score_predictions(
    records: list[dict],
    predictions_by_id: dict[str, list[str]],
) -> tuple[int, int, dict[str, list[bool]]]:
    """Compare predictions to ground truth. Returns (total, correct, by_label)."""
    total = correct = 0
    by_label: dict[str, list[bool]] = defaultdict(list)
    for record in records:
        preds = predictions_by_id.get(str(record["pageid"]), [])
        for i, r in enumerate(record["redactions"]):
            pred = preds[i] if i < len(preds) else ""
            original = r["original"].strip()
            label = r["label"]
            total += 1
            match = pred.lower() == original.lower()
            if match:
                correct += 1
            by_label[label].append(match)
    return total, correct, by_label


def write_report(
    report_path: Path,
    model: str,
    data_path: Path,
    mode: str,
    total: int,
    correct: int,
    by_label: dict[str, list[bool]],
) -> None:
    accuracy = correct / total if total else 0.0
    report_path.parent.mkdir(exist_ok=True)
    with report_path.open("w") as f:
        f.write(f"{'='*80}\nENGRAMMATIC NAMED ENTITY INFERENCE — CLAUDE TOPLINE EVALUATION\n{'='*80}\n")
        f.write(f"Date/Time     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model         : {model}\n")
        f.write(f"Mode          : {mode}\n")
        f.write(f"Data          : {data_path}\n\n")
        f.write(f"{'='*80}\nRESULTS\n{'='*80}\n")
        f.write(f"  Total redactions : {total}\n")
        f.write(f"  Exact matches    : {correct}\n")
        f.write(f"  Accuracy         : {accuracy:.2%}\n")
        f.write(f"\n{'='*80}\nPER-LABEL BREAKDOWN\n{'='*80}\n")
        for label, results in sorted(by_label.items()):
            n = len(results)
            c = sum(results)
            f.write(f"  {label:<12} {c:>4}/{n:<4}  {c/n:.2%}\n")
    json_path = report_path.with_suffix(".json")
    result = {
        "model": model,
        "mode": f"claude-{mode}",
        "data": str(data_path),
        "timestamp": datetime.now().isoformat(),
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "by_label": {
            label: {"correct": sum(res), "total": len(res)}
            for label, res in sorted(by_label.items())
        },
    }
    with json_path.open("w") as f:
        json.dump(result, f, indent=2)


def print_results(
    model: str,
    total: int,
    correct: int,
    by_label: dict[str, list[bool]],
    report_path: Path,
) -> None:
    accuracy = correct / total if total else 0.0
    print(f"\n{'='*48}")
    print(f"  Model            : {model}")
    print(f"  Total redactions : {total}")
    print(f"  Exact matches    : {correct}")
    print(f"  Accuracy         : {accuracy:.2%}")
    print(f"{'='*48}")
    print("\nPer-label breakdown:")
    for label, results in sorted(by_label.items()):
        n = len(results)
        c = sum(results)
        print(f"  {label:<12} {c:>4}/{n:<4}  {c/n:.2%}")
    print(f"\nReport saved → {report_path}")


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------

def run_batch(
    client: anthropic.Anthropic,
    records: list[dict],
    model: str,
    data_path: Path,
) -> None:
    print(f"Model : {model}  (batch mode)")
    print(f"Data  : {data_path}")
    print(f"Articles: {len(records)}\n")

    requests = [
        Request(
            custom_id=str(record["pageid"]),
            params=MessageCreateParamsNonStreaming(
                model=model,
                max_tokens=512,
                system=SYSTEM_PROMPT,
                messages=build_message(record["text"]),
            ),
        )
        for record in records
    ]

    print(f"Submitting batch of {len(requests)} requests...")
    batch = client.messages.batches.create(requests=requests)
    print(f"Batch ID: {batch.id}")

    # Poll until complete
    while True:
        batch = client.messages.batches.retrieve(batch.id)
        counts = batch.request_counts
        print(
            f"  Status: {batch.processing_status} | "
            f"processing={counts.processing} succeeded={counts.succeeded} "
            f"errored={counts.errored}"
        )
        if batch.processing_status == "ended":
            break
        time.sleep(BATCH_POLL_INTERVAL)

    print("\nCollecting results...")
    predictions_by_id: dict[str, list[str]] = {}
    for result in client.messages.batches.results(batch.id):
        pageid = result.custom_id
        record = next((r for r in records if str(r["pageid"]) == pageid), None)
        if record is None:
            continue
        n_expected = len(record["redactions"])
        if result.result.type == "succeeded":
            msg = result.result.message
            raw = next((b.text for b in msg.content if b.type == "text"), "")
            predictions_by_id[pageid] = parse_predictions(raw, n_expected)
        else:
            predictions_by_id[pageid] = [""] * n_expected

    total, correct, by_label = score_predictions(records, predictions_by_id)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = model.replace("-", "_")
    report_path = Path("evals") / f"eval_claude_batch_{model_slug}_{timestamp}.txt"
    write_report(report_path, model, data_path, "batch", total, correct, by_label)
    print_results(model, total, correct, by_label, report_path)


# ---------------------------------------------------------------------------
# Sequential mode (rate-limited)
# ---------------------------------------------------------------------------

def call_claude_sequential(
    client: anthropic.Anthropic,
    model: str,
    text: str,
    n_expected: int,
) -> list[str]:
    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=512,
                system=SYSTEM_PROMPT,
                messages=build_message(text),
            )
            raw = next((b.text for b in response.content if b.type == "text"), "")
            return parse_predictions(raw, n_expected)
        except anthropic.RateLimitError:
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_DELAY * (attempt + 1)
                print(f"\nRate limited — waiting {wait}s...")
                time.sleep(wait)
            else:
                return [""] * n_expected
        except (anthropic.APIStatusError, anthropic.APIConnectionError):
            return [""] * n_expected
    return [""] * n_expected


def run_sequential(
    client: anthropic.Anthropic,
    records: list[dict],
    model: str,
    data_path: Path,
) -> None:
    print(f"Model : {model}  (sequential, ~27 RPM)")
    print(f"Data  : {data_path}")
    print(f"Articles: {len(records)}\n")

    predictions_by_id: dict[str, list[str]] = {}
    for i, record in enumerate(tqdm(records, desc="Articles")):
        n_expected = len(record["redactions"])
        predictions_by_id[str(record["pageid"])] = call_claude_sequential(
            client, model, record["text"], n_expected
        )
        # Rate limiting: sleep between requests (skip after the last one)
        if i < len(records) - 1:
            time.sleep(SEQUENTIAL_DELAY)

    total, correct, by_label = score_predictions(records, predictions_by_id)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = model.replace("-", "_")
    report_path = Path("evals") / f"eval_claude_sequential_{model_slug}_{timestamp}.txt"
    write_report(report_path, model, data_path, "sequential", total, correct, by_label)
    print_results(model, total, correct, by_label, report_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Claude topline eval — Engrammatic NEI")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Claude model ID (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--no-batch",
        action="store_true",
        help="Use sequential API calls instead of the Message Batches API",
    )
    args = parser.parse_args()

    client = anthropic.Anthropic()
    records = load_records(args.data)
    print(f"Loaded {len(records)} evaluable articles from {args.data}")

    if args.no_batch:
        run_sequential(client, records, args.model, args.data)
    else:
        run_batch(client, records, args.model, args.data)


if __name__ == "__main__":
    main()
