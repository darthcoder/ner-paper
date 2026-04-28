"""Topline evaluation: run a local model via LM Studio on the NEI test set.

LM Studio exposes an OpenAI-compatible server at http://localhost:1234/v1.
Load your model in LM Studio, start the local server, then run this script.

The JSON output format matches eval_claude.py and eval.py, so results are
automatically picked up by benchmark_table.py.

Usage:
    uv run python scripts/eval_lmstudio.py --model smollm3-3b
    uv run python scripts/eval_lmstudio.py --model smollm3-3b --base-url http://localhost:1234/v1
    uv run python scripts/eval_lmstudio.py --model smollm3-3b --data data/test_zero_oov.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

DEFAULT_DATA = Path("data/test_zero_oov.jsonl")
DEFAULT_BASE_URL = "http://localhost:1234/v1"

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


def call_model(
    client: OpenAI,
    model: str,
    text: str,
    n_expected: int,
    max_retries: int = 3,
) -> list[str]:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                max_tokens=512,
                temperature=0,
            )
            raw = response.choices[0].message.content or ""
            return parse_predictions(raw, n_expected)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
            else:
                print(f"\nError after {max_retries} attempts: {e}")
                return [""] * n_expected
    return [""] * n_expected


def evaluate(args: argparse.Namespace) -> None:
    client = OpenAI(base_url=args.base_url, api_key="lm-studio")

    print(f"Model    : {args.model}")
    print(f"Base URL : {args.base_url}")
    print(f"Data     : {args.data}\n")

    records = load_records(args.data)
    print(f"Loaded {len(records)} evaluable articles\n")

    total = correct = 0
    by_label: dict[str, list[bool]] = defaultdict(list)

    for record in tqdm(records, desc="Articles"):
        n_tokens = record["_n_tokens"]
        predictions = call_model(client, args.model, record["text"], n_tokens)
        pred_indices = record.get("_pred_indices", list(range(len(record["redactions"]))))
        for r, idx in zip(record["redactions"], pred_indices):
            pred = predictions[idx] if idx < len(predictions) else ""
            original = r["original"].strip()
            total += 1
            match = pred.lower() == original.lower()
            if match:
                correct += 1
            by_label[r["label"]].append(match)

    accuracy = correct / total if total else 0.0

    print(f"\n{'='*48}")
    print(f"  Model            : {args.model}")
    print(f"  Total redactions : {total}")
    print(f"  Exact matches    : {correct}")
    print(f"  Accuracy         : {accuracy:.2%}")
    print(f"{'='*48}")
    print("\nPer-label breakdown:")
    for label, results in sorted(by_label.items()):
        n = len(results)
        c = sum(results)
        print(f"  {label:<12} {c:>4}/{n:<4}  {c/n:.2%}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = re.sub(r"[^a-z0-9]+", "_", args.model.lower()).strip("_")
    stem = f"eval_lmstudio_{model_slug}_{timestamp}"
    Path("evals").mkdir(exist_ok=True)

    txt_path = Path("evals") / f"{stem}.txt"
    with txt_path.open("w") as f:
        f.write(f"{'='*80}\nENGRAMMATIC NAMED ENTITY INFERENCE — LM STUDIO EVALUATION\n{'='*80}\n")
        f.write(f"Date/Time     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model         : {args.model}\n")
        f.write(f"Base URL      : {args.base_url}\n")
        f.write(f"Data          : {args.data}\n\n")
        f.write(f"{'='*80}\nRESULTS\n{'='*80}\n")
        f.write(f"  Total redactions : {total}\n")
        f.write(f"  Exact matches    : {correct}\n")
        f.write(f"  Accuracy         : {accuracy:.2%}\n")
        f.write(f"\n{'='*80}\nPER-LABEL BREAKDOWN\n{'='*80}\n")
        for label, results in sorted(by_label.items()):
            n = len(results)
            c = sum(results)
            f.write(f"  {label:<12} {c:>4}/{n:<4}  {c/n:.2%}\n")

    json_path = Path("evals") / f"{stem}.json"
    result = {
        "model": args.model,
        "mode": "lmstudio",
        "base_url": args.base_url,
        "data": str(args.data),
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

    print(f"\nReport saved → {txt_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LM Studio eval — Engrammatic NEI")
    parser.add_argument("--model", required=True, help="Model name as shown in LM Studio")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help=f"LM Studio server URL (default: {DEFAULT_BASE_URL})")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
