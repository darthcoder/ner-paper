"""Evaluate T5-small span reconstruction on redacted corpora.

Reports:
  - Exact match accuracy overall and per NER label
  - Hallucination rate: predictions that are non-empty but wrong
  - Abstention rate: predictions that are empty (model chose not to guess)

Run against in-domain test set and zero-shot Russian corpus to compare.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from tqdm import tqdm


MODEL_DIR = Path("models/final")
DATA_PATH = Path("data/test_redacted.jsonl")
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128
CHUNK_REDS = 10
CONTEXT_CHARS = 400
NUM_BEAMS = 4


# ---------------------------------------------------------------------------
# Input formatting (mirrors train.py)
# ---------------------------------------------------------------------------

def make_chunks(record: dict) -> list[tuple[str, list[dict]]]:
    """Return list of (input_text, chunk_redactions) for all chunks in record."""
    redactions = sorted(record.get("redactions", []), key=lambda r: r["start"])
    if not redactions:
        return []

    text = record["text"]
    chunks = []

    for chunk_start in range(0, len(redactions), CHUNK_REDS):
        chunk = redactions[chunk_start : chunk_start + CHUNK_REDS]

        span_start = max(0, chunk[0]["start"] - CONTEXT_CHARS)
        span_end = min(len(text), chunk[-1]["end"] + CONTEXT_CHARS)

        parts: list[str] = []
        prev = span_start
        for i, r in enumerate(chunk):
            parts.append(text[prev : r["start"]])
            parts.append(f"<extra_id_{i}>")
            prev = r["end"]
        parts.append(text[prev : span_end])

        chunks.append(("".join(parts), chunk))

    return chunks


def parse_predictions(output_text: str, n: int) -> list[str]:
    """Extract entity predictions from T5 sentinel output.

    Returns a list of length n; empty string where prediction is absent.
    """
    predictions: list[str] = []
    for i in range(n):
        sentinel = f"<extra_id_{i}>"
        next_sentinel = f"<extra_id_{i + 1}>"
        # Match text between this sentinel and the next (or end of string).
        pattern = (
            re.escape(sentinel)
            + r"\s*(.*?)(?="
            + re.escape(next_sentinel)
            + r"|</s>|$)"
        )
        m = re.search(pattern, output_text, re.DOTALL)
        predictions.append(m.group(1).strip() if m else "")
    return predictions


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device: {device}")
    print(f"Model:  {args.model_dir}")
    print(f"Data:   {args.data}\n")

    tokenizer = T5TokenizerFast.from_pretrained(args.model_dir)
    model = T5ForConditionalGeneration.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    total = 0
    correct = 0
    hallucinated = 0   # non-empty, wrong prediction
    abstained = 0      # empty prediction
    by_label: dict[str, list[bool]] = defaultdict(list)

    with open(args.data) as f:
        records = [json.loads(line) for line in f]

    for record in tqdm(records, desc="Articles"):
        chunks = make_chunks(record)
        if not chunks:
            continue

        for input_text, chunk in chunks:
            enc = tokenizer(
                input_text,
                max_length=MAX_INPUT_LENGTH,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                output_ids = model.generate(
                    **enc,
                    max_new_tokens=MAX_TARGET_LENGTH,
                    num_beams=args.num_beams,
                    early_stopping=True,
                )

            output_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
            predictions = parse_predictions(output_text, len(chunk))

            for pred, r in zip(predictions, chunk):
                total += 1
                gold = r["original"].strip().lower()
                pred_norm = pred.lower()

                if pred_norm == gold:
                    correct += 1
                    by_label[r["label"]].append(True)
                elif pred_norm == "":
                    abstained += 1
                    by_label[r["label"]].append(False)
                else:
                    hallucinated += 1
                    by_label[r["label"]].append(False)

    accuracy = correct / total if total else 0.0
    halluc_rate = hallucinated / total if total else 0.0
    abstain_rate = abstained / total if total else 0.0

    print(f"\n{'='*48}")
    print(f"  Total redactions  : {total}")
    print(f"  Exact matches     : {correct}  ({accuracy:.2%})")
    print(f"  Hallucinations    : {hallucinated}  ({halluc_rate:.2%})")
    print(f"  Abstentions       : {abstained}  ({abstain_rate:.2%})")
    print(f"{'='*48}")
    print("\nPer-label breakdown:")
    for label, results in sorted(by_label.items()):
        n = len(results)
        c = sum(results)
        print(f"  {label:<12} {c:>4}/{n:<4}  {c/n:.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate T5 span reconstruction on redacted corpora"
    )
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--num-beams", type=int, default=NUM_BEAMS)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
