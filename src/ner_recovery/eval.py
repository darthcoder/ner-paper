"""Evaluate distilbert-base-uncased MLM for engrammatic named entity inference.

For each [REDACTED:LABEL] token, replaces it with N [MASK] tokens (where N
matches the original entity's subword count), runs the model, and joins the
predicted tokens. Reports exact-match accuracy overall and per NER label.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from datetime import datetime

import torch
from transformers import DistilBertForMaskedLM, DistilBertTokenizerFast
from tqdm import tqdm


MODEL_DIR = Path("models/final")
DATA_PATH = Path("data/test_redacted.jsonl")
MAX_LENGTH = 512
STRIDE = 256


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

REDACTED_RE = re.compile(r"\[REDACTED(?::[A-Z_]+)?\]")


def predict_record(record: dict, model: BertForMaskedLM, tokenizer: BertTokenizerFast, device: torch.device) -> list[tuple[str, str, str]]:
    """Return list of (predicted, original, label) for each redaction."""
    text = record["text"]
    redactions = sorted(record.get("redactions", []), key=lambda r: r["start"])
    if not redactions:
        return []

    results = []

    for r in redactions:
        original = r["original"]
        label = r["label"]

        # Get number of subword tokens for the original entity
        entity_ids = tokenizer.encode(original, add_special_tokens=False)
        n_masks = max(1, len(entity_ids))

        # Build input: replace this redaction's token with N [MASK]s
        masked_text = text[: r["start"]] + " ".join([tokenizer.mask_token] * n_masks) + text[r["end"] :]

        enc = tokenizer(
            masked_text,
            max_length=MAX_LENGTH,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        input_ids = enc["input_ids"][0]
        mask_positions = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]

        if len(mask_positions) == 0:
            results.append(("", original, label))
            continue

        with torch.no_grad():
            logits = model(**enc).logits[0]

        predicted_tokens = [
            tokenizer.decode([logits[pos].argmax().item()]).strip()
            for pos in mask_positions
        ]
        predicted = "".join(predicted_tokens).strip()
        results.append((predicted, original, label))

    return results


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

    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_dir)
    model = DistilBertForMaskedLM.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    total = correct = 0
    by_label: dict[str, list[bool]] = defaultdict(list)

    with open(args.data) as f:
        records = [json.loads(line) for line in f]

    for record in tqdm(records, desc="Articles"):
        for predicted, original, label in predict_record(record, model, tokenizer, device):
            total += 1
            match = predicted.lower() == original.strip().lower()
            if match:
                correct += 1
            by_label[label].append(match)

    accuracy = correct / total if total else 0.0

    print(f"\n{'='*48}")
    print(f"  Total redactions : {total}")
    print(f"  Exact matches    : {correct}")
    print(f"  Truncated (skip) : 0")
    print(f"  Accuracy         : {accuracy:.2%}")
    print(f"{'='*48}")
    print("\nPer-label breakdown:")
    for label, results in sorted(by_label.items()):
        n = len(results)
        c = sum(results)
        print(f"  {label:<12} {c:>4}/{n:<4}  {c/n:.2%}")

    # Save eval report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path("evals") / f"eval_{timestamp}.txt"
    report_path.parent.mkdir(exist_ok=True)
    with report_path.open("w") as f:
        f.write(f"{'='*80}\nNER RECOVERY — EVALUATION REPORT\n{'='*80}\n")
        f.write(f"Date/Time     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model         : {args.model_dir}\n")
        f.write(f"Data          : {args.data}\n\n")
        f.write(f"{'='*80}\nRESULTS\n{'='*80}\n")
        f.write(f"  Total redactions : {total}\n")
        f.write(f"  Exact matches    : {correct}\n")
        f.write(f"  Accuracy         : {accuracy:.2%}\n\n")
        f.write(f"{'='*80}\nPER-LABEL BREAKDOWN\n{'='*80}\n")
        for label, results in sorted(by_label.items()):
            n = len(results)
            c = sum(results)
            f.write(f"  {label:<12} {c:>4}/{n:<4}  {c/n:.2%}\n")
    print(f"\nReport saved → {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate BERT MLM on redacted corpora"
    )
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
