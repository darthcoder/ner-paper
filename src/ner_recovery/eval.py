"""Evaluate fine-tuned DistilBERT on redaction recovery.

Replaces every [REDACTED] span with [MASK], runs one inference pass per
article, takes the top-1 predicted token at each [MASK] position, and
compares against redactions[].original. Reports exact-match accuracy overall
and broken down by NER label.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from transformers import DistilBertForMaskedLM, DistilBertTokenizerFast
from tqdm import tqdm


MODEL_DIR = Path("models/final")
DATA_PATH = Path("data/wwi_redacted.jsonl")
MAX_LENGTH = 512
STRIDE = 256


def evaluate(args: argparse.Namespace) -> None:
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_dir)
    model = DistilBertForMaskedLM.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    total = 0
    correct = 0
    skipped = 0  # [MASK] tokens truncated out of the 512-token window
    by_label: dict[str, list[bool]] = defaultdict(list)

    with open(args.data) as f:
        records = [json.loads(line) for line in f]

    for record in tqdm(records, desc="Articles"):
        redactions = record["redactions"]
        if not redactions:
            continue

        sorted_reds = sorted(redactions, key=lambda r: r["start"])

        # Determine how many tokens each original entity occupies so we can
        # insert exactly that many [MASK] tokens (fixing the single-mask bug).
        red_n_tokens = [
            max(len(tokenizer.encode(r["original"], add_special_tokens=False)), 1)
            for r in sorted_reds
        ]

        # Build masked text with the correct number of [MASK]s per entity.
        parts: list[str] = []
        prev = 0
        for r, n_tok in zip(sorted_reds, red_n_tokens):
            parts.append(record["text"][prev:r["start"]])
            parts.append(" ".join(["[MASK]"] * n_tok))
            prev = r["end"]
        parts.append(record["text"][prev:])
        masked_text = "".join(parts)

        # Tokenize without truncation.
        full_enc = tokenizer(
            masked_text,
            add_special_tokens=False,
            truncation=False,
        )
        all_ids: list[int] = full_enc["input_ids"]

        mask_tok_id = tokenizer.mask_token_id
        all_mask_positions = [i for i, tid in enumerate(all_ids) if tid == mask_tok_id]

        # Group mask token positions by redaction index.
        red_mask_groups: list[list[int]] = []
        pos = 0
        for n_tok in red_n_tokens:
            red_mask_groups.append(all_mask_positions[pos:pos + n_tok])
            pos += n_tok

        # Build lookup: any mask token position → its redaction index.
        tok_to_red: dict[int, int] = {
            tok_idx: red_idx
            for red_idx, group in enumerate(red_mask_groups)
            for tok_idx in group
        }

        evaluated_red_indices: set[int] = set()

        cls_id = tokenizer.cls_token_id
        sep_id = tokenizer.sep_token_id
        pad_id = tokenizer.pad_token_id
        inner = MAX_LENGTH - 2  # slots between [CLS] and [SEP]
        n = len(all_ids)

        start = 0
        while start < n:
            end = min(start + inner, n)
            chunk_ids = all_ids[start:end]

            chunk_mask_global = [i for i in range(start, end) if all_ids[i] == mask_tok_id]
            new_reds_in_chunk = {
                tok_to_red[i] for i in chunk_mask_global
                if i in tok_to_red and tok_to_red[i] not in evaluated_red_indices
            }

            if not new_reds_in_chunk:
                if end == n:
                    break
                start += STRIDE
                continue

            # Build padded input for this window.
            input_ids_win = [cls_id] + chunk_ids + [sep_id]
            pad_len = MAX_LENGTH - len(input_ids_win)
            input_ids_win = input_ids_win + [pad_id] * pad_len
            attention_mask = [1] * (len(chunk_ids) + 2) + [0] * pad_len

            t_input = torch.tensor([input_ids_win], device=device)
            t_mask = torch.tensor([attention_mask], device=device)
            with torch.no_grad():
                logits = model(input_ids=t_input, attention_mask=t_mask).logits

            for red_idx in new_reds_in_chunk:
                group = red_mask_groups[red_idx]
                # All mask tokens for this entity must fall within this window.
                if not all(start <= tok_idx < end for tok_idx in group):
                    continue

                evaluated_red_indices.add(red_idx)
                r = sorted_reds[red_idx]

                # Predict one token per mask position, then decode together.
                predicted_ids = [
                    logits[0, (tok_idx - start) + 1].argmax().item()  # +1 for [CLS]
                    for tok_idx in group
                ]
                prediction = tokenizer.decode(predicted_ids, skip_special_tokens=True).strip()
                match = prediction.lower() == r["original"].lower()
                total += 1
                correct += int(match)
                by_label[r["label"]].append(match)

            if end == n:
                break
            start += STRIDE

        skipped += len(sorted_reds) - len(evaluated_red_indices)

    accuracy = correct / total if total else 0.0

    print(f"\n{'='*42}")
    print(f"  Total redactions : {total}")
    print(f"  Exact matches    : {correct}")
    print(f"  Truncated (skip) : {skipped}")
    print(f"  Accuracy         : {accuracy:.2%}")
    print(f"{'='*42}")
    print("\nPer-label breakdown:")
    for label, results in sorted(by_label.items()):
        n = len(results)
        c = sum(results)
        print(f"  {label:<12} {c:>4}/{n:<4}  {c/n:.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate DistilBERT MLM on redaction recovery"
    )
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR,
                        help="Path to saved model (default: models/final)")
    parser.add_argument("--data", type=Path, default=DATA_PATH,
                        help="Path to wwi_redacted.jsonl")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
