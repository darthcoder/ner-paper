"""Evaluate distilbert-base-uncased MLM for engrammatic named entity inference.

Two inference modes:
  - free (default): argmax over full vocabulary at each [MASK] position
  - constrained: score each candidate from the per-label train set, return best

Constrained mode is inspired by Engram (arXiv:2601.07372) — entity resolution
is a lookup, not generation. Collapsing the search space from ~30k vocab tokens
to ~50-300 per-label candidates makes the task tractable for smaller models.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from datetime import datetime

import torch
from transformers import DistilBertForMaskedLM, DistilBertTokenizerFast
from tqdm import tqdm


MODEL_DIR = Path("models/final")
DATA_PATH = Path("data/test_redacted.jsonl")
TRAIN_PATH = Path("data/train_redacted.jsonl")
MAX_LENGTH = 512


# ---------------------------------------------------------------------------
# Candidate set construction
# ---------------------------------------------------------------------------

def build_candidate_sets(train_path: Path) -> dict[str, list[str]]:
    """Collect all unique entities per NER label from training redactions."""
    candidates: dict[str, set[str]] = defaultdict(set)
    with open(train_path) as f:
        for line in f:
            for r in json.loads(line).get("redactions", []):
                candidates[r["label"]].add(r["original"].strip())
    result = {label: sorted(ents) for label, ents in candidates.items()}
    for label, ents in result.items():
        print(f"  {label:<12} {len(ents):>4} candidates")
    return result


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def get_logits_at_masks(
    context_before: str,
    context_after: str,
    n_masks: int,
    model: DistilBertForMaskedLM,
    tokenizer: DistilBertTokenizerFast,
    device: torch.device,
) -> tuple[torch.Tensor, list[int]] | None:
    """Single forward pass; return (log_probs [n_masks, vocab], mask_positions)."""
    masked_text = context_before + " ".join([tokenizer.mask_token] * n_masks) + context_after
    enc = tokenizer(masked_text, max_length=MAX_LENGTH, truncation=True, return_tensors="pt").to(device)
    input_ids = enc["input_ids"][0]
    mask_positions = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[0].tolist()
    if not mask_positions:
        return None
    with torch.no_grad():
        logits = model(**enc).logits[0]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    return log_probs[mask_positions], mask_positions


def score_candidate_from_logits(
    candidate: str,
    tokenizer: DistilBertTokenizerFast,
    log_probs: torch.Tensor,  # [n_masks, vocab]
) -> float:
    """Score candidate using pre-computed log_probs — no forward pass needed."""
    entity_ids = tokenizer.encode(candidate, add_special_tokens=False)
    score = 0.0
    for i, tok_id in enumerate(entity_ids):
        if i < log_probs.shape[0]:
            score += log_probs[i, tok_id].item()
    return score


def predict_free(
    r: dict,
    text: str,
    model: DistilBertForMaskedLM,
    tokenizer: DistilBertTokenizerFast,
    device: torch.device,
) -> str:
    """Argmax over full vocabulary at each mask position."""
    entity_ids = tokenizer.encode(r["original"], add_special_tokens=False)
    n_masks = max(1, len(entity_ids))
    result = get_logits_at_masks(text[: r["start"]], text[r["end"] :], n_masks, model, tokenizer, device)
    if result is None:
        return ""
    log_probs, _ = result
    tokens = [tokenizer.decode([log_probs[i].argmax().item()]).strip() for i in range(log_probs.shape[0])]
    return "".join(tokens).strip()


def predict_constrained(
    r: dict,
    text: str,
    candidates: list[str],
    model: DistilBertForMaskedLM,
    tokenizer: DistilBertTokenizerFast,
    device: torch.device,
) -> str:
    """One forward pass per unique mask-count, then rank all candidates from logits."""
    # Group candidates by their token length so we run one pass per group
    by_length: dict[int, list[str]] = defaultdict(list)
    for cand in candidates:
        n = max(1, len(tokenizer.encode(cand, add_special_tokens=False)))
        by_length[n].append(cand)

    best_score = -math.inf
    best = candidates[0]

    for n_masks, cands in by_length.items():
        result = get_logits_at_masks(text[: r["start"]], text[r["end"] :], n_masks, model, tokenizer, device)
        if result is None:
            continue
        log_probs, _ = result
        for cand in cands:
            s = score_candidate_from_logits(cand, tokenizer, log_probs)
            if s > best_score:
                best_score = s
                best = cand

    return best


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device:  {device}")
    print(f"Model:   {args.model_dir}")
    print(f"Data:    {args.data}")
    print(f"Mode:    {args.mode}\n")

    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_dir)
    model = DistilBertForMaskedLM.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    candidate_sets: dict[str, list[str]] = {}
    if args.mode == "constrained":
        print("Building candidate sets from training data...")
        candidate_sets = build_candidate_sets(args.train)
        total_candidates = sum(len(v) for v in candidate_sets.values())
        print(f"  Total: {total_candidates} unique entities across {len(candidate_sets)} labels\n")

    total = correct = oov = 0
    by_label: dict[str, list[bool]] = defaultdict(list)

    with open(args.data) as f:
        records = [json.loads(line) for line in f]

    for record in tqdm(records, desc="Articles"):
        text = record["text"]
        for r in sorted(record.get("redactions", []), key=lambda x: x["start"]):
            original = r["original"].strip()
            label = r["label"]
            total += 1

            if args.mode == "constrained":
                cands = candidate_sets.get(label, [])
                if not cands:
                    predicted = ""
                elif original.lower() not in {c.lower() for c in cands}:
                    oov += 1
                    predicted = predict_constrained(r, text, cands, model, tokenizer, device)
                else:
                    predicted = predict_constrained(r, text, cands, model, tokenizer, device)
            else:
                predicted = predict_free(r, text, model, tokenizer, device)

            match = predicted.lower() == original.lower()
            if match:
                correct += 1
            by_label[label].append(match)

    accuracy = correct / total if total else 0.0

    print(f"\n{'='*48}")
    print(f"  Total redactions : {total}")
    print(f"  Exact matches    : {correct}")
    print(f"  Accuracy         : {accuracy:.2%}")
    if args.mode == "constrained":
        print(f"  OOV (not in train candidates): {oov} ({oov/total:.1%})")
    print(f"{'='*48}")
    print("\nPer-label breakdown:")
    for label, results in sorted(by_label.items()):
        n = len(results)
        c = sum(results)
        print(f"  {label:<12} {c:>4}/{n:<4}  {c/n:.2%}")

    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path("evals") / f"eval_{args.mode}_{timestamp}.txt"
    report_path.parent.mkdir(exist_ok=True)
    with report_path.open("w") as f:
        f.write(f"{'='*80}\nENGRAMMATIC NAMED ENTITY INFERENCE — EVALUATION REPORT\n{'='*80}\n")
        f.write(f"Date/Time     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model         : {args.model_dir}\n")
        f.write(f"Data          : {args.data}\n")
        f.write(f"Mode          : {args.mode}\n\n")
        f.write(f"{'='*80}\nRESULTS\n{'='*80}\n")
        f.write(f"  Total redactions : {total}\n")
        f.write(f"  Exact matches    : {correct}\n")
        f.write(f"  Accuracy         : {accuracy:.2%}\n")
        if args.mode == "constrained":
            f.write(f"  OOV              : {oov} ({oov/total:.1%})\n")
        f.write(f"\n{'='*80}\nPER-LABEL BREAKDOWN\n{'='*80}\n")
        for label, results in sorted(by_label.items()):
            n = len(results)
            c = sum(results)
            f.write(f"  {label:<12} {c:>4}/{n:<4}  {c/n:.2%}\n")
    print(f"\nReport saved → {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DistilBERT MLM — Engrammatic NEI")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    parser.add_argument("--mode", choices=["free", "constrained"], default="constrained",
                        help="free: argmax over vocab; constrained: rank train candidates (default)")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
