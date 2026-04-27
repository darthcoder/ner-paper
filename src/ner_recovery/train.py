"""Fine-tune distilbert-base-uncased as a Masked Language Model for engrammatic named entity inference.

Each [REDACTED:LABEL] token in the input is replaced by one [MASK] per subword
token of the original entity. The model is trained to predict those tokens.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertForMaskedLM, DistilBertTokenizerFast, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm


MODEL_NAME = "distilbert-base-uncased"
DATA_PATH = Path("data/train_redacted.jsonl")
OUTPUT_DIR = Path("models")
MAX_LENGTH = 512
STRIDE = 256
BATCH_SIZE = 8
EPOCHS = 3
LR = 5e-5
WARMUP_STEPS = 50


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def make_examples(record: dict, tokenizer: DistilBertTokenizerFast) -> list[dict]:
    """Build MLM examples from one redacted record.

    Groups nearby redactions into shared 512-token windows (~2000 char spans).
    Multiple [MASK] sequences appear in each example; labels filled at each
    MASK position with the corresponding entity subword tokens.
    """
    text = record["text"]
    redactions = sorted(record.get("redactions", []), key=lambda r: r["start"])
    if not redactions:
        return []

    # Group redactions into windows: start a new window when the next redaction
    # is more than WINDOW_CHARS away from the current window start.
    WINDOW_CHARS = 2000
    groups: list[list[dict]] = []
    current_group: list[dict] = []
    window_start = -1

    for r in redactions:
        if window_start < 0 or r["start"] - window_start > WINDOW_CHARS:
            if current_group:
                groups.append(current_group)
            current_group = [r]
            window_start = r["start"]
        else:
            current_group.append(r)
    if current_group:
        groups.append(current_group)

    examples: list[dict] = []
    ctx_pad = 200  # chars of context before/after the group

    for group in groups:
        span_start = max(0, group[0]["start"] - ctx_pad)
        span_end = min(len(text), group[-1]["end"] + ctx_pad)

        # Build local text with all group redactions replaced by [MASK]s
        parts: list[str] = []
        prev = span_start
        all_entity_ids: list[list[int]] = []

        for r in group:
            parts.append(text[prev : r["start"]])
            entity_ids = tokenizer.encode(r["original"], add_special_tokens=False)
            n_masks = max(1, len(entity_ids))
            parts.append(" ".join([tokenizer.mask_token] * n_masks))
            all_entity_ids.append(entity_ids)
            prev = r["end"]
        parts.append(text[prev : span_end])
        local_text = "".join(parts)

        enc = tokenizer(
            local_text,
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"][0].tolist()
        attention_mask = enc["attention_mask"][0].tolist()

        # Fill labels at MASK positions in order of entities
        labels = [-100] * len(input_ids)
        entity_queue = [tok for ids in all_entity_ids for tok in ids]
        eq_idx = 0
        for i, tok in enumerate(input_ids):
            if tok == tokenizer.mask_token_id and eq_idx < len(entity_queue):
                labels[i] = entity_queue[eq_idx]
                eq_idx += 1

        if not any(l != -100 for l in labels):
            continue

        examples.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        })

    return examples


class RedactionDataset(Dataset):
    def __init__(self, path: Path, tokenizer: DistilBertTokenizerFast) -> None:
        self.examples: list[dict] = []
        with open(path) as f:
            for line in f:
                record = json.loads(line)
                self.examples.extend(make_examples(record, tokenizer))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {k: torch.tensor(v) for k, v in self.examples[idx].items()}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    model = DistilBertForMaskedLM.from_pretrained(MODEL_NAME)
    model.to(device)

    print("Building dataset...")
    dataset = RedactionDataset(args.data, tokenizer)
    print(f"  {len(dataset)} training examples from {args.data}")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup,
        num_training_steps=total_steps,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    prev_loss = None
    early_stop_threshold = 0.1

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}: avg loss {avg_loss:.4f}")

        ckpt_dir = args.output_dir / f"checkpoint-epoch{epoch}"
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        print(f"  Checkpoint saved → {ckpt_dir}")

        # Early stopping: check loss improvement
        if prev_loss is not None:
            loss_diff = prev_loss - avg_loss
            print(f"  Loss improvement: {loss_diff:.4f}")
            if loss_diff < early_stop_threshold:
                print(f"  ⚠ Loss improvement < {early_stop_threshold}. Early stopping.")
                break
        prev_loss = avg_loss

    final_dir = args.output_dir / "final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Final model saved → {final_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune distilbert-base-uncased MLM for engrammatic named entity inference"
    )
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--warmup", type=int, default=WARMUP_STEPS)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
