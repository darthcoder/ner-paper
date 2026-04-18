"""Fine-tune T5-small for self-supervised span reconstruction.

Input:  redacted text with [REDACTED] replaced by <extra_id_N> sentinels
Target: <extra_id_0> entity0 <extra_id_1> entity1 ...

No annotation labels are used — the (redacted_text, original) pairs in the
JSONL are the sole training signal.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import T5ForConditionalGeneration, T5TokenizerFast, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm


MODEL_NAME = "t5-small"
DATA_PATH = Path("data/train_redacted.jsonl")
OUTPUT_DIR = Path("models")
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128
CHUNK_REDS = 10       # max redactions per training example
CONTEXT_CHARS = 400   # chars of surrounding context per chunk
BATCH_SIZE = 8
EPOCHS = 7
LR = 3e-4
WARMUP_STEPS = 100


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def make_examples(record: dict, tokenizer: T5TokenizerFast) -> list[dict]:
    """Build T5 span-reconstruction examples from one redacted record.

    Each example covers up to CHUNK_REDS redactions with surrounding context.
    Input sentinels are renumbered from 0 within each chunk.
    """
    redactions = sorted(record.get("redactions", []), key=lambda r: r["start"])
    if not redactions:
        return []

    text = record["text"]
    examples: list[dict] = []

    for chunk_start in range(0, len(redactions), CHUNK_REDS):
        chunk = redactions[chunk_start : chunk_start + CHUNK_REDS]

        # Extract a text span covering the chunk with surrounding context.
        span_start = max(0, chunk[0]["start"] - CONTEXT_CHARS)
        span_end = min(len(text), chunk[-1]["end"] + CONTEXT_CHARS)

        # Build input: replace only THIS chunk's [REDACTED] markers with sentinels.
        parts: list[str] = []
        prev = span_start
        for i, r in enumerate(chunk):
            parts.append(text[prev : r["start"]])
            parts.append(f"<extra_id_{i}>")
            prev = r["end"]
        parts.append(text[prev : span_end])
        input_text = "".join(parts)

        # Build target: <extra_id_0> entity0 <extra_id_1> entity1 ...
        target_text = " ".join(
            f"<extra_id_{i}> {r['original']}" for i, r in enumerate(chunk)
        )

        inp = tokenizer(
            input_text,
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding="max_length",
        )
        tgt = tokenizer(
            target_text,
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
            padding="max_length",
        )
        # Mask padding in labels so it doesn't contribute to loss.
        labels = [
            t if t != tokenizer.pad_token_id else -100
            for t in tgt["input_ids"]
        ]

        examples.append({
            "input_ids": inp["input_ids"],
            "attention_mask": inp["attention_mask"],
            "labels": labels,
        })

    return examples


class RedactionDataset(Dataset):
    def __init__(self, path: Path, tokenizer: T5TokenizerFast) -> None:
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

    tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
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

    final_dir = args.output_dir / "final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Final model saved → {final_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune T5-small for self-supervised span reconstruction"
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
