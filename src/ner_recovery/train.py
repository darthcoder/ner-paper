"""Fine-tune DistilBERT as a masked language model on the redacted WWI corpus.

For each article, [REDACTED] spans are reconstructed to their original entity
text. Those entity token positions are then masked in input_ids so the model
learns to predict the original tokens from surrounding context.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BertForMaskedLM,
    BertTokenizerFast,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from tqdm import tqdm


MODEL_NAME = "bert-base-uncased"
DATA_PATH = Path("data/train_redacted.jsonl")
OUTPUT_DIR = Path("models")
MAX_LENGTH = 512
STRIDE = 256
BATCH_SIZE = 8
EPOCHS = 7
LR = 5e-5
WARMUP_STEPS = 50


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def reconstruct(text: str, redactions: list[dict]) -> tuple[str, list[tuple[int, int]]]:
    """Replace [REDACTED] markers with the original entity text.

    Returns:
        original_text: the reconstructed article text
        spans: list of (start, end) char offsets in original_text for each entity
    """
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    prev = 0
    char_delta = 0  # cumulative offset shift from prior replacements

    for r in sorted(redactions, key=lambda x: x["start"]):
        parts.append(text[prev : r["start"]])
        entity_start = r["start"] + char_delta
        original = r["original"]
        parts.append(original)
        spans.append((entity_start, entity_start + len(original)))
        char_delta += len(original) - (r["end"] - r["start"])
        prev = r["end"]

    parts.append(text[prev:])
    return "".join(parts), spans


def make_examples(
    record: dict,
    tokenizer: DistilBertTokenizerFast,
    max_length: int,
    stride: int,
) -> list[dict]:
    """Build sliding-window MLM examples from one redacted record.

    Each example is a dict with input_ids, attention_mask, and labels, all
    padded to max_length. Labels are -100 everywhere except at entity token
    positions, where they hold the original token id.
    """
    if not record["redactions"]:
        return []

    original_text, entity_spans = reconstruct(record["text"], record["redactions"])

    encoding = tokenizer(
        original_text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=False,
    )
    token_ids: list[int] = encoding["input_ids"]
    offset_map: list[tuple[int, int]] = encoding["offset_mapping"]

    # Mark which token positions overlap with an entity span
    n = len(token_ids)
    is_entity = [False] * n
    for ent_start, ent_end in entity_spans:
        for i, (tok_start, tok_end) in enumerate(offset_map):
            if tok_end > ent_start and tok_start < ent_end:
                is_entity[i] = True

    mask_id: int = tokenizer.mask_token_id
    cls_id: int = tokenizer.cls_token_id
    sep_id: int = tokenizer.sep_token_id
    pad_id: int = tokenizer.pad_token_id
    inner = max_length - 2  # token slots between [CLS] and [SEP]

    examples: list[dict] = []
    start = 0

    while start < n:
        end = min(start + inner, n)
        chunk_ids = token_ids[start:end]
        chunk_entity = is_entity[start:end]

        if any(chunk_entity):
            input_ids = [cls_id]
            labels = [-100]
            for tid, ent in zip(chunk_ids, chunk_entity):
                input_ids.append(mask_id if ent else tid)
                labels.append(tid if ent else -100)
            input_ids.append(sep_id)
            labels.append(-100)

            pad_len = max_length - len(input_ids)
            examples.append({
                "input_ids": input_ids + [pad_id] * pad_len,
                "attention_mask": [1] * len(input_ids) + [0] * pad_len,
                "labels": labels + [-100] * pad_len,
            })

        if end == n:
            break
        start += stride

    return examples


class RedactionDataset(Dataset):
    def __init__(
        self,
        path: Path,
        tokenizer: DistilBertTokenizerFast,
        max_length: int = MAX_LENGTH,
        stride: int = STRIDE,
    ) -> None:
        self.examples: list[dict] = []
        with open(path) as f:
            for line in f:
                record = json.loads(line)
                self.examples.extend(make_examples(record, tokenizer, max_length, stride))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {k: torch.tensor(v) for k, v in self.examples[idx].items()}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    model = BertForMaskedLM.from_pretrained(MODEL_NAME)
    model.to(device)

    print("Building dataset...")
    dataset = RedactionDataset(args.data, tokenizer, args.max_length, args.stride)
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
        description="Fine-tune DistilBERT MLM on the redacted WWI corpus"
    )
    parser.add_argument("--data", type=Path, default=DATA_PATH,
                        help="Path to wwi_redacted.jsonl")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Directory for checkpoints and final model")
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH,
                        help="Token window size (default 512)")
    parser.add_argument("--stride", type=int, default=STRIDE,
                        help="Sliding window stride (default 256)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--warmup", type=int, default=WARMUP_STEPS,
                        help="Linear warmup steps")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
