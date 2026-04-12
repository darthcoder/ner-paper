"""Split wwi_clean.jsonl into 80/20 train/test sets, seeded for reproducibility."""
import json
import random
from pathlib import Path

IN = Path("data/wwi_clean.jsonl")
TRAIN_OUT = Path("data/train_clean.jsonl")
TEST_OUT = Path("data/test_clean.jsonl")
SEED = 42
TEST_RATIO = 0.20

records = [json.loads(line) for line in IN.open()]

rng = random.Random(SEED)
rng.shuffle(records)

split = round(len(records) * TEST_RATIO)
test_records = records[:split]
train_records = records[split:]

for path, subset in [(TRAIN_OUT, train_records), (TEST_OUT, test_records)]:
    with path.open("w") as f:
        for r in subset:
            f.write(json.dumps(r) + "\n")

print(f"Split {len(records)} articles → {len(train_records)} train / {len(test_records)} test")
