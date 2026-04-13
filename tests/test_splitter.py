"""Test train/test split integrity.

Verifies no article overlap between splits and that the total count is
consistent with the source corpus. Run once per split.
"""
import json
from pathlib import Path


TRAIN = Path("data/train_clean.jsonl")
TEST = Path("data/test_clean.jsonl")
SOURCE = Path("data/all_clean.jsonl")


def load_pageids(path: Path) -> set[int]:
    with path.open() as f:
        return {json.loads(line)["pageid"] for line in f}


def test_no_overlap():
    """No article should appear in both train and test."""
    train_ids = load_pageids(TRAIN)
    test_ids = load_pageids(TEST)
    overlap = train_ids & test_ids
    assert not overlap, f"Found {len(overlap)} overlapping pageids: {overlap}"


def test_total_count_matches_source():
    """Train + test should account for all articles in the source corpus."""
    train_ids = load_pageids(TRAIN)
    test_ids = load_pageids(TEST)
    source_ids = load_pageids(SOURCE)
    assert train_ids | test_ids == source_ids, (
        f"Split total {len(train_ids) + len(test_ids)} != "
        f"source total {len(source_ids)}"
    )


def test_split_ratio():
    """Test set should be ~20% of total (within 2 articles tolerance)."""
    train_ids = load_pageids(TRAIN)
    test_ids = load_pageids(TEST)
    total = len(train_ids) + len(test_ids)
    expected_test = round(total * 0.20)
    assert abs(len(test_ids) - expected_test) <= 2, (
        f"Test size {len(test_ids)} too far from expected {expected_test}"
    )
