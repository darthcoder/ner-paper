"""Test the multi-token masking logic from eval.py.

Verifies that for a given entity string, inserting N [MASK] tokens produces
exactly N mask token IDs when tokenized. This is the fix-#1 invariant.

Run often — fast, no GPU required.
"""
import pytest
from transformers import BertTokenizerFast

MODEL_NAME = "bert-base-uncased"
MASK_TOKEN = "[MASK]"

# (entity_text, expected_min_tokens) — min because subword count varies
ENTITIES = [
    "Serbia",
    "Franz Ferdinand",
    "Austria-Hungary",
    "Western Front",
    "Treaty of Versailles",
    "the",
    "1914",
    "Archduke",
]


@pytest.fixture(scope="module")
def tokenizer():
    return BertTokenizerFast.from_pretrained(MODEL_NAME)


@pytest.mark.parametrize("entity", ENTITIES)
def test_mask_count_matches_token_count(tokenizer, entity):
    """Inserting N [MASK]s for an entity must yield exactly N mask token IDs."""
    token_ids = tokenizer.encode(entity, add_special_tokens=False)
    n = max(len(token_ids), 1)

    masked_text = f"Before { ' '.join([MASK_TOKEN] * n) } after."
    encoded = tokenizer(masked_text, add_special_tokens=False)["input_ids"]
    mask_count = encoded.count(tokenizer.mask_token_id)

    assert mask_count == n, (
        f"Entity {entity!r}: expected {n} mask(s), got {mask_count}"
    )


def test_single_token_entity(tokenizer):
    """A single-token entity gets exactly one [MASK]."""
    entity = "France"
    ids = tokenizer.encode(entity, add_special_tokens=False)
    assert len(ids) == 1, f"Expected 'France' to be 1 token, got {len(ids)}"

    masked = f"Hello {MASK_TOKEN} goodbye."
    encoded = tokenizer(masked, add_special_tokens=False)["input_ids"]
    assert encoded.count(tokenizer.mask_token_id) == 1


def test_multi_token_entity(tokenizer):
    """A multi-token entity gets one [MASK] per token."""
    entity = "Franz Ferdinand"
    ids = tokenizer.encode(entity, add_special_tokens=False)
    n = len(ids)
    assert n > 1, "Expected 'Franz Ferdinand' to be multi-token"

    masked = f"Hello { ' '.join([MASK_TOKEN] * n) } goodbye."
    encoded = tokenizer(masked, add_special_tokens=False)["input_ids"]
    assert encoded.count(tokenizer.mask_token_id) == n


def test_decoded_prediction_roundtrip(tokenizer):
    """Decoding predicted token IDs for a known entity reconstructs it."""
    entity = "Serbia"
    ids = tokenizer.encode(entity, add_special_tokens=False)
    decoded = tokenizer.decode(ids, skip_special_tokens=True).strip()
    assert decoded.lower() == entity.lower()
