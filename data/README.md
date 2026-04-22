# data/

All corpus and training data files.

## File Types

### Raw Corpora (`.jsonl`)

- `wwi_extended.jsonl` — ~4,300 Wikipedia WWI articles, cleaned plaintext
  - Fields: `{pageid, title, text}`
  - Source: `fetch_wikipedia_portal.py`

### Processed Data (`.jsonl`)

- `all_clean.jsonl` — Combined and deduplicated from all sources
- `train_clean.jsonl` — 80% split, plain text (432 articles)
- `test_clean.jsonl` — 20% split, plain text (108 articles)

### Redacted Data (`.jsonl`)

- `train_redacted.jsonl` — Training set with redacted entities
- `test_redacted.jsonl` — Test set with redacted entities
- `train_redacted_curated.jsonl` — Training set with noise filtered
- `test_redacted_curated.jsonl` — Test set with noise filtered

## Redaction Schema

Each record in `*_redacted.jsonl`:

```json
{
  "pageid": 4764461,
  "title": "World War I",
  "text": "...Austria-Hungary blamed [REDACTED:GPE], and declared war...",
  "redactions": [
    {
      "original": "Serbia",
      "label": "GPE",
      "start": 1107,
      "end": 1119
    }
  ]
}
```

- `original` — the entity text that was redacted
- `label` — entity type (PERSON, ORG, GPE, etc.)
- `start`, `end` — character offsets in the redacted text

## Entity Labels

After curation, we keep:
- **PERSON** — people (politicians, generals, etc.)
- **ORG** — organizations (governments, military units, etc.)
- **GPE** — geopolitical entities (countries, cities, regions)
- **EVENT** — events (battles, treaties, revolutions)
- **LOC** — locations (geographic features)
- **FAC** — facilities (buildings, infrastructure)
- **NORP** — nationalities/groups (Bolsheviks, Allies, etc.)

We filter out:
- CARDINAL, PERCENT, ORDINAL (numbers/ordinals)
- DATE, TIME (temporal expressions)
- MONEY, QUANTITY (amounts)
- LANGUAGE, LAW, PRODUCT, WORK_OF_ART (low signal)

## Statistics

| File | Articles | Redactions | Unique Entities |
|---|---|---|---|
| `train_redacted.jsonl` | 432 | 14,028 | 8,243 |
| `test_redacted.jsonl` | 108 | 825 | 632 |
| `train_redacted_curated.jsonl` | 407 | 8,188 | ~6,000 |
| `test_redacted_curated.jsonl` | 90 | 559 | ~400 |

(Curated stats: ~58% of training redactions kept after filtering noise)
