# Handoff: train.py Speedup — Spec for Claude Code

## Context

`train.py` (DistilBERT MLM, engrammatic NER inference) takes ~15h per run on
the full corpus on Apple Silicon MPS (24GB M-series, bs=8, fp32). Diagnosis
from code review identified four fixes. Apply **in order**, verify each with a
short timed run (50–100 steps) before moving to the next. No other refactors.

Repo: `~/Work/ner-paper`. Target file: `src/ner_recovery/train.py`.

---

## Fix 1 — Cache tokenized dataset to disk

**Problem**: `RedactionDataset.__init__` re-tokenizes the entire corpus
single-threaded on every run, before training starts.

**Spec**:
- Compute a cache path from the data path, e.g.
  `data/train_redacted.jsonl` → `data/cache/train_redacted.pt`
- On init: if cache exists and is newer than the source jsonl, load it
  (`torch.load`); else build examples as now, then `torch.save` the
  `self.examples` list to the cache.
- Add `--rebuild-cache` flag to force regeneration.
- Print which path was taken (cache hit / rebuild) and elapsed time.

**Verify**: second invocation of training skips straight to "N training
examples" in seconds.

---

## Fix 2 — Dynamic padding (biggest win)

**Problem**: `padding="max_length"` pads every example to 512 tokens.
Most windows are far shorter; compute is wasted on padding.

**Spec**:
- In `make_examples`, tokenize with `truncation=True, max_length=512` but
  **no padding**. Store variable-length lists.
- Add a `collate_fn` to the DataLoader that pads each batch to the longest
  sequence *in that batch* (pad token for input_ids, 0 for attention_mask,
  -100 for labels).
- Optional (only if trivial): sort examples by length once at dataset build,
  then use a batch sampler that keeps similar lengths together. Skip if it
  complicates shuffling — dynamic padding alone is the main win.

**Note**: cache from Fix 1 must be invalidated/rebuilt after this change
(example format changes). Bump a cache version string.

**Verify**: time-per-batch on a 50-step run drops substantially vs baseline.
Record before/after numbers.

---

## Fix 3 — Mixed precision on MPS

**Spec**:
- Wrap the forward+loss in
  `with torch.autocast(device_type="mps", dtype=torch.float16):`
- Keep backward/optimizer outside autocast. No GradScaler on MPS (not
  supported / not needed the way CUDA needs it); if loss goes NaN, fall back
  to bf16 or abort this fix and report.

**Verify**: 50-step run — confirm speedup and that loss values are sane
(finite, similar magnitude to fp32 baseline).

---

## Fix 4 — Retune batch size

**Spec**:
- After Fixes 2–3, try bs=16 then bs=32 on a short run; watch memory.
- Keep the largest bs that fits comfortably (no swapping/pressure warnings).
- If bs changes, note that lr may need a small bump (linear scaling is fine
  to try: bs 8→16 ⇒ lr 5e-5→1e-4), but only if loss curves degrade — don't
  tune preemptively.

---

## Fix 5 — Materialized dev subsample

**Problem**: corpus is now ~80k articles. Dev iterations must not pay the
cost of parsing/tokenizing the full set only to discard most of it.

**Spec**:
- Write a tiny one-off script (or bash one-liner) that creates a fixed dev
  split **as a file**: seeded (seed=42) random 5% of
  `data/train_redacted.jsonl` → `data/train_redacted_dev.jsonl`
  (~4k articles). Same for the test set if eval is also slow.
- This file gets its own tokenization cache via Fix 1 automatically (cache
  path derives from data path).
- Dev runs: `--data data/train_redacted_dev.jsonl`. Full runs unchanged.
- Optionally keep a `--subsample FLOAT` flag as a secondary knob, but the
  materialized file is the primary dev path.
- The dev split is for iteration speed only — report dev-run metrics as
  directional, never as paper numbers.

---

## Reporting back

After each fix, one line: fix name, steps timed, sec/step before → after.
At the end: projected full-run time at final settings.

Do NOT change: windowing logic in `make_examples`, loss/label construction,
early stopping, checkpoint layout.
