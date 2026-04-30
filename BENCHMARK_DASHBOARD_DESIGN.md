# NER Paper Benchmark Dashboard — Design Brief

**Scope**: Static website hosted on GitHub Pages showing how different LLMs and fine-tuned models perform on the Named Entity Recovery task.

**Timeline**: Phase 1 (manual), extensible to Phase 2 (community submissions).

---

## Overview

This dashboard aggregates evaluation results from multiple models and displays them in an interactive leaderboard with filtering, sorting, and detailed breakdowns. It will be **hosted on GitHub Pages** and regenerated whenever new evaluation results are committed to the repository.

---

## Phase 1: Manual Update Workflow

**What happens behind the scenes**:
1. Researcher runs evaluation script (e.g., `uv run python scripts/eval_claude.py`)
2. Result JSON saved to `evals/` directory
3. Developer commits changes to git and pushes to main branch
4. GitHub Actions automatically rebuilds the dashboard from all JSON files in `evals/`
5. Static HTML deployed to GitHub Pages (live in ~2 minutes)

**Designer note**: Build assumes no database or server. Just static HTML/CSS/JS reading from JSON files. Very simple, very fast.

---

## Phase 2: Future Community Extensibility

Eventually: community members can submit their own results via pull requests. The system would validate the JSON, optionally re-run the evaluation, add contributor attribution, and auto-rebuild the dashboard. 

**For now, design with this in mind but don't build for it**. The architecture should cleanly support adding contributor names, submission dates, and PR links later without major redesign.

---

## Key UI Screens

### 1. Main Leaderboard (Homepage)

**Primary display**: Interactive table of all evaluation results.

**Columns** (dynamically generated from available entity types):
- Model name (clickable for details, sortable)
- Model type badge (e.g. "Claude Sonnet", "DistilBERT fine-tuned")
- Inference mode badge (e.g. "Batch", "Constrained")
- Dataset version (which test set used)
- **Overall accuracy** (%) — default sort descending, highlighted
- Per-entity-label columns:
  - PERSON
  - ORG
  - GPE
  - EVENT
  - LOC
  - NORP
  - FAC
  - CARDINAL
  - DATE
  - (any others in results)
- Date evaluated (sortable)

**Interactions**:
- Click any column header to sort
- Hover row to show full details tooltip (exact counts, timestamp)
- Click row to open detail modal
- Highlight the top result visually (e.g., gold background, star icon)

**Example snapshot**:
```
╔═══════════════════════════════════════════════════════════════════════╗
║ Model               │ Type       │ Mode      │ Accuracy │ PERSON │ ORG ║
╟───────────────────────────────────────────────────────────────────────╢
║ Claude Sonnet 4.6 ⭐ │ Frontier   │ Batch     │ 56.3%    │ 56.8%  │ 35% ║
║ Claude Haiku 4.5    │ Frontier   │ Batch     │ 44.0%    │ 29.5%  │ 25% ║
║ DistilBERT (curated)│ Fine-tuned │ Constrain │ 6.6%     │  —     │ —   ║
└───────────────────────────────────────────────────────────────────────┘
```

### 2. Detail Modal (Expand Row)

Opened by clicking a table row. Shows comprehensive info for one model's result:

**Header**:
- Full model name + display name
- Model type & description (e.g., "DistilBERT fine-tuned on 7,200 WWI articles, 9 epochs")
- Link to model if available (e.g., HuggingFace URL, GitHub repo)

**Main content**:
- Evaluation mode (e.g., "Constrained ranking on zero-OOV test set")
- Overall accuracy (large, prominent)
- Per-label accuracy table with exact counts:
  ```
  Entity Type │ Correct │ Total │ Accuracy
  ────────────┼─────────┼───────┼──────────
  PERSON      │   18    │  156  │  11.5%
  ORG         │   22    │  143  │  15.4%
  ...
  ```
- OOV analysis (if applicable): "37% of test entities were out-of-vocabulary"
- Timestamp: "Evaluated on 2026-04-28 at 14:30 UTC"
- Corpus info (if fine-tuned): "Trained on 7,200 articles from curated WWI corpus"

**Phase 2 note**: Add contributor name, submission date, and link to PR when community submissions are enabled.

### 3. Comparison View

Allow user to select 2–3 models and see them side-by-side.

**Layout**:
- Per-label accuracy columns shown for each selected model
- Bar chart showing per-label accuracy comparison (one bar per label, grouped by model)
- Difference scores: "Model A beats Model B by X% on PERSON" etc.
- Highlight where each model is strongest

**Example**:
```
Label   │ Claude Sonnet │ Claude Haiku │ DistilBERT
────────┼───────────────┼──────────────┼───────────
PERSON  │ 56.8%         │ 29.5%        │ —
ORG     │ 34.7%         │ 25.3%        │ —
EVENT   │ 58.1%         │ 35.5%        │ 10.3%
```

### 4. Trend Charts (Optional Phase 1, High Value)

**Line chart**: Accuracy over time (as new evals are run)
- X-axis: timestamp
- Y-axis: accuracy %
- One line per model, color-coded
- Legend with model names
- Show latest result per model by default

**Heatmap**: Per-label performance
- Rows: models
- Columns: entity labels (PERSON, ORG, GPE, etc.)
- Cell color intensity represents accuracy (red = low, green = high)
- Sortable by label or by best-performing model

---

## Sidebar & Filters

**Left sidebar** (persistent, sticky):

- **Filter by Model Type** (checkboxes):
  - [ ] Frontier LLMs (Claude, GPT, etc.)
  - [ ] Fine-tuned models (DistilBERT, custom)
  - [ ] Community submissions (Phase 2)

- **Filter by Inference Mode** (checkboxes):
  - [ ] Batch/Async
  - [ ] Sequential
  - [ ] Constrained ranking
  - [ ] Free generation

- **Filter by Dataset Version** (dropdown):
  - "Zero-OOV test set (316 articles)" — cleanest signal
  - "Full test set (825 articles)" — includes OOV entities
  - (future: other dataset variants)

- **Filter by Date Range** (date pickers):
  - From: [date picker]
  - To: [date picker]

- **Show only latest per model** (toggle):
  - When ON: show max 1 result per model (most recent)
  - When OFF: show all historical results (for trend analysis)

**Header & Footer**:

**Header**:
- Project title: "NER Paper Benchmark Dashboard"
- Brief description: "Comparing LLM and fine-tuned models on Named Entity Recovery"
- Last updated: "Updated 2 minutes ago"
- Link to GitHub repo

**Footer**:
- "How to submit results" (link to CONTRIBUTING guide for Phase 2)
- Link to evals/ directory on GitHub
- Link to benchmark scripts (eval_claude.py, benchmark.py)
- License & attribution
- Disclaimer: "Results reflect different evaluation conditions (model size, training data). Not a ranking. See methodology for details."

---

## Data & Visual Design Considerations

### Dataset Version Awareness

**Critical**: Different models evaluated on different test sets. Must be visually clear:
- When comparing models from different datasets, show a warning: "⚠️ These models were evaluated on different test sets"
- Include dataset version in table/modal (e.g., "zero_oov_v1" or "full_test_v1")
- Consider coloring or grouping rows by dataset version

### Handling Missing Data

Not all models will have results for all entity labels. Design should gracefully handle:
- Empty cells (show "—" or pale color)
- Reordering columns based on which labels have data
- Highlighting which entity types a model actually tested

### Label Flexibility

The model doesn't know in advance which entity types will appear. The dashboard must:
- Dynamically generate table columns based on what's in the JSON
- Support 7 minimum (PERSON, ORG, GPE, EVENT, LOC, NORP, FAC) but allow extras (CARDINAL, DATE, etc.)
- Order columns consistently (most common first, alphabetical as tiebreaker)

### Color & Visual Hierarchy

Suggested approach:
- **Accuracy gradient**: Use a color scale for per-label accuracy cells (red/yellow/green, or neutral/strong)
- **Badge colors**:
  - Frontier models: blue
  - Fine-tuned models: purple
  - Community submissions: green (Phase 2)
- **Highlighting**: Top overall result gets a subtle gold/star treatment
- **Mode badges**: Distinct visual style for batch vs. constrained vs. free

### Typography & Spacing

- Clear hierarchy: model name (large), overall accuracy (highlighted), per-label breakdowns (smaller)
- Generous whitespace in modals for readability
- Monospace font for numbers (accuracy %) for visual alignment
- Mobile-responsive: table scrolls horizontally on small screens, sidebar collapses to hamburger menu

---

## Data Schema

Each evaluation result is a JSON object. **Example**:

```json
{
  "model": "claude-sonnet-4-6",
  "display_name": "Claude Sonnet 4.6",
  "model_type": "frontier",
  "mode": "batch",
  "data": "data/test_zero_oov.jsonl",
  "dataset_version": "zero_oov_v1",
  "timestamp": "2026-04-28T14:30:21.123456",
  "total": 316,
  "correct": 178,
  "accuracy": 0.563,
  "by_label": {
    "NORP": {"correct": 63, "total": 79},
    "GPE": {"correct": 182, "total": 315},
    "EVENT": {"correct": 227, "total": 390},
    "PERSON": {"correct": 177, "total": 311},
    "ORG": {"correct": 87, "total": 251}
  }
}
```

**Key fields**:
- `model`: unique ID (e.g., "claude-sonnet-4-6")
- `display_name`: human-readable name for UI
- `model_type`: "frontier" | "fine-tuned" | "community"
- `mode`: "batch" | "sequential" | "constrained" | "free"
- `dataset_version`: "zero_oov_v1" | "full_test_v1" | ...
- `accuracy`: float 0–1 (displayed as %)
- `by_label`: dict of {label: {correct: int, total: int}}

---

## Interactions & User Flows

### Primary Flow (Casual Visitor)

1. Land on dashboard
2. See leaderboard table, models ranked by overall accuracy
3. Curious about Claude Sonnet → click row → detail modal opens
4. Read per-label breakdown, understand what entity types it's good/bad at
5. Close modal, return to table
6. Filter to "Fine-tuned models only" to see DistilBERT results
7. Compare Claude Sonnet vs. DistilBERT side-by-side
8. Done

### Advanced Flow (Researcher)

1. Filter by dataset version: "Zero-OOV" (cleanest signal)
2. Filter by date: only results from April 2026
3. Sort by PERSON accuracy (interested in pronoun/name inference)
4. Click on top result, examine per-label performance
5. View trend chart to see if accuracy improving over time
6. Export results (future: CSV download feature)

---

## Technical Notes for Implementation

- **Static site**: No server, no database. All state in URL query params (for sharing filters) and browser localStorage (for remembering user preferences).
- **Data loading**: JavaScript reads JSON files from `evals/` directory (served by GitHub Pages).
- **No build step required for design**: HTML/CSS/JS are served as-is; the repo's build process generates `index.html` from the JSON, but that's backend.
- **Accessibility**: WCAG 2.1 AA compliance (color contrast, keyboard navigation, screen reader support).
- **Performance**: Should load in <2s even with 50+ results; data is <500KB JSON.

---

## Future Extensions (Phase 2)

When community submissions are enabled:

1. **Contributor attribution**: Add "Submitted by: @username" to modals
2. **Leaderboard sections**: "Official Benchmarks" vs. "Community Results" tabs
3. **Contributor filter**: Filter by who submitted (dropdown or list)
4. **Submission status**: Show PR number, link to GitHub PR for reviewing
5. **Model cards**: Auto-generate detailed model card from result metadata (description, training procedure, notes)
6. **Historical tracking**: Archive results by date; show which results are "production" vs. "experimental"

Design should make room for these additions (e.g., space in modal footer for contributor info, section headers for separating official vs. community) without cluttering Phase 1.

---

## Deliverables from Design

1. **Wireframes**: Main leaderboard, detail modal, comparison view, chart views
2. **High-fidelity mockups**: Homepage, modal states, responsive mobile view
3. **Component library**: Table, badge, modal, filter panel, chart components
4. **Style guide**: Color palette, typography, spacing, interaction states
5. **Responsive design**: Mockups for desktop (1920px), tablet (768px), mobile (375px)
6. **Interactive prototype** (optional): Clickable Figma/Webflow showing navigation between screens

---

## Success Criteria

- Clean, modern aesthetic (fits GitHub Pages ecosystem)
- Fast to navigate (filters work instantly, modals open smoothly)
- Clear at a glance which model performs best
- Dataset version always visible (prevents misinterpretation)
- Mobile responsive (usable on phone)
- Extensible to community submissions without major redesign

---

## Files & Links

- **Results directory**: `evals/` (contains JSON files)
- **Build script**: `dashboard/build.py` (transforms JSON → HTML)
- **Deploy**: Hosted on GitHub Pages at `github.com/darthcoder/ner-paper/pages`
- **Schema**: `dashboard/schema.json` (defines expected JSON structure)
- **Example result**: `dashboard/sample-result.json`

---

**Ready to design!** Questions? See the technical plan in `/home/user/ner-paper/` for implementation details.
