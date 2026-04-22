#!/usr/bin/env bash
# Fetch the Monroe Doctrine → WWI arc (1823–1914) from Wikipedia.
# Appends into wwi_extended.jsonl; already-fetched titles are skipped.
# Usage: bash scripts/fetch_monroe_wwi.sh [resume_from]
#   resume_from: optional category number or name substring to resume from
# Output: data/wwi_extended.jsonl
# Log:    data/monroe_wwi_fetch.log

set -euo pipefail

OUTPUT="data/wwi_extended.jsonl"
LOG="data/monroe_wwi_fetch.log"
FETCHER="uv run python fetcher.py"

CATEGORIES=(
    # --- Monroe Doctrine & early expansionism ---
    "Category:Monroe Doctrine"
    "Category:Manifest Destiny"
    "Category:Texas Revolution"
    "Category:Mexican-American War"

    # --- Civil War diplomatic context ---
    "Category:American Civil War diplomacy"
    "Category:Trent Affair"
    "Category:Confederate foreign relations"

    # --- Gilded Age / late 19th century ---
    "Category:Spanish-American War"
    "Category:Philippine-American War"
    "Category:Cuban War of Independence"

    # --- American imperialism & Roosevelt era ---
    "Category:American imperialism"
    "Category:Panama Canal history"
    "Category:Roosevelt Corollary"
    "Category:Dollar Diplomacy"

    # --- Pre-WWI / Wilson lead-up ---
    "Category:Taft administration"
    "Category:Progressive Era"
    "Category:Mexican Revolution"
)

RESUME_FROM="${1:-}"

mkdir -p data
exec > >(tee -a "$LOG") 2>&1

echo "========================================"
echo "fetch_monroe_wwi.sh started: $(date)"
echo "Output: $OUTPUT"
echo "Total categories: ${#CATEGORIES[@]}"
[ -n "$RESUME_FROM" ] && echo "Resuming from: $RESUME_FROM"
echo "========================================"

TOTAL=${#CATEGORIES[@]}
SKIPPING=true
[ -z "$RESUME_FROM" ] && SKIPPING=false

for i in "${!CATEGORIES[@]}"; do
    CAT="${CATEGORIES[$i]}"
    NUM=$((i + 1))

    if $SKIPPING; then
        if [[ "$NUM" == "$RESUME_FROM" ]] || [[ "$CAT" == *"$RESUME_FROM"* ]]; then
            SKIPPING=false
        else
            echo "  skip [$NUM/$TOTAL] $CAT"
            continue
        fi
    fi

    UA="ner-recovery/0.1 (research; session=${NUM}-of-${TOTAL})"
    echo ""
    echo "[$NUM/$TOTAL] $(date '+%H:%M:%S') — $CAT"
    $FETCHER --category "$CAT" --output "$OUTPUT" --append --max-depth 2 --sleep 1.0 --user-agent "$UA" || {
        echo "  WARNING: fetcher exited non-zero for: $CAT"
    }
done

echo ""
echo "========================================"
echo "All done: $(date)"
LINES=$(wc -l < "$OUTPUT" 2>/dev/null || echo "?")
echo "Total articles in $OUTPUT: $LINES"
echo "========================================"
