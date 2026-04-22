#!/usr/bin/env bash
# Fetch the full WWI political/military/colonial ecosystem from Wikipedia.
# All categories append into a single JSONL; already-fetched titles are skipped.
# Usage: bash scripts/fetch_wwi_extended.sh
# Output: data/wwi_extended.jsonl
# Log:    data/wwi_extended_fetch.log

set -euo pipefail

OUTPUT="data/wwi_extended.jsonl"
LOG="data/wwi_extended_fetch.log"
FETCHER="uv run python fetcher.py"

CATEGORIES=(
    # --- The spark & pre-war ---
    "Category:Assassination of Archduke Franz Ferdinand"
    "Category:July Crisis"
    "Category:Black Hand (Serbia)"
    "Category:Balkan Wars"
    "Category:Pan-Slavism"
    "Category:Young Bosnia"

    # --- Central Powers political leadership ---
    "Category:Politicians of the German Empire"
    "Category:Austria-Hungary politicians"
    "Category:Young Turks"
    "Category:Committee of Union and Progress"
    "Category:Bulgarian World War I military personnel"

    # --- Ottoman theater ---
    "Category:Ottoman Empire in World War I"
    "Category:Partitioning of the Ottoman Empire"
    "Category:Arab Revolt"

    # --- Entente political leadership ---
    "Category:Russian Empire politicians"
    "Category:French Third Republic politicians"
    "Category:Liberal Party (UK) politicians"
    "Category:Conservative Party (UK) politicians 1910–1945"

    # --- WWI commanders (all sides) ---
    "Category:World War I commanders"
    "Category:World War I military personnel by country"

    # --- USA ---
    "Category:Wilson administration"
    "Category:United States in World War I"

    # --- Colonial & dominion theaters ---
    "Category:Australian World War I military personnel"
    "Category:Canadian World War I military personnel"
    "Category:New Zealand World War I military personnel"
    "Category:South African World War I military personnel"
    "Category:Indian Army during World War I"
    "Category:Gallipoli campaign"
    "Category:Mesopotamian campaign"
    "Category:East African campaign (World War I)"
    "Category:West African campaign (World War I)"

    # --- Asia & Pacific ---
    "Category:China in World War I"
    "Category:Japan in World War I"
    "Category:Siege of Tsingtao"

    # --- Revolution & aftermath ---
    "Category:Russian Revolution"
    "Category:February Revolution"
    "Category:October Revolution"
    "Category:Paris Peace Conference, 1919"
    "Category:Treaty of Versailles"
    "Category:League of Nations"
    "Category:Aftermath of World War I"
)

RESUME_FROM="${1:-}"  # optional: category name or number to resume from

mkdir -p data
exec > >(tee -a "$LOG") 2>&1

echo "========================================"
echo "fetch_wwi_extended.sh started: $(date)"
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

    # Resume logic: match by number (e.g. "10") or category name substring
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
