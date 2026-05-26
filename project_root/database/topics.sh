#!/bin/bash
# Overnight-safe ingestion driver: continues past individual failures
# Uses OpenAlex free API (no API key required)
# All CS-related topics excluded — only non-CS engineering domains ingested.
# Usage: ./topics.sh          (full run)
#        ./topics.sh energy   (run one section by name)

ERROR_LOG="ingest_errors.log"
echo "" > "$ERROR_LOG"   # clear log
FAILED=0
TOTAL=0

topic_covered() {
    local query="$1"
    local max_records="$2"
    local pattern=$(echo "$query" | sed 's/ /%/g')
    local count=$(sqlite3 papers.db "SELECT COUNT(*) FROM papers WHERE normalized_title LIKE '%${pattern}%'")
    [ "$count" -ge "$max_records" ]
}

run_topic() {
    local query="$1"
    local max_records="$2"
    TOTAL=$((TOTAL + 1))
    if topic_covered "$query" "$max_records"; then
        echo "[$TOTAL] $query (max $max_records) — SKIP (already enough papers)"
        return
    fi
    rm -f ingest_offset.txt ingest_cursor.txt
    echo "[$TOTAL] $query (max $max_records)"
    if python ingest_openalex.py "$query" --max-records "$max_records"; then
        echo "    OK"
    else
        echo "    *** FAILED ***"
        echo "FAILED: $query" >> "$ERROR_LOG"
        FAILED=$((FAILED + 1))
    fi
}

run_section() {
    local section="$1"
    if [ -n "$2" ] && [ "$section" != "$2" ]; then
        return
    fi
    echo ""
    echo "======== $section ========"
}

SECTION="${1:-}"

# Civil Engineering
run_section "Civil Engineering" "$SECTION"
run_topic "structural engineering" 2000
run_topic "transportation engineering" 2000
run_topic "geotechnical engineering" 2000

# Mechanical Engineering
run_section "Mechanical Engineering" "$SECTION"
run_topic "thermodynamics" 2000
run_topic "fluid mechanics" 2000
run_topic "finite element analysis" 2000
run_topic "computational mechanics" 2000
run_topic "propulsion" 2000

# Materials Science & Metallurgy
run_section "Materials Science & Metallurgy" "$SECTION"
run_topic "metallurgy" 2000

# Electrical Engineering
run_section "Electrical Engineering" "$SECTION"
run_topic "power electronics" 2000
run_topic "power systems" 2000
run_topic "vlsi" 2000
run_topic "signal processing" 2000
run_topic "instrumentation engineering" 2000
run_topic "communication systems" 2000
run_topic "embedded systems" 2000

# Energy & Chemical Engineering
run_section "Energy & Chemical Engineering" "$SECTION"
run_topic "battery systems" 2000
run_topic "renewable energy" 2000

echo ""
echo "========================================"
echo "All topics finished."
echo "  Total: $TOTAL   Failed: $FAILED"
if [ "$FAILED" -gt 0 ]; then
    echo "  Failures logged in: $ERROR_LOG"
    echo "  Re-run failed topics manually:"
    cat "$ERROR_LOG"
fi
echo "========================================"
