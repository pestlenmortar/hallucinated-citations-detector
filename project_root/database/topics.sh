#!/bin/bash
# Overnight-safe ingestion driver: continues past individual failures
# Usage: ./topics.sh          (full run)
#        ./topics.sh theory   (run one section by name)

ERROR_LOG="ingest_errors.log"
echo "" > "$ERROR_LOG"   # clear log
FAILED=0
TOTAL=0

run_topic() {
    local query="$1"
    local max_records="$2"
    TOTAL=$((TOTAL + 1))
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

# Systems & Architecture
run_section "Systems & Architecture" "$SECTION"
run_topic "operating systems" 5000
run_topic "distributed systems" 2000
run_topic "computer architecture" 2000
run_topic "cloud computing" 2000
run_topic "storage systems" 2000
run_topic "edge computing" 2000
run_topic "database systems" 2000

# Networking
run_section "Networking" "$SECTION"
run_topic "wireless networks" 2000
run_topic "software-defined networking" 2000
run_topic "internet of things" 2000
run_topic "network protocols" 2000

# Security & Privacy
run_section "Security & Privacy" "$SECTION"
run_topic "cybersecurity" 2000
run_topic "cryptography" 2000
run_topic "network security" 2000
run_topic "privacy" 2000
run_topic "blockchain" 2000
run_topic "intrusion detection" 2000

# Software Engineering
run_section "Software Engineering" "$SECTION"
run_topic "software engineering" 5000
run_topic "program analysis" 2000
run_topic "software testing" 2000
run_topic "compiler design" 2000
run_topic "programming languages" 2000
run_topic "formal verification" 2000
run_topic "debugging" 2000

# Theory
run_section "Theory" "$SECTION"
run_topic "algorithms" 5000
run_topic "data structures" 2000
run_topic "computational complexity" 2000
run_topic "graph theory" 2000
run_topic "formal methods" 2000
run_topic "automata theory" 2000
run_topic "approximation algorithms" 2000

# Graphics & HCI
run_section "Graphics & HCI" "$SECTION"
run_topic "computer graphics" 2000
run_topic "image processing" 2000
run_topic "virtual reality" 2000
run_topic "augmented reality" 2000
run_topic "human-computer interaction" 2000
run_topic "visualization" 2000

# Robotics
run_section "Robotics" "$SECTION"
run_topic "robotics" 5000
run_topic "autonomous systems" 2000
run_topic "control theory" 2000
run_topic "multi-agent systems" 2000
run_topic "path planning" 2000
run_topic "simultaneous localization and mapping" 2000

# Parallel & High-Performance Computing
run_section "Parallel & High-Performance Computing" "$SECTION"
run_topic "parallel computing" 2000
run_topic "high-performance computing" 2000
run_topic "gpu computing" 2000

# Other CS
run_section "Other CS" "$SECTION"
run_topic "information retrieval" 2000
run_topic "recommender systems" 2000
run_topic "social network analysis" 2000
run_topic "computational biology" 2000
run_topic "bioinformatics" 2000
run_topic "quantum computing" 2000
run_topic "web search" 2000

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
