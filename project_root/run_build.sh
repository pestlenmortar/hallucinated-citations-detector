#!/usr/bin/env bash
set -o pipefail

BUILD_PID=""
MONITOR_LOG="build_monitor.log"
GPU_LOG="gpu_monitor.log"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DB_PATH="${SCRIPT_DIR}/database/papers.db"
INDEX_OUT="${SCRIPT_DIR}/faiss_index.bin"

cleanup() {
    echo ""
    echo "[MONITOR] Cleaning up ..."
    if [[ -n "$BUILD_PID" ]] && kill -0 "$BUILD_PID" 2>/dev/null; then
        kill "$BUILD_PID" 2>/dev/null
        wait "$BUILD_PID" 2>/dev/null
    fi
    echo "[MONITOR] Exiting."
}
trap cleanup SIGINT SIGTERM EXIT

echo "=============================================="
echo " FAISS Index Build — Continuous Monitor"
echo "=============================================="
echo "DB:       $DB_PATH"
echo "Index:    ${INDEX_OUT}"
echo "Start:    $(date)"
echo "=============================================="
echo ""

# Quick sanity checks
if [[ ! -f "$DB_PATH" ]]; then
    echo "ERROR: Database not found at $DB_PATH"
    exit 1
fi

DB_SIZE=$(stat -c%s "$DB_PATH" 2>/dev/null || stat -f%z "$DB_PATH" 2>/dev/null)
echo "Database size: $(numfmt --to=iec-i --suffix=B $DB_SIZE 2>/dev/null || echo "$DB_SIZE bytes")"
echo ""

# Start the build in background
echo "[MONITOR] Starting FAISS index build ..."
python3 -u "${SCRIPT_DIR}/embeddings/build_index.py" \
    --db "$DB_PATH" \
    --index "$INDEX_OUT" \
    2>&1 | tee build_output.log &
BUILD_PID=$!

echo "[MONITOR] Build PID: $BUILD_PID"
echo ""

# Start GPU monitor in background
(
    while kill -0 "$BUILD_PID" 2>/dev/null; do
        echo "--- $(date +%H:%M:%S) ---" >> "$GPU_LOG"
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits >> "$GPU_LOG" 2>&1
        sleep 10
    done
) &
GPU_MON_PID=$!

# Start system monitor in background
(
    echo "timestamp,cpu_pct,ram_used_mb,ram_total_mb,swap_used_mb,disk_avail_gb" > "$MONITOR_LOG"
    while kill -0 "$BUILD_PID" 2>/dev/null; do
        TIMESTAMP=$(date +%H:%M:%S)
        CPU=$(python3 -c "import psutil; print(psutil.cpu_percent(interval=0.5))" 2>/dev/null || echo "?")
        MEM=$(python3 -c "
import psutil
m = psutil.virtual_memory()
s = psutil.swap_memory()
print(f'{m.used/1024/1024:.0f},{m.total/1024/1024:.0f},{s.used/1024/1024:.0f}')
" 2>/dev/null || echo "?,?,?")
        DISK=$(python3 -c "
import shutil
_, _, free = shutil.disk_usage('${SCRIPT_DIR}')
print(f'{free/1024/1024/1024:.1f}')
" 2>/dev/null || echo "?")
        echo "${TIMESTAMP},${CPU},${MEM},${DISK}" >> "$MONITOR_LOG"
        sleep 5
    done
) &
SYS_MON_PID=$()

# Wait for build to complete
echo "[MONITOR] Monitoring active — watching CPU, RAM, GPU, disk ..."
echo "[MONITOR] Logs: ${MONITOR_LOG} (system), ${GPU_LOG} (GPU), build_output.log (build)"
echo ""

wait "$BUILD_PID"
EXIT_CODE=$?

# Stop monitors
kill "$GPU_MON_PID" "$SYS_MON_PID" 2>/dev/null
wait "$GPU_MON_PID" "$SYS_MON_PID" 2>/dev/null

echo ""
echo "=============================================="
echo " BUILD COMPLETED"
echo " Exit code: $EXIT_CODE"
echo " End time:  $(date)"
echo "=============================================="

if [[ $EXIT_CODE -eq 0 ]]; then
    echo ""
    echo "Output files:"
    ls -lh "${INDEX_OUT}" "${INDEX_OUT}.mapping.json" 2>/dev/null
    echo ""
    echo "=== Peak Resource Usage ==="
    echo "--- System (from ${MONITOR_LOG}) ---"
    column -t -s, "${MONITOR_LOG}" 2>/dev/null || cat "${MONITOR_LOG}"
    echo ""
    echo "--- GPU (from ${GPU_LOG}) ---"
    cat "${GPU_LOG}"
else
    echo ""
    echo "BUILD FAILED with exit code $EXIT_CODE"
    echo "Check build_output.log for errors."
    echo ""
    echo "=== Last 30 lines of build output ==="
    tail -30 build_output.log
fi

exit $EXIT_CODE
