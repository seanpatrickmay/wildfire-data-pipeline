#!/usr/bin/env bash
# Download all 13 fires with the updated pipeline.
# Runs 2 fires in parallel to balance GEE rate limits vs speed.

set -euo pipefail
cd "$(dirname "$0")/.."

FIRES=(
    Kincade Walker CampFire GlassFire CaldorFire MosquitoFire
    SmithRiverComplex DixieFire BobcatFire AugustComplexFire
    CreekFire NorthComplexFire DolanFire
)

PARALLEL=2
LOG_DIR="data/logs"
mkdir -p "$LOG_DIR"

echo "Starting download of ${#FIRES[@]} fires (${PARALLEL} parallel)..."
echo "Logs: $LOG_DIR/"

download_fire() {
    local fire=$1
    echo "[$(date +%H:%M:%S)] Starting $fire"
    .venv/bin/python -m wildfire_pipeline.cli download "$fire" \
        --config config/fires.json --output data \
        > "$LOG_DIR/${fire}.log" 2>&1
    local status=$?
    if [ $status -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] DONE: $fire"
    else
        echo "[$(date +%H:%M:%S)] FAILED: $fire (exit $status)"
    fi
    return $status
}

export -f download_fire

# GNU parallel if available, otherwise xargs
if command -v parallel &> /dev/null; then
    printf '%s\n' "${FIRES[@]}" | parallel -j "$PARALLEL" download_fire {}
else
    printf '%s\n' "${FIRES[@]}" | xargs -P "$PARALLEL" -I {} bash -c 'download_fire "$@"' _ {}
fi

echo ""
echo "All downloads complete. Run 'wildfire process' on each fire next."
