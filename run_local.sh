#!/bin/bash
# Run every shard of a config on this machine (W worker processes).
# Usage: bash run_local.sh CONFIG.json [W]
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: bash run_local.sh CONFIG.json [WORKERS]" >&2
    exit 2
fi
CFG=$1
W=${2:-2}
if [[ ! "$W" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: WORKERS must be a positive integer" >&2
    exit 2
fi
if [[ ! -f "$CFG" ]]; then
    echo "ERROR: configuration file not found: $CFG" >&2
    exit 2
fi
# Resolve caller-relative paths before changing to the package directory.
CFG=$(cd "$(dirname "$CFG")" && pwd)/$(basename "$CFG")
PY=${PYTHON:-python3}
if [[ "$PY" == */* && "$PY" != /* ]]; then
    PY=$(cd "$(dirname "$PY")" && pwd)/$(basename "$PY")
fi
if ! command -v "$PY" >/dev/null 2>&1; then
    echo "ERROR: Python interpreter not found: $PY" >&2
    exit 2
fi
export PYTHON="$PY"
PKG_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$PKG_DIR"
N=$("$PY" "$PKG_DIR/run_shard.py" "$CFG" --list)
if [[ ! "$N" =~ ^[0-9]+$ ]]; then
    echo "ERROR: run_shard.py --list did not return a nonnegative shard count: $N" >&2
    exit 1
fi
for ((k=0; k<N; k++)); do
    "$PY" "$PKG_DIR/run_shard.py" "$CFG" --shard "$k" --workers "$W"
done
