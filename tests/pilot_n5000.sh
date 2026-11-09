#!/bin/bash
# =====================================================================================
# Pilot for the n = 5000 campaign: measure, on a compute node of YOUR cluster, the time
# and memory of one graph pair with the same resources as an array task, before
# submitting 1200 shards.  One job, CPUS workers, one pair per worker (2-10 minutes).
#
#     bash tests/pilot_n5000.sh                        # CPUS=4, MEM_PER_CPU=5G
#     SBATCH_EXTRA="--account=abc --partition=xyz" bash tests/pilot_n5000.sh
#
# It uses slurm/run_array.sbatch (MODE=bench), so the site environment block there applies.
# Reference (one core of the development machine): 117 s per pair, peak 2.9 GB per worker;
# the campaign (24,000 pairs) is then 780 CPU hours, about 12 h on 64 CPUs.
# =====================================================================================
set -euo pipefail
if [[ $# -ne 0 ]]; then
    echo "Usage: CONFIG=configs/n5000_p05.json CPUS=4 PYTHON=python3 bash tests/pilot_n5000.sh" >&2
    exit 2
fi
PY=${PYTHON:-python3}
if [[ "$PY" == */* && "$PY" != /* ]]; then
    PY=$(cd "$(dirname "$PY")" && pwd)/$(basename "$PY")
fi
if ! command -v "$PY" >/dev/null 2>&1; then
    echo "ERROR: Python interpreter not found: $PY" >&2
    exit 2
fi
export PYTHON="$PY"
PKG_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PKG_DIR"
CPUS=${CPUS:-4}; MEM_PER_CPU=${MEM_PER_CPU:-5G}
if [[ ! "$CPUS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: CPUS must be a positive integer" >&2
    exit 2
fi
# CONFIG is absolute or relative to the package root, independent of the caller.
CONFIG=${CONFIG:-configs/n5000_p05.json}
if [[ "$CONFIG" != /* ]]; then CONFIG="$PKG_DIR/$CONFIG"; fi
if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: configuration file not found: $CONFIG" >&2
    exit 2
fi
CONFIG=$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")
OUT_DIR="$PKG_DIR/out/pilot_n5000"
mkdir -p "$OUT_DIR"
EXTRA=()
read -r -a EXTRA <<< "${SBATCH_EXTRA:-}"
if ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch not found; running the pilot here instead"
    "$PY" slurm/cfginfo.py prebuild "$CONFIG"
    "$PY" bench.py --config "$CONFIG" --pairs 1 --workers "$CPUS" --cpus "$CPUS" --target-min 30 --cluster-cpus 64
    exit 0
fi
echo "submitting the pilot job and waiting for it (sbatch --wait) ..."
# Export values directly: commas/spaces in paths need not pass through --export's
# comma-separated assignment syntax.  Preserve the selected interpreter remotely.
export MODE=bench PKG_DIR
error_file=$(mktemp "${TMPDIR:-/tmp}/motif-pilot-sbatch.XXXXXX")
trap 'rm -f "$error_file"' EXIT
submit_status=0
submit_output=$(sbatch --parsable --wait ${EXTRA[@]+"${EXTRA[@]}"} \
    --job-name=mot_pilot_n5000 --chdir="$PKG_DIR" --output="$OUT_DIR/pilot_%j.out" \
    --cpus-per-task="$CPUS" --mem-per-cpu="$MEM_PER_CPU" --time=00:30:00 \
    --export=ALL "$PKG_DIR/slurm/run_array.sbatch" "$CONFIG" 2>"$error_file") || submit_status=$?
if [[ -s "$error_file" ]]; then cat "$error_file" >&2; fi
job_id=${submit_output%%;*}
if [[ ! "$job_id" =~ ^[0-9]+$ || "$submit_output" == *$'\n'* \
      || ( "$submit_output" != "$job_id" && "$submit_output" != "$job_id;"?* ) ]]; then
    echo "ERROR: sbatch did not return a valid --parsable job ID: $submit_output" >&2
    if [[ "$submit_status" -ne 0 ]]; then exit "$submit_status"; fi
    exit 1
fi
printf 'Pilot job: %s\n' "$submit_output"
# Only inspect the log for this submission, never a previous or latest log.
log_file="$OUT_DIR/pilot_$job_id.out"
if [[ -f "$log_file" ]]; then
    if ! cat "$log_file"; then
        echo "ERROR: could not read pilot log: $log_file" >&2
        if [[ "$submit_status" -ne 0 ]]; then exit "$submit_status"; fi
        exit 1
    fi
else
    echo "ERROR: pilot log not found for submitted job $job_id: $log_file" >&2
    if [[ "$submit_status" -ne 0 ]]; then exit "$submit_status"; fi
    exit 1
fi
if [[ "$submit_status" -ne 0 ]]; then
    echo "ERROR: pilot job $job_id failed (sbatch --wait exit $submit_status)" >&2
    exit "$submit_status"
fi
cat <<EOT

The campaign has 4 configs like this one (24,000 pairs): multiply the projected CPU hours
by 4.  A shard of 20 pairs with CPUS=$CPUS workers takes about 20 x (seconds per pair) / $CPUS;
set ARRAY_TIME to at least 3 times that and ARRAY_MEM_PER_CPU above the measured peak
(defaults in the README: 01:00:00 and 5G).
EOT
