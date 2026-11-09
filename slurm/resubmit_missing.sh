#!/bin/bash
# Resubmit only the shards whose output file <out_dir>/shard_KKKKK.npz is missing
# (failed, timed out, preempted, or never ran), then a new aggregation + plots job that
# waits for them (--dependency=afterok).
#
# Usage:  bash slurm/resubmit_missing.sh CONFIG.json [MAX_CONCURRENT]
#
# Shard files are written atomically, so an existing file is always complete.  If jobs of
# this config are still queued or running the script refuses (they may still produce the
# "missing" shards); set FORCE=1 to override.  Without sbatch (or with DRY_RUN=1) it only
# prints the missing indices and the sbatch commands.  NO_AGGREGATE=1 skips the
# aggregation job.  Other knobs (CPUS, ARRAY_TIME, ...) as for submit.sh.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

if [[ $# -lt 1 ]]; then
    sed -n '2,13p' "$0"; exit 2
fi
load_config "$1"
MAXC=${2:-}

N=$("$PY" run_shard.py "$CONFIG" --list)
MISSING=$("$PY" slurm/cfginfo.py missing "$CONFIG" --chunk "$(max_array_size)")
if [[ -z "$MISSING" ]]; then
    echo "all $N shards present in $OUT_DIR"
    echo "aggregate with:  sbatch --chdir=$PKG_DIR --cpus-per-task=1 --mem=${AGG_MEM:-$AGG_MEM_AUTO} --time=$AGG_TIME \\"
    echo "                   --output=$OUT_DIR/logs/%x_%j.out --export=ALL,MODE=aggregate,PKG_DIR=$PKG_DIR \\"
    echo "                   $JOB_SCRIPT $CONFIG"
    echo "     or (small runs) in $PKG_DIR:  python3 aggregate.py $CONFIG && python3 plots.py $CONFIG"
    exit 0
fi
NMISS=$(awk '{s += $3} END {print s}' <<< "$MISSING")
echo "missing shards: $NMISS of $N"
# global shard indices (offset + local index), as a range list
awk '{
    n = split($2, parts, ",");
    for (i = 1; i <= n; i++) {
        m = split(parts[i], ab, "-");
        lo = ab[1] + $1; hi = (m == 2 ? ab[2] : ab[1]) + $1;
        out = out (out == "" ? "" : ",") (lo == hi ? lo : lo "-" hi);
    }
} END { print "missing indices: " out }' <<< "$MISSING"

active=$(squeue_active)
if (( active > 0 )) && [[ "${FORCE:-0}" != 1 ]]; then
    echo "ERROR: $active jobs named $JOB_TAG are still queued/running; wait for them or set FORCE=1" >&2
    exit 1
fi

DEP=""
if [[ "$CACHE_EXISTS" != 1 ]]; then
    echo "WARNING: cache $CACHE_FILE missing; each task builds the library itself" \
         "(run 'python3 slurm/cfginfo.py prebuild $CONFIG' first to avoid that)"
fi
submit_arrays "$DEP" "$MAXC" <<< "$MISSING"
if [[ "${NO_AGGREGATE:-0}" != 1 ]]; then
    submit_aggregate "afterok:$(join_ids : "${ARRAY_IDS[@]}")"
fi
