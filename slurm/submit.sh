#!/bin/bash
# Submit a whole experiment:  prebuild (motif-library cache)  ->  job array (one task per
# shard)  ->  aggregation + plots.
#
# Usage:  bash slurm/submit.sh CONFIG.json [MAX_CONCURRENT]
#
#   MAX_CONCURRENT   maximum number of array tasks running at once (--array=...%MAX).
#                    For a budget of C CPUs and CPUS cpus per task use C / CPUS
#                    (e.g. 100 CPUs, 4 per task -> 25).  Omit for no throttle.
#                    Change it later with:  scontrol update JobId=<array id> ArrayTaskThrottle=<k>
#
# Environment (optional; see slurm/common.sh for the full list):
#   CPUS=4 ARRAY_TIME=02:00:00 ARRAY_MEM_PER_CPU=3G   override the #SBATCH defaults of the array
#   PREBUILD=job     (default) build the cache in a small 1-CPU job; the array waits for it
#                    (--dependency=afterok); skipped if the cache file already exists
#   PREBUILD=login   build it right here on the login node (about 1 min CPU for sizes 3..9)
#   PREBUILD=skip    no prebuild (every task builds the cache itself; safe but wasteful)
#   PREBUILD_TIME=00:30:00 PREBUILD_MEM=4G
#   AGG_TIME=04:00:00 AGG_MEM=<estimated from the config>
#   SBATCH_EXTRA="--account=... --partition=..."   added to every sbatch call
#   DRY_RUN=1        only print the sbatch commands (automatic when sbatch is missing)
#
# Run it from a login shell in which `python3` has numpy (and pynauty for PREBUILD=login).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

if [[ $# -lt 1 ]]; then
    sed -n '2,25p' "$0"; exit 2
fi
load_config "$1"
MAXC=${2:-}
PREBUILD=${PREBUILD:-job}
PREBUILD_TIME=${PREBUILD_TIME:-00:30:00}
PREBUILD_MEM=${PREBUILD_MEM:-4G}

# number of shards, as computed by run_shard.py itself
N=$("$PY" run_shard.py "$CONFIG" --list)
if [[ "$N" != "$NSHARDS" ]]; then
    echo "ERROR: run_shard.py --list says $N shards, cfginfo.py says $NSHARDS" >&2; exit 1
fi
echo "config   : $CONFIG"
echo "package  : $PKG_DIR"
echo "out_dir  : $OUT_DIR   (logs in $OUT_DIR/logs)"
echo "shards   : $N   ($NPAIRS pairs, $NMOTIFS motifs, raw output ~$RAW_GB GB)"

have=$(find "$OUT_DIR" -maxdepth 1 -name 'shard_*.npz' ! -name '*.tmp.npz' 2>/dev/null | wc -l)
if (( have > 0 )); then
    echo "NOTE: $have shard files already exist; their tasks exit immediately."
    echo "      To submit only the missing shards use slurm/resubmit_missing.sh."
fi
active=$(squeue_active)
if (( active > 0 )) && [[ "${FORCE:-0}" != 1 ]]; then
    echo "ERROR: $active jobs named $JOB_TAG are already queued/running; set FORCE=1 to submit anyway" >&2
    exit 1
fi

# ---- (a) prebuild the motif-library cache ---------------------------------------------
DEP=""
if [[ "$CACHE_EXISTS" == 1 ]]; then
    echo "prebuild : cache $CACHE_FILE exists, nothing to do"
elif [[ "$PREBUILD" == login ]]; then
    echo "WARNING: building the motif library on the LOGIN node (sizes 3..9: ~1 min CPU, <1 GB)."
    echo "         Use PREBUILD=job if your site forbids computations on login nodes."
    nice -n 19 "$PY" slurm/cfginfo.py prebuild "$CONFIG"
elif [[ "$PREBUILD" == job ]]; then
    submit_job PRE_ID --job-name="${JOB_TAG}_prebuild" --chdir="$PKG_DIR" \
        --output="$OUT_DIR/logs/%x_%j.out" --cpus-per-task=1 --time="$PREBUILD_TIME" \
        --mem="$PREBUILD_MEM" ${EXTRA[@]+"${EXTRA[@]}"} \
        --export="ALL,MODE=prebuild,PKG_DIR=$PKG_DIR" "$JOB_SCRIPT" "$CONFIG"
    echo "prebuild job $PRE_ID"
    DEP="afterok:$PRE_ID"
elif [[ "$PREBUILD" != skip ]]; then
    echo "ERROR: PREBUILD must be job, login or skip" >&2; exit 2
fi

# ---- (b) the job array ----------------------------------------------------------------
CHUNKS=$("$PY" slurm/cfginfo.py chunks "$N" "$(max_array_size)")
submit_arrays "$DEP" "$MAXC" <<< "$CHUNKS"

# ---- (c) aggregation + plots after every array task succeeded ---------------------------
submit_aggregate "afterok:$(join_ids : "${ARRAY_IDS[@]}")"

cat <<EOF

Monitor:   squeue -u \$USER -n $JOB_TAG,${JOB_TAG}_agg,${JOB_TAG}_prebuild
           sacct -j $(join_ids , "${ARRAY_IDS[@]}") -X --format=JobID,State,Elapsed,MaxRSS
Progress:  ls $OUT_DIR/shard_*.npz | wc -l     (of $N)
If tasks fail or time out, the aggregation job stays pending (DependencyNeverSatisfied):
           scancel $AGG_ID;  bash slurm/resubmit_missing.sh $CONFIG ${MAXC}
EOF
