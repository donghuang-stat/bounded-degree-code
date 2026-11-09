#!/bin/bash
# =====================================================================================
# Job script for the motif-correlation experiment.  One array task = one shard.
#
#   Normally submitted by slurm/submit.sh.  By hand (from the package directory):
#     sbatch --array=0-799%25 slurm/run_array.sbatch configs/cluster_100cpu_day.json
#
#   MODE (environment variable) selects the step; submit.sh / submit_campaign.sh set it:
#     MODE=array      (default) simulate shard  SLURM_ARRAY_TASK_ID + SHARD_OFFSET
#                     (MAX_PAIRS=k: stop each shard after k new pairs; only for testing resume)
#     MODE=prebuild   build the motif-library cache once (1 CPU, ~1-2 min for sizes 3..9)
#     MODE=aggregate  aggregate.py + plots.py after all shards are done
#   For a campaign (the argument is then campaigns/NAME.json):
#     MODE=monitor    progress report + interim figures every MONITOR_INTERVAL minutes
#                     (monitor.py --loop MONITOR_INTERVAL --plots --until-done)
#     MODE=campaign   final results of every config + campaign figures (monitor.py --final);
#                     cancels the monitor job MONITOR_JOB when it is done
#     MODE=bench      time and memory of one pair per worker on this node (tests/pilot_n5000.sh)
# =====================================================================================
#
# ---- Resource requests (defaults for array tasks) -----------------------------------
# #SBATCH lines are read by sbatch before the script runs, so they cannot use shell
# variables.  Edit the values here, or override them at submission time: options on
# the sbatch command line win over these lines, and slurm/submit.sh passes
# CPUS / ARRAY_TIME / ARRAY_MEM_PER_CPU from the environment if they are set.
#
#   cpus-per-task : worker processes per shard (run_shard.py --workers).  Default 4.
#   time          : wall time of ONE shard.  The plans in README.md size shards for
#                   30-60 min with 4 workers; 2 h leaves room for slower nodes.
#   mem-per-cpu   : memory per worker.  Estimated peak per worker (sizes 3..9):
#                   n=200 ~2 GB, n=300 ~4.5 GB (see README, "Memory"; bench.py measures it).
#
#SBATCH --job-name=motifs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=3G
#SBATCH --output=slurm-%x-%A_%a.out
# Uncomment / add site options as needed (or export SBATCH_ACCOUNT, SBATCH_PARTITION):
##SBATCH --account=YOUR_ACCOUNT
##SBATCH --partition=YOUR_PARTITION

set -euo pipefail

CONFIG=${1:?usage: sbatch slurm/run_array.sbatch CONFIG.json   (or CAMPAIGN.json for MODE=monitor|campaign)}
MODE=${MODE:-array}

# Package directory: exported by submit.sh; otherwise the submission directory
# (under SLURM the script itself runs from a spool copy, so $0 is not useful there).
if [[ -z "${PKG_DIR:-}" ]]; then
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        PKG_DIR=${SLURM_SUBMIT_DIR:-$PWD}
    else
        PKG_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
    fi
fi
if [[ ! -f "$PKG_DIR/run_shard.py" ]]; then
    echo "ERROR: run_shard.py not found in PKG_DIR=$PKG_DIR (submit from the package directory or set PKG_DIR)" >&2
    exit 2
fi

# ################ SITE-SPECIFIC ENVIRONMENT: EDIT THIS BLOCK ###########################
# Make a python3 with numpy, scipy, matplotlib and pynauty available on the compute
# nodes (requirements.txt).  Examples -- uncomment / adapt ONE of them:
#
#   module purge
#   module load python/3.11            # site module name differs between clusters
#   source "$HOME/venvs/motifs/bin/activate"
#
#   source "$HOME/miniconda3/etc/profile.d/conda.sh"
#   conda activate motifs
#
# Nothing site-specific is loaded by default.
# ########################################################################################

# One BLAS/OpenMP thread per worker process: parallelism comes from the worker pool.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
PY=${PYTHON:-python3}

cd "$PKG_DIR"
echo "[$(date '+%F %T')] host=$(hostname) mode=$MODE job=${SLURM_JOB_ID:-none} config=$CONFIG"

case "$MODE" in
    array)
        TASK=${SLURM_ARRAY_TASK_ID:?MODE=array needs SLURM_ARRAY_TASK_ID (submit with --array)}
        SHARD=$(( TASK + ${SHARD_OFFSET:-0} ))
        WORKERS=${SLURM_CPUS_PER_TASK:-1}
        echo "shard=$SHARD (array index $TASK + offset ${SHARD_OFFSET:-0}), workers=$WORKERS"
        "$PY" run_shard.py "$CONFIG" --shard "$SHARD" --workers "$WORKERS" ${MAX_PAIRS:+--max-pairs "$MAX_PAIRS"}
        ;;
    prebuild)
        "$PY" slurm/cfginfo.py prebuild "$CONFIG"
        ;;
    aggregate)
        "$PY" aggregate.py "$CONFIG"
        "$PY" plots.py "$CONFIG"
        ;;
    monitor)
        "$PY" monitor.py "$CONFIG" --loop "${MONITOR_INTERVAL:-30}" --plots --until-done
        ;;
    bench)
        WORKERS=${SLURM_CPUS_PER_TASK:-1}
        "$PY" slurm/cfginfo.py prebuild "$CONFIG"
        "$PY" bench.py --config "$CONFIG" --pairs 1 --workers "$WORKERS" --cpus "$WORKERS" \
            --target-min 30 --cluster-cpus "${CLUSTER_CPUS:-64}"
        ;;
    campaign)
        rc=0
        "$PY" monitor.py "$CONFIG" --final || rc=$?
        if [[ -n "${MONITOR_JOB:-}" ]] && command -v scancel >/dev/null 2>&1; then
            scancel "$MONITOR_JOB" 2>/dev/null || true
        fi
        if (( rc == 3 )); then
            echo "campaign incomplete: interim results written; resubmit the missing shards with"
            echo "    bash slurm/submit_campaign.sh $CONFIG"
        fi
        exit $rc
        ;;
    *)
        echo "ERROR: unknown MODE=$MODE" >&2
        exit 2
        ;;
esac
echo "[$(date '+%F %T')] done"
