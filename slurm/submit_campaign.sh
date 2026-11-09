#!/bin/bash
# Submit (or resubmit) a whole campaign: several configs (one per edge probability p) that
# share a CPU budget, plus a progress monitor and a final results job.
#
# Usage:  bash slurm/submit_campaign.sh CAMPAIGN.json [TOTAL_MAX_CONCURRENT]
#         e.g.  CPUS=4 bash slurm/submit_campaign.sh campaigns/n5000.json 16    (64 CPUs)
#
# What it submits:
#   (a) prebuild   build the motif-library cache once (skipped if it exists)
#   (b) arrays     one job array per config, ONLY for shards whose shard_KKKKK.npz is missing
#                  (so the same command resubmits after timeouts, failures or preemption;
#                  unfinished shards resume from their per-pair checkpoints).  The throttle
#                  TOTAL_MAX_CONCURRENT is split evenly between the configs, so all p values
#                  progress at the same pace and interim figures cover every p.
#   (c) monitor    1 CPU; every MONITOR_INTERVAL minutes: progress report + interim figures
#                  (monitor.py --loop MONITOR_INTERVAL --plots --until-done)
#   (d) campaign   after all arrays ended (afterany): final analysis + figures of every
#                  config and of the campaign, manifest; cancels the monitor.  If shards are
#                  missing it writes interim results and exits with status 3.
#
# Environment (optional; see slurm/common.sh for the rest):
#   CPUS=4 ARRAY_TIME=02:00:00 ARRAY_MEM_PER_CPU=5G   resources of one shard (array task)
#   MONITOR=job|none (default job)   MONITOR_INTERVAL=30 (minutes)   MONITOR_TIME=1-00:00:00
#                    (if the monitor job is refused or ends, `python3 monitor.py CAMPAIGN.json
#                    --plots` on the login node does the same, or resubmit only the monitor:
#                    sbatch -c 1 --mem=4G -t 1-00:00:00 --export=ALL,MODE=monitor,PKG_DIR=$PWD \
#                           slurm/run_array.sbatch campaigns/NAME.json)
#   FINAL_TIME=02:00:00 FINAL_MEM=8G
#   MAX_PAIRS=k       testing only: every shard stops after k new pairs (interrupted run)
#   PREBUILD=job|login|skip, SBATCH_EXTRA="--account=... --partition=...", DRY_RUN=1, FORCE=1
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

if [[ $# -lt 1 ]]; then
    sed -n '2,27p' "$0"; exit 2
fi
CAMPAIGN=$(abs_config "$1")
TOTAL_MAXC=${2:-}
cd "$PKG_DIR"
eval "$("$PY" slurm/cfginfo.py campaign "$CAMPAIGN")"
NCFG=${#CONFIGS[@]}
PREBUILD=${PREBUILD:-job}
MONITOR=${MONITOR:-job}
MONITOR_INTERVAL=${MONITOR_INTERVAL:-30}
MONITOR_TIME=${MONITOR_TIME:-1-00:00:00}
FINAL_TIME=${FINAL_TIME:-02:00:00}
FINAL_MEM=${FINAL_MEM:-8G}
PER=""
if [[ -n "$TOTAL_MAXC" ]]; then
    PER=$(( TOTAL_MAXC / NCFG )); (( PER >= 1 )) || PER=1
fi
mkdir -p "$CAMP_OUT/logs"
echo "campaign : $CAMP_NAME ($CAMPAIGN)"
echo "out_dir  : $CAMP_OUT"
echo "configs  : $NCFG; at most ${TOTAL_MAXC:-unlimited} shards at once (${PER:-unlimited} per config)," \
     "${CPUS:-4 (#SBATCH default)} CPUs each"
[[ -n "${MAX_PAIRS:-}" ]] && echo "TEST MODE: MAX_PAIRS=$MAX_PAIRS (shards stop early; resubmit to finish)"

# ---- refuse if jobs of this campaign are still queued or running -------------------------
if command -v squeue >/dev/null 2>&1 && [[ "${FORCE:-0}" != 1 ]]; then
    for c in "${CONFIGS[@]}"; do
        load_config "$c" >/dev/null
        a=$(squeue_active)
        if (( a > 0 )); then
            echo "ERROR: $a jobs named $JOB_TAG are queued/running; wait, scancel them, or set FORCE=1" >&2
            exit 1
        fi
    done
fi

# ---- (a) prebuild the motif-library cache (shared by all configs with the same sizes) -----
DEP=""
load_config "${CONFIGS[0]}"
if [[ "$CACHE_EXISTS" == 1 ]]; then
    echo "prebuild : cache $CACHE_FILE exists"
elif [[ "$PREBUILD" == login ]]; then
    nice -n 19 "$PY" slurm/cfginfo.py prebuild "$CONFIG"
elif [[ "$PREBUILD" == job ]]; then
    OUT_DIR=$CAMP_OUT
    submit_job PRE_ID --job-name="mot_${CAMP_NAME}_prebuild" --chdir="$PKG_DIR" \
        --output="$CAMP_OUT/logs/%x_%j.out" --cpus-per-task=1 --time="${PREBUILD_TIME:-00:30:00}" \
        --mem="${PREBUILD_MEM:-4G}" ${EXTRA[@]+"${EXTRA[@]}"} \
        --export="ALL,MODE=prebuild,PKG_DIR=$PKG_DIR" "$JOB_SCRIPT" "$CONFIG"
    echo "prebuild job $PRE_ID"
    DEP="afterok:$PRE_ID"
fi

# ---- (b) one job array per config, missing shards only ----------------------------------
ALL_ARRAYS=()
TOTAL_MISSING=0
for c in "${CONFIGS[@]}"; do
    load_config "$c"
    N=$("$PY" run_shard.py "$CONFIG" --list)
    MISSING=$("$PY" slurm/cfginfo.py missing "$CONFIG" --chunk "$(max_array_size)")
    NMISS=$(awk '{s += $3} END {print s + 0}' <<< "$MISSING")
    TOTAL_MISSING=$((TOTAL_MISSING + NMISS))
    echo "--- $(basename "$CONFIG"): $NMISS of $N shards to run ($NPAIRS pairs in total), logs in $OUT_DIR/logs"
    if (( NMISS == 0 )); then
        continue
    fi
    submit_arrays "$DEP" "$PER" <<< "$MISSING"
    ALL_ARRAYS+=("${ARRAY_IDS[@]}")
done

OUT_DIR=$CAMP_OUT
if (( TOTAL_MISSING == 0 )); then
    echo "all shards of all configs are present; submitting only the final results job"
fi

# ---- (c) monitor -------------------------------------------------------------------------
MON_ID=""
if [[ "$MONITOR" == job ]] && (( TOTAL_MISSING > 0 )); then
    if submit_job MON_ID --job-name="mot_${CAMP_NAME}_monitor" --chdir="$PKG_DIR" \
        --output="$CAMP_OUT/logs/%x_%j.out" --cpus-per-task=1 --time="$MONITOR_TIME" --mem=4G \
        ${EXTRA[@]+"${EXTRA[@]}"} \
        --export="ALL,MODE=monitor,PKG_DIR=$PKG_DIR,MONITOR_INTERVAL=$MONITOR_INTERVAL" \
        "$JOB_SCRIPT" "$CAMPAIGN"; then
        echo "monitor job $MON_ID (every $MONITOR_INTERVAL min, log $CAMP_OUT/logs/mot_${CAMP_NAME}_monitor_$MON_ID.out)"
    else
        MON_ID=""
        echo "WARNING: the monitor job was not accepted (MONITOR_TIME=$MONITOR_TIME too long for the partition?)."
        echo "         The shards run anyway; use  python3 monitor.py $CAMPAIGN --plots  instead."
    fi
fi

# ---- (d) final results after all arrays ended ---------------------------------------------
FDEP=""
if (( ${#ALL_ARRAYS[@]} > 0 )); then
    FDEP="afterany:$(join_ids : "${ALL_ARRAYS[@]}")"
fi
submit_job FIN_ID --job-name="mot_${CAMP_NAME}_final" --chdir="$PKG_DIR" \
    --output="$CAMP_OUT/logs/%x_%j.out" --cpus-per-task=1 --time="$FINAL_TIME" --mem="$FINAL_MEM" \
    ${FDEP:+--dependency="$FDEP"} ${EXTRA[@]+"${EXTRA[@]}"} \
    --export="ALL,MODE=campaign,PKG_DIR=$PKG_DIR,MONITOR_JOB=$MON_ID" "$JOB_SCRIPT" "$CAMPAIGN"
echo "final results job $FIN_ID (dependency ${FDEP:-none})"

cat <<EOF

Watch progress
  python3 monitor.py $CAMPAIGN                 # status table (cheap; fine on a login node)
  python3 monitor.py $CAMPAIGN --plots         # + interim figures now
  tail -f $CAMP_OUT/progress.log               # written by the monitor job every $MONITOR_INTERVAL min
  tail -f <config out_dir>/logs/*.out          # per-pair progress lines of each shard
  squeue -u \$USER -n $(for c in "${CONFIGS[@]}"; do printf 'mot_%s,' "$(basename "${c%.json}")"; done)mot_${CAMP_NAME}_monitor,mot_${CAMP_NAME}_final
Interim figures : $CAMP_OUT/analysis_partial/   (and <config out_dir>/analysis_partial/)
Final results   : $CAMP_OUT/analysis/           (and <config out_dir>/analysis/)
After timeouts or failures, run the same command again: only missing shards are submitted,
and unfinished shards resume from their checkpoints.
EOF
