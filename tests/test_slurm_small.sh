#!/bin/bash
# =====================================================================================
# Small end-to-end test ON THE CLUSTER, through SLURM.  About 10-30 minutes including
# queueing and about 1 CPU-hour.  Run it on a login node, from the package directory:
#
#     bash tests/test_slurm_small.sh [CAMPAIGN.json]      (default campaigns/test_slurm.json)
#     SBATCH_EXTRA="--account=abc --partition=xyz" bash tests/test_slurm_small.sh
#
# Without a cluster, the same orchestration runs against a stand-in for sbatch/squeue/scancel
# (jobs run on this machine; dependencies, arrays, throttles, exports and scancel emulated):
#     PATH=$PWD/tests/mock_slurm:$PATH bash tests/test_slurm_small.sh campaigns/test_small.json
#
# Campaign campaigns/test_slurm.json: the n = 5000 design (4 values of p, 13 rho values,
# motifs with 3 to 5 edges, interleaved shards) at n = 1000 with 300 pairs per p, 25
# shards of 12 pairs per p.  It runs exactly the scripts of the real campaign:
#
#   stage 1 "interrupted run": submit_campaign.sh with MAX_PAIRS=8, so every shard stops
#           after 8 pairs.  Expected: per-pair progress lines in the shard logs, the monitor
#           job writes progress.log and interim figures, the final job finds the campaign
#           incomplete (exit 3), writes interim results and cancels the monitor;
#   stage 2 "resubmission": the same submit_campaign.sh without MAX_PAIRS.  Only missing
#           shards are submitted, they resume from their checkpoints, the final job writes
#           the final figures and tables;
#   then tests/check_outputs.py.
#
# Knobs: MAXC (concurrent shards, default 16), CPUS (default 4), ARRAY_TIME, ARRAY_MEM_PER_CPU,
# WAIT_MAX (seconds per stage, default 7200), SBATCH_EXTRA, PYTHON.
# =====================================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=${PYTHON:-python3}
export PYTHON="$PY"
# Give each mock test invocation its own scheduler database. A caller can supply
# MOCK_SLURM_DIR explicitly to inspect or reuse a particular sandbox.
SBATCH_BIN=$(command -v sbatch || true)
if [[ -z ${MOCK_SLURM_DIR:-} && -f "$SBATCH_BIN" ]] \
    && [[ $(cd "$(dirname "$SBATCH_BIN")" && pwd -P) == $(cd tests/mock_slurm && pwd -P) ]]; then
    MOCK_SLURM_DIR=$(mktemp -d "${TMPDIR:-/tmp}/motif-mock-slurm.XXXXXX")
    export MOCK_SLURM_DIR
    cleanup_mock() {
        "$PY" tests/mock_slurm/mockslurm.py cleanup || true
        rm -rf "$MOCK_SLURM_DIR"
    }
    trap cleanup_mock EXIT
fi
CAMP=${1:-campaigns/test_slurm.json}
MAXC=${MAXC:-16}
export CPUS=${CPUS:-4} ARRAY_TIME=${ARRAY_TIME:-00:15:00} ARRAY_MEM_PER_CPU=${ARRAY_MEM_PER_CPU:-2G}
export MONITOR_INTERVAL=${MONITOR_INTERVAL:-2} MONITOR_TIME=${MONITOR_TIME:-02:00:00}
export FINAL_TIME=${FINAL_TIME:-00:30:00} FINAL_MEM=${FINAL_MEM:-4G} PREBUILD_TIME=${PREBUILD_TIME:-00:15:00}
eval "$("$PY" slurm/cfginfo.py campaign "$CAMP")"       # CAMP_NAME, CAMP_OUT, CONFIGS
COUT=$CAMP_OUT
CFG_OUTS=(); NAMES=""
for c in "${CONFIGS[@]}"; do
    eval "$("$PY" slurm/cfginfo.py info "$c" | grep -E '^(OUT_DIR|JOB_TAG)=')"
    CFG_OUTS+=("$OUT_DIR"); NAMES+="$JOB_TAG,"
done
NAMES+="mot_${CAMP_NAME}_prebuild,mot_${CAMP_NAME}_monitor,mot_${CAMP_NAME}_final"
LAST_OUT=${CFG_OUTS[${#CFG_OUTS[@]}-1]}

fail() { echo "TEST FAILED: $*" >&2; exit 1; }
command -v sbatch >/dev/null 2>&1 || { echo "sbatch not found: run this on the cluster (elsewhere use tests/test_local_campaign.sh)"; exit 2; }
"$PY" -c "import numpy, scipy, matplotlib, pynauty" 2>/dev/null \
    || fail "python3 on this node lacks numpy/scipy/matplotlib/pynauty (load your environment first; the job script has a site block for the compute nodes)"

wait_jobs() {
    local t0=$SECONDS n
    while :; do
        n=$(squeue -h -u "${USER:-$(id -un)}" -n "$NAMES" 2>/dev/null | wc -l)
        (( n == 0 )) && break
        (( SECONDS - t0 < ${WAIT_MAX:-7200} )) || fail "jobs still queued/running after ${WAIT_MAX:-7200}s (squeue -u \$USER)"
        echo "[$(date +%T)] $n jobs queued/running; $("$PY" monitor.py "$CAMP" --show 0 2>/dev/null | grep TOTAL | sed 's/^ *//')"
        sleep 30
    done
    sleep 5   # let the shared file system settle
}

active=$(squeue -h -u "${USER:-$(id -un)}" -n "$NAMES" 2>/dev/null | wc -l)
(( active == 0 )) || fail "$active test jobs are still queued/running; scancel them first"
echo "######## clean previous test output"
rm -rf "${CFG_OUTS[@]}" "$COUT"

echo "######## stage 1: interrupted run (MAX_PAIRS=${TEST_MAX_PAIRS:-8})"
MAX_PAIRS=${TEST_MAX_PAIRS:-8} bash slurm/submit_campaign.sh "$CAMP" "$MAXC"
wait_jobs
flog=$(ls -t "$COUT"/logs/mot_"${CAMP_NAME}"_final_*.out 2>/dev/null | head -1 || true)
[[ -n "$flog" ]] || fail "no log of the final job in $COUT/logs"
grep -q "campaign incomplete" "$flog" || { tail -20 "$flog"; fail "final job of stage 1 should report an incomplete campaign"; }
echo "final job of stage 1: incomplete campaign detected (log $flog)"
grep -q "stopped with" "$LAST_OUT"/logs/*.out || fail "shard logs do not show the interruption"
echo "--- a shard log:"; head -4 "$(ls "$LAST_OUT"/logs/*.out | head -1)"
[[ -s "$COUT/progress.log" ]] || fail "monitor did not write $COUT/progress.log"
echo "--- progress.log (last block):"; awk '/^=====/{b=""} {b=b $0 "\n"} END{printf "%s", b}' "$COUT/progress.log" | head -12
[[ -f "$COUT/analysis_partial/auc_vs_rho_all_p.png" ]] || fail "no interim campaign figure"
ls "$COUT"/logs/mot_"${CAMP_NAME}"_monitor_*.out >/dev/null 2>&1 || fail "no monitor log"

echo "######## stage 2: resubmission (same command, no MAX_PAIRS)"
bash slurm/submit_campaign.sh "$CAMP" "$MAXC"
wait_jobs
flog=$(ls -t "$COUT"/logs/mot_"${CAMP_NAME}"_final_*.out | head -1)
grep -q "final results:" "$flog" || { tail -30 "$flog"; fail "final job of stage 2 did not write final results"; }
grep -h "already done (resumed)" "$LAST_OUT"/logs/*.out | grep -v " 0 already done" | head -2 || true

echo "######## output checks"
"$PY" tests/check_outputs.py "$CAMP"
echo
echo "SLURM CAMPAIGN TEST PASSED"
echo "  final figures : $COUT/analysis/ and <config out_dir>/analysis/"
echo "  job logs      : $COUT/logs/ and <config out_dir>/logs/"
echo "  accounting    : sacct -u \$USER -S today --name=$NAMES --format=JobID%20,JobName%24,State,Elapsed,MaxRSS"
