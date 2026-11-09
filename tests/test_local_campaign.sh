#!/bin/bash
# =====================================================================================
# Local end-to-end test of the campaign pipeline.  No SLURM needed: runs on a laptop, a
# workstation or (briefly) a login node.  About 1-3 minutes with 2 CPUs.
#
#   bash tests/test_local_campaign.sh            (from the package directory or anywhere)
#   PARALLEL=8 bash tests/test_local_campaign.sh (number of shards run at the same time)
#
# Campaign campaigns/test_small.json: the n = 5000 design (4 values of p, 13 rho values,
# motifs with 3 to 5 edges, interleaved shards) at n = 60 with 340 pairs per p.
#
#   1. "array, interrupted": every shard runs as its own process (PARALLEL at a time, like
#      array tasks) and stops after 6 pairs (--max-pairs), leaving per-pair checkpoints;
#   2. monitor: status table, interim analysis and figures from the checkpoints; the final
#      results step must refuse (exit 3) because shards are missing;
#   3. time limit and hard kill: a shard of a larger graph (n = 800) receives SIGTERM (what
#      SLURM sends at the time limit), is resumed and killed with SIGKILL, then resumed again;
#   4. "resubmission": all shards rerun and resume from their checkpoints;
#   5. final results (monitor.py --final) and tests/check_outputs.py;
#   6. determinism: a fresh run of the same pairs, with another shard size and the blocked
#      task order, in one process, gives bit-identical motif counts.
# =====================================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=${PYTHON:-python3}
export PYTHON="$PY"
PARALLEL=${PARALLEL:-4}
CAMP=campaigns/test_small.json
CFGS=(configs/test_small_p001.json configs/test_small_p005.json configs/test_small_p02.json configs/test_small_p05.json)
TMP=out/test_local_tmp
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg

step() { echo; echo "######## $*"; }
fail() { echo "TEST FAILED: $*" >&2; exit 1; }

step "0. clean previous test output, build the motif library cache"
rm -rf out/test_small_p001 out/test_small_p005 out/test_small_p02 out/test_small_p05 out/campaign_test_small "$TMP"
mkdir -p "$TMP"
"$PY" slurm/cfginfo.py prebuild "${CFGS[0]}"

run_all_shards() {   # $1 = extra run_shard.py options; logs like SLURM array logs
    local extra=$1 c N k
    for c in "${CFGS[@]}"; do
        N=$("$PY" run_shard.py "$c" --list)
        mkdir -p "out/$(basename "${c%.json}")/logs"
        for ((k = 0; k < N; k++)); do echo "$c $k"; done
    done | xargs -P "$PARALLEL" -n 2 sh -c \
        'c=$0; k=$1; d=out/$(basename "${c%.json}")/logs; "$PYTHON" run_shard.py "$c" --shard "$k" --workers 1 '"$extra"' >> "$d/shard_$k.out" 2>&1'
}

step "1. array tasks, interrupted after 6 pairs per shard"
run_all_shards "--max-pairs 6"
n_ck=$(find out/test_small_p0* -path '*partial*' -name 'pair_*.npy' | wc -l)
n_sh=$(find out/test_small_p0* -maxdepth 1 -name 'shard_*.npz' | wc -l)
echo "checkpointed pairs: $n_ck, complete shards: $n_sh"
(( n_ck > 0 && n_sh == 0 )) || fail "expected checkpoints and no complete shard after step 1"
echo "--- a shard log (per-pair progress lines):"
head -5 out/test_small_p05/logs/shard_0.out

step "2. monitor: status, interim figures; --final must refuse"
"$PY" monitor.py "$CAMP" --plots --show 2
for f in out/campaign_test_small/analysis_partial/auc_vs_rho_all_p.png out/campaign_test_small/analysis_partial/power_vs_rho_all_p.png out/campaign_test_small/status.json \
         out/campaign_test_small/progress.log out/test_small_p05/analysis_partial/summary.csv \
         out/test_small_p05/analysis_partial/auc_vs_rho.png; do
    [[ -f "$f" ]] || fail "missing interim output $f"
done
rc=0; "$PY" monitor.py "$CAMP" --final --show 0 > "$TMP/final_refused.log" 2>&1 || rc=$?
(( rc == 3 )) || fail "monitor.py --final should exit 3 on an incomplete campaign (got $rc)"
echo "monitor.py --final on the incomplete campaign: exit 3 as expected"

step "3. time limit (SIGTERM) and hard kill (SIGKILL) during a shard of n = 800, then resume"
cat > "$TMP/kill.json" <<EOF
{"name": "kill", "n": 800, "p": 0.2, "mmin": 3, "mmax": 5, "rhos": [0.0, 0.8], "reps": [12, 12],
 "seed": 99, "shard_size": 24, "order": "interleaved", "out_dir": "$TMP/kill"}
EOF
count_ck() { find "$TMP/kill/partial" -name 'pair_*.npy' ! -name '*.tmp.npy' 2>/dev/null | wc -l; }
# A fixed delay can let a fast machine finish the entire shard without testing a
# signal. Trigger after an actual checkpoint instead; the deadline catches hangs.
ck_dir="$TMP/kill/partial/shard_00000"
rc=0; "$PY" tests/timeout_command.py -s TERM --checkpoint-dir "$ck_dir" --after-checkpoints 1 60 \
    "$PY" run_shard.py "$TMP/kill.json" --shard 0 --workers 2 > "$TMP/kill1.log" 2>&1 || rc=$?
(( rc == 124 )) || fail "TERM test did not time out as expected (exit $rc)"
echo "after SIGTERM: $(count_ck) of 24 pairs checkpointed"; tail -1 "$TMP/kill1.log"
"$PY" -c "import json; s = json.load(open('$TMP/kill/progress/shard_00000.json')); print('  progress file: state', s['state'], s['done'], 'of', s['total']); assert s['state'] == 'interrupted' and 0 < s['done'] < s['total']"
next_ck=$(( $(count_ck) + 1 ))
rc=0; "$PY" tests/timeout_command.py -s KILL --checkpoint-dir "$ck_dir" --after-checkpoints "$next_ck" 60 \
    "$PY" run_shard.py "$TMP/kill.json" --shard 0 --workers 2 > "$TMP/kill2.log" 2>&1 || rc=$?
(( rc == 137 )) || fail "KILL test did not time out as expected (exit $rc)"
echo "after SIGKILL: $(count_ck) of 24 pairs checkpointed"; head -1 "$TMP/kill2.log"
"$PY" run_shard.py "$TMP/kill.json" --shard 0 --workers 2 > "$TMP/kill3.log" 2>&1
head -1 "$TMP/kill3.log"; tail -1 "$TMP/kill3.log"
[[ -f "$TMP/kill/shard_00000.npz" ]] || fail "resumed shard did not complete"
sed -e 's/"shard_size": 24/"shard_size": 5/' -e 's/"interleaved"/"blocked"/' -e "s#$TMP/kill\"#$TMP/kill_ref\"#" \
    "$TMP/kill.json" > "$TMP/kill_ref.json"
N=$("$PY" run_shard.py "$TMP/kill_ref.json" --list)
for ((k = 0; k < N; k++)); do "$PY" run_shard.py "$TMP/kill_ref.json" --shard $k --workers 2 > /dev/null; done

step "4. resubmission: every shard resumes from its checkpoints"
run_all_shards ""
n_sh=$(find out/test_small_p0* -maxdepth 1 -name 'shard_*.npz' | wc -l)
echo "complete shards: $n_sh (expected 68)"
(( n_sh == 68 )) || fail "not all shards complete after resubmission"
grep -h "already done (resumed)" out/test_small_p05/logs/shard_0.out | tail -1

step "5. final results and output checks"
"$PY" monitor.py "$CAMP" --final --show 0
"$PY" tests/check_outputs.py "$CAMP"

step "6. determinism: other shard size + blocked order, uninterrupted, single process"
for c in "${CFGS[0]}" "${CFGS[3]}"; do
    b=$(basename "${c%.json}")
    sed -e 's/"shard_size": 20/"shard_size": 7/' -e 's/"interleaved"/"blocked"/' \
        -e "s#\"out/$b\"#\"$TMP/$b\"#" "$c" > "$TMP/$b.json"
    N=$("$PY" run_shard.py "$TMP/$b.json" --list)
    for ((k = 0; k < N; k++)); do "$PY" run_shard.py "$TMP/$b.json" --shard $k --workers 1 > /dev/null; done
done
"$PY" - "$TMP" <<'EOF'
import glob, sys, numpy as np
tmp = sys.argv[1]
def table(d):
    out = {}
    for f in glob.glob(f"{d}/shard_*.npz"):
        z = np.load(f)
        for k, r, x, e in zip(z["rho_idx"], z["rep"], z["inj"], z["edges"]):
            out[(int(k), int(r))] = (x, e)
    return out
ok = True
for a, b in ((f"{tmp}/kill", f"{tmp}/kill_ref"), ("out/test_small_p001", f"{tmp}/test_small_p001"),
             ("out/test_small_p05", f"{tmp}/test_small_p05")):
    A, B = table(a), table(b)
    same = A.keys() == B.keys() and all(np.array_equal(A[t][0], B[t][0]) and np.array_equal(A[t][1], B[t][1]) for t in A)
    print(f"  {a} vs {b}: {len(A)} pairs, {'bit-identical' if same else 'DIFFERENT'}")
    ok &= same
sys.exit(0 if ok else 1)
EOF

echo
echo "LOCAL CAMPAIGN TEST PASSED"
echo "  interim figures : out/campaign_test_small/analysis_partial/  (from step 2)"
echo "  final figures   : out/campaign_test_small/analysis/, out/test_small_p*/analysis/"
echo "  progress log    : out/campaign_test_small/progress.log"
