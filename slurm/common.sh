# Shared helpers for submit.sh and resubmit_missing.sh (sourced, not executed).
# Environment knobs (all optional):
#   CPUS                cpus per array task (= workers per shard); default: #SBATCH line (4)
#   ARRAY_TIME          wall time per array task, e.g. 02:00:00; default: #SBATCH line
#   ARRAY_MEM_PER_CPU   memory per worker, e.g. 3G;              default: #SBATCH line
#   AGG_TIME, AGG_MEM   aggregation + plotting job (default 04:00:00 and an estimate from the config)
#   SBATCH_EXTRA        extra sbatch options for every job, e.g. "--account=abc --qos=normal"
#                       (sbatch also honours SBATCH_ACCOUNT, SBATCH_PARTITION, ... natively)
#   MAX_ARRAY_SIZE      override the cluster's MaxArraySize (read from scontrol, default 1001)
#   DRY_RUN=1           print the sbatch commands instead of submitting (automatic when
#                       sbatch is not available)
#   PYTHON              python interpreter (default python3)
# Written for bash >= 4.2 (empty arrays are expanded with the ${a[@]+...} idiom).

PKG_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
JOB_SCRIPT="$PKG_DIR/slurm/run_array.sbatch"
PY=${PYTHON:-python3}
AGG_TIME=${AGG_TIME:-04:00:00}
EXTRA=()
if [[ -n "${SBATCH_EXTRA:-}" ]]; then
    read -r -a EXTRA <<< "$SBATCH_EXTRA"
fi

if [[ "${DRY_RUN:-0}" != 1 ]] && ! command -v sbatch >/dev/null 2>&1; then
    echo "NOTE: sbatch not found -> DRY_RUN=1 (sbatch commands are printed, nothing is submitted)"
    DRY_RUN=1
fi
DRY_RUN=${DRY_RUN:-0}
_DRY_COUNTER=0

# abs_config PATH -> absolute path of the config (must exist)
abs_config() {
    local c=$1
    if [[ ! -f "$c" ]]; then
        echo "ERROR: config $c not found" >&2
        exit 2
    fi
    echo "$(cd "$(dirname "$c")" && pwd)/$(basename "$c")"
}

# submit_job VAR sbatch-args...   (stores the job id in VAR)
submit_job() {
    local __var=$1; shift
    local __jid
    if [[ "$DRY_RUN" == 1 ]]; then
        _DRY_COUNTER=$((_DRY_COUNTER + 1)); __jid="DRYRUN$_DRY_COUNTER"
        printf '[dry-run] sbatch'; printf ' %q' "$@"; printf '   # -> job %s\n' "$__jid"
    else
        if ! __jid=$(sbatch --parsable "$@"); then
            echo "ERROR: sbatch failed: sbatch $*" >&2
            return 1
        fi
        __jid=${__jid%%;*}
        echo "$(date '+%F %T') job $__jid: sbatch $*" >> "$OUT_DIR/logs/jobs.txt"
    fi
    printf -v "$__var" '%s' "$__jid"
}

max_array_size() {
    local m=""
    if command -v scontrol >/dev/null 2>&1; then
        m=$(scontrol show config 2>/dev/null | awk -F= '/^MaxArraySize/ {gsub(/[ \t]/, "", $2); print $2}' || true)
    fi
    echo "${MAX_ARRAY_SIZE:-${m:-1001}}"
}

# load_config CONFIG: cd to the package directory and set CONFIG (absolute), NSHARDS,
# NPAIRS, OUT_DIR, CACHE_FILE, CACHE_EXISTS, JOB_TAG, AGG_MEM_AUTO (see slurm/cfginfo.py)
load_config() {
    local info
    CONFIG=$(abs_config "$1")
    cd "$PKG_DIR"
    info=$("$PY" slurm/cfginfo.py info "$CONFIG")
    eval "$info"
    mkdir -p "$OUT_DIR/logs"
}

# squeue_active: number of queued/running jobs of this config (0 if squeue is unavailable)
squeue_active() {
    if command -v squeue >/dev/null 2>&1; then
        { squeue -h -u "${USER:-$(id -un)}" -o '%j' 2>/dev/null || true; } \
            | grep -c -E "^[[:space:]]*${JOB_TAG}(_prebuild)?[[:space:]]*\$" || true
    else
        echo 0
    fi
}

# submit_arrays DEPENDENCY MAX_CONCURRENT  < lines "OFFSET SPEC COUNT"
# Submits one array job per chunk (array indices must be < MaxArraySize; shard = OFFSET + index)
# and stores the job ids in ARRAY_IDS.  With several chunks the throttle is split between
# them, so that at most MAX_CONCURRENT tasks run in total.
submit_arrays() {
    local dep=$1 maxc=$2 lines nchunks per="" off spec cnt id
    local -a opts
    lines=$(cat)
    ARRAY_IDS=()
    if [[ -z "$lines" ]]; then
        return 0
    fi
    nchunks=$(printf '%s\n' "$lines" | wc -l)
    nchunks=$((nchunks + 0))
    if [[ -n "$maxc" ]]; then
        per=$(( maxc / nchunks ))
        if (( per < 1 )); then per=1; fi
        if (( nchunks > 1 )); then
            echo "NOTE: $nchunks array chunks (MaxArraySize $(max_array_size)); throttle $per per chunk"
        fi
    fi
    opts=(--job-name="$JOB_TAG" --chdir="$PKG_DIR" --output="$OUT_DIR/logs/%x_%A_%a.out")
    if [[ -n "${CPUS:-}" ]]; then opts+=(--cpus-per-task="$CPUS"); fi
    if [[ -n "${ARRAY_TIME:-}" ]]; then opts+=(--time="$ARRAY_TIME"); fi
    if [[ -n "${ARRAY_MEM_PER_CPU:-}" ]]; then opts+=(--mem-per-cpu="$ARRAY_MEM_PER_CPU"); fi
    if [[ -n "$dep" ]]; then opts+=(--dependency="$dep"); fi
    while read -r off spec cnt; do
        submit_job id "${opts[@]}" ${EXTRA[@]+"${EXTRA[@]}"} \
            --array="${spec}${per:+%$per}" \
            --export="ALL,MODE=array,PKG_DIR=$PKG_DIR,SHARD_OFFSET=$off${MAX_PAIRS:+,MAX_PAIRS=$MAX_PAIRS}" \
            "$JOB_SCRIPT" "$CONFIG"
        echo "array job $id: $cnt shards (shard = $off + array index)"
        ARRAY_IDS+=("$id")
    done <<< "$lines"
}

# submit_aggregate DEPENDENCY  -> AGG_ID
submit_aggregate() {
    local dep=$1
    local -a opts=(--job-name="${JOB_TAG}_agg" --chdir="$PKG_DIR" --output="$OUT_DIR/logs/%x_%j.out"
                   --cpus-per-task=1 --time="$AGG_TIME" --mem="${AGG_MEM:-$AGG_MEM_AUTO}")
    if [[ -n "$dep" ]]; then opts+=(--dependency="$dep"); fi
    submit_job AGG_ID "${opts[@]}" ${EXTRA[@]+"${EXTRA[@]}"} \
        --export="ALL,MODE=aggregate,PKG_DIR=$PKG_DIR" "$JOB_SCRIPT" "$CONFIG"
    echo "aggregation + plots job $AGG_ID (mem ${AGG_MEM:-$AGG_MEM_AUTO}, time $AGG_TIME), dependency ${dep:-none}"
}

# join_ids SEP ids... -> "id1SEPid2..."
join_ids() {
    local sep=$1; shift
    local out="" x
    for x in "$@"; do out+="${out:+$sep}$x"; done
    echo "$out"
}
