#!/usr/bin/env python3
"""A minimal stand-in for sbatch / squeue / scancel, to test the submission scripts on a
machine without SLURM:

    PATH=$PWD/tests/mock_slurm:$PATH bash tests/test_slurm_small.sh campaigns/test_small.json

Emulated: --array=SPEC[%THROTTLE] (tasks run THROTTLE at a time, default MOCK_PARALLEL=2),
--export=ALL,K=V,..., --output with %x %j %A %a, --job-name, --cpus-per-task
(SLURM_CPUS_PER_TASK, capped at MOCK_CPUS, default 2), --dependency=afterok:/afterany:
(jobs run in submission order; an afterok job whose dependency failed is not run),
--parsable, --wait, --wrap.  Jobs named *_monitor run in the background (squeue lists them,
scancel stops them); all other jobs run to completion inside the sbatch call.  Time and
memory limits are ignored.  State lives in $MOCK_SLURM_DIR (default /tmp/mock_slurm_$USER).
"""
import os, sys, json, signal, subprocess, time, getpass
from concurrent.futures import ThreadPoolExecutor

STATE = os.environ.get("MOCK_SLURM_DIR", f"/tmp/mock_slurm_{getpass.getuser()}")
os.makedirs(STATE, exist_ok=True)
DB = os.path.join(STATE, "jobs.json")


def db_load():
    return json.load(open(DB)) if os.path.exists(DB) else {"next": 1000, "jobs": {}}


def db_save(d):
    json.dump(d, open(DB + ".tmp", "w")); os.replace(DB + ".tmp", DB)


def parse_array(spec):
    thr = None
    if "%" in spec:
        spec, thr = spec.split("%"); thr = int(thr)
    idx = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-"); idx += list(range(int(a), int(b) + 1))
        else:
            idx.append(int(part))
    return idx, thr


def fill(pattern, name, jid, task=None):
    return (pattern.replace("%x", name).replace("%A", str(jid)).replace("%j", str(jid))
            .replace("%a", "" if task is None else str(task)))


def sbatch(argv):
    opts, script, sargs = {}, None, []
    for a in argv:
        if script is None and a.startswith("--"):
            k, _, v = a[2:].partition("=")
            opts[k] = v if _ else True
        elif script is None:
            script = a
        else:
            sargs.append(a)
    d = db_load(); jid = d["next"]; d["next"] += 1
    name = opts.get("job-name", "sbatch")
    env = dict(os.environ)
    exp = opts.get("export", "ALL")
    for kv in exp.split(",")[1:] if exp.startswith("ALL") else exp.split(","):
        if "=" in kv:
            k, v = kv.split("=", 1); env[k] = v
    cpus = min(int(opts.get("cpus-per-task", 1)), int(os.environ.get("MOCK_CPUS", "2")))
    env.update(SLURM_JOB_ID=str(jid), SLURM_CPUS_PER_TASK=str(cpus), SLURM_JOB_NAME=name,
               SLURM_SUBMIT_DIR=os.getcwd())
    cwd = opts.get("chdir", os.getcwd())
    out = opts.get("output", "slurm-%j.out")
    cmd = ["bash", script] + sargs if script else ["bash", "-c", opts["wrap"]]
    dep = opts.get("dependency", "")
    ok_dep = True
    for part in dep.split(","):
        if part.startswith("afterok:"):
            ok_dep &= all(d["jobs"].get(i, {}).get("rc", 1) == 0 for i in part[8:].split(":"))
    rec = dict(name=name, state="PENDING", rc=None, pid=None)
    d["jobs"][str(jid)] = rec; db_save(d)
    print(jid if opts.get("parsable") else f"Submitted batch job {jid}", flush=True)
    if not ok_dep:
        rec.update(state="DependencyNeverSatisfied", rc=1)
    elif name.endswith("_monitor"):
        path = os.path.join(cwd, fill(out, name, jid))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=open(path, "a"), stderr=subprocess.STDOUT,
                             start_new_session=True)
        rec.update(state="RUNNING", pid=p.pid)
    elif "array" in opts:
        idx, thr = parse_array(opts["array"])
        par = thr or int(os.environ.get("MOCK_PARALLEL", "2"))

        def task(i):
            e = dict(env, SLURM_ARRAY_TASK_ID=str(i), SLURM_ARRAY_JOB_ID=str(jid))
            path = os.path.join(cwd, fill(out, name, jid, i))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a") as f:
                return subprocess.call(cmd, cwd=cwd, env=e, stdout=f, stderr=subprocess.STDOUT)
        with ThreadPoolExecutor(par) as ex:
            rcs = list(ex.map(task, idx))
        rec.update(state="COMPLETED" if max(rcs) == 0 else "FAILED", rc=max(rcs))
    else:
        path = os.path.join(cwd, fill(out, name, jid))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            rc = subprocess.call(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)
        rec.update(state="COMPLETED" if rc == 0 else "FAILED", rc=rc)
    d = db_load(); d["jobs"][str(jid)].update(rec); db_save(d)
    return (rec["rc"] or 0) if opts.get("wait") else 0


def alive(pid):
    try:
        os.kill(pid, 0); return True
    except OSError:
        return False


def squeue(argv):
    names = None; fmt = None
    for i, a in enumerate(argv):
        if a == "-n":
            names = set(argv[i + 1].split(","))
        if a == "-o":
            fmt = argv[i + 1]
    d = db_load()
    for jid, r in d["jobs"].items():
        if r["state"] == "RUNNING" and r["pid"] and alive(r["pid"]):
            # reap a finished background job
            try:
                if os.waitpid(r["pid"], os.WNOHANG)[0]:
                    continue
            except ChildProcessError:
                pass
            if names is None or r["name"] in names:
                print(r["name"] if fmt == "%j" else f"{jid} {r['name']} R")


def scancel(argv):
    d = db_load()
    for jid in argv:
        r = d["jobs"].get(jid)
        if r and r["state"] == "RUNNING" and r["pid"]:
            try:
                os.killpg(r["pid"], signal.SIGTERM)
            except OSError:
                pass
            r["state"] = "CANCELLED"
    db_save(d)


def cleanup(argv):
    """Stop only background jobs recorded in this mock's isolated state directory."""
    scancel([jid for jid, rec in db_load()["jobs"].items() if rec["state"] == "RUNNING"])


if __name__ == "__main__":
    raise SystemExit({"sbatch": sbatch, "squeue": squeue, "scancel": scancel,
                      "cleanup": cleanup}[sys.argv[1]](sys.argv[2:]))
