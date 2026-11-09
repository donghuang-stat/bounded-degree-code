#!/usr/bin/env python3
"""Progress monitor, interim results and final results of a campaign (or of one config).

Usage:
    python monitor.py CAMPAIGN.json                     # status table, once
    python monitor.py CAMPAIGN.json --plots             # + interim analysis and figures
    python monitor.py CAMPAIGN.json --loop 30 --plots --until-done
                                                        # every 30 min until all shards
                                                        # are complete (the SLURM monitor job)
    python monitor.py CAMPAIGN.json --final             # final results (if complete; otherwise
                                                        # interim results + list of missing shards)
    python monitor.py CAMPAIGN.json --loop 30 --plots --until-done --final
                                                        # both: interim figures while running,
                                                        # final results at the end (no SLURM needed)
CAMPAIGN.json may also be a single config (a campaign with one member).

Reads (nothing here touches the running jobs):
    <out_dir>/shard_*.npz                   finished shards
    <out_dir>/partial/shard_*/pair_*.npy    checkpointed pairs of unfinished shards
    <out_dir>/progress/shard_*.json         per-shard status written by run_shard.py
Writes into the campaign out_dir:
    status.json            machine-readable status of every config
    progress.log           one block per call (time, pairs done, rate, ETA, running shards)
    monitor_history.csv    (time, config, pairs done), used for the throughput estimate
    analysis_partial/      interim campaign figures and tables (--plots), see campaign_plot.py
    analysis/              final campaign figures and tables (when complete)
    manifest.json          final: configs, timing totals, software versions, code checksums
and, per config, <out_dir>/analysis_partial/ (interim) or <out_dir>/analysis/ (final), see
aggregate.py and plots.py.

Exit status of --final: 0 if all shards are complete and the final results were written,
3 if shards are missing (interim results written instead).
"""
import os, sys, json, glob, time, math, hashlib, platform, subprocess, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from campaign_plot import load_campaign

STALE_MIN = 20          # a running shard without an update for max(STALE_MIN, 3 x wall per pair) is "stale"


class Lock:
    """Best-effort exclusive lock (fcntl.flock) around the analysis step; waits up to
    `wait` seconds.  Where the file system does not support flock, it is a no-op."""
    def __init__(self, path, wait=1800):
        self.path, self.wait, self.f = path, wait, None

    def acquire(self):
        try:
            import fcntl
            self.f = open(self.path, "w")
            t0 = time.time()
            while True:
                try:
                    fcntl.flock(self.f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return True
                except BlockingIOError:
                    if time.time() - t0 > self.wait:
                        return False
                    time.sleep(5)
        except OSError:
            return True

    def release(self):
        if self.f is not None:
            try:
                import fcntl
                fcntl.flock(self.f, fcntl.LOCK_UN)
            except OSError:
                pass
            self.f.close(); self.f = None


def fmt_s(x):
    if x is None or not math.isfinite(x):
        return "?"
    x = int(round(x))
    d, h, m = x // 86400, (x % 86400) // 3600, (x % 3600) // 60
    return (f"{d}d" if d else "") + f"{h}h{m:02d}m"


def scan(cfg):
    """Status of one config from the files on disk."""
    out = cfg["out_dir"]; S = cfg["shard_size"]; T = sum(cfg["reps"])
    N = (T + S - 1) // S
    complete = set()
    for f in glob.glob(os.path.join(out, "shard_*.npz")):
        if not f.endswith(".tmp.npz"):
            complete.add(int(os.path.basename(f)[6:11]))
    done = sum(min(S, T - k * S) for k in complete)
    partial = {}
    for d in glob.glob(os.path.join(out, "partial", "shard_*")):
        k = int(os.path.basename(d)[6:11])
        if k in complete:
            continue
        c = len([f for f in glob.glob(os.path.join(d, "pair_*.npy")) if not f.endswith(".tmp.npy")])
        partial[k] = c
    now = time.time()
    running, stale, cpu = [], [], []
    for f in glob.glob(os.path.join(out, "progress", "shard_*.json")):
        try:
            g = json.load(open(f))
        except Exception:
            continue
        if g.get("sec_per_pair_cpu"):
            cpu.append((g["sec_per_pair_cpu"], g.get("done", 0)))
        k = g["shard"]
        if k in complete or g.get("complete"):
            continue
        age = now - g.get("updated_unix", os.path.getmtime(f))
        lim = max(STALE_MIN * 60, 3 * (g.get("wall_per_pair") or 0))
        fresh = age < lim and g.get("state", "running") == "running"
        (running if fresh else stale).append(dict(g, age=age))
    w = sum(c for _, c in cpu)
    sec_pair = sum(s * c for s, c in cpu) / w if w else None
    return dict(name=cfg.get("name", os.path.basename(out)), p=cfg["p"], n=cfg["n"], total=T, shards=N,
                complete=len(complete), pairs_done=done + sum(partial.values()),
                pairs_in_complete=done, pairs_checkpointed=sum(partial.values()),
                started_shards=len(partial), running=sorted(running, key=lambda g: g["shard"]),
                stale=sorted(stale, key=lambda g: g["shard"]),
                missing=[k for k in range(N) if k not in complete], cpu_sec_per_pair=sec_pair)


def history_rate(path, name, now, done, window_h=1.0):
    """Pairs per hour over the last window_h hours (or since the first record)."""
    rows = []
    if os.path.exists(path):
        for line in open(path):
            t, nm, d = line.strip().split(",")
            if nm == name:
                rows.append((float(t), int(d)))
    rows = [r for r in rows if r[0] < now - 60]
    if not rows:
        return None
    recent = [r for r in rows if r[0] >= now - window_h * 3600]
    t0, d0 = (recent[0] if recent else rows[-1])
    if now - t0 < 60 or done < d0:
        return None
    return (done - d0) / ((now - t0) / 3600)


def run(cmd, log):
    t0 = time.time()
    r = subprocess.run([sys.executable] + cmd, cwd=HERE, capture_output=True, text=True)
    tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
    log(f"    $ python {' '.join(os.path.relpath(c, HERE) if os.path.isabs(c) else c for c in cmd)}"
        f"  [{'ok' if r.returncode == 0 else 'exit ' + str(r.returncode)}, {time.time() - t0:.0f}s]")
    for t in tail:
        log(f"      {t}")
    return r.returncode == 0


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def manifest(camp, cfgs, stats, path):
    import numpy, scipy, matplotlib
    cpu = 0.0
    for cfg in cfgs:
        for f in glob.glob(os.path.join(cfg["out_dir"], "shard_*.npz")):
            if not f.endswith(".tmp.npz"):
                cpu += float(numpy.load(f)["sec"].sum())
    try:
        import pynauty; pyn = getattr(pynauty, "__version__", "?")
    except Exception:
        pyn = "not importable"
    code = {os.path.relpath(f, HERE): md5(f) for f in sorted(glob.glob(os.path.join(HERE, "*.py")))}
    m = dict(campaign=camp.get("name"), title=camp.get("title"), finished=time.strftime("%Y-%m-%d %H:%M:%S"),
             configs={c: json.load(open(c)) for c in camp["config_paths"]},
             pairs={s["name"]: s["total"] for s in stats}, cpu_hours_total=cpu / 3600,
             software=dict(python=platform.python_version(), numpy=numpy.__version__, scipy=scipy.__version__,
                           matplotlib=matplotlib.__version__, pynauty=pyn, platform=platform.platform()),
             code_md5=code)
    with open(path, "w") as f:
        json.dump(m, f, indent=1)


def once(args, camp, cfgs, campaign_path):
    cout = camp["out_dir"]; os.makedirs(cout, exist_ok=True)
    block = []
    lock = Lock(os.path.join(cout, ".monitor.lock"))

    def log(s=""):
        print(s, flush=True); block.append(s)

    def flush_log():
        # one write per block, so that concurrent monitors (monitor job, final job, a user on
        # the login node) do not interleave their lines
        with open(os.path.join(cout, "progress.log"), "a") as f:
            f.write("\n".join(block) + "\n")
        block.clear()

    now = time.time()
    hist = os.path.join(cout, "monitor_history.csv")
    stats = [scan(c) for c in cfgs]
    log(f"===== {time.strftime('%Y-%m-%d %H:%M:%S')}  {camp.get('title') or camp.get('name')}")
    tot_done = sum(s["pairs_done"] for s in stats); tot = sum(s["total"] for s in stats)
    tot_rate = 0.0; rate_known = True
    for s in stats:
        rate = history_rate(hist, s["name"], now, s["pairs_done"])
        s["pairs_per_hour"] = rate
        if rate is None:
            rate_known = False
        else:
            tot_rate += rate
        left = s["total"] - s["pairs_done"]
        eta = left / rate * 3600 if rate else None
        s["eta_seconds"] = 0 if left == 0 else eta
        log(f"  {s['name']:<14s} p={s['p']:<5g} pairs {s['pairs_done']:>6d}/{s['total']:<6d} "
            f"({100 * s['pairs_done'] / s['total']:5.1f}%)  shards {s['complete']:>4d}/{s['shards']:<4d} complete, "
            f"{len(s['running'])} running, {len(s['stale'])} interrupted, "
            f"{s['shards'] - s['complete'] - len(s['running']) - len(s['stale'])} not started"
            + (f", {s['cpu_sec_per_pair']:.3g} CPU s/pair" if s["cpu_sec_per_pair"] else "")
            + (f", {rate:.1f} pairs/h, ETA {fmt_s(eta)}" if rate else ""))
        for g in s["running"][:args.show]:
            log(f"      shard {g['shard']:>5d} on {g.get('host', '?')}: {g['done']}/{g['total']} pairs, "
                f"ETA {fmt_s(g.get('eta_seconds'))}, last update {fmt_s(g['age'])} ago")
        if len(s["running"]) > args.show:
            log(f"      ... {len(s['running']) - args.show} more running shards")
        if s["stale"]:
            log(f"      not running (interrupted, timed out, stopped or stale; resubmit to resume): shards "
                + ",".join(f"{g['shard']}({g['done']}/{g['total']})" for g in s["stale"][:12])
                + (" ..." if len(s["stale"]) > 12 else ""))
    all_done = all(s["complete"] == s["shards"] for s in stats)
    eta = (tot - tot_done) / tot_rate * 3600 if (tot_rate > 0 and rate_known) else None
    log(f"  TOTAL pairs {tot_done}/{tot} ({100 * tot_done / tot:.1f}%)"
        + (f", {tot_rate:.1f} pairs/h, ETA {fmt_s(eta)} (finish ~ {time.strftime('%m-%d %H:%M', time.localtime(now + eta))})"
           if eta else "") + ("  ALL COMPLETE" if all_done else ""))
    with open(hist, "a") as f:
        for s in stats:
            f.write(f"{now:.0f},{s['name']},{s['pairs_done']}\n")
    st = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), campaign=camp.get("name"), pairs_done=tot_done,
              pairs_total=tot, all_complete=all_done, eta_seconds=eta,
              configs=[dict({k: v for k, v in s.items() if k not in ("running", "stale", "missing")},
                            running_shards=[g["shard"] for g in s["running"]],
                            stale_shards=[g["shard"] for g in s["stale"]],
                            missing_shards=len(s["missing"])) for s in stats])
    tmp = os.path.join(cout, "status.json.tmp")
    with open(tmp, "w") as f:
        json.dump(st, f, indent=1)
    os.replace(tmp, os.path.join(cout, "status.json"))

    final = all_done and args.final
    flush_log()
    if (final or args.plots or args.final) and not lock.acquire():
        log("  (another monitor holds the lock; skipping the analysis this time)")
        flush_log()
        return "complete" if all_done else "running"
    if final:
        log("  final results:")
        ok = True
        for c in camp["config_paths"]:
            ok &= run(["aggregate.py", c], log)
            ok &= run(["plots.py", c], log)
        ok &= run(["campaign_plot.py", campaign_path], log)
        manifest(camp, cfgs, stats, os.path.join(cout, "manifest.json"))
        log(f"  manifest: {os.path.join(cout, 'manifest.json')}")
        log(f"  figures and tables: {os.path.join(cout, 'analysis')}/ and <config out_dir>/analysis/")
        flush_log(); lock.release()
        return "final" if ok else "final_with_errors"
    if args.plots or args.final:
        log("  interim results:")
        for c, s in zip(camp["config_paths"], stats):
            if s["pairs_done"] == 0:
                continue
            if run(["aggregate.py", c, "--partial"], log):
                run(["plots.py", c, "--partial"], log)
        run(["campaign_plot.py", campaign_path, "--partial"], log)
        log(f"  interim figures: {os.path.join(cout, 'analysis_partial')}/")
    if args.final and not all_done:
        log("  missing shards: " + ", ".join(f"{s['name']} {len(s['missing'])}/{s['shards']}" for s in stats))
        log(f"  resubmit (only missing shards; unfinished ones resume): bash slurm/submit_campaign.sh {campaign_path}")
    flush_log(); lock.release()
    return "complete" if all_done else "running"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("campaign")
    ap.add_argument("--plots", action="store_true", help="interim analysis and figures")
    ap.add_argument("--loop", type=float, default=None, metavar="MIN", help="repeat every MIN minutes")
    ap.add_argument("--plot-every", type=int, default=1, metavar="K",
                    help="with --loop --plots: interim figures only every K-th iteration")
    ap.add_argument("--until-done", action="store_true", help="with --loop: stop when all shards are complete")
    ap.add_argument("--final", action="store_true", help="final results (interim + missing list if incomplete)")
    ap.add_argument("--show", type=int, default=8, help="running shards listed per config")
    args = ap.parse_args()
    path = os.path.abspath(args.campaign)
    it = 0
    while True:
        camp, cfgs = load_campaign(path)
        plots = args.plots
        args.plots = plots and (it % args.plot_every == 0)
        res = once(args, camp, cfgs, path)
        args.plots = plots
        it += 1
        if args.loop is None:
            break
        if res.startswith("final") or (args.until_done and res == "complete"):
            break
        time.sleep(args.loop * 60)
    if args.final and res == "running":
        sys.exit(3)
    if res == "final_with_errors":
        sys.exit(1)


if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)   # quiet under `| head`
    main()
