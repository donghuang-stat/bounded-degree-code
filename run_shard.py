#!/usr/bin/env python3
"""Simulate one shard of the experiment described by a JSON config.

Every task is a pair (rho index, replicate).  Rho index 0 must be rho = 0 (the null).
Each task samples a correlated Erdos-Renyi pair with its own seed
SeedSequence([seed, n, rho index, replicate]) and stores the exact signed injective
counts inj(M, A_j - p(J - I)) of every motif M in the library, for j = 1, 2, together
with the edge counts of A_1, A_2 and the number of common edges (a check of the
sampled graphs).  The result of a task does not depend on how tasks are split into
shards.  Config key "order": "blocked" (default: shards run through rho index 0, then
1, ...) or "interleaved" (every shard holds a proportional mix of all rho values, so
interim results cover every rho early; recommended for long campaigns).

Progress and checkpoints (useful for large n, where one pair takes minutes):
  * after every finished pair, the result is written to
        <out_dir>/partial/shard_KKKKK/pair_<rho_idx>_<rep>.npy
        layout: [inj of graph 1 (n_motifs), inj of graph 2 (n_motifs), CPU seconds,
                 edges of graph 1, edges of graph 2, common edges]
    so that a killed or timed-out task resumes where it stopped;
  * a progress line is printed (pairs done, seconds per pair, ETA), and
        <out_dir>/progress/shard_KKKKK.json
    is updated (read by monitor.py);
  * when all pairs of the shard are done, they are merged into
        <out_dir>/shard_KKKKK.npz   (rho_idx, rep, inj, sec, edges, host)
    and the partial directory is removed.
  * with "save_graphs": true in the config, the two sampled graphs of every pair are
    also stored, as bit-packed upper triangles, in <out_dir>/graphs/pair_<k>_<r>.npz
    (about 3 MB per pair at n = 5000).  Without it, graphs.py regenerates any pair
    exactly from its seed.

Usage:
    python run_shard.py CONFIG.json --shard K [--workers W]
    python run_shard.py CONFIG.json --list           # number of shards
Shard K is skipped if its output file already exists (safe to resubmit).
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import argparse, json, shutil, signal, socket, sys, time
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from motiflib import MotifLibrary, sample_pair, centered

LIB, CFG = None, None
LAYOUT_KEYS = ("n", "p", "mmin", "mmax", "rhos", "reps", "seed", "shard_size", "order")


def resolve(path):
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(HERE, path))


def load_config(path):
    with open(path) as f:
        cfg = json.load(f)
    assert cfg["rhos"][0] == 0.0, "rho index 0 must be the null"
    assert len(cfg["reps"]) == len(cfg["rhos"])
    cfg.setdefault("cache_dir", os.path.join(HERE, "cache"))
    cfg.setdefault("chunk_size", None)
    cfg.setdefault("order", "blocked")
    assert cfg["order"] in ("blocked", "interleaved")
    cfg["out_dir"] = resolve(cfg["out_dir"])
    cfg["cache_dir"] = resolve(cfg["cache_dir"])
    return cfg


def tasks(cfg):
    T = [(k, r) for k, R in enumerate(cfg["reps"]) for r in range(R)]
    if cfg.get("order", "blocked") == "interleaved":
        reps = cfg["reps"]
        T.sort(key=lambda t: ((t[1] + 0.5) / reps[t[0]], t[0]))
    return T


def shard_tasks(cfg, shard):
    S = cfg["shard_size"]
    return tasks(cfg)[shard * S:(shard + 1) * S]


def check_layout(cfg):
    """The shard -> task map must not change once a run has started."""
    lay = {k: cfg.get(k) for k in LAYOUT_KEYS}
    path = os.path.join(cfg["out_dir"], "layout.json")
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        diff = [k for k in LAYOUT_KEYS if old.get(k) != lay[k]]
        if diff:
            raise SystemExit(f"ERROR: config differs from {path} in {diff}; this out_dir was started "
                             f"with another design. Use a new out_dir.")
    else:
        tmp = f"{path}.{socket.gethostname()}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(lay, f, indent=1)
        os.replace(tmp, path)


def init(cfg, worker=True):
    global LIB, CFG
    if worker:      # SIGTERM (e.g. SLURM time limit) is handled by the main process only
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
    CFG = cfg
    LIB = MotifLibrary(cfg["mmin"], cfg["mmax"], cache_dir=cfg["cache_dir"])


def pair_seed(cfg, k, r):
    return np.random.SeedSequence([cfg["seed"], cfg["n"], k, r])


def work(task):
    k, r = task
    t0 = time.time()
    n, p, rho = CFG["n"], CFG["p"], CFG["rhos"][k]
    rng = np.random.default_rng(pair_seed(CFG, k, r))
    A1, A2 = sample_pair(rng, n, p, rho)
    edges = np.array([A1.sum() / 2, A2.sum() / 2, np.einsum("ij,ij->", A1, A2) / 2])
    if CFG.get("save_graphs"):
        iu = np.triu_indices(n, 1)
        gdir = os.path.join(CFG["out_dir"], "graphs")
        os.makedirs(gdir, exist_ok=True)
        gpath = os.path.join(gdir, f"pair_{k}_{r}.npz")
        tmp = gpath + ".tmp.npz"
        np.savez_compressed(tmp, n=n, p=p, rho=rho, k=k, r=r,
                            A1=np.packbits(A1[iu].astype(bool)), A2=np.packbits(A2[iu].astype(bool)))
        os.replace(tmp, gpath)
        del iu
    out = np.empty((2, len(LIB.motifs)))
    for j, A in enumerate((A1, A2)):
        out[j] = LIB.inj(centered(A, p), chunk_size=CFG["chunk_size"])
    return k, r, out, time.time() - t0, edges


def fmt_s(x):
    x = int(round(x))
    return f"{x // 3600}h{(x % 3600) // 60:02d}m{x % 60:02d}s"


def write_json_atomic(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--shard", type=int)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--max-pairs", type=int, default=None,
                    help="stop after this many new pairs (for testing interruption and resume)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    T = tasks(cfg)
    S = cfg["shard_size"]
    nshards = (len(T) + S - 1) // S
    if a.list:
        print(nshards)
        return
    shard = a.shard if a.shard is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
    if not 0 <= shard < nshards:
        raise SystemExit(f"shard {shard} out of range 0..{nshards - 1}")
    outdir = cfg["out_dir"]
    os.makedirs(outdir, exist_ok=True)
    check_layout(cfg)
    fn = os.path.join(outdir, f"shard_{shard:05d}.npz")
    if os.path.exists(fn):
        print(f"shard {shard} exists, skipping", flush=True)
        return
    mine = T[shard * S:(shard + 1) * S]
    pdir = os.path.join(outdir, "partial", f"shard_{shard:05d}")
    gdir = os.path.join(outdir, "progress")
    os.makedirs(pdir, exist_ok=True); os.makedirs(gdir, exist_ok=True)
    gfile = os.path.join(gdir, f"shard_{shard:05d}.json")
    host = socket.gethostname()
    jobid = os.environ.get("SLURM_ARRAY_JOB_ID", os.environ.get("SLURM_JOB_ID", ""))
    if os.environ.get("SLURM_ARRAY_TASK_ID"):
        jobid += "_" + os.environ["SLURM_ARRAY_TASK_ID"]

    def pair_path(k, r):
        return os.path.join(pdir, f"pair_{k}_{r}.npy")

    done = {t for t in mine if os.path.exists(pair_path(*t))}
    todo = [t for t in mine if t not in done]
    if a.max_pairs is not None:
        todo = todo[:a.max_pairs]
    t0 = time.time()
    start_str = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{start_str}] shard {shard}/{nshards} on {host} (job {jobid or 'local'}): {len(mine)} pairs, "
          f"{len(done)} already done (resumed), {len(todo)} to compute, {a.workers} workers, "
          f"n={cfg['n']}, p={cfg['p']}, motifs {cfg['mmin']}..{cfg['mmax']} edges", flush=True)
    # build or load the library once (cached on disk) before forking
    nm = len(MotifLibrary(cfg["mmin"], cfg["mmax"], cache_dir=cfg["cache_dir"]).motifs)
    secs = []

    last = dict(rate=None, sec=None)

    def status(ndone, rate, eta, complete, sec_mean, state="running"):
        write_json_atomic(gfile, dict(shard=shard, host=host, job=jobid, n=cfg["n"], p=cfg["p"],
                                      total=len(mine), done=ndone, state=state, started=start_str,
                                      updated=time.strftime("%Y-%m-%d %H:%M:%S"), updated_unix=time.time(),
                                      sec_per_pair_cpu=sec_mean, wall_per_pair=rate, eta_seconds=eta,
                                      workers=a.workers, complete=complete))

    status(len(done), None, None, False, None)

    def on_term(signum, frame):
        n_ok = len(done) + len(secs)
        status(n_ok, last["rate"], None, False, last["sec"], state="interrupted")
        print(f"[{time.strftime('%H:%M:%S')}] shard {shard}: signal {signum} (time limit or scancel), "
              f"{n_ok}/{len(mine)} pairs saved; rerun to resume", flush=True)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, on_term)

    def record(res):
        k, r, out, sec, edges = res
        tmp = pair_path(k, r) + ".tmp.npy"
        np.save(tmp, np.concatenate([out.ravel(), [sec], edges]))
        os.replace(tmp, pair_path(k, r))
        secs.append(sec)
        ndone = len(done) + len(secs)
        el = time.time() - t0
        rate = el / len(secs)                       # wall seconds per pair for this task
        eta = rate * (len(mine) - ndone)
        print(f"[{time.strftime('%H:%M:%S')}] shard {shard}: {ndone}/{len(mine)} pairs "
              f"(rho={cfg['rhos'][k]:g} [idx {k}], rep {r}, {sec:.1f}s CPU), elapsed {fmt_s(el)}, "
              f"ETA {fmt_s(eta)}", flush=True)
        last.update(rate=rate, sec=float(np.mean(secs)))
        status(ndone, rate, eta, False, last["sec"])

    if todo:
        if a.workers > 1:
            with Pool(min(a.workers, len(todo)), initializer=init, initargs=(cfg,)) as pool:
                for res in pool.imap_unordered(work, todo, chunksize=1):
                    record(res)
        else:
            init(cfg, worker=False)
            for t in todo:
                record(work(t))

    missing = [t for t in mine if not os.path.exists(pair_path(*t))]
    if missing:
        status(len(mine) - len(missing), last["rate"], None, False, last["sec"], state="stopped")
        print(f"shard {shard}: stopped with {len(missing)} pairs missing (resume by rerunning)", flush=True)
        return
    # merge the pairs of the shard into one file
    ks, rs, inj, sec, edges = [], [], [], [], []
    for k, r in mine:
        v = np.load(pair_path(k, r))
        ks.append(k); rs.append(r); inj.append(v[:2 * nm].reshape(2, nm)); sec.append(v[2 * nm])
        edges.append(v[2 * nm + 1:2 * nm + 4] if len(v) >= 2 * nm + 4 else np.full(3, np.nan))
    tmp = fn + ".tmp.npz"
    np.savez_compressed(tmp, rho_idx=np.array(ks), rep=np.array(rs), inj=np.stack(inj),
                        sec=np.array(sec), edges=np.stack(edges), host=np.array(host))
    os.replace(tmp, fn)
    shutil.rmtree(pdir, ignore_errors=True)
    status(len(mine), (time.time() - t0) / max(1, len(secs)), 0.0, True, float(np.mean(sec)), state="complete")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] shard {shard}: {len(mine)} pairs complete "
          f"({len(secs)} computed now) in {fmt_s(time.time() - t0)} -> {fn}", flush=True)


if __name__ == "__main__":
    main()
