#!/usr/bin/env python3
"""Measure the per-pair cost of run_shard.py on the current node, to rescale the configs.

A "pair" is what one task of run_shard.py does: sample a correlated Erdos-Renyi pair and
compute inj(M, A_j - p) for every motif M of the library, j = 1, 2.

Usage:
    python bench.py --n 200 --mmax 9 [--mmin 3] [--p 0.5] [--pairs 3] [--workers 1]
    python bench.py --config configs/cluster_100cpu_day.json [--pairs 3] [--workers 4]
                    [--cpus 4] [--target-min 40] [--cluster-cpus 100]

With --workers W > 1, W processes run at the same time (W x pairs pairs in total), which
includes the slowdown from sharing caches and memory bandwidth on a node; the reported
"CPU-seconds per pair" is wall time x W / number of pairs.  Run it inside an allocation
with the same --cpus-per-task as the array (e.g. `srun -c 4 --mem-per-cpu=6G python bench.py
--config ... --workers 4`).

Reports: library load/build time, seconds per pair, peak memory (measured RSS and a static
estimate from the evaluation schedule), and with --config the projected CPU hours,
a shard_size for --target-min minutes per shard with --cpus workers, and the wall time
on --cluster-cpus CPUs.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, json, math, resource, sys, time
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from motiflib import MotifLibrary, sample_pair, centered

LIB, ARGS = None, None


def peak_rss_gib(value, platform=None):
    """Convert getrusage's bytes on macOS or KiB on Linux to GiB."""
    platform = sys.platform if platform is None else platform
    return value / 2**30 if platform == "darwin" else value / 2**20


def _init(args):
    global LIB, ARGS
    ARGS = args
    LIB = MotifLibrary(args.mmin, args.mmax, cache_dir=args.cache_dir)


def one_pair(i):
    a = ARGS
    rng = np.random.default_rng(np.random.SeedSequence([987654321, a.n, 99, i]))
    rho = 0.6
    t0 = time.perf_counter()
    A1, A2 = sample_pair(rng, a.n, a.p, rho)
    for A in (A1, A2):
        LIB.inj(centered(A, a.p), chunk_size=a.chunk_size)
    return time.perf_counter() - t0


def static_peak_bytes(lib, n, chunk_size=None):
    """Peak bytes of the arrays alive in MotifLibrary.homs (from its free-after schedule),
    plus three batched matrices for temporaries.  Python/library overhead not included."""
    if chunk_size is None:
        chunk_size = max(1, min(n, int(2.0e7 // (n * n))))
    c = chunk_size

    def size(i):
        kind, b = lib.nodes[i][0], lib.batched[i]
        mult = c if b else 1
        if kind in ("B", "ser", "had"):
            return mult * n * n * 8
        if kind in ("mv", "vp", "rows"):
            return mult * n * 8
        return mult * 8
    alive, powers, peak1 = {}, set(), 0
    for pos, i in enumerate(lib.phase1):
        key = lib.nodes[i]
        if key[0] == "B":
            powers.add(key[1]); alive[i] = 0
        else:
            alive[i] = size(i)
        peak1 = max(peak1, sum(alive.values()) + len(powers) * n * n * 8)
        for ch in lib.free_after.get(pos, ()):
            alive.pop(ch, None)
    kept = sum(alive.values()) + len(powers) * n * n * 8
    alive2, peak2 = {}, 0
    for pos, i in enumerate(lib.phase2):
        if lib.nodes[i][0] != "dot":
            alive2[i] = size(i)
        peak2 = max(peak2, sum(alive2.values()))
        for ch in lib.free_after2.get(pos, ()):
            alive2.pop(ch, None)
    return max(peak1, kept + peak2) + (3 * c * n * n * 8 if lib.phase2 else 0), c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--n", type=int)
    ap.add_argument("--p", type=float)
    ap.add_argument("--mmin", type=int)
    ap.add_argument("--mmax", type=int)
    ap.add_argument("--chunk-size", type=int, default=None, help="as 'chunk_size' in a config")
    ap.add_argument("--pairs", type=int, default=3, help="pairs per worker (default 3)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--cpus", type=int, default=4, help="workers per array task, for the shard_size suggestion")
    ap.add_argument("--target-min", type=float, default=40.0, help="target minutes per shard (default 40)")
    ap.add_argument("--cluster-cpus", type=int, default=None, help="total CPUs, for the wall-time estimate")
    ap.add_argument("--cache-dir", default=os.path.join(HERE, "cache"))
    a = ap.parse_args()
    cfg = {}
    if a.config:
        with open(a.config) as f:
            cfg = json.load(f)
        if cfg.get("cache_dir"):
            a.cache_dir = cfg["cache_dir"]
        if a.chunk_size is None:
            a.chunk_size = cfg.get("chunk_size")
    for k, dflt in (("n", None), ("p", 0.5), ("mmin", 3), ("mmax", None)):
        if getattr(a, k) is None:
            setattr(a, k, cfg.get(k, dflt))
    if a.n is None or a.mmax is None:
        ap.error("give --n and --mmax, or --config")

    t0 = time.time()
    lib = MotifLibrary(a.mmin, a.mmax, cache_dir=a.cache_dir, verbose=True)
    print(f"library sizes {a.mmin}..{a.mmax}: {len(lib.motifs)} motifs, load/build {time.time() - t0:.1f}s")
    est, chunk = static_peak_bytes(lib, a.n, a.chunk_size)
    del lib

    total = a.pairs * a.workers
    t0 = time.perf_counter()
    if a.workers > 1:
        with Pool(a.workers, initializer=_init, initargs=(a,)) as pool:
            per = pool.map(one_pair, range(total), chunksize=1)
    else:
        _init(a)
        per = [one_pair(i) for i in range(total)]
    wall = time.perf_counter() - t0
    cpu_per_pair = wall * a.workers / total
    rss = peak_rss_gib(max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                           resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss))
    print(f"n={a.n} p={a.p} sizes {a.mmin}..{a.mmax}, {a.workers} worker(s), {total} pairs")
    print(f"  seconds per pair (per process): median {np.median(per):.3f}, min {min(per):.3f}, max {max(per):.3f}")
    print(f"  CPU-seconds per pair (wall x workers / pairs): {cpu_per_pair:.3f}")
    print(f"  peak RSS of one process: {rss:.2f} GiB measured; static estimate of the arrays "
          f"{est / 2**30:.2f} GiB (chunk_size {chunk})")
    print(f"  -> suggested --mem-per-cpu: {max(1, math.ceil(1.3 * max(rss, est / 2**30 + 0.15)))}G")

    if cfg:
        pairs = sum(cfg["reps"])
        cpu_s = pairs * cpu_per_pair
        ss = max(1, int(round(a.target_min * 60 * a.cpus / cpu_per_pair)))
        cur = cfg["shard_size"]
        if (a.n, a.mmin, a.mmax) != (cfg["n"], cfg["mmin"], cfg["mmax"]):
            print(f"NOTE: measured n={a.n}, sizes {a.mmin}..{a.mmax}, but the config has n={cfg['n']}, "
                  f"sizes {cfg['mmin']}..{cfg['mmax']}; the projection below uses the measured cost")
        print(f"config {a.config}: {pairs} pairs -> {cpu_s:.3g} CPU-s = {cpu_s / 3600:.0f} CPU-hours")
        print(f"  current shard_size {cur}: {math.ceil(pairs / cur)} shards, "
              f"{cur * cpu_per_pair / a.cpus / 60:.1f} min per shard with {a.cpus} workers")
        print(f"  shard_size for {a.target_min:g} min with {a.cpus} workers: {ss} "
              f"({math.ceil(pairs / ss)} shards)")
        if a.cluster_cpus:
            conc = a.cluster_cpus // a.cpus
            nsh = math.ceil(pairs / cur)
            waves = math.ceil(nsh / conc)
            print(f"  on {a.cluster_cpus} CPUs ({conc} tasks of {a.cpus}): ~{waves * cur * cpu_per_pair / a.cpus / 3600:.1f} h wall; "
                  f"24 h budget {a.cluster_cpus * 86400:.3g} CPU-s, this config uses {100 * cpu_s / (a.cluster_cpus * 86400):.0f}%")
            print(f"  reps scaling factor for 75% of a 24 h budget: {0.75 * a.cluster_cpus * 86400 / cpu_s:.3f}")


if __name__ == "__main__":
    main()
