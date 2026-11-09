#!/usr/bin/env python3
"""Recover the two graphs of any simulated pair.

Every pair (rho index k, replicate r) is sampled from its own seed
SeedSequence([seed, n, k, r]), so the graphs are determined by the config.  This script
loads them from <out_dir>/graphs/ when the run used "save_graphs": true, and otherwise
regenerates them (same code path as run_shard.py), checks them against the edge counts
stored with the pair, and optionally writes them out.

Usage:
    python graphs.py CONFIG.json K R                 # print a summary of pair (K, R)
    python graphs.py CONFIG.json K R --out FILE.npz  # also save A1, A2 (bit-packed upper
                                                     # triangles, numpy.packbits) to FILE.npz
    python graphs.py CONFIG.json K R --edgelist DIR  # write DIR/A1.txt, DIR/A2.txt (i j per line)

In Python:  from graphs import load_pair;  A1, A2 = load_pair(cfg, k, r)   (dense 0/1 arrays)
"""
import os, sys, glob, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from motiflib import sample_pair
from run_shard import load_config, pair_seed


def unpack(bits, n):
    iu = np.triu_indices(n, 1)
    x = np.unpackbits(bits, count=len(iu[0])).astype(float)
    A = np.zeros((n, n)); A[iu] = x
    return A + A.T


def load_pair(cfg, k, r):
    g = os.path.join(cfg["out_dir"], "graphs", f"pair_{k}_{r}.npz")
    if os.path.exists(g):
        d = np.load(g)
        return unpack(d["A1"], cfg["n"]), unpack(d["A2"], cfg["n"]), "stored"
    rng = np.random.default_rng(pair_seed(cfg, k, r))
    A1, A2 = sample_pair(rng, cfg["n"], cfg["p"], cfg["rhos"][k])
    return A1, A2, "regenerated"


def stored_edges(cfg, k, r):
    """Edge counts recorded by run_shard.py for this pair, or None."""
    for f in sorted(glob.glob(os.path.join(cfg["out_dir"], "shard_*.npz"))):
        if f.endswith(".tmp.npz"):
            continue
        d = np.load(f)
        if "edges" not in d:
            continue
        hit = np.nonzero((d["rho_idx"] == k) & (d["rep"] == r))[0]
        if len(hit):
            return d["edges"][hit[0]]
    for f in glob.glob(os.path.join(cfg["out_dir"], "partial", "shard_*", f"pair_{k}_{r}.npy")):
        v = np.load(f)
        return v[-3:]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config"); ap.add_argument("k", type=int); ap.add_argument("r", type=int)
    ap.add_argument("--out"); ap.add_argument("--edgelist")
    a = ap.parse_args()
    cfg = load_config(a.config)
    A1, A2, how = load_pair(cfg, a.k, a.r)
    e = np.array([A1.sum() / 2, A2.sum() / 2, (A1 * A2).sum() / 2])
    N = cfg["n"] * (cfg["n"] - 1) / 2
    print(f"pair (rho_idx={a.k}, rho={cfg['rhos'][a.k]}, rep={a.r}) {how}: n={cfg['n']}, p={cfg['p']}")
    print(f"  edges G1 {e[0]:.0f}, G2 {e[1]:.0f}, common {e[2]:.0f} (expected {N * cfg['p']:.0f}, "
          f"{N * cfg['p']:.0f}, {N * (cfg['p'] ** 2 + cfg['rhos'][a.k] * cfg['p'] * (1 - cfg['p'])):.0f})")
    se = stored_edges(cfg, a.k, a.r)
    if se is not None and np.all(np.isfinite(se)):
        ok = np.array_equal(se, e)
        print(f"  recorded edge counts {se.astype(int).tolist()}: {'match' if ok else 'MISMATCH'}")
        if not ok:
            sys.exit(1)
    if a.out:
        iu = np.triu_indices(cfg["n"], 1)
        np.savez_compressed(a.out, n=cfg["n"], p=cfg["p"], rho=cfg["rhos"][a.k], k=a.k, r=a.r,
                            A1=np.packbits(A1[iu].astype(bool)), A2=np.packbits(A2[iu].astype(bool)))
        print(f"  wrote {a.out}")
    if a.edgelist:
        os.makedirs(a.edgelist, exist_ok=True)
        for name, A in (("A1", A1), ("A2", A2)):
            i, j = np.nonzero(np.triu(A, 1))
            np.savetxt(os.path.join(a.edgelist, f"{name}.txt"), np.c_[i, j], fmt="%d")
        print(f"  wrote {a.edgelist}/A1.txt, A2.txt")


if __name__ == "__main__":
    main()
