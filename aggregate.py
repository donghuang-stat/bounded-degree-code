#!/usr/bin/env python3
"""Aggregate the shards of one config and evaluate motif statistics.

For a family F of motifs, the statistic is
    T_F = sum_{M in F} rho^{e(M)} * w_M * inj(M, B_1) * inj(M, B_2),
    w_M = (n - v(M))! / (n! (p(1-p))^{e(M)} aut(M)),
which is the statistic of the paper with the weights of Proposition 2.1 (up to a
constant factor).  Under the null, E[T_F] = 0 and Var[T_F] = S_F = sum_F rho^{2e(M)};
under the alternative, E[T_F] = S_F, for every n.  For a family whose motifs all
have the same number of edges, the factor rho^{e(M)} is a constant and the test
does not depend on rho.

Outputs (in OUT_DIR = <out_dir>/analysis):
    summary.csv     AUC with DeLong 95% CI, empirical effect size, power at level 0.05
                    (threshold = empirical 95% quantile of an independent calibration half
                    of the null sample), theory AUC Phi(sqrt(S/2)), for every rho,
                    family and size range;
    paired.csv      paired DeLong comparisons on the same graph pairs;
    null_check.csv  empirical E_0[w_M inj_1^2] per motif (should be close to 1);
    motifs.csv      the motif library: index (= column of inj), name, edges, vertices,
                    automorphisms, maximum degree, tree, cycle, edge list;
    all_pairs.npz   all raw data: rho_idx, rho, rep, inj (pairs x 2 graphs x motifs, exact
                    signed injective counts), cpu_sec, edges (edges of G1, G2, common edges);
    graph_check.csv edge density and empirical edge correlation of the sampled graphs per rho
                    (should be close to p and rho), CPU seconds per pair.
Usage: python aggregate.py CONFIG.json [--partial]
  --partial: interim results from finished shards and per-pair checkpoints, written to
             <out_dir>/analysis_partial (used by monitor.py while the jobs run)
"""
import os, sys, json, glob, math, csv
import numpy as np
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from motiflib import MotifLibrary, log_norm_weight
from run_shard import load_config
Z975 = norm.ppf(0.975)


def load(cfg, w, include_partial=False, keep_raw=True):
    """Stream the shards and keep the normalized products w_M inj_1 inj_2 (and, for the null,
    the per-motif sums of w_M inj_1^2), so memory is R x n_motifs per rho.  Per-pair metadata
    (CPU seconds, edge counts) is always kept; the raw counts only if keep_raw.
    With include_partial=True, also read the per-pair checkpoints of unfinished shards
    (<out_dir>/partial/shard_*/pair_<rho_idx>_<rep>.npy) for interim results.  A pair is
    counted once even if it is present both in a shard file and in a checkpoint."""
    files = sorted(glob.glob(os.path.join(cfg["out_dir"], "shard_*.npz")))
    files = [f for f in files if not f.endswith(".tmp.npz")]
    complete = {int(os.path.basename(f)[6:11]) for f in files}
    K = len(cfg["rhos"]); nm = len(w)
    parts = {k: [] for k in range(K)}
    nsq, ncount = np.zeros(nm), 0
    seen, meta, raw = set(), [], []

    def add(ks, reps, inj, sec, edges):
        nonlocal nsq, ncount
        keep = np.array([(int(k), int(r)) not in seen for k, r in zip(ks, reps)], dtype=bool)
        if not keep.any():
            return
        ks, reps, inj, sec, edges = ks[keep], reps[keep], inj[keep], sec[keep], edges[keep]
        seen.update(zip(ks.tolist(), reps.tolist()))
        meta.append(np.column_stack([ks, reps, sec, edges]))
        if keep_raw:
            raw.append(inj)
        for k in np.unique(ks):
            sel = ks == k
            parts[int(k)].append((reps[sel], inj[sel, 0, :] * inj[sel, 1, :] * w))
            if k == 0:
                nsq += (w * inj[sel, 0, :] ** 2).sum(0); ncount += int(sel.sum())

    for f in files:
        d = np.load(f)
        R = len(d["rep"])
        edges = d["edges"] if "edges" in d else np.full((R, 3), np.nan)
        sec = d["sec"] if "sec" in d else np.full(R, np.nan)
        add(d["rho_idx"], d["rep"], d["inj"], sec, edges)
    npartial = 0
    if include_partial:
        ks, rs, inj, sec, edges = [], [], [], [], []
        for f in glob.glob(os.path.join(cfg["out_dir"], "partial", "shard_*", "pair_*.npy")):
            if f.endswith(".tmp.npy") or int(os.path.basename(os.path.dirname(f))[6:11]) in complete:
                continue
            k, r = os.path.basename(f)[5:-4].split("_")
            try:
                v = np.load(f)
            except Exception:          # being written right now
                continue
            ks.append(int(k)); rs.append(int(r)); inj.append(v[:2 * nm].reshape(2, nm)); sec.append(v[2 * nm])
            edges.append(v[2 * nm + 1:2 * nm + 4] if len(v) >= 2 * nm + 4 else np.full(3, np.nan))
        if ks:
            add(np.array(ks), np.array(rs), np.stack(inj), np.array(sec), np.stack(edges))
            npartial = len(ks)
    if not files and npartial == 0:
        raise SystemExit("no shards found")
    prods = {}
    for k, lst in parts.items():
        if not lst:
            continue
        reps = np.concatenate([a for a, _ in lst]); pr = np.concatenate([b for _, b in lst])
        prods[k] = pr[np.argsort(reps, kind="stable")]
    meta = np.concatenate(meta)
    order = np.lexsort((meta[:, 1], meta[:, 0]))
    meta = meta[order]
    raw = np.concatenate(raw)[order] if keep_raw else None
    return dict(prods=prods, null_sq=(nsq / ncount if ncount else None), nfiles=len(files),
                npartial=npartial, meta=meta, raw=raw)


def families(lib, mmin, mmax):
    """Named families: (label, kind, sizes)."""
    out = []
    sizes = list(range(mmin, mmax + 1))
    for kind in ("ALL", "DEG3", "TREE", "CYCLE"):
        for m in sizes:
            out.append((f"{kind}:{m}", kind, (m,)))
        for top in sizes[1:]:
            out.append((f"{kind}:{mmin}-{top}", kind, tuple(range(mmin, top + 1))))
    return out


def members(lib, kind, sizes):
    idx = []
    for a, M in enumerate(lib.motifs):
        if M["m"] not in sizes:
            continue
        if kind == "ALL" or (kind == "DEG3" and M["maxdeg"] <= 3) or (kind == "TREE" and M["tree"]) \
                or (kind == "CYCLE" and M["cycle"]):
            idx.append(a)
    return np.array(idx, dtype=int)


def statistic(prod, idx, e, rho):
    """prod: (R, nmotifs) normalized products w_M inj1 inj2."""
    if len(idx) == 0:
        return None
    return prod[:, idx] @ (rho ** e[idx])


def placements(pos, neg):
    # Mann-Whitney placements via ranks (O(N log N))
    allv = np.concatenate([pos, neg])
    from scipy.stats import rankdata
    r = rankdata(allv)
    rp = rankdata(pos); rn = rankdata(neg)
    npos, nneg = len(pos), len(neg)
    v10 = (r[:npos] - rp) / nneg            # P(pos > neg) for each positive
    v01 = 1.0 - (r[npos:] - rn) / npos      # P(pos > neg) for each negative
    return v10.mean(), v10, v01


def delong(pos, neg):
    a, v10, v01 = placements(pos, neg)
    se = math.sqrt(np.var(v10, ddof=1) / len(pos) + np.var(v01, ddof=1) / len(neg))
    return a, se


def delong_paired(posA, negA, posB, negB):
    a, x1, y1 = placements(posA, negA); b, x2, y2 = placements(posB, negB)
    s10 = np.cov(np.vstack([x1, x2])); s01 = np.cov(np.vstack([y1, y2]))
    se = math.sqrt((s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / len(posA)
                   + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / len(negA))
    return a - b, se


def wilson(k, N):
    if N == 0:
        return (float("nan"),) * 3
    ph = k / N; z = Z975
    c = (ph + z * z / (2 * N)) / (1 + z * z / N)
    h = z * math.sqrt(ph * (1 - ph) / N + z * z / (4 * N * N)) / (1 + z * z / N)
    return ph, c - h, c + h


def write_csv(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    os.replace(tmp, path)


def write_data(cfg, lib, L, outdir):
    """Raw data and checks that do not need a full design:
    motifs.csv      the motif library (index = column of inj), with family membership;
    all_pairs.npz   every pair: rho_idx, rho, rep, inj (pairs x 2 x motifs), cpu_sec,
                    edges (pairs x 3: edges of G1, of G2, common edges);
    graph_check.csv per rho: edge densities and empirical edge correlation of the sampled
                    graphs against p and rho, CPU seconds per pair."""
    rows = []
    for a, M in enumerate(lib.motifs):
        rows.append(dict(index=a, name=M["name"], edges=M["m"], vertices=M["v"], aut=M["aut"],
                         max_degree=M["maxdeg"], tree=int(bool(M["tree"])), cycle=int(bool(M["cycle"])),
                         edge_list=" ".join(f"{u}-{v}" for u, v in M["edges"])))
    write_csv(os.path.join(outdir, "motifs.csv"), rows)
    meta = L["meta"]
    k = meta[:, 0].astype(int); rhos = np.array(cfg["rhos"], float)
    if L["raw"] is not None:
        tmp = os.path.join(outdir, "all_pairs.tmp.npz")
        np.savez_compressed(tmp, rho_idx=k, rho=rhos[k], rep=meta[:, 1].astype(int), inj=L["raw"],
                            cpu_sec=meta[:, 2], edges=meta[:, 3:6], n=cfg["n"], p=cfg["p"],
                            rhos=rhos, motif_names=np.array([M["name"] for M in lib.motifs]),
                            motif_edges=np.array([M["m"] for M in lib.motifs]))
        os.replace(tmp, os.path.join(outdir, "all_pairs.npz"))
    N = cfg["n"] * (cfg["n"] - 1) / 2; p = cfg["p"]
    rows = []
    for kk in np.unique(k):
        sel = k == kk; E = meta[sel, 3:6]
        d1, d2, d12 = E[:, 0] / N, E[:, 1] / N, E[:, 2] / N
        corr = (d12 - d1 * d2) / np.sqrt(d1 * (1 - d1) * d2 * (1 - d2))
        rows.append(dict(rho_idx=int(kk), rho=rhos[kk], pairs=int(sel.sum()), p=p,
                         density1=np.mean(d1), density2=np.mean(d2), edge_corr=np.mean(corr),
                         edge_corr_sd=np.std(corr, ddof=1) if sel.sum() > 1 else float("nan"),
                         cpu_sec_mean=np.mean(meta[sel, 2]), cpu_sec_max=np.max(meta[sel, 2])))
    write_csv(os.path.join(outdir, "graph_check.csv"), rows)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--partial", action="store_true",
                    help="also use checkpoints of unfinished shards; write to <out_dir>/analysis_partial")
    a = ap.parse_args()
    cfg = load_config(a.config)
    lib = MotifLibrary(cfg["mmin"], cfg["mmax"], cache_dir=cfg["cache_dir"])
    n, p = cfg["n"], cfg["p"]
    lw = log_norm_weight(lib, n, p)
    w = np.exp(lw)
    e = np.array([M["m"] for M in lib.motifs], dtype=float)
    keep_raw = sum(cfg["reps"]) * 2 * len(w) * 8 < 2e9
    L = load(cfg, w, include_partial=a.partial, keep_raw=keep_raw)
    prods, nc, nfiles = L["prods"], L["null_sq"], L["nfiles"]
    data = prods
    outdir = os.path.join(cfg["out_dir"], "analysis_partial" if a.partial else "analysis")
    os.makedirs(outdir, exist_ok=True)
    write_data(cfg, lib, L, outdir)
    print(f"{nfiles} complete shards" + (f" + {L['npartial']} checkpointed pairs" if a.partial else "")
          + "; pairs per rho: " + ", ".join(f"{cfg['rhos'][k]}:{len(v)}" for k, v in sorted(data.items())))
    if 0 not in prods or len(prods) < 2:
        raise SystemExit("not enough data yet for the test statistics: need null pairs and at least one "
                         "correlated rho (raw data, graph check and motif table were written)")

    # null check
    if nc is not None:
        with open(os.path.join(outdir, "null_check.csv"), "w", newline="") as f:
            wr = csv.writer(f); wr.writerow(["motif", "m", "v", "aut", "mean_w_inj2"])
            for a, M in enumerate(lib.motifs):
                wr.writerow([M["name"], M["m"], M["v"], M["aut"], f"{nc[a]:.4f}"])
        print(f"null check E0[w inj^2]: mean {nc.mean():.3f}, range [{nc.min():.3f}, {nc.max():.3f}]")

    fams = [(lab, kind, sz, members(lib, kind, sz)) for lab, kind, sz in families(lib, cfg["mmin"], cfg["mmax"])]
    fams = [f for f in fams if len(f[3]) > 0]
    null = prods[0]
    R0 = len(null); half = R0 // 2
    if R0 < 20:
        raise SystemExit(f"not enough null pairs yet ({R0} < 20)")
    rows, stats = [], {}
    for k in sorted(data):
        if k == 0:
            continue
        rho = cfg["rhos"][k]
        # at rho = 0 the weights rho^e(M) vanish; any fixed weights give AUC 1/2 in
        # expectation there, and we use those for rho = 0.5
        rw = rho if rho > 0 else 0.5
        alt = prods[k]
        if len(alt) < 5:
            continue
        for lab, kind, sz, idx in fams:
            t0 = statistic(null, idx, e, rw); t1 = statistic(alt, idx, e, rw)
            stats[(k, lab)] = (t1, t0)
            S = float(np.sum(rho ** (2 * e[idx])))
            auc, se = delong(t1, t0)
            # power at level 0.05: calibrate on the first half of the null sample,
            # evaluate type I error on the second half
            thr = np.quantile(t0[:half], 0.95)
            ti = wilson(int(np.sum(t0[half:] > thr)), R0 - half)
            pw = wilson(int(np.sum(t1 > thr)), len(t1))
            eff = (t1.mean() - t0.mean()) / t0.std(ddof=1)
            # (rows with rho = 0: S = 0, auc_theory = 1/2)
            # the test analyzed in Theorem 2.2: threshold S/2 (E_0 T = 0, E_1 T = S)
            e1 = float(np.mean(t0 >= S / 2)) if S > 0 else float("nan")
            e2 = float(np.mean(t1 < S / 2)) if S > 0 else float("nan")
            rows.append(dict(n=n, p=p, rho=rho, family=lab, kind=kind, sizes=f"{min(sz)}-{max(sz)}",
                             n_motifs=len(idx), S=S, auc_theory=norm.cdf(math.sqrt(S / 2)),
                             auc=auc, auc_lo=auc - Z975 * se, auc_hi=auc + Z975 * se, auc_se=se,
                             power=pw[0], power_lo=pw[1], power_hi=pw[2], typeI=ti[0],
                             effect=eff, effect_theory=math.sqrt(S),
                             typeI_half=e1, typeII_half=e2, errsum_half=e1 + e2,
                             var0_over_S=t0.var(ddof=1) / S if S > 0 else float("nan"),
                             mean1_over_S=t1.mean() / S if S > 0 else float("nan"),
                             R0=R0, R1=len(t1)))
    if not rows:
        raise SystemExit("not enough data yet: every correlated rho has fewer than 5 pairs")
    write_csv(os.path.join(outdir, "summary.csv"), rows)

    # paired comparisons
    mm, MM = cfg["mmin"], cfg["mmax"]
    comps = [(f"ALL:{mm}-{MM}", f"TREE:{mm}-{MM}"), (f"ALL:{mm}-{MM}", f"DEG3:{mm}-{MM}"),
             (f"ALL:{mm}-{MM}", f"ALL:{mm}-{mm + 1}"), (f"ALL:{mm}-{MM}", f"ALL:{mm}-{MM - 2}"),
             (f"ALL:{MM}", f"TREE:{MM}")]
    prow = []
    for k in sorted(data):
        if k == 0:
            continue
        for A, Bn in comps:
            if (k, A) not in stats or (k, Bn) not in stats:
                continue
            (a1, a0), (b1, b0) = stats[(k, A)], stats[(k, Bn)]
            d, se = delong_paired(a1, a0, b1, b0)
            prow.append(dict(rho=cfg["rhos"][k], A=A, B=Bn, diff=d, se=se, z=d / se if se > 0 else float("nan")))
    if prow:
        write_csv(os.path.join(outdir, "paired.csv"), prow)
    print(f"wrote {outdir}/summary.csv ({len(rows)} rows) and paired.csv ({len(prow)} rows)")


if __name__ == "__main__":
    main()
