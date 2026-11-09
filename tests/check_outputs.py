#!/usr/bin/env python3
"""Check the outputs of a campaign (or of one config) after the final results job.

Usage:  python tests/check_outputs.py CAMPAIGN.json|CONFIG.json [--partial] [--recompute K]

Checks, for every config:
  * every shard file is present (not with --partial), every (rho index, replicate) pair occurs
    exactly once, and the number of pairs per rho equals "reps";
  * analysis files: summary.csv, paired.csv, null_check.csv, motifs.csv, graph_check.csv,
    all_pairs.npz and the figures of plots.py (PDF and PNG);
  * all_pairs.npz agrees with the shard files and every count is finite;
  * the sampled graphs: edge density close to p and edge correlation close to rho (a loose
    tolerance from the number of pairs), and the recorded edge counts of a few pairs equal
    those of the graphs regenerated from their seeds (graphs.py);
  * --recompute K: for K random pairs, the motif counts are recomputed from the regenerated
    graphs and must equal the stored ones exactly (default K = 1 if n <= 2000, else 0);
  * summary.csv: AUC in [0, 1], S equals sum rho^(2e) over the family, and the level 0.05
    test is calibrated: median Type I error over all rows <= 0.1 and maximum <= 0.4 (loose,
    so that it also holds for the small test campaigns; at n = 5000 expect about 0.05);
and for the campaign: the campaign figures and tables of campaign_plot.py, status.json.
Exit status 0 if everything passes, 1 otherwise.
"""
import os, sys, csv, glob, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from campaign_plot import load_campaign
from graphs import load_pair
from motiflib import MotifLibrary, centered

FAIL = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        FAIL.append(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("campaign"); ap.add_argument("--partial", action="store_true")
    ap.add_argument("--recompute", type=int, default=None)
    a = ap.parse_args()
    camp, cfgs = load_campaign(a.campaign)
    sub = "analysis_partial" if a.partial else "analysis"
    rng = np.random.default_rng(0)
    for cfg in cfgs:
        out = cfg["out_dir"]; S = cfg["shard_size"]; T = sum(cfg["reps"]); N = -(-T // S)
        print(f"=== {cfg.get('name', out)}  (n={cfg['n']}, p={cfg['p']}, {T} pairs, {N} shards)")
        files = sorted(f for f in glob.glob(os.path.join(out, "shard_*.npz")) if not f.endswith(".tmp.npz"))
        if not a.partial:
            check(len(files) == N, f"{len(files)} of {N} shard files present")
            check(not glob.glob(os.path.join(out, "partial", "shard_*", "pair_*.npy")),
                  "no leftover checkpoints of unfinished shards")
        ks, rs, inj, edges = [], [], [], []
        for f in files:
            d = np.load(f)
            ks.append(d["rho_idx"]); rs.append(d["rep"]); inj.append(d["inj"]); edges.append(d["edges"])
        if files:
            ks, rs, inj, edges = map(np.concatenate, (ks, rs, inj, edges))
            pairs = set(zip(ks.tolist(), rs.tolist()))
            check(len(pairs) == len(ks), "no duplicate pairs in the shard files")
            if not a.partial:
                cnt = np.bincount(ks, minlength=len(cfg["reps"]))
                check(cnt.tolist() == list(cfg["reps"]), f"pairs per rho = reps ({cnt.tolist()})")
            check(bool(np.all(np.isfinite(inj))), "all motif counts finite")
        ad = os.path.join(out, sub)
        names = ["summary.csv", "paired.csv", "null_check.csv", "motifs.csv", "graph_check.csv", "all_pairs.npz"]
        names += [f"{b}.{x}" for b in ("auc_vs_rho", "power_vs_rho", "auc_vs_size", "paired_auc_diff") for x in ("pdf", "png")]
        miss = [x for x in names if not os.path.exists(os.path.join(ad, x))]
        check(not miss, f"{sub}/ has {len(names) - len(miss)} of {len(names)} expected files" + (f" (missing {miss})" if miss else ""))
        ap_path = os.path.join(ad, "all_pairs.npz")
        if os.path.exists(ap_path):
            A = np.load(ap_path)
            if not a.partial and len(files):
                o1 = np.lexsort((rs, ks)); o2 = np.lexsort((A["rep"], A["rho_idx"]))
                same = (len(o1) == len(o2) and np.array_equal(inj[o1], A["inj"][o2])
                        and np.array_equal(edges[o1], A["edges"][o2]))
                check(same, f"all_pairs.npz ({len(A['rep'])} pairs) equals the shard files")
            # sampled graphs: density and edge correlation
            NN = cfg["n"] * (cfg["n"] - 1) / 2; p = cfg["p"]
            E = A["edges"]; d1 = E[:, 0] / NN; d2 = E[:, 1] / NN
            sd = np.sqrt(p * (1 - p) / NN / max(1, len(d1)))
            check(abs(d1.mean() - p) < 6 * sd + 1e-12 and abs(d2.mean() - p) < 6 * sd + 1e-12,
                  f"mean edge densities {d1.mean():.5f}, {d2.mean():.5f} vs p = {p}")
            rows = list(csv.DictReader(open(os.path.join(ad, "graph_check.csv"))))
            bad = []
            for r in rows:
                rho, c, m = float(r["rho"]), float(r["edge_corr"]), int(r["pairs"])
                tol = 6 * np.sqrt((1 + 1 / (p * (1 - p))) / (NN * m)) + 1e-9
                if abs(c - rho) > tol:
                    bad.append(f"rho={rho}: {c:.4f}")
            check(not bad, "empirical edge correlation matches rho for every rho" + (f" ({bad})" if bad else ""))
            # regenerate graphs from seeds
            idx = rng.choice(len(A["rep"]), size=min(3, len(A["rep"])), replace=False)
            ok = True
            for i in idx:
                k, r = int(A["rho_idx"][i]), int(A["rep"][i])
                G1, G2, _ = load_pair(cfg, k, r)
                e = np.array([G1.sum() / 2, G2.sum() / 2, (G1 * G2).sum() / 2])
                ok &= np.array_equal(e, A["edges"][i])
            check(ok, f"edge counts of {len(idx)} pairs regenerated from their seeds match")
            K = a.recompute if a.recompute is not None else (1 if cfg["n"] <= 2000 else 0)
            if K:
                lib = MotifLibrary(cfg["mmin"], cfg["mmax"], cache_dir=cfg["cache_dir"])
                ok = True
                for i in rng.choice(len(A["rep"]), size=min(K, len(A["rep"])), replace=False):
                    k, r = int(A["rho_idx"][i]), int(A["rep"][i])
                    G1, G2, _ = load_pair(cfg, k, r)
                    v = np.stack([lib.inj(centered(G, cfg["p"]), chunk_size=cfg.get("chunk_size")) for G in (G1, G2)])
                    ok &= np.array_equal(v, A["inj"][i])
                check(ok, f"motif counts of {K} pair(s) recomputed from the regenerated graphs match exactly")
        sp = os.path.join(ad, "summary.csv")
        if os.path.exists(sp):
            rows = list(csv.DictReader(open(sp)))
            auc = np.array([float(r["auc"]) for r in rows]); ti = np.array([float(r["typeI"]) for r in rows])
            check(bool(np.all((auc >= 0) & (auc <= 1))), f"{len(rows)} summary rows, AUC in [0, 1]")
            check(np.median(ti) <= 0.1 and ti.max() <= 0.4,
                  f"Type I error at level 0.05: median {np.median(ti):.3f}, max {ti.max():.3f} over {len(ti)} rows")
            lib = MotifLibrary(cfg["mmin"], cfg["mmax"], cache_dir=cfg["cache_dir"])
            from aggregate import members
            e = np.array([M["m"] for M in lib.motifs], float)
            err = max(abs(float(r["S"]) - np.sum(float(r["rho"]) ** (2 * e[members(lib, r["kind"], tuple(
                range(int(r["sizes"].split("-")[0]), int(r["sizes"].split("-")[1]) + 1)))]))) for r in rows)
            check(err < 1e-9, f"S column equals sum rho^(2e) over each family (max error {err:.1e})")
    if "configs" not in json.load(open(a.campaign)):     # a single config: no campaign files
        print(f"\n{'ALL CHECKS PASSED' if not FAIL else f'{len(FAIL)} CHECK(S) FAILED'}")
        sys.exit(1 if FAIL else 0)
    cout = os.path.join(camp["out_dir"], sub)
    names = [f"{b}.{x}" for b in ("auc_vs_rho_all_p", "power_vs_rho_all_p") for x in ("pdf", "png")]
    names += ["campaign_summary.csv", "campaign_paired.csv", "campaign_graph_check.csv", "campaign_numbers.txt"]
    print(f"=== campaign {camp.get('name')}")
    miss = [x for x in names if not os.path.exists(os.path.join(cout, x))]
    check(not miss, f"campaign {sub}/ has {len(names) - len(miss)} of {len(names)} files" + (f" (missing {miss})" if miss else ""))
    st = os.path.join(camp["out_dir"], "status.json")
    if os.path.exists(st):
        s = json.load(open(st))
        check(a.partial or s["all_complete"], f"status.json: {s['pairs_done']}/{s['pairs_total']} pairs, "
              f"all_complete={s['all_complete']}")
    if not a.partial:
        check(os.path.exists(os.path.join(camp["out_dir"], "manifest.json")), "manifest.json written")
    print(f"\n{'ALL CHECKS PASSED' if not FAIL else f'{len(FAIL)} CHECK(S) FAILED'}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
