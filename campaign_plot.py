#!/usr/bin/env python3
"""Campaign-level figures and tables: one panel per config (edge probability p), as in
Figures 6 and 7 of the paper.

Usage:  python campaign_plot.py CAMPAIGN.json [--partial]

Reads <out_dir of each config>/analysis[_partial]/summary.csv and paired.csv and writes
into the campaign out_dir (analysis[_partial]/):
  auc_vs_rho_all_p.pdf/.png     AUC vs rho, one panel per p, four families (connected,
                                degree three, tree, small) with DeLong 95% intervals and the
                                Gaussian predictions Phi(sqrt(S/2))
  power_vs_rho_all_p.pdf/.png   the same for the power at level 0.05
  campaign_summary.csv          all rows of all summary.csv files, with columns n and p
  campaign_paired.csv           all paired comparisons, with column p
  campaign_graph_check.csv      edge density / edge correlation of the sampled graphs per (p, rho)
  campaign_numbers.txt          key numbers (calibration, gaps to the predictions, paired
                                differences at rho in {0.55, 0.7, 0.8}, progress)
"""
import os, sys, csv, json, math, glob
import numpy as np
from scipy.stats import norm
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from motiflib import MotifLibrary
from aggregate import members
from run_shard import load_config, resolve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SQA = math.sqrt(0.3383218568992076)


def read_csv(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k in list(r):
            try:
                r[k] = float(r[k])
            except (TypeError, ValueError):
                pass
    return rows


def progress(cfg):
    """(pairs done, pairs total) from finished shards and per-pair checkpoints."""
    S = cfg["shard_size"]; T = sum(cfg["reps"])
    out = cfg["out_dir"]
    done, complete = 0, set()
    for f in glob.glob(os.path.join(out, "shard_*.npz")):
        if f.endswith(".tmp.npz"):
            continue
        k = int(os.path.basename(f)[6:11]); complete.add(k)
        done += min(S, T - k * S)
    for d in glob.glob(os.path.join(out, "partial", "shard_*")):
        if int(os.path.basename(d)[6:11]) in complete:
            continue
        done += len([f for f in glob.glob(os.path.join(d, "pair_*.npy")) if not f.endswith(".tmp.npy")])
    return done, T


def load_campaign(path):
    """Campaign JSON {"name", "title", "out_dir", "configs": [...]}; a plain config is
    treated as a campaign with one member (out_dir = its own out_dir)."""
    camp = json.load(open(path))
    if "configs" not in camp:
        camp = dict(name=camp.get("name", os.path.basename(path)), title=camp.get("name", ""),
                    out_dir=camp["out_dir"], configs=[os.path.abspath(path)])
    camp["out_dir"] = resolve(camp["out_dir"])
    base = os.path.dirname(os.path.abspath(path))
    paths = []
    for c in camp["configs"]:
        # config paths: relative to the package directory, else to the campaign file
        q = c if os.path.isabs(c) else (resolve(c) if os.path.exists(resolve(c)) else os.path.join(base, c))
        paths.append(q)
    camp["config_paths"] = paths
    return camp, [load_config(q) for q in paths]


def family_label(kind, lo, hi):
    """Use the same singleton/range labels as aggregate.py and plots.py."""
    return f"{kind}:{lo}" if lo == hi else f"{kind}:{lo}-{hi}"


def plot_families(mmin, mmax):
    sizes = tuple(range(mmin, mmax + 1))
    edges = f"{mmin} edges" if mmin == mmax else f"{mmin} to {mmax} edges"
    fams = [(family_label("ALL", mmin, mmax), "ALL", sizes, f"connected ({edges})",
             dict(color="#1f77b4", ls="-", marker="o")),
            (family_label("DEG3", mmin, mmax), "DEG3", sizes, f"degree three ({edges})",
             dict(color="#2ca02c", ls="-.", marker="^")),
            (family_label("TREE", mmin, mmax), "TREE", sizes, f"tree ({edges})",
             dict(color="#d62728", ls="--", marker="s"))]
    # The small family is useful only when it is a proper subset of ALL.
    if mmin + 1 < mmax:
        fams.append((family_label("ALL", mmin, mmin + 1), "ALL", (mmin, mmin + 1),
                     f"small ({mmin} or {mmin + 1} edges)",
                     dict(color="#7f7f7f", ls=":", marker="D")))
    return fams


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        raise SystemExit(__doc__)
    partial = "--partial" in sys.argv
    sub = "analysis_partial" if partial else "analysis"
    camp, cfgs = load_campaign(args[0])
    cout = os.path.join(camp["out_dir"], sub)
    os.makedirs(cout, exist_ok=True)
    mmin, mmax = cfgs[0]["mmin"], cfgs[0]["mmax"]
    lib = MotifLibrary(mmin, mmax, cache_dir=cfgs[0]["cache_dir"])
    e = np.array([M["m"] for M in lib.motifs], dtype=float)
    fams = plot_families(mmin, mmax)
    off = np.linspace(-0.006, 0.006, len(fams))

    def S_of(kind, sizes, rho):
        idx = members(lib, kind, sizes)
        return float(np.sum(rho ** (2 * e[idx])))

    data, allrows, allpaired, prog = [], [], [], []
    for cfg in cfgs:
        d, t = progress(cfg); prog.append((cfg["p"], d, t))
        path = os.path.join(cfg["out_dir"], sub, "summary.csv")
        if not os.path.exists(path):
            data.append((cfg, None, None)); continue
        rows = read_csv(path)
        pp = os.path.join(cfg["out_dir"], sub, "paired.csv")
        paired = read_csv(pp) if os.path.exists(pp) else []
        data.append((cfg, rows, paired))
        allrows += rows
        for x in paired:
            x["p"] = cfg["p"]; allpaired.append(x)
    if not allrows:
        print("campaign_plot: no summary.csv available yet"); return
    frac = sum(d for _, d, _ in prog) / sum(t for _, _, t in prog)
    note = f" (interim: {100 * frac:.0f}% of pairs)" if partial else ""
    title = f"{camp.get('title') or camp.get('name', '')}; n = {cfgs[0]['n']}" + note

    for key, ylab, fname in (("auc", "AUC", "auc_vs_rho_all_p"), ("power", "power at level 0.05", "power_vs_rho_all_p")):
        k = len(data); ncol = 2 if k > 1 else 1; nrow = math.ceil(k / ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(7.0, 2.75 * nrow + 0.5), sharex=True, sharey=True, squeeze=False)
        grid = np.linspace(0, 1, 201)
        handles = None
        for ax, (cfg, rows, _) in zip(axes.ravel(), data):
            for (lab, kind, sizes, desc, st), dx in zip(fams, off):
                S = np.array([S_of(kind, sizes, g) for g in grid])
                th = norm.cdf(np.sqrt(S / 2)) if key == "auc" else 1 - norm.cdf(norm.ppf(0.95) - np.sqrt(S))
                ax.plot(grid, th, color=st["color"], ls=st["ls"], lw=1.0, alpha=0.85)
                if rows is None:
                    continue
                rr = sorted([r for r in rows if r["family"] == lab], key=lambda r: r["rho"])
                if not rr:
                    continue
                x = np.array([r["rho"] for r in rr]); y = np.array([r[key] for r in rr])
                lo = np.array([r[key + "_lo"] for r in rr]); hi = np.array([r[key + "_hi"] for r in rr])
                ax.errorbar(x + dx, y, yerr=[np.maximum(y - lo, 0), np.maximum(hi - y, 0)], color=st["color"], marker=st["marker"], ms=3.0,
                            lw=0, elinewidth=0.9, capsize=1.4, label=desc)
            ax.axvline(SQA, color="k", ls=":", lw=0.8)
            ax.axhline(0.5 if key == "auc" else 0.05, color="gray", lw=0.6)
            ax.set_title(f"$p={cfg['p']:g}$" + ("" if rows is not None else " (no data yet)"))
            ax.grid(alpha=0.25); ax.set_xlim(-0.03, 1.03)
            ax.set_ylim(0.4 if key == "auc" else 0.0, 1.02)
            h, l = ax.get_legend_handles_labels()
            if h and handles is None:
                handles = (h, l)
        for ax in axes.ravel()[len(data):]:
            ax.set_visible(False)
        for ax in axes[-1]:
            ax.set_xlabel(r"correlation $\rho$")
        for ax in axes[:, 0]:
            ax.set_ylabel(ylab)
        if handles:
            fig.legend(*handles, loc="lower center", ncol=4, frameon=False, fontsize=7.5)
        fig.suptitle(title, fontsize=9)
        fig.tight_layout(rect=(0, 0.05, 1, 0.97))
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(cout, f"{fname}.{ext}"), dpi=150)
        plt.close(fig)

    with open(os.path.join(cout, "campaign_summary.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(allrows[0].keys())); wr.writeheader(); wr.writerows(allrows)
    if allpaired:
        with open(os.path.join(cout, "campaign_paired.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(allpaired[0].keys())); wr.writeheader(); wr.writerows(allpaired)
    gc = []
    for cfg in cfgs:
        gp = os.path.join(cfg["out_dir"], sub, "graph_check.csv")
        if os.path.exists(gp):
            for r in read_csv(gp):
                r["n"] = cfg["n"]; gc.append(r)
    if gc:
        with open(os.path.join(cout, "campaign_graph_check.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(gc[0].keys())); wr.writeheader(); wr.writerows(gc)
    labs = [x[0] for x in fams]
    with open(os.path.join(cout, "campaign_numbers.txt"), "w") as f:
        def P(*a):
            print(*a); print(*a, file=f)
        P(title)
        for p, d, t in prog:
            P(f"  p = {p:g}: {d} of {t} pairs done ({100 * d / t:.1f}%)")
        for cfg, rows, paired in data:
            if rows is None:
                continue
            P(f"===== p = {cfg['p']:g}")
            sel = [r for r in rows if r["family"] in labs]
            ti = [r["typeI"] for r in sel]
            hw = [r["auc_hi"] - r["auc"] for r in sel]
            gaps = [r["auc_theory"] - r["auc"] for r in sel if r["rho"] >= 0.5]
            P(f"  Type I error range {min(ti):.3f}..{max(ti):.3f}; max AUC CI half width {max(hw):.3f}")
            if gaps:
                P(f"  prediction minus empirical AUC for rho >= 0.5: {min(gaps):+.3f}..{max(gaps):+.3f}")
            for lab in labs:
                rr = sorted([r for r in sel if r["family"] == lab], key=lambda r: r["rho"])
                P(f"  {lab:9s} AUC " + " ".join(f"{r['rho']:g}:{r['auc']:.3f}" for r in rr))
            for x in paired:
                if x["rho"] in (0.55, 0.7, 0.8) and x["B"] in labs:
                    P(f"  paired rho={x['rho']:g} {x['A']} - {x['B']}: {x['diff']:+.3f} (z {x['z']:.1f})")
    print(f"campaign_plot: wrote {cout}")


if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)   # quiet under `| head`
    main()
