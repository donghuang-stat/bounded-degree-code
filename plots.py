#!/usr/bin/env python3
"""Figures from <out_dir>/analysis/{summary.csv, paired.csv} (written by aggregate.py).

Usage: python plots.py CONFIG.json [--partial]   (--partial: read <out_dir>/analysis_partial)

Writes PDF and PNG files into <out_dir>/analysis/:
  auc_vs_rho      AUC vs rho for ALL:mmin-mmax, DEG3:mmin-mmax, TREE:mmin-mmax and ALL:mmin-(mmin+1);
                  points = simulation with DeLong 95% CI, lines = theory Phi(sqrt(S/2)),
                  S(rho) = sum_{M in family} rho^(2 e(M)) recomputed from the motif library
                  on a fine grid of rho in [0.3, 0.95]; dotted vertical line at sqrt(alpha).
  auc_vs_size     AUC of the single-size families ALL:m and TREE:m vs m, one panel per rho,
                  with theory Phi(sqrt(S/2)).
  power_vs_rho    power at level 0.05 (empirical threshold, Wilson 95% CI) for the families of
                  auc_vs_rho; lines = Gaussian approximation 1 - Phi(z_0.95 - sqrt(S)), i.e.
                  T ~ N(0, S) under the null and N(S, S) under the alternative (the same
                  approximation that gives AUC = Phi(sqrt(S/2))).
  paired_auc_diff paired DeLong AUC differences (A - B) with 95% CI, from paired.csv.
"""
import os, sys, csv, json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from motiflib import MotifLibrary
from aggregate import members
from run_shard import load_config

ALPHA_OTTER = 0.3383218568992076
SQRT_ALPHA = math.sqrt(ALPHA_OTTER)          # 0.58166...
Z95 = norm.ppf(0.95)
LEVEL = 0.05

# one style per family role: color AND line style AND marker differ
STYLE = {
    "ALL":   dict(color="#1f77b4", ls="-",  marker="o"),
    "DEG3":  dict(color="#2ca02c", ls="-.", marker="^"),
    "TREE":  dict(color="#d62728", ls="--", marker="s"),
    "ALL34": dict(color="#7f7f7f", ls=":",  marker="D"),
}
EXTRA_STYLES = [dict(color="#9467bd", ls=(0, (5, 1, 1, 1, 1, 1)), marker="v"),
                dict(color="#8c564b", ls=(0, (1, 1)), marker="P")]

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e3e3e3", "grid.linewidth": 0.6,
    "lines.linewidth": 1.6, "lines.markersize": 5, "errorbar.capsize": 2,
    "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def fam(kind, lo, hi):
    return f"{kind}:{lo}" if lo == hi else f"{kind}:{lo}-{hi}"


def read_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


def ebar(y, lo, hi):
    """Error-bar lengths; clipped at 0 (Wilson bounds at power 0 or 1 can be off by round-off)."""
    return [np.clip(y - lo, 0, None), np.clip(hi - y, 0, None)]


def save(fig, outdir, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), dpi=160)
    plt.close(fig)
    print(f"wrote {outdir}/{name}.pdf/.png")


def otter_line(ax, label=True):
    ax.axvline(SQRT_ALPHA, color="black", ls=(0, (1, 2)), lw=1.0, zorder=1)
    if label:
        ax.annotate(r"$\sqrt{\alpha}\approx 0.5817$", xy=(SQRT_ALPHA, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(3, -3), textcoords="offset points", ha="left", va="top", fontsize=8)


def main():
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    if len(args) != 1:
        raise SystemExit(__doc__)
    cfg = load_config(args[0])
    subdir = "analysis_partial" if "--partial" in sys.argv else "analysis"
    mmin, mmax, n, p = cfg["mmin"], cfg["mmax"], cfg["n"], cfg["p"]
    outdir = os.path.join(cfg["out_dir"], subdir)
    summ = read_csv(os.path.join(outdir, "summary.csv"))
    ppath = os.path.join(outdir, "paired.csv")
    paired = read_csv(ppath) if os.path.exists(ppath) else []
    by = {}
    for r in summ:
        by.setdefault(r["family"], []).append(r)
    for v in by.values():
        v.sort(key=lambda r: r["rho"])
    rhos = sorted({r["rho"] for r in summ})
    R0 = int(summ[0]["R0"]); R1s = sorted({int(r["R1"]) for r in summ})
    sub = (f"n = {n}, p = {p}, motif sizes {mmin}..{mmax}; null pairs {R0}, "
           f"alternative pairs {R1s[0] if len(R1s) == 1 else f'{R1s[0]}-{R1s[-1]}'} per rho")

    # number of motifs per edge count in each family, from the motif library
    lib = MotifLibrary(mmin, mmax, cache_dir=cfg["cache_dir"])
    e = np.array([M["m"] for M in lib.motifs], dtype=float)

    def S_of(kind, sizes, rho):
        idx = members(lib, kind, tuple(sizes))
        return np.sum(np.power.outer(np.asarray(rho, float), 2 * e[idx]), axis=-1)

    # ------------------------------------------------------------------ (i) and (iii)
    main_fams = [("ALL", "ALL", range(mmin, mmax + 1)), ("DEG3", "DEG3", range(mmin, mmax + 1)),
                 ("TREE", "TREE", range(mmin, mmax + 1))]
    if mmin + 1 < mmax:
        main_fams.append(("ALL34", "ALL", range(mmin, mmin + 2)))
    main_fams = [(role, kind, list(sz), fam(kind, sz[0], sz[-1])) for role, kind, sz in main_fams]
    main_fams = [f for f in main_fams if f[3] in by]

    # consistency: S recomputed from the library equals the S column of summary.csv
    worst = 0.0
    for role, kind, sz, lab in main_fams:
        for r in by[lab]:
            if r["S"] > 0:
                worst = max(worst, abs(S_of(kind, sz, r["rho"]) / r["S"] - 1))
    if worst > 1e-9:
        print(f"WARNING: S from the library differs from summary.csv (max rel. diff {worst:.2e})")

    grid = np.linspace(min(0.3, rhos[0]), max(0.95, rhos[-1]), 400)
    xlo = min(0.3, rhos[0]) - 0.02; xhi = max(0.95, rhos[-1]) + 0.02
    dx = 0.0035 * np.arange(len(main_fams)); dx -= dx.mean()   # small dodge for overlapping CIs

    def family_handles():
        h = [Line2D([], [], **{k: v for k, v in STYLE[role].items()}, label=lab) for role, _, _, lab in main_fams]
        return h

    for what in ("auc", "power"):
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for j, (role, kind, sz, lab) in enumerate(main_fams):
            st = STYLE[role]
            S = S_of(kind, sz, grid)
            th = norm.cdf(np.sqrt(S / 2)) if what == "auc" else 1 - norm.cdf(Z95 - np.sqrt(S))
            ax.plot(grid, th, color=st["color"], ls=st["ls"], lw=1.4, zorder=2)
            rr = by[lab]
            x = np.array([r["rho"] for r in rr]) + dx[j]
            y = np.array([r[what] for r in rr])
            lo = np.array([r[f"{what}_lo"] for r in rr]); hi = np.array([r[f"{what}_hi"] for r in rr])
            ax.errorbar(x, y, yerr=ebar(y, lo, hi), fmt=st["marker"], color=st["color"], mfc=st["color"],
                        mec="white", mew=0.6, ms=5.5, elinewidth=1.0, zorder=3)
        otter_line(ax)
        ax.set_xlim(xlo, xhi)
        ax.set_xlabel(r"correlation $\rho$")
        if what == "auc":
            ax.axhline(0.5, color="#9a9a9a", lw=0.8, zorder=1)
            ax.set_ylabel("AUC")
            ymin = min([0.45] + [r["auc"] - 0.03 for _, _, _, lab in main_fams for r in by[lab]])
            ax.set_ylim(max(0.0, ymin), 1.01)
            note = r"points: simulation, DeLong 95% CI;  lines: theory $\Phi(\sqrt{S/2})$"
        else:
            ax.axhline(LEVEL, color="#9a9a9a", lw=0.8, zorder=1)
            ax.set_ylabel(f"power at level {LEVEL}")
            ax.set_ylim(-0.02, 1.02)
            note = r"points: simulation, Wilson 95% CI;  lines: Gaussian approx. $1-\Phi(z_{0.95}-\sqrt{S})$"
        ax.set_title(sub + "\n" + note)
        ax.legend(handles=family_handles(), loc="upper left", frameon=False, handlelength=3.2)
        save(fig, outdir, "auc_vs_rho" if what == "auc" else "power_vs_rho")

    # ------------------------------------------------------------------ (ii) per-size AUC
    sizes = list(range(mmin, mmax + 1))
    alt = [r for r in rhos if r > 0]
    ncol = min(4, len(alt)); nrow = int(math.ceil(len(alt) / ncol))
    head = 0.68                                   # inches reserved for title + legend
    H = 2.3 * nrow + head
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol + 0.4, H),
                             sharex=True, sharey=True, squeeze=False)
    for a, rho in enumerate(alt):
        ax = axes[a // ncol][a % ncol]
        for j, (role, kind) in enumerate((("ALL", "ALL"), ("TREE", "TREE"))):
            st = STYLE[role]
            pts = [(m, r) for m in sizes for r in by.get(fam(kind, m, m), []) if r["rho"] == rho]
            if not pts:
                continue
            m_ = np.array([m for m, _ in pts], float)
            y = np.array([r["auc"] for _, r in pts]); th = np.array([r["auc_theory"] for _, r in pts])
            lo = np.array([r["auc_lo"] for _, r in pts]); hi = np.array([r["auc_hi"] for _, r in pts])
            ax.plot(m_, th, color=st["color"], ls=st["ls"], lw=1.3, zorder=2)
            ax.errorbar(m_ + (j - 0.5) * 0.16, y, yerr=ebar(y, lo, hi), fmt=st["marker"], color=st["color"],
                        mec="white", mew=0.6, ms=5, elinewidth=1.0, zorder=3)
        ax.axhline(0.5, color="#9a9a9a", lw=0.8, zorder=1)
        ax.set_title(rf"$\rho = {rho:g}$" + (r"  ($<\sqrt{\alpha}$)" if rho < SQRT_ALPHA else ""))
        ax.set_xticks(sizes)
    for a in range(len(alt), nrow * ncol):
        axes[a // ncol][a % ncol].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("motif size m (edges)")
    for row in axes:
        row[0].set_ylabel("AUC")
    handles = [Line2D([], [], **STYLE["ALL"], label="ALL:m (all connected motifs with m edges)"),
               Line2D([], [], **STYLE["TREE"], label="TREE:m (trees with m edges)")]
    # fig.text, not suptitle: tight_layout reserves space for a suptitle itself, which would
    # double the header; the header space is reserved explicitly through rect below
    fig.text(0.5, 1 - 0.05 / H, sub + "\n" + r"points: simulation, DeLong 95% CI;  lines: theory $\Phi(\sqrt{S/2})$",
             ha="center", va="top", fontsize=9)
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, handlelength=3.2,
               bbox_to_anchor=(0.5, 1 - 0.40 / H))
    fig.tight_layout(rect=(0, 0, 1, 1 - head / H))
    save(fig, outdir, "auc_vs_size")

    # ------------------------------------------------------------------ (iv) paired differences
    if paired:
        comps = []
        for r in paired:
            if (r["A"], r["B"]) not in comps:
                comps.append((r["A"], r["B"]))

        def style_for(B):
            kind, rng = B.split(":")
            if kind == "TREE":
                return STYLE["TREE"]
            if kind == "DEG3":
                return STYLE["DEG3"]
            if B == fam("ALL", mmin, mmin + 1):
                return STYLE["ALL34"]
            return None
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        k_extra = 0
        dxp = 0.0035 * np.arange(len(comps)); dxp -= dxp.mean()
        handles = []
        for j, (A, B) in enumerate(comps):
            st = style_for(B)
            if st is None or A.split(":")[1] != f"{mmin}-{mmax}":
                st = EXTRA_STYLES[k_extra % len(EXTRA_STYLES)]; k_extra += 1
            rr = sorted([r for r in paired if r["A"] == A and r["B"] == B], key=lambda r: r["rho"])
            seen = set(); rr = [r for r in rr if not (r["rho"] in seen or seen.add(r["rho"]))]
            x = np.array([r["rho"] for r in rr]) + dxp[j]
            d = np.array([r["diff"] for r in rr]); se = np.array([r["se"] for r in rr])
            ax.errorbar(x, d, yerr=norm.ppf(0.975) * np.abs(se), fmt=st["marker"], ls=st["ls"], lw=1.0,
                        color=st["color"], mec="white", mew=0.6, ms=5, elinewidth=1.0)
            handles.append(Line2D([], [], **st, label=f"{A} minus {B}"))
        ax.axhline(0.0, color="#9a9a9a", lw=0.8, zorder=1)
        otter_line(ax)
        ax.set_xlim(xlo, xhi)
        ax.set_xlabel(r"correlation $\rho$")
        ax.set_ylabel("paired AUC difference")
        ax.set_title(sub + "\npaired DeLong differences on the same graph pairs, 95% CI")
        ax.legend(handles=handles, loc="best", frameon=False, handlelength=3.2)
        save(fig, outdir, "paired_auc_diff")


if __name__ == "__main__":
    main()
