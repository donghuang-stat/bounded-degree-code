#!/usr/bin/env python3
"""Redraw manuscript Figures 4--8 from bundled tables and exact family counts.

No graphs are generated and no signed motif counts are recomputed. In Figures
4--7, lines are Gaussian predictions from motif metadata; points and 95%
intervals are the original simulation estimates. Figure 8 adds exact family
counts through signal_growth.py. Numbering follows the manuscript's Section 5.

Usage:
    python plot_paper.py
    python plot_paper.py --data-dir data/paper --out-dir figures/paper
    python plot_paper.py --figures 4 7
"""

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent
ALPHA_OTTER = 0.3383218568992076
SQRT_ALPHA = math.sqrt(ALPHA_OTTER)
P_CODES = ("p001", "p005", "p02", "p05")
P_VALUES = dict(zip(P_CODES, (0.01, 0.05, 0.2, 0.5)))
RHOS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 0.99)
DESIGNS = {
    "n500_m38": dict(n=500, mmin=3, mmax=8, counts=(356, 156, 92, 8)),
    "n5000": dict(n=5000, mmin=3, mmax=5, counts=(20, 16, 11, 8)),
    "n10000_m34": dict(n=10000, mmin=3, mmax=4, counts=(8, 7, 5)),
}
STYLES = {
    "ALL": dict(color="#1f77b4", linestyle="-", marker="o"),
    "DEG3": dict(color="#2ca02c", linestyle="-.", marker="^"),
    "TREE": dict(color="#d62728", linestyle="--", marker="s"),
    "SMALL": dict(color="#7f7f7f", linestyle=":", marker="D"),
}
PLOT_STYLE = {
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8,
    "axes.linewidth": 0.6, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "grid.color": "#dedede", "grid.linewidth": 0.45,
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.facecolor": "white",
}


@dataclass
class Experiment:
    dataset: str
    p_code: str
    n: int
    p: float
    mmin: int
    mmax: int
    summary: List[dict]
    motifs: List[dict]

    def rows(self, family):
        return sorted((row for row in self.summary if row["family"] == family),
                      key=lambda row: row["rho"])


@dataclass
class FamilySpec:
    key: str
    label: str
    count: int
    edge_counts: Dict[int, int]
    role: str


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_table(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Missing paper data file: {}".format(path))
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(bool(rows), "Empty paper data file: {}".format(path))
    return rows


def family_specs(experiment):
    mmin, mmax = experiment.mmin, experiment.mmax
    requests = [
        ("ALL", "ALL:{}-{}".format(mmin, mmax),
         "connected ({} to {} edges)".format(mmin, mmax), mmax),
        ("DEG3", "DEG3:{}-{}".format(mmin, mmax),
         "degree three ({} to {} edges)".format(mmin, mmax), mmax),
        ("TREE", "TREE:{}-{}".format(mmin, mmax),
         "tree ({} to {} edges)".format(mmin, mmax), mmax),
    ]
    if mmax == 4:
        requests[0] = ("ALL", "ALL:3-4", "connected = small (3 or 4 edges)", 4)
    else:
        requests.append(("SMALL", "ALL:3-4", "small (3 or 4 edges)", 4))
    answer = []
    for role, key, label, upper in requests:
        selected = [motif for motif in experiment.motifs
                    if mmin <= motif["edges"] <= upper
                    and (role not in ("DEG3", "TREE")
                         or (role == "DEG3" and motif["max_degree"] <= 3)
                         or (role == "TREE" and motif["tree"] == 1))]
        counts = {}
        for motif in selected:
            edge = motif["edges"]
            counts[edge] = counts.get(edge, 0) + 1
        answer.append(FamilySpec(key=key, label=label, count=len(selected),
                                 edge_counts=counts, role=role))
    return answer


def signal_score(family, rho):
    rho = np.asarray(rho, dtype=float)
    result = np.zeros_like(rho)
    for edges, count in family.edge_counts.items():
        result += count * rho ** (2 * edges)
    return result


def load_experiment(data_dir, dataset_name, p_code):
    require(dataset_name in DESIGNS, "Unknown paper dataset: {}".format(dataset_name))
    require(p_code in P_VALUES, "Unknown paper p panel: {}".format(p_code))
    directory = Path(data_dir) / dataset_name / p_code
    summary_path, motifs_path = directory / "summary.csv", directory / "motifs.csv"
    summary = read_table(summary_path)
    motifs = read_table(motifs_path)
    numeric = ("n", "p", "rho", "n_motifs", "S", "auc_theory", "auc", "auc_lo", "auc_hi",
               "power", "power_lo", "power_hi", "R0", "R1")
    try:
        for row in summary:
            for column in numeric:
                row[column] = float(row[column])
                require(math.isfinite(row[column]), "Nonfinite {} in {}".format(column, summary_path))
        for motif in motifs:
            for column in ("index", "edges", "vertices", "aut", "max_degree", "tree", "cycle"):
                motif[column] = int(motif[column])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid paper data in {}: {}".format(directory, exc)) from exc
    design = DESIGNS[dataset_name]
    experiment = Experiment(dataset_name, p_code, design["n"], P_VALUES[p_code],
                            design["mmin"], design["mmax"], summary, motifs)
    require(all(row["n"] == experiment.n and row["p"] == experiment.p for row in summary),
            "n/p metadata disagrees with {} / {}".format(dataset_name, p_code))
    require(sorted(m["index"] for m in motifs) == list(range(len(motifs))),
            "Motif column indices are incomplete: {}".format(motifs_path))
    require(len({m["name"] for m in motifs}) == len(motifs), "Duplicate motif names: {}".format(motifs_path))
    families = family_specs(experiment)
    require(tuple(f.count for f in families) == design["counts"],
            "Motif family counts disagree with the paper: {} / {}".format(dataset_name, p_code))
    for family in families:
        rows = experiment.rows(family.key)
        require(tuple(row["rho"] for row in rows) == RHOS,
                "Incomplete or duplicate rho values for {} in {}".format(family.key, summary_path))
        for row in rows:
            require(row["n_motifs"] == family.count and row["R0"] == 1200 and row["R1"] == 400,
                    "Family/sample metadata differs from the paper: {}".format(summary_path))
            require(0 <= row["auc"] <= 1 and 0 <= row["power"] <= 1,
                    "AUC/power outside [0,1]: {}".format(summary_path))
            for metric in ("auc", "power"):
                require(row[metric + "_lo"] <= row[metric] + 1e-12
                        and row[metric + "_hi"] >= row[metric] - 1e-12,
                        "Invalid {} interval in {}".format(metric, summary_path))
            score = float(signal_score(family, row["rho"]))
            require(math.isclose(row["S"], score, rel_tol=1e-10, abs_tol=1e-14),
                    "Signal score differs from motif metadata: {}".format(summary_path))
            require(abs(row["auc_theory"] - norm.cdf(math.sqrt(score / 2))) <= 1e-12,
                    "Gaussian prediction differs from motif metadata: {}".format(summary_path))
    return experiment


def load_campaign(data_dir, dataset_name):
    return [load_experiment(data_dir, dataset_name, p_code) for p_code in P_CODES]


def draw_panel(ax, experiment, metric, xlim, legend=False, annotate_threshold=False):
    families = family_specs(experiment)
    grid = np.linspace(max(0.0, xlim[0]), min(1.0, xlim[1]), 501)
    offsets = np.linspace(-0.006, 0.006, len(families))
    for family, offset in zip(families, offsets):
        style = STYLES[family.role]
        score = signal_score(family, grid)
        prediction = (norm.cdf(np.sqrt(score / 2)) if metric == "auc"
                      else norm.sf(norm.ppf(0.95) - np.sqrt(score)))
        ax.plot(grid, prediction, color=style["color"], linestyle=style["linestyle"], linewidth=1)
        rows = [row for row in experiment.rows(family.key) if xlim[0] <= row["rho"] <= xlim[1]]
        x = np.array([row["rho"] for row in rows])
        y = np.array([row[metric] for row in rows])
        lower = np.array([row[metric + "_lo"] for row in rows])
        upper = np.array([row[metric + "_hi"] for row in rows])
        ax.errorbar(x + offset, y, yerr=[np.maximum(0, y - lower), np.maximum(0, upper - y)],
                    fmt=style["marker"], color=style["color"], markersize=2.8,
                    markeredgewidth=0.35, markeredgecolor="white", elinewidth=0.7,
                    capsize=1.1, capthick=0.7, label=family.label, zorder=3)
    ax.axvline(SQRT_ALPHA, color="black", linestyle=":", linewidth=0.7, zorder=1)
    ax.set_xlim(*xlim)
    ax.grid(True)
    if metric == "auc":
        ax.axhline(0.5, color="#777777", linewidth=0.55, zorder=1)
    if annotate_threshold:
        ax.annotate(r"$\sqrt{\alpha}$", xy=(SQRT_ALPHA, 0.505), xytext=(2, 0),
                    textcoords="offset points", fontsize=7, ha="left", va="bottom")
    if legend:
        ax.legend(loc="upper left", fontsize=6.5, handlelength=1.8,
                  borderaxespad=0.5, labelspacing=0.4)


def build_figure4(experiment):
    require(experiment.dataset == "n500_m38" and experiment.p == 0.5,
            "Figure 4 requires n500_m38 / p05")
    import matplotlib.pyplot as plt
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
        for ax, metric in zip(axes, ("auc", "power")):
            draw_panel(ax, experiment, metric, (0.37, 0.83), legend=(metric == "auc"),
                       annotate_threshold=(metric == "auc"))
            ax.set_xlabel(r"correlation $\rho$")
            ax.set_xticks(np.arange(0.4, 0.81, 0.1))
        axes[0].set(ylabel="AUC", ylim=(0.49, 1.025))
        axes[0].set_yticks(np.arange(0.5, 1.01, 0.1))
        axes[1].set(ylabel="power at level 0.05", ylim=(0.0, 1.05))
        axes[1].set_yticks(np.arange(0.2, 1.01, 0.2))
        fig.subplots_adjust(left=0.075, right=0.99, bottom=0.18, top=0.97, wspace=0.24)
    return fig


def build_campaign_figure(experiments):
    require(len(experiments) == 4 and tuple(e.p_code for e in experiments) == P_CODES,
            "A paper campaign figure requires panels p001, p005, p02, p05 in order")
    require(len({e.dataset for e in experiments}) == 1,
            "All four panels must belong to the same campaign")
    import matplotlib.pyplot as plt
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.15), sharex=True, sharey=True)
        for ax, experiment in zip(axes.flat, experiments):
            draw_panel(ax, experiment, "auc", (-0.025, 1.025))
            ax.set_title(r"$p={:g}$".format(experiment.p), pad=4)
            ax.set_ylim(0.4, 1.02)
            ax.set_xticks(np.arange(0, 1.01, 0.2))
            ax.set_yticks(np.arange(0.4, 1.01, 0.1))
        for ax in axes[:, 0]:
            ax.set_ylabel("AUC")
        for ax in axes[-1, :]:
            ax.set_xlabel(r"correlation $\rho$")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.005),
                   ncol=len(labels), fontsize=6.8, handlelength=1.7,
                   columnspacing=1.5, handletextpad=0.6)
        fig.subplots_adjust(left=0.075, right=0.99, bottom=0.14, top=0.955,
                            wspace=0.065, hspace=0.25)
    return fig


def render_paper_figures(data_dir, out_dir, figures=None):
    selected = tuple((4, 5, 6, 7, 8) if figures is None else figures)
    require(bool(selected) and len(set(selected)) == len(selected)
            and all(number in (4, 5, 6, 7, 8) for number in selected),
            "Select distinct figure numbers from 4, 5, 6, 7, 8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Read and validate the simulation tables before writing their figures.
    dataset_for_figure = {5: "n500_m38", 6: "n5000", 7: "n10000_m34"}
    campaigns = {dataset_for_figure[number]: load_campaign(data_dir, dataset_for_figure[number])
                 for number in selected if number in dataset_for_figure}
    focus = None
    if 4 in selected:
        focus = (campaigns["n500_m38"][-1] if "n500_m38" in campaigns
                 else load_experiment(data_dir, "n500_m38", "p05"))
    if 8 in selected:
        import signal_growth
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for number in selected:
        if number == 8:
            growth_outputs = signal_growth.render_figure8(data_dir, out_dir)
            written.extend(growth_outputs)
            for path in growth_outputs:
                print("Wrote {}".format(path))
            continue
        figure = (build_figure4(focus) if number == 4
                  else build_campaign_figure(campaigns[dataset_for_figure[number]]))
        for extension in ("png", "pdf"):
            path = out_dir / "figure{}.{}".format(number, extension)
            with plt.rc_context(PLOT_STYLE):
                figure.savefig(str(path), dpi=300)
            written.append(path)
            print("Wrote {}".format(path))
        plt.close(figure)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/paper")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "figures/paper")
    parser.add_argument("--figures", type=int, nargs="+", choices=(4, 5, 6, 7, 8),
                        default=(4, 5, 6, 7, 8), help="figure numbers to redraw (default: all five)")
    args = parser.parse_args()
    try:
        render_paper_figures(args.data_dir, args.out_dir, args.figures)
    except (OSError, ValueError) as exc:
        parser.exit(2, "plot_paper: {}\n".format(exc))


if __name__ == "__main__":
    main()
