#!/usr/bin/env python3
"""Reproduce Figure 8: motif-family cardinalities and Gaussian predictions.

This module counts isomorphism *types*, not occurrences in a sampled graph.
Connected graph cardinalities are computed from Burnside's lemma and the
unlabelled-component Euler transform.  Tree cardinalities also use the independent
rooted/unrooted tree generating-function identity of Otter.  No network access,
motif-library expansion, or new graph simulation is needed.
"""

import argparse
from collections import Counter
import csv
from functools import lru_cache
import json
import math
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parent
OTTER_ALPHA = 0.3383218568992076
MAX_EDGES = 30


def _nonnegative_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(name + " must be a nonnegative integer")


def integer_partitions(n, minimum=1):
    """Yield increasing tuples representing the integer partitions of n."""
    if n == 0:
        yield ()
        return
    for first in range(minimum, n + 1):
        for rest in integer_partitions(n - first, first):
            yield (first,) + rest


def edge_orbits(partition):
    """Counts of unordered-pair orbit lengths induced by vertex cycles."""
    multiplicities = Counter(partition)
    lengths = sorted(multiplicities)
    result = Counter()
    for i, a in enumerate(lengths):
        count = multiplicities[a]
        # Edges within a vertex cycle, including the even antipodal orbit.
        result[a] += count * ((a - 1) // 2)
        if a % 2 == 0:
            result[a // 2] += count
        # Between two distinct vertex cycles having the same length.
        result[a] += count * (count - 1) // 2 * a
        for b in lengths[i + 1:]:
            common = math.gcd(a, b)
            result[a * b // common] += count * multiplicities[b] * common
    return {length: count for length, count in result.items() if count}


@lru_cache(maxsize=None)
def _orbit_factor(length, count, max_edges):
    return tuple((j * length, math.comb(count, j))
                 for j in range(min(count, max_edges // length) + 1))


def _fixed_graph_polynomial(orbits, max_edges):
    """Coefficients of product_L (1+y**L)**orbits[L], truncated in y."""
    polynomial = [1] + [0] * max_edges
    for length, count in sorted(orbits.items()):
        if length > max_edges:
            continue
        nonzero = [(i, coefficient) for i, coefficient in enumerate(polynomial) if coefficient]
        product = [0] * (max_edges + 1)
        for offset, multiplier in _orbit_factor(length, count, max_edges):
            for degree, coefficient in nonzero:
                if degree + offset <= max_edges:
                    product[degree + offset] += multiplier * coefficient
        polynomial = product
    return polynomial


def _convolution(left, right, max_degree):
    result = [0] * (max_degree + 1)
    for i, a in enumerate(left):
        if a:
            for j in range(max_degree - i + 1):
                if right[j]:
                    result[i + j] += a * right[j]
    return result


@lru_cache(maxsize=4)
def graph_count_tables(max_edges):
    """Return (all_graphs[v][m], connected_graphs[v][m]) as integer tuples.

    All graphs may have isolated vertices.  Their isomorphism types are averaged
    over vertex-permutation conjugacy classes; no labelled graphs are generated.
    The logarithmic derivative of the unlabelled-component product then extracts
    connected graphs.  For m edges a connected graph has at most m+1 vertices.
    """
    _nonnegative_integer(max_edges, "max_edges")
    if max_edges > MAX_EDGES:
        raise ValueError("this implementation is intended for at most 30 edges")
    width = max_edges + 1
    all_graphs = [[1] + [0] * max_edges]
    connected = [[0] * width]
    logarithmic = [[0] * width]
    for vertices in range(1, max_edges + 2):
        factorial = math.factorial(vertices)
        accumulated = [0] * width
        for partition in integer_partitions(vertices):
            denominator = math.prod(length ** count * math.factorial(count)
                                    for length, count in Counter(partition).items())
            class_size, remainder = divmod(factorial, denominator)
            if remainder:
                raise ArithmeticError("nonintegral conjugacy class size")
            polynomial = _fixed_graph_polynomial(edge_orbits(partition), max_edges)
            for edges, count in enumerate(polynomial):
                accumulated[edges] += class_size * count
        row = []
        for value in accumulated:
            quotient, remainder = divmod(value, factorial)
            if remainder:
                raise ArithmeticError("Burnside sum is not divisible by v!")
            row.append(quotient)
        all_graphs.append(row)

        # A(x,y)=product_(v,m) (1-x**v*y**m)**(-C[v,m]).
        # B_v is v times the coefficient of x**v in log A.
        derivative = [vertices * value for value in row]
        for k in range(1, vertices):
            contribution = _convolution(logarithmic[k], all_graphs[vertices - k], max_edges)
            derivative = [a - b for a, b in zip(derivative, contribution)]
        logarithmic.append(derivative)
        primitive = derivative[:]
        for divisor in range(2, vertices + 1):
            if vertices % divisor == 0:
                smaller = vertices // divisor
                for edges in range(max_edges // divisor + 1):
                    primitive[divisor * edges] -= smaller * connected[smaller][edges]
        component = []
        for value in primitive:
            quotient, remainder = divmod(value, vertices)
            if remainder or quotient < 0:
                raise ArithmeticError("invalid connected-component coefficient")
            component.append(quotient)
        if component[0] != int(vertices == 1) or any(component[1:vertices - 1]):
            raise ArithmeticError("connected-graph boundary conditions failed")
        connected.append(component)
    return tuple(map(tuple, all_graphs)), tuple(map(tuple, connected))


def connected_graph_counts(max_edges=30):
    """Isomorphism types by edge count; index 0 is the singleton graph."""
    _, connected = graph_count_tables(max_edges)
    return [sum(row[edges] for row in connected) for edges in range(max_edges + 1)]


@lru_cache(maxsize=8)
def rooted_tree_counts(max_vertices):
    """Rooted unlabelled trees, from R(x)=x exp(sum_k R(x**k)/k)."""
    _nonnegative_integer(max_vertices, "max_vertices")
    rooted = [0] * (max_vertices + 1)
    divisor_sum = [0] * (max_vertices + 1)
    for n in range(1, max_vertices + 1):
        if n == 1:
            rooted[n] = 1
        else:
            value = sum(divisor_sum[k] * rooted[n - k] for k in range(1, n))
            rooted[n], remainder = divmod(value, n - 1)
            if remainder:
                raise ArithmeticError("nonintegral rooted-tree coefficient")
        for multiple in range(n, max_vertices + 1, n):
            divisor_sum[multiple] += n * rooted[n]
    return tuple(rooted)


def unrooted_tree_counts(max_vertices):
    """Unrooted trees indexed by vertices; U=R-(R**2-R(x**2))/2."""
    rooted = rooted_tree_counts(max_vertices)
    result = [0] * (max_vertices + 1)
    for n in range(1, max_vertices + 1):
        convolution = sum(rooted[i] * rooted[n - i] for i in range(1, n))
        corrected = convolution - (rooted[n // 2] if n % 2 == 0 else 0)
        correction, remainder = divmod(corrected, 2)
        if remainder:
            raise ArithmeticError("nonintegral unrooted-tree coefficient")
        result[n] = rooted[n] - correction
    return result


def gaussian_auc(signal):
    """Phi(sqrt(signal/2)); this is a prediction, not a finite-sample identity."""
    if not math.isfinite(signal) or signal < 0:
        raise ValueError("signal must be finite and nonnegative")
    return 0.5 * math.erfc(-math.sqrt(signal) / 2)


def infinite_tree_signal(rho, min_edges=3):
    """Numerical infinite-tree signal below Otter's threshold.

    Evaluate the rooted-tree functional equation instead of assuming a geometric
    tail from a finite sequence.  For k>=2, rho**(2*k)<alpha**2<1/4, so the
    plane-rooted-tree bound controls the two truncated sums.  The remaining
    principal root lies in (0,1) and is found by bisection.  Defaults use 80 rooted
    coefficients and functional powers through k=60, far below double precision error at
    rho=.55.  This is a numerical generating-function evaluation, not symbolic
    infinite arithmetic.
    """
    _nonnegative_integer(min_edges, "min_edges")
    if min_edges > MAX_EDGES:
        raise ValueError("min_edges must not exceed 30")
    if not math.isfinite(rho) or rho < 0 or rho * rho >= OTTER_ALPHA:
        raise ValueError("infinite-tree signal requires 0 <= rho < sqrt(alpha)")
    q = rho * rho
    if q == 0:
        return 1.0 if min_edges == 0 else 0.0
    rooted = rooted_tree_counts(max(80, min_edges + 1))

    def small_rooted(z):
        return sum(coefficient * z ** n for n, coefficient in enumerate(rooted) if coefficient)

    remainder = sum(small_rooted(q ** k) / k for k in range(2, 61))
    target = q * math.exp(remainder)
    if target >= 1 / math.e:
        raise ValueError("rho is too close to the critical value for float64 evaluation")
    lo, hi = 0.0, 1.0
    for _ in range(80):
        middle = (lo + hi) / 2
        if middle * math.exp(-middle) < target:
            lo = middle
        else:
            hi = middle
    r = (lo + hi) / 2
    unrooted = r - (r * r - small_rooted(q * q)) / 2
    low_counts = unrooted_tree_counts(min_edges)
    excluded = sum(low_counts[m + 1] * q ** m for m in range(min_edges))
    return max(0.0, unrooted / q - excluded)


def family_counts(max_edges=30):
    all_counts = connected_graph_counts(max_edges)
    trees = unrooted_tree_counts(max_edges + 1)[1:]
    _, connected = graph_count_tables(max_edges)
    for m in range(max_edges + 1):
        if connected[m + 1][m] != trees[m]:
            raise ArithmeticError("Burnside tree boundary disagrees with Otter's formula")
    return {"connected": all_counts, "trees": trees}


def signal_rows(counts, rhos=(0.55, 0.6), min_edges=3):
    """Rows of single-size contributions and cumulative Gaussian predictions."""
    rows = []
    for rho in rhos:
        if not math.isfinite(rho) or not 0 <= rho <= 1:
            raise ValueError("rho must lie in [0,1]")
        for family in ("connected", "trees"):
            cumulative = 0.0
            for m in range(min_edges, len(counts[family])):
                contribution = counts[family][m] * rho ** (2 * m)
                cumulative += contribution
                rows.append(dict(edges=m, rho=rho, family=family, family_count=counts[family][m],
                                 signal_contribution=contribution, cumulative_signal=cumulative,
                                 auc_gaussian=gaussian_auc(cumulative)))
    return rows


def _empirical_rows(summary, rhos):
    with Path(summary).open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"n", "p", "rho", "kind", "sizes", "family", "R0", "R1", "auc", "auc_lo", "auc_hi"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("empirical summary is missing required metadata/statistic columns")
        source = list(reader)
    if not source or any(float(row["n"]) != 500 or float(row["p"]) != 0.5 for row in source):
        raise ValueError("Figure 8 empirical summaries must have n=500 and p=.5")
    result = []
    for row in source:
        if row["kind"] not in ("ALL", "TREE") or float(row["rho"]) not in rhos:
            continue
        low, high = map(int, row["sizes"].split("-"))
        if low != 3 or high > 8:
            continue
        # A single-size family is cumulative only for three edges.
        if row["family"] != (row["kind"] + (":3" if high == 3 else ":3-" + str(high))):
            continue
        if float(row["R0"]) != 1200 or float(row["R1"]) != 400:
            raise ValueError("Figure 8 empirical intervals require 1200 null and 400 alternative pairs")
        result.append(dict(edges=high, rho=float(row["rho"]),
                           family="connected" if row["kind"] == "ALL" else "trees",
                           auc=float(row["auc"]), auc_lo=float(row["auc_lo"]), auc_hi=float(row["auc_hi"])))
    expected = {(rho, family, edges) for rho in rhos for family in ("connected", "trees")
                for edges in range(3, 9)}
    actual = {(row["rho"], row["family"], row["edges"]) for row in result}
    if actual != expected or len(result) != len(expected):
        raise ValueError("summary must contain n=500,p=.5 cumulative ALL/TREE data at both rhos, 3..8 edges")
    return sorted(result, key=lambda row: (row["rho"], row["family"], row["edges"]))


def _write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_figure8(data_dir, out_dir, max_edges=30, rho_low=0.55, rho_high=0.6):
    """Write Figure 8 and its exact-cardinality/prediction tables; return paths."""
    if not 8 <= max_edges <= MAX_EDGES:
        raise ValueError("Figure 8 requires max_edges between 8 and 30")
    if not 0 < rho_low < math.sqrt(OTTER_ALPHA) < rho_high <= 1:
        raise ValueError("Figure 8 requires rho_low below and rho_high above sqrt(alpha)")
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    summary = data_dir / "n500_m38/p05/summary.csv"
    if not summary.is_file():
        raise FileNotFoundError("missing empirical summaries for Figure 8: " + str(summary))
    empirical = _empirical_rows(summary, (rho_low, rho_high))
    started = time.perf_counter()
    counts = family_counts(max_edges)
    rows = signal_rows(counts, (rho_low, rho_high))
    limit = infinite_tree_signal(rho_low)
    out_dir.mkdir(parents=True, exist_ok=True)
    count_path = out_dir / "family_counts.csv"
    signal_path = out_dir / "signal_growth.csv"
    metadata_path = out_dir / "figure8_metadata.json"
    _write_csv(count_path, [dict(edges=m, connected=counts["connected"][m], trees=counts["trees"][m])
                            for m in range(max_edges + 1)])
    _write_csv(signal_path, rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {"connected": dict(color="#1f77b4", linestyle="-", marker="o", label="all connected"),
             "trees": dict(color="#d62728", linestyle="--", marker="s", label="trees")}
    with plt.rc_context({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                         "legend.fontsize": 9, "pdf.fonttype": 42, "ps.fonttype": 42}):
        figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.25), constrained_layout=True)
        for family, kwargs in style.items():
            selected = [row for row in rows if row["rho"] == rho_low and row["family"] == family]
            axes[0].semilogy([r["edges"] for r in selected], [r["signal_contribution"] for r in selected],
                             color=kwargs["color"], linestyle=kwargs["linestyle"], label=kwargs["label"], linewidth=1.4)
        axes[0].set(xlabel="number of edges $m$", ylabel="signal score of size $m$", ylim=(1e-5, 2.0),
                    title=r"$\rho = %.2f < \sqrt{\alpha}$" % rho_low)
        axes[0].legend(loc="lower left", frameon=False)
        for ax, rho in zip(axes[1:], (rho_low, rho_high)):
            for family, kwargs in style.items():
                selected = [row for row in rows if row["rho"] == rho and row["family"] == family]
                ax.plot([r["edges"] for r in selected], [r["auc_gaussian"] for r in selected],
                        color=kwargs["color"], linestyle=kwargs["linestyle"], linewidth=1.4)
                observations = [r for r in empirical if r["rho"] == rho and r["family"] == family]
                values = [r["auc"] for r in observations]
                errors = [[max(0.0, r["auc"] - r["auc_lo"]) for r in observations],
                          [max(0.0, r["auc_hi"] - r["auc"]) for r in observations]]
                ax.errorbar([r["edges"] for r in observations], values, yerr=errors,
                            fmt=kwargs["marker"], color=kwargs["color"], markersize=3,
                            capsize=2, linewidth=0.8)
            sign = "<" if rho == rho_low else ">"
            ax.set(xlabel="largest number of edges $m$", ylabel="AUC", ylim=(0.5, 1.005),
                   title=r"$\rho = %.2f %s \sqrt{\alpha}$" % (rho, sign))
        axes[1].axhline(gaussian_auc(limit), color="#555555", linestyle=":", linewidth=1.0)
        for ax in axes:
            ax.set_xlim(2.5, max_edges + 0.5)
            ax.grid(alpha=0.22, which="both", linewidth=0.5)
        png_path, pdf_path = out_dir / "figure8.png", out_dir / "figure8.pdf"
        figure.savefig(png_path, dpi=220)
        figure.savefig(pdf_path)
        plt.close(figure)
    metadata = dict(figure=8, max_edges=max_edges, min_edges=3, rhos=[rho_low, rho_high],
                    cardinality_method="Burnside over vertex permutations, then the unlabelled-component Euler transform",
                    tree_count_method="Otter rooted/unrooted generating functions; checked against the connected-graph tree boundary",
                    zero_edge_convention="one connected singleton graph and one singleton tree",
                    curve_interpretation="Gaussian predictions from S=sum_m N_m*rho^(2m), not additional simulation results",
                    empirical_source="data/paper/n500_m38/p05/summary.csv", empirical_n=500, empirical_p=0.5,
                    empirical_intervals="DeLong 95% confidence intervals, 1200 shared null and 400 alternative pairs per rho",
                    infinite_tree_signal=limit, infinite_tree_auc=gaussian_auc(limit),
                    infinite_tree_method="numerical rooted-tree functional equation: 80 coefficients, powers k=2..60, bisection",
                    count_reference="https://oeis.org/A002905", elapsed_seconds=time.perf_counter() - started)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return [png_path, pdf_path, count_path, signal_path, metadata_path]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/paper")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "figures/paper")
    parser.add_argument("--max-edges", type=int, default=30)
    parser.add_argument("--rho-low", type=float, default=0.55)
    parser.add_argument("--rho-high", type=float, default=0.6)
    args = parser.parse_args()
    for path in render_figure8(args.data_dir, args.out_dir, args.max_edges, args.rho_low, args.rho_high):
        print(path)


if __name__ == "__main__":
    main()
