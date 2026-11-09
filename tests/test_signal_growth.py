"""Independent cardinality and paper-value checks for the Figure 8 calculation."""
import math
import csv
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import signal_growth as growth


class SignalGrowthTests(unittest.TestCase):
    def test_edge_orbits_partition_all_unordered_pairs(self):
        for vertices in range(1, 11):
            class_sizes = 0
            for partition in growth.integer_partitions(vertices):
                orbits = growth.edge_orbits(partition)
                self.assertEqual(sum(length * count for length, count in orbits.items()),
                                 vertices * (vertices - 1) // 2)
                from collections import Counter
                z = math.prod(k ** v * math.factorial(v) for k, v in Counter(partition).items())
                class_sizes += math.factorial(vertices) // z
            self.assertEqual(class_sizes, math.factorial(vertices))

    def test_four_vertex_counts(self):
        all_graphs, connected = growth.graph_count_tables(10)
        self.assertEqual(all_graphs[4][:7], (1, 1, 2, 3, 2, 1, 1))
        self.assertEqual(connected[4][:7], (0, 0, 0, 2, 2, 1, 1))
        self.assertEqual([sum(row) for row in all_graphs[:6]], [1, 1, 2, 4, 11, 34])

    def test_connected_counts_against_primary_sequence(self):
        # OEIS A002905 b-file, checked 2026-11-09; implementation computes these
        # independently and never reads this test fixture or downloads the b-file.
        expected = [1, 1, 1, 3, 5, 12, 30, 79, 227, 710, 2322, 8071, 29503,
                    112822, 450141, 1867871, 8037472, 35787667, 164551477,
                    779945969, 3804967442, 19079312775, 98211456209, 518397621443,
                    2802993986619, 15510781288250, 87765472487659, 507395402140211,
                    2994893000122118, 18035546081743772, 110741792670074054]
        self.assertEqual(growth.connected_graph_counts(30), expected)

    def test_unrooted_trees_and_independent_burnside_boundary(self):
        trees = growth.unrooted_tree_counts(10)
        self.assertEqual(trees[1:], [1, 1, 1, 2, 3, 6, 11, 23, 47, 106])
        counts = growth.family_counts(30)
        self.assertEqual(sum(counts["trees"][3:9]), 92)
        self.assertEqual(sum(counts["connected"][3:9]), 356)

    def test_paper_predictions_and_crossing(self):
        counts = growth.family_counts(30)
        rows = growth.signal_rows(counts, (0.55,))
        connected = {r["edges"]: r for r in rows if r["family"] == "connected"}
        self.assertAlmostEqual(connected[8]["auc_gaussian"], 0.628, delta=0.001)
        self.assertLess(connected[20]["cumulative_signal"], 1)
        self.assertGreater(connected[21]["cumulative_signal"], 1)
        self.assertLess(connected[13]["auc_gaussian"], 0.65)
        self.assertGreater(connected[14]["auc_gaussian"], 0.65)
        self.assertAlmostEqual(connected[21]["auc_gaussian"], 0.76, delta=0.01)
        self.assertAlmostEqual(sum(counts["connected"][3:22]) / 1e10, 2.4, delta=0.05)

    def test_infinite_tree_line_against_long_independent_series(self):
        rho = 0.55
        trees = growth.unrooted_tree_counts(501)
        partial = math.fsum(trees[m + 1] * rho ** (2 * m) for m in range(3, 501))
        limit = growth.infinite_tree_signal(rho)
        # The rooted-tree bound at x=1/3 gives a tail <= this value.
        tail_bound = 3 * (3 * rho * rho) ** 501
        self.assertLess(tail_bound, 2.3e-21)
        self.assertAlmostEqual(limit, partial, delta=5e-15)
        self.assertAlmostEqual(limit, 0.12075194100294309, delta=5e-15)
        self.assertAlmostEqual(growth.gaussian_auc(limit), 0.597048656169832, delta=5e-15)
        with self.assertRaises(ValueError):
            growth.infinite_tree_signal(0.6)

    def test_invalid_inputs_and_empty_signal(self):
        self.assertEqual(growth.connected_graph_counts(0), [1])
        self.assertEqual(growth.gaussian_auc(0), 0.5)
        self.assertEqual(growth.infinite_tree_signal(0), 0)
        for invalid in (-1, 31, 1.2, True):
            with self.assertRaises(ValueError):
                growth.connected_graph_counts(invalid)

    def test_empirical_overlay_matches_paper_and_rejects_wrong_experiment(self):
        summary = Path(__file__).resolve().parents[1] / "data/paper/n500_m38/p05/summary.csv"
        rows = growth._empirical_rows(summary, (0.55, 0.6))
        self.assertEqual(len(rows), 24)
        point = next(row for row in rows if row["rho"] == 0.55 and
                     row["family"] == "connected" and row["edges"] == 8)
        self.assertAlmostEqual(point["auc"], 0.650, delta=0.001)
        with summary.open(newline="") as stream:
            reader = csv.DictReader(stream)
            fields, source = reader.fieldnames, list(reader)
        with tempfile.TemporaryDirectory() as directory:
            invalid = Path(directory) / "summary.csv"
            with invalid.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(dict(row, p="0.2") for row in source)
            with self.assertRaisesRegex(ValueError, "n=500 and p=.5"):
                growth._empirical_rows(invalid, (0.55, 0.6))


if __name__ == "__main__":
    unittest.main()
