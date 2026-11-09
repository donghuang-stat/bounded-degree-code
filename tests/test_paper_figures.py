"""Numerical and structural checks for the figures drawn from archived tables.

Run from the repository root with:
    python -m unittest discover -s tests -p 'test_paper_figures.py' -v

These tests read the bundled paper data without rerunning any simulations.
"""

import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "paper"
sys.path.insert(0, str(ROOT))
import plot_paper


class PaperDataTests(unittest.TestCase):
    def test_figure4_reference_auc_values(self):
        experiment = plot_paper.load_experiment(DATA, "n500_m38", "p05")
        expected = {
            "ALL:3-8": 0.6501333333333332,
            "TREE:3-8": 0.6053624999999999,
            "ALL:3-4": 0.6233333333333333,
        }
        for family, auc in expected.items():
            with self.subTest(family=family):
                selected = [row for row in experiment.rows(family)
                            if float(row["rho"]) == 0.55]
                self.assertEqual(len(selected), 1)
                self.assertAlmostEqual(float(selected[0]["auc"]), auc, places=12)

    def test_n10000_empirical_zero_correlation_point_is_preserved(self):
        experiment = plot_paper.load_experiment(DATA, "n10000_m34", "p05")
        selected = [row for row in experiment.rows("ALL:3-4")
                    if float(row["rho"]) == 0.0]
        self.assertEqual(len(selected), 1)
        row = selected[0]
        self.assertAlmostEqual(float(row["auc"]), 0.5446875, places=12)
        self.assertAlmostEqual(float(row["auc_lo"]), 0.5122624975575361, places=12)
        self.assertAlmostEqual(float(row["auc_hi"]), 0.5771125024424639, places=12)
        self.assertGreater(float(row["auc_lo"]), 0.5)

    def test_family_counts_and_duplicate_small_family(self):
        expected = {
            "n500_m38": [356, 156, 92, 8],
            "n5000": [20, 16, 11, 8],
            "n10000_m34": [8, 7, 5],
        }
        for dataset, counts in expected.items():
            with self.subTest(dataset=dataset):
                experiment = plot_paper.load_experiment(DATA, dataset, "p05")
                specs = plot_paper.family_specs(experiment)
                self.assertEqual(len(specs), len(counts))
                self.assertEqual(sorted((spec.count for spec in specs), reverse=True), counts)
                self.assertEqual(len({spec.key for spec in specs}), len(specs))
                for spec in specs:
                    self.assertTrue(spec.label)
                    self.assertEqual(sum(spec.edge_counts.values()), spec.count)
                    self.assertTrue(all(experiment.mmin <= int(edges) <= experiment.mmax
                                        for edges in spec.edge_counts))

    def test_campaign_order_metadata_and_sorted_rows(self):
        metadata = {"n500_m38": (500, 3, 8),
                    "n5000": (5000, 3, 5),
                    "n10000_m34": (10000, 3, 4)}
        expected_rhos = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6,
                         0.7, 0.8, 0.9, 0.99]
        for dataset, expected in metadata.items():
            with self.subTest(dataset=dataset):
                experiments = plot_paper.load_campaign(DATA, dataset)
                self.assertEqual(len(experiments), 4)
                self.assertEqual([experiment.p for experiment in experiments],
                                 [0.01, 0.05, 0.2, 0.5])
                for experiment in experiments:
                    self.assertEqual((experiment.n, experiment.mmin, experiment.mmax), expected)
                    self.assertTrue(experiment.summary)
                    self.assertTrue(experiment.motifs)
                    self.assertEqual([float(row["rho"]) for row in experiment.rows("ALL:3-4")],
                                     expected_rhos)

    def test_missing_tables_identify_the_missing_path(self):
        # Only temporary copies are changed; the bundled data remain untouched.
        for missing_name in ("summary.csv", "motifs.csv"):
            with self.subTest(missing=missing_name), tempfile.TemporaryDirectory() as directory:
                data_dir = Path(directory)
                target = data_dir / "n500_m38" / "p05"
                target.mkdir(parents=True)
                for name in ("summary.csv", "motifs.csv"):
                    if name != missing_name:
                        shutil.copyfile(DATA / "n500_m38" / "p05" / name, target / name)
                with self.assertRaises(FileNotFoundError) as raised:
                    plot_paper.load_experiment(data_dir, "n500_m38", "p05")
                missing_path = target / missing_name
                message = str(raised.exception)
                self.assertIn(missing_name, message)
                self.assertTrue(str(missing_path) in message
                                or str(missing_path.resolve()) in message, message)


class PaperFigureTests(unittest.TestCase):
    def test_figure4_has_two_populated_panels(self):
        experiment = plot_paper.load_experiment(DATA, "n500_m38", "p05")
        figure = plot_paper.build_figure4(experiment)
        try:
            self.assertEqual(len(figure.axes), 2)
            self.assertTrue(all(axis.has_data() for axis in figure.axes))
        finally:
            plt.close(figure)

    def test_figures5_to7_have_four_populated_panels(self):
        for dataset in ("n500_m38", "n5000", "n10000_m34"):
            with self.subTest(dataset=dataset):
                figure = plot_paper.build_campaign_figure(
                    plot_paper.load_campaign(DATA, dataset))
                try:
                    self.assertEqual(len(figure.axes), 4)
                    self.assertTrue(all(axis.has_data() for axis in figure.axes))
                finally:
                    plt.close(figure)


if __name__ == "__main__":
    unittest.main()
