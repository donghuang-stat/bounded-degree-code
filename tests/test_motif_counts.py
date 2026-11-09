#!/usr/bin/env python3
"""Independent numerical and reproducibility checks for the exact counter.

Run with ``python3 -m unittest discover -s tests -p test_motif_counts.py -v``.
All 356 motifs with 3..8 edges are checked against direct injective enumeration
at n=7, including the zero count of motifs with more than seven vertices.  The
counter's batched quotient branch is exercised with chunk sizes 1, 2 and 3.
Temporary caches are removed after the test; no example or campaign is changed.
"""

from collections import Counter
import itertools
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from motiflib import MotifLibrary, centered, sample_pair
from run_shard import pair_seed, shard_tasks


def direct_injective_count(matrix, motif):
    """Enumerate vertex injections; no quotient expansion or compiler is used."""
    if motif["v"] > len(matrix):
        return 0.0
    terms = []
    for image in itertools.permutations(range(len(matrix)), motif["v"]):
        product = 1.0
        for left, right in motif["edges"]:
            product *= float(matrix[image[left], image[right]])
        terms.append(product)
    # fsum makes the oracle's summation independent of NumPy/BLAS reductions.
    return math.fsum(terms)


def scheduled_graphs(config, reverse_shards=False):
    """Materialize the same pair IDs under a different shard/order schedule."""
    count = math.ceil(sum(config["reps"]) / config["shard_size"])
    shards = list(range(count))
    if reverse_shards:
        shards.reverse()
    answer = {}
    for shard in shards:
        for key in shard_tasks(config, shard):
            if key in answer:
                raise AssertionError("duplicate pair ID in the shard schedule")
            rho_index, replicate = key
            generator = np.random.default_rng(pair_seed(config, rho_index, replicate))
            answer[key] = sample_pair(generator, config["n"], config["p"],
                                      config["rhos"][rho_index])
    return answer


class ExactMotifCountTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="motif-count-test-")
        cls.library = MotifLibrary(3, 8, cache_dir=cls.temporary.name)
        generator = np.random.default_rng(719041)
        upper = np.triu_indices(7, 1)
        graph = np.zeros((7, 7))
        graph[upper] = generator.random(len(upper[0])) < 0.2
        graph += graph.T
        bernoulli = centered(graph, 0.2)
        weighted = np.zeros((7, 7))
        weighted[upper] = generator.uniform(-0.45, 0.7, len(upper[0]))
        weighted += weighted.T
        cls.matrices = {"centered_bernoulli": bernoulli, "general_signed": weighted}
        cls.reference = {}
        cls.actual = {}
        for label, matrix in cls.matrices.items():
            cls.reference[label] = np.array([
                direct_injective_count(matrix, motif) for motif in cls.library.motifs
            ])
            cls.actual[label] = {
                chunk: cls.library.inj(matrix, chunk_size=chunk) for chunk in (1, 2, 3)
            }

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_complete_motif_family_and_batched_quotients(self):
        self.assertEqual(Counter(motif["m"] for motif in self.library.motifs),
                         {3: 3, 4: 5, 5: 12, 6: 30, 7: 79, 8: 227})
        self.assertEqual(len(self.library.motifs), 356)
        # qtw stores the n-exponent: 4 denotes the treewidth-three branch.
        self.assertIn(4, self.library.qtw)
        self.assertTrue(self.library.phase2)
        self.assertTrue(any(self.library.batched))
        self.assertTrue(any(self.library.nodes[i][0] == "dot"
                            for i in self.library.phase2))

    def test_all_356_motifs_match_direct_injections(self):
        for label in self.matrices:
            for chunk in (1, 2, 3):
                with self.subTest(matrix=label, chunk_size=chunk):
                    np.testing.assert_allclose(
                        self.actual[label][chunk], self.reference[label],
                        rtol=1e-9, atol=2e-9,
                        err_msg="exact counter differs from direct vertex injections",
                    )

    def test_motifs_larger_than_host_have_zero_count(self):
        impossible = np.array([motif["v"] > 7 for motif in self.library.motifs])
        self.assertTrue(impossible.any())
        for label in self.matrices:
            self.assertTrue(np.array_equal(self.reference[label][impossible],
                                           np.zeros(impossible.sum())))
            np.testing.assert_allclose(self.actual[label][2][impossible], 0.0,
                                       rtol=0, atol=2e-9)

    def test_chunk_boundaries_and_tail_are_executed(self):
        observed = set()
        original = self.library._eval

        def record_chunk(key, values, n, chunk):
            if chunk is not None:
                observed.add(tuple(chunk))
            return original(key, values, n, chunk)

        self.library._eval = record_chunk
        try:
            actual = self.library.inj(self.matrices["general_signed"], chunk_size=2)
        finally:
            del self.library._eval
        self.assertEqual(observed, {(0, 2), (2, 4), (4, 6), (6, 7)})
        np.testing.assert_allclose(actual, self.reference["general_signed"],
                                   rtol=1e-9, atol=2e-9)
        for label in self.matrices:
            for chunk in (2, 3):
                np.testing.assert_allclose(self.actual[label][1], self.actual[label][chunk],
                                           rtol=1e-10, atol=2e-9)

    def test_correlated_graphs_have_the_requested_structure(self):
        p, n = 0.2, 80
        upper = np.triu_indices(n, 1)
        edge_trials = len(upper[0])
        density_tolerance = 6 * math.sqrt(p * (1 - p) / edge_trials)
        for index, rho in enumerate((0.0, 0.55, 0.8, 1.0)):
            generator = np.random.default_rng(np.random.SeedSequence([714, index]))
            first, second = sample_pair(generator, n, p, rho)
            for graph in (first, second):
                self.assertEqual(graph.shape, (n, n))
                self.assertTrue(np.array_equal(graph, graph.T))
                self.assertTrue(np.array_equal(np.diag(graph), np.zeros(n)))
                self.assertTrue(np.isin(graph, (0.0, 1.0)).all())
                self.assertLess(abs(float(graph[upper].mean()) - p), density_tolerance)
            measured = float(np.corrcoef(first[upper], second[upper])[0, 1])
            self.assertLess(abs(measured - rho), 6 / math.sqrt(edge_trials))
            if rho == 1:
                self.assertTrue(np.array_equal(first, second))

    def test_graph_identity_is_independent_of_shards_and_order(self):
        first = dict(n=18, p=0.2, rhos=[0.0, 0.55, 0.8], reps=[4, 3, 2],
                     seed=20417, shard_size=3, order="blocked")
        second = dict(first, shard_size=4, order="interleaved")
        graph_a = scheduled_graphs(first)
        graph_b = scheduled_graphs(second, reverse_shards=True)
        expected = {(index, replicate) for index, count in enumerate(first["reps"])
                    for replicate in range(count)}
        self.assertEqual(set(graph_a), expected)
        self.assertEqual(set(graph_b), expected)
        seed_states = set()
        for key in sorted(expected):
            for left, right in zip(graph_a[key], graph_b[key]):
                self.assertTrue(np.array_equal(left, right))
            state = tuple(pair_seed(first, *key).generate_state(4).tolist())
            seed_states.add(state)
            documented = np.random.SeedSequence([first["seed"], first["n"], key[0], key[1]])
            self.assertEqual(state, tuple(documented.generate_state(4).tolist()))
        self.assertEqual(len(seed_states), len(expected))
        changed = dict(first, seed=first["seed"] + 1)
        changed_graphs = scheduled_graphs(changed)
        self.assertTrue(any(not np.array_equal(graph_a[key][0], changed_graphs[key][0])
                            for key in expected))


if __name__ == "__main__":
    unittest.main(verbosity=2)
