# Third-party dependencies

The MIT license in this repository applies to the original project code and
included research summary data. Third-party packages are installed separately;
their licenses and notices remain applicable to those packages. No dependency
source code is vendored here.

- [NumPy](https://numpy.org/doc/stable/license.html): array operations and BLAS access.
- [SciPy](https://scipy.org/): distributions and statistical utilities; consult its distributed license.
- [Matplotlib](https://matplotlib.org/stable/project/license.html): figure rendering.
- [pynauty](https://github.com/pdobsan/pynauty#license): graph isomorphism and automorphisms. Its upstream distribution is GPLv3, and the underlying nauty code has its own licensing terms. Consult the upstream distribution's COPYING and nauty COPYRIGHT files.

The Figure 8 implementation computes mathematical cardinalities independently
using Burnside's lemma, an Euler transform, and Otter's generating-function
identities. Known connected-graph cardinalities used to validate the implementation
are documented by [OEIS A002905](https://oeis.org/A002905), also cited in the manuscript.
The runtime does not download an OEIS database or depend on network access.
