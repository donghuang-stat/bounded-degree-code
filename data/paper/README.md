# Numerical summary data

These tables are exported from the three simulation campaigns described in
Section 5 of the November 2026 manuscript. Each campaign has four edge
probabilities, 6,000 graph pairs per probability, and 24,000 pairs in total.
There are 72,000 simulated graph pairs across the three campaigns.

Directories are `n500_m38/`, `n5000/`, and `n10000_m34/`. Each contains
`p001`, `p005`, `p02`, and `p05`, corresponding to p=0.01, 0.05, 0.2, and 0.5.

| File | Content |
|---|---|
| `summary.csv` | AUC, DeLong standard errors and intervals, power with Wilson intervals, Type I point estimates, moments and signal scores for each family and rho |
| `paired.csv` | Paired AUC differences and standard errors on the same graph pairs |
| `motifs.csv` | Motif column order, edge/vertex counts, automorphisms, maximum degree and edge lists |
| `null_check.csv` | Per-motif null second-moment diagnostics |
| `graph_check.csv` | Densities, edge correlations and elapsed time diagnostics for the generated graph pairs |

The first half of the 1,200 null pairs calibrates the power-test threshold;
the other half estimates Type I error. AUC uses all 1,200 null pairs and
400 alternative pairs at each rho. The null sample is shared across correlations.
The configuration contains separate null and alternative-control entries with
rho=0. The summary's rho=0 rows refer to the alternative control.

Simulation inputs are in `configs/` and `campaigns/`. For p=0.01, 0.05, 0.2,
and 0.5, the seeds are 5000001, 5000005, 5000020, and 5000050 respectively.
The per-pair SeedSequence additionally contains n, rho index and replicate index.

The tables support redrawing Figures 4-7 and the empirical points in Figure 8.
They are summaries, not the original per-pair counts. New ROC curves, alternative
weights or new raw-count checks require the pairwise NPZ files or regeneration
with the supplied configurations. Raw shards and adjacency matrices are not
included in this public package.

The original `cpu_sec_mean`/`cpu_sec_max` fields in `graph_check.csv` are elapsed
wall seconds per pair. They are not allocated CPU hours or process CPU time.

`manifest.json` records table checksums, sample sizes and configuration paths.
Only scientific tables are included; machine-specific job logs, usernames,
working directories and scheduler configuration are excluded.
