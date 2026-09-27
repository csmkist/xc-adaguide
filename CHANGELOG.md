# Changelog

## 1.0.0 (2026-09)

First public release, used for all results in the paper.

- Shifted optimum drawn inside the central 80% of each box, so it is always feasible; checked by a self-test for axis-aligned and rotated functions.
- Sobol outputs centred before estimation; S1 reported unclipped; standard error of ST returned.
- Property 1 checked on the clipped bounds, for dimensions that meet the interior condition.
- Consistency test: one statistic (mean per-seed r) for the point estimate, bootstrap CI and permutation p; threshold r_min calibrated per dimension; Spearman and Kendall reported.
- Fixed seeds for permutation and bootstrap streams (zlib.crc32).
- Maglev case study uses the same optimizer core as the benchmark study.
- Probability matching with a floor p_min = 0.05; JADE Lehmer mean for mu_F.
- Local importance steps inward at a bound and divides by the actual displacement.
- Eq. (12) returns w_hat = 0.5 when all weights coincide.
- Friedman test returns p = 1 for fully tied blocks.
- Optional switches, off by default: Eq. (12) gate, per-module timing, per-run output of the parameter sweep.
- Extended experiments P1-P8 and the 12-gain quadrotor case study.
