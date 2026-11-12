# Hamilton (2018) Filter on US Real GDP

Demonstrates Hamilton's (2018) regression-based filter alongside HP and
classical Beveridge-Nelson decompositions on log real GDP. The paper
("Why You Should Never Use the Hodrick-Prescott Filter," *ReStat* 100(5))
argues that HP introduces spurious dynamics and that a simple
regression alternative does better.

## Specification

| | |
|---|---|
| **Series** | log real GDP × 100 (`GDPC1` from FRED-QD) |
| **Sample** | 1959Q1–2024Q4 |
| **Filters compared** | HP (λ=1600); Hamilton (h=8, p=4); Beveridge-Nelson (AR(p) by BIC); KMW-BN (AR(12), δ̄=0.25) |

The KMW (2018) Beveridge-Nelson filter is implemented in
`lib.kmw_bn_filter` via the closed-form Dickey-Fuller-style restriction
Σφ = 1 − 1/√δ̄ (see the KMW replication for the paper-faithful
treatment; the paper's own US δ̄ is 0.24 by automatic selection).

Mirrors [entries/hamilton_2018_filter.yaml](../../../skills/model-dictionary/entries/hamilton_2018_filter.yaml).

## Results

The Hamilton filter loses the first p+h−1 = 11 quarters. The paper's own
GDP cycle standard deviation is **3.38** (regression residuals,
1947:1–2016:1, Table 2 p. 840; random-walk variant 3.69). The Hamilton
cycle on this shorter 1959–2024 FRED-QD sample is somewhat smaller than
the paper's value.

In this GDP sample the Hamilton cycle has a larger amplitude than the HP
cycle. This is not a general property of the two filters. The paper
itself makes **no** amplitude comparison with HP. Its indictments are
spurious dynamics, end-of-sample behaviour, and the λ=1600 having no
statistical foundation (the GDP MLE λ is 0.245, Table 1). The classical
BN cycle is close to zero. This is the empirical anomaly that motivates
the KMW (2018) δ̄-constrained refinement: classical BN assigns most of
the Δlog-GDP variance to the permanent component, leaving tiny
transitory residuals. The KMW-BN cycle is much larger than the
classical BN cycle.

**Results.** Run `python replicate.py` in this folder. It writes
`outputs/report.html` (interactive report), `outputs/cycles.csv` and
`outputs/trends.csv` (cycles and trends of the four filters by date) and
`outputs/research_log.json`, which includes the comparison with
`paper_targets.yaml`. Numbers depend on the FRED vintage you download.

## Library implementation

Uses [`lib.hp_filter`](../../lib/filters.py),
[`lib.hamilton_filter`](../../lib/filters.py),
[`lib.beveridge_nelson`](../../lib/filters.py), and
[`lib.kmw_bn_filter`](../../lib/filters.py), validated by the filter
unit tests in [tests/test_filters.py](../../../tests/test_filters.py)
covering:

- Additive decomposition: trend + cycle = input
- HP filter matches `statsmodels.tsa.filters.hp_filter` exactly
- Hamilton residuals have mean zero (OLS property)
- BN companion eigenvalues < 1 (stationarity of AR(p) on Δy)
- BN cycle for a true random walk is near-zero (DGP-consistency)

## How to run

```bash
python scripts/replications/hamilton_2018_filter/replicate.py
```

## Citation

```bibtex
@article{hamilton2018,
  author = {Hamilton, James D.},
  title = {Why You Should Never Use the Hodrick-Prescott Filter},
  journal = {Review of Economics and Statistics},
  volume = {100},
  number = {5},
  pages = {831--843},
  year = {2018}
}
```
