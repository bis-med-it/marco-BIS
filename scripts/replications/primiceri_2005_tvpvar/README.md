# Primiceri (2005) TVP-VAR: methodological demo

Demonstrates the time-varying-coefficients portion of Primiceri (2005),
"Time Varying Structural Vector Autoregressions and Monetary Policy"
*Review of Economic Studies* 72(3), 821-852.

## Scope of this replication

The original paper specifies:
- **Time-varying VAR coefficients** β_t with random-walk drift
- **Stochastic volatility** Σ_t via the Kim-Shephard-Chib mixture algorithm
- 10,000 Gibbs iterations (paper p. 830; first 2,000 discarded)

This replication exercises the homoskedastic variant; the library also
ships the full stochastic-volatility sampler
([`lib.fit_tvp_var_sv_primiceri`](../../lib/tvp.py)), not used here to
keep the demo fast.

## Specification

| | |
|---|---|
| **Variables** | Inflation (dlog GDPCTPI × 400, annualised %), Unemployment, 3-month T-bill |
| **Lags** | 2 quarterly |
| **Sample** | Requested 1953Q1–2001Q3 (Primiceri's window); FRED-QD starts in 1959, so the run uses 1959Q2–2001Q3 (170 quarters) |
| **Training sample** | first 40 quarters of the available data (the paper's length, not its 1953–1962 years) |
| **Q prior scale** | 0.0001 = k_Q² with k_Q = 0.01 (paper p. 831) |
| **Gibbs** | 300 iterations, 100 burn-in → 200 stored draws |

Mirrors [entries/primiceri_2005_tvpvar.yaml](../../../skills/model-dictionary/entries/primiceri_2005_tvpvar.yaml).

## Results: own-lag-1 coefficient paths (paper-faithful prior, demo run)

In the current run the posterior-mean own-lag-1 coefficient paths of all
three equations are almost flat over the sample. The VAR evaluated at the
last-period posterior mean is stable.

**Results.** Run `python replicate.py` in this folder. It writes
`outputs/report.html` (interactive report), `outputs/own_lag1_path.csv`
(the posterior-mean own-lag-1 paths) and `outputs/research_log.json`,
which includes the comparison with `paper_targets.yaml`. Numbers depend on
the FRED-QD vintage you download and on the seed in `config.yaml`.

With Primiceri's actual calibration (k_Q² = 1e-4) and this demo-scale run
(300 iterations), the coefficient paths drift only modestly. The prior is
tight, and the paper's headline time-variation lives mainly
in the STOCHASTIC-VOLATILITY channel and in functions of the full system
(impulse responses, the inflation-unemployment trade-off), not in large
swings of individual VAR coefficients.

The prior calibration is paper-faithful and pinned by
`tests/test_conventions.py::TestTVPPriorConventions`.

Caveats:

1. **No stochastic volatility in this baseline run** (the lib implements
   it as `fit_tvp_var_sv_primiceri`; this replication exercises the
   homoskedastic variant). Without SV, regime variation that belongs to
   Σ_t is forced into β_t or suppressed by the tight Q prior.
2. **Demo-scale draws.** 300 iterations vs the paper's 10,000 (2,000
   burn-in). Production analysis should scale up (see the `gibbs` block in
   config.yaml).
3. **Training rows are reused.** `lib.fit_tvp_var` builds the β_0 and Q
   priors from OLS on the first 40 effective observations (1959Q4–1969Q3
   here) and then also estimates the coefficient path over those same
   rows. Primiceri uses the training sample only for the prior and
   starts the estimation after it.

## Library implementation

Uses [`lib.fit_tvp_var`](../../lib/tvp.py), validated by the unit tests
in [`tests/test_tvp.py`](../../../tests/test_tvp.py). They include a
constant-DGP recovery test: with a near-zero Q prior the TVP posterior
collapses to OLS. `TVPVARFit` is a `VARFit` subclass, so `cholesky_irf`
works on it directly.

## How to run

```bash
python scripts/replications/primiceri_2005_tvpvar/replicate.py
```

## Citation

```bibtex
@article{primiceri2005,
  author = {Primiceri, Giorgio E.},
  title = {Time Varying Structural Vector Autoregressions and Monetary Policy},
  journal = {Review of Economic Studies},
  volume = {72},
  number = {3},
  pages = {821--852},
  year = {2005}
}
```
