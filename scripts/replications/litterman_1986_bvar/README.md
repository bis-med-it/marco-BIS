# Litterman (1986): Bayesian VAR with the Minnesota prior

Canonical replication of the seminal Minnesota-prior Bayesian VAR from
Litterman (1986), *JBES* 4(1), 25–38, "Forecasting with Bayesian Vector
Autoregressions—Five Years of Experience". Fits Litterman's seven-variable
quarterly system at the paper's hyperparameters, reports an own-lag-1
shrinkage diagnostic against an unrestricted OLS VAR, and tests the
paper's headline with a rolling-window out-of-sample forecast comparison.

Mirrors [`entries/litterman_1986_bvar.yaml`](../../../skills/model-dictionary/entries/litterman_1986_bvar.yaml).

## Specification

| | |
|---|---|
| **Variables** (paper p. 29) | real GNP growth, GNP-deflator inflation, unemployment rate, log M1, log investment, commercial-paper rate, change in business inventories |
| **Frequency** | quarterly |
| **Lags** | 6 (paper p. 29) |
| **Sample** | paper: 1948:1–1979:3; replication: 1959Q1–2019Q4 (FRED-QD; modern OOS demonstration) |
| **Prior** | Minnesota, `λ₁=0.2, λ₂(θ)=0.2, λ₃=1` (harmonic), `μ=0, δ=0`, `asymmetric=True` |
| **Prior mean** (`rw_deltas`) | `[1,1,1,1,1,1,1]`: own-first-lag mean = **1 for all** variables (random-walk prior, p. 29) |
| **Identification** | Cholesky (illustrative IRF to a commercial-paper-rate shock; the paper's focus is forecasting) |

## Specification notes (pp. 29–32)

- **Transforms.** Real GNP and the deflator enter as **growth / inflation
  rates** (p. 29), not log levels.
- **Prior mean = 1 for all** own-first-lags (random-walk prior, p. 29),
  including the stationary variables.
- `μ`/`δ` (sum-of-coefficients, dummy-initial-obs) are **off**. They are
  not part of the original specification the paper describes (which dates
  to Litterman's 1979-80 work; DLS 1984 and Sims 1993 introduced them).

## Substitutions (FRED-QD proxies)

| Paper variable (p. 29) | FRED-QD | Notes |
|---|---|---|
| real GNP, annual growth | `GDPC1`, `dlog_x_400` | real GDP, annualized quarterly growth |
| GNP-deflator inflation | `GDPCTPI`, `dlog_x_400` | annualized inflation |
| unemployment rate | `UNRATE`, level | unchanged |
| M1 (log level) | `M1REAL` → `nominalise(M1REAL, CPIAUCSL)`, log | FRED-QD publishes only real M1, deflated by the CPI; the CPI is used for this step only (the GDP deflator stays a model variable) |
| gross private domestic investment (log) | `GPDIC1`, log | real GPDI |
| 4–6 month commercial paper rate | `CP3M`, level | 3-month CP rate (the 4–6mo series is discontinued) |
| change in business inventories | `diff(BUSINVx)` | **proxy**: NIPA change-in-inventories is not in FRED-QD; differenced total business inventories stands in |

The paper's 1948–1979 sample predates FRED-QD (which starts 1959Q1); the
rolling OOS demonstration runs **1959Q1–2019Q4** (ends before the 2020 M1
redefinition / COVID) for enough rolling origins.

## Results

**Own-lag-1 shrinkage (Minnesota prior in action).** The research log
compares each own-lag-1 coefficient under OLS, the BVAR posterior mean
and the prior mean of 1. Growth-rate variables (rgnp, defl) have low OLS
own-persistence and the prior pulls them up toward 1; the level
variables sit closer to unity already.

**Out-of-sample forecast RMSE, Litterman's actual headline.**
Rolling-window OOS: 80-quarter window, step=1 (consecutive quarterly
origins, the spacing the Diebold-Mariano correction assumes), horizons
1..12, 163 origins, both models scored on the same matched forecast
cells. BVAR (paper hyperparameters) vs unrestricted OLS VAR(6), reported
at h = 1, 4, 8 and 12. In the current run the BVAR has the lower RMSE in
every reported (variable, horizon) cell. Diebold-Mariano tests
(squared-error loss, paired by origin) reject equal predictive accuracy
in most cells. This is in line with Litterman's 1986 claim (his Table 2,
p. 30): the Minnesota prior produces substantially better quarterly
forecasts than the unrestricted multivariate alternative.

The Cholesky IRF to a commercial-paper-rate shock is reported in `outputs/`
as an illustration only. Litterman's model is a forecasting tool, not a
structural one, so the IRF carries no causal interpretation.

**Results.** Run `python replicate.py` in this folder. It writes
`outputs/report.html` (interactive report), `outputs/numerical_results.json`
(IRFs) and `outputs/research_log.json`, which holds the shrinkage
diagnostic, the RMSE and Diebold-Mariano tables and the comparison with
`paper_targets.yaml`. Numbers depend on the FRED vintage you download and
on the seed in `config.yaml`.

## How to run

```bash
python scripts/replications/litterman_1986_bvar/replicate.py
```

The rolling OOS re-fits the BVAR and OLS VAR at every origin. See the
Results section for the files it writes.

## Citation

```bibtex
@article{litterman1986bvar,
  author  = {Litterman, Robert B.},
  title   = {Forecasting with Bayesian Vector Autoregressions:
             Five Years of Experience},
  journal = {Journal of Business \& Economic Statistics},
  volume  = {4},
  number  = {1},
  pages   = {25--38},
  year    = {1986}
}
```
