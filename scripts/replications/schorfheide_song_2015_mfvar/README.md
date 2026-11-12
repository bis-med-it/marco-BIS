# Schorfheide-Song (2015) Mixed-Frequency BVAR: 4-variable demo

Methodological demonstration of the Schorfheide-Song (2015) mixed-
frequency Bayesian VAR using FRED-MD (monthly) + FRED-QD (quarterly) data.

## Specification

| | |
|---|---|
| **Monthly variables** | IP growth, CPI inflation, FFR (all from FRED-MD) |
| **Quarterly variable** | Real GDP growth (from FRED-QD) |
| **Lags** | 4 |
| **State lags** | 5 (Mariano-Murasawa aggregation window) |
| **Sample** | 1960-01 to 2019-12 (monthly) |
| **Prior** | Minnesota with `lambda_1=0.2, lambda_2=0.5, mu=1.0, delta=1.0` |
| **Gibbs** | 500 iterations, 100 burn-in, 1 thin |
| **Aggregation** | Mariano-Murasawa flow weights `[1/3, 2/3, 1, 2/3, 1/3]` |

Quarterly GDP growth is placed on the monthly index at end-of-quarter
months (March, June, September, December) and is NaN at intra-quarter
months. The state-space measurement equation enforces

    y^Q_t = (1/3) y^M_t + (2/3) y^M_{t-1} + y^M_{t-2} + (2/3) y^M_{t-3} + (1/3) y^M_{t-4}

exactly. The Gibbs sampler alternates:

1. Draw the latent monthly GDP path from the Durbin-Koopman simulation
   smoother, conditional on the VAR parameters
2. Draw the VAR parameters from the NIW posterior, conditional on the
   imputed complete monthly dataset

Mirrors [entries/schorfheide_song_2015_mfvar.yaml](../../../skills/model-dictionary/entries/schorfheide_song_2015_mfvar.yaml).

## Results

In the current run, at every end-of-quarter month the Mariano-Murasawa
weighted sum of the past 5 posterior-mean monthly GDP states matches the
observed quarterly GDP growth up to numerical precision. The estimated
MF-VAR is stable.

This is the measurement-equation constraint built into the state-space
model and is enforced by the Kalman simulation smoother. The Gibbs
sampler produces a posterior-mean monthly state path that interpolates
the quarterly GDP growth observations.

**Results.** Run `python replicate.py` in this folder. It writes
`outputs/report.html` (interactive report),
`outputs/state_path_posterior_mean.csv` (the posterior-mean monthly state
path, including imputed monthly GDP growth) and
`outputs/research_log.json`, which includes the comparison with
`paper_targets.yaml`. Numbers depend on the FRED-MD and FRED-QD vintages
you download and on the seed in `config.yaml`.

Intra-quarter months are imputed from the VAR's dynamic structure and
the contemporaneous correlation with the monthly indicators.

## Library implementation

Uses [`lib.fit_mfvar_bayesian`](../../lib/mfvar.py), which builds on:

- [`lib.state_space`](../../lib/state_space.py): Kalman / simulation
  smoother (thin wrappers around `statsmodels.tsa.statespace.SimulationSmoother`)
- [`lib.bvar`](../../lib/bvar.py): Minnesota BVAR machinery and posterior draws
- [`lib.mariano_murasawa_weights`](../../lib/state_space.py): flow-aggregation weights

Validated by unit tests in [`tests/test_mfvar.py`](../../../tests/test_mfvar.py):

- MFVARSpec validation (lag bounds, empty-vars rejection)
- Measurement-design construction (monthly identity, quarterly MM weights)
- Exact aggregation arithmetic
- Posterior-mean state in the all-monthly degenerate case matches data
- Gibbs runs end-to-end and stores draws
- Imputation at quarter-ends close to observed

Plus tests for the state-space module and the Gibbs module.

## Limitations vs Schorfheide-Song 2015

| Original SS 2015 | This replication |
|---|---|
| 11 variables (3 quarterly, 8 monthly) | 4 variables (1 quarterly, 3 monthly) |
| Log LEVELS (UNR/FF/TB divided by 100, p. 371) | Growth rates (GDP, IP, CPI) + FFR level |
| 3-month average of latent log levels; ragged edge via time-varying observation dimension (Durbin-Koopman) | Mariano-Murasawa 5-weight identity on growth rates (the level identity rewritten in differences) |
| p(m) = 6 monthly lags (MDD exploration, p. 372) | 4 monthly lags |
| Sample 1968:M1; real-time origins 1997:M7-2010:M1 | 1960-2019, final vintage |
| MDD-optimised hyperparameters (lambda_1 = 0.09, lambda_2 = 4.30 at the peak, p. 373) | Fixed canonical Minnesota |
| Real-time vintages | Final-vintage only |
| Forecast evaluation vs MIDAS / quarterly VAR | Just imputation demo |

Extending to the full 11-variable SS spec would need the paper's
log-level, 3-month-average measurement equation, which
`lib.fit_mfvar_bayesian` does not provide (it implements the
Mariano-Murasawa growth-rate weights only). MDD hyperparameter optimisation is not
available for the MF-VAR: `lib.optimize_minnesota_hyperparameters` covers
the non-mixed BVAR only.

## How to run

```bash
python scripts/replications/schorfheide_song_2015_mfvar/replicate.py
```

## Citation

```bibtex
@article{schorfheidesong2015,
  author = {Schorfheide, Frank and Song, Dongho},
  title = {Real-Time Forecasting with a Mixed-Frequency VAR},
  journal = {Journal of Business \& Economic Statistics},
  volume = {33},
  number = {3},
  pages = {366--380},
  year = {2015}
}
```
