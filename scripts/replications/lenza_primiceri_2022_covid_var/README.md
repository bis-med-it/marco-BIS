# Lenza-Primiceri (2022): BVAR with COVID Scaling

Canonical replication of Lenza & Primiceri (2022) "How to Estimate a
Vector Autoregression after March 2020" (JAE 37(4), 688-699). The paper
addresses the practical question every macro forecaster faces post-2020:
how do you fit a VAR on a sample that includes the pandemic without
either dropping the data (information loss) or letting the 10-sigma
2020Q2 observations contaminate the parameter estimates?

LP's answer: a per-observation variance scaling `s_t` that down-weights
COVID-era innovations in the GLS sense, leaving the transmission
mechanism (B matrix) intact but allowing larger innovation variance
during the pandemic.

Mirrors [`entries/lenza_primiceri_2022_covid_var.yaml`](../../../skills/model-dictionary/entries/lenza_primiceri_2022_covid_var.yaml).

## Specification

| | |
|---|---|
| **Setup** | 4-variable quarterly BVAR, 5 lags, 1960Q1 - 2024Q4 (260 obs) |
| **Variables** | real GDP, GDP deflator, unemployment rate, federal funds rate (FRED-QD: `GDPC1`, `GDPCTPI`, `UNRATE`, `FEDFUNDS`) |
| **Prior** | Minnesota (λ₁=0.2, λ₂=0.5, λ₃=1, λ₄=100, μ=1, δ=1) |
| **Scaling** | `s_bar=5`, `rho=0.8` (decay rate); plateau through 2020Q2-2020Q4, then geometric decay back to 1.0 |
| **Library function** | `lib.fit_minnesota_bvar(..., obs_scale=s_t_array)` (the per-observation `obs_scale` weighting parameter) |

`obs_scale` is a per-observation scale factor (`s_t`) on the post-lag
effective sample. The fitter pre-scales `Y_act` and `X_act` rows by
`1/s_t` before stacking with the Minnesota dummies. This is equivalent to
GLS with diagonal weight matrix `diag(1/s_t^2)`. The prior dummies retain
their original weight, so high-`s_t` observations contribute less to
the posterior while the prior pulls the same amount.

## Three setups compared

This replication runs the same data through three different BVAR
estimation strategies to isolate the LP method's contribution:

1. **Naïve**: standard BVAR on the full sample with no COVID handling.
   The 2020Q2 unemployment spike (3.5% → 13.0% q/q) is treated as a
   typical observation; parameters absorb it.
2. **Drop COVID**: standard BVAR on the sample with 2020Q2 and 2020Q3
   dropped (2 quarters; 2020Q4 onward stays in). The two pandemic
   quarters no longer affect the parameter estimates, but
   the residual covariance under-states the tail risk that the
   pandemic period revealed.
3. **LP-scaled**: standard BVAR on the **full** sample with `obs_scale`
   set per the parameterisation above. Parameter estimates should
   match the drop case (COVID observations down-weighted) while
   keeping all data in the sample.

## Results

**Parameter recovery.** The unemployment own-lag-1 coefficient is the
headline. The naïve fit pulls it far below the value the drop-COVID fit
gives: the 2020Q2 spike drags the autoregressive coefficient down as the
BVAR tries to "explain" the pandemic. **The drop and LP-scaled fits give
similar values, both far from the naïve one.** Across the four own-lag-1
coefficients, the LP-scaled fit is much closer to the drop fit than the
naïve fit is.

LP's contribution: recover drop-style parameter estimates **without**
dropping any data, so all 260 quarterly observations remain in the
sample for posterior moment-matching and forecast-uncertainty
quantification.

**Results.** Run `python replicate.py` in this folder. It writes
`outputs/report.html` (interactive report) and
`outputs/research_log.json`, which holds the own-lag coefficients and
residual standard deviations of the three setups and the comparison with
`paper_targets.yaml`. Numbers depend on the FRED vintage you download.

## Residual variance: a deviation from the paper

The naïve fit has the largest residual standard deviations, most of all
for unemployment. The LP-scaled residual std (on the unit-scale
innovation `u_t`) is about the same as or **slightly smaller** than the
drop case, not larger as the paper's qualitative description
(drop-COVID density forecasts are "excessively sharp", p. 689) might
suggest. Two reasons:

1. The IW posterior on Σ has more degrees of freedom in LP (255 effective
   observations with weighted contributions) than in drop (253 unweighted),
   pulling `Sigma_u` slightly tighter for LP.
2. The paper's "wider bands" claim refers to **forecast uncertainty going
   forward**, which requires propagating a posterior distribution over
   future `s_t` (e.g., non-zero probability of another pandemic-scale
   event). This replication uses a fixed `s_t` path with no future
   randomness, so the residual-Σ comparison doesn't show the LP advantage
   for forecast bands.

The forecast-band comparison needs forward simulation with stochastic
`s_t` and is not included (see Caveats). The result shown here is that
LP scaling brings the coefficients close to the drop-COVID estimates.

## Library implementation

| Step | Function |
|---|---|
| Data load | `lib.Spec.from_replication` + `lib.load_data` |
| BVAR estimation | `lib.fit_minnesota_bvar(data, lags=5, prior=..., obs_scale=s_t)` |
| Per-obs scaling | `obs_scale` parameter; supported on the symmetric (natural-conjugate) prior only (the asymmetric Litterman branch rejects it) |
| Validation | `tests/test_bvar.py::TestObsScale` |

## How to run

```bash
python scripts/replications/lenza_primiceri_2022_covid_var/replicate.py
```

See the Results section for the files it writes.

## Caveats

* **No theta estimation**: s_t fixed at demo values (the paper ESTIMATES
  [s_bar_0, s_bar_1, s_bar_2, rho], posterior peaks ~17/65/20 and rho just
  below 0.8, Fig 1 p. 692).
* **No real-time forecast-band comparison**: not included in this
  replication; the paper's "wider bands" claim requires forward
  simulation with stochastic `s_t`.
* **Quarterly, 4-variable spec**: simpler than the paper's 7-variable
  monthly illustration (p. 691). The quarterly 4-variable setup has
  not been compared with the paper's 7-variable monthly illustration.

## Citation

```bibtex
@article{lenzaprimiceri2022,
  author  = {Lenza, Michele and Primiceri, Giorgio E.},
  title   = {How to Estimate a Vector Autoregression after March 2020},
  journal = {Journal of Applied Econometrics},
  volume  = {37},
  number  = {4},
  pages   = {688--699},
  year    = {2022}
}
```
