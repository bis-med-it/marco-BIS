# Jordà (2005): Local Projections

Canonical replication of the local-projections impulse-response estimator from
Jordà (2005), demonstrated against a closed-form Cholesky-identified VAR on
the same data. The two methods agree at short horizons (when the VAR is
well-specified) and diverge at long horizons (where LP is the robustness
check Jordà's paper champions).

Mirrors [`entries/jorda_2005_lp.yaml`](../../../skills/model-dictionary/entries/jorda_2005_lp.yaml).

## Specification

| | |
|---|---|
| **Data** | FRED-QD, four variables: real GDP, GDP deflator, commodity prices (PPIACO), federal funds rate |
| **Sample** | 1965Q1–2019Q4 (220 quarters; pre-COVID) |
| **Variable order (Cholesky)** | rgdp → deflator → commodity → ffr |
| **Lags** | 4 (matched between VAR and LP control set) |
| **Horizon** | 20 quarters |
| **Identification** | Recursive (Cholesky on VAR residual covariance); structural FFR shock = standardised innovation in the FFR equation |
| **VAR bands** | Asymptotic MC, 2000 draws |
| **LP bands** | Newey-West HAC with `lags = horizon + 1` (LP-standard rule) |

## Pipeline

```
load_dataset (FRED-QD)
  → lib.transform (log/level)
    → lib.fit_var(data, lags=4)               # parametric VAR
       ├→ lib.cholesky_irf(fit, ...)          # closed-form structural IRF
       └→ extract structural shock = chol⁻¹ · residuals → "ffr" column
          → lib.lp_irf(data, shock, horizon)  # local projections
```

The Cholesky-identified FFR shock is the same object both methods use: for
`cholesky_irf` it's implicit (through the lower-triangular `P` factor); for
`lp_irf` it's passed as the `shock` argument. This gives a clean apples-to-apples
comparison: same data, same identification, two estimators. The shock is
extracted by the inline helper `extract_cholesky_shock` in `replicate.py`.

Validated by the unit tests in [`tests/test_lp.py`](../../../tests/test_lp.py),
plus the Cholesky / VAR tests covering identification and inference.

## Results

Real GDP falls after a contractionary monetary shock in both estimators.
The VAR and LP output responses are close for about the first three
years. At longer horizons the VAR output loss persists, while the LP
response fades toward zero. For the federal funds rate the two methods
agree at short horizons and disagree from about three years on: this
fitted VAR's rate path stays positive, while the LP estimates turn
negative. The exercise does not show which path is closer to the true
response. On average the LP bands are wider than the VAR's
asymptotic-MC bands. Jordà flags this as the cost of horizon-by-horizon
estimation.

**Results.** Run `python replicate.py` in this folder. It writes
`outputs/report.html` (interactive report), `outputs/numerical_results.json`
(IRFs) and `outputs/research_log.json`, which holds the VAR-vs-LP
comparison table, the out-of-sample forecast comparison and the
comparison with `paper_targets.yaml`. Numbers depend on the FRED vintage
you download and on the seed in `config.yaml`.

## How to run

```bash
python scripts/replications/jorda_2005_lp/replicate.py
```

## Notes

* **Why LP and VAR are close at h=0**: both use the same Cholesky-identified
  shock, the FFR-equation residual orthogonalised against the
  contemporaneous shocks of the variables ordered before it. At h=0 the LP
  coefficient on this shock estimates the same impact effect as the
  relevant column of the Cholesky factor `P`. The two are not identical:
  the LP regresses on the extracted shock series, and the VAR values are
  medians over parameter draws. The replication script prints both side
  by side.
* **Why they can diverge at long horizons**: the VAR extrapolates its lag
  structure, which gains precision but can add bias if the lags are
  misspecified. LP estimates each horizon separately, which reduces that
  bias but adds variance; LP point estimates are also subject to
  finite-sample bias (see `skills/model-selection/SKILL.md`).
* **The "price puzzle"** (deflator increases after a contractionary shock)
  shows up in both estimators at short horizons. Adding commodity prices
  partially attenuates it but does not eliminate it. This is a known feature
  of recursive identification on this data.

## Scope vs original paper

Jordà (2005) presents LP as a *general method*, not tied to any one
application. His single empirical illustration (p. 175) is **monetary**: a
3-variable system of the output gap (real GDP vs CBO potential),
GDP-deflator inflation and the federal funds rate, on quarterly US data
**1955:I–2003:I**, VAR(4), Cholesky order gap→inflation→FFR, shown with both
linear and **cubic (nonlinear)** projections.

This replication demonstrates the method on a **CEE-style 4-variable
monetary VAR** (real GDP, deflator, commodity, FFR; 1965–2019) rather than
Jordà's exact 3-variable output-gap system. This is the setting in which LP is
most often deployed as a robustness check against a benchmark VAR. The
**fiscal and state-dependent LP applications** often associated with the
method are *later* developments (e.g., Ramey 2016, Ramey-Zubairy 2018;
Auerbach-Gorodnichenko 2012 is a smooth-transition VAR that motivates the
state weights), **not part of Jordà 2005**; they fit on the same `lib.lp_irf`
interface and live in separate replications/applications.

## Citation

```bibtex
@article{jorda2005lp,
  author  = {Jordà, Òscar},
  title   = {Estimation and Inference of Impulse Responses by Local Projections},
  journal = {American Economic Review},
  volume  = {95},
  number  = {1},
  pages   = {161--182},
  year    = {2005}
}
```
