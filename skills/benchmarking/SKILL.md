---
name: benchmarking
description: >
  Universal evaluation skill for any estimation method in the framework.
  Runs standardised forecast accuracy tests against baseline models and
  published results, economic sensibility checks, and correctness
  verification against matched reference implementations.
  Model-specific fitting comes from the estimation code; this skill
  evaluates its outputs. Use on request after an estimation run to check
  the pipeline and the model's forecast performance, or during paper
  preparation for formal benchmarking. Triggers when the user says
  "benchmark", "evaluate", "test the model", "how well does this perform",
  or "compare forecasts". Residual-diagnostic requests go to the
  diagnostics skill.
---

# Benchmarking

## Design principle

One skill, any model. The skill evaluates standard outputs (IRFs, FEVDs,
forecasts) and holds no model-specific logic. A check whose input a method
does not produce is skipped and the reason stated.

Forecast benchmarking needs, for each model, either a complete refit and
forecast recipe (a `fit_fn` for `lib.rolling_forecasts`, with any tuning done
on the training window only) or matched out-of-sample forecast records with
their provenance. The estimation code supplies the model-specific fitting.
For an unchanged fit, use its existing diagnostics rather than recomputing
them.

## Two modes

### Development mode (fast check)

A quick pass/fail check that runs in minutes, after an estimation run.

### Paper mode (full suite, for publication)

The full benchmarking suite, for publication-grade evaluations. Use when
preparing results for submission.

---

## Development mode

### Inputs

```yaml
benchmark_request:
  mode: "development"
  results: output from any estimation method (standard interface)
  data: the dataset used for estimation
  holdout_start: date (default: last 20% of sample)
```

### Step 1: Baseline generation

Automatically generate these baselines from the same data using established
packages:

| Baseline | Implementation | Purpose |
|----------|---------------|---------|
| Random walk | `statsmodels` or trivial shift | Naive benchmark; failing to beat it is a finding to report |
| AR(1) per variable | `statsmodels.tsa.ar_model.AutoReg` | Univariate benchmark |
| AR(p) per variable | `statsmodels.tsa.ar_model.AutoReg` with BIC selection | Stronger univariate benchmark |
| VAR(1) flat prior | `statsmodels.tsa.api.VAR` | Multivariate floor |
| Sample mean | trivial computation | Unconditional benchmark |

Use whichever package is most efficient. The agent should use existing
libraries, not reimplement standard models from scratch.

### Step 2: Forecast evaluation

Using the holdout sample:

1. Estimate the model on the training sample only. **No look-ahead
   anywhere in the pipeline:** hyperparameters, lag orders, or any other
   tuned quantity must be selected on the training window only (or
   re-tuned inside each origin). A "winner" tuned on the full sample has
   already seen the holdout, and its apparent OOS edge is leakage, not
   skill. State in the report which window the tuning used.
2. Produce forecasts at horizons h = 1, 4, 8 (quarterly) or h = 1, 6, 12
   (monthly).
3. Compute RMSFE for the estimated model's output and each baseline,
   **on matched samples**: every compared model is scored on the same
   valid (origin, horizon, variable) cells (`lib.forecast_score_matched`
   does this join), so a failed fit in one model cannot shift the
   comparison onto different origins.
4. Compute RMSFE ratios: skill output / random walk. Values < 1.0 mean the
   model beats the random walk.

**Criteria for development:**

| Check | Type | Pass | Not passed |
|-------|------|------|------------|
| RMSFE ratio to RW at h=1 | Empirical (reported) | < 1.0 for at least half the variables | ≥ 1.0 for most variables: flag and report |
| RMSFE ratio to AR(1) at h=1 | Empirical (reported) | < 1.1 (within 10% of AR) | > 1.3 (much worse than AR): flag and report |
| Any variable RMSFE = NaN or Inf | Pipeline (fail) | None | Any present |
| Forecast variance | Pipeline (fail) | Finite and reasonable | Explosive or zero |

Only the pipeline checks fail the run. Failing to beat the random walk or
the AR benchmarks is an empirical finding to report, not evidence of an
estimation error.

### Step 3: Economic sensibility check

Model-agnostic checks using only the standardised IRF output:

| Check | Method | Pass | Fail |
|-------|--------|------|------|
| Stability | Max eigenvalue modulus from the fit's diagnostics (`lib.diagnostics(...).stability`); skipped without a fit | Reported and classified (stationary / unit-root / explosive) against the declared model-class or replication criterion | An explosive root not disclosed, or IRFs that diverge |
| IRF convergence, transitory shocks | Response at max horizon < 10% of peak response | Yes | No |
| IRF convergence, permanent shocks (long-run-identified, or level responses in unit-root systems) | Response **flattens** (change over the last ~8 periods small relative to peak) rather than vanishing; demanding die-out here fails correct models by construction (e.g. Blanchard-Quah supply on output) | Flattens | Still trending at max horizon |
| Sign at impact | Check against expected signs from knowledge base (if available) | Match | Mismatch: reported for investigation, not a fail (see the specification skill) |
| IRF magnitude | Peak response within plausible range for the variable | Reasonable | Implausibly large or zero |

Classify each shock as transitory or permanent BEFORE applying the
convergence check (the identification scheme tells you: long-run /
permanent-transitory schemes name their permanent shocks; recursive and
sign schemes on stationary transforms are transitory).

If no knowledge base is available (e.g., first run on a new dataset), skip
the sign check and only test stability and convergence.

### Step 4: Correctness verification

If a reference package is available (configurable), estimate the same model
with the same specification in the reference package and compare. A
reference counts only if it implements the same prior, estimator,
identification and normalisation as the model being checked:

```yaml
reference_packages:
  var_ols:
    python: "statsmodels.tsa.api.VAR"     # same OLS estimator and ordering
    r: "vars::VAR"
  bvar:
    python: null                          # needs the same prior and hyperparameters; else skip
    r: "bvartools::bvar"                  # only with a matched prior; posterior simulation
  local_projections:
    python: "linearmodels"                # or custom OLS loop
    r: "lpirfs::lp_lin"
  sign_restrictions:
    python: null                          # no standard package; skip
    r: "VARsignR"                         # posterior simulation
  tvp_var:
    python: null                          # no standard package; skip
    r: "bvarsv"                           # posterior simulation
```

**Comparison metric.** For deterministic estimators (OLS VAR with the same
ordering, OLS local projections), compute the maximum absolute difference in
impulse response point estimates across all variables and horizons. It
should be close to numerical precision; a larger difference flags a
potential implementation error. References that use posterior simulation
(bvartools, VARsignR, bvarsv) need matched algorithms, draw counts and seeds.
For these, compare posterior medians and band quantiles with a tolerance
scaled to each variable's units and to the Monte Carlo error.

A comparison against a classical OLS VAR, a random walk or another
benchmark with a different estimator is empirical performance (Step 2),
not a correctness check.

If no matched reference is available for the model class (e.g., sign-restricted
VAR in Python), skip this step and rely on sensibility checks only.

### Step 5: Report

Output a concise development report (illustrative example; the numbers are
hypothetical):

```
============================================
DEVELOPMENT BENCHMARK REPORT
Model: BVAR (Minnesota prior)
Date: 2026-04-11
============================================

FORECAST ACCURACY (RMSFE ratio to random walk; empirical, reported)
          h=1     h=4     h=8
GDP       0.87    0.79    0.91    beats RW
CPI       0.94    0.85    0.88    beats RW
FFR       1.02    0.93    0.87    RW better at h=1
UNEMP     0.91    0.82    0.95    beats RW

SENSIBILITY
Stability:          PASS (max eigenvalue = 0.94)
IRF convergence:    PASS (all responses < 5% of peak at h=40)
Sign checks:        3/3 expected signs at impact (reported)

CORRECTNESS
Matched reference (same prior and hyperparameters): none   SKIPPED

OVERALL: PASS (pipeline and stability checks)
============================================
```

**If any check fails or is flagged,** the report highlights it and lists
causes to check:

```
FLAG: RMSFE ratio to RW > 1.0 for all variables at all horizons.
This is an empirical finding to report. Before reporting it, check for a
data transformation or forecast-alignment error, and whether the prior is
too tight (shrinking to a prior mean far from the data). If none applies,
report the result as it stands.
```

---

## Paper mode

### Inputs

```yaml
benchmark_request:
  mode: "paper"
  results: output from any estimation method (standard interface)
  data: the dataset used for estimation
  holdout_start: date
  competitors: list of competitor model specifications (see below)
  specification_variants: list (for cross-specification comparison)
```

### Step 1: Full competitor set

Extends development baselines with configurable external competitors:

```yaml
competitors:
  # Always included (computed internally)
  internal:
    - random_walk
    - ar1
    - ar_bic
    - var1_flat
    - sample_mean

  # Configurable (require additional packages or data), for example:
  external:
    - name: "LASSO-VAR"
      implementation: "sklearn.linear_model.LassoCV on stacked VAR system"
      package: "scikit-learn"

  # Published results (no computation needed; just comparison)
  published:
    - name: "GLP (2015)"
      source: "Giannone, Lenza, Primiceri (2015), Table 2"
      metric: "MSFE levels by system size (BVAR vs flat-prior VAR, factor model, random walk); derive RMSFE ratios as sqrt(MSFE_model / MSFE_RW) from the same row"
      notes: "applicable to US data with similar variable set"
```

**Key design: competitors are data, not code.** The competitor list is
provided at runtime (config.yaml ships no standing block for it). Adding a
new competitor means adding one entry to the list, not modifying the
benchmarking skill.

### Step 2: Cross-specification comparison

Takes a list of specification variants, estimates each, and compares:

```yaml
specification_variants:
  - name: "Agent-specified"
    source: "framework output"
  - name: "Fixed textbook"
    specification: {variables: [...], lags: 4, prior: {lambda_1: 0.1, ...}}
  - name: "GLP-optimal"
    method: "maximise marginal likelihood over hyperparameter grid (on the
      initial training window only when this variant enters an OOS
      comparison (full-sample tuning leaks the evaluation observations))"
  - name: "Expert-specified"
    specification: {provided by user}
  - name: "Cross-validated"
    method: "ridge/LASSO with CV on VAR coefficients"
```

Each variant is estimated and evaluated identically. The benchmarking skill
does not know or care how the specification was determined. It receives
the standardised output and evaluates it.

### Step 3: Statistical tests

For all pairwise comparisons:

- **Diebold-Mariano test** (Diebold and Mariano, 1995) for forecast accuracy,
  via `lib.diebold_mariano` (Newey-West HAC variance with the Harvey,
  Leybourne, Newbold 1997 small-sample correction and t(n-1) p-values).
  Requirements the caller must meet (the function cannot check them):
  - **Consecutive forecast origins.** The HAC truncation (`horizon - 1`)
    and the HLN factor assume the loss series' spacing equals one forecast
    period: run the rolling exercise with `step=1`, or pass the horizon
    in origin-spacing units.
  - **Matched losses.** Pair the two models' losses at the same origins
    (join on origin before differencing), never two independently
    NaN-dropped arrays.
  - **Degenerate differentials raise.** A nonzero CONSTANT loss
    differential is an undefined test (one model dominates by a fixed
    amount). Report the losses directly; the function raises rather than
    returning p = 1.
- **Giacomini-White test** (2006) for conditional predictive ability. It is not
  in `scripts/lib/`; run it separately.
- Report p-values. Use * p<0.10, ** p<0.05, *** p<0.01 convention.

### Step 4: Economic checks alongside forecast accuracy

Report the economic checks next to predictive performance, in an unranked
table: stability classification (maximum root), whether IRFs settle, impact
and peak signs, peak magnitudes, and whether a price puzzle appears. Do not
combine expected signs, magnitudes or the absence of a price puzzle into a
model-quality score, and do not rank models by such a score. Unexpected
results are reported and investigated, not penalised.

### Step 5: Paper-ready output

Generate:

1. **Summary table** (main text): one row per model/specification, columns
   for RMSFE at each horizon, the economic checks (unranked), and DM test results.
2. **Detailed tables** (appendix): per-variable, per-horizon RMSFE.
3. **RMSFE bar charts** (main text): grouped by horizon, coloured by method.

All generated via the results-presentation skill in the appropriate style.

---

## Interface

### Input contract

The benchmarking skill accepts the standard output from any estimation
method. No model-specific fields are required. If a field is missing (e.g.,
no FEVD for LP), the benchmarking skill skips the corresponding evaluation.

### Output contract

```yaml
benchmark_results:
  mode: "development" | "paper"
  model_evaluated: string
  forecast_accuracy:
    rmsfe: {variable -> {horizon -> value}}
    rmsfe_ratio_to_rw: {variable -> {horizon -> value}}
    lps: {variable -> {horizon -> value}} (paper mode only)
    crps: {variable -> {horizon -> value}} (paper mode only)
  baselines:
    {baseline_name -> {variable -> {horizon -> rmsfe}}}
  competitors: (paper mode only)
    {competitor_name -> {variable -> {horizon -> rmsfe}}}
  statistical_tests: (paper mode only)
    diebold_mariano: {pair -> {horizon -> {statistic, p_value}}}
  sensibility:
    stability: bool
    irf_convergence: bool
    sign_checks: {shock -> {variable -> match/mismatch}}
  correctness: (if a matched reference is available)
    max_irf_difference: float
    pass: bool
  overall_pass: bool (development mode)
  charts: list of file paths (paper mode)
  summary_tables: list of file paths (paper mode)
```

---

## Adding new competitor models

To add a new competitor, include an entry in the `competitors` list of the
runtime benchmark request:

```yaml
- name: "My New Model"
  implementation: "description of how to run it"
  package: "package_name"
```

The benchmarking skill generates the competitor's forecasts using the
specified package, then evaluates identically to all other competitors.
No changes to the benchmarking skill itself.

If the competitor is a published result (no code to run), add to
`competitors.published` instead and provide the metric values manually.

## References

- Diebold, F.X. and Mariano, R.S. (1995). Comparing Predictive Accuracy. JBES.
- Harvey, D., Leybourne, S., Newbold, P. (1997). Testing the Equality of
  Prediction Mean Squared Errors. International Journal of Forecasting,
  13(2), 281-291.
- Giacomini, R. and White, H. (2006). Tests of Conditional Predictive Ability.
  Econometrica.
- Giannone, D., Lenza, M., Primiceri, G.E. (2015). Prior Selection for VARs. RESTAT.
- McCracken, M.W. and Ng, S. (2016). FRED-MD: A Monthly Database for
  Macroeconomic Research. Journal of Business & Economic Statistics.
