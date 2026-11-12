# Framework conventions

Authoritative reference for the **mathematical conventions** used across
`scripts/lib/`. When writing equations from code, or translating a paper's
math into a library call, consult this file first.

Tests pin down code behaviour (`tests/test_conventions.py`); this file
pins down **how to describe that behaviour in equations**, plus the
choices the library makes among multiple defensible conventions.

If you find code that violates a convention listed here, that's a bug:
fix it in your copy. If a convention here is missing or unclear, that's
a documentation bug, so extend this file.

---

## 1. Variables

### Variable scaling in `data` passed to `fit_var` / `fit_minnesota_bvar` etc.

The framework is **convention-agnostic about variable scaling**. Whatever
you put in the DataFrame is what the VAR sees. Common choices:

| Choice | When to use | What the IRF means |
|---|---|---|
| Natural log of level (`np.log(GDP)`) | Standard for output, prices, indices | IRF in log-units; multiply by 100 for percent |
| `100 * np.log(level)` | When matching papers that use this scaling | IRF directly in percent |
| Raw level (rates, ratios) | For interest rates, spreads, V/U ratios | IRF in level units (e.g., percentage points for rates) |
| Growth rates / first differences | For stationarity-by-construction | IRF in growth-rate units |

**Naming convention**: column names should signal the transformation
applied. Examples: `log_real_gdp`, `us_1y_rate`, `gdp_yoy_growth`. When
in doubt, document in the replication's config.yaml.

### What `Y_t` means in an equation

When writing a math equation derived from code:

- If the code's input `data` is in log levels → `Y_t` in the equation
  means the log level. Don't add another log.
- If the code's input is `100*log` → `Y_t` is already in percent units.
  Differences `Y_t - Y_{t-4}` give percent inflation directly.
- If the code's input is the raw level → `Y_t` is the level, and you
  must explicitly write `log Y_t` to mean the log.

**Common mistake**: writing
$\log Y_t - \log Y_{t-4}$ when $Y_t$ in the code is already a log
variable. Either drop the $\log$ or rename the equation symbol.

---

## 2. Horizon conventions

Two distinct objects use horizons in this framework. They use **different
indexing conventions on purpose** because the math demands it.

### IRF horizon: `h = 0, 1, ..., H` (impact at `h=0`)

`IRFContainer.median` has shape `(H+1, n_vars, n_shocks)`. Row 0 is the
impact response (same period as the shock). Row H is the response H
periods later.

Same convention for `lp_irf` and `lp_iv_irf`.

Math:
$$\text{IRF}_h = \Phi_h \cdot s, \qquad h = 0, 1, \ldots, H$$

where $\Phi_0 = I_n$ and $\Phi_h = \sum_{j=1}^{\min(h,p)} B_j \Phi_{h-j}$.

### FEVD horizon: `H = 1, 2, ..., H_max` (no `H=0` reported)

A single number `horizon=H` to `lib.structural_fevd` or
`lib.generalised_fevd` specifies the **H-step-ahead forecast-error
variance**, which sums **H terms** (NOT H+1):

$$\text{FEV}_k(H) = \sum_{j=0}^{H-1} (\Phi_j \Sigma_u \Phi_j^\top)_{kk}$$

$$\text{FEVD}_{k, j}(H) = \frac{\sum_{s=0}^{H-1} (\Phi_s s_j)_k^2}{\text{FEV}_k(H)}$$

So `horizon=8` = 2-year-ahead FEV for quarterly data = sum of 8 terms.

**Common mistake**: summing `j=0..H`
instead of `j=0..H-1`, giving `H+1` terms. Use `lib.structural_fevd`
rather than rolling your own. Tests pin this:
`tests/test_conventions.py::test_fevd_horizon_off_by_one_regression`.

### Forecast horizon: `h = 1, 2, ..., H` (`h=1` = one-step-ahead)

`lib.iterated_forecast` and `lib.conditional_forecast` return forecasts
starting at `h=1` (the next period after the last observation).

---

## 3. Reduced-form fit conventions

### `B_hat` column / row layout (`lib.fit_var`)

`B_hat` is a `(k, n)` matrix where `n` is the number of endogenous
variables and `k` depends on the deterministic + exogenous block:

```
B_hat row layout (top to bottom):
  rows  0  ..  n*p - 1            lag coefficients (lag1_var1, lag1_var2, ..., lagp_varn)
  row   n*p                       constant       (if trend in {"c", "ct"})
  row   n*p + 1                   linear trend   (if trend == "ct")
  rows  n*p + (c+d) ..            exog block     (if exog is provided)
                                  with c=0/1, d=0/1 depending on trend
```

Accessors:

- `fit.lag_matrices()` → list of `p` matrices each `(n, n)`. Slices rows
  `0..n*p` and reshapes/transposes.
- `fit.B_hat[n*p, :]` → constant (trend `"c"`/`"ct"`).
- `fit.B_hat[n*p+1, :]` → trend (trend `"ct"`). The `[-1]`/`[-2]`
  shortcuts hold only without exogenous regressors.
- `fit.exog_coefs()` → exog block (only when `has_exog=True`).

**Trend convention**: the trend column values in `X_act` are
`1, 2, ..., T_eff`, where `T_eff = T - p` (this library's convention;
statsmodels and JMulTi use `p+1, ..., T`, a pure intercept
reparameterisation: `const_here = const_sm + p·slope`).

### `Sigma_u` divisor

`fit.Sigma_u` is computed with a degrees-of-freedom-adjusted divisor (not
the bias-uncorrected `1/T_eff`), and the convention differs between paths:

- **Plain VAR fit via statsmodels**: divisor is `T_eff - k` where `k` is
  the number of **regressors per equation** (= `lags*n + constant + trend`).
  This is the Lütkepohl 2005 §3.2.2 small-sample correction. Worked example:
  T=200, n=2, p=2, trend="c" gives T_eff=198, k=5, divisor=193 (not 196).
- **VARX fit (when `exog` is provided)**: divisor is `T_eff` (pure ML
  estimator with no Bessel correction). Divergence from the plain path is
  documented in `fit_var`'s docstring.
- **Panel estimators** (`lib.panel.fit_*_var`): all four estimators
  (`fit_pooled_var`, `fit_mean_group`, `fit_partial_pooling_var`,
  `fit_global_var`) use the ML divisor (1/T). `fit_mean_group` rescales
  the per-unit `fit_var` Lütkepohl Sigma_u to ML before averaging, so
  residual covariances are on a common scale across panel estimators.

**Implication for hand-derived equations**: `Sigma_u != cov(u, ddof=1)`
in general. The ratio `Sigma_u / cov(u, ddof=1) = (T_eff - 1) / (T_eff - k)`
is ~1 + (k-1)/T_eff, about 2% on T=200, k=5. If you need a specific
scaling, compute it from residuals directly.

---

## 4. Structural identification conventions

### Cholesky (`lib.cholesky_impact`, `lib.cholesky_irf`)

`cholesky_impact(Sigma_u)` returns the **lower-triangular** Cholesky
factor `L` such that `L @ L.T = Sigma_u`. Diagonal entries are positive
by convention.

Ordering = variable order in the `data` DataFrame. Shock `j` has zero
impact effect on the variables ordered before it (`0..j-1`); variable
`j` and later variables can respond on impact.

**Asymptotic-MC band sampling (Σ-uncertainty included)**:
`VARFit.parameter_draws` draws ``Sigma`` alongside ``B`` from an
inverse-Wishart recentred on the point estimate (``scale =
(T-k-n-1) Sigma_hat, df = T-k``, so that ``E[Sigma_draw] = Sigma_hat``
exactly). This recentres the textbook flat-prior posterior
``IW(U'U, T-k)``, whose mean sits ``(T-k)/(T-k-n-1)`` above
``Sigma_hat``, by O((n+1)/T) on purpose. Sampling Σ propagates its
uncertainty through ``L = chol(Sigma)`` and brings the asymptotic-MC
bands to nominal frequentist coverage
(`tests/test_coverage.py::TestCholeskyAsymptoticMCCoverage`).

To hold Σ fixed:
``fit.parameter_draws(..., include_sigma_uncertainty=False)``.

### Proxy-SVAR (`lib.proxy_svar_irf`)

Three normalisations (the third, `normalization="reliability"`, is for a
single instrument; see the `proxy_svar_irf` docstring):

**`normalization="unit_policy"`**:

$$s_p = 1, \qquad s_k = \frac{\Sigma_{uz, k}}{\Sigma_{uz, p}}$$

where $\Sigma_{uz, k} = \tfrac{1}{T_{\text{eff}}} \sum_t u_{t,k} z_t$
(or `(u * z[:, None]).mean(axis=0)` in code).

**`normalization="one_sd"`**:

$$s_p = \text{np.std}(u_p, \text{ddof}=1), \qquad s_k = s_p \cdot \frac{\Sigma_{uz, k}}{\Sigma_{uz, p}}$$

**The ddof=1 detail**: the library uses sample standard deviation with
Bessel correction (`np.std(..., ddof=1)`), NOT `sqrt(Sigma_u[p, p])`.
The two differ by ~`sqrt(T_eff / (T_eff - k_ar))`, small but non-zero
on short samples. Pinned by
`test_one_sd_normalisation_h0_policy_equals_std_u_policy`.

**Instrument observed on a subsample**: `lib.proxy_svar_irf` keeps the
full residual sample and zero-fills unobserved instrument periods
(impact ratios are invariant to the fill; see the NaN-semantics note in
the docstring). Some other implementations instead trim to the common
sample and demean within it. On the Gertler-Karadi (2015) data the two
conventions differ by 2.1e-2 in the impact ratios and about 26%
(relative) in the GK fn.4 reliability scale. The first-stage F
statistic uses only the periods where the instrument is observed
(regression-tested in `tests/test_identification.py`).

**Instrument mean**: the cross-covariance
`(residuals * z[:, None]).mean(axis=0)` divides a sample product by
`T_eff`. Under `fit_var(trend="c")` the OLS residuals have zero mean,
so this equals `Cov(u, z)` whether or not `z` has zero mean
(high-frequency FOMC surprises are mean-zero by construction; narrative
shocks usually are). Some other implementations instead regress
residuals on `[1, z]`; results are equivalent under `trend="c"`. If you
fit with `trend="n"` and pass a non-zero-mean instrument, mean-centre
the instrument first to avoid spurious correlation.

### Sign restrictions (`lib.sign_restricted_irf`)

QR rotation with diagonal-sign normalisation: `Q = Q_raw @ diag(sign(diag(R)))`.

**Single-shock mode** (one key in `sign_restrictions`): check COLUMN 0 of
the Haar rotation only (with a sign flip allowed). This is the canonical
Uhlig (2005) accept/reject, since the first column of a Haar-distributed
`Q` is uniform on the unit sphere. This samples the identified set under
the uniform-conditional measure, which IS part of the reported object
(Baumeister-Hamilton 2015). `column_scan=True` instead scans all `n`
columns and accepts the first match. It is more rotation-efficient but
induces a non-canonical measure (later columns are accepted conditional
on earlier ones failing; KS-distinguishable, band edges shift by several
percent).

**Multi-shock mode** (multiple keys; Rubio-Ramirez-Waggoner-Zha 2010):
FIXED assignment: shock `j` is checked against column `j` of `Q`, with a
per-column sign flip. Haar columns are exchangeable, so the fixed
assignment loses no generality in distribution and acceptance does not
depend on the order the shocks are listed in.

**Residual bootstrap** (`bands="residual_bootstrap"`): resampled
residuals are rescaled by `sqrt(T/(T-k))` (Stine 1987) and the bootstrap
Sigma uses the same `T-k` divisor as the point estimator, so bootstrap
draws are centred on the point IRF.

Restriction types and metadata tags:

| Restrictions active | `metadata["identification"]` |
|---|---|
| Sign only | `"sign_restrictions"` |
| Sign + narrative | `"narrative_sign_restrictions"` |
| Sign + FEV-share | `"sign_fev_share_restrictions"` |
| All three | `"narrative_sign_fev_share_restrictions"` |

### Zero + sign restrictions (`lib.zero_sign_restricted_irf`)

Arias, Rubio-Ramírez & Waggoner (2018): joint zero AND sign restrictions
on the structural impact matrix. Implements Algorithm 2 (simplified to
contemporaneous zero restrictions, h=0).

**Algorithm**: builds the rotation `Q` one column at a time. For shock j,
the column `Q[:, j]` must lie in the joint null space of:

1. ``Z_j = (rows of P = chol(Σ_u) corresponding to zero-restricted
   variables for shock j)``, so that `(P Q[:, j])[v] = 0` for every
   restricted `v`.
2. ``Q[:, 1..j-1]'`` (orthogonality with previously-placed columns).

A random vector is drawn from this joint null space, normalised, and
becomes `Q[:, j]`. After all columns are placed, sign restrictions are
checked on `A_0 = P Q` via the standard column-flipping logic.

**Key invariant**: zero restrictions are algorithmically guaranteed,
not statistical. The accepted IRF satisfies `IRF[0, v, j] == 0`
exactly for every (v, j) in `zero_restrictions[shock_j]`. Pinned by
``tests/test_conventions.py::TestZeroSignRestrictionConventions``.

**Scope**: supports only `horizon=0` zero restrictions
(the impact period). Multi-horizon ARRW 2018 restrictions are not
implemented.

**Backward compat**: `zero_restrictions=None` (or empty) delegates to
`sign_restricted_irf` byte-identically. The `identification` metadata
tag is `"zero_sign_restrictions"` when zeros are active, otherwise
`"sign_restrictions"`.

### Long-run (`lib.long_run_irf`)

Blanchard-Quah 1989: structural impact `S` such that the long-run
matrix `(I - B(1))^{-1} S` is lower triangular. Permanent shocks
ordered first.

### Heteroskedasticity (`lib.heteroskedasticity_irf`)

Rigobon 2003. Two-regime variance ratio assumed; structural shocks
identified from cross-regime covariance differences.

**Normalisation**: the generalised eigendecomposition fixes the shock
covariance to the identity in the SECOND-sorted regime label (e.g.
recessions in the shipped replication). All reported IRFs are responses
to "one-standard-deviation-in-regime-2" shocks, and columns follow
scipy's ascending-eigenvalue order, which is economically arbitrary
(use `sign_normalization` to pin signs).

### Inference caveats for set-identified and weakly-identified models

These are the standard published critiques of the inference the lib
reports for the methods above. They are *documented limitations of the
methods themselves*, not implementation bugs; each is also noted in the
corresponding docstring.

1. **Pointwise quantiles across rotations are not a model (Fry-Pagan
   2011, JEL).** `sign_restricted_irf` / `zero_sign_restricted_irf`
   bands are pointwise cross-draw quantiles; the pointwise median is
   generally not the IRF of ANY accepted rotation, and connecting
   pointwise medians across horizons/variables mixes rotations. Report
   the bands as summaries of the admissible set, not as "the model's"
   response. A Fry-Pagan median-target rotation IS implemented:
   `lib.fry_pagan_median_target(draws)` selects the single accepted
   rotation closest to the pointwise median (standardised cells), and
   `sign_restricted_irf` and `zero_sign_restricted_irf` attach it
   automatically at `metadata["median_target"]` (JSON-safe; `irf` as
   nested lists). Report it alongside the pointwise bands when a
   coherent single-model response is needed.

2. **The uniform-Haar rotation prior is informative (Baumeister-Hamilton
   2015, ECMA).** "Agnostic" sign restrictions still place a non-trivial
   prior over the identified set through the uniform draw on Q; posterior
   bands do NOT shrink to the identified set as T grows. Conclusions that
   hinge on the band edges inherit that prior.

3. **Zero+sign draws are unweighted (ARRW 2018, Algorithm 3).** Drawing
   Q conditional on zero restrictions (Algorithm 2, as implemented)
   induces a non-uniform density over the identified set; ARRW's
   importance weights (their Algorithm 3 volume elements) are NOT
   applied, so cross-draw quantiles are with respect to the conditional
   draw measure, not the conditionally-agnostic prior. One consequence:
   shocks that carry the same number of zeros are drawn in the order the
   caller lists them, and that order changes the reported distribution.
   A variable may carry both a zero on impact and a sign restriction for
   the same shock; the sign then applies from horizon 1.

4. **Narrative restrictions use rejection, not ADRR's importance
   sampler.** Draws violating narrative restrictions are discarded and
   survivors equally weighted; ADRR (2018, p. 2812) note this is not
   their Algorithm 1 (which reweights by the inverse probability that
   the narrative holds). Documented in the ADRR replication README.
   A contribution restriction refers to one date and, by default, to the
   cumulative contribution since the start of the sample. The paper's
   examples restrict the one-period unexpected movement
   (`contribution="unexpected"`), with the `overwhelming` criterion in
   its monetary application. Windows of several periods are not
   supported.

5. **Long-run identification is fragile (Faust-Leeper 1997, JBES).**
   `C(1)` is a frequency-zero object estimated with vanishing precision;
   near-unit-root dynamics make BQ-style conclusions sensitive to
   low-frequency specification. Additionally, band draws with singular
   `(I - ΣA)` are skipped, which censors the most explosive (widest) draws.

6. **Rigobon bands exclude identification uncertainty.** The band loop
   holds the rotation `H` fixed at its point estimate (only the VAR
   coefficients vary); sampling error in the regime covariances (often
   the dominant uncertainty) is NOT reflected. The weak-identification
   flag is a heuristic eigen-gap threshold, not a formal test
   (Lütkepohl-Netšunajev 2017; Lewis 2021).

### Bayesian Local Projections (`lib.blp_irf`)

Ferreira, Miranda-Agrippino & Ricco (2025, REStat 107(5); 2021 WP design). Shrinks each horizon's LP coefficient
toward the iterated-VAR IRF at that horizon, with shrinkage strength λ.

**Convex-combination form** (simplified BLP per the natural-conjugate
prior scaling where prior precision matches data precision):

```
β^BLP_{h, j} = (β^OLS_{h, j} + λ · β^VAR_{h, j}) / (1 + λ)
σ^BLP_{h, j} = σ^OLS_{h, j} / √(1 + λ)
```

* `β^OLS_{h, j}`: frequentist LP coefficient at horizon h, response j
  (recovered via the existing `lp_irf` HAC-based pipeline).
* `β^VAR_{h, j}`: iterated-VAR IRF at horizon h on variable j, using
  the OLS LP at h=0 as the structural impact loading. Built from a
  baseline VAR(`var_lags`) fit.
* `λ`: `shrinkage` kwarg. λ=0 → pure OLS LP; λ=1 → equal weight;
  λ=∞ → pure VAR.

**Algorithmic invariants** (pinned by
`tests/test_conventions.py::TestBLPConventions`):

* At λ=0, BLP point estimate and bands are byte-identical to `lp_irf`.
* At h=0, the BLP coefficient is invariant to λ, because the prior
  mean at h=0 IS the OLS LP coefficient at h=0 (the impact loading
  used to build the VAR prior).
* Band shrinkage is exactly `1/√(1+λ)` at every horizon.

**Identification tag**: `"bayesian_lp"`; method tag:
`"bayesian_local_projections"`.

**Scope**: this implementation handles the shock coefficient only.
The full MAR 2021 framework also allows Minnesota-style priors on the
control coefficients; that is not implemented. The shock-coefficient shrinkage
captures the headline BLP advantage (reduced LP variance at long
horizons).

### State-dependent local projections (`lib.state_dependent_lp_irf`)

Auerbach-Gorodnichenko 2012 / Ramey-Zubairy 2018. Two-regime LP with
the shock effect allowed to differ across regimes.

Estimating equation per horizon h and response j:

```
y_{j, t+h} = α^E_h + (α^R_h − α^E_h) · F_t
           + β^E_{jh} · s_t + (β^R_{jh} − β^E_{jh}) · F_t · s_t
           + γ_h^T Z_{t−1:t−p} + δ_h^T F_t · Z_{t−1:t−p} + u_{j, t+h}
```

where `s_t` is the shock and `F_t ∈ [0, 1]` is the state weight
(1 = recession, 0 = expansion). The intercept is regime-dependent
(`α^E_h` in expansion, `α^R_h` in recession); implementation includes
`F_t` as a separate regressor so the constant shifts with the regime.

* **Binary state** (NBER dummy, Harding-Pagan dummy): `F_t ∈ {0, 1}`
* **Continuous state** (smooth-transition logistic, AG 2012):
  `F_t ∈ [0, 1]`

The same machinery handles both.

`interact_controls=True` (default) includes `F_t · Z_{t−1:t−p}`, the
Ramey-Zubairy 2018 "saturated" specification, allowing regime-dependent
dynamics. `interact_controls=False` omits these interaction terms;
shock coefficient is still regime-interacted but the regime affects
the conditional mean only through the contemporaneous shock channel.

**Returned container shape**: `(H+1, n_response, 2)` where the last
dim is `(expansion, recession)`. `shock_names` are
`[f"{shock_name}_{state_names[0]}", f"{shock_name}_{state_names[1]}"]`,
default `("expansion", "recession")`.

**Recession SE formula**:
`Var(β^R) = Var(β^E) + Var(β^R − β^E) + 2 · Cov(β^E, β^R − β^E)`,
where the covariance is pulled from the joint HAC covariance matrix
of the regression coefficients. Tiny negative variances from
finite-sample HAC are replaced with NaN in the bands.

`metadata["identification"]` is the literal `"state_dependent_lp"`;
`metadata["state_mean"]` reports the sample mean of `F`.

### Global / panel VAR conventions (`lib.fit_global_var`)

Stacks the per-unit panel into a single VAR on
`N = n_units * n_vars_per_unit` endogenous variables, with stacked
variable name `"{unit}_{var}"` and column order `(u_0_v_0, u_0_v_1,
..., u_0_v_K-1, u_1_v_0, ...)`: unit-major, variable-minor.

Minnesota shrinkage with four-way block structure on
`B_{ij}^{(p'), kl}` (effect of unit `j`'s variable `l` at lag `p'`
on unit `i`'s variable `k`):

| Block | λ_eff |
|---|---|
| Own unit, own variable (i=j, k=l) | λ₁ |
| Own unit, cross variable (i=j, k≠l) | λ₁ · λ₂ |
| Cross unit, own variable (i≠j, k=l) | λ₁ · λ_cross_unit |
| Cross unit, cross variable (i≠j, k≠l) | λ₁ · λ₂ · λ_cross_unit |

Lag decay: `(p')^λ₃` divides every prior std.
Variance scaling: each block's prior std is multiplied by `σ_eq / σ_col`
where σ are presample AR(p) residual stds (Sims-Zha / BGR convention).

Prior mean: random walk on own-own at lag 1 (`B_{ii}^{(1), kk} = 1`),
zero elsewhere.

`lambda_cross_unit = 0` collapses cross-unit blocks to exactly zero
(pinned by
`tests/test_panel.py::TestGlobalVAR::test_recovers_block_diagonal_when_cross_unit_zero`).

Per-block accessor: `fit.cross_unit_lag_block(unit_to, unit_from, lag)`
returns the K×K sub-block. Summary statistic
`fit.average_cross_unit_spillover(lag)` returns the average Frobenius
norm of cross-blocks divided by the average Frobenius norm of own-blocks
at the named lag (0 = no spillovers; 1 = cross-blocks as large as
own-blocks).

The fit subclasses `VARFit` and presents an N-variable VAR(p) interface,
so all identification routines (`cholesky_irf`, `sign_restricted_irf`,
`proxy_svar_irf`) work directly on the stacked system.

### Non-Gaussian SVAR (`lib.nongaussian_svar_irf`)

Lanne-Meitz-Saikkonen 2017. Identifies the structural matrix `B` in
`u_t = B ε_t` up to column permutation and sign by exploiting
non-Gaussianity of the structural shocks.

Algorithm:

1. **Pre-whiten** residuals: `Z = L^{-1} u` with `L = chol(Σ_u)`.
2. **Find orthogonal Q** such that `ε_t = Q^⊤ z_t` are approximately
   independent (and maximally non-Gaussian). Two methods:

   * `method="fastica"`: Hyvärinen (1999) symmetric (parallel) FastICA
     fixed-point algorithm. Distribution-free. Returns unmixing
     `W = Q^⊤`. Tag in `ica_method` metadata: `"fastica"`.
   * `method="pml_t"`: LMS §3 parametric pseudo-MLE assuming each
     structural shock is i.i.d. standardised Student-t with per-shock
     degrees-of-freedom `ν_j`. Optimises `(Q, ν)` jointly via L-BFGS-B
     over the Givens-angle parameterisation of orthogonal matrices.
     Warm-starts from the FastICA solution plus `pml_t_n_restarts`
     random restarts; the highest-likelihood result wins.
     Tag in `ica_method` metadata: `"pml_t"`.

3. **Recover** `B = L Q` (note the transpose: for FastICA,
   `Q = W^⊤`; for pml_t, `Q` is returned directly).

Identification family tag: `"nongaussian_svar"` for both methods. The
`ica_method` metadata field distinguishes which inner algorithm ran.

The standardised Student-t pdf for unit-variance scaling (used by
`method="pml_t"`):

$$f(x; \nu) = \frac{1}{\sigma} \cdot t_\nu\!\left(\frac{x}{\sigma}\right), \qquad \sigma = \sqrt{\frac{\nu - 2}{\nu}}$$

with `ν > 2` required (for finite variance). The optimiser parameterises
`ν` via `log(ν − 2)` so the unconstrained optimiser respects the bound.

Diagnostics returned:

* `jarque_bera`: per-shock JB stat + p-value
* `skewness`, `excess_kurtosis`: per-shock sample moments (Fisher form)
* `n_gaussian_like`: number of shocks with JB p > 0.10
* `weak_nongaussianity`: True if `n_gaussian_like >= 2` (LMS allow at
  most one Gaussian component; non-rejection is finite-sample evidence,
  not proof). `max_jb_pvalue` is also reported.
* `fastica_converged`, `fastica_n_iter`, `fastica_final_diff`
  (always populated, since FastICA is the warm-start even under `pml_t`)
* When `method == "pml_t"` additionally:
  `pml_t_log_likelihood`, `pml_t_nu` (per-shock ν estimates),
  `pml_t_converged`, `pml_t_n_restarts_tried`,
  `pml_t_best_restart_index`

The `weak_nongaussianity` flag is the main check. LMS identification
fails on near-Gaussian data, and silent unreliability is the
worst-case mode. Always inspect this before trusting LMS output.

### Sign restrictions + higher moments (AFM 2023)

Andrade-Ferroni-Melosi 2023 extends sign-restricted SVARs with
inequality restrictions on the implied structural shock series'
higher moments. Added as a kwarg `higher_moment_restrictions` on
`lib.sign_restricted_irf`; it composes cleanly with `narrative_restrictions`
(ADRR 2018) and `fev_share_restrictions`.

**Restriction schema** (single entry):

```python
# Threshold (single-shock or per-shock in multi-shock mode):
{"moment": "skewness" | "excess_kurtosis",
 "lower": float, "upper": float,   # both optional, default -inf / +inf
 "abs": bool,                       # if True, apply bounds to |moment|
 "shock": "name"}                   # required in multi-shock mode

# Cross-shock comparison (multi-shock mode only):
{"shock": "a", "moment": "...",
 "vs": "b", "comparison": "greater" | "less",
 "margin": float}                   # optional, default 0.0; strict-margin gap
```

**Moment names**:

* `"skewness"`: `scipy.stats.skew(bias=False)`, the adjusted
  Fisher-Pearson estimator (G1).
* `"excess_kurtosis"`: `scipy.stats.kurtosis(fisher=True, bias=False)`,
  **Fisher form** (excess kurtosis = kurt − 3) with **bias=False**
  (the adjusted estimator G2). A Gaussian shock has `excess_kurtosis = 0`.
  Leptokurtic = positive.

  **Comparison with MATLAB**: MATLAB `kurtosis(x,0)` is the
  bias-corrected Pearson kurtosis, equal to SciPy
  `kurtosis(x, fisher=False, bias=False)`; subtract 3 to compare with our
  excess-kurtosis value. MATLAB's default `kurtosis(x)` (flag 1) is the
  biased moment estimator (SciPy `bias=True`).

**Implementation**: for a candidate column `Q[:, c]`, the implied
structural shock is `ε_t = u_t^⊤ (P^{-⊤} Q[:, c])` where `P = chol(Σ_d)`.
Same projection used by narrative restrictions, reused here.

**Identification tag**: gains `"higher_moment"` bit (e.g.,
`"sign_higher_moment_restrictions"` or
`"multi_shock_sign_higher_moment_restrictions"`).

**Default pin**:
`tests/test_conventions.py::TestAFMHigherMomentConventions::test_none_is_byte_identical_to_baseline`
locks `higher_moment_restrictions=None` as byte-identical to omitting
the argument.

---

## 4b. BVAR (Minnesota) prior conventions

`lib.MinnesotaPrior` follows the Sims-Zha (1998) / Bańbura-Giannone-
Reichlin (2010) convention, encoded via dummy observations stacked with
the actual data (Theil mixed estimation).

### Hyperparameter directionality

| Parameter | Smaller value means | Effect |
|---|---|---|
| `lambda_1` (overall tightness) | **Tighter** shrinkage to prior mean | Posterior closer to RW prior; less data influence |
| `lambda_2` (cross-variable tightness) | **Tighter** cross-variable shrinkage | Off-diagonal lag coefficients pulled to 0 (asymmetric mode only) |
| `lambda_3` (lag decay) | Slower decay across lags | Distant lags less aggressively shrunk |
| `lambda_4` (constant tightness) | **Tighter** prior on the intercept | Less commonly varied; default 100 = nearly flat |
| `mu` (sum-of-coefficients) | Weaker SOC prior; `mu=0` disables | Larger `mu` enforces RW-like sum-of-own-lags behaviour |
| `delta` (dummy-initial-obs) | Weaker single-unit-root prior; `delta=0` disables | Larger `delta` strengthens common-trend prior |

### `lambda_2` is mode-dependent (critical)

Under the **default symmetric** BGR (2010) form (`asymmetric=False`):

  The prior covariance has Kronecker structure $\Sigma \otimes V_0$
  where $V_0 = (X_d'X_d)^{-1}$ is the regressor covariance factor shared
  by all equations (its inverse is the prior precision). This
  structure CANNOT encode different shrinkage for own-lag vs
  cross-variable coefficients within the same equation. So `lambda_2`
  is a **silent no-op** in symmetric mode.

  Pinned by `tests/test_conventions.py::test_lambda_2_no_op_under_symmetric_mode`.

Under the **asymmetric** Litterman (1986) / Sims-Zha (1998) form
(`asymmetric=True`):

  The fit becomes equation-by-equation Ridge with per-equation
  precision matrices. `lambda_2` actively scales cross-variable
  coefficient priors. Smaller `lambda_2` → tighter cross-variable
  shrinkage → off-diagonals pulled toward 0.

  Pinned by `tests/test_conventions.py::test_lambda_2_has_effect_under_asymmetric_mode`.

Tradeoffs of asymmetric mode:

* `Sigma_u` is treated as a fixed plug-in (residual covariance from
  per-equation fits), NOT drawn from an Inverse-Wishart posterior.
* `BVARFit.parameter_draws` in asymmetric mode samples beta from
  per-equation Gaussian posteriors with `Sigma_u` held constant.
* `lib.log_marginal_data_density` raises `NotImplementedError` in
  asymmetric mode (no closed-form MDD).

### Dummy observation count = signal of which sub-priors are active

`fit.n_dummies` reports the number of dummy rows stacked on top of the
data. Composition (for an $n$-variable VAR with $p$ lags):

  * $n \cdot p$ rows for own-lag tightness (`lambda_1`, `lambda_3`;
    always active).
  * $n$ rows for the Sigma prior (`Yd = diag(σ)`, `Xd = 0`) in the
    symmetric/conjugate form; absent with `asymmetric=True`.
  * 1 row for constant tightness (`lambda_4`) if `trend="c"`.
  * one row per variable with a non-zero `rw_deltas` entry ($n$ rows by
    default) for the sum-of-coefficients prior, only if `mu > 0`.
  * 1 row for dummy-initial-observation prior, only if `delta > 0`.

Pinned by `test_mu_zero_disables_sum_of_coefficients_dummies` and
`test_delta_zero_disables_dummy_initial_observation`.

### `rw_deltas` = prior mean of own-lag-1 coefficients

Default `[1, 1, ..., 1]` = random-walk prior for every variable. Set to
0 for variables you believe are I(0) / white-noise around mean (e.g.,
unemployment gap, output gap). Mixed `[1, 1, 0]` is fine
(variable-by-variable).

The sum-of-coefficients rows are scaled by `rw_deltas` (BGR 2010), so a
variable set to 0 gets no sum-of-coefficients row. The
dummy-initial-observation row is not affected.

### Marginal likelihood and lag length

`lib.log_marginal_data_density` is comparable across hyperparameters for
the same data and lag length. It is not comparable across lag lengths
when the fits are given the same data frame: a fit with `lags=p` explains
the observations from `p + 1` on, so each lag length explains a different
set of rows and the comparison favours the longest lag. To compare lag
lengths, drop leading rows so that every fit explains the same
observations (for a maximum lag `P`, pass `data.iloc[P - p:]`).

### `obs_scale` (GLS per-observation variance scaling), Lenza-Primiceri 2022

Adding `obs_scale=s_t` (length `T - lags`) pre-scales the `(Y_act, X_act)`
rows by `1/s_t` before stacking with the Minnesota dummies. The dummies
retain original weight; high-`s_t` observations contribute less to the
posterior. `obs_scale=None` (default) is byte-identical to the
unscaled fit. Use for COVID-era variance scaling.

---

## 4c. TVP-VAR prior conventions (`lib.fit_tvp_var`)

The Primiceri (2005) TVP-VAR places independent priors on:

* **β_0** (initial-period VAR coefficients):
  $\beta_0 \sim N(\beta_{OLS}^{train}, 4 \cdot V_{OLS}^{train})$
  where $V_{OLS}^{train} = \Sigma_{OLS}^{train} \otimes (X^\top X)^{-1}$ from
  the training-sample OLS. The factor 4 is Primiceri's looser prior on β_0.

* **Q** (random-walk drift covariance of β_t):
  $Q \sim IW\big(k_Q^2 \cdot \tau \cdot V_{OLS}^{train}, \tau\big)$
  with $\tau$ = `training_sample` and $k_Q^2$ = `Q_prior_scale`
  (**default 1e-4**, Primiceri's $k_Q = 0.01$ *squared*, p. 831).

  Two pinned conventions (`tests/test_conventions.py::TestTVPPriorConventions`):

  1. The IW scale **multiplies** by $\tau$, not divides (Primiceri 2005
     Appendix C).
  2. The scale uses $V_{OLS}^{train}$, **not** the β_0 prior covariance
     $4 V_{OLS}$. The factor 4 belongs to the β_0 prior only.

* **Σ** (constant residual covariance): IW with `df = Sigma_prior_df`
  (default `n+2`, just-proper), scale = `Sigma_OLS_train * df`.

For the SV extension (`fit_tvp_var_sv`, `fit_tvp_var_sv_primiceri`), the
same Q prior calibration applies. The other priors do not follow the
paper:

* log-volatility innovations: IG(3, 0.01) on each variance (mean 0.005),
  not scaled by the training sample;
* first-period log-volatility: N(0, 10);
* covariance states (`fit_tvp_var_sv_primiceri` only):
  IW(`S_prior_scale` · τ · I, τ), with `S_prior_scale` default 1e-4. The
  paper scales this prior by the variance of the training-sample OLS
  covariance coefficients. Here the scale is a fixed number, so check
  that it suits the units of the data: a value of 0.01 produces visible
  drift in the covariance states even on data with constant covariances.

Three further conventions:

* the training sample enters twice, in the prior and again in the
  estimation sample (the paper drops it from estimation), so paths are
  reported for all `T - lags` dates;
* the constant inside `log(y² + c)` of the volatility sampler is 1e-6
  (the paper uses 0.001). Volatility estimates are flattened when the
  residuals are very small in absolute size (standard deviation near
  0.001 or below), so express the data in percent;
* the mixture indicators are drawn after the coefficient and covariance
  blocks and immediately before the volatilities, which is the order of
  the Del Negro and Primiceri (2015) correction.

---

## 5. Bootstrap conventions

### Wild bootstrap (`lib.proxy_svar_irf` default)

Rademacher weights $\omega_t \in \{-1, +1\}$ IID per period:

$$u_t^* = \omega_t \hat u_t, \qquad z_t^* = \omega_t z_t$$

The same $\omega_t$ multiplies both residuals and instruments
(preserves the proxy-SVAR cross-covariance structure within the draw).

### Residual bootstrap (`lib.residual_bootstrap_draws`, `lib.cholesky_irf`)

IID resampling with replacement from the empirical residual
distribution. `T_eff` draws per bootstrap iteration.

Stability filter: draws are discarded if the resampled VAR's companion
matrix has any eigenvalue with modulus `>= eig_tol` (default 1.0).
Filter ceiling at `4 * n_draws` attempts before raising.

**Asymmetry note**: the proxy-SVAR bootstraps (`_proxy_wild_bootstrap`,
`_proxy_moving_block_bootstrap`, and the multi-instrument variant) use a
slightly looser hardcoded tolerance of `1.01` rather than 1.0. Proxy
applications typically run near-unit-root levels VARs, and a strict 1.0
filter discards a large share of re-estimated draws, biasing the bands.
The 1.01 ceiling keeps marginally-explosive re-estimates whose IRFs are
still well-behaved over the reported horizon.

### Exog block during bootstrap

When `fit.has_exog`, exogenous variables stay **fixed** across
bootstrap replications (treated as deterministic). Only the
endogenous residuals are resampled.

---

## 6. Container conventions

| Container | Shape | Indexing |
|---|---|---|
| `IRFContainer.median` | `(H+1, n_vars, n_shocks)` | `[h, var, shock]`; `h=0` is impact |
| `FEVDContainer.median` | `(H+1, n_vars, n_shocks)` | Same axes; `[h, ...]` is the h-step-ahead share for h>=1 (h=0 row degenerate / set to impact share by convention) |
| `HistoricalDecompositionContainer` | dict of DataFrames, each `(T_eff, n_vars)` | `.shocks[name]`, `.deterministic_plus_initial`, `.other_shocks` |
| `ForecastContainer` | `(H, n_vars)` for point; `(n_paths, H, n_vars)` for density | Forecast starts at h=1 |
| `ConditionalForecastContainer` | Same as ForecastContainer | Conditioning info in metadata |

### Metadata fields

All containers carry a `metadata: dict` field. Recommended keys:

- `method`: free-form description ("VAR / Cholesky", "narrative sign restrictions")
- `identification`: machine-readable tag (see Section 4)
- `n_lags`, `n_obs_effective`
- `seed`: RNG seed for reproducibility
- `parameter_uncertainty`: bool
- For sign-restricted: `n_attempted`, `n_accepted`, `acceptance_rate`,
  `n_rejected_by_narrative`, `n_rejected_by_fev_share`
- For proxy-SVAR: `first_stage` dict with `F`, `gamma`, `R2`, `n_obs`

---

## 7. Time indexing and sample alignment

### Effective sample

For a VAR(p), `T_eff = T - p`. `Y_act` has shape `(T_eff, n)` starting
at index `p` of the input data. Residuals are aligned with `Y_act`:
`fit.residuals[i]` is the innovation in `fit.data.iloc[p + i]`.

### Date alignment in `fit.data.index`

- `fit.data.index[:p]` → initial conditions (no residual computed).
- `fit.data.index[p:]` → effective sample, one residual per period.
- `fit.data.index[p + i]` → the period corresponding to
  `fit.residuals[i]` and `fit.Y_act[i]`.

### PeriodIndex vs Timestamp alignment

`load_dataset("fred_qd", ...)` returns end-of-quarter Timestamps (March,
June, September, December). Pandas' `.resample("QS").mean()` returns
start-of-quarter Timestamps (January, April, July, October). A naive
merge across the two produces all-NaN rows.

**When mixing FRED-QD with externally-resampled data**, convert both to
`PeriodIndex('Q')` before merging, then convert back to Timestamp at the
end. The pattern:

```python
q = monthly.resample("QS").mean()
q.index = q.index.to_period("Q")          # align on PeriodIndex('Q')
merged = fredqd_df.set_axis(fredqd_df.index.to_period("Q")).join(q)
merged.index = merged.index.to_timestamp()  # back to Timestamps
```

---

## 8. Common gotchas / FAQ

### "The paper says 100·log; my data is log"

If your data is in natural log and the paper is in 100·log, you have
two options:
1. Multiply your data by 100 before fitting, recovering paper-scale IRFs
   directly.
2. Keep natural log, then multiply IRFs by 100 to get percent.

Both work; pick one and stay consistent.

### "FEVD at the 2-year horizon: H=8 or H=7?"

For quarterly data use `horizon=8`, which sums 8 terms (j=0..7). See
§2 "Horizon conventions".

### "Why doesn't `np.std(u, ddof=1)` equal `sqrt(Sigma_u[i,i])`?"

The plain VAR divides by `T_eff - k`, not `T_eff - 1`. See §3
"`Sigma_u` divisor" for the exact ratio.

### "Why does `trend='ct'` matter vs `trend='c'`?"

The trend column adds a `1, 2, ..., T_eff` regressor to `X_act`, absorbing
deterministic linear time trends out of the lag coefficients. On series
with persistent low-frequency variation (commodity prices, broad CPI),
`'ct'` materially changes lag coefficients and IRF persistence.
Pinned by `tests/test_var.py::test_trend_ct_adds_trend_row_to_b_hat`.

### "When do I use HD vs FEVD?"

- **HD (historical decomposition)** decomposes the observed path of
  $Y_t$ at each $t$ into contributions of each shock plus deterministic
  + initial conditions. It's a **temporal** decomposition: "how much of
  Y_t in 2022Q1 came from which shocks?"
- **FEVD (forecast-error variance decomposition)** decomposes the
  variance of the H-step-ahead forecast error into shock contributions.
  It's a **horizon-distribution** statement: "at the 2-year horizon,
  what share of the variance is from shock j?"

The two answer different questions. Both have valid horizon conventions
but they don't correspond one-to-one.

---

## 8b. Independent cross-checks

Cross-checks against independent implementations are described in
DEVELOPMENT.md §"Independent cross-checks"; the convention differences
they surfaced are documented in place in this file.

## 9. Extending this file

If you add a convention choice to `lib.` (a new normalisation option,
horizon convention or container), document it here and pair it with a
closed-form test in `tests/test_conventions.py` where possible.
