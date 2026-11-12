# Library

Reusable Python primitives for the MaRCo@BIS framework. Imported by
all replication scripts under [`../replications/`](../replications/), and by
the orchestrator-driven workflow defined under `skills/`.

Design rule: where `statsmodels`, `scipy` or `numpy` cover a step (OLS VAR
fit, state-space smoothing, information criteria), the library calls them.
Methods those packages do not expose (Bayesian VARs and their Gibbs
samplers, sign-restriction sampling, proxy SVAR, FAVAR rotation, BN filters
and others) are implemented here from the papers and pinned by the tests.
HTML reports use `plotly`.

## Modules

| Module | What it does |
|---|---|
| [`irf.py`](irf.py) | `IRFContainer` / `FEVDContainer`, the standardised output contract defined in [DEVELOPMENT.md](../../DEVELOPMENT.md#standardised-irf-structure) |
| [`var.py`](var.py) | `fit_var()` (wraps `statsmodels.tsa.api.VAR`), companion form, MA representation, stability, lag selection. `VARFit.parameter_draws()` provides the generic uncertainty interface |
| [`bvar.py`](bvar.py) | `fit_minnesota_bvar()` (with `exogenous_block=` for small-open-economy block exogeneity), `MinnesotaPrior`, `BVARFit` (subclass of `VARFit` with NIW posterior draws); `fit_steady_state_bvar()` / `SteadyStatePrior` (Villani 2009 mean-adjusted / steady-state prior); `log_marginal_data_density()` and `optimize_minnesota_hyperparameters()` for GLP (2015) hyperparameter selection. Plugs directly into `cholesky_irf()` and `long_run_irf()` with `bands="parameter_draws"`, and into `sign_restricted_irf()` with `parameter_uncertainty=True` (that function has no `bands` argument) |
| [`lp.py`](lp.py) | `lp_irf()` (Jordà 2005), `lp_iv_irf()` (Stock-Watson 2018 LP-IV) with HAC bands |
| [`identification.py`](identification.py) | `cholesky_irf()`, `sign_restricted_irf()` (sign / narrative / magnitude / FEV-share / higher-moment restrictions), `zero_sign_restricted_irf()` (ARRW 2018), `proxy_svar_irf()` (GK / Mertens-Ravn / Stock-Watson; `normalization="reliability"`), `long_run_irf()` (Blanchard-Quah), `heteroskedasticity_irf()` (Rigobon 2003). Auto-dispatch to asymptotic or posterior uncertainty based on the fit type |
| [`inference.py`](inference.py) | `asymptotic_mc_draws()`, `residual_bootstrap_draws()`, `draw_parameter_samples()` |
| [`filters.py`](filters.py) | `hp_filter()`, `hamilton_filter()` (Hamilton 2018), `beveridge_nelson()`, `kmw_bn_filter()` (Kamber-Morley-Wong 2018: constrained AR(p) re-estimation with target signal-to-noise ratio), `FilterResult` |
| [`connectedness.py`](connectedness.py) | `connectedness()` (Diebold-Yilmaz 2012), `generalised_fevd()`, `rolling_connectedness()` |
| [`factors.py`](factors.py) | `fit_favar()`, `favar_irf()` (Bernanke-Boivin-Eliasz 2005 two-step PC FAVAR with slow/fast rotation) |
| [`state_space.py`](state_space.py) | `StateSpaceModel`, `kalman_filter_smooth()`, `simulation_smoother()` (thin wrappers around `statsmodels.tsa.statespace`), `mariano_murasawa_weights()` for flow aggregation |
| [`gibbs.py`](gibbs.py) | `draw_niw_posterior()` (Normal-Inverse-Wishart conjugate draws), `draw_state_path()` (single Durbin-Koopman simulation-smoother draw). Foundation for MF-VAR, TVP-VAR, DFM |
| [`mfvar.py`](mfvar.py) | `fit_mfvar_bayesian()` (Schorfheide-Song 2015 / Bańbura-Modugno 2014 mixed-frequency BVAR via Gibbs); `MFVARSpec`, `MFVARFit` subclassing `VARFit` so identification routines work directly |
| [`tvp.py`](tvp.py) | `fit_tvp_var()` (time-varying coefs, constant Σ); `fit_tvp_var_sv()` (TVP-VAR + diagonal SV); `fit_tvp_var_sv_primiceri()` (FULL Primiceri 2005: time-varying β_t, A_t, AND h_t); `TVPVARSpec`, `TVPVARFit` |
| [`nongaussian_svar.py`](nongaussian_svar.py) | `nongaussian_svar_irf()` (Lanne-Meitz-Saikkonen 2017): FastICA initialisation + parametric pseudo-MLE under non-Gaussian independent components |
| [`panel.py`](panel.py) | Panel VAR estimators (Canova-Ciccarelli 2013): pooled, mean-group, Bayesian partial-pooling / hierarchical, stacked cross-country VAR (`fit_global_var`; not a Dees et al. GVAR) |
| [`dating.py`](dating.py) | Business-cycle dating: Harding-Pagan (quarterly) and a simplified Bry-Boschan-style monthly routine (no smoothing stages) |
| [`validation.py`](validation.py) | `compare_to_paper()` checks a run's outputs against a replication's `paper_targets.yaml`, printing the pass/fail table the smoke tests parse |
| [`stochastic_volatility.py`](stochastic_volatility.py) | Kim-Shephard-Chib (1998) 7-component mixture for log χ²(1); `draw_log_h_path()` for univariate SV sampling. Reusable across TVP-VAR-SV and Cogley-Sargent applications |
| [`dfm.py`](dfm.py) | `fit_dfm_bayesian()` (Bayesian DFM via Gibbs with arbitrary missing data; a substitution for Bańbura-Modugno 2014's EM estimator); `DFMSpec`, `DFMFit`. Handles ragged-edge / NaN observations natively |
| [`report.py`](report.py) | `VARReport`: interactive HTML report with Plotly charts and diagnostics tables |
| [`forecast.py`](forecast.py) | `iterated_forecast()` (point + density forecasts for any `VARFit` subclass, with optional parameter-uncertainty path simulation), `rolling_forecasts()` (expanding/rolling-window driver), `forecast_score()` (MSE/RMSE/MAE by horizon or variable), `diebold_mariano()` (HAC-robust pairwise forecast-comparison test), `ForecastContainer` |
| [`specs.py`](specs.py) | `Spec` (immutable dataclass with `data`/`model`/`irf`/`forecast`/`metadata` blocks with deep-merge `update`), `Spec.from_config(path)` / `Spec.from_replication(id)` factories, `lib.run(spec)` (driver dispatching to fit + identification + forecast for supported specs: var/bvar fits with cholesky, sign-restriction, long-run or proxy-SVAR identification; raises `NotImplementedError` otherwise), `SpecResult` |
| [`diagnostics.py`](diagnostics.py) | `diagnostics(fit)` (one-button residual + stability + lag-selection report), `DiagnosticReport` with `summary()` (human-readable) + `to_log_dict()` (JSON-safe). Tests per-variable Ljung-Box / Jarque-Bera / ARCH-LM, classifies stability as stationary / marginal-near-unit-root / non-stationary, runs the AIC/BIC/HQIC selection across `1..max_lag_search` and flags disagreements with the current `p`. Warnings list aggregates actionable concerns |
| [`sensitivity.py`](sensitivity.py) | `sensitivity(base_spec, grid, metric_fn=None)` (hyperparameter sweep over `lib.run`; dotted-path grid keys like `"model.prior.lambda_1"` map to deep-merged spec overrides; Cartesian product over multi-axis grids), `SensitivityReport` with `.table` DataFrame + `.results` list + `.best(by, ascending)` helper. Default metric extractor pulls `max_eig`, `aic`, `bic`, `hqic`, `log_mdd` (when applicable). |
| [`data/`](data/) | Two-step pattern: `load_dataset(source, series, start, end, vintage)` returns raw, structurally-clean data + provenance metadata in `df.attrs` (source, vintage, frequency, url, loaded_at, per-series source code + recommended_transform from FRED-MD codes). `transform(df, transforms)` applies per-column transforms, because the decision belongs with the model. Available transforms: standard FRED-MD codes (`level`, `log`, `diff`, `log_diff`/`dlog`, `diff_log_diff`/`ddlog`, `pct_change`) plus scaled variants (`log_x_100`, `dlog_x_100`/`log_diff_pct`/`dlog_pct`, `dlog_x_400`/`dlog_ann_q`). Runtime fetchers ship for FRED, DBnomics, ONS, BoE IADB plus `csv_file`/`external_archive` (`lib.fetch_data`, cache-on-fetch); the pre-collected FRED-MD/QD CSVs remain the default path |
| [`preprocess/`](preprocess/) | Complex preprocessing utilities: operations that span columns or wrap heavy third-party tooling. `deflate` / `nominalise` (price-deflator conversion), `aggregate` (frequency aggregation M→Q etc. with per-column flow/rate/stock methods) and `impute` (linear / Kalman / EM via Shumway-Stoffer). See [`preprocess/README.md`](preprocess/README.md) for what is not included, the design, and how it fits into a `replicate.py`. Trend-cycle filters (HP, Hamilton, BN, KMW-BN) stay in `filters.py`; per-column univariate transforms stay in `data/transforms.py` |

## Module dependency graph (core modules only; acyclic at module load)

`A ──► B` means B imports A when it loads. Imports inside functions are
not shown (`lp.py` imports `var.py` only inside `blp_irf`; `report.py`
imports `irf` and `diagnostics` helpers inside functions).

```
  var.py ──┬──► inference.py ──► identification.py ──► factors.py
           ├──► identification.py
           ├──► bvar.py
           ├──► connectedness.py
           └──► factors.py

  irf.py ──┬──► identification.py
           ├──► factors.py
           └──► lp.py

  state_space.py ──► gibbs.py

  filters.py, report.py, data/   ← import none of the modules above
```

The `data/` subpackage has no dependencies on other lib modules.

MF-VAR imports `bvar`, `var` and `state_space`; TVP-VAR imports `var` and
`state_space`; DFM imports `factors` and `state_space`.

Each module loads without circular dependencies. The library can be
extended by adding a new file at any DAG level; existing modules don't
need to be touched as long as the new module follows the imports-down rule.

`inference.py` imports its shared VAR helpers (`_ensure_psd`, `_build_x_row`,
`extract_lag_matrices`, `ma_representation`) from `var.py`, which keeps the
graph acyclic.

Top-level imports:

```python
import lib   # scripts/ is on sys.path

# Data, in two steps: load raw, then transform.
# Step 1: structurally-clean raw data with provenance metadata in df.attrs.
raw = lib.load_dataset(
    source="fred_md",
    series={"indpro": "INDPRO", "cpi": "CPIAUCSL", "ffr": "FEDFUNDS"},
    start="1965-01", end="2003-12",
)
# raw.attrs has: source, frequency, vintage, url, loaded_at,
# sample_start_actual / sample_end_actual / n_obs, plus per-series:
# series[alias] = {source_code, recommended_transform (from FRED-MD codes)}

# Step 2: per-model transforms (the transform choice belongs with the model)
data = lib.transform(raw, {"indpro": "log", "cpi": "log", "ffr": "level"})

# Model + identification
fit = lib.fit_var(data, lags=4)
irf = lib.cholesky_irf(fit, horizon=40, bands="asymptotic_mc", n_draws=2000)

# Report
report = lib.VARReport(title="…", subtitle="…")
report.add_irf(irf_point=irf.median,
               ci_68_lo=irf.lower_68, ci_68_hi=irf.upper_68,
               ci_90_lo=irf.lower_90, ci_90_hi=irf.upper_90,
               horizons=irf.horizons,
               var_names=…, shock_labels=…)
report.save("outputs/report.html")
irf.save_snapshot("outputs/numerical_results.json")
```

## Module-to-replication map

The table maps each module to the canonical replications it serves.

| Module | Serves |
|---|---|
| `var.py` + `identification.py` (`cholesky_irf`) | Sims 1980, CEE 1996, CEE 1999, Kilian 2009 |
| `bvar.py` | Litterman 1986, BGR 2010 large BVAR, GLP 2015 hyperparameter optimisation, Lenza-Primiceri 2022 (observation scaling) |
| `lp.py` | Jordà 2005, Stock-Watson 2018 LP-IV, Auerbach-Gorodnichenko 2012 |
| `identification.py` (`proxy_svar_irf`) | Gertler-Karadi 2015, Mertens-Ravn 2013, Romer-Romer 2004 |
| `identification.py` (`long_run_irf`) | Blanchard-Quah 1989, Galí 1999 |
| `identification.py` (`sign_restricted_irf` incl. narrative and higher-moment restrictions, `zero_sign_restricted_irf`) | Uhlig 2005, Antolín-Díaz-Rubio-Ramírez 2018, Andrade-Ferroni-Melosi 2023 |
| `identification.py` (`heteroskedasticity_irf`) | Rigobon 2003 |
| `nongaussian_svar.py` | Lanne-Meitz-Saikkonen 2017 |
| `filters.py` | Hamilton 2018, Kamber-Morley-Wong 2018 BNF, plus HP / Beveridge-Nelson |
| `forecast.py` (`conditional_forecast`) | Bańbura-Giannone-Lenza 2015; forecast evaluation across replications |
| `connectedness.py` | Diebold-Yilmaz 2012 |
| `factors.py` | BBE 2005 FAVAR |
| `dfm.py` | Bańbura-Modugno 2014 DFM |
| `mfvar.py` | Schorfheide-Song 2015 |
| `tvp.py` + `stochastic_volatility.py` | Primiceri 2005 |
| `panel.py` | Canova-Ciccarelli 2013 |
| `dating.py` | business-cycle dating utilities (no dedicated replication) |
| `specs.py` / `diagnostics.py` / `sensitivity.py` / `validation.py` | cross-cutting: declarative runs, residual diagnostics, hyperparameter sweeps, paper-target validation |

## Versioning

The library has no version number of its own: it is versioned with the
archive release (the version in `CITATION.cff`). To regenerate identical
numbers, use the same release and the same data files; the FRED-MD/QD
CSVs are downloaded separately and revised with each vintage.

## Testing

Unit tests live in `tests/` (1,000+ across one test module per lib module,
plus convention / DGP-recovery / coverage / doc-consistency layers; see
`tests/README.md`). Each module should have:

* A canonical correctness check (e.g., for `var.py`, a differential test
  against `statsmodels.tsa.api.VAR`, plus the Lütkepohl textbook oracle
  pinning the conventions).
* End-to-end coverage via the replication smoke tests (`--runslow`), which
  run every replication and fail on any failing `paper_targets.yaml` target.
