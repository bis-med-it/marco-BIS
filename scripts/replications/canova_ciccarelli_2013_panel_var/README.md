# Canova-Ciccarelli (2013) Panel VAR: G4 country panel

Replication of the methodological taxonomy in **Canova & Ciccarelli
(2013)**, *Panel Vector Autoregressive Models: A Survey* (in: Fomby,
Kilian, & Murphy, eds., *VAR Models in Macroeconomics*, Advances in
Econometrics vol. 32, Emerald, pp. 205–246; ECB Working Paper 1507),
on a G4 country panel: **UK, US, JP, DE** × **{IPI, CPI, LTR, STR}**,
monthly 1980-2019.

The panel design (countries, variables, 4 lags, short-rate shock) follows
the panel example in the Canova–Ferroni BVAR_ toolbox; the data (FRED,
1980-2019) and the code are this replication's own.

## Why this isn't a single-headline replication

Canova-Ciccarelli 2013 is a **survey paper**, not an empirical headline
paper. It covers a taxonomy of panel-VAR methods (unit-by-unit OLS,
pooling, mean-group, partial pooling / exchangeable priors, factor
shrinkage on the coefficient vector, full panel VAR with
interdependence). The survey itself contains **no simulations and no
empirical estimation**.

This replication is therefore:

* A **methodological demonstration**: fit the three main estimators
  from the survey (pooled, mean-group, global VAR) on real G4 data and
  illustrate the Canova-Ciccarelli messages: (1) pooling vs
  heterogeneity is a real trade-off, (2) cross-country spillovers can
  be probed from a single integrated GVAR.
* **Not a numerical reproduction**: the survey reports no numbers to
  match.

## What this script does

1. **Loads the G4 staging CSV** built by `prepare_data.py` from FRED.
2. **Fits a pooled VAR(4)** with unit fixed effects on the stacked panel.
3. **Fits a mean-group estimator** (Pesaran-Smith 1995) on the same panel.
4. **Fits a global VAR(4) with cross-unit interdependence**,
   the survey's "panel VAR with dynamic interdependencies" (§3.2)
   estimated with large-scale-BVAR-style shrinkage (§6.1). GVAR-
   flavoured but **not** a Dees et al. (2007) GVAR: no a-priori weight
   matrices (a GVAR is "a restricted large scale VAR, where variables
   of different units in an equation are weighted according to w_i",
   §6.4 p. 41). Stacks all units into a single VAR of dimension N=16
   with Minnesota shrinkage; cross-unit blocks are tightened by an
   extra `lambda_cross_unit = 0.05` factor.
5. **Cholesky IRFs** on the pooled fit (ordering `[ipi, cpi, ltr, str]`).
   The last shock is the monetary-tightening shock.
6. **Cross-country IRFs from the GVAR**: identify a US monetary
   tightening as the Cholesky shock to `US_str`, trace responses in
   UK / JP / DE variables (the headline GVAR output).
7. **Compares pooled vs mean-group** lag-1 coefficient matrices. The
   max-absolute-difference is a heterogeneity diagnostic in the spirit
   of the survey's Pesaran-Smith discussion (pooled estimators are
   inconsistent under dynamic heterogeneity while the mean group is
   consistent for the average effect, p. 15).
8. **Validates against `paper_targets.yaml`**: stability + qualitative
   IRF signs.

## Data

| Variable | US | UK | JP | DE |
|---|---|---|---|---|
| IPI (log) | `INDPRO` | `GBRPROINDMISMEI` | `JPNPROINDMISMEI` | `DEUPROINDMISMEI` |
| CPI (log) | `CPIAUCSL` | `GBRCPIALLMINMEI` | `JPNCPIALLMINMEI` | `DEUCPIALLMINMEI` |
| LTR (level) | `GS10` | `IRLTLT01GBM156N` | `IRLTLT01JPM156N` | `IRLTLT01DEM156N` |
| STR (level) | `TB3MS` | `IR3TIB01GBM156N` | `IRSTCB01JPM156N` | `IR3TIB01DEM156N` |

Note **JP STR** uses the central bank short-term rate (Bank of Japan
overnight rate proxy); other countries use 3-month interbank / T-bill.
Documented heterogeneity in the short-rate series; doesn't affect the
pooled VAR's estimation (each unit's STR is the "domestic policy
indicator").

**Sample windows** differ across units (JP LTR only from 1989), and the
panel module handles different T per unit:
* US, UK, DE: 480 monthly observations (1980M1–2019M12)
* JP: 372 monthly observations (1989M1–2019M12)
* Post-lag effective total: 1796 observations

## Global VAR specification

Stacked dimension: N = 4 countries × 4 variables = **16 endogenous
variables**. With 4 lags each equation has **64 lag coefficients plus
an intercept** (65 regressors; 1,040 coefficients across the 16
equations). The common index starts in 1989-01 because of JP's late
start: 372 months, or 368 observations per equation after the lags.
Minnesota shrinkage regularises the many cross-country coefficients.

Hyperparameter setting: `lambda_1 = 0.1`, `lambda_2 = 0.5`,
`lambda_cross_unit = 0.05` (the prior SD of every cross-country
coefficient is multiplied by 0.05, i.e. 20 times tighter than the
matching within-country coefficient), `lambda_3 = 1.0`.

The US monetary tightening in the global VAR is the Cholesky shock to
the `US_str` column (the last-ordered policy variable within the US
country group). Responses are traced in all four countries over
horizons 0 to 24.

## Results

**Results.** Run `python replicate.py` in this folder (after
`prepare_data.py`). It writes `outputs/report.html` (interactive
report), `outputs/numerical_results.json` (the pooled-VAR impulse
responses) and `outputs/research_log.json`, which holds the stability,
pooled-versus-mean-group, global-VAR and cross-country figures and the
comparison with `paper_targets.yaml`. Numbers depend on the FRED
vintage you download.

In this run:

* **Stability.** The pooled, mean-group and global VAR fits are all
  stable, with the largest root close to one. This is what monthly
  log-level data without a trend usually give.
* **Pooling versus heterogeneity.** The pooled and mean-group lag-1
  coefficient matrices differ markedly. This is the Canova-Ciccarelli
  point that a single pooled coefficient set need not match the
  average of the country coefficients when units differ.
* **Pooled monetary tightening.** The short rate rises on impact,
  industrial production is lower after one year, and CPI rises
  slightly, a mild price puzzle. The VAR has no commodity price index.
* **Global VAR spillovers.** Cross-country lag blocks are small next
  to own-country blocks, as the tight `lambda_cross_unit = 0.05` prior
  implies. After a US tightening the US short rate moves most; foreign
  short rates move less and peak late in the horizon. Foreign output
  and CPI responses are very small. With this prior the run cannot
  separate small true spillovers from the effect of the shrinkage.

## How to run

```
# Build staging CSV (one-time; downloads over the FRED graph URL)
python scripts/replications/canova_ciccarelli_2013_panel_var/prepare_data.py

# Run replication
python scripts/replications/canova_ciccarelli_2013_panel_var/replicate.py
```

Outputs land in `scripts/replications/canova_ciccarelli_2013_panel_var/outputs/`.

## Key references

* Pesaran, M. H. & Smith, R. (1995). "Estimating long-run relationships
  from dynamic heterogeneous panels." *Journal of Econometrics* 68(1),
  79–113.

## Not covered

Time-varying-coefficient and factor-augmented panel VARs are not
implemented. `lib.fit_partial_pooling_var` and the exchangeable-prior
`lib.fit_hierarchical_panel_var` exist but are not run here.

## Citation

```bibtex
@incollection{canovaciccarelli2013,
  author    = {Canova, Fabio and Ciccarelli, Matteo},
  title     = {Panel Vector Autoregressive Models: A Survey},
  booktitle = {VAR Models in Macroeconomics — New Developments and
               Applications: Essays in Honor of Christopher A. Sims},
  editor    = {Fomby, Thomas B. and Kilian, Lutz and Murphy, Anthony},
  series    = {Advances in Econometrics},
  volume    = {32},
  pages     = {205--246},
  publisher = {Emerald},
  year      = {2013},
  note      = {Also as ECB Working Paper 1507}
}
```
