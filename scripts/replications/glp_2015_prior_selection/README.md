# Giannone-Lenza-Primiceri (2015): Optimal Minnesota Hyperparameters

Demonstrates GLP's closed-form marginal-data-density optimisation of the
Minnesota prior hyperparameters, applied to the [CEE (1999)](../cee_1999_monetary_svar/)
seven-variable BVAR.

## Specification

| | |
|---|---|
| **Setup** | CEE (1999) seven-variable quarterly BVAR (see [`cee_1999_monetary_svar/`](../cee_1999_monetary_svar/)) |
| **Initial prior** | lambda_1=0.1, lambda_2=0.5, lambda_3=1.0, lambda_4=100, mu=1.0, delta=1.0 |
| **Parameters optimised** | lambda_1, mu, delta (jointly) |
| **Method** | scipy `Nelder-Mead`, maximising log marginal data density |
| **MDD formula** | Closed-form NIW dummy-observation MDD (Karlsson 2013, GLP 2015 eq. 5) |

GLP 2015 use the natural-conjugate prior, with covariance `Sigma ⊗ V_0`
and one `V_0 = (X_d'X_d)^{-1}` shared by all equations. A separate
cross-variable tightness `lambda_2` needs a per-equation prior (Litterman
1986; Sims-Zha 1998), which loses the closed-form MDD. So this replication
optimises `{lambda_1, mu, delta}` only, and `lambda_2` stays at its initial
value.

Mirrors [entries/glp_2015_prior_selection.yaml](../../../skills/model-dictionary/entries/glp_2015_prior_selection.yaml).

## Relationship to GLP's paper (read this first)

This replication demonstrates an **empirical-Bayes version** of GLP's
hierarchical approach: it maximises the marginal data density over the
Minnesota hyperparameters (the flat-hyperprior case GLP discuss in Section
II), where GLP's own estimates use hyperpriors and Metropolis draws of the
hyperparameters (p. 439). The test bed is the CEE 1999 seven-variable
monetary BVAR. It is **not** a reproduction of GLP's
own empirical results. Divergences from the paper:

| | GLP 2015 (paper) | This replication |
|---|---|---|
| System | small (3-var), medium (7-var), large (22-var) | CEE 1999 7-var monetary VAR |
| Transforms | annualized log levels (logs × 4) | log levels (CEE convention) |
| Lags | 5 | 4 (CEE) |
| Sample | 1959Q1–2008Q4 | 1965Q1–1995Q4 (CEE-style; the CEE replication itself uses 1965Q3–1995Q2) |
| Hyperparameters | hierarchical posterior for λ, μ, δ with **Gamma hyperpriors** (modes 0.2/1/1), Metropolis draws (p. 439) | λ₁, μ, δ by Nelder-Mead MDD-max, **no hyperpriors** |
| Headline | Table 2 MSFE: BVAR beats flat-prior VAR & RW; tighter λ for larger systems | large in-sample Bayes factor; no-look-ahead OOS roughly at parity with the default prior |

The **paper-faithful spec** (variables, log×4 transforms, 5 lags, 1959–2008,
Gamma hyperpriors) is recorded in the
[dictionary entry](../../../skills/model-dictionary/entries/glp_2015_prior_selection.yaml).

## Results

**Results.** Run `python replicate.py` in this folder. It writes
`outputs/report.html` (interactive report), `outputs/numerical_results.json`
(impulse responses under the optimised prior) and
`outputs/research_log.json`, which holds the initial and optimised
hyperparameters, both log MDDs, the out-of-sample RMSE and
Diebold-Mariano tables, and the comparison with `paper_targets.yaml`.
The console output also prints the default-versus-optimised impulse
responses. Numbers depend on the FRED vintage you download and on the
seed in `config.yaml`.

In this run the optimisation raises the log MDD by a large
margin, so the data clearly prefer the optimised prior to the default.
The optimal overall tightness lambda_1 is looser than the 0.1 default and
close to Litterman's canonical 0.2. The sum-of-coefficients (mu) and
dummy-initial-observation (delta) priors both end up stronger than their
starting values, inside the search bounds.

> **Methodological note.** The MDD uses a proper normal-inverse-Wishart
> prior, including the Sigma-prior dummy block. Without that block the
> prior is improper and the density is not defined. Only MDD differences
> are meaningful, since the level depends on the normalising constant.

## IRF comparison: Real GDP response to monetary policy shock

Under the optimised prior, real GDP falls after a monetary contraction
and stays lower over the horizon. The deflator rises slightly at first
and turns negative later in the horizon, so the price puzzle is mild and
temporary in this run.

## Out-of-sample forecast comparison: does MDD-optimal beat default OOS?

Rolling-window OOS evaluation: 60-quarter window, step=1 (consecutive
quarterly origins), horizons 1..12, both models scored on the same
matched forecast cells, reported at horizons 1, 4, 8 and 12 for the
seven variables. The hyperparameters for this exercise are tuned on the
**initial 60-quarter training window only** and then frozen. The
full-sample optimum above serves the in-sample IRF exhibit, and reusing
it here would let the evaluation observations inform the prior at every
origin.

In this run the tuned prior has the lower RMSE in fewer than half
of the cells. Diebold-Mariano tests reject equal accuracy in only a
minority of cells, and those rejections cut both ways: the tuned prior
does better for the deflator, the default does better for output and the
federal funds rate. So the GLP machinery delivers a large in-sample
Bayes factor, but with the prior tuned only on the training window its
pseudo-out-of-sample accuracy on this dataset is roughly at parity with
the default Litterman prior.

## Library implementation

Uses [`lib.log_marginal_data_density`](../../lib/bvar.py) and
[`lib.optimize_minnesota_hyperparameters`](../../lib/bvar.py); unit tests
are in [`tests/test_bvar.py`](../../../tests/test_bvar.py).

## How to run

```bash
python scripts/replications/glp_2015_prior_selection/replicate.py
```

## Notes

Nelder-Mead reports convergence under its default tolerances (the
evaluation count is in `research_log.json`). This is the best value
found from one starting point, not necessarily the global optimum.

## Citation

```bibtex
@article{glp2015,
  author = {Giannone, Domenico and Lenza, Michele and Primiceri, Giorgio E.},
  title = {Prior Selection for Vector Autoregressions},
  journal = {Review of Economics and Statistics},
  volume = {97},
  number = {2},
  pages = {436--451},
  year = {2015}
}
```
