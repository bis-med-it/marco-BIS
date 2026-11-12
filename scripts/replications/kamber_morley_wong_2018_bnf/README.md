# Kamber-Morley-Wong (2018): BN Filter on US Real GDP

Canonical replication of KMW's headline result: imposing the paper's low
signal-to-noise ratio delta = psi(1)^2 (trend-shock variance over
forecast-error variance) on the AR(p) for first-differenced log real GDP
produces a **larger, more persistent, NBER-aligned** Beveridge-Nelson cycle.
The restriction is the exact linear constraint sum(phi) = 1 - 1/sqrt(delta-bar)
(delta-bar = 0.25 => sum(phi) = -1), estimated by closed-form constrained OLS.
In this sample the KMW cycle at delta-bar = 0.25 is several times larger
than the textbook BN cycle. Its average is negative in all seven NBER
recessions the sample covers.

Mirrors [`entries/kamber_morley_wong_2018_bnf.yaml`](../../../skills/model-dictionary/entries/kamber_morley_wong_2018_bnf.yaml).

## Specification

| | |
|---|---|
| **Data** | FRED-QD `GDPC1` (real GDP) |
| **Sample** | 1959Q1–2019Q4 (244 obs, pre-COVID to avoid 2020Q2 outlier) |
| **Transform** | `log_x_100` (so cycle units are log-percentage points) |
| **AR order** | 12 (pinned for reproducibility; KMW use 12 for US GDP) |
| **delta-bar (imposed)** | 0.25 (rounded from the paper's delta-bar = 0.24, chosen by automatic amplitude-to-noise selection, p. 554; => sum(phi) = -1, psi(1) = 0.5) |
| **Filter** | [`lib.kmw_bn_filter`](../../lib/filters.py) (exact linear restriction via Dickey-Fuller substitution; closed-form OLS) |

## Results

The unconstrained AR(12) on dlog GDP implies a delta-hat well above 1.
Trend shocks dominate and the BN cycle is small and noisy (the textbook
critique). Imposing delta-bar = 0.25 fixes the trend-shock variance at a
quarter of the one-step forecast-error variance. In this sample the
resulting cycle is several times larger than the unconstrained BN cycle,
more persistent, and recession-shaped.

**delta-bar sensitivity.** The script also fits delta-bar = 0.10, 0.25,
0.50 and 1.00 (sum(phi) = 1 - 1/sqrt(delta-bar), psi(1) = sqrt(delta-bar)).

* Cycle amplitude falls as delta-bar rises: a lower trend-shock share
  pushes more variation into the cycle.
* delta-bar = 1 imposes sum(phi) = 0, i.e. a unit long-run multiplier
  psi(1) = 1: the trend shock carries the full one-step forecast-error
  variance. Short-run growth dynamics can remain. A zero sum does not
  require each AR coefficient to be zero, and it is NOT the unconstrained
  fit, whose delta-hat is well above 1.
* All constrained fits in the grid are stationary.

**NBER recession alignment.** The research log reports the cycle mean and
minimum within each NBER peak-to-trough window. The average KMW gap is
negative in all seven recessions the sample covers, the paper's signature
property. The deepest readings are in Volcker II (1981-82) and the GFC.
The average unconstrained BN gap is (counterintuitively) positive in all
seven: the classical "small jagged cycle" problem KMW set out to fix.

**Inflation forecasting.** In the research-log exercise the AR + KMW gap
has the lowest RMSE at h = 4 and 8 and is about equal to the AR baseline
at h = 1. The differences against the BN gap are not statistically
significant (DM test).

Recessions before 1959 (1948-49, 1953-54, 1957-58) show as `n/a` because
FRED-QD's `GDPC1` starts in 1959Q1. The 1960-61 recession also shows as
`n/a` because differencing and the AR(12) lags use up the early quarters;
the first cycle value is 1962Q1.

**Results.** Run `python replicate.py` in this folder. It writes
`outputs/report.html` (interactive report), `outputs/cycles.csv` (BN and
KMW cycles by date) and `outputs/research_log.json`, which holds the
delta-bar sensitivity, the recession table, the inflation forecast
comparison and the comparison with `paper_targets.yaml`. Numbers depend
on the FRED vintage you download.

## Library implementation

| Step | Function |
|---|---|
| Data load | [`lib.load_dataset`](../../lib/data/loader.py) + [`lib.transform`](../../lib/data/transforms.py) |
| Unconstrained BN | [`lib.beveridge_nelson`](../../lib/filters.py) |
| Constrained KMW BN | [`lib.kmw_bn_filter`](../../lib/filters.py) (sum(phi) = 1 - 1/sqrt(delta-bar) imposed exactly via the Dickey-Fuller substitution phi_1 = s - sum(phi_j, j>=2); closed-form OLS, no optimizer) |

Validated by unit tests in
[`tests/test_filters.py::TestKMW_BN`](../../../tests/test_filters.py),
covering: the exact sum(phi) = 1 - 1/sqrt(delta-bar) contract across
delta values; the larger-cycle headline on a GDP-like DGP; amplitude
monotonicity in delta-bar; trend+cycle identity; stationarity;
NaN / invalid-delta / too-short-series rejection; IC-based lag selection.

## How to run

```bash
python scripts/replications/kamber_morley_wong_2018_bnf/replicate.py
```

## Scope vs original paper

KMW (2018) include several additional analyses we don't replicate here:

* **Comparison against HP** at λ=1600 (the paper's only λ). See the
  sibling replication [`hamilton_2018_filter/`](../hamilton_2018_filter/)
  which runs HP, Hamilton, BN, and KMW-BN side by side on the same
  series.
* **The paper's full out-of-sample design**: Nelson (2008) growth
  regressions and an ADL **CPI** Phillips curve with SIC-chosen lags,
  h = 1–8, expanding window from 1970Q1, DM bands. (This script's
  research-log exercise is a simplified cousin: GDP-deflator inflation,
  expanding window starting from 60 quarters, AR(4)+gap, with the gaps
  estimated on the full sample, so it is not a real-time evaluation.) The growth-forecast regression is
  not implemented.
* **Real-time revision analysis**: the paper's central reliability
  claim (revision SD/RMS less than 1/4 of the ex-post gap SD, vs well
  over 1/2 for HP and other popular methods; ~90% same-sign,
  pp. 557-558). Not implemented (needs real-time data vintages).
* **Bayesian shrinkage estimation**: the paper's baseline imposes a
  Minnesota prior phi*_j ~ N(0, 0.5/j²) (eq. 7); the lib uses the
  acknowledged MLE/OLS variant without shrinkage.
* **Automatic delta-bar selection** (eq. 8) and the **2006Q1
  structural-break / rolling 40-quarter demeaning** treatment (§IV.A).
* **Cross-country / multi-series application.**

## Notes

* **AR order**: the paper uses p=12 for US quarterly GDP (with the
  shrinkage prior making the high order safe; unconstrained SIC
  favours p=1, fn 1). With BIC on this sample, `lib._select_ar_order`
  picks a smaller p (typically p=1–4); we pin p=12 in config to match
  the paper's specification.
* **Sample window**: ending at 2019Q4 excludes the COVID
  outlier. The 2020Q2 -8.9% q/q draw distorts the AR fit (and hence the
  unconstrained delta-hat) severely if included; KMW's original paper
  predates COVID.
* **No structural interpretation**: the BNF cycle is a *statistical*
  trend-cycle decomposition, not a structural output gap. It says
  nothing about productive capacity or potential output in the
  Solow-growth sense.

## Citation

```bibtex
@article{kmw2018bnf,
  author  = {Kamber, Güneş and Morley, James and Wong, Benjamin},
  title   = {Intuitive and Reliable Estimates of the Output Gap from
             a Beveridge-Nelson Filter},
  journal = {Review of Economics and Statistics},
  volume  = {100},
  number  = {3},
  pages   = {550--566},
  year    = {2018}
}
```
