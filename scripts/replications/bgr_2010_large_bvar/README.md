# Banbura-Giannone-Reichlin (2010): Large BVARs

Canonical replication of BGR's *scaling rule* for the Minnesota prior:
as the number of variables in a BVAR grows, the overall-tightness
hyperparameter λ₁ must **shrink** so the large BVAR's in-sample fit on
a fixed benchmark set of small-VAR variables matches the small VAR
baseline. Without that calibration, the large BVAR overfits: its
in-sample fit looks superficially better but its out-of-sample
forecasting (per the paper) is worse.

Mirrors [`entries/bgr_2010_large_bvar.yaml`](../../../skills/model-dictionary/entries/bgr_2010_large_bvar.yaml).

## Specification

| | |
|---|---|
| **Data** | FRED-MD monthly, 1959-2019 (732 obs, pre-COVID) |
| **Benchmark** | 3-variable small VAR (`INDPRO`, `CPIAUCSL`, `FEDFUNDS`) |
| **System sizes** | small (3), medium (7), large (17) |
| **Lags** | 13 monthly (paper's choice) |
| **Default prior** | Minnesota with `λ₁=0.20, λ₂=0.5, λ₃=1, μ=δ=1` |
| **Calibration target** | log-average MSE on the benchmark variables matches the small OLS VAR baseline (ratio = 1.0) |
| **Search method** | `scipy.optimize.brentq` on `λ₁ ∈ [0.001, 1.0]` |

## Pipeline

```
lib.load_dataset (FRED-MD, 17 series)
  -> lib.transform (log/level)
    -> lib.fit_var on the small 3-var benchmark
       -> record target log-average benchmark MSE
    -> for each {small, medium, large} system:
         lib.fit_minnesota_bvar at default λ₁ (no calibration)
           -> record default-prior fit ratio
         medium and large systems only:
         brentq over λ₁ matching benchmark log-MSE to the small-VAR target
           -> lib.fit_minnesota_bvar at λ₁*
           -> record calibrated fit ratio
```

The calibration loop calls `lib.fit_minnesota_bvar` inside `brentq`, with
one BVAR fit per evaluation. The benchmark MSE is the mean squared
residual of each benchmark variable.

Validated by the BVAR tests in [`tests/test_bvar.py`](../../../tests/test_bvar.py)
plus the standard VAR / data / transform tests.

## Results

* **λ₁* shrinks as system size grows.** In this run the 7-variable and
  17-variable systems both need a prior much tighter than the default
  λ₁ = 0.20, and the 17-variable system needs the tightest.
* **Without calibration**, the default-λ₁ medium and large BVARs fit
  the benchmark variables *better* in sample than the small VAR. This
  is the overfitting BGR warn against: the large BVAR has 17 × 13 = 221
  lag coefficients per equation, and at default tightness it absorbs
  noise. Paper's central claim: that in-sample-looking-good translates
  to out-of-sample looking *bad*.
* **With calibration**, the fit ratio equals 1 by construction: the
  large BVAR is restricted enough that its benchmark fit equals (not
  exceeds) the small VAR's. This is the disciplined large BVAR that BGR
  show forecasts as well as or better than smaller systems.

**Results.** Run `python replicate.py` in this folder. It writes
`outputs/report.html` (interactive report) and `outputs/research_log.json`,
which holds the calibrated λ₁* and fit ratios per system, the
out-of-sample RMSE comparison described below and the comparison with
`paper_targets.yaml`. Numbers depend on the FRED vintage you download.

## Out-of-sample RMSE: BGR's actual headline

Rolling-window OOS evaluation (240-month window = 20 years, step=12 =
re-fit yearly, horizons 1..12). Three setups compared on the
**benchmark variables** (`INDPRO`, `CPIAUCSL`, `FEDFUNDS`):

* Small VAR(13) on the 3 benchmark variables alone (the baseline).
* Large BVAR(13) on 17 variables at **default λ₁=0.20** (the
  uncalibrated overfitting setup).
* Large BVAR(13) on 17 variables at a **λ₁* calibrated on the
  initial 240-month training window only**, which is the information
  available at the first forecast origin. (The full-sample λ₁* belongs to the
  in-sample calibration exercise above and is not used here.)

In this run the default large BVAR forecasts worse than the small VAR
on average across the benchmark variables and horizons, and the
calibrated large BVAR is close to the small VAR on average. This is in
line with BGR's point that calibration is what makes large BVARs
usable. The calibrated large BVAR has a lower RMSE than the small VAR
for `indpro` at every horizon and for `ffr` at most horizons, but a
higher RMSE for `cpi` at every horizon, so the gain is not across the
board.

## What this replication does not cover

* **The paper's FAVAR forecast comparison** (BGR Section 3, Table III). BGR
  benchmark the calibrated large BVAR against the Bernanke-Boivin-Eliasz
  (2005) FAVAR (on the Stock-Watson dataset), *not* a Stock-Watson DFM.
  Reproducing that head-to-head is out of scope here; the factor machinery
  exists in [`lib.factors`](../../lib/factors.py).
* **The structural analysis** (BGR Section 4): a recursive-SVAR
  monetary-policy shock (IRFs + variance decomposition; larger systems
  resolve the price puzzle). This is not reproduced in this forecasting-focused demo.
* **Real-time vintages.** This replication uses the current FRED-MD
  vintage. The paper used vintage-aware data; a vintage-aware data
  layer is not implemented.

## How to run

```bash
python scripts/replications/bgr_2010_large_bvar/replicate.py
```

Most of the run time goes to the brentq calibration (several BVAR fits
per system) and the OOS rolling-window evaluation (one re-fit per year
× 3 setups × 12-horizon forecasts), dominated by the large 17-variable
BVAR.

## Notes

* **Why such a tight prior for the large system?** With 17 variables ×
  13 lags + 1 intercept = 222 coefficients per equation against 732 obs,
  the unrestricted OLS would have only 3.3 obs per parameter. The tight
  prior compensates for that.
* **Choice of metric.** BGR's exact calibration uses an in-sample
  one-step-ahead MSFE matching condition; we use the *log* of the
  benchmark-variable MSE so the metric is scale-invariant and the root
  search is well-behaved for `brentq`. The economics of the comparison are
  unchanged.
* **The 17 variables.** Drawn from FRED-MD: output (`INDPRO`,
  `IPMANSICS`), prices (`CPIAUCSL`, `PCEPI`), rates (`FEDFUNDS`, `GS1`,
  `GS10`), labour (`PAYEMS`, `UNRATE`, `AWHMAN`, `CES0600000008`,
  `CES2000000008`), money (`M1SL`, `M2SL`), real activity (`HOUST`,
  `DPCERA3M086SBEA`, `BUSINVx`). A subset of BGR's variables (their
  large system has 131).

## Citation

```bibtex
@article{bgr2010lbvar,
  author  = {Bańbura, Marta and Giannone, Domenico and Reichlin, Lucrezia},
  title   = {Large {Bayesian} Vector Auto Regressions},
  journal = {Journal of Applied Econometrics},
  volume  = {25},
  number  = {1},
  pages   = {71--92},
  year    = {2010}
}
```
