# Stock-Watson (2018) LP-IV with Jarocinski-Karadi MP Shocks

Demonstrates the LP-IV methodology of Stock & Watson (2018), *EJ* 128(610),
917–948, applied with the Jarocinski-Karadi (2020) high-frequency monetary
policy shock as the external instrument for the federal funds rate. Response
variables: CPI, CPI excluding food (`CPIULFSL`; config key `core_cpi`),
industrial production, and FFR.

## Specification

| | |
|---|---|
| **Method** | Local projections IV (scaled reduced-form) |
| **Responses** | CPI (log), CPI excluding food (`CPIULFSL`, log), industrial production (log), FFR (level) |
| **Endogenous policy variable** | Federal funds rate |
| **External instrument** | Jarocinski-Karadi (2020) `MP_median` (median-rotation HF surprise) |
| **Controls** | 4 lags of all 4 responses + 4 lags of the instrument |
| **Sample** | 1990M2–2024M1 (FRED-MD intersected with JK shock availability) |
| **Horizon** | 0–48 months |
| **HAC** | Newey-West with `h + 1` lags |

## Methodology vs the original Stock-Watson 2018 paper

The dictionary entry [stock_watson_2018_lpiv.yaml](../../../skills/model-dictionary/entries/stock_watson_2018_lpiv.yaml)
documents the canonical LP-IV approach. Stock & Watson's empirical
illustration (§4) uses **Gertler-Karadi (2015)'s Fed Funds futures (FFF)**
high-frequency surprises (monthly 1990m1–2012m6); this replication substitutes
the more recent Jarociński-Karadi (2020) HF instrument and extends the sample
through 2024. (Romer-Romer narrative shocks are only a *cited* external-IV
example in the paper, not its worked illustration.)

The library implements scaled-reduced-form LP-IV: for each horizon ``h``,
the impulse response is the reduced-form coefficient ``β_h^RF`` (from
regressing ``y_{t+h}`` on the instrument plus controls) divided by the
first-stage coefficient ``γ`` (from regressing the policy variable on the
instrument plus controls). This is equivalent to 2SLS in the just-identified
case but numerically more stable.

## Results

In the current run the first-stage F-statistic is above the rule-of-thumb
threshold of 10. CPI and CPI excluding food rise at long horizons, and
industrial production falls to a trough in the second year before
recovering. The 90% bands on these three responses include zero at almost
every horizon.

**Results.** Run `python replicate.py` in this folder. It writes
`outputs/report.html` (interactive report), `outputs/numerical_results.json`
(which includes the first-stage statistics) and `outputs/research_log.json`,
which includes the comparison with `paper_targets.yaml`. Numbers depend on
the FRED-MD vintage you download and on the cached JK shock file.

The run uses the CONTIGUOUS merged sample (1990-02 to 2024-01, 408
months): the few JK months without usable announcements (September 2001
above all) enter as zero surprise. Dropping those rows instead would
corrupt the calendar lags and remove high-leverage observations, and it
gives a larger first-stage coefficient. Long-horizon LP-IV magnitudes are
imprecise (the 90% bands are wide); treat peak values as qualitative.

Small differences against 2SLS implementations reflect HAC covariance
numerics and sample edge effects, not specification.

## Interpretation

The positive response of CPI and CPI excluding food to a contractionary monetary shock is
the well-known "price puzzle" at LP-IV horizons, although the bands are
wide. Industrial production shows the expected contractionary hump
followed by overshoot. The FFR response is positive on impact, turns
negative in the second year, and rises far above its impact value at long
horizons. The JK decomposition already separates
central-bank-information effects into `CBI_median`; this replication uses
the monetary-policy component `MP_median`, so information effects are not
an obvious explanation for the price puzzle here, and this exercise does
not establish its cause.

## How to run

```bash
python scripts/replications/stock_watson_2018_lpiv/replicate.py
```

Internet access is required only on the *first* run: the JK shock CSV is
fetched via `lib.fetch_data(api="external_archive", ...)` and cached at
`data/jk_shocks/dataset.csv` (with per-series provenance in
`data/jk_shocks/metadata.yaml`). Subsequent runs (and any other
replication referencing `dataset_id="jk_shocks"`, such as
gertler_karadi_2015) read from the cache without an HTTP call.

## Citations

```bibtex
@article{stockwatson2018,
  author = {Stock, James H. and Watson, Mark W.},
  title = {Identification and Estimation of Dynamic Causal Effects in Macroeconomics Using External Instruments},
  journal = {Economic Journal},
  volume = {128},
  number = {610},
  pages = {917--948},
  year = {2018}
}

@article{jarocinskikaradi2020,
  author = {Jarocinski, Marek and Karadi, Peter},
  title = {Deconstructing Monetary Policy Surprises — The Role of Information Shocks},
  journal = {American Economic Journal: Macroeconomics},
  volume = {12},
  number = {2},
  pages = {1--43},
  year = {2020}
}
```
