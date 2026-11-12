# Romer-Romer (2004): Narrative Monetary Shocks via Proxy-SVAR

Canonical replication of Romer-Romer (2004)'s narrative monetary policy
shock applied as an **external instrument** in a 4-variable proxy-SVAR.
Same architectural pattern as
[`gertler_karadi_2015_proxy_svar/`](../gertler_karadi_2015_proxy_svar/):
only the instrument changes (Greenbook-purged narrative residuals
instead of high-frequency FOMC-window surprises).

Mirrors [`entries/romer_romer_2004_narrative_mp.yaml`](../../../skills/model-dictionary/entries/romer_romer_2004_narrative_mp.yaml).

## ⚠ Data requirement (user must stage)

The RR shock series is **not on a stable public URL**. The user must
stage a CSV at:

```
data/staging/romer_romer_shocks.csv
```

with the layout:

```
date,rr_shock
1969-01-01,0.005
1969-02-01,-0.012
...
```

`data/staging/` is not distributed; it holds files a user must download
by hand. If the file is missing, the script exits with a message pointing
to the sources below.

## Where to obtain the RR shock series

| Source | Range | Notes |
|---|---|---|
| Romer-Romer (2004) AER replication archive (openICPSR 116025) | 1969M1–1996M12 | Original paper data |
| Coibion (2012) AEJ:Macro replication archive | extended to 2002 | Same methodology, later sample |
| Wieland-Yang (2020) JMCB replication archive | extends into the late 2000s | Updated Greenbook handling |
| Aruoba-Drechsel (2024, NBER WP 32417) | see the paper | Text-augmented shocks from pre-meeting staff documents (Greenbooks/Tealbooks, Redbooks/Beigebooks) |

After downloading from any of these, rename the shock column to
`rr_shock`, ensure the date column is the first column, and save to
`data/staging/romer_romer_shocks.csv`.

## Specification

| | |
|---|---|
| **Macro data** | FRED-MD monthly: `INDPRO`, `CPIAUCSL`, `UNRATE`, `FEDFUNDS` |
| **Instrument** | Romer-Romer narrative MP shock (staged CSV) |
| **Lags** | 12 monthly |
| **Policy variable** | `FEDFUNDS` |
| **Normalisation** | unit-FFR shock on impact |
| **Bands** | Wild bootstrap (Rademacher), 1 000 draws (the GK/MR legacy convention); Jentsch and Lunsford (2019) show this wild bootstrap is asymptotically invalid for proxy SVARs, so the bands have no established coverage (the library's `bands="moving_block_bootstrap"` is the recommended alternative for this single-instrument case) |

## Pipeline

```
lib.load_dataset (FRED-MD, 4 macro series)
  -> lib.transform (log/level)
    -> lib.fetch_data(api="csv_file") for the staged RR shock CSV
      -> inner-join macro + shock; instrument = shock column
        -> lib.fit_var (VAR(12), trend='c')
          -> lib.proxy_svar_irf (wild-bootstrap, RR instrument)
            -> save IRF snapshot + research log
```

All identification machinery is the same code path as the gertler_karadi
replication, validated by the proxy-SVAR unit tests in
[`tests/test_identification.py`](../../../tests/test_identification.py).

## Expected results

The paper offers two sets of magnitudes for a 1-percentage-point
contractionary shock:

* **Single-equation estimates** (eq. 2/3, pp. 1069-73): IP falls from
  month ~5 to **-4.3%**, flat over months 22-27, back to -1.7% at
  month 48; the **PPI for finished goods** is virtually unchanged for
  22 months, then falls to **-5.9%** at month 48, with no price puzzle.
* **The paper's own VAR** (3 variables, 36 lags, cumulated shock,
  p. 1079): output peak **-2.9%**, prices -5.0% at 48 months. These are smaller,
  "presumably [because] a substantial part of the interest rate
  movement is undone quickly".
* **FFR**: rises more than one-for-one with the shock for the first
  few months (fn 19, p. 1072).

Notes for this replication: the price variable here is **CPI**
(FRED-MD) while the paper's baseline is the PPI for finished goods
(CPI ex shelter robustness: -3.6% at 48m); the paper estimates **no
unemployment response** (UNRATE in this VAR has no paper counterpart);
the **first-stage F** is a framework diagnostic (RR 2004 contains no
proxy-SVAR), and 12 lags is a framework convention (the paper's fn 27
shows 12 lags understate effects relative to its 36-lag baseline).

**Results.** This replication's results require the staged RR shock
file, so this README reports none. After staging, run
`python scripts/replications/romer_romer_2004_narrative_mp/replicate.py`
from the repository root. It writes `outputs/report.html`
(interactive report), `outputs/numerical_results.json` and
`outputs/research_log.json`, which includes the comparison with
`paper_targets.yaml`. Numbers depend on the FRED-MD vintage you download,
on the shock file you stage and on the seed in `config.yaml`.

## How to run

```bash
# 1. Stage data/staging/romer_romer_shocks.csv (see above).
# 2. Run:
python scripts/replications/romer_romer_2004_narrative_mp/replicate.py
```

If the file isn't staged, the script exits cleanly with sourcing
instructions; no outputs are produced.

## Scope vs original paper

Romer-Romer (2004) is half shock-construction (Section I), half
implications (Section II): the paper's own outcome analysis is
single-equation distributed-lag regressions plus a 3-variable recursive
VAR with the cumulated shock (36 lags). This replication takes the
shock series as given (per the dictionary entry's
`model.class: "shock_series"` annotation) and demonstrates the
**post-2004 proxy-SVAR usage** (Stock-Watson 2018) rather than either
of the paper's own designs.

The shock-construction step (regression of intended FFR changes on
Greenbook forecasts) is **not** reproduced here: it requires Greenbook
data and FOMC transcript coding that are upstream of FRED. The Aruoba-
Drechsel (2024) NLP-based update is the closest end-to-end
reconstruction.

## Citation

```bibtex
@article{romer_romer2004,
  author  = {Romer, Christina D. and Romer, David H.},
  title   = {A New Measure of Monetary Shocks: Derivation and Implications},
  journal = {American Economic Review},
  volume  = {94},
  number  = {4},
  pages   = {1055--1084},
  year    = {2004}
}
```
