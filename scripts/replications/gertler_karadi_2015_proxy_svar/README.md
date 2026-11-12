# Gertler-Karadi (2015) Proxy-SVAR: FRED-MD substitutes

Demonstrates the proxy-SVAR identification method of Gertler & Karadi (2015),
*AEJ:Macro* 7(1), 44–76, using
[`lib.proxy_svar_irf`](../../lib/identification.py).

## Specification

| | |
|---|---|
| **Model class** | Reduced-form VAR with proxy-SVAR identification |
| **Variables** | GS1 (1-Year Treasury, level), log IP, log CPI, **BAA-AAA spread (level)** |
| **Lags** | 12 monthly |
| **Sample** | 1990M2–2024M1 (JK MP shock availability) |
| **Policy variable** | GS1 |
| **Instrument** | **Jarocinski-Karadi MP_median** (HF FOMC-window surprise) |
| **Normalisation** | unit shock to GS1 |
| **Bands** | Wild bootstrap (Rademacher), 1 000 replications (the GK/MR legacy convention); Jentsch and Lunsford (2019) show this wild bootstrap is asymptotically invalid for proxy SVARs, so the bands have no established coverage |

## Substitutions vs the original paper

This replication uses FRED-MD series plus the Jarocinski-Karadi shock
file, which is fetched on the first run. Two GK series aren't in FRED-MD:

| Original GK 2015 | Substitute used here |
|---|---|
| **Excess Bond Premium** (Gilchrist-Zakrajšek 2012, updated by Favara et al.) | **BAA-AAA credit spread** (FRED-MD: `BAA - AAA`) |
| **ff4** (3-month fed funds future surprise in 30-min window) | **Jarocinski-Karadi MP_median** (HF FOMC surprise, monthly aggregated) |

Both substitutions are conceptually aligned (credit-risk spread and HF
monetary surprise), but they differ in construction. `MP_median` is the
monetary-policy component of Jarocinski-Karadi's (2020) decomposition of
FOMC surprises (the central-bank-information component is the separate
`CBI_median` series), aggregated to monthly frequency, and it covers a
later sample than ff4. As a result the responses below are not directly
comparable to GK's published figures.

## What the proxy-SVAR mechanics produce

**Results.** Run `python replicate.py` in this folder. It writes
`outputs/report.html` (interactive report), `outputs/numerical_results.json`
and `outputs/research_log.json`, which holds the first-stage statistics and
the comparison with `paper_targets.yaml`. Numbers depend on the FRED-MD
vintage you download, the cached JK shock file and the seed in
`config.yaml`.

The first-stage F is below the usual threshold of 10, so the instrument
is weak in this sample. The largest companion root is slightly above one
in this run. The responses are normalised so GS1 rises by 1.0 on
impact, and several signs differ from GK's figures:

- log IP rises after the "monetary tightening" rather than falling,
  and turns negative only at long horizons
- log CPI is mildly positive over most of the horizon (price puzzle)
- The credit spread rises on impact, as in GK, but only briefly: it is
  negative by month 12

**The cause of these differences is not established here.**
Candidate explanations are the two substitutions, the later and longer
sample (1990–2024, including the effective-lower-bound years) and the weak
first stage (F < 10), which makes the identified responses imprecise.
`MP_median` already excludes JK's central-bank-information component, so
information effects are not an obvious explanation. The direct test would
be GK's original setup (ff4, the EBP and GK's sample), which is not
implemented here (see below).

## Checks in this replication

The run reports the first-stage F and sets the weak-instrument flag when
F < 10. The GS1 response at h=0 is exactly 1.0 by the unit
normalisation. The wild-bootstrap bands are kept for comparison with GK
only (see the Bands row above). Unit tests for `lib.proxy_svar_irf` are
in `tests/test_identification.py`.

## How to run

```bash
python scripts/replications/gertler_karadi_2015_proxy_svar/replicate.py
```

Internet access is required only on the *first* run, when the JK shock CSV is
fetched via [`lib.fetch_data(api="external_archive", ...)`](../../lib/data/fetcher.py)
and cached at `data/jk_shocks/dataset.csv` with per-series provenance in
`data/jk_shocks/metadata.yaml`. Subsequent runs (and any other replication
referencing `dataset_id="jk_shocks"`) read from the cache without an HTTP
call. FRED-MD itself is loaded via [`lib.load_dataset`](../../lib/data/loader.py)
and transformed via [`lib.transform`](../../lib/data/transforms.py); the
BAA-AAA spread is computed in `replicate.py` before the transform step.

## GK's own setup (not implemented)

GK use `ff4` (AEA replication archive, DOI 10.1257/mac.20130329), the
Gilchrist-Zakrajšek excess bond premium, a 1979M7–2012M6 VAR sample and
a 1991M1–2012M6 instrument sample (p. 55).

## Citations

```bibtex
@article{gertlerkaradi2015,
  author = {Gertler, Mark and Karadi, Peter},
  title = {Monetary Policy Surprises, Credit Costs, and Economic Activity},
  journal = {American Economic Journal: Macroeconomics},
  volume = {7},
  number = {1},
  pages = {44--76},
  year = {2015}
}
```
