# Mertens-Ravn (2013): Tax Shocks via Multi-Instrument Proxy SVAR

Canonical replication of Mertens & Ravn (2013) "The Dynamic Effects of
Personal and Corporate Income Tax Changes in the United States" (AER
103(4), 1212-1247). Distinguishes personal income tax shocks from
corporate income tax shocks via TWO narrative external instruments and
TWO endogenous policy variables (APITR and ACITR, the average personal /
corporate income tax rates), identified jointly via the §II algebra.

Uses the multi-instrument extension to `lib.proxy_svar_irf`.

Mirrors [`entries/mertens_ravn_2013_tax_proxy_svar.yaml`](../../../skills/model-dictionary/entries/mertens_ravn_2013_tax_proxy_svar.yaml).

## Specification

| | |
|---|---|
| **Setup** | 7-variable quarterly SVAR, 4 lags, 1950Q1-2006Q4 |
| **Endogenous** | APITR, ACITR, PITB, CITB, log real FEDERAL govt purchases p.c., log real GDP p.c., log real per-capita debt HELD BY THE PUBLIC (p. 1245). Overall ordering doesn't matter (proxy identification), but the WITHIN-policy-block normalisation does: the paper uses a triangular S_1 reporting both orderings; the lib's `unit_policy` (H_1^p = I_2) is a stricter documented deviation, benign at the paper's estimated shock correlation of -0.07 (p. 1225) |
| **Policy variables** | APITR, ACITR |
| **Instruments** | Personal narrative tax shock, Corporate narrative tax shock |
| **Identification** | Multi-instrument proxy SVAR (MR 2013 §II); H_1^p = I_2 normalisation; wild bootstrap for bands (n_draws=1000) |
| **Library function** | `lib.proxy_svar_irf(fit, instrument=Z, policy_var=["apitr", "acitr"], ...)` (multi-instrument path) |

## ⚠ Data requirement (user must stage)

APITR / ACITR / PITB / CITB are NOT in FRED-QD; they require BEA NIPA
constructions. The narrative tax shock series are from the Mertens-Ravn
data archive. **All nine columns + a date column must be pre-staged**
at:

```
data/staging/mertens_ravn_2013.csv
```

Layout:

```
date,apitr,acitr,pitb,citb,log_real_g,log_real_gdp,log_real_debt,personal_shock,corporate_shock
1950-01-01,...
1950-04-01,...
...
```

`data/staging/` is not distributed (per the established cache-on-fetch
convention for archive-sourced data). If the file is missing, the
script exits with a clear message pointing to the sources below.

## Where to obtain the data

| Component | Source |
|---|---|
| APITR, ACITR, PITB, CITB | Mertens-Ravn 2013 AER replication archive: search "Mertens Ravn 2013" on the [AEA article page](https://www.aeaweb.org/articles?id=10.1257/aer.103.4.1212), download the replication package, and use the constructed quarterly series. |
| Personal & corporate narrative shocks | Same archive (columns `PITAX_NAR` and `CITAX_NAR`); quarterly post-WWII series. |
| Macro controls | Either reuse from the archive OR compute (MR Appendix B, pp. 1245-46). Real GDP (`GDPC1`, NIPA T1.1.3 L1) and real FEDERAL purchases (`FGCEC1`, NIPA T1.1.3 L22; NOT total-government GCEC1) are **already real**: divide each by population and log, with no further deflation. Federal debt HELD BY THE PUBLIC (Favero-Giavazzi DEBTHP; `FYGFDPUN`-type) is nominal: divide by the GDP deflator and by population, then log. Population is total population aged 16+ (Francis-Ramey `nipop16`; FRED `CNP16OV` is the closest public substitute). |

After downloading, ensure column names match the layout above and save
to `data/staging/mertens_ravn_2013.csv`. Once the CSV is in place,
running `python scripts/replications/mertens_ravn_2013_tax_proxy_svar/replicate.py`
produces the full IRF set with no further configuration.

## What the replication produces

* **First-stage diagnostics**: per (instrument, policy) pair F-statistic,
  γ, R²; condition number of the policy-block residual-instrument
  covariance Σ_p (low = instruments are well-separated). Min own-pair
  F flagged STRONG / WEAK at the conventional threshold of 10.
* **IRFs to a 1pp personal tax cut**: sign-flipped from the
  unit-policy convention so the policy variable falls by 1pp on impact.
  The paper reports that output rises 1.4% in the first quarter and
  peaks at 1.8% after 3 quarters (p. 1227, Fig. 2), with consumption as
  the main channel; debt rises mechanically.
* **IRFs to a 1pp corporate tax cut**: similar reporting. The paper
  reports a smaller and less persistent output response than for
  personal taxes (p. 1228, Fig. 3), with investment as the main channel.
  Consumption and investment are not in this 7-variable system.

**Results.** This replication's results require the staged data file,
so this README reports none. After staging, run `python replicate.py` in
this folder. It writes `outputs/report.html` (interactive report),
`outputs/numerical_results.json` (the IRF snapshot) and
`outputs/research_log.json` (specification and first-stage diagnostics),
which includes the comparison with `paper_targets.yaml`. Numbers depend
on the archive vintage you stage and on the seed in `config.yaml`.

## Library implementation

| Step | Function |
|---|---|
| Data load | CSV from `data/staging/` (file read inline; not via `lib.fetch_data` because the file is user-staged from a non-public archive) |
| Transform | `lib.transform` |
| VAR estimation | `lib.fit_var` (7-var quarterly VAR(4)) |
| **Multi-instrument proxy SVAR** | `lib.proxy_svar_irf(fit, instrument=Z, policy_var=["apitr", "acitr"], normalization="unit_policy")`; the multi-instrument dispatch detects 2D instrument + list of policy variables |
| Bootstrap inference | Built into `proxy_svar_irf`: the MR/GK-legacy Rademacher wild bootstrap, which Jentsch and Lunsford (2019) show is asymptotically invalid for proxy SVARs (see the caveat below) |

Validated by the multi-instrument tests in
[`tests/test_identification.py::TestProxySVARMultiInstrument`](../../../tests/test_identification.py).
The bootstrap test checks band shape only, not coverage.

## Substitutions / caveats

* **Jentsch-Lunsford (2019) moving-block bootstrap**: implemented in the
  lib for single-instrument proxy-SVARs (`bands="moving_block_bootstrap"`)
  but not on the multi-instrument path used here, so this replication
  uses the GK-style wild bootstrap. Jentsch and Lunsford (2019) show that
  bootstrap is asymptotically invalid for proxy SVARs, so the reported
  bands are a legacy convention for comparison with MR's figures, not
  intervals with established coverage; do not read significance from them.
* **APITR / ACITR construction depends on the archive vintage**. The
  staging CSV is the user's responsibility. If you use a different
  construction, the point estimates will shift. The effect on the qualitative story has
  not been tested.
* **No Romer-Romer narrative reconstruction**. The replication consumes
  the MR archive's instruments directly; we don't re-build them from
  the Romer-Romer (2010) source narratives.

## How to run

```bash
# After staging data/staging/mertens_ravn_2013.csv:
python scripts/replications/mertens_ravn_2013_tax_proxy_svar/replicate.py
```

The bootstrap is the dominant cost; reduce `irf.n_draws` in
`config.yaml` for faster iteration.

If the staging CSV is absent, the script exits with `*** Mertens-Ravn
2013 staging CSV not found ***` and a list of the expected columns plus
the AER archive link.

## Citation

```bibtex
@article{mertensravn2013,
  author  = {Mertens, Karel and Ravn, Morten O.},
  title   = {The Dynamic Effects of Personal and Corporate Income Tax
             Changes in the United States},
  journal = {American Economic Review},
  volume  = {103},
  number  = {4},
  pages   = {1212--1247},
  year    = {2013}
}
```
