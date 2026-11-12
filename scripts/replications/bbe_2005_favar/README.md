# Bernanke-Boivin-Eliasz (2005) FAVAR: methodological demo

A two-step PC-based FAVAR on FRED-MD monthly data, demonstrating the
[`lib.fit_favar` / `lib.favar_irf`](../../lib/factors.py) implementation.

## Specification

| | |
|---|---|
| **Model class** | FAVAR (two-step PC) with BBE slow/fast rotation |
| **Information panel** | 19 hand-picked FRED-MD variables (real activity, prices, rates, money, asset prices) |
| **Slow block** | 9 variables (INDPRO, PAYEMS, UNRATE, CPIAUCSL, CPIULFSL, PCEPI, RPI, real PCE, wages) |
| **Policy variable** | FEDFUNDS |
| **Factors** | K = 3 |
| **Lags** | 13 monthly |
| **Sample** | 1965M1–2001M8 (paper: 1959M1–2001M8, p. 404; truncated here as a replication choice) |
| **Identification** | Cholesky on the augmented state [F_1, F_2, F_3, FFR], FFR ordered last |
| **Bands** | Parameter-draws (asymptotic), 1 000 draws |

Mirrors [entries/bbe_2005_favar.yaml](../../../skills/model-dictionary/entries/bbe_2005_favar.yaml).

## What this replication demonstrates

- IRFs of the original (un-standardised) panel variables come from
  loadings estimated on the full augmented state `[F, FFR]`. BBE's
  equation (2) includes the observed policy rate among the regressors,
  so panel variables (interest rates above all) can respond to the
  policy shock on impact.
- Bands are projected through Λ draw by draw, then summarised by
  quantiles.

## Results

In this run short and long interest rates rise on impact after the
FFR shock. After the first few months industrial production falls and
unemployment rises, and both stay on that side of zero through the
second year. The median responses of the price indexes are above zero
at every horizon to 48 months, a price puzzle.

**Results.** Run `python replicate.py` in this folder. It writes
`outputs/report.html` (interactive report), `outputs/numerical_results.json`
and `outputs/research_log.json`, which includes the comparison with
`paper_targets.yaml`. Numbers depend on the FRED vintage you download
and on the seed in `config.yaml`.

## What this replication does NOT do

- **Use BBE's full 120-series panel**: we use 19 FRED-MD series for
  reproducibility from the unpacked archive. With a 19-variable panel the factor
  extraction is much noisier than BBE's; the slow/fast partition is also
  smaller (9 vs BBE's 70 slow and 50 fast series, Appendix 1).
- **Use the paper's stationarity transforms**: BBE transform each
  series per Appendix 1 codes (e.g. IP and CPI enter as first
  differences of logs, pp. 403-404); this demo uses log-levels for
  several of them. A leading candidate for the sign issues below.
- **Match BBE's shock units**: the paper standardises the shock to a
  25-basis-point FFR innovation and plots responses in
  standard-deviation units (p. 405); this demo uses a +1-unit Cholesky
  shock.
- **Match BBE's bands**: the paper reports 90% intervals from a
  Kilian (1998) bootstrap that accounts for factor-estimation
  uncertainty (pp. 399, 407); this demo uses asymptotic parameter-draw
  bands.
- **Match BBE's published IRFs**: see Results for the qualitative
  pattern and the price puzzle. Magnitudes are not comparable with the
  paper's 25bp, standard-deviation-unit convention. Candidate causes of
  the gaps are (a) the log-level transforms above, (b) the small panel,
  and (c) sample composition (1965–2001 spans regime shifts).

## Library implementation

Uses `lib.fit_favar` and `lib.favar_irf`; unit tests are in
[`tests/test_factors.py`](../../../tests/test_factors.py).

## How to run

```bash
python scripts/replications/bbe_2005_favar/replicate.py
```

## Citation

```bibtex
@article{bbe2005,
  author = {Bernanke, Ben S. and Boivin, Jean and Eliasz, Piotr},
  title = {Measuring the Effects of Monetary Policy: A Factor-Augmented Vector Autoregressive (FAVAR) Approach},
  journal = {Quarterly Journal of Economics},
  volume = {120},
  number = {1},
  pages = {387--422},
  year = {2005}
}
```
