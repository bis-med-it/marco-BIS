# Andrade, Ferroni, Melosi (2023): Higher-Moment SVAR

Combines standard multi-shock sign restrictions (RWZ 2010) with
higher-moment inequality restrictions on the structural shock series
(AFM 2023). The headline AFM claim: adding skewness / kurtosis
restrictions sharpens set-identified rotations and narrows IRF
credible bands relative to sign-only.

This replication tests that claim on a US monetary VAR (INDPRO / CPI /
FEDFUNDS / BAA-FFR spread, monthly 1965-2007). This is a methodological
cousin of the paper's Application 1, which uses the **Uhlig (2005) dataset**
(real GDP, GDP deflator, commodity price index, FFR; 1965m1–2003m12,
VAR(12)) with the MP shock's **robust excess kurtosis > 1.2** and
output unrestricted. Our FRED-MD system differs (monthly activity/price
proxies; BAAFFM as a financial-conditions stand-in).

**Result in this sample**: the higher-moment restriction does
*not* tighten bands. Most sign-passing rotations fail the kurtosis
filter, and the bands from the rotations that pass are wider than the
sign-only bands. See §"Interpretation and limitations" below for
candidate explanations and for when AFM's tightening claim is more
likely to transfer.

## Specification

| | |
|---|---|
| **Model class** | Reduced-form VAR + multi-shock sign restrictions (RWZ 2010) |
| **Variables** | log(INDPRO), log(CPIAUCSL), FEDFUNDS (level), BAAFFM (BAA − FFR, level) |
| **Lags** | 12 monthly |
| **Sample** | 1965-01 to 2007-12 (516 obs, T_eff = 504) |
| **Identification** | Three shocks: `demand_shock`, `supply_shock`, `monetary_shock` |
| **Higher-moment restriction** | `excess_kurtosis(monetary_shock) ≥ 1.0` |
| **Acceptance target** | 500 rotations per pass |

Sign restrictions (impact, `h=0`):

| Shock | indpro | cpi | ffr | spread |
|---|:---:|:---:|:---:|:---:|
| demand_shock | + | + | + | (unrestricted) |
| supply_shock | + | − | (unrestricted) | (unrestricted) |
| monetary_shock | − | − | + | + |

Mirrors [entries/andrade_ferroni_melosi_2023_higher_moments.yaml](../../../skills/model-dictionary/entries/andrade_ferroni_melosi_2023_higher_moments.yaml).

## Results

In this run most rotations that pass the sign restrictions fail the
kurtosis filter. Under sign plus higher-moment restrictions, the median
response to the contractionary monetary shock shows industrial
production falling and the spread widening in the first year, and the
FFR rising on impact and then falling below zero within a few months.

The average 68% band is wider under the higher-moment restriction than
under sign restrictions alone, for all four variables. This contrasts
with the paper's qualitative sharpening claim ("significantly sharpens
identification"; output's positive responses "chopped away", prices
about half the sign-only response, p. 30; the paper states no
percentage figure). See §"Interpretation and limitations" next.

**Results.** Run `python replicate.py` in this folder. It writes
`outputs/report.html` (interactive report), `outputs/numerical_results.json`
and `outputs/research_log.json`, which includes the acceptance counts
for both passes, the band-width comparison and the comparison with
`paper_targets.yaml`. Numbers depend on the FRED vintage you download
and on the seed in `config.yaml`.

## Interpretation and limitations

### Why bands widen rather than tighten

The experiment does not isolate why the bands widen. Candidate
explanations include sampling noise in the kurtosis statistic, which
parameter draws pass the filter, and differences from the paper's data
and estimators. The paper's own setting differs in these ways:

* **Robust percentile-based moment estimators** (eq. 1; Moors /
  Crow-Siddiqui kurtosis) instead of sample kurtosis, which makes them less
  sensitive to small-sample noise;
* **Petrova (2022) robust Bayesian posterior with flat priors** and a
  fourth-moment shrinkage estimator for Σ, giving a different
  parameter-uncertainty pattern from our OLS + asymptotic draws;
* **A threshold calibrated externally** (below the minimum robust
  kurtosis, 1.4, across the Table-1 proxy menu measured on 134–518
  high-frequency observations) and a **restriction horizon of impact +
  6 months** on the signs.

### Form of the restriction

The Pass-2 restriction is an own-shock kurtosis threshold, the form the
paper uses (US: robust excess kurtosis > 1.2; EA: skewness > 1 and
kurtosis > 3). The library also supports cross-shock kurtosis
comparisons, which the paper does not use and this replication does not
run.

## Library implementation

Uses `lib.sign_restricted_irf(..., higher_moment_restrictions=...)`
([`scripts/lib/identification.py`](../../lib/identification.py)). Unit
tests are in
[`tests/test_higher_moment_svar.py`](../../../tests/test_higher_moment_svar.py).

## Citation

```bibtex
@techreport{andrade_ferroni_melosi_2023,
  author      = {Andrade, Philippe and Ferroni, Filippo and Melosi, Leonardo},
  title       = {Identification Using Higher-Order Moments Restrictions},
  institution = {Federal Reserve Bank of Chicago},
  number      = {WP 2023-28},
  year        = {2023}
}
```

## How to run

```bash
python scripts/replications/andrade_ferroni_melosi_2023_higher_moments/replicate.py
```

Outputs:

| File | What it is |
|---|---|
| `outputs/report.html` | Pass-2 IRFs with 68% and 90% bands from asymptotic parameter draws and random rotations (sign + HM) |
| `outputs/numerical_results.json` | Pass-2 IRF tensor snapshot |
| `outputs/research_log.json` | Both passes, band-width comparison, config, provenance |
