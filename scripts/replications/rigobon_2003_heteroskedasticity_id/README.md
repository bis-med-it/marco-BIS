# Rigobon (2003) Heteroskedasticity Identification: US recessions as regime

Demonstrates Rigobon's (2003) identification-through-heteroskedasticity
methodology on a three-variable US macro/financial VAR, using NBER
recession dates as the variance-regime indicator. The original paper
identifies contagion in Argentine/Brazilian/Mexican EMBI+ sovereign
yields (12 event/tranquil windows, a common shock, and a US-rate
control). We don't have those data in FRED-MD, so this is a substitute
application of the same identification scheme.

## Specification

| | |
|---|---|
| **Model class** | Reduced-form VAR with heteroskedasticity-based identification |
| **Variables** | Industrial production growth (%), 10-year Treasury yield (level), S&P 500 return (%) |
| **Lags** | 6 monthly |
| **Sample** | 1959-01 to 2024-12 (constrained by FRED-MD coverage) |
| **Regime indicator** | NBER recessions (1 during 9 recessions, 0 otherwise) |
| **Identification** | Generalised eigenvalue decomposition (Σ_recession, Σ_expansion) |
| **Bands** | Parameter draws, 1000 |

Mirrors [entries/rigobon_2003_heteroskedasticity_id.yaml](../../../skills/model-dictionary/entries/rigobon_2003_heteroskedasticity_id.yaml).

## Results

In the current run all three variance ratios (expansion variance ÷
recession variance) are below one, so every structural shock has a higher
variance in recessions. The ratios are separated enough that the library
does not flag weak identification, but the separation is modest. The
heteroskedasticity-identified impact matrix differs markedly from the
Cholesky factor of the same Σ_u.

For the yield-dominated shock (`shock_2`), the 10-year yield rises and its
90% band stays above zero over the 24-month horizon. Stock returns rise on
impact and turn negative within the first year. The IP-growth response is
positive at first and fades.

**Results.** Run `python replicate.py` in this folder. It writes
`outputs/report.html` (interactive report, with the Cholesky IRFs as an
alternative), `outputs/numerical_results.json` (which includes the regime
diagnostics) and `outputs/research_log.json`, which includes the
comparison with `paper_targets.yaml`. Numbers depend on the FRED-MD
vintage you download and on the seed in `config.yaml`.

The regime split uses 104 NBER recession months and 687 expansion months
in the full transformed sample (791 months). The validation targets count
the post-lag effective sample: 104 recession + 681 expansion = 785.

### Identification strength

The library's diagnostic is the **minimum pairwise separation of the
variance ratios on the log scale**, a criterion invariant to swapping the
two regime labels (an absolute eigenvalue gap is not). The library flags
weak identification when the minimum log separation is below the 0.1
heuristic threshold. In the current run it is above the threshold, but
not by a wide margin.

Rigobon-style identification requires the variance ratios to DIFFER
across shocks. For US recessions all three shocks scale up by broadly
similar factors, so the shock-specific separation is modest and the
implied structural matrix remains sensitive to small changes in the
regime definition, even though the ratios are not numerically
coincident.

For the original Rigobon (2003) application (EM sovereign bonds with
country-specific crises), different bonds load on different crises, so
the variance ratios differ substantially and identification is strong.
The US recession application here is a methodological demo; for
informative Rigobon-ID on US data, see Lanne-Lütkepohl (2008) or
Lewis (2021, REStud) who use higher-frequency volatility variation;
Lewis, D. J., "Robust Inference in Models Identified via
Heteroskedasticity" (Review of Economics and Statistics) is the formal
version of the eigenvalue-spread heuristic flagged above.

## Library implementation

Uses [`lib.heteroskedasticity_irf`](../../lib/identification.py),
validated by unit tests in
[`tests/test_identification.py::TestHeteroskedasticityID`](../../../tests/test_identification.py):

- Decomposition exactly reconstructs both regime covariances
- Identical regimes correctly produce weak-identification flag
- Wrong number of regimes rejected
- Misaligned regime length rejected
- IRF output has correct shape

## Substitutions vs original Rigobon (2003)

| Rigobon (2003) | This replication |
|---|---|
| Argentine, Brazilian, Mexican sovereign bond returns | US IP growth, 10Y yield, S&P 500 return |
| Six crisis/event windows + six tranquil windows, nested subsamples | Binary NBER recession indicator |
| Daily frequency | Monthly frequency |
| Strong identification (different bonds load on different crises) | **Modest separation** (US shocks scale similarly in recessions) |

## How to run

```bash
python scripts/replications/rigobon_2003_heteroskedasticity_id/replicate.py
```

## Citation

```bibtex
@article{rigobon2003,
  author = {Rigobon, Roberto},
  title = {Identification through Heteroskedasticity},
  journal = {Review of Economics and Statistics},
  volume = {85},
  number = {4},
  pages = {777--792},
  year = {2003}
}
```
