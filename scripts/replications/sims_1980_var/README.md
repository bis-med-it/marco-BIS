# Sims (1980): Macroeconomics and Reality

Replicates the six-variable VAR(4) of Sims (1980), *Econometrica* 48(1), 1–48.

This is the foundational macroeconometric VAR paper: a six-variable
recursive system on quarterly US data, fit by OLS, with Cholesky-identified
impulse responses traced 40 quarters out.

## Specification

| | |
|---|---|
| **Model class** | reduced-form VAR with Cholesky identification (OLS; constant + time trend) |
| **Variables** | M1 (log), real GNP (log), unemployment (level), wages (log), GNP deflator (log), import prices (log) |
| **Cholesky ordering** | M1 → real GNP → unemployment → wages → GNP deflator → import prices (paper p. 21: M, Y, U, W, P, PM) |
| **Lags** | 4 quarterly (paper tested 4 vs 8, accepted 4) |
| **Sample** | paper: 1949Q1–1975Q4; replication: **1959Q1–1975Q4** (FRED-QD starts 1959) |
| **Frequency** | Quarterly |
| **Bands** | Monte Carlo from asymptotic coefficient distribution, 2 000 draws |

Mirrors [entries/sims_1980_var.yaml](../../../skills/model-dictionary/entries/sims_1980_var.yaml). For data alignment notes, see `config.yaml`.

## Specification notes

Verified field-by-field against the paper; US system spec is **§B, p. 18**
(variables, sample, lags, trend), the ordering is **§G, p. 21**, and the
quarterly frequency is confirmed by the IRF figure axes (p. 35–41). Two
points worth attention, plus documented proxy substitutions:

- **Sample.** The paper's US fit period is **1949–1975** (p. 18).
  FRED-QD only reaches back to 1959Q1, so the replication runs on
  **1959Q1–1975Q4** (68 obs), matching the paper's 1975 end.
- **Deterministic trend.** The paper states *"the regressions all
  included time trends"* (p. 18), so the VAR uses `trend: "ct"`
  (constant + linear trend). The US system has **no seasonal dummies**
  (those were Germany-only).
- **Proxy substitutions** (FRED stand-ins): nominal M1 reconstructed as
  `M1REAL × CPIAUCSL / 100` (`lib.preprocess.nominalise` with the CPI, the index FRED deflates M1 by; exact at monthly frequency, an approximation with quarterly averages); wages = `CES0600000008`; price level = `GDPCTPI`;
  import prices = `WPSID61` (a PPI, and only a rough proxy for Sims's
  import-price index).

## Results

In the current run a positive M1 innovation (M1 ordered first) raises real
output and lowers unemployment in the first few quarters. Both responses
then reverse. The price level rises for several quarters and then falls
back, so the rise does not last over the full 40-quarter horizon.

The 90% bands are wide and widen sharply at long horizons. The largest
companion-matrix root is slightly above one in modulus. (The price-puzzle
discussion and the commodity-price remedy belong to Sims (1992) and CEE
(1999), not this M1-shock baseline.)

**Results.** Run `python replicate.py` in this folder. It writes
`outputs/report.html` (interactive report), `outputs/numerical_results.json`
and `outputs/research_log.json`, which includes the comparison with
`paper_targets.yaml`. Numbers depend on the FRED-QD vintage you download and
on the seed in `config.yaml`.

## How to run

```bash
cd <repo root>
python scripts/replications/sims_1980_var/replicate.py
```

## Reproducibility

Re-running regenerates identical numbers only with the same code,
`config.yaml`, random seed (`general.random_seed`), Python dependencies
**and the same FRED-QD data file**. The panel is downloaded separately
(`data/fred_qd/UPDATE.md`) and revised with each monthly vintage, so keep
the data file you used alongside the run.

## Citation

If you use this replication, please cite Sims (1980) and, where appropriate,
the MaRCo@BIS framework (see `CITATION.cff` at the repository root):

```bibtex
@article{sims1980,
  author = {Sims, Christopher A.},
  title = {Macroeconomics and Reality},
  journal = {Econometrica},
  volume = {48},
  number = {1},
  pages = {1--48},
  year = {1980}
}
```
