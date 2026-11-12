# Galí (1999): Technology vs Non-Technology Shocks

Replicates Galí (1999), *AER* 89(1), 249–271, "Technology, Employment, and
the Business Cycle: Do Technology Shocks Explain Aggregate Fluctuations?".
A bivariate VAR in productivity growth and hours growth, identified by the
restriction that **only technology shocks may have a permanent effect on
the level of labour productivity** (pp. 255–257). The paper's headline
anti-RBC finding: conditional on technology shocks, productivity and hours
comove *negatively*: hours **fall** after a positive technology shock.

## Specification

| | |
|---|---|
| **Model class** | Reduced-form VAR with long-run identification |
| **Variables** | Δlog productivity (×100), Δlog hours (×100); productivity is **constructed** as log GDP − log hours, exactly as in the paper (p. 258) |
| **Order in VAR** | productivity first (carries the identifying unit root), hours second |
| **Lags** | 4 quarterly (**not stated in the published paper**; see Specification notes) |
| **Sample** | paper: 1948:1–1994:4 (p. 258); replication: 1959Q1–1994Q4 (FRED-QD start) |
| **Identification** | Lower-triangular long-run impact: shock 2 (non-technology) has zero permanent effect on the productivity *level* (paper's single restriction, p. 257) |
| **Sign normalisation** | Technology shock raises productivity on impact; non-technology raises hours (signs are irrelevant for the conditional correlations) |
| **Bands** | Asymptotic Monte Carlo on the VAR coefficients, 2 000 draws (paper fn. 17 uses 500) |
| **Headline statistics** | Conditional correlations of Δx and Δn from the structural MA coefficients (paper eq. 24), sums truncated at h = 400 (paper fn. 16) |

Mirrors [entries/gali_1999_technology_var.yaml](../../../skills/model-dictionary/entries/gali_1999_technology_var.yaml).

## Specification notes

Documented deviations:

- **Hours series.** The paper uses "total employee-hours in nonagricultural
  establishments" (Citibase, p. 258), which includes government. The
  closest FRED-QD aggregate is `HOANBS` (nonfarm **business** hours), a
  narrower sector concept; there is no exact FRED-QD counterpart.
  Productivity inherits the mismatch in its denominator, since it is
  constructed as log(`GDPC1`) − log(`HOANBS`), mirroring the paper's own
  construction x = log GDP − log hours.
- **Sample.** Truncated to **1959Q1–1994Q4** (FRED-QD floor; paper starts
  1948:1). The Korean-war-era cycles are excluded. Same handling as the
  Blanchard-Quah replication.
- **Lag order.** The published article nowhere states the VAR lag order
  (it is deferred to the unpublished appendix / NBER WP 5721). We use
  **4 quarterly lags**, the convention of the subsequent replication
  literature, and do not treat 4 as paper-pinned (see the lag-order
  section below).
- **Caveat: detrended-variant non-technology correlation.** Table 1
  Panel B's non-technology correlation (+0.35) does **not** reproduce on
  this truncated proxy sample. In this run (4 lags) it is
  negative, and its sign depends on the lag order. The
  technology-conditional correlation, the claim the variant exists to
  support, is negative. `paper_targets.yaml` asserts only the technology
  leg.
- **Caveat: HP-filtered technology-component correlation.** The paper's
  −0.02 (p. 266) is a near-zero object. On the truncated sample it moves
  a lot with the lag order, so it is not asserted numerically. The
  falsifiable form of the Section-IV claim, that non-technology
  components dominate the cyclical comovement, *is* asserted (the
  non-technology correlation must exceed the technology one by more
  than 0.25).

## What the long-run restriction means

With productivity ordered first, the structural impact matrix `H`
satisfies `H H' = Σ_u` and `C(1) H` is lower triangular, where
`C(1) = (I − Σ A_j)^{-1}` is the long-run multiplier. The (1,2) entry of
`C(1) H` is zero by construction:

> The CUMULATIVE response of Δlog productivity to a non-technology shock
> is zero. Only technology shocks move the *level* of productivity
> permanently. Both shocks remain free to have permanent effects on hours
> and output, which is exactly what distinguishes Galí's scheme from
> applying Blanchard-Quah to output (p. 256).

The conditional correlations (paper eq. 24) are then computed from the
structural MA coefficients: corr(Δx, Δn | shock j) =
Σ_h C₁ⱼ(h)C₂ⱼ(h) / √(Σ_h C₁ⱼ(h)² · Σ_h C₂ⱼ(h)²), truncated at h = 400.

## Replicated results

**Results.** Run `python replicate.py` in this folder. It writes
`outputs/report.html` (interactive report), `outputs/numerical_results.json`
and `outputs/research_log.json`, which holds the conditional-correlation
and historical-decomposition blocks and the comparison with
`paper_targets.yaml`. Numbers depend on the FRED vintage you download and
on the seed in `config.yaml`.

The paper's values below are from the cited pages. The right column says
only what this run (1959Q1–1994Q4, 4 lags) shows qualitatively.

| Statistic | Paper (48:1–94:4) | This run |
|---|---:|---|
| corr(Δx, Δn \| technology) | **−0.82** (s.e. 0.12; Table 1, p. 259) | strongly negative, somewhat more so than the paper |
| corr(Δx, Δn \| non-technology) | **+0.26** (s.e. 0.12; Table 1) | positive, somewhat larger |
| corr(Δx, Δn) unconditional | **−0.26** (s.e. 0.08; Table 1) | mildly negative, close to the paper |
| Panel B (detrended hours), technology | **−0.81** (s.e. 0.11; Table 1) | negative, smaller in size |
| Productivity impact, 1-s.d. technology shock | ~ +0.6% (p. 258) | positive, a little smaller |
| Hours impact, technology shock | negative, persistent (abstract, Figure 2) | negative |
| HP (λ = 1600) GDP–hours correlation, technology components | −0.02 (p. 266) | positive (lag-sensitive, not asserted) |
| HP GDP–hours correlation, non-technology components | **+0.97** (p. 266) | strongly positive, close to the paper |

The paper's headline holds in this run: hours fall after a
positive technology shock, and the business-cycle comovement of GDP and
hours comes mainly from non-technology shocks.

## Lag-order sensitivity (the published paper does not state the lag order)

`replicate.py` runs one lag order, set by `lags` in `config.yaml`. To
check sensitivity, change `lags` and re-run. The HP technology-component
correlation and the detrended non-technology correlation depend on the
lag order, which is why neither is asserted.

## Library implementation

Uses [`lib.long_run_irf`](../../lib/identification.py) (same machinery as
the Blanchard-Quah replication, covered by unit tests),
[`lib.historical_decomposition`](../../lib/irf.py) (exact reconstruction;
the run records its reconstruction error in `research_log.json`), and
[`lib.hp_filter`](../../lib/filters.py).

## How to run

```bash
python scripts/replications/gali_1999_technology_var/replicate.py
```

## Citation

```bibtex
@article{gali1999,
  author = {Gal{\'i}, Jordi},
  title = {Technology, Employment, and the Business Cycle: Do Technology
           Shocks Explain Aggregate Fluctuations?},
  journal = {American Economic Review},
  volume = {89},
  number = {1},
  pages = {249--271},
  year = {1999}
}
```
