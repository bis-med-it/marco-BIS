# Blanchard-Quah (1989): Demand vs Supply Decomposition

Replicates Blanchard & Quah (1989), *AER* 79(4), 655–673, "The Dynamic
Effects of Aggregate Demand and Supply Disturbances". A bivariate VAR
identified by the long-run restriction that demand shocks have zero
permanent effect on the level of real GDP.

## Specification

| | |
|---|---|
| **Model class** | Reduced-form VAR with long-run identification |
| **Variables** | Δlog real GDP (×100, percent quarterly growth), unemployment rate (level) |
| **Order in VAR** | Δlog GDP first, unemployment second |
| **Lags** | 8 quarterly (paper p. 661) |
| **Sample** | paper: 1950Q2–1987Q4 (p. 661); replication: data 1959Q1–1987Q4 (FRED-QD starts in 1959Q1) |
| **Data treatment** | BQ "case (c)": raw series + VAR constant (see Specification notes) |
| **Identification** | Lower-triangular long-run impact: shock 1 (supply) has permanent effect on log GDP; shock 2 (demand) has zero long-run effect |
| **Sign normalisation** | Both shocks normalised to raise Δlog GDP on impact |
| **Bands** | Asymptotic Monte Carlo on the VAR coefficients, 2 000 draws |

Mirrors [entries/blanchard_quah_1989.yaml](../../../skills/model-dictionary/entries/blanchard_quah_1989.yaml).

## Specification notes

The replication keeps the paper's two variables, **8 lags** (p. 661),
quarterly frequency and the long-run restriction that **demand shocks have
no permanent effect on the level of output** (§I, p. 657: Σⱼ a₁₁(j)=0;
supply ordered first, lower-triangular long run). It deviates as follows:

- **Unemployment series.** BQ use the SA unemployment rate for **males age
  20+** (a homogeneous labor group, fn 5). The replication uses total
  `UNRATE` as a proxy; male-20+ is not in FRED-QD.
- **Output / sample.** Real GDP (`GDPC1`) for the paper's real GNP; sample
  truncated to **1959Q1–1987Q4** (paper is 1950Q2–1987Q4; FRED-QD starts 1959).
- **Data treatment: why this uses "case (c)".** BQ's headline Figs 1–2 use
  the **base case**: output growth demeaned with a **1974Q1 break** plus
  linearly **detrended unemployment** (p. 661). Case (c), raw series plus
  a constant, is one of the alternative treatments BQ report as giving
  results qualitatively similar to the base case (p. 662). It
  was adopted here after the base case was run on this truncated proxy
  sample and gave the opposite result: the break-demeaning of output growth
  turns the supply shock's permanent output effect negative (detrending
  unemployment alone does not). The choice was therefore made with the
  result in view. Under case (c) the permanent supply effect is positive,
  as in BQ. `replicate.py` does not run the base case.
- **Caveat: supply→unemployment.** BQ (p. 663) report that a positive supply
  shock *slightly increases* unemployment initially before it falls. This
  replication shows supply→unemployment negative throughout (no initial bump).
  The feature is delicate and sensitive to the male-20+/UNRATE and sample differences.
  The `paper_targets` do not assert the supply→U initial sign.

## What the long-run restriction means

The structural impact matrix `H` satisfies `H H' = Σ_u` (orthogonality) and
`C(1) H` is lower triangular, where `C(1) = (I − Σ A_j)^{-1}` is the
long-run multiplier. The (1,2) entry of `C(1) H` is zero by construction,
which means:

> The CUMULATIVE response of Δlog GDP to a demand shock is zero in the
> long run; equivalently, demand shocks have no permanent effect on
> the level of log GDP.

Algorithmically this is implemented by Cholesky-decomposing the long-run
reduced-form covariance `C(1) Σ_u C(1)' = L L'` and setting `H = C(1)^{-1} L`.

## Results

In this run the cumulative median response of Δlog GDP to the supply
shock (summed over horizons 0 to 40) is clearly positive, a permanent
positive effect on the level of output. The cumulative response to the
demand shock is close to zero, as the identifying restriction requires.
This matches BQ's narrative, with one caveat (see Specification notes):

- **Supply** shocks have permanent positive effects on output and lower
  unemployment persistently; BQ's small initial rise in unemployment is
  not reproduced (see Specification notes).
- **Demand** shocks raise output growth on impact and the effect fades
  to zero in the long run. Unemployment falls, with the largest decline
  within the first year, and then returns to zero.

**Median vs point estimate.** The reported cumulative response is the sum
over horizons 0-40 of the pointwise median IRFs. It is not the median of
draw-level cumulative responses, and it can differ from the deterministic
point-estimate cumulative. Say which object you are quoting.

The demand-shock cumulative isn't exactly zero because we sum only over
horizons 0..40 and because the median is taken horizon by horizon across
draws. The exact long-run restriction is at h → ∞; the test
suite confirms convergence to zero at h=200 with synthetic data (see
[`tests/test_identification.py::TestLongRun::test_long_run_restriction_holds`](../../../tests/test_identification.py)).

**Results.** Run `python replicate.py` in this folder. It writes
`outputs/report.html` (interactive report), `outputs/numerical_results.json`
(IRF medians and bands) and `outputs/research_log.json`, which includes
the comparison with `paper_targets.yaml`. Numbers depend on the FRED
vintage you download and on the seed in `config.yaml`.

## Library implementation

Uses [`lib.long_run_irf`](../../lib/identification.py); unit tests are in
[`tests/test_identification.py`](../../../tests/test_identification.py).

## How to run

```bash
python scripts/replications/blanchard_quah_1989/replicate.py
```

## Citation

```bibtex
@article{blanchard1989,
  author = {Blanchard, Olivier J. and Quah, Danny},
  title = {The Dynamic Effects of Aggregate Demand and Supply Disturbances},
  journal = {American Economic Review},
  volume = {79},
  number = {4},
  pages = {655--673},
  year = {1989}
}
```
