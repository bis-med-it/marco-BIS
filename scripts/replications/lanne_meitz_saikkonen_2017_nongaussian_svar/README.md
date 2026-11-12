# Lanne, Meitz, Saikkonen (2017): Non-Gaussian SVAR

Demonstrates the LMS (2017) identification result: when the STRUCTURAL
shocks are mutually independent and at most one of them is Gaussian,
the structural matrix `B` in `u_t = B ε_t` is point-identified up to
column permutation and scaling (sign, once shock variances are
normalised to one). No Cholesky, sign, zero, or external-instrument
restrictions are needed (Proposition 1).

The paper's own application is a 4-variable quarterly SVAR(2) on the
output gap, GDP-deflator inflation, the Kansas City Financial Conditions
Index and the FFR (1990 to 2008Q2, T = 74), which rejects recursive
identification. The KCFCI is not in FRED-MD, so this replication uses a
standard small US monetary VAR instead (industrial production, CPI,
federal funds rate; monthly 1960 to 2007, pre-ZLB). It is a
methodological demo, run side by side with Cholesky on the same data and
with explicit non-Gaussianity diagnostics.

## Specification

| | |
|---|---|
| **Model class** | Reduced-form VAR with non-Gaussian SVAR identification |
| **Variables** | log(INDPRO), log(CPIAUCSL), FEDFUNDS (level) |
| **Lags** | 12 monthly |
| **Sample** | 1960-01 to 2007-12 (576 obs, T_eff=564) |
| **Identification** | (1) Cholesky baseline, FFR ordered last; (2) LMS / FastICA with `tanh` non-linearity; (3) LMS / `pml_t` parametric MLE with t-innovations, warm-started from FastICA + 3 random restarts |
| **Bands** | Residual (IID) bootstrap, which keeps the residuals' non-Gaussianity; 200 draws (config; the paper's own application uses 1000) |

Mirrors [entries/lanne_meitz_saikkonen_2017_nongaussian_svar.yaml](../../../skills/model-dictionary/entries/lanne_meitz_saikkonen_2017_nongaussian_svar.yaml).

## Why this sample

* **Pre-ZLB end (2007-12)**: the federal funds rate is bounded below by
  zero from 2009-2015, which compresses residual variance and likely
  violates the i.i.d. structural shock assumption. Pre-ZLB the FFR
  residual is clearly non-Gaussian (heavy-tailed; see diagnostics below).
* **Start 1960-01**: FRED-MD's effective start for all three series.
  Includes the Volcker disinflation (1979-82) which gives the FFR
  residual its dramatic kurtosis.

## Results

### Non-Gaussianity

All three recovered shocks reject Gaussianity (Jarque-Bera) at any
conventional level. The monetary policy shock is by far the most
non-Gaussian, with heavy negative skew and very large excess kurtosis.
FastICA (symmetric form, `tanh` non-linearity) converges.

**Implication for LMS**: the non-Gaussianity condition LMS identification
needs is clearly met. The `weak_nongaussianity` flag is False.
Identification also requires the structural shocks to be mutually
independent, which is not tested here (see "What this exercise does not
test").

### pml_t convergence and degrees-of-freedom estimates

The pml_t estimator (≈ step 2 of LMS's §4.5 three-step procedure,
repo normalisation) jointly estimates the orthogonal rotation
`Q` (via Givens-angle parameterisation) and per-shock t-distribution
degrees-of-freedom `ν_j` via L-BFGS-B, warm-started from FastICA's `Q`
plus 3 random restarts.

* pml_t converges.
* The recovered degrees of freedom `ν̂` (smaller = heavier-tailed) are
  well below 30 (the framework's "near-Gaussian" threshold) for all
  three shocks. The price shock has the heaviest tails.

That supports the non-Gaussianity half of the LMS conditions; the
independence half is not tested.

### FastICA vs pml_t agreement

The two methods produce nearly identical structural impact matrices
(the research log reports their relative Frobenius distance). The
largest difference is in the (ffr, price) entry, which is small in
absolute terms under both methods.

This is the expected outcome: FastICA is distribution-free and recovers
B from the data's non-Gaussian structure; pml_t imposes a t-distribution
parametric form on the same structure. When residuals are well-described
by t-innovations (as they are here), the two methods coincide. pml_t
adds the per-shock ν estimates as a bonus: a quantitative measure of
tail-heaviness that FastICA doesn't provide.

### Recovered structural impact matrix B

Both B matrices are in `outputs/research_log.json` (`B_point`).
(Columns assigned to shocks by largest absolute contemporaneous impact;
signs pinned so that each named shock raises its own variable on impact.)

The block structure is nearly diagonal: each shock's largest
contemporaneous impact is on the variable it's labelled with. The
off-diagonals are small but generally non-zero, capturing the
contemporaneous spillovers that the Cholesky restriction forbids by
construction.

### LMS vs Cholesky: same monetary shock?

Under Cholesky the contemporaneous impact of the MP shock on output and
prices is zero by construction. LMS allows small non-zero
contemporaneous spillovers to output and prices. The Cholesky zeros are
an *imposed* identifying assumption (no contemporaneous response of
output / prices to monetary shocks); LMS's data-driven identification
finds those spillovers are indeed small but non-zero, consistent with
the recursive ordering being approximately right.

### Dynamic IRFs to the contractionary MP shock

The saved IRFs are the pml_t-identified responses (horizons in months).
The qualitative picture is the textbook contractionary MP shock:

* **FFR**: rises on impact, stays elevated for years, and decays toward
  zero by 48 months.
* **INDPRO**: positive on impact (spillover effect), then contracts and
  stays below zero.
* **CPI**: small **positive** response throughout, the **price puzzle**
  (Sims 1992). Adding a commodity-price proxy or replacing FEDFUNDS
  with a high-frequency monetary surprise typically resolves this; left
  uncorrected here to maintain methodological focus.

The script prints the Cholesky baseline IRFs to the console for
comparison. Any similarity between the two is descriptive, conditional
on the LMS assumptions; it is not a statistical test of the recursive
zero restrictions.

**Results.** Run `python replicate.py` in this folder. It writes
`outputs/report.html` (interactive report with pml_t IRFs and bootstrap
bands), `outputs/numerical_results.json` (IRF tensor) and
`outputs/research_log.json`, which holds both B matrices, the
non-Gaussianity diagnostics and the comparison with
`paper_targets.yaml`. Numbers depend on the FRED vintage you download
and on the seed in `config.yaml`.

## Interpretation and limitations

This replication is a **methodological demo**, not a paper-faithful
empirical replication (it differs from the paper's own
4-variable quarterly KCFCI application).
The estimators are not LMS's own: FastICA is a repo substitution
consistent with Prop. 1, and pml_t (parametric t-ML) approximates step 2
of LMS's §4.5 three-step procedure.

What this exercise **does not test**:

* **Mutual independence of structural shocks** (the stronger LMS
  assumption beyond uncorrelatedness). Pairwise distance-correlation
  tests on the recovered shocks would be the appropriate check; they are
  not implemented here.
* **Sensitivity to the choice of `tanh` vs `cube` vs `exp` non-linearity**
  (FastICA). Robust FastICA users typically re-run with multiple `g`
  and check the permutation/sign-equivalent agreement.
* **Parametric form in pml_t**: only the standardised t-distribution is
  supported, so sensitivity to the parametric assumption is not tested.

What LMS contributes here vs the Cholesky baseline: no ordering choice is
needed. The columns of B are recovered from the data, and the "monetary
policy" shock is the component with the largest contemporaneous impact on
FFR, not the shock orthogonal to all other variables on impact.

## Library implementation

Uses [`lib.nongaussian_svar_irf`](../../lib/nongaussian_svar.py),
validated by the unit tests in
[`tests/test_nongaussian_svar.py`](../../../tests/test_nongaussian_svar.py).
The tests cover both the FastICA and the pml_t paths, including recovery
of a known mixing matrix and known ν on synthetic data and the
`weak_nongaussianity` flag.

## Citation

```bibtex
@article{lanne_meitz_saikkonen_2017,
  author  = {Lanne, Markku and Meitz, Mika and Saikkonen, Pentti},
  title   = {Identification and Estimation of Non-Gaussian Structural Vector Autoregressions},
  journal = {Journal of Econometrics},
  volume  = {196},
  number  = {2},
  pages   = {288--304},
  year    = {2017}
}
```

## How to run

```bash
python scripts/replications/lanne_meitz_saikkonen_2017_nongaussian_svar/replicate.py
```
