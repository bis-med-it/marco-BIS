# Auerbach-Gorodnichenko (2012): State-Dependent Fiscal Multipliers

State-dependent fiscal-multiplier demonstration based on Auerbach &
Gorodnichenko (2012). **AG's own estimator is a smooth-transition VAR (STVAR)**
estimated by MCMC maximum likelihood (Chernozhukov-Hong 2003); the framework
has no STVAR primitive, so this replication **substitutes a Ramey-Zubairy
(2018)-style state-dependent local projection** with the same identification
(Cholesky G-shock, Blanchard-Perotti timing) and the same logistic
smooth-transition state. The *estimator* differs from the paper; the
*identification and state construction* follow AG.

> **Sample limitation.** On the 1959Q1-2008Q4 sample (FRED-QD starts in
> 1959) the expansion multiplier exceeds the recession multiplier, the
> reverse of AG's 1947-2008 finding. See "Divergence from AG's headline"
> below.
>
> To try AG's 1947 start, stage `GDPC1`, `GCEC1` and `FGRECPTx` from FRED
> from 1947Q1 as a registered dataset (`data/registry.yaml`) and set
> `data.source` to it. The estimator still differs from AG's STVAR, so
> this would not by itself explain the reversal.

Mirrors [`entries/auerbach_gorodnichenko_2012.yaml`](../../../skills/model-dictionary/entries/auerbach_gorodnichenko_2012.yaml).

## Specification

| | |
|---|---|
| **Data** | FRED-QD quarterly (`GCEC1`, `GDPC1`, `FGRECPTx`) |
| **Sample** | 1959Q1–2008Q4 (200 obs) |
| **Variables** | `g` (log govt consumption + investment), `gdp` (log real GDP), `tax` (log federal receipts) |
| **Lags** | 4 |
| **Identification** | Cholesky with `g` ordered first (Blanchard-Perotti timing) |
| **State variable** | 7-quarter moving average of `Δlog GDP`, normalised to z ∼ N(0,1) over the sample |
| **State function** | `F(z) = exp(-γ z) / (1 + exp(-γ z))` with `γ = 1.5`; complementary logistic, `F=1` in recession (low z), `F=0` in expansion (high z) |
| **LP horizon** | 20 quarters |
| **Bands** | Newey-West HAC, `lag = h + 1` |

## State-interacted regression

At each horizon `h`:

```
y_{t+h} = α_E + α_R · F_{t-1}
          + β_E · shock_t + β_diff · (F_{t-1} · shock_t)
          + Γ_E · Z_{t-1} + Γ_diff · (F_{t-1} · Z_{t-1}) + ε_{t+h}
```

where `Z_{t-1}` stacks 4 lags of (g, gdp, tax). The **expansion**
coefficient is `β_E` (the shock-only coefficient, active when F≈0).
The **recession** coefficient is `β_E + β_diff` (the sum, active when
F≈1). The interacted controls absorb regime-dependent lag dynamics so
the shock coefficients can be interpreted as the regime-specific direct
response.

The state-interacted regression uses `lib.state_dependent_lp_irf`.
The Cholesky shock extraction matches the helper used by
[`jorda_2005_lp/`](../jorda_2005_lp/).

## Cumulative dollar multiplier

```
M(h) = [Y/G] × cumulative dlog(Y)_h / cumulative dlog(G)_h
```

The Y/G factor (the sample mean of Y/G over 1959-2008, recorded in
`outputs/research_log.json`) converts log-unit
elasticities to dollar-per-dollar multipliers. Computed separately for
each regime using its regime-specific `β` path.

## Results (this implementation)

In this run the expansion multiplier is larger than the recession
multiplier at each reported horizon (4, 8 and 16 quarters), the reverse
of AG's ordering, and the recession multiplier turns negative by 16
quarters. The 68% band on the recession-regime GDP response includes
zero at every horizon, while the expansion-regime band lies above zero
at most horizons.

**Results.** Run `python replicate.py` in this folder. It writes
`outputs/report.html` (interactive report) and `outputs/research_log.json`,
which holds the regime-specific responses with 68% and 90% bands, the
multipliers and the comparison with `paper_targets.yaml`. Numbers depend
on the FRED vintage you download.

## Divergence from AG's headline

**AG 2012 report** (Table 1, p. 9): recession multiplier of 2.48 at its
maximum (2.24 cumulative), expansion multiplier of 0.57 at its maximum
(-0.33 cumulative), over the 1947Q1-2008Q4 sample.

**This implementation** finds the **opposite ordering** over
1959Q1-2008Q4: the expansion multiplier exceeds the recession
multiplier. The sign of regime-dependence reverses.

Checks run during validation, not part of `replicate.py`: F = 1 matches
the lowest-growth quarters; a steeper transition (γ = 3) gives the same
ordering at h=4; and a linear LP gives a multiplier between the two
regime values at h=4-16.

**Why the divergence?** Three candidate explanations, which this
replication cannot separate:

1. **Estimator**: this replication substitutes a Ramey-Zubairy-style
   state-dependent local projection for AG's smooth-transition VAR (see
   the box above), so the comparison changes the estimator as well as
   the data.
2. **Sample period**: AG use 1947Q1-2008Q4, while FRED-QD starts in
   1959Q1. The three early postwar recessions (1948-49, 1953-54,
   1957-58) are therefore absent from this sample.
3. **Tax variable**: FRED-QD's `FGRECPTx` is real federal receipts;
   AG's tax variable is constructed from BEA NIPA. The controls are
   close but not identical.

The data substitutions are forced by FRED-QD's availability and the
estimator substitution by the library's coverage. Identification and
state construction follow AG, but the divergence is a finding about this
combined setup, not a test of AG's result on its own terms.

## Library implementation

| Step | Function |
|---|---|
| Data load | [`lib.load_dataset`](../../lib/data/loader.py) + [`lib.transform`](../../lib/data/transforms.py) |
| Cholesky shock | [`lib.fit_var`](../../lib/var.py); inline `chol⁻¹ · residuals` |
| State-LP regression | [`lib.state_dependent_lp_irf`](../../lib/lp.py) (Newey-West HAC, `lag = h + 1`) |

Validated by the LP unit tests in
[`tests/test_lp.py`](../../../tests/test_lp.py) which cover the
Cholesky-shock-feeds-LP pattern under linear specifications.

## How to run

```bash
python scripts/replications/auerbach_gorodnichenko_2012/replicate.py
```

## Scope vs original paper

* The paper also reports a tax-multiplier variant (govt revenue
  shocks instead of spending shocks). We focus on the spending
  multiplier, the headline result.
* AG2012 uses additional robustness checks (alternative state
  variables such as the output gap and unemployment rate; alternative
  VAR ordering; sub-sample stability tests). Not reproduced here.

## Citation

```bibtex
@article{auerbach_gorodnichenko_2012,
  author  = {Auerbach, Alan J. and Gorodnichenko, Yuriy},
  title   = {Measuring the Output Responses to Fiscal Policy},
  journal = {American Economic Journal: Economic Policy},
  volume  = {4},
  number  = {2},
  pages   = {1--27},
  year    = {2012}
}
```
