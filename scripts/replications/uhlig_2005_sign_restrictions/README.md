# Uhlig (2005): Sign-Restricted Monetary VAR

Replicates Uhlig (2005), *Journal of Monetary Economics* 52(2), 381–419,
"What are the effects of monetary policy on output? Results from an agnostic
identification procedure."

## Specification

| | |
|---|---|
| **Model class** | Reduced-form VAR with sign-restricted identification |
| **Variables** (paper-faithful, 6) | Industrial production (log), CPI (log), commodity prices (log), Federal funds rate (level), non-borrowed reserves (log), **Total Reserves** (log) |
| **Lags** | 12 monthly |
| **Sample** | 1965M1–2003M12 (FRED-MD) |
| **Sign restrictions** (monetary contraction, horizons 0–5) | FFR+, CPI−, commodity−, NB reserves−. **Output and Total Reserves unrestricted** |
| **Identification algorithm** | Pure-sign-restriction (the paper's main approach, Section 3.1, Fig. 6). QR rotation; accept rotations whose first column matches the sign restrictions; joint parameter + rotation sampling. |
| **Target accepted draws** | 1000 |
| **Parameter draws** | Reduced-form parameters are drawn from the framework's recentred asymptotic sampler (inverse-Wishart with df = T−k, centred on the OLS covariance), not from the paper's weak-prior Normal-Wishart posterior (Appendix B, p. 410: df = T). Before sign filtering, the covariance draws are centred about 17% higher than the paper's posterior mean. The exercise does not isolate how much of any difference from the paper this explains. |

Mirrors [entries/uhlig_2005_sign_restrictions.yaml](../../../skills/model-dictionary/entries/uhlig_2005_sign_restrictions.yaml).

## Variable substitutions (paper → replication)

The paper's literal data spec uses sources / interpolation that this
replication substitutes pragmatically. All three substitutions are
documented; their effect on the headline was not tested.

| Paper variable | Paper source | Replication uses | Rationale |
|---|---|---|---|
| Real GDP (monthly) | Bernanke-Mihov dataset, Moench-Uhlig (2004) interpolation using INDPRO | **raw INDPRO** | Interpolation method not re-implemented; INDPRO is the standard public proxy for monthly real activity |
| GDP deflator (monthly) | Bernanke-Mihov dataset, Moench-Uhlig (2004) interpolation using CPI + PPI | **raw CPIAUCSL** | Same reason; CPI is the standard monthly price index |
| Commodity price index | Dow Jones Spot Average (`_DJSD`, Global Financial Data; proprietary) | **WPSID62** (FRED PPI: crude materials) | Paper's series is paywalled; a raw-commodity PPI is the closest public proxy |
| Total reserves | TOTRESNS | **TOTRESNS** | ✓ Direct |
| Non-borrowed reserves | BOGNONBR | NONBORRES | ✓ Direct (FRED-MD naming convention) |
| Federal funds rate | FEDFUNDS | FEDFUNDS | ✓ Direct (in levels, not logs) |

## The headline contribution

Uhlig's key point: prior sign-restriction work imposed the negative
output response. Uhlig leaves it **unrestricted** and asks whether the
data alone signs it. Result: **no**. Under pure sign restrictions the
output response is not robustly negative: the 90% credible band
includes zero across most horizons in the first two years.

Paper Section 1 (p. 384) summary: *"Contractionary monetary policy
shocks have an ambiguous effect on real GDP. With 2/3 probability, a
typical shock will move real GDP by up to ±0.2 percent, consistent with
the conventional view, but also consistent with e.g. monetary
neutrality."*

## Replicated result

In this run (seed 42) the 90% band on the output response includes zero
at every month from 1 to 24, consistent with Uhlig's agnostic finding; the
median output response is positive. The proxy substitutions were not
tested for their effect on the result.

**Results.** Run `python replicate.py` in this folder. It writes
`outputs/report.html` (interactive report), `outputs/numerical_results.json`
and `outputs/research_log.json`, which includes the comparison with
`paper_targets.yaml` and the output-response checks under `key_findings`.
Numbers depend on the FRED-MD vintage you download and on the seed in
`config.yaml`.

## Methodology note: pure-sign vs. penalty function

The paper presents two approaches:

- **Pure-sign-restriction approach** (Section 3.1, Figs. 6-8): the
  paper's MAIN approach. Generates rotations from a uniform prior on
  the unit sphere; accepts those satisfying the sign restrictions.
- **Penalty-function approach** (Appendix B.2 / Section B.3, Fig. 14):
  an alternative producing sharper bands at the cost of additional
  implicit restrictions (linear penalty for sign violations + small
  reward for going further in the correct direction).

Paper p. 391: *"The pure-sign-restriction approach is cleaner for my
task at hand... Therefore, I focus on it in the main body of the
paper."* This replication uses pure-sign.

## How to run

```bash
python scripts/replications/uhlig_2005_sign_restrictions/replicate.py
```

## Citation

```bibtex
@article{uhlig2005,
  author = {Uhlig, Harald},
  title = {What are the effects of monetary policy on output? Results from an agnostic identification procedure},
  journal = {Journal of Monetary Economics},
  volume = {52},
  number = {2},
  pages = {381--419},
  year = {2005}
}
```

## Fry-Pagan median target

The pointwise-median IRF mixes rotations and is not itself a model
(Fry and Pagan 2011). The run's `IRFContainer` carries a coherent
single-rotation summary at `metadata["median_target"]` (the accepted
draw closest to the pointwise median), via `lib.fry_pagan_median_target`.
