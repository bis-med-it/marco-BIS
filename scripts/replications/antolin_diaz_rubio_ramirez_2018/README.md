# Antolín-Díaz & Rubio-Ramírez (2018): Narrative Sign Restrictions

Canonical replication of ADRR (2018) "Narrative Sign Restrictions for
SVARs" (AER 108, 2802–2829). The paper introduces a tractable extension
to the standard sign-restriction identification scheme: augment the
QR-rotation acceptance loop with restrictions on the **structural shock
series at specific historical dates** ("on October 1979, the monetary
policy shock was positive"). ADRR show that narrative information can
sharpen identification without imposing the strong assumptions of
recursive or proxy-SVAR identification.

Uses `lib.sign_restricted_irf` with its `narrative_restrictions` argument:
a list of dated restrictions (sign of the shock, or largest contributor to
a variable) checked as a second rejection filter after the standard sign
check.

Mirrors [`entries/antolin_diaz_rubio_ramirez_2018.yaml`](../../../skills/model-dictionary/entries/antolin_diaz_rubio_ramirez_2018.yaml).

## Specification

| | |
|---|---|
| **Setup** | 4-variable US monthly monetary VAR(6) on FRED-MD 1965M1-2019M12 |
| **Variables** | industrial production, CPI, federal funds rate, 10-year yield |
| **Sign restrictions** | contractionary monetary policy shock: `ip ≤ 0`, `cpi ≤ 0`, `ffr ≥ 0` on impact |
| **Narrative restrictions** | Type 1 (sign of shock): (i) Volcker tightening (Oct 1979, shock > 0); (ii) late-Greenspan tightening (May 2000, shock > 0). Type 2 (historical contribution): (iii) in Oct 1979 the monetary policy shock was the largest positive contributor to the federal funds rate |
| **IRF horizon** | 24 months |
| **Sampler** | target 1000 accepted rotations, max 50,000 attempts; canonical single-candidate draw (first column of the Haar rotation); parameter uncertainty via asymptotic posterior |

Note: the paper's headline application is the global oil market
(Section IV) with restrictions tied to episodes like the Iranian
revolution (Dec 1978–Jan 1979), the Iran-Iraq war (Sep–Oct 1980), and
the Gulf War (Aug 1990). The paper **excludes** the Oct 1973 OAPEC
embargo as contested (p. 2814–2815). The paper also has a
monetary-policy application (Section V, pp. 2822–2826): Uhlig's
six-variable VAR with 12 lags, 1965M1–2007M11, output left unrestricted,
and narrative restrictions on October 1979. It reports its results
graphically (Fig. 6) and gives no band-width ratios. This replication is
a simpler variant and does not reproduce that exercise: four variables,
six lags, output restricted on impact, a sample to 2019 and an added May
2000 date. The identification logic is the same (but see "What this
replication does NOT cover" below re: the importance sampler).

The contribution restriction (iii) uses the library defaults: the
cumulative contribution since the start of the sample and the "most
important" criterion. The paper's October 1979 restriction is on the
unexpected change in that month with the "overwhelming" criterion
(`contribution: unexpected`, `criterion: overwhelming` in the
restriction).

## Results

In this run the narrative restrictions reject most sign-admissible
rotations, so the narrative-consistent subset is a small part of the
sign-restricted identified set. The run reaches its 50,000-attempt
ceiling well short of the 1,000-draw target, so the narrative-side bands
rest on far fewer draws and are noisier than the sign-only ones.

The clearest effect is on `spread` (the 10-year yield), which no sign
restriction constrains: its 68% band is much narrower on impact and at
short horizons, consistent with ADRR's point that narrative information
is most informative where sign restrictions are silent. The narrowing
fades by twelve months and reverses at 24, and the `ffr` and `cpi` bands
widen at most horizons, so this run does not show a general tightening
of the bands. The rejection sampler (below), the small number of
accepted draws and the different application are candidate reasons.

**Results.** Run `python replicate.py` in this folder. It writes
`outputs/report.html` (interactive report), `outputs/numerical_results.json`
and `outputs/research_log.json`, which includes the acceptance counts,
the band-width comparison by variable and horizon, and the comparison
with `paper_targets.yaml`. Numbers depend on the FRED vintage you
download and on the seed in `config.yaml`.

## Algorithm

For each rotation draw $Q$:
1. Standard sign check: take the first column of $Q$ (the canonical
   uniform draw of the impulse vector) and check its impact-period IRFs
   against the sign pattern, allowing a sign flip.
2. **Narrative check**: compute the structural shock series
   $\\varepsilon_t = u_t' (P^{-T} Q_{:,c})$ for the accepted column $c$,
   where $P = \\text{chol}(\\Sigma_u)$ and $u_t$ is the reduced-form
   residual. Reject the rotation if any date in
   `narrative_restrictions` fails its sign check.
3. Both checks pass → accept the rotation.

## Library implementation

| Step | Function |
|---|---|
| Data load | `lib.Spec.from_replication` + `lib.load_data` |
| VAR estimation | `lib.fit_var` |
| **Narrative sign restrictions** | `lib.sign_restricted_irf(..., narrative_restrictions=[...])` |
| Parameter uncertainty | Asymptotic posterior draws via `fit.parameter_draws` |

Tests are in
[`tests/test_identification.py::TestNarrativeSignRestrictions`](../../../tests/test_identification.py).

## What this replication does NOT cover

**ADRR's importance sampler.** The paper's algorithm (§III, p. 2811–2813)
*reweights and resamples* accepted draws, because narrative restrictions
truncate the **likelihood** (not the prior, unlike traditional sign
restrictions); the paper is explicit that simply discarding violating draws is
incorrect (p. 2812). The library's `sign_restricted_irf` instead implements
the simpler **rejection** form (discard rotations that violate the narrative
restrictions). This is the main methodological simplification vs the paper.
On this example the narrative restrictions narrow the `spread` bands at
short horizons but widen the `ffr` and `cpi` bands at most horizons (see
Results above); the posterior is not the exact ADRR posterior.

## How to run

```bash
python scripts/replications/antolin_diaz_rubio_ramirez_2018/replicate.py
```

The rotation search (up to 1000 accepted rotations × asymptotic
parameter draws) is the dominant cost. Reduce
`identification.target_accepted` for faster iteration.

## Citation

```bibtex
@article{antolindiazrubioramirez2018,
  author  = {Antolín-Díaz, Juan and Rubio-Ramírez, Juan F.},
  title   = {Narrative Sign Restrictions for SVARs},
  journal = {American Economic Review},
  volume  = {108},
  pages   = {2802--2829},
  year    = {2018}
}
```
