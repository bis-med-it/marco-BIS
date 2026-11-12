# CEE (1996), Monetary Policy Shocks: Evidence from the Flow of Funds

Replicates the Section II core of Christiano, Eichenbaum & Evans (1996),
*REStat* 78(1), 16–34: a six-variable quarterly VAR with the paper's TWO
Cholesky-identified policy-shock measures (the federal funds rate and
minus log nonborrowed reserves), plus the D_t variants (M1, unemployment)
and the no-PCOM price-puzzle demonstration.

## Specification

| | |
|---|---|
| **Model class** | OLS VAR(4), constant (eq. 2 p. 17; q = 4 p. 18), quarterly |
| **Benchmark variables** | log real GDP (Y), log GDP deflator (P), log commodity prices (PCOM), federal funds rate (FF), **minus log nonborrowed reserves (NBRD)**, log total reserves (TR); p. 18 |
| **Identification** | Cholesky. FF policy shock: (Y, P, PCOM, FF, NBRD, TR). NBRD policy shock: (Y, P, PCOM, NBRD, FF, TR). Both orderings verbatim p. 18; positive shocks contractionary under both measures (the minus-log convention) |
| **D_t variants** | M1 ordered last (Table 1); UNEMP ordered *just before the policy variable* (p. 18 exception for production indicators); no-PCOM system (Figure 4 price puzzle) |
| **Sample** | 1960Q1–1992Q4 (p. 18), 132 obs, T_eff = 128 |
| **Bands** | ±1 s.e. Monte Carlo, **500 draws** (fn. 8 p. 19: Doan 1990, ex. 10.1) |
| **Table convention** | "quarters 1–2" = IRF horizons 0–1; "quarters 3–4" = horizons 2–3 (note to Table 1, p. 20) |

Mirrors [entries/cee_1996_flow_of_funds.yaml](../../../skills/model-dictionary/entries/cee_1996_flow_of_funds.yaml).

## Scope: what is and isn't replicated

The paper has two halves. The **monetary/macro half (Section II, Tables
1–2, Figure 4)** is replicated here from the pre-collected FRED-QD panel (downloaded
once per `data/fred_qd/UPDATE.md`).
The **flow-of-funds half (Tables 4–7: BNET, BSHORT, household HNET,
sectoral FINET/GNET)** requires Flow of Funds Z.1 sectoral flows that are
not in FRED-QD; the dictionary entry carries the paper's full F.101/F.104
line-number recipe (appendix p. 34) for staging the data via
`data/staging/`. This half is out of scope.

## Specification notes

- **PCOM proxy.** The paper's "index of sensitive commodity prices" has
  no stated source (an appendix gap, flagged in the entry). Baseline uses
  PPIACO; the paper's own fn. 11 sanctions the **crude-materials PPI**
  (WPSID62), which is run as a disclosed robustness branch and matters
  (below).
- **Reserves vintage.** FRED-QD's `NONBORRES` omits extended credit (the
  paper's series is NBR *plus extended credit*, appendix item 2) and FRED
  reserves carry unadjusted breaks (1984 Continental Illinois; Dec-1990 /
  Apr-1992 reserve-requirement changes). See `data/fred_qd/UPDATE.md`
  "Known series caveats".
- **Deflator/GDP vintage.** Modern chain-weighted GDPC1/GDPCTPI vs the
  paper's 1987-dollar fixed-weight aggregates (December-1993 vintage).
- **Sample reading.** "1960:Q1–1992:Q4" treated as the data window
  (T_eff = 128); the text does not resolve window-vs-regression-sample
  (entry verified.notes item d).
- **Stability.** The largest companion root is at about one, the
  unit-root class expected for a six-variable log-levels system. The
  responses stay bounded over the 24-quarter horizon.

## Replicated results: what reproduces

**Results.** Run `python replicate.py` in this folder. It writes
`outputs/report.html` (interactive report), `outputs/numerical_results.json`
and `outputs/research_log.json`, which includes the comparison with
`paper_targets.yaml`. Numbers depend on the FRED vintage you download and
on the seed in `config.yaml`.

The paper's values below are from the cited tables. The right column
says only what this run shows qualitatively.

| Object | Paper | This run |
|---|---:|---|
| σ(FF shock), annual rate | 0.79 (p. 18) | somewhat larger |
| FF own response, q1–2 avg | +0.821 (Table 1) | positive, similar size |
| Real GDP, q3–4 avg | −0.437 (Table 2) | negative, somewhat smaller |
| Output decline delayed (q1–2 ≈ 0, then falls) | yes (Table 2) | yes |
| Deflator flat ~year with PCOM (no puzzle) | yes (p. 21, Figure 4) | yes |
| **Price puzzle without PCOM** | deflator rises 2+ years (p. 21) | yes |
| M1 falls, q1–2 | −0.166 (Table 1) | falls |
| Unemployment rises | +0.087 at q3–4 (Table 2) | rises later, at h4–8; the q3–4 average is about zero |
| Commodity prices fall, q3–4 (fn-11 crude-materials PCOM) | −0.144 (Table 2) | falls, by more than in the paper |
| FF rises after NBRD shock, q1–2 | +0.443 (Table 1) | rises, by much less |
| TR falls after NBRD shock (the one cross-measure difference) | −0.795 (Table 1) | falls, by much more |

The signature CEE finding holds in this run: no price puzzle with
commodity prices in the system, and a puzzle without them.

## NOT asserted (documented deviations)

These paper values are computed and recorded in `research_log.json` but
not asserted:

- **The reserves-vintage cluster**: the paper's corr(FF, NBRD shocks) of
  0.49 (p. 18), its NBRD response to an FF shock of +0.751 and its
  initial TR insulation of +0.014 (Table 1). This run gives a
  correlation near zero, a much smaller NBRD response, and no initial
  TR insulation. `NONBORRES` omits extended credit and FRED reserves are
  not break-adjusted; the largest replication NBRD shocks land on the
  institutional episodes (1991Q1, 1984Q3, 1992Q2/Q4). A pre-1984
  diagnostic subsample also gives a correlation near zero, so the
  divergence is broad, not concentrated in those episodes. Modern FRED
  reserves data do not reproduce the paper's reserves-market moments.
- **The PPIACO-baseline PCOM decline** (−0.144, Table 2): with PPIACO
  the commodity price index *rises*, while the paper's own fn-11
  crude-materials alternative falls. The fn-11 variant is asserted
  instead.
- **The FEVD point value** (29.744, Table 2): this run's share is
  well above it on the modern vintage. Only the qualitative claim
  (substantial share, at least 20%) is asserted.

## Library implementation

Uses [`lib.cholesky_irf`](../../lib/identification.py),
[`lib.structural_shocks`](../../lib/irf.py) (shock-series recovery),
[`lib.structural_fevd`](../../lib/irf.py), and
[`lib.preprocess.nominalise`](../../lib/preprocess/deflation.py) for the M1
reconstruction.

## How to run

```bash
python scripts/replications/cee_1996_flow_of_funds/replicate.py
```

The script runs six VARs with 500 band draws each. Outputs:

- `outputs/report.html`: interactive report (benchmark FF system)
- `outputs/numerical_results.json`: IRF snapshot
- `outputs/research_log.json`: full provenance, including the
  reserves-vintage diagnostics and the fn-11 robustness block

## Citation

```bibtex
@article{cee1996,
  author = {Christiano, Lawrence J. and Eichenbaum, Martin and Evans, Charles},
  title = {The Effects of Monetary Policy Shocks: Evidence from the Flow
           of Funds},
  journal = {Review of Economics and Statistics},
  volume = {78},
  number = {1},
  pages = {16--34},
  year = {1996}
}
```
