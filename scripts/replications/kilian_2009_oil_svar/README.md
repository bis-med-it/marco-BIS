# Kilian (2009): Not All Oil Price Shocks Are Alike

**Paper:** Lutz Kilian (2009), "Not All Oil Price Shocks Are Alike: Disentangling
Demand and Supply Shocks in the Crude Oil Market," *American Economic Review*
99(3), 1053–1069.

**Dictionary entry:** [`skills/model-dictionary/entries/kilian_2009_oil_svar.yaml`](../../../skills/model-dictionary/entries/kilian_2009_oil_svar.yaml)

## What this does

A recursive (Cholesky) SVAR of the global crude oil market. It identifies three
structural shocks (oil **supply**, **aggregate demand**, oil-market-specific
**precautionary demand**) from a 3-variable monthly VAR(24) over 1973m2–2007m12.
`replicate.py` is a thin orchestration: `lib.fit_var` → `lib.cholesky_irf`. The
method is point-identified and deterministic.

```
python scripts/replications/kilian_2009_oil_svar/replicate.py
```

It fetches Kilian's `data.txt` on first run (cached under `data/`), estimates the
VAR, computes the IRFs, validates against `paper_targets.yaml`, and writes
`outputs/` (numerical_results.json, research_log.json, report.html).

## Verification

Point responses agree to approximately machine precision with the Fig. 3
values obtained from the author's official replication files (openICPSR
113299). All nine Fig. 3 responses are reproduced; the remaining
differences are of floating-point size. The dictionary entry was checked
field by field against the AER 99(3) paper, with page references.

## Results

An oil-specific (precautionary) demand shock raises the real oil price on
impact. The response peaks within a few months and is persistent. In
Kilian's Fig. 3 the impact response is about +5.9% and
the peak about +8.5%. The current run is close to both values.

**Results.** Run `python replicate.py` in this folder. It writes
`outputs/report.html` (interactive report), `outputs/numerical_results.json`
and `outputs/research_log.json`, which includes the comparison with
`paper_targets.yaml`. The data are Kilian's deposited `data.txt` and the
method is point-identified, so the numbers do not depend on a data
vintage or a seed.

## Data

`data.txt` = Kilian's deposited data (oil-production growth, the real-economic-
activity index, the real price of oil), obtained from the public **JCR-Econ (2025)
replication repo** [`github.com/richryan/jcre-oil`](https://github.com/richryan/jcre-oil)
(`kilianAER2009replication/data.txt`) and verified **byte-identical (SHA-256)** to
the `data.txt` in Kilian's official **openICPSR 113299** package.

## Provenance

The computation uses the framework's own `lib` (Python), not Kilian's MATLAB
code, with the same data and the same recipe.
