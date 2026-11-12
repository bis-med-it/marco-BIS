# CEE (1999): Monetary Policy VAR

Replicates Christiano, Eichenbaum & Evans (1999) "Monetary Policy Shocks:
What Have We Learned and to What End?", *Handbook of Macroeconomics* 1A,
65–148. Seven-variable OLS VAR(4) with Cholesky-identified monetary policy
shock, the canonical workhorse for recursive monetary VAR analysis.

## Specification

| | |
|---|---|
| **Model class** | OLS VAR (matches the paper's §3 estimator, p. 73) |
| **Variables** | Real GDP (log), GDP deflator (log), commodity prices (log), FFR (level), total reserves (log), non-borrowed reserves (log), nominal M1 (log) |
| **Cholesky ordering** | rgdp → deflator → commodity → FFR → totreserves → nbreserves → m1 |
| **Lags** | 4 quarterly |
| **Sample** | 1965Q3–1995Q2 (120 obs; VAR(4) → T_eff = 116) |
| **Bands** | Asymptotic Monte Carlo on the VAR coefficient distribution, 2 000 draws |

Mirrors [entries/cee_1999_monetary_svar.yaml](../../../skills/model-dictionary/entries/cee_1999_monetary_svar.yaml).

## Specification notes

The benchmark spec is **§4.2, pp. 83–84** (first benchmark: FFR as the
policy instrument). Two notes and the proxy substitutions:

- **Sample** **1965Q3–1995Q2** (p. 84: *"quarterly data
  over the sample period 1965:3–1995:2"*).
- **Reserves order**: **total reserves before non-borrowed
  reserves**, matching the paper's benchmark list `Y, P, PCOM, FF, TR,
  NBR, M` (p. 83). By CEE's own **Proposition 4.1(iii)** (p. 82) the
  monetary-policy-shock IRFs are *invariant to the ordering within the
  slow and fast blocks*, so the ordering is a faithfulness point that
  leaves the policy-shock responses unchanged.
- **Proxy substitutions** (FRED-available stand-ins for the paper's
  series): commodity prices = log `PPIACO` (paper uses the *smoothed
  change* in a sensitive-commodity index, a BEA leading-indicator
  component); non-borrowed reserves = `NONBORRES` (paper: nonborrowed
  reserves *plus extended credit*); deflator = `GDPCTPI`; nominal M1
  reconstructed as `M1REAL × CPIAUCSL / 100` (`lib.preprocess.nominalise` with the CPI, the index FRED deflates M1 by; exact at monthly frequency, an approximation with quarterly averages).

## Results: monetary contraction (FFR ↑)

**Results.** Run `python replicate.py` in this folder. It writes
`outputs/report.html` (interactive report), `outputs/numerical_results.json`
and `outputs/research_log.json`, which includes the comparison with
`paper_targets.yaml`. Numbers depend on the FRED vintage you download and
on the seed in `config.yaml`.

In this run the responses to a contractionary funds-rate shock
match CEE's qualitative picture:

- Real GDP is zero on impact by construction (ordered before the funds
  rate), then falls with a delay and recovers toward zero later in the
  horizon.
- The deflator is roughly flat at first and then falls. There is no
  lasting price puzzle in this run.
- Commodity prices are zero on impact by construction. They rise
  slightly for a short time and then fall.
- M1 falls within the first year.

## Estimator

CEE estimate the system by OLS (p. 73), and so does this replication. The
BVAR machinery is shown in the Litterman 1986, BGR 2010 and GLP 2015
replications.

## How to run

```bash
python scripts/replications/cee_1999_monetary_svar/replicate.py
```

## Citation

```bibtex
@incollection{christiano1999,
  author = {Christiano, Lawrence J. and Eichenbaum, Martin and Evans, Charles L.},
  title = {Monetary Policy Shocks: What Have We Learned and to What End?},
  booktitle = {Handbook of Macroeconomics},
  editor = {Taylor, John B. and Woodford, Michael},
  volume = {1A},
  pages = {65--148},
  publisher = {Elsevier},
  year = {1999}
}
```
