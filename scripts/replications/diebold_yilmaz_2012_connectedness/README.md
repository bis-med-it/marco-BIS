# Diebold-Yilmaz (2012) Connectedness: monthly FRED-MD proxies

Demonstrates the Diebold-Yilmaz (2012) connectedness methodology on four
US asset-class volatility proxies from FRED-MD. The original paper uses
daily Parkinson (1980) high-low range volatility on stocks, bonds, FX, and
commodities (1999–2010); this replication uses **monthly absolute
log-returns** of analogous FRED-MD series (1973–2024) as the volatility
proxy. The methodology is paper-faithful; the data is a sensible
monthly substitute.

## Specification

| | |
|---|---|
| **Method** | Generalised (Pesaran-Shin) FEVD + DY 2012 connectedness measures |
| **Variables** | 4 asset-class volatility proxies |
| **VAR lags** | 4 monthly |
| **FEVD horizon** | 10 months (DY used 10 days) |
| **Rolling window** | 60 months (DY used 200 days) |
| **Sample** | 1973-01 to 2024-12 |
| **Identification** | None; connectedness is identification-free via the generalised FEVD |

Variable construction:

| Asset class | Series | Volatility proxy |
|---|---|---|
| Stocks | S&P 500 | abs(d log price) |
| Bonds | GS10 (10Y Treasury yield) | abs(d yield) |
| FX | TWEXAFEGSMTHx (broad dollar index) | abs(d log) |
| Commodity | OILPRICEx (crude oil) | abs(d log price) |

Mirrors [entries/diebold_yilmaz_2012_connectedness.yaml](../../../skills/model-dictionary/entries/diebold_yilmaz_2012_connectedness.yaml).

## Results

**Results.** Run `python replicate.py` in this folder. It writes
`outputs/report.html` (interactive report), `outputs/rolling_connectedness.csv`
(the rolling total, directional and net spillovers by month) and
`outputs/research_log.json`, which holds the full-sample connectedness
table and the comparison with `paper_targets.yaml`. Numbers depend on the
FRED-MD vintage you download.

In this run:

- **Full sample.** Stock volatility is the only net transmitter. Bond,
  FX and commodity volatilities are net receivers. In the paper stocks
  are also the largest net transmitter (Table 2, p. 61). The full-sample
  total index is below the paper's 12.6% (Table 2, p. 61).
- **Rolling 60-month windows.** The total index rises sharply in stress
  episodes. Its highest value falls in late 2008, which matches the
  paper's account of the surge after Lehman (pp. 62-64). The levels are
  not comparable with the paper's daily-volatility index because of the
  monthly proxy.

## Library implementation

Uses [`lib.connectedness`](../../lib/connectedness.py) and
[`lib.rolling_connectedness`](../../lib/connectedness.py); unit tests are
in [`tests/test_connectedness.py`](../../../tests/test_connectedness.py).

## Substitutions vs original DY 2012

The data substitution from daily Parkinson high-low volatility to monthly
absolute log-returns affects:

1. **Magnitude of spillovers**: daily volatility has more high-frequency
   information; monthly proxies are smoother
2. **Resolution of peaks**: DY's daily series captures intraday Lehman
   events; our monthly resolution lumps them into a late-2008 spike
3. **Sample period**: we extend through 2024 (DY ended in 2010)

For a paper-faithful replication with daily Parkinson high-low volatility, the
authors' replication data is on the [Diebold-Yilmaz connectedness
website](https://financialconnectedness.org/).

## How to run

```bash
python scripts/replications/diebold_yilmaz_2012_connectedness/replicate.py
```

The rolling-window estimation re-fits the VAR at every window. Outputs: `rolling_connectedness.csv`, `report.html`, and
`research_log.json` in `outputs/`.

## Citation

```bibtex
@article{diebold2012,
  author = {Diebold, Francis X. and Yilmaz, Kamil},
  title = {Better to Give than to Receive: Predictive Directional Measurement of Volatility Spillovers},
  journal = {International Journal of Forecasting},
  volume = {28},
  number = {1},
  pages = {57--66},
  year = {2012}
}
```
