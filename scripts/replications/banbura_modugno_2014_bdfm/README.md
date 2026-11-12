# Bańbura-Modugno (2014) DFM with missing data: Bayesian Gibbs demo

(The paper itself is frequentist ML via a modified EM algorithm; the
Gibbs sampler here is this repo's documented substitution.)

Demonstrates a Bayesian dynamic factor model with arbitrary missing-data
pattern handling, the central contribution of Bańbura & Modugno (2014),
*JAE* 29(1), 133-160.

## Scope of this replication

The model class is the paper's: a state-space DFM whose Kalman filter and
simulation smoother handle missing observations. This replication
estimates it by Bayesian Gibbs sampling (`lib.fit_dfm_bayesian`) instead
of EM, on 15 FRED-MD series with a simulated ragged edge, and compares
the Bayesian factors with principal-component factors. The table under
"Limitations vs the original paper" lists the other differences.

## Specification

| | |
|---|---|
| **Model class** | Bayesian DFM with AR(1) factor dynamics |
| **Information panel** | 15 FRED-MD variables (real activity, prices, labour, rates, money, asset prices) |
| **Factors** | 3 |
| **Standardisation** | z-score each column ignoring NaN |
| **Sample** | 1970-02 to 2019-12 (599 months) |
| **Ragged edge** | NaN last month of PAYEMS, INDPRO, UNRATE, CPIAUCSL (simulates real-time publication lags) |
| **Gibbs** | 200 iterations, 50 burn-in → 150 stored draws |

Mirrors [entries/banbura_modugno_2014_bdfm.yaml](../../../skills/model-dictionary/entries/banbura_modugno_2014_bdfm.yaml).

## Results

In this run the posterior-mean factor AR(1) matrix is stable but close
to a unit root, so the factors are highly persistent. Factors are
identified only up to rotation and sign; the first Bayesian factor
tracks the first principal-component factor closely. For the 4
variables with a missing last month (December 2019), the smoother fills
the gap from the factor representation.

**Results.** Run `python replicate.py` in this folder. It writes
`outputs/report.html` (summary report), `outputs/factor_path.csv`
(posterior-mean factor path), `outputs/loadings.csv` (posterior-mean
loadings matrix Λ) and `outputs/research_log.json`, which includes the
comparison with `paper_targets.yaml`. Numbers depend on the FRED vintage
you download and on the seed in `config.yaml`.

## Library implementation

Uses [`lib.fit_dfm_bayesian`](../../lib/dfm.py), which builds on:
- [`lib.state_space`](../../lib/state_space.py): Kalman simulation smoother
  with native NaN handling
- [`lib.gibbs`](../../lib/gibbs.py): NIW conjugate sampling

Unit tests are in [`tests/test_dfm.py`](../../../tests/test_dfm.py).

## Limitations vs the original paper

| Original BM 2014 | This replication |
|---|---|
| EM algorithm | Gibbs sampling |
| Euro-area panel (14/46/101 series, Table V) | US FRED-MD (15 series) |
| Pseudo real-time: single 15-Oct-2009 vintage + replicated publication delays (fn 16) | Final-vintage, simulated ragged edge |
| GDP nowcasting RMSE evaluation | Imputation demo only |
| Multiple frequency mixes | Monthly only |

For a paper-faithful BM 2014 reproduction, the authors' EM code is in
the Journal of Applied Econometrics Data Archive.

## How to run

```bash
python scripts/replications/banbura_modugno_2014_bdfm/replicate.py
```

The 200 Gibbs iterations of the 15-variable DFM simulation smoother are
the dominant cost.

## Citation

```bibtex
@article{banbura2014,
  author = {Bańbura, Marta and Modugno, Michele},
  title = {Maximum Likelihood Estimation of Factor Models on Datasets with Arbitrary Pattern of Missing Data},
  journal = {Journal of Applied Econometrics},
  volume = {29},
  number = {1},
  pages = {133--160},
  year = {2014}
}
```
