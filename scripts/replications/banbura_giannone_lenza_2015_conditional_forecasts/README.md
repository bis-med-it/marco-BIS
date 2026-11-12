# Bańbura-Giannone-Lenza (2015): Conditional Forecasts via Kalman Smoothing

Canonical replication of BGL (2015) "Conditional Forecasts and Scenario
Analysis with Vector Autoregressions for Large Cross-Sections" (IJF
31(3), 739–756). The paper's contribution is a clean algorithm for
computing conditional forecasts in large (B)VARs by casting the VAR in
companion-form state space and treating the conditioning paths as
"observations" in a Kalman smoother.

Implemented as a library function: `lib.conditional_forecast(fit,
horizon, conditioning, n_paths, parameter_uncertainty, seed)`. Works on
`VARFit` and its subclasses (`BVARFit`, `MFVARFit`, `TVPVARFit`).
`DFMFit` is a separate class, and conditional forecasting from a dynamic
factor model is not implemented.

Mirrors [`entries/banbura_giannone_lenza_2015_conditional_forecasts.yaml`](../../../skills/model-dictionary/entries/banbura_giannone_lenza_2015_conditional_forecasts.yaml).

## Specification

| | |
|---|---|
| **Setup** | CEE-style 7-variable quarterly BVAR (US, 1965Q1–2019Q4) |
| **Variables** | real GDP, GDP deflator, commodity prices, FFR, non-borrowed reserves, total reserves, real M1 |
| **Prior** | Minnesota, fixed at the paper's hyperprior modes (λ₁=0.2, μ=1, δ=1); the paper estimates them by MH, as in GLP. λ₂=0.5 has no effect in the conjugate prior used here. λ₃=1 and a diffuse λ₄ match the paper. |
| **Forecast horizon** | 8 quarters |
| **Density draws** | 1000 paths via Kalman simulation smoother |
| **Library function** | `lib.conditional_forecast` |

Note: the paper's empirical illustration uses a 26-variable Euro-area
quarterly panel; this replication uses a smaller US CEE-style spec.
The methodology is variable-agnostic: the contribution is the Kalman
algorithm, not the specific data.

## Scenarios demonstrated

1. **Unconditional**: standard h-step BVAR forecast.
2. **Higher-for-longer**: FFR pinned at 5.0% for 8 quarters (8 quarters
   of restrictive monetary policy).
3. **ZLB**: FFR pinned at 0.25% for 8 quarters (effective zero lower
   bound, accommodative monetary policy).

## Algorithm

For a VAR(p) cast in companion-form state space:

    state α_t = [y_t; y_{t-1}; ...; y_{t-p+1}]   (n_vars * p,)
    y_t = Z α_t                                  Z = [I_n | 0 | ... | 0]
    α_{t+1} = T α_t + c + R η_t                  T = companion matrix
                                                 R = [I_n; 0; ...]
                                                 Q = Σ_u

The Kalman smoother is run on an extended observation matrix:

* Pre-horizon: actual in-sample data
* Horizon: conditioning values where pinned, NaN where not

statsmodels' `KalmanSmoother` natively handles NaN by dropping the
corresponding rows of the measurement equation per time step. The
smoothed mean of the state's first `n_vars` block gives the conditional
mean forecast for `y_{T+1}, ..., y_{T+horizon}`. Conditioning values
are pinned exactly post-smoother to eliminate residual numerical noise.

For density forecasts, the Durbin-Koopman simulation smoother
(`lib.state_space.simulation_smoother`) draws joint state paths from
the conditional distribution, then we extract the forecast block.

## Headline results

In this run both scenarios hold the FFR exactly at the pinned path. The
shifts against the unconditional forecast are small in this CEE-style
spec. The high-FFR scenario shifts the deflator up and the low-FFR
scenario shifts it down, with the gap growing over the 8 quarters. This
is consistent with reverse causality in the reduced-form BVAR: without
structural identification, pinning FFR draws on all shocks, not just
monetary ones (see "Caveats" below). The real GDP shifts are small and
change sign across horizons.

**Results.** Run `python replicate.py` in this folder. It writes
`outputs/report.html` (interactive report) and `outputs/research_log.json`,
which holds the unconditional and scenario paths with 68% bands and the
comparison with `paper_targets.yaml`. Numbers depend on the FRED vintage
you download and on the seed in `config.yaml`.

## Library implementation

| Step | Function |
|---|---|
| Data load | `lib.Spec.from_replication` + `lib.load_data` |
| BVAR estimation | `lib.fit_minnesota_bvar` |
| **Conditional forecast** | `lib.conditional_forecast(bvar, horizon, conditioning, n_paths, seed)` |
| State-space cast | `lib.forecast._build_var_state_space` (internal companion-form constructor) |
| Kalman smoother | `lib.state_space.kalman_filter_smooth` |
| Density forecasts | `lib.state_space.simulation_smoother` |

Tests are in
[`tests/test_forecast.py::TestConditionalForecast`](../../../tests/test_forecast.py).

## Caveats: what BGL 2015 does NOT solve

The shipped replication does **pure** conditional forecasts: the
conditioning path is treated as a future observation that gets
explained by ALL shocks. This is appropriate for **forecasting**
("what does the model predict for GDP if FFR follows path X?") but
not for **structural counterfactuals** ("what if monetary policy had
been path X, all else equal?").

For structural scenarios (Section 3.4 of the paper), identification
has to be layered on top, and the paper's mechanism replaces Σ with
Σ̄ = γ_{s+1}γ′_{s+1} + ⋯ + γ_nγ′_n over the forecast window. The
non-selected shocks are SET TO ZERO and the retained shocks drive
everything (with fn. 17's compatibility caveat). Structural scenarios
are not implemented here.

Other limitations:

* **Exact-path conditioning only**: soft conditioning (truncated
  regions, Waggoner-Zha 1999) and entropic tilting (Robertson-
  Tallman-Whitelaw 2005) not implemented. (BGL 2015 itself is also
  exact-path only.)
* **No multi-shock decomposition**: the smoother gives a joint
  conditional distribution; it doesn't decompose which shocks drove
  the conditioning path.

## How to run

```bash
python scripts/replications/banbura_giannone_lenza_2015_conditional_forecasts/replicate.py
```

The 1000 density paths are the dominant cost; reduce
`forecast.n_paths` in `config.yaml` for faster iteration.

## Citation

```bibtex
@article{banburagiannonelenza2015,
  author  = {Bańbura, Marta and Giannone, Domenico and Lenza, Michele},
  title   = {Conditional Forecasts and Scenario Analysis with Vector
             Autoregressions for Large Cross-Sections},
  journal = {International Journal of Forecasting},
  volume  = {31},
  number  = {3},
  pages   = {739--756},
  year    = {2015}
}
```
