# Replications gallery

One folder per paper. Each replication is a self-contained, browseable folder
intended as a public-good reference: download or click into one folder, read
the spec, and run `replicate.py` to regenerate the outputs (a minute or less
for most replications); full provenance lands in `outputs/research_log.json`
after a run.

## Conventions

```
replications/
└── <dictionary_id>/                  ← matches skills/model-dictionary/entries/<id>.yaml
    ├── README.md                     ← paper citation, summary, expected results
    ├── config.yaml                   ← all spec choices (mirrors dictionary entry)
    ├── paper_targets.yaml            ← (recommended) declared checks for validation.
    │                                     NB: these mix published benchmarks with
    │                                     structural identities and replication-derived
    │                                     magnitudes; each file labels which is which,
    │                                     so "N/N" means "all declared checks pass",
    │                                     not "N published numbers reproduced"
    ├── replicate.py                  ← thin script: load config -> call lib -> save outputs
    └── outputs/
        ├── report.html               ← interactive Plotly report
        ├── numerical_results.json    ← IRF snapshot, for IRF methods (others write their own CSVs)
        ├── research_log.json         ← full provenance + paper-target comparison
        └── *.pdf / *.tex (optional)  ← figures and tables for print
```

The folder name matches the dictionary `id` exactly. Search the dictionary
to find a replication, search this folder to find a paper.

The `replicate.py` is intentionally thin (a few hundred lines at most). Heavy lifting lives in
[`scripts/lib/`](../lib/). When the library improves, every replication
benefits without touching the per-paper script.

## Paper-target validation (recommended for new replications)

Each replication should declare the specific numbers the paper reports in
`paper_targets.yaml`, then call `lib.compare_to_paper` at the end of
`replicate.py`. The pattern catches paper-vs-our-result divergences **at
build time**, the analogue at the replication layer of what
[`tests/test_conventions.py`](../../tests/test_conventions.py) does at
the library layer.

Example `paper_targets.yaml`:

```yaml
targets:
  - description: "Core CPI pass-through (peak)"
    location: "Section 2.2, US column"
    paper_value: 0.21
    paper_units: "percent, 10% energy commodity shock"
    our_key: "core_cpi_peak_per_10pct_shock"
    tolerance: 0.10
    tolerance_kind: "absolute"             # or "relative"

  - description: "Energy FEV at 2y horizon"
    location: "Footnote 5"
    paper_value: 0.19
    paper_units: "share (0-1)"
    our_key: "energy_fev_2y_share"
    tolerance: 0.50
    tolerance_kind: "relative"
```

In `replicate.py`:

```python
import lib

our_results = {
    "core_cpi_peak_per_10pct_shock": float(...),
    "energy_fev_2y_share": float(...),
    # ...
}
report = lib.compare_to_paper(our_results, HERE / "paper_targets.yaml")
print(report.table())
# Persist for provenance
log["paper_target_comparison"] = report.to_dict()
```

The act of writing the YAML forces a careful read of the paper before
declaring replication success. The pass/fail column makes ambiguous
deviations explicit instead of letting them get buried in a
"substitutions" table.

## Index

**Fidelity.** *Paper specification* follows the paper's main model design,
with the substitutions noted. *Paper system, modified method* keeps the
paper's variable roles but changes the model, estimation, identification or
inference. *Method demo* illustrates selected ideas of the paper or uses its
shock series, on a different system or data; it may also adapt the method.
The table names the main departures. Full differences and results are in
each folder's README.

**Data.** FRED-MD and FRED-QD are downloaded once by following
[`data/fred_md/UPDATE.md`](../../data/fred_md/UPDATE.md) and
[`data/fred_qd/UPDATE.md`](../../data/fred_qd/UPDATE.md). Sources marked
"fetched" download on the first run. Files marked "required" must be staged
by you; the script exits with sourcing instructions if they are missing.

| Dictionary entry | Folder | Method | Fidelity to the paper | Data |
|---|---|---|---|---|
| [sims_1980_var](../../skills/model-dictionary/entries/sims_1980_var.yaml) | [`sims_1980_var/`](sims_1980_var/) | VAR + Cholesky | **Paper specification**: sample starts 1959; a PPI (WPSID61) stands in for import prices | FRED-QD |
| [cee_1999_monetary_svar](../../skills/model-dictionary/entries/cee_1999_monetary_svar.yaml) | [`cee_1999_monetary_svar/`](cee_1999_monetary_svar/) | OLS VAR + Cholesky | **Paper specification**: log PPIACO replaces the smoothed commodity-price change; NONBORRES omits extended credit | FRED-QD |
| [cee_1996_flow_of_funds](../../skills/model-dictionary/entries/cee_1996_flow_of_funds.yaml) | [`cee_1996_flow_of_funds/`](cee_1996_flow_of_funds/) | OLS VAR + Cholesky, dual policy-shock measures (FF and −log NBR) | **Paper specification**: Section II only (flow-of-funds half out of scope); PPIACO for commodities | FRED-QD |
| [uhlig_2005_sign_restrictions](../../skills/model-dictionary/entries/uhlig_2005_sign_restrictions.yaml) | [`uhlig_2005_sign_restrictions/`](uhlig_2005_sign_restrictions/) | Sign-restricted VAR (pure-sign approach) | **Paper system, modified method**: asymptotic parameter draws replace the paper's weak-prior posterior; INDPRO, CPI and WPSID62 proxies | FRED-MD |
| [kilian_2009_oil_svar](../../skills/model-dictionary/entries/kilian_2009_oil_svar.yaml) | [`kilian_2009_oil_svar/`](kilian_2009_oil_svar/) | Recursive oil-market SVAR (VAR(24) + Cholesky) | **Paper specification**: Kilian's own deposited data; no substitution | Kilian's deposited data (fetched) |
| [jorda_2005_lp](../../skills/model-dictionary/entries/jorda_2005_lp.yaml) | [`jorda_2005_lp/`](jorda_2005_lp/) | Local projections | **Method demo**: CEE-style 4-variable monetary system instead of Jordà's 3-variable output-gap system | FRED-QD |
| [litterman_1986_bvar](../../skills/model-dictionary/entries/litterman_1986_bvar.yaml) | [`litterman_1986_bvar/`](litterman_1986_bvar/) | BVAR + Minnesota | **Paper specification**: FRED-QD proxies; sample 1959-2019 instead of 1948-1979 | FRED-QD |
| [bgr_2010_large_bvar](../../skills/model-dictionary/entries/bgr_2010_large_bvar.yaml) | [`bgr_2010_large_bvar/`](bgr_2010_large_bvar/) | BGR-inspired shrinkage calibration | **Method demo**: 17-variable FRED-MD subset instead of 131 variables; log-average MSE calibration instead of the paper's relative-MSFE fit | FRED-MD |
| [romer_romer_2004_narrative_mp](../../skills/model-dictionary/entries/romer_romer_2004_narrative_mp.yaml) | [`romer_romer_2004_narrative_mp/`](romer_romer_2004_narrative_mp/) | Proxy-SVAR + narrative shock | **Method demo**: RR shock as a proxy-SVAR instrument in a 4-variable VAR, not the paper's designs | FRED-MD + staged RR shock file (required) |
| [auerbach_gorodnichenko_2012](../../skills/model-dictionary/entries/auerbach_gorodnichenko_2012.yaml) | [`auerbach_gorodnichenko_2012/`](auerbach_gorodnichenko_2012/) | State-dependent LP | **Paper system, modified method**: state-dependent local projection instead of the paper's smooth-transition VAR | FRED-QD |
| [stock_watson_2018_lpiv](../../skills/model-dictionary/entries/stock_watson_2018_lpiv.yaml) | [`stock_watson_2018_lpiv/`](stock_watson_2018_lpiv/) | Scaled-reduced-form LP-IV | **Method demo**: Jarociński-Karadi instrument and 1990-2024 sample instead of GK futures surprises | FRED-MD + JK shocks (fetched) |
| [gertler_karadi_2015_proxy_svar](../../skills/model-dictionary/entries/gertler_karadi_2015_proxy_svar.yaml) | [`gertler_karadi_2015_proxy_svar/`](gertler_karadi_2015_proxy_svar/) | Proxy-SVAR | **Method demo**: JK instrument, BAA-AAA spread for the EBP, 1990-2024 sample | FRED-MD + JK shocks (fetched) |
| [blanchard_quah_1989](../../skills/model-dictionary/entries/blanchard_quah_1989.yaml) | [`blanchard_quah_1989/`](blanchard_quah_1989/) | Long-run identification | **Paper specification**: case (c); total unemployment for the male 20+ rate; real GDP for GNP | FRED-QD |
| [gali_1999_technology_var](../../skills/model-dictionary/entries/gali_1999_technology_var.yaml) | [`gali_1999_technology_var/`](gali_1999_technology_var/) | Long-run identification (technology shocks) | **Paper specification**: nonfarm business hours for total employee-hours; sample starts 1959 | FRED-QD |
| [hamilton_2018_filter](../../skills/model-dictionary/entries/hamilton_2018_filter.yaml) | [`hamilton_2018_filter/`](hamilton_2018_filter/) | Trend-cycle filter | **Paper specification**: FRED-QD real GDP over 1959-2024 instead of 1947-2016 | FRED-QD |
| [kamber_morley_wong_2018_bnf](../../skills/model-dictionary/entries/kamber_morley_wong_2018_bnf.yaml) | [`kamber_morley_wong_2018_bnf/`](kamber_morley_wong_2018_bnf/) | KMW BN filter (delta-bar = psi(1)^2 restriction, closed-form constrained OLS) | **Paper system, modified method**: constrained OLS without the paper's shrinkage prior; delta-bar 0.25 (paper 0.24) | FRED-QD |
| [bbe_2005_favar](../../skills/model-dictionary/entries/bbe_2005_favar.yaml) | [`bbe_2005_favar/`](bbe_2005_favar/) | FAVAR (two-step PC) | **Method demo**: 19 FRED-MD series instead of the paper's 120-series panel | FRED-MD |
| [glp_2015_prior_selection](../../skills/model-dictionary/entries/glp_2015_prior_selection.yaml) | [`glp_2015_prior_selection/`](glp_2015_prior_selection/) | Empirical-Bayes BVAR hyperparameter selection | **Method demo**: CEE 1999 7-variable VAR instead of GLP's 3/7/22-variable systems | FRED-QD |
| [diebold_yilmaz_2012_connectedness](../../skills/model-dictionary/entries/diebold_yilmaz_2012_connectedness.yaml) | [`diebold_yilmaz_2012_connectedness/`](diebold_yilmaz_2012_connectedness/) | Generalised FEVD connectedness | **Method demo**: monthly return and yield-change volatility proxies 1973-2024 instead of daily range volatility 1999-2010 | FRED-MD |
| [rigobon_2003_heteroskedasticity_id](../../skills/model-dictionary/entries/rigobon_2003_heteroskedasticity_id.yaml) | [`rigobon_2003_heteroskedasticity_id/`](rigobon_2003_heteroskedasticity_id/) | Heteroskedasticity ID | **Method demo**: US IP, 10-year yield and stocks with NBER regimes, not EM bonds | FRED-MD |
| [schorfheide_song_2015_mfvar](../../skills/model-dictionary/entries/schorfheide_song_2015_mfvar.yaml) | [`schorfheide_song_2015_mfvar/`](schorfheide_song_2015_mfvar/) | Mixed-frequency BVAR | **Method demo**: 4 variables in growth rates instead of 11 in log levels | FRED-MD + FRED-QD |
| [primiceri_2005_tvpvar](../../skills/model-dictionary/entries/primiceri_2005_tvpvar.yaml) | [`primiceri_2005_tvpvar/`](primiceri_2005_tvpvar/) | Homoskedastic TVP-VAR | **Paper system, modified method**: paper's three variables and Q-prior calibration; constant covariance, so no time-varying contemporaneous coefficients or stochastic volatility | FRED-QD |
| [banbura_modugno_2014_bdfm](../../skills/model-dictionary/entries/banbura_modugno_2014_bdfm.yaml) | [`banbura_modugno_2014_bdfm/`](banbura_modugno_2014_bdfm/) | Bayesian DFM with ragged edge | **Method demo**: 15-series US FRED-MD panel and Gibbs, not the euro-area panel and EM | FRED-MD |
| [mertens_ravn_2013_tax_proxy_svar](../../skills/model-dictionary/entries/mertens_ravn_2013_tax_proxy_svar.yaml) | [`mertens_ravn_2013_tax_proxy_svar/`](mertens_ravn_2013_tax_proxy_svar/) | Multi-instrument proxy SVAR (unit-policy impact block) | **Paper system, modified method**: authors' archive data; identity policy-impact block instead of the paper's triangular tax-shock decomposition | Staged file from the authors' AER archive (required) |
| [antolin_diaz_rubio_ramirez_2018](../../skills/model-dictionary/entries/antolin_diaz_rubio_ramirez_2018.yaml) | [`antolin_diaz_rubio_ramirez_2018/`](antolin_diaz_rubio_ramirez_2018/) | Narrative sign restrictions (Type 1 + Type 2; rejection only) | **Method demo**: 4-variable monetary VAR, not the paper's applications; no ADRR importance weighting | FRED-MD |
| [canova_ciccarelli_2013_panel_var](../../skills/model-dictionary/entries/canova_ciccarelli_2013_panel_var.yaml) | [`canova_ciccarelli_2013_panel_var/`](canova_ciccarelli_2013_panel_var/) | Pooled / mean-group / stacked Minnesota VAR | **Method demo**: selected survey approaches on a G4 panel; the survey has no empirical benchmark | FRED (built by `prepare_data.py`) |
| [lanne_meitz_saikkonen_2017_nongaussian_svar](../../skills/model-dictionary/entries/lanne_meitz_saikkonen_2017_nongaussian_svar.yaml) | [`lanne_meitz_saikkonen_2017_nongaussian_svar/`](lanne_meitz_saikkonen_2017_nongaussian_svar/) | Non-Gaussian SVAR (FastICA + approximate LMS step-2 t-ML) | **Method demo**: 3-variable monthly US VAR instead of the 4-variable quarterly KCFCI system | FRED-MD |
| [andrade_ferroni_melosi_2023_higher_moments](../../skills/model-dictionary/entries/andrade_ferroni_melosi_2023_higher_moments.yaml) | [`andrade_ferroni_melosi_2023_higher_moments/`](andrade_ferroni_melosi_2023_higher_moments/) | Sign + sample-kurtosis restrictions | **Method demo**: 4-variable monetary VAR; sample moments and asymptotic draws replace the paper's robust moments and posterior | FRED-MD |
| [banbura_giannone_lenza_2015_conditional_forecasts](../../skills/model-dictionary/entries/banbura_giannone_lenza_2015_conditional_forecasts.yaml) | [`banbura_giannone_lenza_2015_conditional_forecasts/`](banbura_giannone_lenza_2015_conditional_forecasts/) | Conditional forecasts (Kalman, BGL 2015) | **Method demo**: US 7-variable CEE-style BVAR instead of a 26-variable euro-area panel | FRED-QD |
| [lenza_primiceri_2022_covid_var](../../skills/model-dictionary/entries/lenza_primiceri_2022_covid_var.yaml) | [`lenza_primiceri_2022_covid_var/`](lenza_primiceri_2022_covid_var/) | BVAR with fixed COVID scales (GLS) | **Method demo**: 4-variable quarterly BVAR instead of the paper's 7-variable monthly illustration | FRED-QD |

User-specific analyses live under [`scripts/applications/`](../applications/);
this folder holds only the canonical paper replications.

Every dictionary entry has a paired replication folder.
New papers enter via Mode B: draft the entry, then add the folder here
once the spec is verified. `gali_1999_technology_var/` (text PDF) and
`cee_1996_flow_of_funds/` (image-only scan) are the worked examples of
that pipeline.

## Adding a new replication

1. Copy `sims_1980_var/` as a template into `<your_id>/`.
2. Edit `config.yaml` to match the dictionary entry.
3. Edit `replicate.py` to load the right data and call the right library
   functions (`lib.fit_var`, `lib.cholesky_irf`, `lib.sign_restricted_irf`, …).
4. Run it; verify the outputs make economic sense.
5. Keep `README.md`, `config.yaml`, `paper_targets.yaml` and
   `replicate.py`. The `outputs/` folder is excluded from version
   control; others regenerate it by running `replicate.py`
   (a minute or less for most replications), with provenance recorded in
   `outputs/research_log.json` after the run.

## Reproducibility

`outputs/numerical_results.json` is a rounded local snapshot of the IRF
you can diff across runs. The automated check against the paper is
`paper_targets.yaml`: the replication smoke tests (`--runslow`) run every
replication and fail on any failing target.

Runs are tied to the archive release (the version in `CITATION.cff`) and
to the data vintage; see
[`scripts/lib/README.md`](../lib/README.md#versioning).
