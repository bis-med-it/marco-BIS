"""Reusable library for the MaRCo@BIS framework.

Top-level imports for the common surface:

>>> from lib import fit_var, cholesky_irf, sign_restricted_irf, IRFContainer, VARReport

Modules:

* :mod:`lib.var` — reduced-form VAR fitting and linear-algebra primitives.
* :mod:`lib.identification` — Cholesky and sign-restriction identification.
* :mod:`lib.inference` — Monte Carlo and bootstrap bands.
* :mod:`lib.irf` — standardised IRF / FEVD containers (the DEVELOPMENT.md
  interface contract).
* :mod:`lib.report` — HTML report generation.

Also implemented (see ``scripts/lib/README.md`` for the authoritative
inventory): ``bvar`` (Minnesota BVAR), ``lp`` (Jordà LP + LP-IV + BLP),
``factors`` (principal-components FAVAR), ``filters`` (HP / one-sided HP / Hamilton /
BN / KMW-BN / Baxter-King / Christiano-Fitzgerald), ``connectedness``
(Diebold-Yilmaz GVD), ``forecast`` (iterated + conditional Bańbura-Giannone-
Lenza 2015), ``mfvar`` (state-space MF-VAR), ``tvp``, ``dfm``, ``panel``,
``nongaussian_svar``, ``dating``, and the ``identification`` proxy / long-run /
sign / zero-sign / heteroskedasticity routines.
"""

from .irf import (
    IRFContainer,
    FEVDContainer,
    HistoricalDecompositionContainer,
    historical_decomposition,
    structural_fevd,
    structural_shocks,
)
from .var import (
    VARFit,
    fit_var,
    select_lag_order,
    build_design_matrices,
    extract_lag_matrices,
    companion_matrix,
    max_eigenvalue_modulus,
    ma_representation,
)
from .bvar import (
    BVARFit,
    MinnesotaPrior,
    fit_minnesota_bvar,
    build_minnesota_dummies,
    log_marginal_data_density,
    optimize_minnesota_hyperparameters,
    SteadyStatePrior,
    SteadyStateBVARFit,
    fit_steady_state_bvar,
)
from .lp import lp_irf, lp_iv_irf, build_lp_controls, state_dependent_lp_irf, blp_irf
from .filters import (
    FilterResult,
    hp_filter,
    one_sided_hp_filter,
    hamilton_filter,
    beveridge_nelson,
    kmw_bn_filter,
    bk_filter,
    cf_filter,
)
from .connectedness import (
    ConnectednessResult,
    connectedness,
    generalised_fevd,
    rolling_connectedness,
)
from .factors import FAVARFit, fit_favar, favar_irf
from .state_space import (
    StateSpaceModel,
    KalmanResults,
    kalman_filter_smooth,
    simulation_smoother,
    log_likelihood as ss_log_likelihood,
    mariano_murasawa_weights,
    simple_flow_weights,
)
from .gibbs import draw_niw_posterior, draw_state_path
from .stochastic_volatility import (
    draw_log_h_path,
    sample_mixture_indicators,
)
from .mfvar import MFVARSpec, MFVARFit, fit_mfvar_bayesian
from .tvp import (
    TVPVARSpec, TVPVARFit,
    fit_tvp_var, fit_tvp_var_sv, fit_tvp_var_sv_primiceri,
)
from .dfm import DFMSpec, DFMFit, fit_dfm_bayesian
from .panel import (
    PanelVARFit,
    fit_pooled_var,
    fit_mean_group,
    fit_partial_pooling_var,
    fit_global_var,
    HierarchicalPanelPrior,
    HierarchicalPanelVARFit,
    fit_hierarchical_panel_var,
)
from .dating import DatingResult, bry_boschan_dating, harding_pagan_dating
from .identification import (
    cholesky_irf,
    fry_pagan_median_target,
    sign_restricted_irf,
    zero_sign_restricted_irf,
    cholesky_impact,
    proxy_svar_irf,
    long_run_irf,
    heteroskedasticity_irf,
)
from .nongaussian_svar import (
    nongaussian_svar_irf,
    fastica_symmetric,
    whiten_residuals,
    pml_t_estimate,
)
from .inference import (
    asymptotic_mc_draws,
    residual_bootstrap_draws,
    draw_parameter_samples,
)
from .report import VARReport, draft_narrative
from .data import load_dataset, transform, fetch_data
from .forecast import (
    ForecastContainer,
    ConditionalForecastContainer,
    iterated_forecast,
    conditional_forecast,
    rolling_forecasts,
    forecast_score,
    forecast_score_matched,
    density_scores,
    diebold_mariano,
)
from .specs import Spec, SpecResult, load_data, run
from .diagnostics import DiagnosticReport, diagnostics
from .sensitivity import SensitivityReport, sensitivity
from .validation import ComparisonReport, TargetEntry, compare_to_paper
from . import preprocess   # exposes deflate, nominalise, aggregate; see preprocess/__init__.py

__all__ = [
    "IRFContainer",
    "FEVDContainer",
    "HistoricalDecompositionContainer",
    "historical_decomposition",
    "structural_fevd",
    "structural_shocks",
    "VARFit",
    "fit_var",
    "select_lag_order",
    "build_design_matrices",
    "extract_lag_matrices",
    "companion_matrix",
    "max_eigenvalue_modulus",
    "ma_representation",
    "BVARFit",
    "MinnesotaPrior",
    "fit_minnesota_bvar",
    "SteadyStatePrior",
    "SteadyStateBVARFit",
    "fit_steady_state_bvar",
    "build_minnesota_dummies",
    "log_marginal_data_density",
    "optimize_minnesota_hyperparameters",
    "lp_irf",
    "lp_iv_irf",
    "build_lp_controls",
    "state_dependent_lp_irf",
    "blp_irf",
    "FilterResult",
    "hp_filter",
    "one_sided_hp_filter",
    "hamilton_filter",
    "beveridge_nelson",
    "kmw_bn_filter",
    "bk_filter",
    "cf_filter",
    "ConnectednessResult",
    "connectedness",
    "generalised_fevd",
    "rolling_connectedness",
    "FAVARFit",
    "fit_favar",
    "favar_irf",
    "StateSpaceModel",
    "KalmanResults",
    "kalman_filter_smooth",
    "simulation_smoother",
    "ss_log_likelihood",
    "mariano_murasawa_weights",
    "simple_flow_weights",
    "draw_niw_posterior",
    "draw_state_path",
    "MFVARSpec",
    "MFVARFit",
    "fit_mfvar_bayesian",
    "TVPVARSpec",
    "TVPVARFit",
    "fit_tvp_var",
    "fit_tvp_var_sv",
    "fit_tvp_var_sv_primiceri",
    "DFMSpec",
    "DFMFit",
    "fit_dfm_bayesian",
    "PanelVARFit",
    "fit_pooled_var",
    "fit_mean_group",
    "fit_partial_pooling_var",
    "fit_global_var",
    "HierarchicalPanelPrior",
    "HierarchicalPanelVARFit",
    "fit_hierarchical_panel_var",
    "DatingResult",
    "bry_boschan_dating",
    "harding_pagan_dating",
    "draw_log_h_path",
    "sample_mixture_indicators",
    "cholesky_irf",
    "fry_pagan_median_target",
    "sign_restricted_irf",
    "zero_sign_restricted_irf",
    "cholesky_impact",
    "proxy_svar_irf",
    "long_run_irf",
    "heteroskedasticity_irf",
    "nongaussian_svar_irf",
    "fastica_symmetric",
    "whiten_residuals",
    "pml_t_estimate",
    "asymptotic_mc_draws",
    "residual_bootstrap_draws",
    "draw_parameter_samples",
    "VARReport",
    "draft_narrative",
    "load_dataset",
    "transform",
    "fetch_data",
    "ForecastContainer",
    "ConditionalForecastContainer",
    "iterated_forecast",
    "conditional_forecast",
    "rolling_forecasts",
    "forecast_score",
    "forecast_score_matched",
    "density_scores",
    "diebold_mariano",
    "Spec",
    "SpecResult",
    "run",
    "load_data",
    "DiagnosticReport",
    "diagnostics",
    "SensitivityReport",
    "sensitivity",
    "ComparisonReport",
    "TargetEntry",
    "compare_to_paper",
]
