"""Tests for the panel VAR estimators (Canova-Ciccarelli 2013 family)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import lib


def _make_homogeneous_panel(n_units: int = 3, T: int = 200, n_vars: int = 2,
                             seed: int = 0,
                             B_true: np.ndarray | None = None,
                             unit_means: dict[str, np.ndarray] | None = None):
    """Generate a panel where all units share the same VAR(1) dynamics."""
    rng = np.random.default_rng(seed)
    if B_true is None:
        B_true = np.array([[0.5, 0.2], [0.1, 0.4]])
    panel: dict[str, pd.DataFrame] = {}
    for i in range(n_units):
        unit = f"unit_{i}"
        mean = unit_means.get(unit, np.zeros(n_vars)) if unit_means else np.zeros(n_vars)
        Y = np.zeros((T, n_vars))
        for t in range(1, T):
            Y[t] = mean + B_true @ (Y[t - 1] - mean) + rng.standard_normal(n_vars) * 0.5
        panel[unit] = pd.DataFrame(
            Y, columns=[f"y{j+1}" for j in range(n_vars)],
            index=pd.date_range("1990-01-01", periods=T, freq="QS"),
        )
    return panel, B_true


def _make_heterogeneous_panel(n_units: int = 4, T: int = 200, n_vars: int = 2,
                                seed: int = 0):
    """Generate a panel where each unit has its own VAR(1) coefficients
    drawn from a common distribution.
    """
    rng = np.random.default_rng(seed)
    B_bar = np.array([[0.5, 0.2], [0.1, 0.4]])
    panel: dict[str, pd.DataFrame] = {}
    per_unit_B: dict[str, np.ndarray] = {}
    for i in range(n_units):
        unit = f"unit_{i}"
        # Each unit's B is a small perturbation of B_bar
        B_i = B_bar + rng.normal(0, 0.05, size=B_bar.shape)
        per_unit_B[unit] = B_i
        Y = np.zeros((T, n_vars))
        for t in range(1, T):
            Y[t] = B_i @ Y[t - 1] + rng.standard_normal(n_vars) * 0.5
        panel[unit] = pd.DataFrame(
            Y, columns=[f"y{j+1}" for j in range(n_vars)],
            index=pd.date_range("1990-01-01", periods=T, freq="QS"),
        )
    return panel, B_bar, per_unit_B


class TestPooledVAR:

    def test_basic_fit_returns_panelvarfit(self):
        panel, _ = _make_homogeneous_panel()
        fit = lib.fit_pooled_var(panel, lags=1, trend="c")
        assert isinstance(fit, lib.PanelVARFit)
        assert isinstance(fit, lib.VARFit)
        assert fit.method == "pooled"
        assert fit.n_units == 3
        assert set(fit.units) == {"unit_0", "unit_1", "unit_2"}

    def test_recovers_true_B_in_homogeneous_panel(self):
        """With homogeneous DGP, pooled VAR should recover B_true closely."""
        panel, B_true = _make_homogeneous_panel(n_units=5, T=300, seed=42)
        fit = lib.fit_pooled_var(panel, lags=1, trend="c")
        B_estimated = fit.lag_matrices()[0]
        # Per-element diff < 0.05 with N=5, T=300
        np.testing.assert_allclose(B_estimated, B_true, atol=0.05)

    def test_per_unit_intercepts(self):
        """Unit intercepts should differ when units have different means."""
        unit_means = {
            "unit_0": np.array([1.0, 2.0]),
            "unit_1": np.array([-0.5, 3.0]),
            "unit_2": np.array([0.0, 0.0]),
        }
        panel, B_true = _make_homogeneous_panel(n_units=3, T=500, unit_means=unit_means)
        fit = lib.fit_pooled_var(panel, lags=1, trend="c")
        # Per-unit intercepts should differ
        intercepts = np.stack([fit.unit_intercepts[u] for u in fit.units])
        assert intercepts.shape == (3, 2)
        # All unit intercepts should be roughly different
        ranges = intercepts.max(axis=0) - intercepts.min(axis=0)
        assert (ranges > 0.1).all(), f"Unit intercepts too similar: {intercepts}"

    def test_long_format_input(self):
        """Long-format DataFrame with unit column should work identically."""
        panel, _ = _make_homogeneous_panel()
        # Convert to long format
        long_rows = []
        for u, df in panel.items():
            long_df = df.copy()
            long_df["unit"] = u
            long_rows.append(long_df)
        long = pd.concat(long_rows)

        fit_dict = lib.fit_pooled_var(panel, lags=1, trend="c")
        fit_long = lib.fit_pooled_var(long, lags=1, trend="c", unit_col="unit")
        # Same shared lag block
        np.testing.assert_allclose(fit_dict.lag_matrices()[0], fit_long.lag_matrices()[0],
                                   atol=1e-10)

    def test_cholesky_irf_works_on_pooled_fit(self):
        """Identification routines should accept PanelVARFit directly."""
        panel, _ = _make_homogeneous_panel(n_units=3, T=300)
        fit = lib.fit_pooled_var(panel, lags=1, trend="c")
        irf = lib.cholesky_irf(fit, horizon=10, bands="none")
        assert irf.median.shape == (11, 2, 2)
        # Impact is lower triangular by Cholesky convention
        assert abs(irf.median[0, 0, 1]) < 1e-12

    def test_panel_with_different_T_per_unit(self):
        """Units with different sample lengths should be handled."""
        panel, _ = _make_homogeneous_panel()
        # Shorten unit_2 to T=50
        panel["unit_2"] = panel["unit_2"].iloc[:50]
        fit = lib.fit_pooled_var(panel, lags=1, trend="c")
        assert fit.fit_info["T_per_unit"]["unit_2"] == 50 - 1  # T - p
        assert fit.fit_info["T_per_unit"]["unit_0"] == 200 - 1

    def test_drops_units_with_too_few_obs(self):
        """Units with T <= p should be dropped with a warning."""
        panel, _ = _make_homogeneous_panel()
        panel["short_unit"] = panel["unit_0"].iloc[:3]   # T=3 with lags=4 → drop
        with pytest.warns(UserWarning, match="insufficient observations"):
            fit = lib.fit_pooled_var(panel, lags=4, trend="c")
        assert "short_unit" not in fit.units
        assert fit.n_units == 3

    def test_rejects_nan(self):
        panel, _ = _make_homogeneous_panel()
        panel["unit_0"].iloc[5, 0] = np.nan
        with pytest.raises(ValueError, match="NaN values"):
            lib.fit_pooled_var(panel, lags=1, trend="c")

    def test_rejects_mismatched_columns(self):
        panel, _ = _make_homogeneous_panel()
        panel["unit_0"] = panel["unit_0"].rename(columns={"y1": "other_name"})
        with pytest.raises(ValueError, match="has columns"):
            lib.fit_pooled_var(panel, lags=1, trend="c")


class TestMeanGroup:

    def test_basic_fit_returns_panelvarfit(self):
        panel, _ = _make_homogeneous_panel()
        fit = lib.fit_mean_group(panel, lags=1, trend="c")
        assert fit.method == "mean_group"
        assert fit.per_unit_B is not None
        assert set(fit.per_unit_B.keys()) == {"unit_0", "unit_1", "unit_2"}

    def test_averaged_coefficients(self):
        """B_bar should equal the average of per-unit B."""
        panel, B_bar, per_unit_B = _make_heterogeneous_panel(n_units=4, T=200)
        fit = lib.fit_mean_group(panel, lags=1, trend="c")
        # Average per-unit B (lag block only)
        per_unit_lag = np.stack([extract for u, extract in
                                  ((u, B[:2, :]) for u, B in fit.per_unit_B.items())])
        manual_mean = per_unit_lag.mean(axis=0)
        # fit.B_hat for mean-group has the SAME shape as VARFit's B_hat
        # (it averages the full per-unit B_hat, not just lag block)
        # Lag block average is computed via lag_matrices()
        lag_bar = fit.lag_matrices()[0]
        np.testing.assert_allclose(lag_bar.T, manual_mean, atol=1e-10)

    def test_unit_lag_matrices_accessor(self):
        panel, _, _ = _make_heterogeneous_panel(n_units=3)
        fit = lib.fit_mean_group(panel, lags=1, trend="c")
        ulm = fit.unit_lag_matrices()
        assert set(ulm.keys()) == {"unit_0", "unit_1", "unit_2"}
        for u in ulm:
            assert len(ulm[u]) == 1   # p=1
            assert ulm[u][0].shape == (2, 2)

    def test_pooled_lag_matrices_not_available(self):
        """Pooled fits don't have per_unit_B; accessor should raise cleanly."""
        panel, _ = _make_homogeneous_panel()
        fit = lib.fit_pooled_var(panel, lags=1, trend="c")
        with pytest.raises(ValueError, match="only defined for mean-group"):
            fit.unit_lag_matrices()

    def test_mean_group_recovers_B_bar(self):
        """Under heterogeneity, mean-group should converge to B_bar."""
        panel, B_bar, _ = _make_heterogeneous_panel(n_units=8, T=400)
        fit = lib.fit_mean_group(panel, lags=1, trend="c")
        B_estimated = fit.lag_matrices()[0]
        np.testing.assert_allclose(B_estimated, B_bar, atol=0.1)


class TestPartialPoolingPVAR:
    """Bayesian partial-pooling PVAR. gamma controls shrinkage toward
    the common mean.
      gamma = 0       : no shrinkage; reduces to mean-group per-unit OLS
      gamma = infinity: full shrinkage; every unit collapses to B_bar
      intermediate    : bias-variance trade-off
    """

    def test_gamma_zero_equals_mean_group(self):
        """gamma=0 should reproduce per-unit OLS (mean-group's per-unit Bs)."""
        panel, _, _ = _make_heterogeneous_panel(n_units=4, T=200)
        fit_mg = lib.fit_mean_group(panel, lags=1, trend="c")
        fit_pp = lib.fit_partial_pooling_var(panel, lags=1, trend="c", gamma=0.0)
        # Per-unit B matrices should match exactly
        for u in fit_mg.units:
            np.testing.assert_allclose(fit_mg.per_unit_B[u],
                                       fit_pp.per_unit_B[u], atol=1e-10)
        assert fit_pp.method == "partial_pooling"

    def test_gamma_infinity_collapses_to_pooled(self):
        """gamma extremely large should collapse all per-unit Bs to the pooled B_bar."""
        panel, _, _ = _make_heterogeneous_panel(n_units=4, T=200)
        fit_pool = lib.fit_pooled_var(panel, lags=1, trend="c")
        fit_pp = lib.fit_partial_pooling_var(panel, lags=1, trend="c", gamma=1e9)
        # Every per-unit B should equal the pooled B_bar to ~1e-6
        for u in fit_pp.units:
            np.testing.assert_allclose(fit_pp.per_unit_B[u], fit_pool.B_hat,
                                       atol=1e-6)

    def test_intermediate_gamma_reduces_dispersion(self):
        """As gamma rises, cross-unit dispersion of B_i should monotonically fall."""
        panel, _, _ = _make_heterogeneous_panel(n_units=4, T=200)
        gammas = [0.0, 1.0, 100.0, 1e9]
        dispersions = []
        for g in gammas:
            fit = lib.fit_partial_pooling_var(panel, lags=1, trend="c", gamma=g)
            B_stack = np.stack(list(fit.per_unit_B.values()))
            dispersion = (B_stack.max(axis=0) - B_stack.min(axis=0)).mean()
            dispersions.append(dispersion)
        # Strictly decreasing in gamma
        for i in range(1, len(dispersions)):
            assert dispersions[i] <= dispersions[i - 1] + 1e-10, (
                f"dispersion not monotone in gamma: {dispersions}"
            )
        # Endpoint check: gamma=1e9 dispersion < 1e-3
        assert dispersions[-1] < 1e-3

    def test_negative_gamma_raises(self):
        panel, _, _ = _make_heterogeneous_panel(n_units=3)
        with pytest.raises(ValueError, match="gamma must be >= 0"):
            lib.fit_partial_pooling_var(panel, lags=1, trend="c", gamma=-1.0)

    def test_unknown_common_mean_string_raises(self):
        panel, _, _ = _make_heterogeneous_panel(n_units=3)
        with pytest.raises(ValueError, match="common_mean string must be"):
            lib.fit_partial_pooling_var(panel, lags=1, trend="c",
                                         common_mean="not_a_real_option")

    def test_common_mean_zero_shrinks_toward_zero(self):
        """With common_mean='zero' and high gamma, Bs should be near zero."""
        panel, _, _ = _make_heterogeneous_panel(n_units=3, T=200)
        fit = lib.fit_partial_pooling_var(panel, lags=1, trend="c",
                                            gamma=1e9, common_mean="zero")
        for u in fit.units:
            assert np.abs(fit.per_unit_B[u]).max() < 1e-3

    def test_common_mean_array_accepted(self):
        """Explicit np.ndarray as common_mean is accepted."""
        panel, _, _ = _make_heterogeneous_panel(n_units=3, T=200)
        # Need to know the right shape — fit one unit
        from lib.var import fit_var as _fit
        tmp = _fit(panel["unit_0"], lags=1, trend="c")
        custom_mean = np.ones_like(tmp.B_hat) * 0.3
        fit = lib.fit_partial_pooling_var(panel, lags=1, trend="c",
                                            gamma=1e9, common_mean=custom_mean)
        for u in fit.units:
            np.testing.assert_allclose(fit.per_unit_B[u], custom_mean, atol=1e-6)

    def test_irf_works_on_partial_pooling_fit(self):
        """PanelVARFit inheritance: Cholesky IRF should accept the fit."""
        panel, _, _ = _make_heterogeneous_panel(n_units=3, T=200)
        fit = lib.fit_partial_pooling_var(panel, lags=1, trend="c", gamma=1.0)
        irf = lib.cholesky_irf(fit, horizon=4, bands="none")
        assert irf.median.shape == (5, 2, 2)


class TestPanelVARFitInheritance:
    """PanelVARFit must work as a drop-in VARFit for identification routines."""

    def test_lag_matrices_accessor(self):
        panel, B_true = _make_homogeneous_panel(n_units=3, T=200)
        fit = lib.fit_pooled_var(panel, lags=2, trend="c")
        lags = fit.lag_matrices()
        assert len(lags) == 2
        assert lags[0].shape == (2, 2)
        assert lags[1].shape == (2, 2)

    def test_max_eigenvalue_modulus(self):
        panel, _ = _make_homogeneous_panel()
        fit = lib.fit_pooled_var(panel, lags=1, trend="c")
        eig = lib.max_eigenvalue_modulus(fit.lag_matrices())
        assert 0 < eig < 1   # B_true is stationary

    def test_ma_representation(self):
        panel, _ = _make_homogeneous_panel()
        fit = lib.fit_pooled_var(panel, lags=1, trend="c")
        Phi = fit.ma_rep(5)
        assert Phi.shape == (6, 2, 2)
        np.testing.assert_allclose(Phi[0], np.eye(2))


# ---------------------------------------------------------------------------
# Global VAR (full cross-unit interdependence) tests
# ---------------------------------------------------------------------------


def _make_gvar_panel(
    n_units: int = 2,
    T: int = 500,
    K: int = 2,
    cross_unit_strength: float = 0.05,
    seed: int = 0,
):
    """Generate a panel with known cross-unit spillovers.

    DGP: each unit's own dynamics are a stationary VAR(1) with
    autocorrelation 0.6 (own) and 0.1 (cross-var, within-unit), plus
    a cross-unit spillover of magnitude ``cross_unit_strength`` from
    each unit's variables to every other unit.
    """
    rng = np.random.default_rng(seed)
    N = n_units * K
    A = np.zeros((N, N))
    for i_u in range(n_units):
        for j_u in range(n_units):
            block_rows = slice(i_u * K, (i_u + 1) * K)
            block_cols = slice(j_u * K, (j_u + 1) * K)
            if i_u == j_u:
                # Within-unit: 0.6 on diagonal, 0.1 on off-diagonal
                block = 0.6 * np.eye(K) + 0.1 * (np.ones((K, K)) - np.eye(K))
            else:
                # Cross-unit: identity-shaped at cross_unit_strength
                block = cross_unit_strength * np.eye(K)
            A[block_rows, block_cols] = block

    burn = 100
    Y = np.zeros((T + burn, N))
    for t in range(1, T + burn):
        Y[t] = A @ Y[t - 1] + rng.standard_normal(N) * 0.5
    Y = Y[burn:]

    var_names = [f"y{j+1}" for j in range(K)]
    idx = pd.period_range("1990-01", periods=T, freq="M").to_timestamp()
    panel = {}
    for i_u in range(n_units):
        unit = f"unit_{i_u}"
        panel[unit] = pd.DataFrame(
            Y[:, i_u * K : (i_u + 1) * K],
            columns=var_names, index=idx,
        )
    return panel, A


class TestGlobalVAR:

    def test_basic_fit_returns_panelvarfit(self):
        panel, _ = _make_gvar_panel(n_units=2, T=300, K=2)
        fit = lib.fit_global_var(
            panel, lags=1, lambda_1=0.5, lambda_cross_unit=0.5,
        )
        assert isinstance(fit, lib.PanelVARFit)
        assert fit.method == "global_var"
        assert fit.n_units == 2
        assert fit.n_vars == 4   # stacked = n_units * K
        assert fit.variable_names == ["unit_0_y1", "unit_0_y2",
                                       "unit_1_y1", "unit_1_y2"]

    def test_compatible_with_cholesky_irf(self):
        """fit_global_var produces a VARFit-compatible result that
        works with all existing identification routines."""
        panel, _ = _make_gvar_panel(n_units=2, T=300, K=2)
        fit = lib.fit_global_var(panel, lags=1, lambda_1=0.5)
        irf = lib.cholesky_irf(fit, horizon=5, bands="none")
        # Shape: (H+1, n_vars_stacked, n_shocks_stacked)
        assert irf.median.shape == (6, 4, 4)

    def test_recovers_block_diagonal_when_cross_unit_zero(self):
        """When lambda_cross_unit=0, all cross-unit blocks should be
        exactly 0 (the prior fully shrinks them to zero)."""
        panel, _ = _make_gvar_panel(n_units=3, T=300, K=2)
        fit = lib.fit_global_var(
            panel, lags=1, lambda_1=0.5, lambda_cross_unit=0.0,
        )
        for ui in fit.units:
            for uj in fit.units:
                if ui == uj:
                    continue
                block = fit.cross_unit_lag_block(ui, uj, lag=1)
                assert np.allclose(block, 0.0, atol=1e-10), (
                    f"cross-unit block ({ui}, {uj}) should be exactly 0 "
                    f"with lambda_cross_unit=0; got max |entry|="
                    f"{np.abs(block).max()}"
                )

    def test_recovers_dgp_on_long_sample(self):
        """With loose prior and T=2000, the global VAR should recover
        the DGP coefficients within ~10% on each block."""
        panel, A_true = _make_gvar_panel(
            n_units=2, T=2000, K=2, cross_unit_strength=0.08, seed=42,
        )
        fit = lib.fit_global_var(
            panel, lags=1, lambda_1=10.0, lambda_2=1.0,
            lambda_cross_unit=1.0, trend="n",
        )
        # The fit's B_hat (lag block) should be close to A_true^T (column convention)
        B_lag = fit.B_hat[: fit.n_vars * fit.p, :]   # (N, N)
        # extract_lag_matrices reshapes: lag_mats[0] is (N, N) such that
        # Y_t ~ lag_mats[0] @ Y_{t-1}, which matches our A_true.
        lag_mats = fit.lag_matrices()
        A_est = lag_mats[0]
        diff = float(np.linalg.norm(A_est - A_true, ord="fro"))
        rel_err = diff / float(np.linalg.norm(A_true, ord="fro"))
        assert rel_err < 0.1, (
            f"GVAR coefficient recovery error too large: rel_err={rel_err:.3f}"
            f"\nA_true:\n{A_true}\nA_est:\n{A_est}"
        )

    def test_cross_unit_lag_block_shape(self):
        panel, _ = _make_gvar_panel(n_units=2, T=300, K=3)
        fit = lib.fit_global_var(panel, lags=2, lambda_1=0.5)
        for ui in fit.units:
            for uj in fit.units:
                for lag in (1, 2):
                    block = fit.cross_unit_lag_block(ui, uj, lag=lag)
                    assert block.shape == (3, 3)

    def test_cross_unit_lag_block_validates_args(self):
        panel, _ = _make_gvar_panel(n_units=2, T=300, K=2)
        fit = lib.fit_global_var(panel, lags=1, lambda_1=0.5)
        with pytest.raises(ValueError, match="unknown unit_to"):
            fit.cross_unit_lag_block("unknown", "unit_0", lag=1)
        with pytest.raises(ValueError, match="unknown unit_from"):
            fit.cross_unit_lag_block("unit_0", "unknown", lag=1)
        with pytest.raises(ValueError, match="lag must be in"):
            fit.cross_unit_lag_block("unit_0", "unit_1", lag=0)
        with pytest.raises(ValueError, match="lag must be in"):
            fit.cross_unit_lag_block("unit_0", "unit_1", lag=2)

    def test_average_cross_unit_spillover_low_when_dgp_diagonal(self):
        """When DGP has zero cross-unit spillovers, the recovered
        spillover ratio should be small."""
        panel, _ = _make_gvar_panel(
            n_units=2, T=600, K=2, cross_unit_strength=0.0, seed=7,
        )
        fit = lib.fit_global_var(
            panel, lags=1, lambda_1=0.3, lambda_cross_unit=0.05, trend="n",
        )
        ratio = fit.average_cross_unit_spillover(lag=1)
        # With tight cross-unit prior, ratio should be << 1 (close to 0)
        assert ratio < 0.10, (
            f"Cross-unit spillover ratio {ratio:.3f} unexpectedly large "
            f"when DGP has no spillover"
        )

    def test_average_cross_unit_spillover_higher_when_dgp_has_spillovers(self):
        """When DGP has substantial cross-unit spillovers, the ratio
        should be higher than with zero-spillover DGP at the same prior."""
        panel_off, _ = _make_gvar_panel(
            n_units=2, T=600, K=2, cross_unit_strength=0.0, seed=10,
        )
        panel_on, _ = _make_gvar_panel(
            n_units=2, T=600, K=2, cross_unit_strength=0.15, seed=10,
        )
        prior_kwargs = dict(
            lags=1, lambda_1=0.5, lambda_cross_unit=0.5, trend="n",
        )
        fit_off = lib.fit_global_var(panel_off, **prior_kwargs)
        fit_on = lib.fit_global_var(panel_on, **prior_kwargs)
        ratio_off = fit_off.average_cross_unit_spillover(lag=1)
        ratio_on = fit_on.average_cross_unit_spillover(lag=1)
        assert ratio_on > ratio_off, (
            f"Spillover ratio should increase with DGP spillover strength; "
            f"got ratio_off={ratio_off:.3f}, ratio_on={ratio_on:.3f}"
        )

    def test_mismatched_indices_raise(self):
        panel, _ = _make_gvar_panel(n_units=2, T=200, K=2)
        # Shift the second unit's index — should raise
        units = list(panel.keys())
        panel[units[1]] = panel[units[1]].iloc[1:].copy()   # drop one row, shifts index
        with pytest.raises(ValueError, match="common index"):
            lib.fit_global_var(panel, lags=1, lambda_1=0.5)

    def test_negative_lambda_raises(self):
        panel, _ = _make_gvar_panel(n_units=2, T=200, K=2)
        with pytest.raises(ValueError, match="lambda"):
            lib.fit_global_var(panel, lags=1, lambda_1=-0.1)
        with pytest.raises(ValueError, match="lambda"):
            lib.fit_global_var(panel, lags=1, lambda_cross_unit=-0.1)

    def test_cross_unit_lag_block_only_for_global_var(self):
        """Calling cross_unit_lag_block on a pooled fit should raise."""
        panel, _ = _make_homogeneous_panel(n_units=2, T=200)
        fit = lib.fit_pooled_var(panel, lags=1, trend="c")
        with pytest.raises(ValueError, match="only defined for global-VAR"):
            fit.cross_unit_lag_block("unit_0", "unit_1", lag=1)

    def test_fit_info_records_hyperparameters(self):
        panel, _ = _make_gvar_panel(n_units=2, T=200, K=2)
        fit = lib.fit_global_var(
            panel, lags=1, lambda_1=0.3, lambda_2=0.4,
            lambda_cross_unit=0.05, lambda_3=2.0,
        )
        for key in ("lambda_1", "lambda_2", "lambda_cross_unit", "lambda_3",
                    "n_units", "n_vars_per_unit", "unit_var_layout"):
            assert key in fit.fit_info
        assert fit.fit_info["lambda_1"] == 0.3
        assert fit.fit_info["lambda_cross_unit"] == 0.05
        assert fit.fit_info["n_vars_per_unit"] == 2

    def test_tighter_cross_unit_prior_shrinks_cross_unit_blocks(self):
        """Same data, tighter lambda_cross_unit → smaller cross-unit Frobenius
        norms compared to looser prior."""
        panel, _ = _make_gvar_panel(
            n_units=2, T=400, K=2, cross_unit_strength=0.05, seed=1,
        )
        loose = lib.fit_global_var(
            panel, lags=1, lambda_1=0.5, lambda_cross_unit=1.0, trend="n",
        )
        tight = lib.fit_global_var(
            panel, lags=1, lambda_1=0.5, lambda_cross_unit=0.01, trend="n",
        )
        ratio_loose = loose.average_cross_unit_spillover(lag=1)
        ratio_tight = tight.average_cross_unit_spillover(lag=1)
        assert ratio_tight < ratio_loose, (
            f"Tighter prior should produce smaller cross-unit ratio; "
            f"loose={ratio_loose:.4f}, tight={ratio_tight:.4f}"
        )


class TestHierarchicalPanelVAR:
    """Hierarchical random-coefficient Bayesian panel VAR (Canova-Ciccarelli),
    which completes fit_partial_pooling_var with a learned exchangeable
    prior (common mean delta_bar and per-coefficient pooling tau^2)."""

    @staticmethod
    def _lag_dev(B, B_true):
        # B is (k, n) with rows [lag1 vars, const]; A_1 = B[:n].T
        n = B_true.shape[0]
        return float(np.abs(B[:n, :].T - B_true).mean())

    def test_recovers_common_dynamics(self):
        panel, B_true = _make_homogeneous_panel(n_units=6, T=200, seed=3)
        fit = lib.fit_hierarchical_panel_var(
            panel, lags=1, trend="c", n_draws=600, burn_in=400, seed=1)
        assert isinstance(fit, lib.PanelVARFit)
        assert fit.method == "hierarchical"
        assert np.allclose(fit.lag_matrices()[0], B_true, atol=0.1)
        assert fit.pooling_tau2.shape == fit.B_hat.shape
        assert fit.common_mean_B.shape == fit.B_hat.shape

    def test_shrinks_toward_common_at_short_T(self):
        # short T => noisy per-unit OLS; since the units truly share B_true,
        # hierarchical pooling should pull the unit estimates closer to it.
        panel, B_true = _make_homogeneous_panel(n_units=8, T=35, seed=5)
        hier = lib.fit_hierarchical_panel_var(
            panel, lags=1, trend="c", n_draws=500, burn_in=300, seed=1)
        mg = lib.fit_mean_group(panel, lags=1, trend="c")
        hier_dev = np.mean([self._lag_dev(hier.per_unit_B[u], B_true) for u in hier.units])
        ols_dev = np.mean([self._lag_dev(mg.per_unit_B[u], B_true) for u in hier.units])
        assert hier_dev < ols_dev

    def test_irf_and_parameter_draws(self):
        panel, B_true = _make_homogeneous_panel(n_units=5, T=150, seed=2)
        fit = lib.fit_hierarchical_panel_var(
            panel, lags=1, trend="c", n_draws=300, burn_in=200, seed=1)
        irf = lib.cholesky_irf(fit, horizon=8, bands="none")
        assert irf.median.shape == (9, 2, 2)
        draws = fit.parameter_draws(n_draws=40, seed=2)
        assert len(draws) > 0
        B, S = draws[0]
        assert B.shape == fit.B_hat.shape and S.shape == (2, 2)

    def test_long_format_and_determinism(self):
        panel, _ = _make_homogeneous_panel(n_units=4, T=120, seed=0)
        kw = dict(lags=1, trend="c", n_draws=200, burn_in=100, seed=9)
        a = lib.fit_hierarchical_panel_var(panel, **kw)
        b = lib.fit_hierarchical_panel_var(panel, **kw)
        assert np.allclose(a.common_mean_B, b.common_mean_B)
        long = pd.concat([df.assign(unit=u) for u, df in panel.items()])
        c = lib.fit_hierarchical_panel_var(long, unit_col="unit", **kw)
        assert c.n_units == 4


class TestGlobalVarUnitInvariance:
    """Acceptance check: shrinkage must combine the prior with the
    LIKELIHOOD precision X'X/sigma^2, so re-expressing every series in
    different units leaves all dimensionless lag coefficients unchanged."""

    def test_lag_coefficients_invariant_to_common_rescale(self):
        panel, _ = _make_homogeneous_panel(n_units=2, T=200, seed=3)
        fit1 = lib.fit_global_var(panel, lags=1, trend="n")
        panel_scaled = {u: df * 100.0 for u, df in panel.items()}
        fit2 = lib.fit_global_var(panel_scaled, lags=1, trend="n")
        B1 = fit1.lag_matrices()[0]
        B2 = fit2.lag_matrices()[0]
        # With EVERY series rescaled by the same factor, lag coefficients
        # are dimensionless and must agree.
        assert np.allclose(B1, B2, atol=1e-8), np.max(np.abs(B1 - B2))

    def test_lag_coefficients_transform_back_under_mixed_rescale(self):
        panel, _ = _make_homogeneous_panel(n_units=2, T=200, seed=4)
        fit1 = lib.fit_global_var(panel, lags=1, trend="n")
        # Scale only the second variable of each unit by 1000
        scale = np.array([1.0, 1000.0])
        panel_scaled = {u: df * scale for u, df in panel.items()}
        fit2 = lib.fit_global_var(panel_scaled, lags=1, trend="n")
        B1 = fit1.lag_matrices()[0]
        B2 = fit2.lag_matrices()[0]
        # lag_matrices returns A with A[i, j] multiplying variable j in
        # equation i, so the conversion back is A_orig[i, j] =
        # A_scaled[i, j] * s_j / s_i.
        n_units = 2
        s = np.tile(scale, n_units)
        B2_back = B2 * (s[None, :] / s[:, None])
        assert np.allclose(B1, B2_back, rtol=1e-6, atol=1e-8), \
            np.max(np.abs(B1 - B2_back))


class TestHierarchicalTrendMetadata:
    def test_trend_recorded_in_fit_info(self):
        panel, _ = _make_homogeneous_panel(n_units=3, T=80, seed=5)
        fit = lib.fit_hierarchical_panel_var(
            panel, lags=1, trend="n", n_draws=100, burn_in=50, seed=1)
        assert fit.fit_info["trend"] == "n"
        fit_c = lib.fit_hierarchical_panel_var(
            panel, lags=1, trend="c", n_draws=100, burn_in=50, seed=1)
        assert fit_c.fit_info["trend"] == "c"
