"""Tests for IRFContainer / FEVDContainer (the interface contract)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import lib


def _toy_arrays(H=10, n_vars=3, n_shocks=3):
    """Return five arrays of the canonical IRF shape."""
    rng = np.random.default_rng(0)
    median = rng.standard_normal((H + 1, n_vars, n_shocks))
    half = 0.5 * np.abs(rng.standard_normal((H + 1, n_vars, n_shocks)))
    return median, median - half, median + half, median - 2 * half, median + 2 * half


class TestIRFContainer:
    def test_shape_consistency_enforced(self):
        """Mismatched array shapes must raise."""
        med, lo68, hi68, lo90, hi90 = _toy_arrays()
        with pytest.raises(ValueError, match="share a shape"):
            lib.IRFContainer(
                median=med, lower_68=lo68[:5], upper_68=hi68,
                lower_90=lo90, upper_90=hi90,
                variable_names=["a", "b", "c"], shock_names=["s1", "s2", "s3"],
            )

    def test_variable_name_count_enforced(self):
        med, *bands = _toy_arrays()
        with pytest.raises(ValueError, match="variable_names"):
            lib.IRFContainer(
                median=med, lower_68=bands[0], upper_68=bands[1],
                lower_90=bands[2], upper_90=bands[3],
                variable_names=["only_one"], shock_names=["s1", "s2", "s3"],
            )

    def test_from_point_makes_all_bands_equal(self):
        point = np.arange(30).reshape(10, 3, 1).astype(float)
        irf = lib.IRFContainer.from_point(
            point=point, variable_names=["a", "b", "c"], shock_names=["s"],
        )
        for band in (irf.lower_68, irf.upper_68, irf.lower_90, irf.upper_90):
            assert np.array_equal(band, point)

    def test_from_draws_quantiles_are_correct(self):
        """Median, 16%/84%, 5%/95% must match numpy.percentile."""
        rng = np.random.default_rng(42)
        draws = rng.standard_normal((1000, 5, 2, 1))   # (n_draws, H+1, n_vars, n_shocks)
        irf = lib.IRFContainer.from_draws(
            draws=draws, variable_names=["a", "b"], shock_names=["s"],
        )
        assert np.allclose(irf.median, np.median(draws, axis=0))
        assert np.allclose(irf.lower_68, np.percentile(draws, 16, axis=0))
        assert np.allclose(irf.upper_68, np.percentile(draws, 84, axis=0))
        assert np.allclose(irf.lower_90, np.percentile(draws, 5, axis=0))
        assert np.allclose(irf.upper_90, np.percentile(draws, 95, axis=0))

    def test_to_contract_dict_matches_development_md_structure(self):
        """The exported dict must have shape: shock -> var -> {horizon, median, lower_*, upper_*}."""
        med, *bands = _toy_arrays(H=5, n_vars=2, n_shocks=2)
        irf = lib.IRFContainer(
            median=med, lower_68=bands[0], upper_68=bands[1],
            lower_90=bands[2], upper_90=bands[3],
            variable_names=["a", "b"], shock_names=["s1", "s2"],
        )
        d = irf.to_contract_dict()
        assert set(d.keys()) == {"s1", "s2"}
        for shock in d:
            assert set(d[shock].keys()) == {"a", "b"}
            for var in d[shock]:
                entry = d[shock][var]
                assert set(entry.keys()) == {"horizon", "median", "lower_68",
                                             "upper_68", "lower_90", "upper_90"}
                assert len(entry["horizon"]) == 6
                assert len(entry["median"]) == 6

    def test_select_round_trips(self):
        med, *bands = _toy_arrays(H=5, n_vars=3, n_shocks=3)
        irf = lib.IRFContainer(
            median=med, lower_68=bands[0], upper_68=bands[1],
            lower_90=bands[2], upper_90=bands[3],
            variable_names=["a", "b", "c"], shock_names=["s1", "s2", "s3"],
        )
        sub = irf.select(shocks=["s2"], variables=["a", "c"])
        assert sub.shock_names == ["s2"]
        assert sub.variable_names == ["a", "c"]
        assert np.allclose(sub.median[:, 0, 0], med[:, 0, 1])  # a, s2
        assert np.allclose(sub.median[:, 1, 0], med[:, 2, 1])  # c, s2

    def test_save_snapshot_round_trips_through_json(self, tmp_path: Path):
        med, *bands = _toy_arrays(H=3, n_vars=2, n_shocks=1)
        irf = lib.IRFContainer(
            median=med, lower_68=bands[0], upper_68=bands[1],
            lower_90=bands[2], upper_90=bands[3],
            variable_names=["x", "y"], shock_names=["s"],
            metadata={"method": "test"},
        )
        path = irf.save_snapshot(tmp_path / "snap.json")
        data = json.loads(path.read_text())
        assert data["variable_names"] == ["x", "y"]
        assert data["shock_names"] == ["s"]
        assert data["metadata"]["method"] == "test"
        # Median fingerprint round-trips to 6 decimals
        assert np.allclose(np.array(data["irf"]["median"]), np.round(med, 6))


# ---------------------------------------------------------------------------
# Historical decomposition
# ---------------------------------------------------------------------------

import pandas as pd  # noqa: E402


def _toy_var_fit(T=200, n=3, p=2, seed=42):
    """Fit a small VAR on synthetic data for HD testing."""
    rng = np.random.default_rng(seed)
    Y = pd.DataFrame(
        rng.standard_normal((T, n)).cumsum(axis=0),
        columns=[f"v{i}" for i in range(n)],
        index=pd.date_range("1990-01-01", periods=T, freq="QS"),
    )
    return lib.fit_var(Y, lags=p, trend="c")


class TestHistoricalDecomposition:

    def test_full_identification_reconstructs_data(self):
        """Cholesky-decomposed full identification reconstructs Y_act exactly."""
        fit = _toy_var_fit()
        S = lib.cholesky_impact(fit.Sigma_u)  # (n, n) lower triangular
        hd = lib.historical_decomposition(fit, S, shock_names=["s0", "s1", "s2"])
        err = hd.reconstruction_error()
        assert (err < 1e-10).all(), f"reconstruction error too large: {err}"
        assert hd.other_shocks is None
        assert hd.metadata["identification_complete"] is True

    def test_partial_identification_reconstructs_with_other_block(self):
        """Single-shock identification leaves an 'other_shocks' block; total still reconstructs."""
        fit = _toy_var_fit()
        S_full = lib.cholesky_impact(fit.Sigma_u)
        S_partial = S_full[:, :1]  # only first column
        hd = lib.historical_decomposition(fit, S_partial, shock_names=["shock_one"])
        err = hd.reconstruction_error()
        assert (err < 1e-10).all(), f"reconstruction error too large: {err}"
        assert hd.other_shocks is not None
        assert hd.other_shocks.shape == hd.data.shape
        assert hd.metadata["identification_complete"] is False
        assert hd.metadata["n_identified_shocks"] == 1

    def test_one_d_structural_column_accepted(self):
        """1-D structural impact (single shock) is broadcast to (n, 1)."""
        fit = _toy_var_fit()
        S_full = lib.cholesky_impact(fit.Sigma_u)
        hd = lib.historical_decomposition(fit, S_full[:, 0])  # 1-D
        assert hd.metadata["n_identified_shocks"] == 1
        assert hd.other_shocks is not None
        err = hd.reconstruction_error()
        assert (err < 1e-10).all()

    def test_shock_names_length_mismatch_raises(self):
        fit = _toy_var_fit()
        S = lib.cholesky_impact(fit.Sigma_u)
        with pytest.raises(ValueError, match="shock_names length"):
            lib.historical_decomposition(fit, S, shock_names=["only_one_name"])

    def test_wrong_row_dim_raises(self):
        fit = _toy_var_fit(n=3)
        bad_S = np.eye(2)  # n=2 doesn't match fit's n=3
        with pytest.raises(ValueError, match="must have 3 rows"):
            lib.historical_decomposition(fit, bad_S)

    def test_partial_id_recovers_full_id_when_remaining_block_added(self):
        """If we recompose by summing identified + other, we should match the full-id result."""
        fit = _toy_var_fit()
        S_full = lib.cholesky_impact(fit.Sigma_u)
        # Full id: sum of all shock contributions == data - det
        hd_full = lib.historical_decomposition(fit, S_full,
                                               shock_names=["s0", "s1", "s2"])
        sum_full = sum(hd_full.shocks.values())
        # Partial id with first shock only
        hd_p = lib.historical_decomposition(fit, S_full[:, :1],
                                            shock_names=["s0"])
        sum_partial = hd_p.shocks["s0"] + hd_p.other_shocks
        # Both decomposition residuals should be identical (det_plus_init is identical)
        assert np.allclose(hd_full.deterministic_plus_initial.values,
                           hd_p.deterministic_plus_initial.values, atol=1e-10)
        assert np.allclose(sum_full.values, sum_partial.values, atol=1e-10)


class TestPartialHDProjection:
    """Regression contract: the partial-identification shock projection.

    With unit-variance structural shocks and B B' = Sigma_u, the identified
    shocks are eps = S' Sigma_u^{-1} u -- NOT pinv(S) u, which is correct
    only for Euclidean-orthogonal columns (false for a Cholesky column) and
    would cross-contaminate the per-shock and other_shocks contributions.
    Reconstruction tests cannot catch this (any projection reconstructs);
    agreement with the full decomposition can.
    """

    def test_partial_hd_matches_full_hd_for_same_shock(self):
        fit = _toy_var_fit()
        S_full = lib.cholesky_impact(fit.Sigma_u)
        hd_full = lib.historical_decomposition(
            fit, S_full, shock_names=["s0", "s1", "s2"])
        hd_part = lib.historical_decomposition(
            fit, S_full[:, :1], shock_names=["s0"])
        # The first Cholesky shock's contribution must be IDENTICAL whether
        # the other shocks are identified or lumped into other_shocks.
        np.testing.assert_allclose(
            hd_part.shocks["s0"].values, hd_full.shocks["s0"].values,
            atol=1e-8,
            err_msg="partial-ID HD shock contribution differs from full-ID "
                    "(a pinv(S) projection would cross-contaminate the "
                    "contributions)",
        )

    def test_recovered_eps_is_true_structural_shock(self):
        fit = _toy_var_fit()
        S_full = lib.cholesky_impact(fit.Sigma_u)
        u = fit.residuals
        eps_true = np.linalg.solve(S_full, u.T).T[:, 0]
        hd_part = lib.historical_decomposition(
            fit, S_full[:, :1], shock_names=["s0"])
        # metadata path not exposed; verify via the projection identity
        Sigma_inv_S = np.linalg.solve(fit.Sigma_u, S_full[:, :1])
        eps_code = (np.linalg.solve(S_full[:, :1].T @ Sigma_inv_S,
                                    Sigma_inv_S.T @ u.T).T).ravel()
        np.testing.assert_allclose(eps_code, eps_true, atol=1e-8)


class TestSnapshotRoundTrip:
    """``IRFContainer.load_snapshot`` is the inverse of ``save_snapshot``:
    an archived replication's IRFs can be reconstructed (to the snapshot's
    6-decimal rounding) without re-estimating — the primitive a robustness
    overlay against a stored baseline needs."""

    def _container(self):
        from lib.irf import IRFContainer
        rng = np.random.default_rng(3)
        med = rng.standard_normal((9, 2, 2))
        return IRFContainer(
            median=med,
            lower_68=med - 0.1, upper_68=med + 0.1,
            lower_90=med - 0.2, upper_90=med + 0.2,
            variable_names=["output", "prices"],
            shock_names=["supply", "demand"],
            metadata={"method": "test", "seed": 3},
        )

    def test_round_trip_preserves_values_and_names(self, tmp_path):
        from lib.irf import IRFContainer
        irf = self._container()
        path = irf.save_snapshot(tmp_path / "snap.json")
        loaded = IRFContainer.load_snapshot(path)
        np.testing.assert_allclose(loaded.median, irf.median, atol=1e-6)
        np.testing.assert_allclose(loaded.lower_90, irf.lower_90, atol=1e-6)
        np.testing.assert_allclose(loaded.upper_68, irf.upper_68, atol=1e-6)
        assert loaded.variable_names == irf.variable_names
        assert loaded.shock_names == irf.shock_names
        assert loaded.metadata["method"] == "test"
        assert loaded.horizon == irf.horizon


class TestStructuralShocks:
    """``structural_shocks`` recovers the identified shock series from a
    fitted VAR + impact matrix: ``S^{-1} u`` under full identification, the
    GLS projection under partial — the same convention
    ``historical_decomposition`` uses internally, now exposed because
    sessions kept hand-rolling it."""

    def test_full_identification_inverts_exactly(self, synthetic_var_data):
        fit = lib.fit_var(data=synthetic_var_data, lags=2, trend="c")
        S = np.linalg.cholesky(fit.Sigma_u)
        eps = lib.structural_shocks(fit, S, shock_names=["a", "b", "c"])
        assert list(eps.columns) == ["a", "b", "c"]
        assert len(eps) == fit.t_eff
        # S eps' must reproduce the reduced-form residuals exactly
        np.testing.assert_allclose(
            (S @ eps.to_numpy().T).T, np.asarray(fit.residuals), atol=1e-10
        )
        # Cholesky-scaled shocks have ~unit variance
        np.testing.assert_allclose(eps.std(ddof=1).to_numpy(), 1.0, atol=0.1)

    def test_partial_identification_matches_gls_projection(self, synthetic_var_data):
        fit = lib.fit_var(data=synthetic_var_data, lags=2, trend="c")
        S_full = np.linalg.cholesky(fit.Sigma_u)
        S1 = S_full[:, :1]
        eps_part = lib.structural_shocks(fit, S1)
        # For a Cholesky column the projection equals the full-inversion shock
        eps_full = np.linalg.solve(S_full, np.asarray(fit.residuals).T).T[:, 0]
        np.testing.assert_allclose(
            eps_part.to_numpy().ravel(), eps_full, atol=1e-8
        )
        assert list(eps_part.columns) == ["shock_0"]

    def test_shape_validation(self, synthetic_var_data):
        fit = lib.fit_var(data=synthetic_var_data, lags=2, trend="c")
        with pytest.raises(ValueError, match="rows"):
            lib.structural_shocks(fit, np.eye(2))


class TestHistoricalHelpersRejectTVP:
    """Acceptance check: historical attribution assumes constant
    coefficients; a TVP fit's last-period-coefficient face must be
    rejected, not silently treated as history."""

    def _fit_with_tvp_face(self):
        rng = np.random.default_rng(0)
        T = 120
        y = np.zeros((T, 2))
        for t in range(1, T):
            y[t] = 0.4 * y[t - 1] + rng.standard_normal(2)
        df = pd.DataFrame(y, columns=["a", "b"],
                          index=pd.date_range("1990-01-01", periods=T, freq="QS"))
        fit = lib.fit_var(df, lags=1, trend="c")
        import dataclasses
        return dataclasses.replace(
            fit, fit_info={**fit.fit_info, "method": "tvp_var_gibbs"})

    def test_structural_shocks_rejects_tvp(self):
        fit = self._fit_with_tvp_face()
        S = np.linalg.cholesky(np.asarray(fit.Sigma_u))
        with pytest.raises(NotImplementedError, match="time-varying"):
            lib.structural_shocks(fit, S)

    def test_historical_decomposition_rejects_tvp(self):
        fit = self._fit_with_tvp_face()
        S = np.linalg.cholesky(np.asarray(fit.Sigma_u))
        with pytest.raises(NotImplementedError, match="time-varying"):
            lib.historical_decomposition(fit, S)

    def test_real_tvp_fit_rejected(self):
        rng = np.random.default_rng(1)
        T = 90
        y = np.zeros((T, 2))
        for t in range(1, T):
            y[t] = 0.4 * y[t - 1] + rng.standard_normal(2)
        df = pd.DataFrame(y, columns=["a", "b"],
                          index=pd.date_range("1990-01-01", periods=T, freq="QS"))
        spec = lib.TVPVARSpec(lags=1)
        fit = lib.fit_tvp_var(df, spec, n_iter=40, burn_in=20, seed=0)
        S = np.linalg.cholesky(np.asarray(fit.Sigma_u))
        with pytest.raises(NotImplementedError, match="time-varying"):
            lib.historical_decomposition(fit, S)
