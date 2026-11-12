"""Tests for ``lib.diagnostics`` — the one-button residual + stability +
lag-selection report.

Validation strategy:

1. **Clean DGP → clean diagnostics** — i.i.d. Gaussian innovations should
   produce no Ljung-Box / Jarque-Bera / ARCH rejections.
2. **Known-bad DGP → flagged** — strong AR(1) errors in residuals should
   trip Ljung-Box; pure ARCH(1) shocks should trip ARCH-LM; t-distributed
   shocks should trip Jarque-Bera.
3. **Stability classification** — stationary AR(1) (φ=0.5) labelled
   "stationary"; near-unit-root (φ=0.99) labelled "marginal"; explosive
   coefficient labelled "non-stationary".
4. **Lag-selection comparison** — AIC/BIC/HQIC recommendations are
   integer values in the searched range; warnings flag disagreements
   with the current p.
5. **JSON serialisation** — ``to_log_dict()`` is JSON-safe.
6. **Container API** — fields populated, summary() produces a string.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import lib  # noqa: E402
from lib.diagnostics import DiagnosticReport, diagnostics  # noqa: E402

warnings.filterwarnings("ignore", category=Warning)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_var(T=300, n=2, phi=0.5, innov_kind="gaussian", seed=0):
    """Synthetic VAR(1) data. phi can be a scalar diagonal coefficient or a
    matrix; innov_kind ∈ {'gaussian', 'arch', 't', 'ar1_residuals'}."""
    rng = np.random.default_rng(seed)
    A = np.eye(n) * phi if np.isscalar(phi) else np.asarray(phi)

    y = np.zeros((T, n))
    if innov_kind == "gaussian":
        for t in range(1, T):
            y[t] = A @ y[t-1] + rng.normal(0, 1, n)
    elif innov_kind == "t":
        # Heavy-tailed innovations (t with 3 df) — should trip JB
        for t in range(1, T):
            y[t] = A @ y[t-1] + rng.standard_t(3, n)
    elif innov_kind == "arch":
        # Pure ARCH(1) variance dynamics: sigma_t^2 = 0.1 + 0.85 * eps_{t-1}^2
        sigma2 = np.full(n, 1.0)
        eps = np.zeros((T, n))
        for t in range(1, T):
            sigma2 = 0.1 + 0.85 * eps[t-1] ** 2
            eps[t] = np.sqrt(sigma2) * rng.normal(0, 1, n)
            y[t] = A @ y[t-1] + eps[t]
    else:
        raise ValueError(innov_kind)
    idx = pd.date_range("2000-01-01", periods=T, freq="QS")
    return pd.DataFrame(y, index=idx, columns=[f"x{i}" for i in range(n)])


@pytest.fixture
def clean_var_fit():
    """Stationary VAR(1) with iid Gaussian innovations — should produce a
    diagnostically clean fit."""
    data = _make_var(T=400, n=2, phi=0.5, innov_kind="gaussian", seed=7)
    return lib.fit_var(data, lags=1, trend="c")


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------


class TestReportContainer:
    def test_returns_diagnostic_report(self, clean_var_fit):
        report = diagnostics(clean_var_fit)
        assert isinstance(report, DiagnosticReport)
        assert report.fit_class == "VARFit"
        assert report.n_vars == 2
        assert report.p == 1

    def test_summary_is_string(self, clean_var_fit):
        report = diagnostics(clean_var_fit)
        s = report.summary()
        assert isinstance(s, str)
        assert "Diagnostics for VARFit" in s
        assert "Stability:" in s
        assert "Residual diagnostics" in s
        assert "Lag selection" in s

    def test_to_log_dict_is_json_safe(self, clean_var_fit):
        report = diagnostics(clean_var_fit)
        d = report.to_log_dict()
        # Must serialise without TypeError
        json.dumps(d)
        assert d["fit_class"] == "VARFit"
        assert "stability" in d
        assert "residuals" in d


# ---------------------------------------------------------------------------
# Clean-DGP: no warnings (or very few)
# ---------------------------------------------------------------------------


class TestCleanDGP:
    def test_stationary_iid_gaussian_clean(self, clean_var_fit):
        report = diagnostics(clean_var_fit, max_lag_search=4)
        # Stability should be 'stationary'
        assert report.stability["status"] == "stationary"
        # All Ljung-Box / JB / ARCH p-values should be > 0.01
        for var, r in report.residuals.items():
            assert r["ljung_box_pvalue"] > 0.01, f"{var}: LB rejects on clean DGP"
            assert r["jarque_bera_pvalue"] > 0.01, f"{var}: JB rejects on clean DGP"
            assert r["arch_pvalue"] > 0.01, f"{var}: ARCH rejects on clean DGP"


# ---------------------------------------------------------------------------
# Stability classification
# ---------------------------------------------------------------------------


class TestStabilityClassification:
    def test_stationary(self, clean_var_fit):
        report = diagnostics(clean_var_fit, max_lag_search=2)
        assert report.stability["status"] == "stationary"
        assert report.stability["stable"] is True

    def test_marginal_near_unit_root(self):
        """φ = 0.99 with long sample → OLS estimate stays near 1 → marginal."""
        data = _make_var(T=800, n=2, phi=0.99, innov_kind="gaussian", seed=11)
        fit = lib.fit_var(data, lags=1, trend="c")
        report = diagnostics(fit, max_lag_search=2)
        # Either marginal or non-stationary depending on finite-sample noise —
        # both indicate the near-unit-root regime.
        assert report.stability["status"] in (
            "marginal (near unit root)", "non-stationary / unstable",
        )

    def test_unstable_at_or_above_unit_root(self):
        """Explosive coefficient gives max |eig| > 1 — flagged unstable."""
        data = _make_var(T=200, n=2, phi=1.02, innov_kind="gaussian", seed=13)
        fit = lib.fit_var(data, lags=1, trend="c")
        report = diagnostics(fit, max_lag_search=2)
        assert report.stability["status"] == "non-stationary / unstable"
        assert report.stability["stable"] is False
        assert any("outside the unit circle" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# Residual-test flagging on known-bad DGPs
# ---------------------------------------------------------------------------


class TestKnownBadDGPs:
    def test_arch_dgp_flags_arch(self):
        """Innovations driven by ARCH(1) — at least one variable should trip
        the ARCH-LM test."""
        data = _make_var(T=600, n=2, phi=0.3, innov_kind="arch", seed=17)
        fit = lib.fit_var(data, lags=1, trend="c")
        report = diagnostics(fit, max_lag_search=2)
        arch_pvals = [r["arch_pvalue"] for r in report.residuals.values()]
        assert min(arch_pvals) < 0.05, "ARCH DGP did not trip ARCH-LM"
        assert any("conditional heteroskedasticity" in w for w in report.warnings)

    def test_t_dgp_flags_jarque_bera(self):
        """t(3) innovations — JB should reject normality on at least one var."""
        data = _make_var(T=600, n=2, phi=0.3, innov_kind="t", seed=19)
        fit = lib.fit_var(data, lags=1, trend="c")
        report = diagnostics(fit, max_lag_search=2)
        jb_pvals = [r["jarque_bera_pvalue"] for r in report.residuals.values()]
        assert min(jb_pvals) < 0.05, "t-distributed DGP did not trip JB"

    def test_underfit_lag_flags_ljung_box(self):
        """If true DGP is AR(p) with p>1 but we fit with p=1, residuals
        should show autocorrelation."""
        rng = np.random.default_rng(23)
        T, n = 400, 2
        y = np.zeros((T, n))
        A1 = np.diag([0.3, 0.4])
        A2 = np.diag([0.5, 0.4])    # second lag is large — AR(1) underfit
        for t in range(2, T):
            y[t] = A1 @ y[t-1] + A2 @ y[t-2] + rng.normal(0, 1, n)
        data = pd.DataFrame(y, index=pd.date_range("2000-01-01", periods=T, freq="QS"),
                             columns=["x0", "x1"])
        fit = lib.fit_var(data, lags=1, trend="c")
        report = diagnostics(fit, max_lag_search=4)
        lb_pvals = [r["ljung_box_pvalue"] for r in report.residuals.values()]
        assert min(lb_pvals) < 0.05, "AR(2)-as-AR(1) misfit didn't trip LB"
        # And the lag-selection comparison should prefer p > 1
        ls = report.lag_selection
        assert ls["aic_recommended"] > 1


# ---------------------------------------------------------------------------
# Lag selection
# ---------------------------------------------------------------------------


class TestLagSelection:
    def test_table_present_and_ordered(self, clean_var_fit):
        report = diagnostics(clean_var_fit, max_lag_search=6)
        ls = report.lag_selection
        assert ls is not None
        assert "table" in ls
        assert len(ls["table"]) >= 5     # at least p=1..5 entries
        # All entries should have aic/bic/hqic floats
        for row in ls["table"]:
            assert isinstance(row["p"], int)
            assert "aic" in row and "bic" in row and "hqic" in row

    def test_recommendations_in_range(self, clean_var_fit):
        report = diagnostics(clean_var_fit, max_lag_search=6)
        ls = report.lag_selection
        for ic in ("aic", "bic", "hqic"):
            rec = ls.get(f"{ic}_recommended")
            assert rec is not None
            assert 1 <= rec <= 6

    def test_max_lag_search_zero_skips_lag_selection(self, clean_var_fit):
        report = diagnostics(clean_var_fit, max_lag_search=0)
        assert report.lag_selection is None


# ---------------------------------------------------------------------------
# Works on BVAR too
# ---------------------------------------------------------------------------


class TestBVARIntegration:
    def test_works_on_bvar(self):
        rng = np.random.default_rng(29)
        T, n = 300, 2
        y = np.zeros((T, n))
        for t in range(1, T):
            y[t] = 0.5 * y[t-1] + rng.normal(0, 1, n)
        data = pd.DataFrame(
            y, index=pd.date_range("2000-01-01", periods=T, freq="QS"),
            columns=["x0", "x1"],
        )
        prior = lib.MinnesotaPrior(
            lambda_1=0.1, lambda_2=0.5, lambda_3=1.0,
            lambda_4=100.0, mu=1.0, delta=1.0,
        )
        fit = lib.fit_minnesota_bvar(data, lags=1, prior=prior, trend="c")
        report = diagnostics(fit, max_lag_search=4)
        assert report.fit_class == "BVARFit"
        assert report.n_vars == 2
        assert report.p == 1
        # Should still produce a stability + residuals + lag_selection report
        assert "stability" in report.to_log_dict()


# ---------------------------------------------------------------------------
# Package wiring
# ---------------------------------------------------------------------------


def test_lib_exports():
    assert lib.diagnostics is diagnostics
    assert lib.DiagnosticReport is DiagnosticReport


def test_lib_all_lists_diagnostics_surface():
    assert "diagnostics" in lib.__all__
    assert "DiagnosticReport" in lib.__all__


class TestLjungBoxMultipleLags:
    """``lb_lags`` accepts a sequence: one Ljung-Box test per requested lag,
    each stored under ``residuals[var]["ljung_box"][lag]`` — the multi-lag
    convention the diagnostics skill specifies for monthly data (6/12/24).
    The scalar ``ljung_box_pvalue`` keeps reporting the first lag so older
    consumers are unaffected."""

    def test_sequence_yields_per_lag_results(self, synthetic_var_data):
        fit = lib.fit_var(data=synthetic_var_data, lags=2, trend="c")
        dr = diagnostics(fit, max_lag_search=0, lb_lags=[4, 12])
        for var, r in dr.residuals.items():
            assert set(r["ljung_box"]) == {4, 12}
            for lag in (4, 12):
                assert 0.0 <= r["ljung_box"][lag]["p_value"] <= 1.0
            assert r["ljung_box_pvalue"] == r["ljung_box"][4]["p_value"]

    def test_int_still_works_and_is_labelled(self, synthetic_var_data):
        fit = lib.fit_var(data=synthetic_var_data, lags=2, trend="c")
        dr = diagnostics(fit, max_lag_search=0, lb_lags=8)
        for var, r in dr.residuals.items():
            assert set(r["ljung_box"]) == {8}
            assert r["ljung_box_pvalue"] == r["ljung_box"][8]["p_value"]
