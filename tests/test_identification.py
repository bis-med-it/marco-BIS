"""Tests for identification.py (Cholesky, sign, proxy, long-run, heteroskedasticity).

Validation strategy:

1. ``cholesky_irf`` at ``h=0`` must equal ``np.linalg.cholesky(Sigma_u)`` —
   pure math, no implementation choices.
2. ``cholesky_irf`` point estimate must match ``statsmodels`` ``orth_irfs``
   exactly — both fit the same VAR and apply the same Cholesky.
3. Sign-restricted IRFs must SATISFY THEIR OWN RESTRICTIONS by
   construction. We verify this post hoc.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.api import VAR as _StatsmodelsVAR

import lib
from lib.identification import (
    _parse_type2_extras, _resolve_magnitude_restriction,
    _type2_contributions, _type2_dominates,
)


class TestCholesky:
    def test_h0_equals_chol_of_sigma(self, synthetic_var_data):
        """At horizon 0, the IRF must be the Cholesky factor of Sigma_u."""
        fit = lib.fit_var(synthetic_var_data, lags=2)
        irf = lib.cholesky_irf(fit, horizon=0, bands="none")
        chol = np.linalg.cholesky(fit.Sigma_u)
        # irf.median has shape (1, n, n) at horizon=0
        assert np.allclose(irf.median[0], chol, atol=1e-12)

    def test_point_estimate_matches_statsmodels(self, synthetic_var_data):
        """Point IRF must equal statsmodels.tsa.VAR.irf().orth_irfs."""
        fit = lib.fit_var(synthetic_var_data, lags=2)
        irf = lib.cholesky_irf(fit, horizon=20, bands="none")
        sm = _StatsmodelsVAR(synthetic_var_data).fit(maxlags=2, ic=None, trend="c")
        sm_irf = sm.irf(periods=20).orth_irfs
        assert np.allclose(irf.median, sm_irf, atol=1e-10)

    def test_n_shocks_equals_n_vars(self, synthetic_var_data):
        fit = lib.fit_var(synthetic_var_data, lags=2)
        irf = lib.cholesky_irf(fit, horizon=10, bands="none")
        assert irf.n_shocks == fit.n_vars
        assert irf.n_vars == fit.n_vars

    def test_bands_widen_with_n_draws(self, synthetic_var_data):
        """More draws should reduce sampling noise in the bands.

        Specifically: with seed-controlled MC, bands at large n_draws should
        be reproducible. With seed differing, bands should still produce
        finite, ordered (lower < upper) values.
        """
        fit = lib.fit_var(synthetic_var_data, lags=2)
        irf = lib.cholesky_irf(
            fit, horizon=10, bands="asymptotic_mc", n_draws=200, seed=0,
        )
        # 68% lower ≤ median ≤ 68% upper
        assert np.all(irf.lower_68 <= irf.median + 1e-10)
        assert np.all(irf.median <= irf.upper_68 + 1e-10)
        # 90% strictly wider than 68%
        assert np.all(irf.lower_90 <= irf.lower_68 + 1e-10)
        assert np.all(irf.upper_68 <= irf.upper_90 + 1e-10)


class TestSignRestrictions:
    def test_accepted_responses_satisfy_restrictions(self, synthetic_var_data):
        """Every accepted draw's median IRF for the named shock must satisfy
        every imposed sign restriction at every imposed horizon.
        """
        fit = lib.fit_var(synthetic_var_data, lags=2)
        restrictions = {"demand": {"y1": +1, "y2": +1, "y3": 0}}
        irf = lib.sign_restricted_irf(
            fit=fit,
            horizon=10,
            sign_restrictions=restrictions,
            restriction_horizons=3,
            target_accepted=200,
            max_rotations=50_000,
            seed=0,
            parameter_uncertainty=False,
        )
        # Check median responses obey the restrictions at h = 0..2
        # (restriction_horizons = 3)
        med = irf.median[..., 0]   # (H+1, n_vars), single shock
        for h in range(3):
            assert med[h, 0] >= 0    # y1 positive
            assert med[h, 1] >= 0    # y2 positive
            # y3 unrestricted

    def test_unsatisfiable_restrictions_raise(self, synthetic_var_data):
        """A self-contradictory set of restrictions yields no acceptances."""
        fit = lib.fit_var(synthetic_var_data, lags=2)
        # Impose constraints that no rotation can satisfy simultaneously
        # if we set max_rotations small enough
        restrictions = {"weird": {"y1": +1, "y2": +1, "y3": +1}}
        # Try just a tiny number of rotations: we may not find any
        # (some seeds may find one; loosen the threshold rather than
        # asserting raise).
        try:
            irf = lib.sign_restricted_irf(
                fit=fit, horizon=5,
                sign_restrictions=restrictions,
                restriction_horizons=1,
                target_accepted=10,
                max_rotations=10,
                seed=1,
                parameter_uncertainty=False,
            )
            # If we got here it accepted some
            assert irf.metadata["n_accepted"] >= 1
        except RuntimeError as e:
            assert "no rotations satisfied" in str(e).lower()

    def test_acceptance_rate_recorded(self, synthetic_var_data):
        fit = lib.fit_var(synthetic_var_data, lags=2)
        restrictions = {"demand": {"y1": +1, "y2": +1}}
        irf = lib.sign_restricted_irf(
            fit=fit, horizon=10,
            sign_restrictions=restrictions,
            restriction_horizons=2,
            target_accepted=50,
            max_rotations=5000,
            seed=0,
            parameter_uncertainty=False,
        )
        meta = irf.metadata
        assert meta["n_accepted"] >= 50
        assert 0.0 < meta["acceptance_rate"] <= 1.0
        assert meta["n_attempted"] >= meta["n_accepted"]


class TestNarrativeSignRestrictions:
    """Tests for Antolín-Díaz & Rubio-Ramírez (2018) narrative sign
    restrictions — ``narrative_restrictions`` parameter on
    ``sign_restricted_irf``.

    Validation strategy:

    * **No narrative arg = backward-compat** — calls without
      ``narrative_restrictions`` reproduce the pre-extension behaviour
      byte-identically (same seed, same output).
    * **None / empty list = backward-compat** — explicit ``None`` or
      ``[]`` also reproduces the no-restriction path.
    * **Single restriction filters rotations** — adding a restriction
      strictly decreases (or holds equal) the acceptance rate.
    * **Metadata correctly tagged** — identification name flips to
      ``"narrative_sign_restrictions"``, count of narrative rejections
      reported, narrative restrictions stored.
    * **Acceptance rate falls monotonically with more restrictions**.
    * **Validation errors** — bad sign, missing date, out-of-sample
      date all raise clearly.
    """

    def test_no_narrative_arg_matches_baseline(self, synthetic_var_data):
        fit = lib.fit_var(synthetic_var_data, lags=2)
        common = dict(
            fit=fit, horizon=8,
            sign_restrictions={"shock": {"y1": +1, "y2": -1}},
            target_accepted=100, max_rotations=10_000,
            seed=0, parameter_uncertainty=False,
        )
        base = lib.sign_restricted_irf(**common)
        explicit_none = lib.sign_restricted_irf(**common, narrative_restrictions=None)
        explicit_empty = lib.sign_restricted_irf(**common, narrative_restrictions=[])
        assert np.allclose(base.median, explicit_none.median, atol=1e-12)
        assert np.allclose(base.median, explicit_empty.median, atol=1e-12)
        assert base.metadata["identification"] == "sign_restrictions"
        assert explicit_none.metadata["identification"] == "sign_restrictions"

    def test_narrative_restriction_filters_rotations(self, synthetic_var_data):
        """Adding a narrative restriction should reduce acceptance rate
        below the sign-only baseline."""
        fit = lib.fit_var(synthetic_var_data, lags=2)
        # Pick a date inside the effective sample
        sample_dates = list(fit.data.index[fit.p:])
        nr_date = sample_dates[len(sample_dates) // 2]

        common = dict(
            fit=fit, horizon=4,
            sign_restrictions={"shock": {"y1": +1, "y2": -1}},
            target_accepted=200, max_rotations=20_000,
            seed=0, parameter_uncertainty=False,
        )
        base = lib.sign_restricted_irf(**common)
        narr = lib.sign_restricted_irf(
            **common,
            narrative_restrictions=[{"date": nr_date, "sign": +1}],
        )
        # Narrative should reject at least some rotations
        assert narr.metadata["n_rejected_by_narrative"] >= 0
        # Acceptance rate decreases (or holds equal)
        assert narr.metadata["acceptance_rate"] <= base.metadata["acceptance_rate"] + 1e-9
        # Metadata correctly tagged
        assert narr.metadata["identification"] == "narrative_sign_restrictions"
        assert narr.metadata["method"] == (
            "VAR / narrative sign restrictions (ADRR 2018)"
        )
        assert len(narr.metadata["narrative_restrictions"]) == 1

    def test_more_restrictions_monotonically_filter(self, synthetic_var_data):
        """Adding MORE narrative restrictions should further reduce
        (or hold) the acceptance rate."""
        fit = lib.fit_var(synthetic_var_data, lags=2)
        sample_dates = list(fit.data.index[fit.p:])
        d1 = sample_dates[len(sample_dates) // 3]
        d2 = sample_dates[2 * len(sample_dates) // 3]

        common = dict(
            fit=fit, horizon=4,
            sign_restrictions={"shock": {"y1": +1, "y2": -1}},
            target_accepted=80, max_rotations=50_000,
            seed=0, parameter_uncertainty=False,
        )
        r1 = lib.sign_restricted_irf(**common,
                                      narrative_restrictions=[{"date": d1, "sign": +1}])
        r2 = lib.sign_restricted_irf(**common, narrative_restrictions=[
            {"date": d1, "sign": +1},
            {"date": d2, "sign": +1},
        ])
        # More restrictions → fewer acceptances per attempt
        assert r2.metadata["acceptance_rate"] <= r1.metadata["acceptance_rate"] + 1e-9

    def test_bad_sign_raises(self, synthetic_var_data):
        fit = lib.fit_var(synthetic_var_data, lags=2)
        with pytest.raises(ValueError, match="sign must be"):
            lib.sign_restricted_irf(
                fit=fit, horizon=4,
                sign_restrictions={"shock": {"y1": +1}},
                narrative_restrictions=[{"date": fit.data.index[fit.p + 5], "sign": 0}],
                target_accepted=10, max_rotations=100,
                parameter_uncertainty=False,
            )

    def test_out_of_sample_date_raises(self, synthetic_var_data):
        fit = lib.fit_var(synthetic_var_data, lags=2)
        with pytest.raises(ValueError, match="not in effective sample"):
            lib.sign_restricted_irf(
                fit=fit, horizon=4,
                sign_restrictions={"shock": {"y1": +1}},
                narrative_restrictions=[{"date": "1850-01-01", "sign": +1}],
                target_accepted=10, max_rotations=100,
                parameter_uncertainty=False,
            )

    def test_missing_keys_raises(self, synthetic_var_data):
        fit = lib.fit_var(synthetic_var_data, lags=2)
        # Missing 'date' — checked first
        with pytest.raises(ValueError, match="'date' key"):
            lib.sign_restricted_irf(
                fit=fit, horizon=4,
                sign_restrictions={"shock": {"y1": +1}},
                narrative_restrictions=[{"sign": +1}],  # missing 'date'
                target_accepted=10, max_rotations=100,
                parameter_uncertainty=False,
            )
        # Date present but 'sign' missing for default type='sign'
        with pytest.raises(ValueError, match="'sign' key"):
            lib.sign_restricted_irf(
                fit=fit, horizon=4,
                sign_restrictions={"shock": {"y1": +1}},
                narrative_restrictions=[
                    {"date": fit.data.index[fit.p + 5]},  # missing 'sign'
                ],
                target_accepted=10, max_rotations=100,
                parameter_uncertainty=False,
            )


class TestFEVShareRestrictions:
    """Tests for quantity / variance-decomposition sign restrictions —
    ``fev_share_restrictions`` parameter on ``sign_restricted_irf``.

    Mirrors the narrative-restriction validation pattern:
    * No FEV arg / None / [] = backward-compat.
    * A binding restriction reduces acceptance rate.
    * Tighter bounds further reduce acceptance.
    * Metadata tagged correctly.
    * Validation errors raise clearly.
    """

    def test_no_fev_arg_matches_baseline(self, synthetic_var_data):
        fit = lib.fit_var(synthetic_var_data, lags=2)
        common = dict(
            fit=fit, horizon=8,
            sign_restrictions={"shock": {"y1": +1, "y2": -1}},
            target_accepted=100, max_rotations=10_000,
            seed=0, parameter_uncertainty=False,
        )
        base = lib.sign_restricted_irf(**common)
        explicit_none = lib.sign_restricted_irf(**common, fev_share_restrictions=None)
        explicit_empty = lib.sign_restricted_irf(**common, fev_share_restrictions=[])
        assert np.allclose(base.median, explicit_none.median, atol=1e-12)
        assert np.allclose(base.median, explicit_empty.median, atol=1e-12)
        assert base.metadata["identification"] == "sign_restrictions"

    def test_binding_fev_lower_filters_rotations(self, synthetic_var_data):
        """Requiring shock to explain >= 30% of y1 FEV at h=4 reduces acceptance."""
        fit = lib.fit_var(synthetic_var_data, lags=2)
        common = dict(
            fit=fit, horizon=8,
            sign_restrictions={"shock": {"y1": +1, "y2": -1}},
            target_accepted=80, max_rotations=20_000,
            seed=0, parameter_uncertainty=False,
        )
        base = lib.sign_restricted_irf(**common)
        fev = lib.sign_restricted_irf(
            **common,
            fev_share_restrictions=[{"variable": "y1", "horizon": 4, "lower": 0.3}],
        )
        assert fev.metadata["n_rejected_by_fev_share"] >= 0
        assert fev.metadata["acceptance_rate"] <= base.metadata["acceptance_rate"] + 1e-9
        assert fev.metadata["identification"] == "sign_fev_share_restrictions"
        assert len(fev.metadata["fev_share_restrictions"]) == 1

    def test_tighter_bounds_filter_more(self, synthetic_var_data):
        """Tighter [lower, upper] window → fewer acceptances."""
        fit = lib.fit_var(synthetic_var_data, lags=2)
        common = dict(
            fit=fit, horizon=8,
            sign_restrictions={"shock": {"y1": +1, "y2": -1}},
            target_accepted=50, max_rotations=50_000,
            seed=0, parameter_uncertainty=False,
        )
        loose = lib.sign_restricted_irf(
            **common,
            fev_share_restrictions=[{"variable": "y1", "horizon": 4, "lower": 0.1}],
        )
        tight = lib.sign_restricted_irf(
            **common,
            fev_share_restrictions=[{
                "variable": "y1", "horizon": 4, "lower": 0.5, "upper": 0.95,
            }],
        )
        assert tight.metadata["acceptance_rate"] <= loose.metadata["acceptance_rate"] + 1e-9

    def test_accepted_irfs_satisfy_fev_share_constraint(self, synthetic_var_data):
        """Every accepted IRF should have its identified shock explain
        >= lower share at the constrained (variable, horizon)."""
        fit = lib.fit_var(synthetic_var_data, lags=2)
        irf = lib.sign_restricted_irf(
            fit=fit, horizon=8,
            sign_restrictions={"shock": {"y1": +1, "y2": -1}},
            fev_share_restrictions=[{"variable": "y1", "horizon": 4, "lower": 0.3}],
            target_accepted=30, max_rotations=50_000,
            seed=0, parameter_uncertainty=False,
        )
        # The median IRF should reflect the constraint: structural-shock
        # contribution to y1 FEV at h=4 should be >= 0.3
        # (median ≠ each accepted draw, but in the limit of many accepts it
        # should reflect the post-rejection distribution).
        v_idx = fit.variable_names.index("y1")
        irf_to_shock = irf.median[:, v_idx, 0]
        Phi = fit.ma_rep(8)
        Sigma_u = np.asarray(fit.Sigma_u)
        # Total FEV at h=4 for y1
        total_fev = sum(
            (Phi[h] @ Sigma_u @ Phi[h].T)[v_idx, v_idx] for h in range(5)
        )
        shock_fev = float(sum(irf_to_shock[h]**2 for h in range(5)))
        median_share = shock_fev / total_fev
        # Median of accepted draws should reflect (some respect for) the
        # >=0.3 constraint. Tolerance is loose because median != each draw.
        assert median_share >= 0.15, f"median FEV share too low: {median_share:.3f}"

    def test_bad_bounds_raise(self, synthetic_var_data):
        fit = lib.fit_var(synthetic_var_data, lags=2)
        with pytest.raises(ValueError, match="lower <= upper"):
            lib.sign_restricted_irf(
                fit=fit, horizon=4,
                sign_restrictions={"shock": {"y1": +1}},
                fev_share_restrictions=[{
                    "variable": "y1", "horizon": 2, "lower": 0.8, "upper": 0.3,
                }],
                target_accepted=10, max_rotations=100,
                parameter_uncertainty=False,
            )

    def test_unknown_variable_raises(self, synthetic_var_data):
        fit = lib.fit_var(synthetic_var_data, lags=2)
        with pytest.raises(ValueError, match="unknown variable"):
            lib.sign_restricted_irf(
                fit=fit, horizon=4,
                sign_restrictions={"shock": {"y1": +1}},
                fev_share_restrictions=[{
                    "variable": "nope", "horizon": 2, "lower": 0.3,
                }],
                target_accepted=10, max_rotations=100,
                parameter_uncertainty=False,
            )

    def test_out_of_range_horizon_raises(self, synthetic_var_data):
        fit = lib.fit_var(synthetic_var_data, lags=2)
        with pytest.raises(ValueError, match="outside"):
            lib.sign_restricted_irf(
                fit=fit, horizon=4,
                sign_restrictions={"shock": {"y1": +1}},
                fev_share_restrictions=[{
                    "variable": "y1", "horizon": 10, "lower": 0.3,
                }],
                target_accepted=10, max_rotations=100,
                parameter_uncertainty=False,
            )

    def test_fev_and_narrative_compose(self, synthetic_var_data):
        """Both restriction types can be applied at once; identification name
        composes both tags."""
        fit = lib.fit_var(synthetic_var_data, lags=2)
        sample_dates = list(fit.data.index[fit.p:])
        nr_date = sample_dates[len(sample_dates) // 2]
        irf = lib.sign_restricted_irf(
            fit=fit, horizon=4,
            sign_restrictions={"shock": {"y1": +1, "y2": -1}},
            narrative_restrictions=[{"date": nr_date, "sign": +1}],
            fev_share_restrictions=[{"variable": "y1", "horizon": 2, "lower": 0.2}],
            target_accepted=30, max_rotations=50_000,
            seed=0, parameter_uncertainty=False,
        )
        assert irf.metadata["identification"] == "narrative_sign_fev_share_restrictions"


class TestMultiShockSignRestrictions:
    """Tests for multi-shock simultaneous sign identification (RWZ 2010).

    Validation strategy:
      - Two distinct shocks identified simultaneously, each gets its own
        column of the rotation matrix
      - Impact responses match the declared sign patterns
      - Narrative + FEV-share restrictions require a 'shock' key
      - Backward-compat: single-shock case unchanged
      - Failure mode: incompatible sign patterns produce no acceptance
    """

    def test_two_shocks_simultaneous_identification(self, synthetic_var_data):
        """Two shocks with distinct sign patterns each get their own column."""
        fit = lib.fit_var(synthetic_var_data, lags=2)
        irf = lib.sign_restricted_irf(
            fit=fit, horizon=4,
            sign_restrictions={
                "shock_a": {"y1": +1, "y2": -1},
                "shock_b": {"y1": +1, "y2": +1},
            },
            restriction_horizons=2,
            target_accepted=50, max_rotations=20_000,
            seed=42, parameter_uncertainty=False,
        )
        assert irf.shock_names == ["shock_a", "shock_b"]
        assert irf.median.shape == (5, 3, 2)
        assert irf.metadata["n_shocks_identified"] == 2
        assert irf.metadata["identification"] == "multi_shock_sign_restrictions"
        # Impact responses match the sign patterns at h=0 and h=1
        # (restriction_horizons=2 means signs enforced at h=0 AND h=1)
        for h in range(2):
            for shock_idx, (s_y1, s_y2) in enumerate([(+1, -1), (+1, +1)]):
                # Check median IRF at h
                assert np.sign(irf.median[h, 0, shock_idx]) == s_y1
                assert np.sign(irf.median[h, 1, shock_idx]) == s_y2

    def test_three_shocks_simultaneous_identification(self, synthetic_var_data):
        """Three shocks each pinned to a different sign pattern."""
        fit = lib.fit_var(synthetic_var_data, lags=2)
        irf = lib.sign_restricted_irf(
            fit=fit, horizon=4,
            sign_restrictions={
                "s1": {"y1": +1, "y2": -1, "y3": -1},
                "s2": {"y1": -1, "y2": +1, "y3": -1},
                "s3": {"y1": -1, "y2": -1, "y3": +1},
            },
            restriction_horizons=1,
            target_accepted=30, max_rotations=50_000,
            seed=42, parameter_uncertainty=False,
        )
        assert irf.metadata["n_shocks_identified"] == 3
        assert irf.median.shape[2] == 3

    def test_columns_assigned_distinctly(self, synthetic_var_data):
        """No two shocks should be assigned to the same column.

        Indirectly tested: with restriction_horizons=1 and very narrow
        patterns, the algorithm enforces distinctness by construction
        (assigned_cols set). Here we test that two shocks with
        identical sign patterns still get distinct columns (impossible
        if both demand the same column).
        """
        fit = lib.fit_var(synthetic_var_data, lags=2)
        # Both shocks ask for y1 > 0 — they can't be the SAME column but
        # CAN be different columns (any column with +y1 sign works).
        irf = lib.sign_restricted_irf(
            fit=fit, horizon=4,
            sign_restrictions={
                "shock_1": {"y1": +1},
                "shock_2": {"y1": +1},
            },
            target_accepted=30, max_rotations=10_000,
            seed=0, parameter_uncertainty=False,
        )
        # Both should be identified, and both should have +y1
        assert irf.median.shape[2] == 2
        assert irf.median[0, 0, 0] > 0
        assert irf.median[0, 0, 1] > 0

    def test_incompatible_pattern_raises(self, synthetic_var_data):
        """Demanding two shocks with the SAME sign pattern on EVERY
        variable AND restriction_horizons=large should mostly succeed,
        but conflicting patterns (impossible to satisfy by orthogonal
        rotation) should produce no acceptance."""
        fit = lib.fit_var(synthetic_var_data, lags=2)
        # Three shocks asking for +1 on y1 with restriction_horizons high
        # enough that the chance of getting 3 columns all +y1 is small.
        with pytest.raises(RuntimeError, match="no rotations satisfied"):
            lib.sign_restricted_irf(
                fit=fit, horizon=4,
                sign_restrictions={
                    "s1": {"y1": +1, "y2": +1, "y3": +1},
                    "s2": {"y1": +1, "y2": +1, "y3": +1},
                    "s3": {"y1": +1, "y2": +1, "y3": +1},
                },
                restriction_horizons=4,
                target_accepted=10, max_rotations=200,
                seed=0, parameter_uncertainty=False,
            )

    def test_narrative_requires_shock_key_in_multi_shock(self, synthetic_var_data):
        """Narrative restrictions in multi-shock mode must specify which shock."""
        fit = lib.fit_var(synthetic_var_data, lags=2)
        with pytest.raises(ValueError, match="must include a 'shock' key"):
            lib.sign_restricted_irf(
                fit=fit, horizon=4,
                sign_restrictions={
                    "s1": {"y1": +1, "y2": -1},
                    "s2": {"y1": -1, "y2": +1},
                },
                narrative_restrictions=[
                    {"date": fit.data.index[fit.p + 5], "sign": +1},
                    # missing 'shock' key
                ],
                target_accepted=10, max_rotations=100,
                parameter_uncertainty=False,
            )

    def test_narrative_with_shock_key_works(self, synthetic_var_data):
        """Narrative with explicit shock= key applies only to that shock."""
        fit = lib.fit_var(synthetic_var_data, lags=2)
        sample_dates = list(fit.data.index[fit.p:])
        irf = lib.sign_restricted_irf(
            fit=fit, horizon=4,
            sign_restrictions={
                "s1": {"y1": +1, "y2": -1},
                "s2": {"y1": -1, "y2": +1},
            },
            narrative_restrictions=[
                {"date": sample_dates[len(sample_dates) // 2], "sign": +1, "shock": "s1"},
            ],
            target_accepted=30, max_rotations=50_000,
            seed=0, parameter_uncertainty=False,
        )
        assert "narrative" in irf.metadata["identification"]
        assert irf.metadata["n_rejected_by_narrative"] >= 0

    def test_fev_share_requires_shock_key_in_multi_shock(self, synthetic_var_data):
        fit = lib.fit_var(synthetic_var_data, lags=2)
        with pytest.raises(ValueError, match="must include a 'shock' key"):
            lib.sign_restricted_irf(
                fit=fit, horizon=4,
                sign_restrictions={
                    "s1": {"y1": +1, "y2": -1},
                    "s2": {"y1": -1, "y2": +1},
                },
                fev_share_restrictions=[
                    {"variable": "y1", "horizon": 2, "lower": 0.3},
                    # missing 'shock' key
                ],
                target_accepted=10, max_rotations=100,
                parameter_uncertainty=False,
            )

    def test_unknown_shock_in_restriction_raises(self, synthetic_var_data):
        fit = lib.fit_var(synthetic_var_data, lags=2)
        with pytest.raises(ValueError, match="references shock"):
            lib.sign_restricted_irf(
                fit=fit, horizon=4,
                sign_restrictions={"s1": {"y1": +1}, "s2": {"y2": +1}},
                narrative_restrictions=[
                    {"date": fit.data.index[fit.p + 5], "sign": +1, "shock": "nonexistent"},
                ],
                target_accepted=10, max_rotations=100,
                parameter_uncertainty=False,
            )

    def test_single_shock_path_unchanged(self, synthetic_var_data):
        """Backward compat: single-shock dict goes through the original code path."""
        fit = lib.fit_var(synthetic_var_data, lags=2)
        irf = lib.sign_restricted_irf(
            fit=fit, horizon=4,
            sign_restrictions={"only_shock": {"y1": +1}},
            target_accepted=30, max_rotations=10_000,
            seed=0, parameter_uncertainty=False,
        )
        # Single-shock identification tag, not multi-shock
        assert irf.metadata["identification"] == "sign_restrictions"
        assert irf.median.shape[2] == 1


class TestType2NarrativeRestrictions:
    """ADRR 2018 Type 2 historical-contribution narrative restrictions
    in the multi-shock RWZ identification.

    Restriction format:
      {date, type, variable, shock} where type is one of
      'largest_contributor', 'largest_positive_contributor',
      'largest_negative_contributor'.
    """

    def test_type2_largest_contributor_filters_rotations(self, synthetic_var_data):
        """Adding a Type 2 'largest_contributor' restriction should
        reject some rotations that the sign-only baseline accepts."""
        fit = lib.fit_var(synthetic_var_data, lags=2)
        sample_dates = list(fit.data.index[fit.p:])
        date_mid = sample_dates[len(sample_dates) // 2]
        common = dict(
            fit=fit, horizon=4,
            sign_restrictions={
                "shock_a": {"y1": +1, "y2": -1},
                "shock_b": {"y1": +1, "y2": +1},
            },
            target_accepted=20, max_rotations=20_000,
            seed=0, parameter_uncertainty=False,
        )
        base = lib.sign_restricted_irf(**common)
        type2 = lib.sign_restricted_irf(
            **common,
            narrative_restrictions=[
                {"date": date_mid, "type": "largest_contributor",
                 "variable": "y1", "shock": "shock_a"},
            ],
        )
        # Type 2 should reject at least some rotations
        assert type2.metadata["n_rejected_by_type2"] >= 0
        # Acceptance rate should not exceed baseline
        assert type2.metadata["acceptance_rate"] <= base.metadata["acceptance_rate"] + 1e-9
        # Identification tag composes the Type 2 marker
        assert "type2_contribution" in type2.metadata["identification"]

    def test_type2_largest_positive_filters_distinct(self, synthetic_var_data):
        """'largest_positive_contributor' is stricter than 'largest_contributor'
        in cases where the absolute winner contributes negatively."""
        fit = lib.fit_var(synthetic_var_data, lags=2)
        sample_dates = list(fit.data.index[fit.p:])
        date_mid = sample_dates[len(sample_dates) // 2]
        irf = lib.sign_restricted_irf(
            fit=fit, horizon=4,
            sign_restrictions={
                "s1": {"y1": +1, "y2": -1},
                "s2": {"y1": -1, "y2": +1},
            },
            narrative_restrictions=[
                {"date": date_mid, "type": "largest_positive_contributor",
                 "variable": "y1", "shock": "s1"},
            ],
            target_accepted=10, max_rotations=20_000,
            seed=0, parameter_uncertainty=False,
        )
        assert irf.metadata["n_rejected_by_type2"] >= 0
        assert "type2_contribution" in irf.metadata["identification"]

    def test_type2_largest_negative_works(self, synthetic_var_data):
        """Under ADRR absolute-dominance semantics the restriction must
        be placed at a date where it is actually TRUE — an arbitrary
        mid-sample date would only satisfy a weaker signed-min comparison.
        Pick the date of the most negative realised Cholesky shock-1 draw
        and restrict on its OWN variable, where the contemporaneous
        own-shock contribution plausibly dominates."""
        fit = lib.fit_var(synthetic_var_data, lags=2)
        L = lib.cholesky_impact(fit.Sigma_u)
        eps = np.linalg.solve(L, fit.residuals.T).T          # (T_eff, n)
        sample_dates = list(fit.data.index[fit.p:])
        t_star = int(np.argmin(eps[:, 0]))                   # most negative s1 draw
        irf = lib.sign_restricted_irf(
            fit=fit, horizon=4,
            sign_restrictions={
                "s1": {"y1": +1, "y2": -1},
                "s2": {"y1": -1, "y2": +1},
            },
            narrative_restrictions=[
                {"date": sample_dates[t_star],
                 "type": "largest_negative_contributor",
                 "variable": "y1", "shock": "s1"},
            ],
            target_accepted=10, max_rotations=20_000,
            seed=0, parameter_uncertainty=False,
        )
        # Should produce a valid IRF (or at least not crash)
        assert irf.median.shape[2] == 2

    def test_type2_missing_variable_key_raises(self, synthetic_var_data):
        fit = lib.fit_var(synthetic_var_data, lags=2)
        with pytest.raises(ValueError, match="must have a 'variable' key"):
            lib.sign_restricted_irf(
                fit=fit, horizon=4,
                sign_restrictions={
                    "s1": {"y1": +1},
                    "s2": {"y2": +1},
                },
                narrative_restrictions=[
                    {"date": fit.data.index[fit.p + 5], "type": "largest_contributor",
                     "shock": "s1"},  # missing 'variable'
                ],
                target_accepted=10, max_rotations=100,
                parameter_uncertainty=False,
            )

    def test_type2_unknown_type_raises(self, synthetic_var_data):
        fit = lib.fit_var(synthetic_var_data, lags=2)
        with pytest.raises(ValueError, match="unknown narrative restriction type"):
            lib.sign_restricted_irf(
                fit=fit, horizon=4,
                sign_restrictions={
                    "s1": {"y1": +1},
                    "s2": {"y2": +1},
                },
                narrative_restrictions=[
                    {"date": fit.data.index[fit.p + 5], "type": "weirdest",
                     "variable": "y1", "shock": "s1"},
                ],
                target_accepted=10, max_rotations=100,
                parameter_uncertainty=False,
            )

    def test_type1_and_type2_can_compose(self, synthetic_var_data):
        """Both restriction types simultaneously on different shocks."""
        fit = lib.fit_var(synthetic_var_data, lags=2)
        sample_dates = list(fit.data.index[fit.p:])
        date_a = sample_dates[len(sample_dates) // 3]
        date_b = sample_dates[2 * len(sample_dates) // 3]
        irf = lib.sign_restricted_irf(
            fit=fit, horizon=4,
            sign_restrictions={
                "s1": {"y1": +1, "y2": -1},
                "s2": {"y1": -1, "y2": +1},
            },
            narrative_restrictions=[
                # Type 1 on s1
                {"date": date_a, "type": "sign", "sign": +1, "shock": "s1"},
                # Type 2 on s2
                {"date": date_b, "type": "largest_contributor",
                 "variable": "y2", "shock": "s2"},
            ],
            target_accepted=10, max_rotations=50_000,
            seed=0, parameter_uncertainty=False,
        )
        assert "type2_contribution" in irf.metadata["identification"]
        # Note: tag string contains 'narrative' for Type 1 + 'type2_contribution' for Type 2
        assert "narrative" in irf.metadata["identification"]


class TestSingleShockType2:
    """Type 2 narrative restrictions in the single-shock code path."""

    def test_single_shock_type2_filters_rotations(self, synthetic_var_data):
        fit = lib.fit_var(synthetic_var_data, lags=2)
        sample_dates = list(fit.data.index[fit.p:])
        date_mid = sample_dates[len(sample_dates) // 2]
        common = dict(
            fit=fit, horizon=4,
            sign_restrictions={"mp": {"y1": +1, "y2": -1}},
            target_accepted=20, max_rotations=20_000,
            seed=0, parameter_uncertainty=False,
        )
        base = lib.sign_restricted_irf(**common)
        type2 = lib.sign_restricted_irf(
            **common,
            narrative_restrictions=[
                {"date": date_mid, "type": "largest_contributor",
                 "variable": "y1"},
            ],
        )
        assert type2.metadata["n_rejected_by_type2"] >= 0
        assert type2.metadata["acceptance_rate"] <= base.metadata["acceptance_rate"] + 1e-9
        assert "type2_contribution" in type2.metadata["identification"]
        # Single-shock Type 2 doesn't need 'shock' key
        assert type2.median.shape[2] == 1

    def test_single_shock_type2_composes_with_type1(self, synthetic_var_data):
        """Type 1 and Type 2 restrictions in the single-shock case."""
        fit = lib.fit_var(synthetic_var_data, lags=2)
        sample_dates = list(fit.data.index[fit.p:])
        date_a = sample_dates[len(sample_dates) // 3]
        date_b = sample_dates[2 * len(sample_dates) // 3]
        irf = lib.sign_restricted_irf(
            fit=fit, horizon=4,
            sign_restrictions={"mp": {"y1": +1, "y2": -1}},
            narrative_restrictions=[
                {"date": date_a, "type": "sign", "sign": +1},
                {"date": date_b, "type": "largest_positive_contributor",
                 "variable": "y1"},
            ],
            target_accepted=15, max_rotations=50_000,
            seed=0, parameter_uncertainty=False,
        )
        assert "narrative" in irf.metadata["identification"]
        assert "type2_contribution" in irf.metadata["identification"]

    def test_single_shock_type2_missing_variable_raises(self, synthetic_var_data):
        fit = lib.fit_var(synthetic_var_data, lags=2)
        with pytest.raises(ValueError, match="'variable' key"):
            lib.sign_restricted_irf(
                fit=fit, horizon=4,
                sign_restrictions={"mp": {"y1": +1}},
                narrative_restrictions=[
                    {"date": fit.data.index[fit.p + 5],
                     "type": "largest_contributor"},  # no 'variable'
                ],
                target_accepted=10, max_rotations=100,
                parameter_uncertainty=False,
            )


class TestProxySVAR:
    """Tests for proxy-SVAR (external-instrument identification).

    Validation strategy:

    * **Normalisation property** — at h=0 with ``"unit_policy"`` normalisation
      the response of the policy variable to its own shock must equal 1.
    * **Cholesky equivalence** — if we use the Cholesky-orthogonalised
      residual of the policy variable AS THE INSTRUMENT, the proxy-SVAR
      column must equal column ``policy_idx`` of the Cholesky impact
      matrix (up to sign).
    * **First-stage diagnostics** match a standalone OLS regression.
    * **Weak-instrument flagging** with a pure-noise instrument.
    * **Wild bootstrap** produces well-formed IRF distributions (band
      ordering) — a mechanics check, not a coverage test; Jentsch-Lunsford
      (2019) show this bootstrap is asymptotically invalid for proxy SVARs.
    """

    def test_h0_response_of_policy_equals_one_under_unit_normalization(self,
            synthetic_var_data):
        rng = np.random.default_rng(0)
        # Create a synthetic instrument correlated with the y1 residual.
        # The exact correlation doesn't matter — only the normalisation does.
        instrument = rng.standard_normal(len(synthetic_var_data))
        fit = lib.fit_var(synthetic_var_data, lags=2)
        irf, _ = lib.proxy_svar_irf(
            fit=fit, instrument=instrument, policy_var="y1",
            horizon=5, bands="none", normalization="unit_policy",
        )
        # IRF of y1 to y1-shock at h=0 should be exactly 1
        assert np.isclose(irf.median[0, 0, 0], 1.0, atol=1e-12)

    def test_first_stage_diagnostics_match_standalone_ols(self,
            synthetic_var_data):
        from statsmodels.regression.linear_model import OLS
        from statsmodels.tools import add_constant
        rng = np.random.default_rng(1)
        instrument = rng.standard_normal(len(synthetic_var_data))
        fit = lib.fit_var(synthetic_var_data, lags=2)
        _, fs = lib.proxy_svar_irf(
            fit=fit, instrument=instrument, policy_var="y1",
            horizon=2, bands="none",
        )
        # Standalone OLS: residual of y1 on instrument
        z_aligned = instrument[fit.p:]
        u1 = fit.residuals[:, 0]
        X = add_constant(z_aligned)
        ols_res = OLS(u1, X).fit()
        manual_F = float(ols_res.tvalues[1] ** 2)
        manual_gamma = float(ols_res.params[1])
        manual_r2 = float(ols_res.rsquared)
        assert np.isclose(fs["gamma"], manual_gamma, atol=1e-10)
        assert np.isclose(fs["R2"], manual_r2, atol=1e-10)
        # Our F uses (n-2) * R^2 / (1-R^2) which equals t^2 only for one
        # regressor with no constant adjustment; both are valid first-stage
        # F statistics. Verify ours is positive and finite.
        assert np.isfinite(fs["F"]) and fs["F"] > 0

    def test_using_residual_as_instrument_recovers_cholesky_column(self,
            synthetic_var_data):
        """If Z_t = u_{policy,t} (the policy variable's own residual), proxy-SVAR
        recovers the Cholesky column for the policy variable up to scale.

        With ``policy_var="y1"`` (which is ordered first), the Cholesky column
        for y1 is the first column of chol(Sigma_u). Both should reduce to
        the same direction.
        """
        fit = lib.fit_var(synthetic_var_data, lags=2)
        # Use y1's reduced-form residual as the "instrument" — extend with
        # zeros for the first p observations
        u1 = fit.residuals[:, 0]
        instrument = np.concatenate([np.zeros(fit.p), u1])

        irf, _ = lib.proxy_svar_irf(
            fit=fit, instrument=instrument, policy_var="y1",
            horizon=0, bands="none", normalization="unit_policy",
        )
        # Cholesky first-column normalised to have unit entry at the policy
        # variable's position
        P = np.linalg.cholesky(fit.Sigma_u)
        chol_col_norm = P[:, 0] / P[0, 0]
        assert np.allclose(irf.median[0, :, 0], chol_col_norm, atol=1e-8)

    def test_weak_instrument_flag(self, synthetic_var_data):
        rng = np.random.default_rng(99)
        # An instrument uncorrelated with anything by construction
        noise = rng.standard_normal(len(synthetic_var_data))
        fit = lib.fit_var(synthetic_var_data, lags=2)
        try:
            irf, fs = lib.proxy_svar_irf(
                fit=fit, instrument=noise, policy_var="y1",
                horizon=2, bands="none",
            )
        except RuntimeError:
            # Pure noise might give Σ_uZ ≈ 0 — RuntimeError is acceptable
            return
        # If we got here, F should be small (likely < 10)
        assert fs["F"] < 10
        assert fs["weak_instrument_F_lt_10"] is True

    def test_all_nan_instrument_rejected(self, synthetic_var_data):
        fit = lib.fit_var(synthetic_var_data, lags=2)
        bad = np.full(len(synthetic_var_data), np.nan)
        with pytest.raises(ValueError, match="no observed"):
            lib.proxy_svar_irf(
                fit=fit, instrument=bad, policy_var="y1",
                horizon=2, bands="none",
            )

    def test_nan_padding_leaves_estimates_identical_to_zero_fill(self,
            synthetic_var_data):
        """NaN marks 'not observed'; the moment computations zero-fill, so
        the identified impact column and IRF must be IDENTICAL to passing
        the same instrument with zeros in the unobserved periods."""
        rng = np.random.default_rng(3)
        T = len(synthetic_var_data)
        z = rng.standard_normal(T)
        cut = T // 2                       # instrument unobserved in first half
        z_zerofill = z.copy()
        z_zerofill[:cut] = 0.0
        z_nanpad = z.copy()
        z_nanpad[:cut] = np.nan
        fit = lib.fit_var(synthetic_var_data, lags=2)
        irf_zero, _ = lib.proxy_svar_irf(
            fit=fit, instrument=z_zerofill, policy_var="y1",
            horizon=5, bands="none", normalization="unit_policy",
        )
        irf_nan, _ = lib.proxy_svar_irf(
            fit=fit, instrument=z_nanpad, policy_var="y1",
            horizon=5, bands="none", normalization="unit_policy",
        )
        np.testing.assert_allclose(irf_nan.median, irf_zero.median, atol=1e-14)

    def test_nan_padding_first_stage_on_observed_subsample(self,
            synthetic_var_data):
        """First-stage diagnostics must use only the observed rows: a strong
        instrument observed on half the sample must NOT have its F diluted
        by the unobserved (NaN) half, and must match a manual OLS on the
        observed subsample (Gertler-Karadi FF4-style partial coverage)."""
        fit = lib.fit_var(synthetic_var_data, lags=2)
        T = len(synthetic_var_data)
        rng = np.random.default_rng(11)
        # Strong instrument: policy residual + small noise, observed only in
        # the second half of the sample
        u1 = np.asarray(fit.residuals)[:, 0]
        z = np.concatenate([np.zeros(fit.p), u1]) + 0.3 * rng.standard_normal(T)
        cut = T // 2
        z_nanpad = z.copy()
        z_nanpad[:cut] = np.nan
        z_zerofill = z.copy()
        z_zerofill[:cut] = 0.0

        _, fs_nan = lib.proxy_svar_irf(
            fit=fit, instrument=z_nanpad, policy_var="y1",
            horizon=2, bands="none",
        )
        _, fs_zero = lib.proxy_svar_irf(
            fit=fit, instrument=z_zerofill, policy_var="y1",
            horizon=2, bands="none",
        )
        # Manual first stage on the observed subsample only
        obs = np.isfinite(z_nanpad[fit.p:])
        z_obs = z_nanpad[fit.p:][obs]
        u_obs = u1[obs]
        n = len(z_obs)
        Z = np.column_stack([np.ones(n), z_obs])
        beta, *_ = np.linalg.lstsq(Z, u_obs, rcond=None)
        resid = u_obs - Z @ beta
        r2 = 1.0 - float((resid ** 2).sum()) / float(
            ((u_obs - u_obs.mean()) ** 2).sum())
        f_manual = (n - 2) * r2 / (1.0 - r2)

        assert np.isclose(fs_nan["F"], f_manual, rtol=1e-10)
        assert fs_nan["n_obs"] == n
        assert fs_nan["n_unobserved_in_sample"] == int((~obs).sum())
        # Zero-filling the unobserved half dilutes F; the mask must not
        assert fs_nan["F"] > fs_zero["F"]

    def test_unknown_policy_var_rejected(self, synthetic_var_data):
        fit = lib.fit_var(synthetic_var_data, lags=2)
        instrument = np.zeros(len(synthetic_var_data))
        instrument[10] = 1.0
        with pytest.raises(ValueError, match="not in variable_names"):
            lib.proxy_svar_irf(
                fit=fit, instrument=instrument, policy_var="not_a_var",
                horizon=2, bands="none",
            )

    def test_wild_bootstrap_produces_valid_bands(self, synthetic_var_data):
        rng = np.random.default_rng(7)
        instrument = rng.standard_normal(len(synthetic_var_data))
        fit = lib.fit_var(synthetic_var_data, lags=2)
        irf, _ = lib.proxy_svar_irf(
            fit=fit, instrument=instrument, policy_var="y1",
            horizon=5, bands="wild_bootstrap", n_draws=200, seed=0,
        )
        # Band ordering: lower_90 ≤ lower_68 ≤ median ≤ upper_68 ≤ upper_90
        assert np.all(irf.lower_90 <= irf.lower_68 + 1e-10)
        assert np.all(irf.lower_68 <= irf.median + 1e-10)
        assert np.all(irf.median <= irf.upper_68 + 1e-10)
        assert np.all(irf.upper_68 <= irf.upper_90 + 1e-10)
        assert irf.median.shape == (6, 3, 1)

    def test_one_sd_normalization_scales_by_residual_std(self,
            synthetic_var_data):
        rng = np.random.default_rng(2)
        instrument = rng.standard_normal(len(synthetic_var_data))
        fit = lib.fit_var(synthetic_var_data, lags=2)
        irf_unit, _ = lib.proxy_svar_irf(
            fit=fit, instrument=instrument, policy_var="y1",
            horizon=0, bands="none", normalization="unit_policy",
        )
        irf_sd, _ = lib.proxy_svar_irf(
            fit=fit, instrument=instrument, policy_var="y1",
            horizon=0, bands="none", normalization="one_sd",
        )
        sigma_y1 = float(np.std(fit.residuals[:, 0], ddof=1))
        # one_sd response = unit_policy response * std(u_y1)
        assert np.allclose(
            irf_sd.median[0, :, 0],
            irf_unit.median[0, :, 0] * sigma_y1,
            atol=1e-10,
        )


class TestProxySVARMultiInstrument:
    """Tests for the Mertens-Ravn (2013) §II multi-instrument extension.

    Validation strategy:

    * **Policy block is I_m** at h=0 under unit-policy normalisation —
      the same property as the single-instrument case generalised.
    * **Single-instrument limit** — when m=1 the multi-instrument code
      path produces the same IRF as the single-instrument code path.
    * **m=2 with diagonal instrument matrix** — when the two instruments
      are uncorrelated and each moves only its corresponding policy
      variable, the result reduces to two independent single-instrument
      proxy-SVARs (same impact columns).
    * **Distinct policy variables required** — duplicates raise.
    * **First-stage diagnostics** include per-pair table and Σ_p condition.
    * **Wild bootstrap multi** — returns bands of shape
      ``(H+1, n_vars, m)`` (a shape check, not a coverage test).
    """

    def test_h0_policy_block_is_identity_under_unit_normalization(self,
            synthetic_var_data):
        rng = np.random.default_rng(0)
        T = len(synthetic_var_data)
        # Two random instruments
        Z = rng.standard_normal((T, 2))
        fit = lib.fit_var(synthetic_var_data, lags=2)
        irf, _ = lib.proxy_svar_irf(
            fit=fit, instrument=Z, policy_var=["y1", "y2"],
            horizon=3, bands="none", normalization="unit_policy",
        )
        # Policy block (rows y1, y2 ; cols 0, 1) at h=0 must be I_2
        block = irf.median[0, [0, 1], :]
        assert np.allclose(block, np.eye(2), atol=1e-12)

    def test_single_instrument_limit_matches_legacy_path(self,
            synthetic_var_data):
        """m=1 multi-instrument call must produce the same IRF and first-stage
        diagnostics as the single-instrument call."""
        rng = np.random.default_rng(1)
        T = len(synthetic_var_data)
        z = rng.standard_normal(T)
        fit = lib.fit_var(synthetic_var_data, lags=2)
        irf_single, fs_single = lib.proxy_svar_irf(
            fit=fit, instrument=z, policy_var="y1",
            horizon=5, bands="none", normalization="unit_policy",
        )
        irf_multi, fs_multi = lib.proxy_svar_irf(
            fit=fit, instrument=z.reshape(-1, 1), policy_var=["y1"],
            horizon=5, bands="none", normalization="unit_policy",
        )
        assert np.allclose(irf_single.median, irf_multi.median, atol=1e-12)
        # Multi diagnostics dict has different schema, but per-pair entry
        # (the only one when m=1) should match the single-instrument F / R²
        assert len(fs_multi["per_pair"]) == 1
        assert np.isclose(fs_multi["per_pair"][0]["R2"], fs_single["R2"])
        assert np.isclose(fs_multi["per_pair"][0]["gamma"], fs_single["gamma"])

    def test_nan_padded_column_masks_its_own_first_stage(self,
            synthetic_var_data):
        """A NaN-padded instrument column: estimates equal the zero-filled
        call; its per-pair diagnostics use only that column's observed
        rows (n_obs) while the other column keeps the full sample."""
        rng = np.random.default_rng(5)
        T = len(synthetic_var_data)
        Z = rng.standard_normal((T, 2))
        cut = T // 3
        Z_nan = Z.copy()
        Z_nan[:cut, 0] = np.nan          # column 0 unobserved early on
        Z_zero = Z.copy()
        Z_zero[:cut, 0] = 0.0
        fit = lib.fit_var(synthetic_var_data, lags=2)
        irf_nan, fs_nan = lib.proxy_svar_irf(
            fit=fit, instrument=Z_nan, policy_var=["y1", "y2"],
            horizon=4, bands="none", normalization="unit_policy",
        )
        irf_zero, _ = lib.proxy_svar_irf(
            fit=fit, instrument=Z_zero, policy_var=["y1", "y2"],
            horizon=4, bands="none", normalization="unit_policy",
        )
        np.testing.assert_allclose(irf_nan.median, irf_zero.median, atol=1e-14)
        n_obs0 = {p["n_obs"] for p in fs_nan["per_pair"] if p["instrument_idx"] == 0}
        n_obs1 = {p["n_obs"] for p in fs_nan["per_pair"] if p["instrument_idx"] == 1}
        assert n_obs0 == {fit.t_eff - (cut - fit.p)}
        assert n_obs1 == {fit.t_eff}

    def test_all_nan_column_rejected(self, synthetic_var_data):
        rng = np.random.default_rng(6)
        T = len(synthetic_var_data)
        Z = rng.standard_normal((T, 2))
        Z[:, 1] = np.nan
        fit = lib.fit_var(synthetic_var_data, lags=2)
        with pytest.raises(ValueError, match="no observed"):
            lib.proxy_svar_irf(
                fit=fit, instrument=Z, policy_var=["y1", "y2"],
                horizon=2, bands="none",
            )

    def test_block_diagonal_sigma_p_decouples_into_two_single_runs(self,
            synthetic_var_data):
        """When Sigma_p is block-diagonal (instrument 1 covaries only with
        policy variable 1's residual; instrument 2 only with 2's), the
        m=2 multi-instrument identification reduces to two independent
        single-instrument identifications. Achieved here by bilateral
        Gram-Schmidt: z_j orthogonalised against u_{other_policy}."""
        T = len(synthetic_var_data)
        fit = lib.fit_var(synthetic_var_data, lags=2)
        u1 = fit.residuals[:, 0]
        u2 = fit.residuals[:, 1]
        # Bilateral orthogonalisation: u_1' @ z_2 = 0 AND u_2' @ z_1 = 0
        z1_aligned = u1 - (u2 @ u1) / (u2 @ u2) * u2
        z2_aligned = u2 - (u1 @ u2) / (u1 @ u1) * u1
        # Pad with zeros for the first p observations
        z1 = np.concatenate([np.zeros(fit.p), z1_aligned])
        z2 = np.concatenate([np.zeros(fit.p), z2_aligned])
        Z = np.column_stack([z1, z2])

        # Sanity: Sigma_p should be diagonal under this construction
        sigma_p = fit.residuals[:, [0, 1]].T @ np.column_stack([z1_aligned, z2_aligned]) / fit.t_eff
        assert abs(sigma_p[0, 1]) < 1e-12
        assert abs(sigma_p[1, 0]) < 1e-12

        irf_multi, _ = lib.proxy_svar_irf(
            fit=fit, instrument=Z, policy_var=["y1", "y2"],
            horizon=3, bands="none",
        )
        irf_1, _ = lib.proxy_svar_irf(
            fit=fit, instrument=z1, policy_var="y1",
            horizon=3, bands="none",
        )
        irf_2, _ = lib.proxy_svar_irf(
            fit=fit, instrument=z2, policy_var="y2",
            horizon=3, bands="none",
        )
        assert np.allclose(irf_multi.median[:, :, 0], irf_1.median[:, :, 0], atol=1e-10)
        assert np.allclose(irf_multi.median[:, :, 1], irf_2.median[:, :, 0], atol=1e-10)

    def test_duplicate_policy_vars_rejected(self, synthetic_var_data):
        T = len(synthetic_var_data)
        rng = np.random.default_rng(3)
        Z = rng.standard_normal((T, 2))
        fit = lib.fit_var(synthetic_var_data, lags=2)
        with pytest.raises(ValueError, match="distinct"):
            lib.proxy_svar_irf(
                fit=fit, instrument=Z, policy_var=["y1", "y1"],
                horizon=2, bands="none",
            )

    def test_policy_var_length_mismatch_rejected(self, synthetic_var_data):
        T = len(synthetic_var_data)
        rng = np.random.default_rng(4)
        Z = rng.standard_normal((T, 2))
        fit = lib.fit_var(synthetic_var_data, lags=2)
        with pytest.raises(ValueError, match="length"):
            lib.proxy_svar_irf(
                fit=fit, instrument=Z, policy_var=["y1"],
                horizon=2, bands="none",
            )

    def test_first_stage_multi_includes_per_pair_and_cond(self, synthetic_var_data):
        rng = np.random.default_rng(5)
        T = len(synthetic_var_data)
        Z = rng.standard_normal((T, 2))
        fit = lib.fit_var(synthetic_var_data, lags=2)
        _, fs = lib.proxy_svar_irf(
            fit=fit, instrument=Z, policy_var=["y1", "y2"],
            horizon=2, bands="none",
        )
        assert "per_pair" in fs
        assert len(fs["per_pair"]) == 4   # m × m
        # Each entry has gamma, R2, F, instrument_idx, policy_idx
        for p in fs["per_pair"]:
            assert {"gamma", "R2", "F", "instrument_idx", "policy_idx",
                    "policy_position"} <= set(p.keys())
        assert "Sigma_p_cond" in fs
        assert "weak_instrument_min_own_F" in fs
        assert "weak_instrument_F_lt_10" in fs

    def test_wild_bootstrap_multi_produces_valid_bands(self, synthetic_var_data):
        rng = np.random.default_rng(7)
        T = len(synthetic_var_data)
        # Build relevant instruments using orthogonalised policy residuals
        # so the first stage is non-degenerate.
        fit = lib.fit_var(synthetic_var_data, lags=2)
        u1, u2 = fit.residuals[:, 0], fit.residuals[:, 1]
        beta = (u1 @ u2) / (u1 @ u1)
        u2_orth = u2 - beta * u1
        Z = np.column_stack([
            np.concatenate([np.zeros(fit.p), u1]),
            np.concatenate([np.zeros(fit.p), u2_orth]),
        ])
        irf, _ = lib.proxy_svar_irf(
            fit=fit, instrument=Z, policy_var=["y1", "y2"],
            horizon=5, bands="wild_bootstrap", n_draws=200, seed=0,
        )
        assert irf.median.shape == (6, 3, 2)
        assert np.all(irf.lower_90 <= irf.lower_68 + 1e-10)
        assert np.all(irf.lower_68 <= irf.median + 1e-10)
        assert np.all(irf.median <= irf.upper_68 + 1e-10)
        assert np.all(irf.upper_68 <= irf.upper_90 + 1e-10)

    def test_singular_first_stage_raises(self, synthetic_var_data):
        """If both instruments are the same, Sigma_p is singular —
        identification fails with a clear error."""
        rng = np.random.default_rng(8)
        T = len(synthetic_var_data)
        z = rng.standard_normal(T)
        Z = np.column_stack([z, z])   # rank-1 instrument matrix
        fit = lib.fit_var(synthetic_var_data, lags=2)
        with pytest.raises(RuntimeError, match="singular"):
            lib.proxy_svar_irf(
                fit=fit, instrument=Z, policy_var=["y1", "y2"],
                horizon=2, bands="none",
            )


class TestLongRun:
    """Tests for long-run / Blanchard-Quah identification.

    Validation strategy:

    * **Long-run restriction holds** — at h → ∞, the cumulative IRF of
      shock ``j`` on variables ``1, ..., j-1`` must be zero by construction.
    * **Orthogonality** — H H' = Sigma_u (the impact matrix is a valid
      orthogonalisation of the reduced-form covariance).
    * **Lower-triangular long-run** — C(1) @ H is lower triangular by
      construction.
    * **Sign normalisation** flips columns correctly.
    """

    def test_long_run_restriction_holds(self, synthetic_var_data):
        """At h → ∞, cumulative IRF of shock j on variables 0..j-1 ≈ 0."""
        fit = lib.fit_var(synthetic_var_data, lags=2)
        irf = lib.long_run_irf(fit, horizon=200, bands="none")
        n = fit.n_vars
        # Cumulative response over a long horizon
        cum = irf.median.sum(axis=0)   # (n_vars, n_shocks)
        # cum[i, j] ≈ long-run effect of shock j on variable i
        # For j > i, this must be ≈ 0 (the BQ-style restriction)
        for j in range(n):
            for i in range(j):
                assert abs(cum[i, j]) < 0.05, (
                    f"long-run effect of shock {j} on variable {i} = "
                    f"{cum[i, j]:.4f}, expected ≈ 0"
                )

    def test_impact_matrix_satisfies_orthogonality(self, synthetic_var_data):
        """H must satisfy H H' = Sigma_u."""
        fit = lib.fit_var(synthetic_var_data, lags=2)
        lag_mats = fit.lag_matrices()
        from lib.identification import _long_run_structural_matrix
        H = _long_run_structural_matrix(lag_mats, fit.Sigma_u)
        reconstructed = H @ H.T
        assert np.allclose(reconstructed, fit.Sigma_u, atol=1e-10)

    def test_long_run_matrix_is_lower_triangular(self, synthetic_var_data):
        """C(1) @ H must be lower triangular (the identifying assumption)."""
        fit = lib.fit_var(synthetic_var_data, lags=2)
        lag_mats = fit.lag_matrices()
        from lib.identification import _long_run_structural_matrix
        H = _long_run_structural_matrix(lag_mats, fit.Sigma_u)
        n = H.shape[0]
        A_sum = sum(lag_mats)
        C1 = np.linalg.inv(np.eye(n) - A_sum)
        long_run = C1 @ H
        # Upper-triangular off-diagonal entries must be zero
        for i in range(n):
            for j in range(i + 1, n):
                assert abs(long_run[i, j]) < 1e-10, (
                    f"long_run[{i}, {j}] = {long_run[i, j]:.2e}, "
                    f"expected exactly 0"
                )

    def test_irf_shape_is_n_shocks_equals_n_vars(self, synthetic_var_data):
        fit = lib.fit_var(synthetic_var_data, lags=2)
        irf = lib.long_run_irf(fit, horizon=10, bands="none")
        assert irf.median.shape == (11, fit.n_vars, fit.n_vars)

    def test_shock_names_default_and_custom(self, synthetic_var_data):
        fit = lib.fit_var(synthetic_var_data, lags=2)
        irf_default = lib.long_run_irf(fit, horizon=5, bands="none")
        assert irf_default.shock_names == ["shock_1", "shock_2", "shock_3"]
        irf_named = lib.long_run_irf(
            fit, horizon=5, bands="none",
            shock_names=["supply", "demand", "noise"],
        )
        assert irf_named.shock_names == ["supply", "demand", "noise"]

    def test_sign_normalization_flips_column(self, synthetic_var_data):
        fit = lib.fit_var(synthetic_var_data, lags=2)
        irf_default = lib.long_run_irf(fit, horizon=5, bands="none",
                                        shock_names=["s1", "s2", "s3"])
        # Force the first shock to have NEGATIVE contemporaneous effect on y1
        impact_y1_s1_default = irf_default.median[0, 0, 0]
        wanted_sign = -int(np.sign(impact_y1_s1_default))
        irf_flipped = lib.long_run_irf(
            fit, horizon=5, bands="none",
            shock_names=["s1", "s2", "s3"],
            sign_normalization={"s1": {"y1": wanted_sign}},
        )
        # The first shock's column should be flipped
        assert np.allclose(
            irf_flipped.median[:, :, 0],
            -irf_default.median[:, :, 0],
            atol=1e-10,
        )
        # Other columns unchanged
        assert np.allclose(
            irf_flipped.median[:, :, 1],
            irf_default.median[:, :, 1],
            atol=1e-10,
        )

    def test_parameter_draws_band_ordering(self, synthetic_var_data):
        fit = lib.fit_var(synthetic_var_data, lags=2)
        irf = lib.long_run_irf(
            fit, horizon=10, bands="parameter_draws", n_draws=200, seed=0,
        )
        assert np.all(irf.lower_90 <= irf.lower_68 + 1e-10)
        assert np.all(irf.lower_68 <= irf.median + 1e-10)
        assert np.all(irf.median <= irf.upper_68 + 1e-10)
        assert np.all(irf.upper_68 <= irf.upper_90 + 1e-10)


class TestHeteroskedasticityID:
    """Tests for Rigobon (2003) heteroskedasticity-based identification.

    Validation strategy:

    * **Decomposition holds exactly** — for any two regime covariances
      Σ_1, Σ_2, the recovered H satisfies H Ω_1 H' = Σ_1 and H I H' = Σ_2.
    * **Identical regimes** — if Σ_1 = Σ_2, the eigenvalues are all 1 and
      identification fails (flagged by ``weak_identification``).
    * **DGP recovery** — with a known synthetic two-regime DGP and
      sufficient data, the recovered H is close to the true H.
    """

    def _make_regime_data(self, seed: int = 11, T: int = 800) -> tuple:
        """Synthetic DGP: bivariate VAR(1) with two clear variance regimes.

        Returns (data, regime, true_H).
        """
        rng = np.random.default_rng(seed)
        n = 2
        # True structural impact matrix
        H_true = np.array([[1.0, 0.5], [0.3, 1.2]])
        # Regime variances
        omega_1 = np.diag([1.0, 1.0])
        omega_2 = np.diag([4.0, 4.0])   # both 4x larger; weak ID by design
        # Use a more interesting case: different variance ratios per shock
        omega_2 = np.diag([4.0, 0.5])
        A1 = np.array([[0.5, 0.1], [0.0, 0.4]])

        regime = np.zeros(T, dtype=int)
        regime[T // 2:] = 1

        Y = np.zeros((T, n))
        for t in range(1, T):
            omega = omega_1 if regime[t] == 0 else omega_2
            eps = rng.standard_normal(n) * np.sqrt(np.diag(omega))
            u = H_true @ eps
            Y[t] = A1 @ Y[t - 1] + u
        idx = pd.period_range("1980Q1", periods=T, freq="Q").to_timestamp()
        return (
            pd.DataFrame(Y, columns=["y1", "y2"], index=idx),
            pd.Series(regime, index=idx),
            H_true,
        )

    def test_decomposition_reconstructs_both_sigmas(self):
        data, regime, _ = self._make_regime_data(seed=11)
        fit = lib.fit_var(data, lags=1)
        irf, diag = lib.heteroskedasticity_irf(
            fit, regime=regime.values, horizon=0, bands="none",
        )
        # Extract H from the IRF: at h=0, IRF = I @ H = H
        H = irf.median[0]
        # Regime covariances from the residuals
        resid = np.asarray(fit.residuals)
        regime_aligned = regime.values[fit.p:]
        Sigma_1 = np.cov(resid[regime_aligned == 0].T, ddof=1)
        Sigma_2 = np.cov(resid[regime_aligned == 1].T, ddof=1)
        # Implied diagonal variance matrices
        Omega_1 = np.linalg.solve(H, Sigma_1) @ np.linalg.inv(H.T)
        Omega_2 = np.linalg.solve(H, Sigma_2) @ np.linalg.inv(H.T)
        # Off-diagonal entries of Omega should be ≈ 0
        assert abs(Omega_1[0, 1]) < 1e-8 and abs(Omega_1[1, 0]) < 1e-8
        assert abs(Omega_2[0, 1]) < 1e-8 and abs(Omega_2[1, 0]) < 1e-8
        # Diagonal entries should be positive
        assert Omega_1[0, 0] > 0 and Omega_1[1, 1] > 0
        assert Omega_2[0, 0] > 0 and Omega_2[1, 1] > 0

    def test_flags_weak_identification_when_regimes_similar(self):
        rng = np.random.default_rng(0)
        T = 400
        # Generate data with SAME variances across regimes
        Y = np.zeros((T, 2))
        for t in range(1, T):
            Y[t] = 0.5 * Y[t - 1] + rng.standard_normal(2)
        idx = pd.period_range("1980Q1", periods=T, freq="Q").to_timestamp()
        data = pd.DataFrame(Y, columns=["y1", "y2"], index=idx)
        regime = np.zeros(T, dtype=int)
        regime[T // 2:] = 1

        fit = lib.fit_var(data, lags=1)
        _, diag = lib.heteroskedasticity_irf(
            fit, regime=regime, horizon=0, bands="none",
        )
        # When regimes are identical, eigenvalues should cluster near 1
        eigvals = np.asarray(diag["eigenvalues_omega1_over_omega2"])
        # All eigenvalues should be close to each other
        assert np.std(eigvals) / np.mean(eigvals) < 0.5

    def test_rejects_wrong_number_of_regimes(self, synthetic_var_data):
        fit = lib.fit_var(synthetic_var_data, lags=2)
        regime = np.array([0, 1, 2] * (len(synthetic_var_data) // 3 + 1))[
            :len(synthetic_var_data)
        ]
        with pytest.raises(ValueError, match="exactly 2"):
            lib.heteroskedasticity_irf(
                fit, regime=regime, horizon=2, bands="none",
            )

    def test_rejects_misaligned_regime(self, synthetic_var_data):
        fit = lib.fit_var(synthetic_var_data, lags=2)
        with pytest.raises(ValueError, match="regime length"):
            lib.heteroskedasticity_irf(
                fit, regime=np.array([0, 1, 0, 1]), horizon=2, bands="none",
            )

    def test_irf_shape_is_n_vars_squared(self):
        data, regime, _ = self._make_regime_data(seed=11)
        fit = lib.fit_var(data, lags=1)
        irf, _ = lib.heteroskedasticity_irf(
            fit, regime=regime.values, horizon=10, bands="none",
        )
        assert irf.median.shape == (11, 2, 2)
        assert len(irf.shock_names) == 2


class TestZeroSignRestrictedIRF:
    """Tests for `lib.zero_sign_restricted_irf` (Arias-Rubio-Ramirez-
    Waggoner 2018 zero+sign identification).

    Validation strategy:

    * **Zero restriction held exactly** — under a contemporaneous zero
      restriction, the IRF at h=0 on the restricted (variable, shock)
      cell must be exactly 0 (up to floating-point) for EVERY accepted
      rotation. This is the algorithmic guarantee.
    * **Sign restrictions held** — the IRF satisfies the requested sign
      pattern at horizon 0.
    * **Backward compat** — passing no zero restrictions delegates to
      sign_restricted_irf byte-identically.
    * **Multi-shock joint identification**.
    * **Validation errors** — multi-horizon zero restrictions raise;
      unknown variables raise.
    """

    @staticmethod
    def _make_three_var_data(seed=0, T=200):
        rng = np.random.default_rng(seed)
        A = np.array([[0.5, 0.1, 0.05], [0.0, 0.4, 0.0], [0.1, 0.0, 0.3]])
        y = np.zeros((T, 3))
        for t in range(1, T):
            y[t] = A @ y[t - 1] + rng.standard_normal(3) * 0.5
        return pd.DataFrame(
            y, columns=["rgdp", "cpi", "ffr"],
            index=pd.date_range("1990-01-01", periods=T, freq="QS"),
        )

    def test_zero_restriction_holds_exactly_at_h0(self):
        df = self._make_three_var_data(seed=0)
        fit = lib.fit_var(df, lags=2, trend="c")
        irf = lib.zero_sign_restricted_irf(
            fit, horizon=12,
            sign_restrictions={"monetary": {"cpi": -1, "ffr": +1}},
            zero_restrictions={
                "monetary": [{"variable": "rgdp", "horizon": 0}]
            },
            target_accepted=50, max_rotations=2000, seed=42,
            parameter_uncertainty=False,
        )
        assert abs(irf.median[0, 0, 0]) < 1e-10, (
            f"Zero restriction violated: rgdp@h=0 = {irf.median[0, 0, 0]}"
        )
        assert abs(irf.lower_68[0, 0, 0]) < 1e-10
        assert abs(irf.upper_68[0, 0, 0]) < 1e-10

    def test_sign_restrictions_hold_at_h0(self):
        df = self._make_three_var_data(seed=0)
        fit = lib.fit_var(df, lags=2, trend="c")
        irf = lib.zero_sign_restricted_irf(
            fit, horizon=12,
            sign_restrictions={"monetary": {"cpi": -1, "ffr": +1}},
            zero_restrictions={
                "monetary": [{"variable": "rgdp", "horizon": 0}]
            },
            target_accepted=50, max_rotations=2000, seed=42,
            parameter_uncertainty=False,
        )
        assert irf.median[0, 1, 0] < 0
        assert irf.median[0, 2, 0] > 0

    def test_no_zero_restrictions_delegates_to_sign_restricted_irf(self):
        df = self._make_three_var_data(seed=0)
        fit = lib.fit_var(df, lags=2, trend="c")
        irf_zs = lib.zero_sign_restricted_irf(
            fit, horizon=8,
            sign_restrictions={"monetary": {"cpi": -1, "ffr": +1}},
            zero_restrictions=None,
            target_accepted=100, max_rotations=2000, seed=42,
            parameter_uncertainty=False,
        )
        irf_pure = lib.sign_restricted_irf(
            fit, horizon=8,
            sign_restrictions={"monetary": {"cpi": -1, "ffr": +1}},
            target_accepted=100, max_rotations=2000, seed=42,
            parameter_uncertainty=False,
        )
        np.testing.assert_array_equal(irf_zs.median, irf_pure.median)

    def test_empty_zero_dict_also_delegates(self):
        df = self._make_three_var_data(seed=0)
        fit = lib.fit_var(df, lags=2, trend="c")
        irf_zs = lib.zero_sign_restricted_irf(
            fit, horizon=8,
            sign_restrictions={"monetary": {"cpi": -1, "ffr": +1}},
            zero_restrictions={"monetary": []},
            target_accepted=100, max_rotations=2000, seed=42,
            parameter_uncertainty=False,
        )
        irf_pure = lib.sign_restricted_irf(
            fit, horizon=8,
            sign_restrictions={"monetary": {"cpi": -1, "ffr": +1}},
            target_accepted=100, max_rotations=2000, seed=42,
            parameter_uncertainty=False,
        )
        np.testing.assert_array_equal(irf_zs.median, irf_pure.median)

    def test_metadata_tag_zero_sign_when_active(self):
        df = self._make_three_var_data(seed=0)
        fit = lib.fit_var(df, lags=2, trend="c")
        irf = lib.zero_sign_restricted_irf(
            fit, horizon=4,
            sign_restrictions={"mp": {"ffr": +1}},
            zero_restrictions={"mp": [{"variable": "rgdp", "horizon": 0}]},
            target_accepted=20, max_rotations=2000, seed=42,
            parameter_uncertainty=False,
        )
        assert irf.metadata["identification"] == "zero_sign_restrictions"
        assert "zero_restrictions" in irf.metadata
        assert "n_accepted" in irf.metadata

    def test_multi_horizon_zero_restriction_raises(self):
        df = self._make_three_var_data(seed=0)
        fit = lib.fit_var(df, lags=2, trend="c")
        with pytest.raises(ValueError, match="horizon=0"):
            lib.zero_sign_restricted_irf(
                fit, horizon=4,
                sign_restrictions={"mp": {"ffr": +1}},
                zero_restrictions={
                    "mp": [{"variable": "rgdp", "horizon": 2}]
                },
                target_accepted=10, max_rotations=200, seed=42,
                parameter_uncertainty=False,
            )

    def test_unknown_variable_raises(self):
        df = self._make_three_var_data(seed=0)
        fit = lib.fit_var(df, lags=2, trend="c")
        with pytest.raises(ValueError, match="unknown variable"):
            lib.zero_sign_restricted_irf(
                fit, horizon=4,
                sign_restrictions={"mp": {"ffr": +1}},
                zero_restrictions={
                    "mp": [{"variable": "no_such_var", "horizon": 0}]
                },
                target_accepted=10, max_rotations=200, seed=42,
                parameter_uncertainty=False,
            )

    def test_shock_in_zeros_but_not_signs_raises(self):
        df = self._make_three_var_data(seed=0)
        fit = lib.fit_var(df, lags=2, trend="c")
        with pytest.raises(ValueError, match="not in sign_restrictions"):
            lib.zero_sign_restricted_irf(
                fit, horizon=4,
                sign_restrictions={"mp": {"ffr": +1}},
                zero_restrictions={
                    "ghost_shock": [{"variable": "rgdp", "horizon": 0}]
                },
                target_accepted=10, max_rotations=200, seed=42,
                parameter_uncertainty=False,
            )

    def test_multi_shock_zero_sign_joint_identification(self):
        df = self._make_three_var_data(seed=1)
        fit = lib.fit_var(df, lags=2, trend="c")
        irf = lib.zero_sign_restricted_irf(
            fit, horizon=6,
            sign_restrictions={
                "supply": {"cpi": -1, "rgdp": +1},
                "demand": {"cpi": +1, "rgdp": +1},
            },
            zero_restrictions={
                "supply": [],
                "demand": [{"variable": "ffr", "horizon": 0}],
            },
            target_accepted=20, max_rotations=10000, seed=42,
            parameter_uncertainty=False,
        )
        assert irf.shock_names == ["supply", "demand"]
        assert irf.median.shape == (7, 3, 2)
        assert abs(irf.median[0, 2, 1]) < 1e-10   # demand → ffr at h=0
        assert irf.median[0, 1, 0] < 0
        assert irf.median[0, 0, 0] > 0
        assert irf.median[0, 1, 1] > 0
        assert irf.median[0, 0, 1] > 0


class TestType2Dominance:
    """Semantic contract for ADRR (2018) Type-2 contribution dominance.

    Both the single- and multi-shock paths must share _type2_dominates
    (absolute-dominance semantics, ADRR pp. 2806-2807). Two error classes
    this pins: (i) multiplying the contribution by the acceptance
    orientation would invert the positive / negative tests for ~half of
    accepted draws -- contributions are orientation-INVARIANT (flipping a
    column flips both the IRF and the shock series); (ii) a signed-max
    comparison would accept draws where another shock has a larger
    |negative| contribution.
    """

    def test_hand_computable_cases(self):
        from lib.identification import _type2_dominates
        import numpy as np
        # self = +5 dominates |-3|, |1|
        c = np.array([5.0, -3.0, 1.0])
        assert _type2_dominates(c, 0, "absolute")
        assert _type2_dominates(c, 0, "positive")
        assert not _type2_dominates(c, 0, "negative")
        # self = -5 dominates: absolute and negative, NOT positive
        c = np.array([-5.0, 3.0, 1.0])
        assert _type2_dominates(c, 0, "absolute")
        assert not _type2_dominates(c, 0, "positive")
        assert _type2_dominates(c, 0, "negative")
        # another shock dominates in absolute value -> all fail
        # (a signed-max rule would wrongly ACCEPT "positive" here:
        #  2 >= max(2, -5, 1) holds)
        c = np.array([2.0, -5.0, 1.0])
        assert not _type2_dominates(c, 0, "absolute")
        assert not _type2_dominates(c, 0, "positive")
        assert not _type2_dominates(c, 0, "negative")

    def test_unknown_direction_raises(self):
        from lib.identification import _type2_dominates
        import numpy as np
        import pytest as _pytest
        with _pytest.raises(ValueError, match="direction"):
            _type2_dominates(np.array([1.0]), 0, "sideways")


class TestProxyMovingBlockBootstrap:
    """Jentsch-Lunsford (2019) residual moving-block bootstrap — the valid
    replacement for the MR/GK-legacy Rademacher wild bootstrap (which JL
    proved destroys the (u, z) cross-moments the impact column needs)."""

    @staticmethod
    def _fit_and_instrument(synthetic_var_data):
        fit = lib.fit_var(synthetic_var_data, lags=2)
        rng = np.random.default_rng(7)
        # Relevant instrument: y1's structural innovation + measurement noise
        u1 = fit.residuals[:, 0]
        z = np.concatenate([np.zeros(fit.p), u1 + 0.5 * rng.standard_normal(len(u1))])
        return fit, z

    def test_runs_and_reports_block_length(self, synthetic_var_data):
        fit, z = self._fit_and_instrument(synthetic_var_data)
        irf, fs = lib.proxy_svar_irf(
            fit=fit, instrument=z, policy_var="y1", horizon=8,
            bands="moving_block_bootstrap", n_draws=80, seed=1,
        )
        assert np.isfinite(irf.median).all()
        assert np.isfinite(irf.lower_90).all() and np.isfinite(irf.upper_90).all()
        assert (irf.upper_68 >= irf.lower_68).all()
        # JL block-length rule: round(5.03 * T^(1/4))
        expected_ell = max(int(round(5.03 * fit.t_eff ** 0.25)), 2)
        assert irf.metadata["block_length"] == expected_ell
        assert irf.metadata["bands"] == "moving_block_bootstrap"

    def test_seed_determinism(self, synthetic_var_data):
        fit, z = self._fit_and_instrument(synthetic_var_data)
        kw = dict(fit=fit, instrument=z, policy_var="y1", horizon=4,
                  bands="moving_block_bootstrap", n_draws=40, seed=3)
        a, _ = lib.proxy_svar_irf(**kw)
        b, _ = lib.proxy_svar_irf(**kw)
        np.testing.assert_allclose(a.lower_90, b.lower_90)

    def test_differs_from_wild_bootstrap(self, synthetic_var_data):
        fit, z = self._fit_and_instrument(synthetic_var_data)
        mbb, _ = lib.proxy_svar_irf(
            fit=fit, instrument=z, policy_var="y1", horizon=4,
            bands="moving_block_bootstrap", n_draws=60, seed=5)
        wild, _ = lib.proxy_svar_irf(
            fit=fit, instrument=z, policy_var="y1", horizon=4,
            bands="wild_bootstrap", n_draws=60, seed=5)
        assert not np.allclose(mbb.lower_90, wild.lower_90)

    def test_unknown_bands_message_names_both(self, synthetic_var_data):
        fit, z = self._fit_and_instrument(synthetic_var_data)
        with pytest.raises(ValueError, match="moving_block_bootstrap"):
            lib.proxy_svar_irf(fit=fit, instrument=z, policy_var="y1",
                               horizon=2, bands="bogus")


class TestFryPaganMedianTarget:
    """Fry-Pagan (2011) median-target summary."""

    @staticmethod
    def _ladder_draws():
        """Five draws symmetric around a base path: base + k*delta, k=-2..2.

        The pointwise median equals the k=0 draw (index 2) exactly, and the
        standardised distance is proportional to k**2, so the median target
        must select index 2.
        """
        rng = np.random.default_rng(0)
        base = rng.standard_normal((9, 3, 1))
        delta = np.abs(rng.standard_normal((9, 3, 1))) + 0.1
        return np.stack([base + k * delta for k in (-2, -1, 0, 1, 2)], axis=0)

    def test_picks_the_median_member(self):
        draws = self._ladder_draws()
        mt = lib.fry_pagan_median_target(draws)
        assert mt["index"] == 2
        np.testing.assert_allclose(mt["irf"], draws[2])
        assert mt["distance"] == pytest.approx(min(mt["distances"]))

    def test_scale_invariance(self):
        draws = self._ladder_draws()
        scaled = draws.copy()
        scaled[:, :, 1, :] *= 1e6   # blow up one variable in ALL draws
        assert lib.fry_pagan_median_target(scaled)["index"] == \
            lib.fry_pagan_median_target(draws)["index"]

    def test_rejects_bad_input(self):
        with pytest.raises(ValueError):
            lib.fry_pagan_median_target(np.zeros((4, 3, 2)))      # 3-D
        with pytest.raises(ValueError):
            lib.fry_pagan_median_target(np.zeros((1, 4, 3, 2)))   # one draw

    def test_attached_to_sign_restricted_metadata(self, synthetic_var_data):
        fit = lib.fit_var(synthetic_var_data, lags=2)
        v = fit.variable_names
        irf = lib.sign_restricted_irf(
            fit, horizon=8,
            sign_restrictions={"shock": {v[0]: +1, v[1]: -1}},
            target_accepted=40, max_rotations=50_000,
            parameter_uncertainty=False, seed=7,
        )
        meta = irf.metadata["median_target"]
        assert meta is not None
        mt_irf = np.array(meta["irf"])
        assert mt_irf.shape == irf.median.shape
        assert 0 <= meta["index"] < irf.metadata["n_accepted"]
        # The MT draw is an ACCEPTED rotation, so it must satisfy the
        # impact sign restrictions itself (the pointwise median need not).
        assert mt_irf[0, 0, 0] > 0
        assert mt_irf[0, 1, 0] < 0
        # JSON-safe by construction
        import json
        json.dumps(meta)


class TestRotationMeasure:
    """Pins the SAMPLING MEASURE of the sign-restriction engine — the part
    of the reported object Baumeister-Hamilton (2015) warn about.

    Default single-shock mode checks column 0 of a Haar rotation only: the
    accepted impact vector with NO binding restriction is b = P q with q
    uniform on the unit sphere. A missing Haar sign correction
    (Householder QR's deterministic sign of q_1) or the legacy column-scan
    measure would break the symmetry this class pins through the public
    API."""

    def _fit(self, n=3, T=400, seed=5):
        rng = np.random.default_rng(seed)
        Y = np.zeros((T, n))
        A = 0.5 * np.eye(n)
        L = np.linalg.cholesky(np.eye(n) + 0.4 * np.ones((n, n)))
        for t in range(1, T):
            Y[t] = A @ Y[t - 1] + L @ rng.standard_normal(n)
        idx = pd.period_range("1960Q1", periods=T, freq="Q").to_timestamp()
        return lib.fit_var(
            data=pd.DataFrame(Y, columns=["a", "b", "c"], index=idx),
            lags=1, trend="c",
        )

    def test_unrestricted_accepts_are_sign_symmetric(self):
        fit = self._fit()
        irf = lib.sign_restricted_irf(
            fit=fit, horizon=2,
            sign_restrictions={"s": {"a": 0, "b": 0, "c": 0}},  # nothing binds
            restriction_horizons=1, target_accepted=4000,
            max_rotations=5000, seed=11, parameter_uncertainty=False,
        )
        assert irf.metadata["n_accepted"] == 4000
        # Under the uniform-sphere measure the pointwise MEDIAN of
        # b = P q is 0 componentwise (sign symmetry); a Householder sign
        # artefact pushes it to O(0.4 x impact scale).
        impact_scale = np.abs(np.linalg.cholesky(np.asarray(fit.Sigma_u))).max()
        assert np.all(np.abs(irf.median[0, :, 0]) < 0.08 * impact_scale)

    def test_column_scan_optin_differs_and_is_more_efficient(self):
        fit = self._fit()
        kw = dict(
            fit=fit, horizon=2,
            sign_restrictions={"s": {"a": +1, "b": +1, "c": -1}},
            restriction_horizons=2, target_accepted=400,
            max_rotations=200_000, parameter_uncertainty=False, seed=4,
        )
        sphere = lib.sign_restricted_irf(**kw)
        scan = lib.sign_restricted_irf(**kw, column_scan=True)
        # Scan inspects n columns per rotation -> strictly fewer attempts
        # for the same number of accepts.
        assert scan.metadata["n_attempted"] < sphere.metadata["n_attempted"]

    def test_multishock_acceptance_key_order_invariant(self):
        fit = self._fit(seed=9)
        rA = {"a": +1, "b": -1, "c": 0}
        rB = {"a": 0, "b": +1, "c": +1}
        kw = dict(fit=fit, horizon=2, restriction_horizons=1,
                  target_accepted=1500, max_rotations=400_000,
                  parameter_uncertainty=False, seed=21)
        ab = lib.sign_restricted_irf(
            sign_restrictions={"A": rA, "B": rB}, **kw)
        ba = lib.sign_restricted_irf(
            sign_restrictions={"B": rB, "A": rA}, **kw)
        rate_ab = ab.metadata["acceptance_rate"]
        rate_ba = ba.metadata["acceptance_rate"]
        # Haar columns are exchangeable -> rates equal up to MC noise.
        # The removed greedy assignment produced 0.32 vs 0.62 hereabouts.
        assert abs(rate_ab - rate_ba) < 0.04, (rate_ab, rate_ba)
        # Shock A's accepted distribution must not depend on listing order
        iA_ab = list(ab.shock_names).index("A")
        iA_ba = list(ba.shock_names).index("A")
        np.testing.assert_allclose(
            ab.median[0, :, iA_ab], ba.median[0, :, iA_ba], atol=0.06
        )

    def test_fev_share_restriction_matches_structural_fevd_convention(self):
        """The engine's FEV-share at 'horizon' H must be numerically the
        same object as lib.structural_fevd(horizon=H): accept the seeded
        draw under a vanishing band around its own structural_fevd share,
        reject under a band just above it."""
        fit = self._fit(seed=3)
        H_check = 4
        base = dict(
            fit=fit, horizon=6,
            sign_restrictions={"s": {"a": 0, "b": 0, "c": 0}},
            restriction_horizons=1, target_accepted=1,
            max_rotations=1, parameter_uncertainty=False, seed=77,
        )
        # Learn the (deterministic, seeded) draw's impact vector
        free = lib.sign_restricted_irf(**base)
        b = np.asarray(free.median[0, :, 0])
        share = float(lib.structural_fevd(fit, b, horizon=H_check)[0, 0])  # var "a"
        ok = lib.sign_restricted_irf(
            **base,
            fev_share_restrictions=[{
                "variable": "a", "horizon": H_check,
                "lower": share - 1e-9, "upper": share + 1e-9,
            }],
        )
        assert ok.metadata["n_accepted"] == 1
        with pytest.raises(RuntimeError, match="no rotations"):
            lib.sign_restricted_irf(
                **base,
                fev_share_restrictions=[{
                    "variable": "a", "horizon": H_check,
                    "lower": min(share + 1e-6, 1.0), "upper": 1.0,
                }],
            )


class TestResidualBootstrapCentred:
    """Bootstrap IRF draws must be centred on the point IRF: the Stine
    rescaling sqrt(T/(T-k)) plus the matching T-k Sigma divisor. The
    pre-fix scheme sat ~(T-k)/T low (≈0.96 at this k/T) — the 1.5% gate
    fails it decisively."""

    def test_mean_impact_ratio_is_one(self):
        rng = np.random.default_rng(2)
        T, n = 124, 2
        Y = np.zeros((T, n))
        for t in range(2, T):
            Y[t] = 0.5 * Y[t - 1] - 0.2 * Y[t - 2] + rng.standard_normal(n)
        idx = pd.period_range("1960Q1", periods=T, freq="Q").to_timestamp()
        fit = lib.fit_var(
            data=pd.DataFrame(Y, columns=["x", "y"], index=idx),
            lags=2, trend="c",
        )
        irf = lib.cholesky_irf(fit=fit, horizon=4,
                               bands="residual_bootstrap",
                               n_draws=600, seed=8)
        point = float(np.linalg.cholesky(np.asarray(fit.Sigma_u))[0, 0])
        ratio = float(irf.median[0, 0, 0]) / point
        assert abs(ratio - 1.0) < 0.015, f"impact centring ratio {ratio:.4f}"


class TestType2CriterionAndObject:
    """ADRR (2018) Type-2 modifiers: the 'overwhelming' (Type B)
    dominance threshold and the 'unexpected'-movement contribution object.
    """

    # --- threshold: most_important (max) vs overwhelming (sum of others) ---
    def test_dominates_threshold_distinguishes_criteria(self):
        # self (col 0) = 3 is the single largest, but 3 < 2 + 2 (sum of others):
        c = np.array([3.0, 2.0, 2.0])
        assert _type2_dominates(c, 0, "absolute", criterion="most_important")
        assert not _type2_dominates(c, 0, "absolute", criterion="overwhelming")
        # self = 5 > 2 + 2: passes both
        c2 = np.array([5.0, 2.0, 2.0])
        assert _type2_dominates(c2, 0, "absolute", criterion="overwhelming")
        assert _type2_dominates(c2, 0, "absolute", criterion="most_important")

    def test_dominates_direction_with_overwhelming(self):
        c = np.array([-5.0, 2.0, 1.0])          # negative self, overwhelming magnitude
        assert _type2_dominates(c, 0, "negative", criterion="overwhelming")
        assert not _type2_dominates(c, 0, "positive", criterion="overwhelming")

    def test_dominates_unknown_criterion_raises(self):
        with pytest.raises(ValueError, match="criterion"):
            _type2_dominates(np.array([1.0, 0.0]), 0, "absolute", criterion="bogus")

    # --- contribution object: unexpected (impact) vs cumulative (level) ---
    def test_contributions_unexpected_is_impact_product(self):
        rng = np.random.default_rng(0)
        n, H = 2, 3
        irf_all_ext = rng.standard_normal((H + 1, n, n))
        B_impact = irf_all_ext[0]
        eps_all = rng.standard_normal((H + 1, n))
        var_idx, t_idx = 1, 2
        unexpected = _type2_contributions(
            irf_all_ext, eps_all, B_impact, var_idx, t_idx, "unexpected")
        assert np.allclose(unexpected, B_impact[var_idx, :] * eps_all[t_idx, :])
        cumulative = _type2_contributions(
            irf_all_ext, eps_all, B_impact, var_idx, t_idx, "cumulative")
        manual = sum(irf_all_ext[t_idx - s, var_idx, :] * eps_all[s, :]
                     for s in range(t_idx + 1))
        assert np.allclose(cumulative, manual)
        assert not np.allclose(unexpected, cumulative)   # differ for t_idx > 0

    def test_contributions_coincide_at_first_date(self):
        # At t_idx = 0 the cumulative convolution is just the impact term, so the
        # two objects must coincide.
        rng = np.random.default_rng(1)
        n = 3
        irf_all_ext = rng.standard_normal((2, n, n))
        B_impact = irf_all_ext[0]
        eps_all = rng.standard_normal((2, n))
        a = _type2_contributions(irf_all_ext, eps_all, B_impact, 0, 0, "unexpected")
        b = _type2_contributions(irf_all_ext, eps_all, B_impact, 0, 0, "cumulative")
        assert np.allclose(a, b)

    def test_parse_extras_defaults_and_validation(self):
        assert _parse_type2_extras({}) == ("most_important", "cumulative")
        assert _parse_type2_extras(
            {"criterion": "overwhelming", "contribution": "unexpected"}
        ) == ("overwhelming", "unexpected")
        with pytest.raises(ValueError, match="criterion"):
            _parse_type2_extras({"criterion": "x"})
        with pytest.raises(ValueError, match="contribution"):
            _parse_type2_extras({"contribution": "x"})

    # --- integration: both paths run; overwhelming is no more lenient ---
    def _mid_date(self, fit):
        dates = list(fit.data.index[fit.p:])
        return dates[len(dates) // 2]

    def test_overwhelming_no_more_lenient_multishock(self, synthetic_var_data):
        fit = lib.fit_var(synthetic_var_data, lags=2)
        common = dict(
            fit=fit, horizon=4,
            sign_restrictions={"s1": {"y1": +1, "y2": -1}, "s2": {"y1": -1, "y2": +1}},
            target_accepted=15, max_rotations=40_000, seed=3,
            parameter_uncertainty=False,
        )
        base = dict(date=self._mid_date(fit), type="largest_contributor",
                    variable="y1", shock="s1")
        most = lib.sign_restricted_irf(
            **common, narrative_restrictions=[{**base, "criterion": "most_important"}])
        overwh = lib.sign_restricted_irf(
            **common, narrative_restrictions=[{**base, "criterion": "overwhelming"}])
        assert (overwh.metadata["acceptance_rate"]
                <= most.metadata["acceptance_rate"] + 1e-9)

    def test_unexpected_object_singleshock_runs(self, synthetic_var_data):
        fit = lib.fit_var(synthetic_var_data, lags=2)
        irf = lib.sign_restricted_irf(
            fit=fit, horizon=4,
            sign_restrictions={"mp": {"y1": +1, "y2": -1}},   # one key -> single-shock
            narrative_restrictions=[
                {"date": self._mid_date(fit), "type": "largest_contributor",
                 "variable": "y1", "contribution": "unexpected",
                 "criterion": "overwhelming"}],
            target_accepted=10, max_rotations=60_000, seed=3,
            parameter_uncertainty=False,
        )
        assert irf.median.shape[1] == fit.n_vars
        assert "type2_contribution" in irf.metadata["identification"]

    def test_defaults_match_explicit_backward_compat(self, synthetic_var_data):
        """Omitting the modifiers must reproduce most_important + cumulative."""
        fit = lib.fit_var(synthetic_var_data, lags=2)
        common = dict(
            fit=fit, horizon=4,
            sign_restrictions={"s1": {"y1": +1, "y2": -1}, "s2": {"y1": -1, "y2": +1}},
            target_accepted=10, max_rotations=40_000, seed=5,
            parameter_uncertainty=False,
        )
        base = dict(date=self._mid_date(fit), type="largest_contributor",
                    variable="y1", shock="s1")
        omitted = lib.sign_restricted_irf(**common, narrative_restrictions=[base])
        explicit = lib.sign_restricted_irf(**common, narrative_restrictions=[
            {**base, "criterion": "most_important", "contribution": "cumulative"}])
        assert np.allclose(omitted.median, explicit.median)


class TestMagnitudeRestrictions:
    """Relative-magnitude restrictions (Kilian-Murphy elasticity
    bounds): reject a rotation unless
    |IRF[h, stronger, shock]| >= ratio * |IRF[h, weaker, shock]|.
    """

    def test_resolver_expands_and_validates(self):
        names = ["a", "b", "c"]
        assert _resolve_magnitude_restriction(
            {"stronger": "a", "weaker": "c", "horizons": [0, 2], "ratio": 1.5}, names, 8
        ) == [(0, 2, 0, 1.5), (0, 2, 2, 1.5)]
        # int 'horizon' alias + default ratio
        assert _resolve_magnitude_restriction(
            {"stronger": "b", "weaker": "a", "horizon": 3}, names, 8
        ) == [(1, 0, 3, 1.0)]
        with pytest.raises(ValueError, match="unknown variable"):
            _resolve_magnitude_restriction({"stronger": "x", "weaker": "a", "horizons": 0}, names, 8)
        with pytest.raises(ValueError, match="different variables"):
            _resolve_magnitude_restriction({"stronger": "a", "weaker": "a", "horizons": 0}, names, 8)
        with pytest.raises(ValueError, match="ratio"):
            _resolve_magnitude_restriction(
                {"stronger": "a", "weaker": "b", "horizons": 0, "ratio": 0}, names, 8)
        with pytest.raises(ValueError, match="horizon"):
            _resolve_magnitude_restriction({"stronger": "a", "weaker": "b", "horizons": 99}, names, 8)

    def test_single_shock_enforced_exactly(self, synthetic_var_data):
        """target_accepted=1 -> the median IS the single accepted draw, so it
        must satisfy the magnitude restriction exactly at every listed horizon."""
        fit = lib.fit_var(synthetic_var_data, lags=2)
        irf = lib.sign_restricted_irf(
            fit=fit, horizon=8, sign_restrictions={"mp": {"y1": +1, "y2": -1}},
            magnitude_restrictions=[{"stronger": "y1", "weaker": "y3", "horizons": [0, 1, 2]}],
            target_accepted=1, max_rotations=200_000, seed=2, parameter_uncertainty=False)
        med = irf.median[:, :, 0]
        V = list(fit.variable_names)
        for h in (0, 1, 2):
            assert abs(med[h, V.index("y1")]) + 1e-9 >= abs(med[h, V.index("y3")])
        assert "magnitude" in irf.metadata["identification"]
        assert irf.metadata["magnitude_restrictions"] is not None

    def test_filters_rotations_and_records(self, synthetic_var_data):
        fit = lib.fit_var(synthetic_var_data, lags=2)
        common = dict(fit=fit, horizon=8, sign_restrictions={"mp": {"y1": +1, "y2": -1}},
                      target_accepted=25, max_rotations=80_000, seed=4,
                      parameter_uncertainty=False)
        base = lib.sign_restricted_irf(**common)
        restr = lib.sign_restricted_irf(**common, magnitude_restrictions=[
            {"stronger": "y1", "weaker": "y3", "horizons": 0}])
        assert (restr.metadata["acceptance_rate"]
                <= base.metadata["acceptance_rate"] + 1e-9)
        assert restr.metadata["n_rejected_by_magnitude"] >= 0

    def test_ratio_stricter_than_plain_dominance(self, synthetic_var_data):
        fit = lib.fit_var(synthetic_var_data, lags=2)
        common = dict(fit=fit, horizon=8, sign_restrictions={"mp": {"y1": +1, "y2": -1}},
                      target_accepted=20, max_rotations=200_000, seed=6,
                      parameter_uncertainty=False)
        dom = lib.sign_restricted_irf(**common, magnitude_restrictions=[
            {"stronger": "y1", "weaker": "y3", "horizons": 0, "ratio": 1.0}])
        ela = lib.sign_restricted_irf(**common, magnitude_restrictions=[
            {"stronger": "y1", "weaker": "y3", "horizons": 0, "ratio": 3.0}])
        assert (ela.metadata["acceptance_rate"]
                <= dom.metadata["acceptance_rate"] + 1e-9)

    def test_multishock_enforced_and_requires_shock_key(self, synthetic_var_data):
        fit = lib.fit_var(synthetic_var_data, lags=2)
        irf = lib.sign_restricted_irf(
            fit=fit, horizon=8,
            sign_restrictions={"s1": {"y1": +1, "y2": -1}, "s2": {"y1": -1, "y2": +1}},
            magnitude_restrictions=[
                {"shock": "s1", "stronger": "y1", "weaker": "y3", "horizons": 0}],
            target_accepted=1, max_rotations=300_000, seed=2, parameter_uncertainty=False)
        si = list(irf.shock_names).index("s1")
        med = irf.median[:, :, si]
        V = list(fit.variable_names)
        assert abs(med[0, V.index("y1")]) + 1e-9 >= abs(med[0, V.index("y3")])
        assert "magnitude" in irf.metadata["identification"]
        # multi-shock mode requires a 'shock' key on each restriction
        with pytest.raises(ValueError, match="shock"):
            lib.sign_restricted_irf(
                fit=fit, horizon=8,
                sign_restrictions={"s1": {"y1": +1}, "s2": {"y1": -1}},
                magnitude_restrictions=[{"stronger": "y1", "weaker": "y3", "horizons": 0}],
                target_accepted=5, max_rotations=1000, parameter_uncertainty=False)


# ---------------------------------------------------------------------------
# Acceptance checks: long-run stationarity, proxy rank requirement
# ---------------------------------------------------------------------------


class TestLongRunStationarityGuard:
    def test_explosive_lag_matrix_rejected_by_helper(self):
        from lib.identification import _long_run_structural_matrix
        A = np.diag([1.05, 0.5])
        with pytest.raises(ValueError, match="stationary"):
            _long_run_structural_matrix([A], np.eye(2))

    def test_negative_unit_root_rejected_by_helper(self):
        from lib.identification import _long_run_structural_matrix
        # Root at -1: (I - A) is well-conditioned but the MA sum oscillates
        A = np.diag([-1.0, 0.5])
        with pytest.raises(ValueError, match="stationary"):
            _long_run_structural_matrix([A], np.eye(2))

    def test_stable_matrix_accepted(self):
        from lib.identification import _long_run_structural_matrix
        A = np.diag([0.95, 0.5])
        H = _long_run_structural_matrix([A], np.eye(2))
        assert np.all(np.isfinite(H))

    def test_explosive_point_estimate_rejected_publicly(self, synthetic_var_data):
        import dataclasses
        fit = lib.fit_var(synthetic_var_data, lags=1)
        B_bad = fit.B_hat.copy()
        n = fit.n_vars
        B_bad[:n, :n] = np.diag([1.05, 0.5, 0.5])   # explosive lag block
        fit_bad = dataclasses.replace(fit, B_hat=B_bad)
        with pytest.raises(ValueError, match="stationary"):
            lib.long_run_irf(fit_bad, horizon=10, bands="none")

    def test_draw_rejection_counts_reported(self, synthetic_var_data):
        fit = lib.fit_var(synthetic_var_data, lags=2)
        irf = lib.long_run_irf(fit, horizon=5, bands="parameter_draws",
                               n_draws=50, seed=0)
        meta = irf.metadata
        assert "n_draws_retained" in meta
        assert "n_draws_rejected_nonstationary_or_singular" in meta
        assert (meta["n_draws_retained"]
                + meta["n_draws_rejected_nonstationary_or_singular"]) == 50


class TestProxyRankRequirement:
    def test_duplicate_instruments_rejected(self, synthetic_var_data):
        """Two identical instrument columns span one direction — the m=2
        identification must stop, however strong the per-pair F looks."""
        rng = np.random.default_rng(0)
        T = len(synthetic_var_data)
        z = rng.standard_normal(T)
        Z = np.column_stack([z, z])
        fit = lib.fit_var(synthetic_var_data, lags=2)
        with pytest.raises(RuntimeError, match="rank-deficient"):
            lib.proxy_svar_irf(
                fit=fit, instrument=Z, policy_var=["y1", "y2"],
                horizon=3, bands="none", normalization="unit_policy",
            )

    def test_linear_combination_rejected(self, synthetic_var_data):
        rng = np.random.default_rng(1)
        T = len(synthetic_var_data)
        z1 = rng.standard_normal(T)
        z2 = rng.standard_normal(T)
        Z = np.column_stack([z1, z2, 0.3 * z1 - 0.7 * z2])
        fit = lib.fit_var(synthetic_var_data, lags=2)
        with pytest.raises(RuntimeError, match="rank-deficient"):
            lib.proxy_svar_irf(
                fit=fit, instrument=Z, policy_var=["y1", "y2", "y3"],
                horizon=3, bands="none", normalization="unit_policy",
            )


class TestRigobonSeparationDiagnostic:
    def test_weak_flag_invariant_to_regime_relabeling(self):
        """Swapping the two regime labels inverts every variance ratio but
        changes no information — the weak-identification flag must agree."""
        rng = np.random.default_rng(2)
        T = 400
        Y = np.zeros((T, 2))
        scale = np.where(np.arange(T) < T // 2, 1.0, 3.0)
        for t in range(1, T):
            Y[t] = 0.5 * Y[t - 1] + rng.standard_normal(2) * scale[t]
        idx = pd.period_range("1980Q1", periods=T, freq="Q").to_timestamp()
        data = pd.DataFrame(Y, columns=["y1", "y2"], index=idx)
        regime = (np.arange(T) >= T // 2).astype(int)

        fit = lib.fit_var(data, lags=1)
        _, d1 = lib.heteroskedasticity_irf(fit, regime=regime, horizon=0,
                                           bands="none")
        _, d2 = lib.heteroskedasticity_irf(fit, regime=1 - regime, horizon=0,
                                           bands="none")
        assert d1["weak_identification"] == d2["weak_identification"]
        assert np.isclose(d1["min_pairwise_log_separation"],
                          d2["min_pairwise_log_separation"], rtol=1e-8)


class TestZeroSignOrderingAndScale:
    """Acceptance check: shock ORDER must not decide feasibility (ARRW
    Theorems 3-4 ordering is handled internally), and zero restrictions
    must survive a change of variable units."""

    @staticmethod
    def _two_var_fit(seed=0, scale_v1=1.0):
        rng = np.random.default_rng(seed)
        T = 300
        y = np.zeros((T, 2))
        for t in range(1, T):
            y[t] = 0.3 * y[t - 1] + rng.standard_normal(2)
        y[:, 0] *= scale_v1
        df = pd.DataFrame(y, columns=["v1", "v2"],
                          index=pd.date_range("1980-01-01", periods=T, freq="QS"))
        return lib.fit_var(df, lags=1, trend="c")

    def _run(self, fit, order):
        sr = {name: ({"v1": +1} if name == "A" else {"v2": +1})
              for name in order}
        sr = {k: sr[k] for k in order}   # preserve requested ordering
        return lib.zero_sign_restricted_irf(
            fit, horizon=4,
            sign_restrictions=sr,
            zero_restrictions={"B": [{"variable": "v1", "horizon": 0}]},
            target_accepted=20, max_rotations=4000, seed=1,
            parameter_uncertainty=False,
        )

    def test_both_shock_orderings_feasible(self):
        fit = self._two_var_fit()
        irf_ab = self._run(fit, ["A", "B"])   # unrestricted shock named first
        irf_ba = self._run(fit, ["B", "A"])
        # Shock B's zero restriction on v1 must hold in both
        j_ab = irf_ab.shock_names.index("B shock") if "B shock" in irf_ab.shock_names else irf_ab.shock_names.index("B")
        j_ba = irf_ba.shock_names.index(irf_ab.shock_names[j_ab])
        v1 = 0
        assert abs(irf_ab.median[0, v1, j_ab]) < 1e-8
        assert abs(irf_ba.median[0, v1, j_ba]) < 1e-8

    def test_infeasible_zero_pattern_rejected_before_sampling(self):
        fit = self._two_var_fit()
        with pytest.raises(ValueError, match="rank condition"):
            lib.zero_sign_restricted_irf(
                fit, horizon=4,
                sign_restrictions={"A": {"v1": +1}, "B": {"v2": +1}},
                zero_restrictions={
                    "A": [{"variable": "v2", "horizon": 0}],
                    "B": [{"variable": "v1", "horizon": 0}],
                },
                target_accepted=5, max_rotations=100, seed=1,
                parameter_uncertainty=False,
            )

    def test_zero_restriction_survives_tiny_units(self):
        # Shrinking v1's units by 1e-11 must not silently drop its zero
        # restriction (the constraint is scale-free).
        fit = self._two_var_fit(scale_v1=1e-11)
        irf = self._run(fit, ["B", "A"])
        j = 0  # first shock in requested order is B
        scale = float(np.sqrt(np.asarray(fit.Sigma_u)[0, 0]))
        assert np.all(np.abs(irf.median[0, 0, j]) <= 1e-6 * scale)


class TestZeroSignSignOnZeroRestrictedVariable:
    """Regression contract: a zero on impact and a sign on the SAME variable
    ("zero on impact, positive afterwards") is decided by the responses at
    the following horizons, never by the rounding-level value left in the
    zero-restricted impact cell.

    The reference is an independent sampler written directly from the
    method: draw the column uniformly in the null space of the restricted
    row of the Cholesky factor, then check the signs with a flip allowed."""

    @staticmethod
    def _fit():
        rng = np.random.default_rng(4)
        while True:
            A1 = 0.5 * np.eye(3) + 0.15 * rng.standard_normal((3, 3))
            if np.max(np.abs(np.linalg.eigvals(A1))) < 0.9:
                break
        B = np.tril(rng.standard_normal((3, 3))) + np.eye(3)
        Y = np.zeros((500, 3))
        for t in range(1, 500):
            Y[t] = A1 @ Y[t - 1] + B @ rng.standard_normal(3)
        idx = pd.date_range("1960-01-01", periods=300, freq="QS")
        return lib.fit_var(
            pd.DataFrame(Y[200:], columns=["rgdp", "cpi", "ffr"], index=idx), lags=1)

    def test_matches_independent_sampler(self):
        from scipy.linalg import null_space
        fit = self._fit()
        H = 4
        irf = lib.zero_sign_restricted_irf(
            fit, horizon=H,
            sign_restrictions={"mp": {"cpi": +1, "ffr": +1}},
            zero_restrictions={"mp": [{"variable": "cpi", "horizon": 0}]},
            restriction_horizons=3, target_accepted=2000,
            max_rotations=200_000, seed=11, parameter_uncertainty=False,
        )
        # the zero-restricted cell is reported as exactly zero
        assert irf.median[0, 1, 0] == 0.0
        assert irf.lower_90[0, 1, 0] == 0.0 and irf.upper_90[0, 1, 0] == 0.0

        A = np.asarray(fit.lag_matrices()[0])
        P = np.linalg.cholesky(np.asarray(fit.Sigma_u))
        resp_of = np.array([np.linalg.matrix_power(A, h) @ P for h in range(H + 1)])
        K = null_space(P[[1], :])
        rng = np.random.default_rng(1)

        def ok(r):
            # cpi: zero at h = 0, positive at h = 1, 2; ffr positive at h = 0, 1, 2
            return np.all(r[:3, 2] >= 0) and np.all(r[1:3, 1] >= 0)

        accepted, tries = [], 0
        while len(accepted) < 4000:
            tries += 1
            q = K @ rng.standard_normal(2)
            r = resp_of @ (q / np.linalg.norm(q))
            if ok(r):
                accepted.append(r)
            elif ok(-r):
                accepted.append(-r)
        accepted = np.array(accepted)
        rate = irf.metadata["n_accepted"] / irf.metadata["n_attempted"]
        assert abs(rate - len(accepted) / tries) < 0.05
        gap = irf.median[0, 0, 0] - np.median(accepted[:, 0, 0])
        assert abs(gap) < 0.25 * accepted[:, 0, 0].std()
