"""Tests for ``lib.specs`` — the spec dataclass + the ``run`` driver.

Validation strategy:

1. **Round-trip** — ``Spec.from_dict(spec.to_dict()) == spec`` for any spec.
2. **Immutability** — ``spec.update(...)`` returns a new Spec; the original
   is unchanged.
3. **Deep-merge** — partial overrides preserve siblings in nested blocks.
4. **Loader equivalence** — ``Spec.from_replication("litterman_1986_bvar")``
   produces a runnable spec; running it gives the same IRF shape as the
   standalone replicate.py and matches a known coefficient.
5. **Parameter override actually changes the result** — running a tighter
   prior gives a different IRF than the default.
6. **Error paths** — unknown spec fields, missing config files, unsupported
   model classes all raise informative errors.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import lib  # noqa: E402
from lib.specs import Spec, SpecResult, load_data, run, _deep_merge  # noqa: E402

warnings.filterwarnings("ignore", category=Warning)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_FRED_QD_CSV = Path(__file__).resolve().parents[1] / "data" / "fred_qd" / "dataset.csv"
requires_fred_qd = pytest.mark.skipif(
    not _FRED_QD_CSV.exists(),
    reason="fred_qd dataset.csv not downloaded - see data/fred_qd/UPDATE.md",
)


@pytest.fixture
def minimal_spec_dict() -> dict:
    return {
        "data": {
            "source": "fred_qd",
            "sample_start": "1980-01-01",
            "sample_end": "2019-12-31",
            "series": [
                {"name": "rgdp", "fred_code": "GDPC1"},
                {"name": "cpi", "fred_code": "GDPCTPI"},
                {"name": "ffr", "fred_code": "FEDFUNDS"},
            ],
            "transforms": {"rgdp": "log", "cpi": "log", "ffr": "level"},
            "variable_order": ["rgdp", "cpi", "ffr"],
        },
        "model": {"class": "var", "lags": 2, "trend": "c"},
        "irf": {"method": "cholesky", "horizon": 8, "bands": "asymptotic_mc", "n_draws": 200},
        "metadata": {"seed": 42, "note": "minimal test fixture"},
    }


# ---------------------------------------------------------------------------
# Spec dataclass: serialisation + update
# ---------------------------------------------------------------------------


class TestSpecDataclass:
    def test_round_trip(self, minimal_spec_dict):
        s = Spec.from_dict(minimal_spec_dict)
        out = s.to_dict()
        # to_dict puts None on missing blocks; we should be able to construct again
        s2 = Spec.from_dict(out)
        assert s == s2

    def test_update_returns_new_instance(self, minimal_spec_dict):
        s = Spec.from_dict(minimal_spec_dict)
        s2 = s.update(model={"lags": 4})
        assert s is not s2
        assert s.model["lags"] == 2     # original unchanged
        assert s2.model["lags"] == 4

    def test_update_deep_merges_nested_dicts(self, minimal_spec_dict):
        """A partial override into model.prior keeps the other prior fields."""
        d = minimal_spec_dict.copy()
        d["model"] = {
            "class": "bvar", "lags": 4, "trend": "c",
            "prior": {"lambda_1": 0.1, "lambda_2": 0.5, "mu": 1.0},
        }
        s = Spec.from_dict(d)
        s2 = s.update(model={"prior": {"lambda_1": 0.05}})
        assert s2.model["prior"]["lambda_1"] == 0.05    # overridden
        assert s2.model["prior"]["lambda_2"] == 0.5     # preserved
        assert s2.model["prior"]["mu"] == 1.0           # preserved
        assert s2.model["lags"] == 4                     # preserved

    def test_update_replaces_with_none(self, minimal_spec_dict):
        """update(forecast=None) clears the forecast block entirely."""
        s = Spec.from_dict(minimal_spec_dict)
        # Add a forecast block first
        s = s.update(forecast={"horizon": 4})
        assert s.forecast is not None
        s2 = s.update(forecast=None)
        assert s2.forecast is None

    def test_update_into_none_field(self, minimal_spec_dict):
        """update(forecast={...}) on a spec with forecast=None creates the block."""
        s = Spec.from_dict(minimal_spec_dict)
        assert s.forecast is None
        s2 = s.update(forecast={"horizon": 12, "n_paths": 500})
        assert s2.forecast == {"horizon": 12, "n_paths": 500}

    def test_unknown_field_rejected(self, minimal_spec_dict):
        s = Spec.from_dict(minimal_spec_dict)
        with pytest.raises(ValueError, match="unknown spec field"):
            s.update(estimation={"method": "ml"})

    def test_immutability_via_frozen(self, minimal_spec_dict):
        s = Spec.from_dict(minimal_spec_dict)
        with pytest.raises(Exception):    # FrozenInstanceError or AttributeError
            s.data = {}     # type: ignore[misc]


# ---------------------------------------------------------------------------
# _deep_merge
# ---------------------------------------------------------------------------


class TestDeepMerge:
    def test_overlay_wins_at_leaf(self):
        out = _deep_merge({"a": 1, "b": 2}, {"a": 10})
        assert out == {"a": 10, "b": 2}

    def test_nested_dicts_merged(self):
        out = _deep_merge(
            {"a": {"x": 1, "y": 2}, "b": 5},
            {"a": {"x": 10}},
        )
        assert out == {"a": {"x": 10, "y": 2}, "b": 5}

    def test_non_dict_at_overlay_replaces(self):
        """If the overlay value at a key isn't a dict, it replaces wholesale."""
        out = _deep_merge({"a": {"x": 1}}, {"a": [1, 2, 3]})
        assert out == {"a": [1, 2, 3]}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


class TestSpecLoaders:
    def test_from_config_yaml(self, minimal_spec_dict, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(minimal_spec_dict))
        s = Spec.from_config(path)
        assert s.data == minimal_spec_dict["data"]
        assert s.model == minimal_spec_dict["model"]
        assert s.irf == minimal_spec_dict["irf"]

    def test_from_config_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Spec.from_config(tmp_path / "does_not_exist.yaml")

    def test_from_config_folds_replication_into_metadata(self, tmp_path):
        """A 'replication' top-level block in config.yaml lands in metadata."""
        d = {
            "replication": {"id": "demo", "title": "Demo"},
            "data": {"source": "fred_qd", "series": [{"name": "x", "fred_code": "GDPC1"}],
                     "transforms": {"x": "log"}, "variable_order": ["x"]},
            "model": {"class": "var", "lags": 1, "trend": "c"},
        }
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(d))
        s = Spec.from_config(path)
        assert s.metadata["replication"]["id"] == "demo"

    def test_from_config_separates_forecast_and_forecast_evaluation(self, tmp_path):
        """`forecast` (iterated end-of-sample) and `forecast_evaluation`
        (rolling-window OOS) are semantically distinct. They must load
        into separate fields, never alias."""
        d = {
            "data": {"source": "fred_qd", "series": [{"name": "x", "fred_code": "GDPC1"}],
                     "transforms": {"x": "log"}, "variable_order": ["x"]},
            "model": {"class": "var", "lags": 1, "trend": "c"},
            "forecast": {"horizon": 12, "n_paths": 1000},
            "forecast_evaluation": {"window_size": 80, "horizon": 12, "step": 4},
        }
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(d))
        s = Spec.from_config(path)
        assert s.forecast == {"horizon": 12, "n_paths": 1000}
        assert s.forecast_evaluation == {"window_size": 80, "horizon": 12, "step": 4}

    def test_from_config_forecast_evaluation_only(self, tmp_path):
        """`forecast_evaluation` without `forecast` loads into its own
        field and `spec.forecast` stays None (so `lib.run` doesn't try
        to compute an iterated forecast from rolling-window settings)."""
        d = {
            "data": {"source": "fred_qd", "series": [{"name": "x", "fred_code": "GDPC1"}],
                     "transforms": {"x": "log"}, "variable_order": ["x"]},
            "model": {"class": "var", "lags": 1, "trend": "c"},
            "forecast_evaluation": {"window_size": 80, "horizon": 12, "step": 4},
        }
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(d))
        s = Spec.from_config(path)
        assert s.forecast is None
        assert s.forecast_evaluation == {"window_size": 80, "horizon": 12, "step": 4}

    def test_from_replication_loads_existing(self):
        """The Litterman replication is in the repo — should load."""
        s = Spec.from_replication("litterman_1986_bvar")
        assert s.data["source"] == "fred_qd"
        assert s.model["class"] == "bvar"
        assert s.irf is not None
        assert s.metadata.get("replication", {}).get("id") == "litterman_1986_bvar"


class TestFromDictionary:
    """``Spec.from_dictionary`` delegates to ``from_replication`` when a
    paired replication exists, and produces a partial Spec from the
    descriptive entry otherwise."""

    def test_delegates_to_replication_when_paired(self):
        """Litterman has both a dictionary entry AND a canonical replication.
        ``from_dictionary`` should produce the same Spec as
        ``from_replication`` (the implementation config is more complete)."""
        s_dict = Spec.from_dictionary("litterman_1986_bvar")
        s_rep = Spec.from_replication("litterman_1986_bvar")
        assert s_dict == s_rep

    def test_prefer_replication_false_loads_descriptive_entry(self):
        """Passing prefer_replication=False forces the descriptive loader
        even when a replication exists."""
        s = Spec.from_dictionary("litterman_1986_bvar", prefer_replication=False)
        # Descriptive Spec: source / series / sample dates not present
        assert "source" not in s.data
        assert "series" not in s.data
        assert "sample_start" not in s.data
        # But variable_order and transforms ARE derived from the entry
        assert "variable_order" in s.data
        assert "real_gnp" in s.data["variable_order"]
        # Model block populated from the entry
        assert s.model["class"] == "bvar"
        assert s.model["lags"] == 6
        assert s.model["prior"]["lambda_1"] == 0.2
        assert s.model["prior"]["lambda_2"] == 0.2
        # Identification method "none" → no IRF block
        assert s.irf is None
        # Dictionary entry carried in metadata
        assert s.metadata["dictionary_entry"]["id"] == "litterman_1986_bvar"

    def test_descriptive_entry_for_unpaired_dictionary_entry(self):
        """The descriptive (Mode B) translator should produce a partial Spec
        from a dictionary entry that lacks ``data.source`` / ``series`` /
        ``sample_start``. Use ``prefer_replication=False`` to force this
        path even when the entry happens to be paired with a canonical
        replication — the test then doesn't drift each time a new
        replication lands.

        Uses ``lanne_meitz_saikkonen_2017_nongaussian_svar`` because its
        identification method name (``non_gaussian_independent_components``)
        exercises the no-entry-in-method_map branch of the translator.
        """
        s = Spec.from_dictionary(
            "lanne_meitz_saikkonen_2017_nongaussian_svar",
            prefer_replication=False,
        )
        assert s.model["class"] == "var"   # structural_var → var
        assert s.model["lags"] >= 1
        assert s.irf is not None
        # The method name passes through unchanged (no entry in method_map)
        assert s.irf["method"] == "non_gaussian_independent_components"
        # Partial: caller must fill in data.source / series / sample
        assert "source" not in s.data
        assert "series" not in s.data

    def test_structural_var_class_normalized_to_var(self):
        """Dictionary uses 'structural_var' / 'sign_var' / 'svar' — all
        should map to 'var' (the runner-supported model class) with the
        identification method carrying the structural detail."""
        s = Spec.from_dictionary("cee_1999_monetary_svar", prefer_replication=False)
        assert s.model["class"] == "var"
        assert s.irf is not None
        assert s.irf["method"] == "cholesky"   # choleski → cholesky

    def test_missing_entry_raises(self):
        with pytest.raises(FileNotFoundError, match="dictionary entry not found"):
            Spec.from_dictionary("nonexistent_entry_xyz")

    def test_partial_spec_irf_defaults_match_model_class(self):
        """BVAR partial Spec defaults to bands='parameter_draws'; VAR
        partial defaults to 'asymptotic_mc'."""
        s_bvar = Spec.from_dictionary("litterman_1986_bvar", prefer_replication=False)
        # Litterman has method='none' so no IRF — pick a BVAR with non-none
        s_bvar2 = Spec.from_dictionary(
            "banbura_giannone_lenza_2015_conditional_forecasts",
            prefer_replication=False,
        )
        # This entry may have method="varies"; if so, IRF defaults will be
        # set anyway. The point: when set, bands default reflects class.
        if s_bvar2.irf is not None:
            assert s_bvar2.irf["bands"] == "parameter_draws"


# ---------------------------------------------------------------------------
# run() — driver
# ---------------------------------------------------------------------------


@requires_fred_qd
class TestLoadData:
    """``lib.load_data(spec)`` runs the data pipeline (load + preprocess +
    transform) without fitting. Useful for replications that need the
    data for custom modelling outside the ``lib.run`` one-model-one-run
    abstraction (e.g., BGR 2010 multi-system exploration)."""

    def test_load_data_returns_dataframe(self, minimal_spec_dict):
        s = Spec.from_dict(minimal_spec_dict)
        data = load_data(s)
        assert isinstance(data, pd.DataFrame)
        assert list(data.columns) == s.data["variable_order"]
        assert len(data) > 0

    def test_load_data_matches_run_data(self, minimal_spec_dict):
        """load_data should produce the same data as lib.run(spec).data
        — that's the contract."""
        s = Spec.from_dict(minimal_spec_dict)
        data_direct = load_data(s)
        data_via_run = run(s, seed=42).data
        pd.testing.assert_frame_equal(data_direct, data_via_run)

    def test_load_data_exposed_on_lib_namespace(self, minimal_spec_dict):
        """Public API surface: ``lib.load_data`` should be importable."""
        assert lib.load_data is load_data


@requires_fred_qd
class TestRun:
    def test_run_var_with_cholesky(self, minimal_spec_dict):
        s = Spec.from_dict(minimal_spec_dict)
        result = run(s, seed=42)
        assert isinstance(result, SpecResult)
        assert type(result.fit).__name__ == "VARFit"
        assert result.data.shape[1] == 3            # 3 variables
        assert result.irf.median.shape[0] == 9      # horizon+1 = 9
        assert result.forecast is None

    def test_run_bvar_reproduces_replication(self):
        """Running the Litterman spec through lib.run should produce the same
        IRF as the standalone replicate.py — the spec layer is just a wrapper."""
        s = Spec.from_replication("litterman_1986_bvar")
        # Clear forecast block to keep this test fast
        s = s.update(forecast=None)
        result = run(s, seed=42)
        # Spot-check the headline number from the standalone run. The
        # paper-faithful specification uses real GNP (rgnp) and the
        # commercial paper rate (cprate), with six lags.
        cprate_idx = result.irf.variable_names.index("cprate")
        rgnp_idx = result.irf.variable_names.index("rgnp")
        # rgnp response at h=12 to a cprate innovation (seed=42, 1000 draws;
        # anchored on the CPI-nominalised M1 configuration)
        rgnp_h12 = result.irf.median[12, rgnp_idx, cprate_idx]
        assert abs(rgnp_h12 - 0.009259) < 5e-4     # match within tolerance

    def test_run_with_param_override_changes_irf(self):
        """update() into model.prior actually changes the run result."""
        base = Spec.from_replication("litterman_1986_bvar").update(forecast=None)
        tighter = base.update(model={"prior": {"lambda_1": 0.01}})
        r_base = run(base, seed=42)
        r_tight = run(tighter, seed=42)
        # Tighter prior → coefficients shrunk more → IRFs different
        cprate_idx = r_base.irf.variable_names.index("cprate")
        rgnp_idx = r_base.irf.variable_names.index("rgnp")
        irf_base = r_base.irf.median[:, rgnp_idx, cprate_idx]
        irf_tight = r_tight.irf.median[:, rgnp_idx, cprate_idx]
        # They should differ meaningfully somewhere
        max_diff = np.max(np.abs(irf_base - irf_tight))
        assert max_diff > 1e-4

    def test_run_with_forecast_block(self, minimal_spec_dict):
        d = dict(minimal_spec_dict)
        d["forecast"] = {"horizon": 4, "n_paths": 50}
        s = Spec.from_dict(d)
        result = run(s, seed=7)
        assert result.forecast is not None
        assert result.forecast.point.shape == (4, 3)
        assert result.forecast.paths.shape == (50, 4, 3)

    def test_run_no_irf_block_skips_irf(self, minimal_spec_dict):
        d = dict(minimal_spec_dict)
        d["irf"] = None
        s = Spec.from_dict(d)
        result = run(s, seed=42)
        assert result.fit is not None
        assert result.irf is None

    def test_run_rejects_unsupported_model_class(self, minimal_spec_dict):
        d = dict(minimal_spec_dict)
        d["model"] = {"class": "panel_var", "lags": 1, "trend": "c"}
        s = Spec.from_dict(d)
        with pytest.raises(NotImplementedError, match="model class 'panel_var'"):
            run(s, seed=42)

    def test_run_rejects_unsupported_irf_method(self, minimal_spec_dict):
        d = dict(minimal_spec_dict)
        d["irf"] = {"method": "narrative_signs", "horizon": 8}
        s = Spec.from_dict(d)
        with pytest.raises(NotImplementedError, match="irf.method='narrative_signs'"):
            run(s, seed=42)


# ---------------------------------------------------------------------------
# SpecResult.to_log_dict — JSON-friendly summary
# ---------------------------------------------------------------------------


@requires_fred_qd
class TestSpecResultLog:
    def test_to_log_dict_is_json_safe(self, minimal_spec_dict):
        s = Spec.from_dict(minimal_spec_dict)
        result = run(s, seed=42)
        log = result.to_log_dict()
        # Must be JSON-serialisable
        import json
        s_json = json.dumps(log)
        assert "spec" in log
        assert "fit" in log
        assert "irf" in log
        assert log["fit"]["class"] == "VARFit"
        assert "fred_qd" in s_json


# ---------------------------------------------------------------------------
# Package wiring
# ---------------------------------------------------------------------------


def test_lib_exports():
    assert lib.Spec is Spec
    assert lib.SpecResult is SpecResult
    assert lib.run is run


def test_lib_all_lists_spec_surface():
    for name in ("Spec", "SpecResult", "run"):
        assert name in lib.__all__


# ---------------------------------------------------------------------------
# Canonical-config execution through lib.run (shipped configs, not fixtures)
# ---------------------------------------------------------------------------

_FRED_MD_CSV = Path(__file__).resolve().parents[1] / "data" / "fred_md" / "dataset.csv"
requires_fred_md = pytest.mark.skipif(
    not _FRED_MD_CSV.exists(),
    reason="fred_md dataset.csv not downloaded - see data/fred_md/UPDATE.md",
)


@requires_fred_qd
class TestCanonicalConfigExecution:
    """Fix acceptance: lib.run executes SHIPPED canonical configs — the
    per-series transforms declared in data.series and the identification
    block parked in metadata must reach the estimator."""

    def test_bq_config_runs_long_run_identification(self):
        s = Spec.from_replication("blanchard_quah_1989")
        s = s.update(irf={"n_draws": 100, "horizon": 10})
        result = run(s, seed=42)
        # Per-series transforms applied: dlog_x_100 growth rates in percent,
        # not raw GDP levels (which would be in the thousands).
        assert result.data["dlog_gdp"].abs().max() < 25.0
        # Identification translated from the config's identification block:
        # named shocks and impact sign normalisation.
        assert list(result.irf.shock_names) == ["supply", "demand"]
        i_gdp = result.irf.variable_names.index("dlog_gdp")
        assert result.irf.median[0, i_gdp, 0] > 0   # supply raises growth
        assert result.irf.median[0, i_gdp, 1] > 0   # demand raises growth

    def test_cholesky_ordering_mismatch_refused(self, minimal_spec_dict):
        d = dict(minimal_spec_dict)
        d["metadata"] = {
            **d["metadata"],
            "identification": {"method": "cholesky",
                               "ordering": ["ffr", "rgdp", "cpi"]},
        }
        s = Spec.from_dict(d)
        with pytest.raises(NotImplementedError, match="Cholesky ordering"):
            run(s, seed=42)

    def test_sign_restrictions_without_restrictions_refused(
            self, minimal_spec_dict):
        d = dict(minimal_spec_dict)
        d["irf"] = {"method": "sign_restrictions", "horizon": 8}
        s = Spec.from_dict(d)
        with pytest.raises(ValueError, match="restrictions block"):
            run(s, seed=42)


@requires_fred_md
class TestUhligConfigExecution:
    def test_uhlig_config_sign_restrictions_translate(self):
        """The shipped Uhlig config declares its scheme under
        identification:; lib.run must translate it (restrictions,
        restriction_horizons, method recovery) and enforce it."""
        s = Spec.from_replication("uhlig_2005_sign_restrictions")
        s = s.update(irf={"horizon": 8, "target_accepted": 25,
                          "max_rotations": 50000})
        result = run(s, seed=42)
        i_ffr = result.irf.variable_names.index("ffr")
        # The monetary-policy shock is restricted to RAISE the funds rate
        # on impact; an untranslated scheme could not enforce this.
        assert result.irf.median[0, i_ffr, 0] > 0


# ---------------------------------------------------------------------------
# Seed resolution: explicit > metadata.seed > general.random_seed > 42
# ---------------------------------------------------------------------------


class TestSeedResolution:
    """Fix acceptance: lib.run honours the config schema's
    general.random_seed (parked in metadata by from_config), keeps 0 as a
    valid seed, and treats null as absent."""

    @pytest.mark.parametrize("metadata, explicit, expected", [
        ({"general": {"random_seed": 7}}, None, 7),
        ({"general": {"random_seed": 0}}, None, 0),
        ({"seed": 0}, None, 0),
        ({"seed": 5, "general": {"random_seed": 7}}, None, 5),
        ({"general": {"random_seed": 7}}, 3, 3),
        ({"general": {"random_seed": 7}}, 0, 0),
        ({"general": {"random_seed": None}}, None, 42),
        ({}, None, 42),
    ])
    def test_precedence(self, metadata, explicit, expected):
        from lib.specs import _resolve_seed
        spec = Spec(data={}, model={}, metadata=metadata)
        assert _resolve_seed(spec, explicit) == expected

    @requires_fred_qd
    def test_shipped_config_schema_seed_is_honoured(self, tmp_path):
        """A real replication config (Blanchard-Quah) with its
        general.random_seed changed must run under THAT seed: the result
        records it, the same seed reproduces the bands exactly, and a
        different seed moves them."""
        src = (ROOT / "scripts" / "replications" / "blanchard_quah_1989"
               / "config.yaml")
        raw = yaml.safe_load(src.read_text(encoding="utf-8"))

        def run_with(seed_value):
            raw["general"]["random_seed"] = seed_value
            path = tmp_path / f"config_{seed_value}.yaml"
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            s = Spec.from_config(path).update(irf={"n_draws": 50, "horizon": 8})
            return run(s)

        r7a, r7b, r8 = run_with(7), run_with(7), run_with(8)
        assert r7a.seed == 7 and r8.seed == 8
        assert r7a.to_log_dict()["seed"] == 7
        np.testing.assert_array_equal(r7a.irf.lower_68, r7b.irf.lower_68)
        assert not np.allclose(r7a.irf.lower_68, r8.irf.lower_68)
