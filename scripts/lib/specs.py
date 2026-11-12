"""Specifications — declarative bridge between the dictionary and the runner.

A :class:`Spec` is a self-contained description of a model run: data source,
transforms, model class, hyperparameters, identification, and optional IRF /
forecast settings. :func:`run` is the driver: given a supported Spec (var or
bvar fits; cholesky, sign-restriction, long-run or proxy-SVAR identification)
it loads the data, applies preprocess + transforms, fits the model, computes
IRFs / forecasts, and returns a :class:`SpecResult`; other specs raise
``NotImplementedError``.

The point of this layer:

* The model dictionary (``skills/model-dictionary/entries/*.yaml``) is
  *descriptive*. The replication configs (``scripts/replications/*/config.yaml``)
  are *implementation-specific*. Specs sit in between: a programmatic,
  editable representation that can be constructed from a config, mutated,
  composed, and re-run without re-writing the replicate.py boilerplate.

* The canonical replications under ``scripts/replications/`` become "frozen
  specs" — one specific Spec instance per paper, kept around for the public
  gallery. Any other analysis is a Spec on the fly:

    >>> spec = lib.Spec.from_replication("litterman_1986_bvar")
    >>> tighter = spec.update(model={"prior": {"lambda_1": 0.05}})
    >>> result = lib.run(tighter)
    >>> result.irf.median[0]              # IRF on impact
    >>> result.fit.fit_info["prior"]      # resolved prior settings

The runner supports the model classes most replications use today:
``var`` and ``bvar``. Identification: ``cholesky``, ``sign_restrictions``,
``proxy_svar``, ``long_run``, plus pass-through for forecasting-only runs.
Other model classes (``mfvar``, ``tvp_var``, ``dfm``) can be added by
extending :func:`_dispatch_fit`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Spec dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Spec:
    """A self-contained model-run specification.

    Top-level blocks (dicts so they're easy to deep-merge and serialise):

    * ``data`` — source, sample, series, transforms, variable_order, preprocess.
    * ``model`` — class ('var' or 'bvar'), lags, trend, prior, ...
    * ``irf`` — identification + IRF settings (method, horizon, bands, ...).
      None to skip IRF computation.
    * ``forecast`` — iterated forecast from the end of the sample (horizon,
      n_paths, parameter_uncertainty). Drives :func:`lib.iterated_forecast`.
      None to skip.
    * ``forecast_evaluation`` — rolling-window OOS evaluation settings
      (window_size, horizon, step, horizons_to_report). Distinct from
      ``forecast``: drives a multi-origin rolling refit + scoring pipeline,
      not a single end-of-sample forecast. Currently consumed by replication
      scripts directly (not by :func:`run`). None to skip.
    * ``metadata`` — free-form provenance (paper citation, notes, etc.).

    Specs are immutable. Use :meth:`update` for deep-merge overrides.

    The block contents follow the existing replication config.yaml shape,
    so any canonical replication config can be loaded as a Spec via
    :meth:`from_replication`. :func:`run` executes only the supported
    model / identification combinations; other replications use their
    dedicated scripts.
    """

    data: dict[str, Any]
    model: dict[str, Any]
    irf: dict[str, Any] | None = None
    forecast: dict[str, Any] | None = None
    forecast_evaluation: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # -------- editing --------

    def update(self, **overrides) -> "Spec":
        """Return a new Spec with deep-merged overrides.

        Override values are merged into the existing block dicts (one level
        deep — nested dicts in fields like ``data.transforms`` are themselves
        deep-merged recursively, so partial updates work the right way).

        Example::

            spec = Spec.from_replication("litterman_1986_bvar")
            tighter = spec.update(
                model={"prior": {"lambda_1": 0.05}},
                irf={"horizon": 40},
            )

        Result: ``tighter.model["prior"]["lambda_1"] == 0.05`` while all
        other prior fields (lambda_2, mu, delta, rw_deltas, ...) are
        preserved from the original.
        """
        kwargs = {}
        allowed = {"data", "model", "irf", "forecast", "forecast_evaluation", "metadata"}
        for k, v in overrides.items():
            if k not in allowed:
                raise ValueError(
                    f"unknown spec field {k!r}; allowed: {sorted(allowed)}"
                )
            current = getattr(self, k)
            if current is None:
                kwargs[k] = copy.deepcopy(v)
            elif v is None:
                kwargs[k] = None
            else:
                kwargs[k] = _deep_merge(current, v)
        return replace(self, **kwargs)

    # -------- (de)serialisation --------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Spec":
        # Accept either flat dict or already-Spec; tolerate missing optional fields.
        return cls(
            data=copy.deepcopy(d.get("data", {})),
            model=copy.deepcopy(d.get("model", {})),
            irf=copy.deepcopy(d.get("irf")) if d.get("irf") else None,
            forecast=copy.deepcopy(d.get("forecast")) if d.get("forecast") else None,
            forecast_evaluation=(
                copy.deepcopy(d.get("forecast_evaluation"))
                if d.get("forecast_evaluation") else None
            ),
            metadata=copy.deepcopy(d.get("metadata", {})),
        )

    @classmethod
    def from_config(cls, config_path: str | Path) -> "Spec":
        """Load a Spec from a config.yaml file (e.g., a replication folder).

        ``forecast`` (iterated end-of-sample forecast) and
        ``forecast_evaluation`` (rolling-window OOS comparison) are loaded
        into separate fields. They are semantically distinct and must not
        be aliased.
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"config not found: {path}")
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"config {path} did not parse to a dict")
        # Anything at top level that isn't a recognised Spec block gets
        # folded into metadata, so replications can keep their own
        # bespoke blocks (e.g., GLP's `glp:` optimiser settings, BGR's
        # ad-hoc keys, paper-specific extension fields) without losing
        # them through Spec.from_config. The recognised blocks are
        # those that map to Spec dataclass fields.
        recognised = {
            "data", "model", "irf", "forecast", "forecast_evaluation",
            "metadata",
        }
        metadata = dict(raw.get("metadata", {}))
        for k, v in raw.items():
            if k in recognised:
                continue
            metadata[k] = v
        return cls(
            data=copy.deepcopy(raw.get("data", {})),
            model=copy.deepcopy(raw.get("model", {})),
            irf=copy.deepcopy(raw.get("irf")) if raw.get("irf") else None,
            forecast=copy.deepcopy(raw.get("forecast")) if raw.get("forecast") else None,
            forecast_evaluation=(
                copy.deepcopy(raw.get("forecast_evaluation"))
                if raw.get("forecast_evaluation") else None
            ),
            metadata=metadata,
        )

    @classmethod
    def from_replication(cls, replication_id: str) -> "Spec":
        """Load a Spec from ``scripts/replications/<id>/config.yaml``."""
        # Resolve relative to lib package (this file's parent.parent)
        lib_root = Path(__file__).resolve().parent
        scripts_root = lib_root.parent
        path = scripts_root / "replications" / replication_id / "config.yaml"
        return cls.from_config(path)

    @classmethod
    def from_dictionary(
        cls, entry_id: str, prefer_replication: bool = True,
    ) -> "Spec":
        """Load a Spec from a dictionary entry (``skills/model-dictionary/entries/<id>.yaml``).

        The dictionary describes a paper's specification; a runnable Spec
        needs implementation-specific details (FRED codes, exact sample
        dates, IRF horizon, etc.). Two modes:

        * **Paired replication exists** (``scripts/replications/<id>/``):
          delegate to :meth:`from_replication` — the implementation config
          is more complete than the descriptive entry. This is the
          orchestrator's Mode A path.

        * **No paired replication**: translate the descriptive entry into
          a *partial* Spec — model block + prior + identification filled
          from the entry, but ``data.series`` (FRED codes) and
          ``data.sample_start/end`` left absent. The caller (typically
          the orchestrator's Mode B / Mode C path) is expected to fill
          these via ``.update(data={"source": ..., "series": {...}, ...})``
          before calling :func:`run`.

        Parameters
        ----------
        entry_id : str
            The dictionary entry id (filename stem under
            ``skills/model-dictionary/entries/``).
        prefer_replication : bool, default True
            If True and a paired replication exists, delegate to it.
            Pass False to force loading from the descriptive entry even
            when a replication is available.

        Returns
        -------
        Spec
            The replication config as a Spec if delegating to a
            replication (:func:`run` executes it only for supported
            model / identification combinations); partial (data.series
            and data.sample_start/end absent) otherwise.

        Raises
        ------
        FileNotFoundError
            If neither a paired replication nor a dictionary entry exists
            for ``entry_id``.
        """
        lib_root = Path(__file__).resolve().parent
        scripts_root = lib_root.parent
        project_root = lib_root.parent.parent

        if prefer_replication:
            rep_path = scripts_root / "replications" / entry_id / "config.yaml"
            if rep_path.exists():
                return cls.from_config(rep_path)

        entry_path = (
            project_root / "skills" / "model-dictionary"
            / "entries" / f"{entry_id}.yaml"
        )
        if not entry_path.exists():
            raise FileNotFoundError(
                f"dictionary entry not found: {entry_path}. "
                f"Available entries are in skills/model-dictionary/entries/."
            )
        raw = yaml.safe_load(entry_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"entry {entry_path} did not parse to a dict")
        return cls._from_dictionary_entry(raw)

    @classmethod
    def _from_dictionary_entry(cls, entry: dict) -> "Spec":
        """Translate a descriptive dictionary entry into a partial Spec."""

        def _slug(name: str) -> str:
            return name.lower().replace(" ", "_").replace("-", "_")

        # Dictionary uses descriptive class names (structural_var, sign_var, ...)
        # that mostly all reduce to "var" with different identification
        # methods. Map to runner-supported classes; leave unsupported
        # specialist classes (lp, dfm, favar, panel_var, tvp_var, mfvar,
        # filter, shock_series) untouched — they'll surface as a clear
        # NotImplementedError from lib.run when the user tries to execute.
        class_map = {
            "structural_var": "var",
            "svar": "var",
            "sign_var": "var",
        }
        method_map = {
            "choleski": "cholesky",
            "long_run_restrictions": "long_run",
            "external_instrument": "proxy_svar",
            "sign_restrictions": "sign_restrictions",
            "narrative_sign_restrictions": "sign_restrictions",
        }

        model_entry = entry.get("model") or {}
        cls_raw = model_entry.get("class", "var")
        # Lags may be descriptive ("varies by application", "user-defined", ...)
        # in unpaired entries. Coerce to int when possible; otherwise default
        # to 4 — the caller is expected to override via .update() before run.
        lags_raw = model_entry.get("lags", 4)
        try:
            lags_int = int(lags_raw)
        except (TypeError, ValueError):
            lags_int = 4
        model_block: dict[str, Any] = {
            "class": class_map.get(cls_raw, cls_raw),
            "lags": lags_int,
            "trend": "c",
        }
        hp = model_entry.get("hyperparameters") or {}
        if hp and model_block["class"] == "bvar":
            prior_keys = (
                "lambda_1", "lambda_2", "lambda_3", "lambda_4",
                "mu", "delta", "asymmetric",
            )
            prior = {k: hp[k] for k in prior_keys if k in hp}
            if prior:
                model_block["prior"] = prior

        endogenous = (model_entry.get("variables") or {}).get("endogenous") or []
        data_block: dict[str, Any] = {}
        if endogenous:
            data_block["variable_order"] = [_slug(v["name"]) for v in endogenous]
            data_block["transforms"] = {
                _slug(v["name"]): v.get("transform", "level")
                for v in endogenous if "transform" in v
            }
        # source, series (FRED codes), sample_start/end, preprocess: not
        # in the descriptive entry — caller fills these in via .update().

        ident = entry.get("identification") or {}
        method = ident.get("method", "none")
        if method == "none":
            irf_block = None
        else:
            # Heuristic: some entries spell out a long descriptive method
            # (e.g., "Gibbs sampling — ..."). Only translate when the raw
            # token matches our known map; otherwise pass through so
            # lib.run raises a clear NotImplementedError.
            method_norm = method_map.get(method.strip().lower().split()[0]
                                          if isinstance(method, str) else method,
                                          method)
            irf_block = {
                "method": method_norm,
                "horizon": 20,
                "bands": (
                    "parameter_draws" if model_block["class"] == "bvar"
                    else "asymptotic_mc"
                ),
                "n_draws": 1000,
            }

        # Carry the source entry for orchestrator inspection / debugging.
        metadata = {"dictionary_entry": copy.deepcopy(entry)}

        return cls(
            data=data_block,
            model=model_block,
            irf=irf_block,
            forecast=None,
            forecast_evaluation=None,
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class SpecResult:
    """Output of :func:`run`.

    Attributes
    ----------
    spec : Spec
        The input spec (for provenance).
    data : pd.DataFrame
        The post-preprocess + post-transform data passed to the estimator.
    fit : VARFit subclass
        The fitted model. Always present.
    irf : IRFContainer or None
        Computed if ``spec.irf`` was set.
    forecast : ForecastContainer or None
        Computed if ``spec.forecast`` was set.
    extras : dict
        Anything else the runner needs to expose (e.g., the proxy_svar
        first-stage F-stat). Keyed by step name.
    timestamp_utc : str
        ISO timestamp at the end of the run.
    seed : int or None
        The random seed the run actually used (see :func:`run`).
    """

    spec: Spec
    data: pd.DataFrame
    fit: Any
    irf: Any = None
    forecast: Any = None
    extras: dict[str, Any] = field(default_factory=dict)
    timestamp_utc: str = ""
    seed: int | None = None

    def to_log_dict(self) -> dict[str, Any]:
        """Compact JSON-friendly summary suitable for research_log.json."""
        out: dict[str, Any] = {
            "spec": self.spec.to_dict(),
            "timestamp_utc": self.timestamp_utc,
            "seed": self.seed,
            "data": {
                "n_obs": int(len(self.data)),
                "sample_start": str(self.data.index[0].date()) if len(self.data) else None,
                "sample_end": str(self.data.index[-1].date()) if len(self.data) else None,
                "variables": list(self.data.columns),
            },
            "fit": {
                "class": type(self.fit).__name__,
                "p": int(getattr(self.fit, "p", 0)),
                "n_vars": int(getattr(self.fit, "n_vars", 0)),
                "stable": bool(self.fit.stable()) if hasattr(self.fit, "stable")
                          and callable(getattr(self.fit, "stable", None)) else None,
                "fit_info": _to_json_safe(getattr(self.fit, "fit_info", {})),
            },
            "extras": _to_json_safe(self.extras),
        }
        if self.irf is not None:
            out["irf"] = {
                "shape": list(self.irf.median.shape),
                "horizon": int(self.irf.horizons[-1]) if hasattr(self.irf, "horizons") else None,
                "variable_names": list(getattr(self.irf, "variable_names", [])),
            }
        if self.forecast is not None:
            out["forecast"] = {
                "point_shape": list(self.forecast.point.shape),
                "horizon": int(self.forecast.horizons[-1]),
                "n_paths": self.forecast.metadata.get("n_paths"),
                "parameter_uncertainty": self.forecast.metadata.get("parameter_uncertainty"),
            }
        return out


# ---------------------------------------------------------------------------
# run() — driver
# ---------------------------------------------------------------------------


def run(spec: Spec, seed: int | None = None) -> SpecResult:
    """Execute a Spec end-to-end.

    Pipeline:
      1. Load data via ``lib.load_dataset`` (source from ``spec.data["source"]``).
      2. Apply preprocess steps (deflate / nominalise / aggregate).
      3. Apply per-column transforms via ``lib.transform``.
      4. Restrict to ``spec.data["variable_order"]`` (model variables).
      5. Fit the model — dispatched on ``spec.model["class"]``.
      6. Compute IRF if ``spec.irf`` is set — dispatched on
         ``spec.irf["method"]``, falling back to the config's
         ``identification.method`` (parked in ``spec.metadata``). A spec
         that declares NEITHER raises ``ValueError``: the runner never
         silently defaults to Cholesky, which would execute a different
         model than the one specified.
      7. Compute forecast if ``spec.forecast`` is set.

    Parameters
    ----------
    spec : Spec
    seed : int, optional
        Random seed for any stochastic component (IRF bands, sign-restriction
        rotations, forecast density paths). Resolution order: this argument;
        else ``metadata["seed"]``; else the config's ``general.random_seed``
        (which :meth:`Spec.from_config` parks in ``metadata["general"]``);
        else 42. ``0`` is a valid seed and is kept; a null value counts as
        absent. The seed used is recorded on the result (``SpecResult.seed``).

    Returns
    -------
    SpecResult

    Raises
    ------
    ValueError
        If ``spec.irf`` is set but no identification method is declared
        anywhere (no silent Cholesky default), or if the prepared data
        has internal gaps (the retained rows must be a contiguous suffix
        of the raw calendar — silent row-drops would corrupt lags).
    NotImplementedError
        If a model class or identification method isn't yet supported by the
        dispatcher. Add the dispatch in ``_dispatch_fit`` / ``_dispatch_irf``.
    """
    # Local import to avoid a circular dependency at module load time
    import lib

    seed = _resolve_seed(spec, seed)
    np.random.seed(seed)

    # 1-3. Data load + preprocess + transform
    data = _load_and_transform(spec, lib)

    # 4. Fit
    fit, fit_extras = _dispatch_fit(spec, data, lib, seed=seed)

    extras: dict[str, Any] = {"fit": fit_extras} if fit_extras else {}

    # 5. IRF (optional)
    irf, irf_extras = None, None
    if spec.irf is not None:
        irf, irf_extras = _dispatch_irf(spec, fit, data, lib, seed=seed)
        if irf_extras:
            extras["irf"] = irf_extras

    # 6. Forecast (optional)
    forecast = None
    if spec.forecast is not None:
        forecast = _run_forecast(spec, fit, lib, seed=seed)

    return SpecResult(
        spec=spec,
        data=data,
        fit=fit,
        irf=irf,
        forecast=forecast,
        extras=extras,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        seed=seed,
    )


def _resolve_seed(spec: Spec, seed: int | None) -> int:
    """Effective seed: explicit argument > metadata.seed > general.random_seed > 42."""
    if seed is not None:
        return int(seed)
    meta = spec.metadata or {}
    general = meta.get("general")
    candidates = (
        meta.get("seed"),
        general.get("random_seed") if isinstance(general, dict) else None,
    )
    for cand in candidates:
        if cand is not None:
            return int(cand)
    return 42


# ---------------------------------------------------------------------------
# Internal: data + preprocess + transform
# ---------------------------------------------------------------------------


def load_data(spec: Spec) -> pd.DataFrame:
    """Load + preprocess + transform a Spec's ``data`` block, without fitting.

    Useful when a replication uses the spec for the data pipeline but does
    custom modelling (e.g., multi-system exploration like BGR 2010, or any
    workflow that doesn't fit ``lib.run``'s "one model, one run" shape).

    Equivalent to the data-loading step of ``lib.run`` but with no fit
    dispatch, no IRF, no forecast. ``spec.data["variable_order"]`` (if set)
    is honoured and rows with any NaN are dropped.

    Parameters
    ----------
    spec : Spec

    Returns
    -------
    pd.DataFrame
        Wide-format data with ``df.attrs`` preserved from
        ``lib.load_dataset`` plus any ``preprocess_applied`` entries
        appended by the preprocess steps.
    """
    # Local import (matches lib.run's pattern) to avoid circular dependency
    import lib
    return _load_and_transform(spec, lib)


def _load_and_transform(spec: Spec, lib) -> pd.DataFrame:
    data_cfg = spec.data
    series_specs = data_cfg.get("series") or []
    if isinstance(series_specs, list):
        series_map = {s["name"]: s["fred_code"] for s in series_specs}
    elif isinstance(series_specs, dict):
        series_map = dict(series_specs)
    else:
        raise ValueError(
            f"data.series must be a list of {{name, fred_code}} or a dict; "
            f"got {type(series_specs).__name__}"
        )

    raw = lib.load_dataset(
        source=data_cfg["source"],
        series=series_map,
        start=data_cfg.get("sample_start"),
        end=data_cfg.get("sample_end"),
    )

    # Preprocess steps (operate on raw before transform)
    for step in data_cfg.get("preprocess") or []:
        op = step["operation"]
        if op == "nominalise":
            raw[step["output"]] = lib.preprocess.nominalise(
                raw[step["real"]], raw[step["deflator"]],
            )
        elif op == "deflate":
            raw[step["output"]] = lib.preprocess.deflate(
                raw[step["nominal"]], raw[step["deflator"]],
            )
        elif op == "aggregate":
            # In-place reassignment of raw to the aggregated frame
            raw = lib.preprocess.aggregate(
                raw, target_freq=step["target_freq"], method=step["method"],
            )
        else:
            raise ValueError(f"unknown preprocess operation: {op!r}")

    transforms = dict(data_cfg.get("transforms") or {})
    # Canonical replication configs declare transforms PER VARIABLE
    # (data.variables: [{name, fred_code, transform}, ...]); an empty flat
    # map must not silently mean "no transforms".
    # Canonical configs declare per-series transforms inside data.series
    # entries ({name, fred_code, transform}); some older specs use a
    # data.variables list. Read BOTH — an empty flat map must never
    # silently mean "no transforms" for a config that declares them.
    for key in ("series", "variables"):
        var_entries = data_cfg.get(key)
        if isinstance(var_entries, list):
            for v in var_entries:
                if isinstance(v, dict) and v.get("name") and v.get("transform"):
                    transforms.setdefault(str(v["name"]), str(v["transform"]))
    var_order = data_cfg.get("variable_order") or list(raw.columns)
    out = lib.transform(raw[var_order], transforms).dropna()
    # The retained rows must be a contiguous SUFFIX of the raw calendar
    # (transforms legitimately create leading NaNs; internal NaNs mean
    # gaps that row-based lags would silently paper over).
    if len(out) and not out.index.equals(raw.index[len(raw) - len(out):]):
        raise ValueError(
            "spec data preparation: transformed data has internal gaps "
            "(NaNs inside the sample, not just at the start). Impute or "
            "select a contiguous sample before estimation."
        )
    return out


# ---------------------------------------------------------------------------
# Internal: fit dispatcher
# ---------------------------------------------------------------------------


def _dispatch_fit(
    spec: Spec, data: pd.DataFrame, lib, seed: int,
) -> tuple[Any, dict[str, Any] | None]:
    """Fit the model. Returns (fit_object, optional extras dict)."""
    model_cfg = spec.model
    model_class = model_cfg.get("class")
    lags = int(model_cfg["lags"])
    trend = model_cfg.get("trend", "c")

    if model_class == "var":
        return lib.fit_var(data, lags=lags, trend=trend), None

    if model_class == "bvar":
        prior_cfg = model_cfg.get("prior") or {}
        prior = lib.MinnesotaPrior(
            lambda_1=prior_cfg.get("lambda_1", 0.1),
            lambda_2=prior_cfg.get("lambda_2", 0.5),
            lambda_3=prior_cfg.get("lambda_3", 1.0),
            lambda_4=prior_cfg.get("lambda_4", 100.0),
            mu=prior_cfg.get("mu", 1.0),
            delta=prior_cfg.get("delta", 1.0),
            rw_deltas=prior_cfg.get("rw_deltas"),
            asymmetric=prior_cfg.get("asymmetric", False),
        )
        return lib.fit_minnesota_bvar(data, lags=lags, prior=prior, trend=trend), None

    raise NotImplementedError(
        f"model class {model_class!r} not yet supported by lib.run; "
        f"supported: 'var', 'bvar'. Extend lib.specs._dispatch_fit."
    )


# ---------------------------------------------------------------------------
# Internal: IRF dispatcher
# ---------------------------------------------------------------------------


def _dispatch_irf(
    spec: Spec, fit, data: pd.DataFrame, lib, seed: int,
) -> tuple[Any, dict[str, Any] | None]:
    """Compute IRF. Returns (irf_container, optional extras)."""
    irf_cfg = spec.irf or {}
    # Canonical configs put identification settings (method, restrictions,
    # shock names, sign normalisation) under a top-level `identification:`
    # block, which from_config parks in metadata. Translate the supported
    # fields into the runner's inputs; explicit irf keys win.
    _ident_cfg = (spec.metadata or {}).get("identification") or {}
    _translatable = ("restrictions", "restriction_horizons",
                     "target_accepted", "max_rotations",
                     "shock_names", "sign_normalization")
    irf_cfg = {
        **{k: _ident_cfg[k] for k in _translatable if k in _ident_cfg},
        **irf_cfg,
    }
    method = irf_cfg.get("method")
    if method is None:
        # Recover the declared method from the identification block — a
        # missing declaration must FAIL, never silently default to
        # Cholesky and execute a different model.
        declared = _ident_cfg.get("method")
        _map = {
            "cholesky": "cholesky", "recursive": "cholesky",
            "sign_restrictions": "sign_restrictions",
            "long_run": "long_run",
            "external_instrument": "proxy_svar", "proxy_svar": "proxy_svar",
        }
        if declared in _map:
            method = _map[declared]
        elif declared is None:
            raise ValueError(
                "lib.run: no irf.method and no identification.method "
                "declared. Refusing to default to Cholesky — state the "
                "identification explicitly (or use the replication's own "
                "replicate.py, which encodes it)."
            )
        else:
            raise NotImplementedError(
                f"lib.run: declared identification.method={declared!r} is "
                f"not supported by the generic runner (supported: "
                f"{sorted(_map)}). Use the replication's own replicate.py."
            )
    horizon = int(irf_cfg["horizon"])
    bands = irf_cfg.get("bands", "asymptotic_mc")
    n_draws = int(irf_cfg.get("n_draws", 1000))

    if method == "cholesky":
        if _ident_cfg.get("ordering"):
            _ord = [str(v) for v in _ident_cfg["ordering"]]
            if _ord != list(fit.variable_names):
                raise NotImplementedError(
                    "lib.run: the config declares a Cholesky ordering "
                    f"{_ord!r} that differs from the fitted variable order "
                    f"{list(fit.variable_names)!r}. The generic runner does "
                    "not reorder — set data.variable_order to the Cholesky "
                    "ordering, or use the replication's own replicate.py."
                )
        return lib.cholesky_irf(
            fit, horizon=horizon, bands=bands, n_draws=n_draws, seed=seed,
        ), None

    if method == "sign_restrictions":
        _restr = irf_cfg.get("restrictions")
        if _restr is None:
            raise ValueError(
                "lib.run: sign-restriction identification needs a "
                "restrictions block ({shock: {variable: sign}}) in the "
                "irf or identification config — or use the replication's "
                "own replicate.py, which encodes the full scheme."
            )
        return lib.sign_restricted_irf(
            fit,
            horizon=horizon,
            sign_restrictions=_restr,
            restriction_horizons=int(irf_cfg.get("restriction_horizons", 1)),
            target_accepted=int(irf_cfg.get("target_accepted", 1000)),
            max_rotations=int(irf_cfg.get("max_rotations", 100_000)),
            seed=seed,
        ), None

    if method == "proxy_svar":
        # The caller is responsible for providing the instrument via
        # `irf_cfg["instrument"]` (an array-like aligned with data.index).
        instrument = irf_cfg.get("instrument")
        if instrument is None:
            raise ValueError(
                "irf.method='proxy_svar' requires irf['instrument'] = array of "
                "shock realisations aligned with data.index"
            )
        irf_obj, fs = lib.proxy_svar_irf(
            fit=fit,
            instrument=np.asarray(instrument),
            policy_var=irf_cfg["policy_variable"],
            horizon=horizon,
            shock_name=irf_cfg.get("shock_name", "shock"),
            bands=bands,
            n_draws=n_draws,
            seed=seed,
            normalization=irf_cfg.get("normalization", "unit_policy"),
        )
        return irf_obj, {"first_stage": fs}

    if method == "long_run":
        return lib.long_run_irf(
            fit, horizon=horizon, bands=bands, n_draws=n_draws, seed=seed,
            shock_names=irf_cfg.get("shock_names"),
            sign_normalization=irf_cfg.get("sign_normalization"),
        ), None

    raise NotImplementedError(
        f"irf.method={method!r} not yet supported by lib.run; "
        f"supported: 'cholesky', 'sign_restrictions', 'proxy_svar', 'long_run'."
    )


# ---------------------------------------------------------------------------
# Internal: forecast
# ---------------------------------------------------------------------------


def _run_forecast(spec: Spec, fit, lib, seed: int) -> Any:
    """Compute an iterated forecast per spec.forecast settings."""
    fc_cfg = spec.forecast or {}
    return lib.iterated_forecast(
        fit,
        horizon=int(fc_cfg["horizon"]),
        n_paths=fc_cfg.get("n_paths"),
        parameter_uncertainty=bool(fc_cfg.get("parameter_uncertainty", False)),
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursive dict merge — overlay wins, with nested dicts merged in-place."""
    out = copy.deepcopy(base)
    for k, v in overlay.items():
        if (
            isinstance(v, dict)
            and isinstance(out.get(k), dict)
        ):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _to_json_safe(obj):
    """Recursively normalise numpy scalars/arrays to plain Python for JSON."""
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj
