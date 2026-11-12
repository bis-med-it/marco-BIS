"""Cache-on-fetch infrastructure for runtime data sources.

Fetched series persist to disk in a per-dataset directory (``data/<dataset_id>/``)
with per-series provenance metadata. Subsequent calls (from any project) read
from the cache without hitting the API. This turns ephemeral API fetches into a
durable, growing data asset.

Public API:

>>> data = lib.fetch_data(
...     api="ons",
...     series={"indpro": "economy/economicoutputandproductivity/output/timeseries/k222/diop"},  # ONS: URL path
...     dataset_id="uk_macro",                    # cache destination
...     start="1992-10", end="2026-04",
... )

Each ``api`` resolves to a :class:`BaseFetcher` subclass registered in
:mod:`lib.data.fetchers`. The fetcher is responsible only for talking to the
external source and returning ``{alias: pd.Series}``; this module handles cache
read/write, metadata merging, and registry updates.

Cache layout::

    data/<dataset_id>/
    ├── dataset.csv        # all fetched series, joined on date index
    └── metadata.yaml      # per-series provenance (api, source_spec, endpoint,
                           # fetched_at, sample bounds, n_obs)

The registry's gitignored local overlay (``data/registry.local.yaml``) is
auto-updated with a new entry when
the dataset is first created.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml

from .loader import load_dataset
from .registry import load_registry

# Registry of available fetchers: api_name -> BaseFetcher subclass.
# Populated by lib.data.fetchers/__init__.py.
_FETCHER_REGISTRY: dict[str, type["BaseFetcher"]] = {}


def register_fetcher(api_name: str, fetcher_class: type["BaseFetcher"]) -> None:
    """Register a fetcher class under ``api_name``.

    Called by source-specific modules at import time. Allows
    ``lib.fetch_data(api="ons", ...)`` to dispatch to ``OnsFetcher``.
    """
    _FETCHER_REGISTRY[api_name] = fetcher_class


def get_fetcher(api_name: str) -> "BaseFetcher":
    """Instantiate the fetcher registered under ``api_name``.

    Raises
    ------
    KeyError
        If ``api_name`` is not registered. Hint to import the relevant
        fetcher module (e.g., ``import lib.data.fetchers.ons``) — though
        importing ``lib.data.fetchers`` auto-registers all built-ins.
    """
    if api_name not in _FETCHER_REGISTRY:
        available = sorted(_FETCHER_REGISTRY)
        raise KeyError(
            f"No fetcher registered for api={api_name!r}. "
            f"Registered: {available}. "
            f"If you've added a new source, ensure its module is imported."
        )
    return _FETCHER_REGISTRY[api_name]()


class BaseFetcher(ABC):
    """Source-specific fetcher protocol.

    A fetcher knows how to talk to one external data source (ONS, BoE iadb,
    ECB SDMX, …) and return time series. It is *stateless* — instantiation
    has no side effects. The orchestration in :func:`fetch_data` handles
    caching, provenance, and registry updates.
    """

    #: Short identifier used in api lookups and metadata (e.g., ``"ons"``).
    name: str = ""

    #: Human-readable description for documentation / metadata.
    description: str = ""

    #: URL pattern or base endpoint. Documentation only; the fetcher
    #: constructs concrete URLs internally.
    endpoint: str = ""

    @abstractmethod
    def fetch(
        self,
        series: Mapping[str, Any],
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> dict[str, pd.Series]:
        """Fetch the requested series from the source.

        Parameters
        ----------
        series : dict
            ``{alias: source_spec}``. ``source_spec`` format is fetcher-specific
            (a bare code, a tuple ``(code, dataset)``, etc.). Document the
            expected shape in the fetcher's docstring.
        start, end : str or pd.Timestamp, optional
            Sample bounds. May be ignored by some APIs that return the
            full history regardless; the cache merge filters at load time.

        Returns
        -------
        dict[str, pd.Series]
            ``{alias: pd.Series}``. Each series is indexed by date.
            Missing aliases should be omitted from the result with a warning
            (don't raise — caller decides what to do with partial fetches).
        """
        ...


def fetch_data(
    api: str,
    series: Mapping[str, Any],
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    dataset_id: str | None = None,
    refresh: str = "cache_first",
    dropna: bool = True,
    *,
    registry_path: str | Path | None = None,
) -> pd.DataFrame:
    """Fetch series from a runtime API, cache them, return as a DataFrame.

    Wraps :class:`BaseFetcher` subclasses with cache-on-fetch behaviour:

    1. Identify which requested series are already in ``data/<dataset_id>/``.
    2. Fetch only the missing ones via the source-specific fetcher.
    3. Merge new series into ``dataset.csv``; update ``metadata.yaml``.
    4. Auto-register the dataset in ``data/registry.local.yaml`` if it's new.
    5. Return ``lib.load_dataset(source=dataset_id, series=all_requested)``.

    Parameters
    ----------
    api : str
        Source identifier (e.g., ``"ons"``, ``"boe_iadb"``, ``"csv_file"``).
        Resolved via :func:`get_fetcher`.
    series : dict
        ``{alias: source_spec}``. The exact shape of ``source_spec`` depends
        on the fetcher: a string code for most sources, a URL path string for ONS.
    start, end : str or pd.Timestamp, optional
        Sample bounds for the returned DataFrame.
    dataset_id : str, optional
        Cache destination under ``data/``. Defaults to ``api`` (so
        ``api="ons"`` caches to ``data/ons/``). Set explicitly to combine data
        from multiple APIs in one dataset (``dataset_id="uk_macro"`` aggregates
        ONS + BoE iadb).
    refresh : str, default "cache_first"
        - ``"cache_first"``: skip API for series already in cache
        - ``"force"``: re-fetch all requested series (overwrites cache)
        - ``"stale_only"``: re-fetch only series whose ``fetched_at`` exceeds
          the source's cache-invalidation policy (not implemented; raises)
    dropna : bool, default True
        Passed through to :func:`lib.load_dataset`. Set False for series
        with legitimate internal gaps (e.g. event-based instrument series
        with excluded months) and handle the NaNs explicitly — the loader
        refuses to silently row-drop internal gaps.
    registry_path : path, optional
        Override the registry path. Used in tests.

    Returns
    -------
    pd.DataFrame
        Loaded data with provenance in ``df.attrs`` (same shape as
        :func:`lib.load_dataset`). ``df.attrs["fetcher"]`` carries the API name
        and per-series fetch metadata.
    """
    if dataset_id is None:
        dataset_id = api
    if refresh not in ("cache_first", "force", "stale_only"):
        raise ValueError(
            f"refresh must be 'cache_first' | 'force' | 'stale_only'; got {refresh!r}"
        )
    if refresh == "stale_only":
        # There is no cache-age policy, so raise instead of behaving like
        # cache_first.
        raise NotImplementedError(
            "refresh='stale_only' is not implemented (there are no cache-age "
            "policies). Use 'force' to re-fetch or 'cache_first' to accept "
            "the cache."
        )

    # 1. Resolve fetcher
    fetcher = get_fetcher(api)

    # 2. Locate the dataset directory in the cache
    registry = load_registry(registry_path)
    project_root: Path = registry["_project_root"]
    dataset_dir = project_root / "data" / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)
    csv_path = dataset_dir / "dataset.csv"
    meta_path = dataset_dir / "metadata.yaml"

    # 3. Decide what to fetch
    spec = dict(series)
    existing_cols: set[str] = set()
    if csv_path.exists():
        # Read the cache even under force refresh: force refetches the
        # REQUESTED aliases but must preserve other cached columns.
        existing = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        existing_cols = set() if refresh == "force" else set(existing.columns)
    else:
        existing = None

    # Cached-alias reuse requires the SOURCE to match stored provenance:
    # the same alias requested with a different source specification must be
    # refetched, or the user silently keeps estimating with the old series.
    stale_aliases: set[str] = set()
    if existing_cols:
        _stored: dict = {}
        if meta_path.exists():
            try:
                _meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
                _stored = {v.get("series_id"): v for v in _meta.get("variables", [])}
            except Exception:
                _stored = {}
        for alias, src in spec.items():
            if alias not in existing_cols:
                continue
            # Identity = (provider, source spec): the SAME series code
            # requested from a DIFFERENT provider is a different series
            # until proven otherwise. Missing provenance (no metadata.yaml,
            # no record for the alias, or empty fields) cannot be verified
            # and is refetched.
            rec = _stored.get(alias) or {}
            stored_src = rec.get("source_spec")
            stored_api = rec.get("api")
            src_changed = stored_src is None or str(stored_src) != str(src)
            api_changed = stored_api is None or str(stored_api) != str(api)
            if src_changed or api_changed:
                warnings.warn(
                    f"cached alias {alias!r} was fetched from api="
                    f"{stored_api!r} spec={stored_src!r} but is now "
                    f"requested from api={api!r} spec={src!r}; "
                    f"refetching with the new identity.",
                    UserWarning, stacklevel=2,
                )
                stale_aliases.add(alias)

    to_fetch = {
        alias: src for alias, src in spec.items()
        if refresh == "force" or alias not in existing_cols
        or alias in stale_aliases
    }

    if to_fetch:
        # 4. Call the fetcher. We deliberately pass start=None, end=None so
        # the fetcher returns the full available history of each series. The
        # cache stores the full series; load_dataset (step 8) filters to the
        # user's requested sample. This means a second project requesting a
        # wider sample doesn't trigger a re-fetch.
        fetched = fetcher.fetch(to_fetch)

        # A stale-identity alias (provider or source spec changed) MUST come
        # back from the new source; keeping the old source's values under the
        # new label would silently mislabel the series. Checked here so the
        # guard also covers partial fetch results, not just total failure.
        _failed_stale = {a for a in (stale_aliases & set(to_fetch))
                         if a not in (fetched or {})}
        if _failed_stale:
            raise RuntimeError(
                f"fetcher {api!r} failed to refetch "
                f"{sorted(_failed_stale)!r}, whose cached values were "
                f"fetched under a DIFFERENT or unrecorded identity (provider "
                f"or source spec). Refusing to fall back to the old source's values "
                f"and label them as the new one — fix the fetch or request "
                f"the original identity."
            )

        # 5. Merge into the cache
        if fetched:
            fetched_df = pd.DataFrame(fetched)
            fetched_df.index = pd.DatetimeIndex(fetched_df.index)
            fetched_df.index.name = "date"
            if fetched_df.index.has_duplicates:
                _dups = fetched_df.index[fetched_df.index.duplicated()][:5]
                raise ValueError(
                    f"fetcher {api!r} returned DUPLICATE dates for "
                    f"{sorted(to_fetch)!r} (e.g. {list(map(str, _dups))}). "
                    f"Refusing to cache a nonunique calendar — downstream "
                    f"joins would silently duplicate observations. Fix the "
                    f"source file / date parsing (check 'date_format')."
                )
            if existing is not None:
                # TRUE outer join on the union of date indices (column-wise
                # assignment would silently truncate newly fetched history to
                # the old index), preferring fetched values for refreshed
                # columns and preserving every previously cached column.
                union_idx = existing.index.union(fetched_df.index)
                merged = existing.reindex(union_idx)
                for col in fetched_df.columns:
                    merged[col] = fetched_df[col].reindex(union_idx)
            else:
                merged = fetched_df
            merged.sort_index().to_csv(csv_path)

            # 6. Update metadata
            _update_metadata(meta_path, fetcher, to_fetch, fetched_df)

            # 7. Update registry
            _ensure_registered(
                registry_path or _default_registry_path(project_root),
                dataset_id,
                fetcher,
            )
        else:
            # Fetcher returned nothing — every requested series failed.
            # If there's no cached fallback, give a clear error rather than
            # falling through to load_dataset (which would emit a confusing
            # "source not in registry" KeyError).
            if existing is None or existing.empty:
                failed = list(to_fetch)
                raise RuntimeError(
                    f"fetcher {api!r} failed to fetch any of {failed!r}; "
                    f"no cached data exists at data/{dataset_id}/. Check the "
                    f"warnings above for the API errors, then verify the URL "
                    f"constants in scripts/lib/data/fetchers/{api}.py and "
                    f"the series specs in your config."
                )
            warnings.warn(
                f"fetcher {api!r} returned no new data; falling back to cache.",
                UserWarning,
                stacklevel=2,
            )

    # 8. Return via the standard loader. dropna=False lets callers keep
    # series with legitimate internal gaps (e.g. event-based instrument
    # series with excluded months) and handle them explicitly.
    df = load_dataset(
        source=dataset_id,
        series={alias: alias for alias in spec},  # cache stores under alias
        start=start, end=end,
        dropna=dropna,
        registry_path=registry_path,
    )
    # Add fetcher metadata for downstream consumers
    df.attrs["fetcher_api"] = api
    df.attrs["dataset_id"] = dataset_id
    return df


def _default_registry_path(project_root: Path) -> Path:
    return project_root / "data" / "registry.yaml"


def _update_metadata(
    meta_path: Path,
    fetcher: BaseFetcher,
    to_fetch: Mapping[str, Any],
    fetched_df: pd.DataFrame,
) -> None:
    """Append/update per-series provenance in ``metadata.yaml``."""
    if meta_path.exists():
        meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
    else:
        meta = {"dataset_id": meta_path.parent.name, "variables": []}
    meta.setdefault("variables", [])

    # Index existing entries by series_id for upsert
    existing_by_id = {v.get("series_id"): i for i, v in enumerate(meta["variables"])}
    now = datetime.now(timezone.utc).isoformat()

    for alias, source_spec in to_fetch.items():
        if alias not in fetched_df.columns:
            continue  # fetcher didn't return this one
        col = fetched_df[alias]
        entry = {
            "series_id": alias,
            "source_spec": str(source_spec),
            "api": fetcher.name,
            "endpoint": fetcher.endpoint,
            "fetched_at": now,
            "sample_start": str(col.first_valid_index().date())
            if col.first_valid_index() is not None else None,
            "sample_end": str(col.last_valid_index().date())
            if col.last_valid_index() is not None else None,
            "n_obs": int(col.notna().sum()),
        }
        if alias in existing_by_id:
            meta["variables"][existing_by_id[alias]] = entry
        else:
            meta["variables"].append(entry)

    meta["last_updated"] = now
    meta_path.write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")


def _ensure_registered(
    registry_path: Path,
    dataset_id: str,
    fetcher: BaseFetcher,
) -> None:
    """Register ``dataset_id`` in the gitignored local overlay if needed.

    Cache-on-fetch entries go to ``data/registry.local.yaml`` (merged with
    the tracked registry by ``load_registry``), so the committed
    ``data/registry.yaml`` never churns — or loses its hand-written
    comments to ``yaml.safe_dump`` — as machine-local fetches accumulate.
    """
    registry_path = Path(registry_path)
    if registry_path.exists():
        main = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
        if any(e.get("id") == dataset_id for e in main.get("datasets", [])):
            return

    local_path = registry_path.parent / "registry.local.yaml"
    if local_path.exists():
        registry_data = yaml.safe_load(local_path.read_text(encoding="utf-8")) or {"datasets": []}
    else:
        registry_data = {"datasets": []}
    registry_data.setdefault("datasets", [])

    if any(e.get("id") == dataset_id for e in registry_data["datasets"]):
        return

    registry_data["datasets"].append({
        "id": dataset_id,
        "name": f"{dataset_id} (fetched via {fetcher.name})",
        "description": f"Cache-on-fetch dataset, sourced from {fetcher.description!r}",
        "path": f"data/{dataset_id}/",
        "frequency": "mixed",
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "verified_by": f"fetcher:{fetcher.name}",
    })
    local_path.write_text(yaml.safe_dump(registry_data, sort_keys=False), encoding="utf-8")
