"""Unified loader for pre-collected datasets.

Delivers structurally-clean raw data (DatetimeIndex, named columns, sample-
filtered) plus provenance metadata in ``df.attrs``. Per-model transforms
(log, diff, log_diff, ...) are applied separately via :func:`lib.transform`
— the data layer's job is structure and provenance, not model-specific
preparation.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from .registry import get_dataset_path, load_registry
from .transforms import FRED_MD_CODE_TO_NAME


def _read_fred_transform_codes(csv_dir: Path) -> dict[str, str]:
    """Read FRED-MD ``transform_codes.csv`` if present.

    The FRED-MD format has two metadata rows at the top — ``,0`` and
    ``sasdate,Transform:`` — followed by ``series_code,float_code`` rows.
    Returns ``{source_code: recommended_transform_name}``.
    """
    path = csv_dir / "transform_codes.csv"
    if not path.exists():
        return {}
    try:
        raw = pd.read_csv(
            path, header=None, skiprows=2, names=["series", "code"],
        )
    except Exception:  # malformed file — fall back to no recommendations
        return {}
    mapping: dict[str, str] = {}
    for _, row in raw.iterrows():
        try:
            code = int(row["code"])
        except (ValueError, TypeError):
            continue
        name = FRED_MD_CODE_TO_NAME.get(code)
        if name is not None:
            mapping[str(row["series"])] = name
    return mapping


def load_dataset(
    source: str,
    series: Mapping[str, str] | Sequence[str],
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    vintage: str | None = None,
    dropna: bool = True,
    ignore_missing: bool = False,
    *,
    registry_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load raw data from a pre-collected dataset.

    Returns structurally-clean raw data — DataFrame with DatetimeIndex,
    named columns, sample-filtered. Apply transforms via :func:`lib.transform`
    as a separate step.

    Parameters
    ----------
    source : str
        Dataset id in ``data/registry.yaml`` (e.g., ``"fred_md"``, ``"fred_qd"``).
    series : dict or list
        If dict: ``{alias: source_code}``. Output columns named by alias.
        If list: ``[source_code, ...]``. Output columns named by source code.
    start, end : str or pd.Timestamp, optional
        Sample period bounds. If None, no truncation.
    vintage : str, optional
        Reserved for future vintage support. Recorded in metadata.
    dropna : bool, default True
        Drop LEADING/TRAILING rows containing NaN in the loaded series.
        Rows missing INSIDE the requested sample are never silently
        dropped — that would corrupt calendar lags and differences —
        and raise ``ValueError`` instead (impute, choose a contiguous
        sample, or pass ``dropna=False`` and handle the gaps
        explicitly). For a dataset registered as ``monthly`` or
        ``quarterly``, a period with no row at all is treated the same
        way: ``ValueError`` by default, a warning with ``dropna=False``.
    ignore_missing : bool, default False
        If True, requested series that aren't in the dataset are silently
        skipped (with a warning), and the load proceeds with the available
        ones. The skipped codes are recorded in ``df.attrs["missing_series"]``.
        If False (default), missing series raise ``KeyError``.
        Useful for large information panels (BBE 2005 FAVAR) where the exact
        column list depends on the dataset vintage.
    registry_path : path, optional
        Override the registry path. Used in tests; production code finds it.

    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex, columns named by alias (or source code).
        ``df.attrs`` carries provenance metadata: ``source``, ``name``,
        ``description``, ``frequency``, ``vintage``, ``url``, ``last_updated``,
        ``verified_by``, ``loaded_at``, ``sample_start_actual``,
        ``sample_end_actual``, ``n_obs``, a ``series`` dict mapping each
        alias to ``{source_code, recommended_transform}`` (the latter from the
        dataset's ``transform_codes.csv`` if present), and (if any series
        were skipped via ``ignore_missing``) a ``missing_series`` list.

    Raises
    ------
    KeyError
        ``source`` is not in the registry, or (with ``ignore_missing=False``)
        a requested series is not in the dataset, or all requested series
        are missing.
    FileNotFoundError
        Source is in the registry but its CSV is missing.
    ValueError
        The CSV's first column does not parse as dates (e.g. a raw
        FRED ``current.csv`` with its metadata rows still in place).
        With ``dropna=True``: observations are missing INSIDE the
        requested sample (internal calendar gaps, including absent
        periods of a monthly or quarterly dataset) — see ``dropna``
        above.

    Examples
    --------
    >>> raw = lib.load_dataset(
    ...     source="fred_md",
    ...     series={"indpro": "INDPRO", "cpi": "CPIAUCSL"},
    ...     start="1965-01", end="2003-12",
    ... )
    >>> data = lib.transform(raw, {"indpro": "log", "cpi": "log"})

    Tolerate vintage-dependent missing series in a large information panel::

        >>> info = lib.load_dataset(
        ...     source="fred_md",
        ...     series=["INDPRO", "CPIAUCSL", "DISCONTINUED_SERIES"],
        ...     ignore_missing=True,
        ... )
        >>> info.attrs["missing_series"]
        ['DISCONTINUED_SERIES']
    """
    # 1. Resolve registry entry + CSV path
    registry = load_registry(registry_path)
    csv_path = get_dataset_path(source, registry=registry)
    entry = next((e for e in registry.get("datasets", []) if e["id"] == source), {})

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset CSV not found at {csv_path}.\n"
            f"Source {source!r} is in the registry but the file is missing. "
            f"See data/{source}/UPDATE.md for download instructions."
        )

    # 2. Load CSV (date index, named columns)
    raw = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if len(raw) and not isinstance(raw.index, pd.DatetimeIndex):
        raise ValueError(
            f"Dataset {source!r}: the first column of {csv_path} does not "
            f"parse as dates (first entry: {raw.index[0]!r}). A raw FRED "
            f"current.csv still has its metadata rows at the top; run the "
            f"cleaning recipe in data/{source}/UPDATE.md to write dataset.csv."
        )
    raw = raw.sort_index()
    if raw.index.has_duplicates:
        _dups = raw.index[raw.index.duplicated()][:5]
        raise ValueError(
            f"Dataset {source!r} has DUPLICATE dates in its calendar "
            f"(e.g. {list(map(str, _dups))}) — joins against it would "
            f"silently duplicate observations. Repair or re-download the "
            f"cache at {csv_path}."
        )

    # 3. Normalize series spec to {alias: source_code}
    if isinstance(series, Mapping):
        spec = dict(series)
    else:
        spec = {s: s for s in series}

    # 4. Validate series exist in dataset (or filter if ignore_missing)
    missing = [code for code in spec.values() if code not in raw.columns]
    if missing:
        if ignore_missing:
            warnings.warn(
                f"Series {missing} not found in {source!r}; skipping "
                f"(ignore_missing=True). Available: {len(raw.columns)} columns.",
                UserWarning,
                stacklevel=2,
            )
            spec = {alias: code for alias, code in spec.items() if code in raw.columns}
            if not spec:
                raise KeyError(
                    f"None of the requested series were found in {source!r}. "
                    f"All {len(missing)} requested codes are missing."
                )
        else:
            sample_cols = list(raw.columns[:5])
            raise KeyError(
                f"Series {missing} not found in {source!r} "
                f"({len(raw.columns)} columns total, sample: {sample_cols}...)"
            )

    # 5. Filter by sample period (record requested bounds)
    requested_start = pd.Timestamp(start) if start is not None else None
    requested_end = pd.Timestamp(end) if end is not None else None
    if requested_start is not None:
        raw = raw.loc[requested_start:]
    if requested_end is not None:
        raw = raw.loc[:requested_end]

    # 5b. A monthly or quarterly dataset must not skip a period. An absent
    # row never shows up as NaN, so the dropna guard below cannot see it,
    # and row-based differences and lags would treat the periods on either
    # side as adjacent. Periods (not timestamps) keep FRED-QD's
    # quarter-ending-month labels valid.
    freq = {"monthly": "M", "quarterly": "Q"}.get(
        str(entry.get("frequency", "")).lower())
    if freq and len(raw) > 1:
        periods = raw.index.to_period(freq)
        absent = pd.period_range(periods.min(), periods.max(),
                                 freq=freq).difference(periods)
        if len(absent):
            msg = (
                f"load_dataset: {len(absent)} period(s) absent from the "
                f"{entry.get('frequency')} calendar of {source!r} inside "
                f"the requested sample (first: {absent[0]}). Row-based "
                f"differences and lags would treat the periods on either "
                f"side as adjacent. Repair or re-download the dataset, or "
                f"choose a sample without the gap."
            )
            if dropna:
                raise ValueError(msg)
            warnings.warn(msg, UserWarning, stacklevel=2)

    # 6. Select and alias columns
    df = pd.DataFrame({alias: raw[code] for alias, code in spec.items()},
                      index=raw.index)
    if dropna:
        pre_drop_index = df.index
        df = df.dropna()
        # Dropping rows must only trim the EDGES. Deleting an internal date
        # silently corrupts calendar-based transforms downstream (a two-month
        # movement becomes a one-period 'growth' observation) and makes
        # non-adjacent dates look adjacent to lag constructions.
        if len(df) and len(df) != len(pre_drop_index):
            lo, hi = df.index[0], df.index[-1]
            expected_inner = pre_drop_index[(pre_drop_index >= lo)
                                            & (pre_drop_index <= hi)]
            if len(expected_inner) != len(df):
                missing = expected_inner.difference(df.index)
                raise ValueError(
                    f"load_dataset: {len(missing)} internal observation(s) "
                    f"missing inside the requested sample (first: "
                    f"{missing[0]}). Row-dropping would corrupt calendar "
                    f"lags and differences. Either impute "
                    f"(lib.preprocess.impute), choose a contiguous sample, "
                    f"or pass dropna=False and handle the gaps explicitly."
                )

    # 7. Read per-series recommended transforms if dataset provides them
    recommended = _read_fred_transform_codes(csv_path.parent)

    # 8. Populate provenance metadata in df.attrs
    series_meta = {}
    for alias, source_code in spec.items():
        item: dict[str, str] = {"source_code": source_code}
        if source_code in recommended:
            item["recommended_transform"] = recommended[source_code]
        series_meta[alias] = item

    df.attrs = {
        "source": source,
        "name": entry.get("name"),
        "description": entry.get("description"),
        "frequency": entry.get("frequency"),
        "vintage": vintage or entry.get("vintage"),
        "url": entry.get("url"),
        "last_updated": entry.get("last_updated"),
        "verified_by": entry.get("verified_by"),
        "loaded_at": datetime.now(timezone.utc).isoformat(),
        "sample_start_requested": str(requested_start.date()) if requested_start else None,
        "sample_end_requested": str(requested_end.date()) if requested_end else None,
        "sample_start_actual": str(df.index.min().date()) if len(df) else None,
        "sample_end_actual": str(df.index.max().date()) if len(df) else None,
        "n_obs": int(len(df)),
        "series": series_meta,
        "missing_series": missing if ignore_missing else [],
    }
    return df
