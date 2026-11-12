"""External-archive fetcher — author GitHub repos and journal supplements.

Used for shock series and other replication inputs that live in *static*
external files (not maintained APIs). Typical sources:

* Jarociński-Karadi high-frequency monetary surprises (GitHub)
* Stock-Watson MP shocks (replication archive)
* Romer-Romer narrative MP shocks (academic supplement)
* Mertens-Ravn proxy tax shocks (AER supplement)

These archives are usually published once with a paper and don't change. The
:func:`lib.fetch_data` column-level cache handles refresh: parsed series
persist locally, and the HTTP layer is only touched when a series is missing
from the cache. Within a single ``fetch()`` call, the *raw bytes* of each
URL are downloaded only once even if multiple aliases come from the same
archive.

Series spec shape — string form (simple)::

    series = {
        "ff4": "https://example.org/archive/jk_shocks.csv:MP1",
    }

The string is split on the **last** ``":"`` so that ``"https://..."`` (which
contains a colon after the scheme) parses correctly.

Series spec shape — dict form (when the CSV needs special parsing)::

    series = {
        "ebp": {
            "url":         "https://example.org/data/ebp.csv",
            "column":      "ebp",
            "date_column": "date",       # default: first column of the CSV
            "date_format": "%Y-%m-%d",   # default: pandas auto-infer
            # Optional pandas read_csv kwargs:
            "skiprows":  2,
            "encoding":  "latin-1",
            "sep":       ",",
            "header":    0,
            "thousands": ",",
            "na_values": ["NA", "."],
        },
        # Or, for CSVs that split the date across columns (e.g., the
        # Jarociński-Karadi shocks file has separate ``year`` and ``month``):
        "mp_shock": {
            "url":          "https://example.com/jk_shocks.csv",
            "column":       "MP_median",
            "date_columns": ["year", "month"],   # joined with '-' and parsed
            # date_format defaults to "%Y-%m" for a 2-column list; override
            # via "date_format" for non-default layouts.
        },
    }
"""

from __future__ import annotations

import io
import warnings
from typing import Any, Mapping
from urllib import request
from urllib.error import HTTPError, URLError

import pandas as pd

from ..fetcher import BaseFetcher, register_fetcher

_READ_CSV_KWARG_KEYS = (
    "skiprows", "encoding", "sep", "header", "thousands", "na_values",
)


class ExternalArchiveFetcher(BaseFetcher):
    """Fetcher for static CSVs hosted at author / journal archives."""

    name = "external_archive"
    description = "Author / journal supplementary archives (static CSVs)"
    endpoint = "https://"

    def fetch(
        self,
        series: Mapping[str, Any],
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> dict[str, pd.Series]:
        bytes_cache: dict[str, bytes] = {}
        out: dict[str, pd.Series] = {}

        for alias, spec in series.items():
            try:
                cfg = self._parse_spec(spec)
            except ValueError as exc:
                raise ValueError(
                    f"external_archive series spec for {alias!r}: {exc} "
                    f"(got {spec!r})"
                ) from exc

            url = cfg["url"]
            if url not in bytes_cache:
                try:
                    bytes_cache[url] = self._http_get_bytes(url)
                except (HTTPError, URLError) as exc:
                    warnings.warn(
                        f"external_archive fetch failed for alias={alias!r} "
                        f"(url={url}): {exc}",
                        UserWarning,
                        stacklevel=2,
                    )
                    continue

            try:
                df = self._read_csv(bytes_cache[url], cfg)
            except (ValueError, KeyError) as exc:
                warnings.warn(
                    f"external_archive: failed to parse archive at {url} "
                    f"for alias={alias!r}: {exc}",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            column = cfg["column"]
            if column not in df.columns:
                warnings.warn(
                    f"external_archive: column {column!r} not in archive {url}; "
                    f"available: {list(df.columns)}. Skipping alias={alias!r}.",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            ser = df[column].rename(alias)
            if start is not None:
                ser = ser.loc[pd.Timestamp(start):]
            if end is not None:
                ser = ser.loc[:pd.Timestamp(end)]
            out[alias] = ser
        return out

    # -------- helpers --------

    @staticmethod
    def _parse_spec(spec: Any) -> dict[str, Any]:
        """Normalise spec to a dict with at least ``url`` and ``column``."""
        if isinstance(spec, Mapping):
            url = spec.get("url")
            column = spec.get("column")
            if not url or not column:
                raise ValueError("dict spec must have non-empty 'url' and 'column' keys")
            cfg: dict[str, Any] = {
                "url": url,
                "column": column,
                "date_column": spec.get("date_column"),
                "date_columns": spec.get("date_columns"),
                "date_format": spec.get("date_format"),
            }
            for k in _READ_CSV_KWARG_KEYS:
                if k in spec:
                    cfg[k] = spec[k]
            return cfg
        if isinstance(spec, str):
            # rpartition on ":" so URLs containing ':' (scheme, port) work —
            # the user must put the column name *after* any URL colons.
            url, sep, column = spec.rpartition(":")
            if not sep or not url or not column:
                raise ValueError("string spec must be '<url>:<column>'")
            return {"url": url, "column": column, "date_column": None, "date_format": None}
        raise ValueError("expected a string '<url>:<column>' or a dict with 'url' + 'column'")

    @staticmethod
    def _http_get_bytes(url: str, timeout: int = 60) -> bytes:
        req = request.Request(url, headers={
            "Accept": "text/csv,text/plain,*/*",
            "User-Agent": "Mozilla/5.0 (compatible; marco-bis/1.0)",
        })
        with request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return resp.read()

    @staticmethod
    def _read_csv(data: bytes, cfg: Mapping[str, Any]) -> pd.DataFrame:
        """Parse CSV bytes into a date-indexed DataFrame.

        Date handling:

        * If ``cfg["date_columns"]`` is a list (e.g., ``["year", "month"]``),
          those columns are joined with ``"-"`` and parsed via
          ``pd.to_datetime`` (with optional ``cfg["date_format"]``). Useful for
          academic archives that split the date across multiple integer
          columns. Month/day are zero-padded automatically.
        * Else takes ``cfg["date_column"]`` if given, defaulting to the first
          column of the CSV. Parses with optional ``cfg["date_format"]``.

        Remaining columns are coerced to numeric (``errors="coerce"``) so
        string values become NaN rather than blocking the load.
        """
        kwargs = {k: cfg[k] for k in _READ_CSV_KWARG_KEYS if k in cfg}
        df = pd.read_csv(io.BytesIO(data), **kwargs)

        date_cols = cfg.get("date_columns")
        if date_cols:
            missing = [c for c in date_cols if c not in df.columns]
            if missing:
                raise KeyError(
                    f"date_columns {missing!r} not in CSV; available: {list(df.columns)}"
                )
            # Zero-pad each component so 'YYYY-M-D' becomes 'YYYY-MM-DD' for
            # parsing. The first column (year) is left as-is; subsequent
            # components are padded to width 2.
            parts = df[date_cols[0]].astype(str)
            for c in date_cols[1:]:
                parts = parts + "-" + df[c].astype(str).str.zfill(2)
            fmt = cfg.get("date_format")
            if fmt is None and len(date_cols) == 2:
                fmt = "%Y-%m"  # common case: year + month
            df.index = pd.to_datetime(parts, format=fmt)
            df = df.drop(columns=list(date_cols))
        else:
            date_col = cfg.get("date_column")
            if date_col is None:
                date_col = df.columns[0]
            if date_col not in df.columns:
                raise KeyError(
                    f"date column {date_col!r} not in CSV; available: {list(df.columns)}"
                )
            _fmt = cfg.get("date_format")
            if _fmt is None and pd.api.types.is_numeric_dtype(df[date_col]):
                raise ValueError(
                    f"external_archive: date column {date_col!r} is numeric "
                    f"(e.g. a bare year like 1990) — pandas would parse it "
                    f"as NANOSECOND timestamps near 1970. Pass an explicit "
                    f"'date_format' (e.g. '%Y') or 'date_columns' in the "
                    f"series spec."
                )
            df.index = pd.to_datetime(df[date_col], format=_fmt)
            df = df.drop(columns=[date_col])

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df


register_fetcher(ExternalArchiveFetcher.name, ExternalArchiveFetcher)
