"""Local-CSV "fetcher" — stage manually-downloaded data into the cache.

Useful when an external source isn't yet supported by a dedicated fetcher (or
the user already has the data on disk from a previous download). Treat a local
CSV as if it were an API response: extract specified columns, join them into
``data/<dataset_id>/dataset.csv``, and register the dataset normally.

Series spec shape passed via ``lib.fetch_data(api="csv_file", series=...)``::

    series = {
        "indpro": "data/staging/ons_k222.csv:K222",
        "cpi":    "data/staging/ons_d7bt.csv:value",
    }

Each value is ``"<csv_path>:<column_name>"``. The CSV file must have its
dates in the first column (used as the index) and the requested
``column_name``. Relative paths are resolved against the current working
directory; use an absolute path or run from the repository root.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ..fetcher import BaseFetcher, register_fetcher


class CsvFileFetcher(BaseFetcher):
    """Loads series from local CSV files."""

    name = "csv_file"
    description = "Local CSV staging — manually-downloaded data"
    endpoint = "file://"

    def fetch(
        self,
        series: Mapping[str, Any],
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> dict[str, pd.Series]:
        out: dict[str, pd.Series] = {}
        for alias, spec in series.items():
            try:
                csv_path, column = self._parse_spec(spec)
            except ValueError as exc:
                raise ValueError(
                    f"csv_file series spec for {alias!r} must be "
                    f"'<path>:<column>'; got {spec!r}: {exc}"
                ) from exc

            path = Path(csv_path)
            if not path.is_absolute():
                # Resolve relative to current working directory (typically project root)
                path = Path.cwd() / csv_path
            if not path.exists():
                raise FileNotFoundError(
                    f"csv_file fetcher: {path} not found for alias={alias!r}"
                )

            df = pd.read_csv(path, index_col=0, parse_dates=True)
            if column not in df.columns:
                raise KeyError(
                    f"csv_file fetcher: column {column!r} not in {path}; "
                    f"available: {list(df.columns)}"
                )
            ser = df[column].rename(alias)
            if start is not None:
                ser = ser.loc[pd.Timestamp(start):]
            if end is not None:
                ser = ser.loc[:pd.Timestamp(end)]
            out[alias] = ser
        return out

    @staticmethod
    def _parse_spec(spec: Any) -> tuple[str, str]:
        if not isinstance(spec, str) or ":" not in spec:
            raise ValueError("expected '<path>:<column>'")
        path, _, column = spec.rpartition(":")
        if not path or not column:
            raise ValueError("path and column must both be non-empty")
        return path, column


register_fetcher(CsvFileFetcher.name, CsvFileFetcher)
