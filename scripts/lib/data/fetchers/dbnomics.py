"""DBnomics — unified access to ~80 macro data providers via one JSON API.

DBnomics (https://db.nomics.world) aggregates ECB, OECD, IMF, BIS, Eurostat,
World Bank, and many national statistics offices behind a single, free,
**no-key** JSON API. One fetcher therefore reaches most of the world's public
macro series — handy for euro-area work, cross-country panels, and any source
not covered by a dedicated fetcher.

API::

    https://api.db.nomics.world/v22/series?series_ids=<id>&observations=1

returns ``series.docs[i]`` with parallel ``period`` / ``period_start_day`` /
``value`` arrays. An invalid id comes back as HTTP 200 with empty ``docs`` and
an ``errors`` entry (handled below — warn and skip, like any other source).

Series-spec shape via ``lib.fetch_data(api="dbnomics", series=...)``::

    series = {
        "ea_hicp": "ECB/ICP/M.U2.N.000000.4.ANR",   # euro-area HICP, monthly
        "us_cpi":  "OECD/MEI/USA.CPALTT01.GP.M",
        "us_gdp":  "IMF/IFS/Q.US.NGDP_SA_XDC",       # quarterly
    }

i.e. ``{alias: "PROVIDER/DATASET/SERIES"}`` — a DBnomics series id (browse them
at https://db.nomics.world). The index uses DBnomics' ``period_start_day``
(period-start ISO dates), so quarterly / annual series align with the rest of
the library; missing observations (DBnomics ``"NA"`` / null) become ``NaN``.
"""

from __future__ import annotations

import json
import warnings
from typing import Any, Mapping
from urllib import parse, request
from urllib.error import HTTPError, URLError

import pandas as pd

from ..fetcher import BaseFetcher, register_fetcher

_BASE_URL = "https://api.db.nomics.world/v22/series"


class DbnomicsFetcher(BaseFetcher):
    """Fetcher for DBnomics — one API over ~80 providers (ECB, OECD, IMF, …)."""

    name = "dbnomics"
    description = "DBnomics aggregator (ECB / OECD / IMF / BIS / Eurostat / World Bank / …)"
    endpoint = _BASE_URL

    def fetch(
        self,
        series: Mapping[str, Any],
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> dict[str, pd.Series]:
        out: dict[str, pd.Series] = {}
        for alias, spec in series.items():
            try:
                series_id = self._parse_spec(spec)
            except ValueError as exc:
                raise ValueError(
                    f"DBnomics series spec for {alias!r} must be a "
                    f"'PROVIDER/DATASET/SERIES' id string (e.g. "
                    f"'ECB/ICP/M.U2.N.000000.4.ANR'); got {spec!r}: {exc}"
                ) from exc

            url = f"{_BASE_URL}?{parse.urlencode({'series_ids': series_id, 'observations': 1})}"
            try:
                payload = self._http_get_json(url)
            except (HTTPError, URLError) as exc:
                warnings.warn(
                    f"DBnomics fetch failed for alias={alias!r} "
                    f"(series_id={series_id}, url={url}): {exc}",
                    UserWarning, stacklevel=2,
                )
                continue

            docs = (payload.get("series") or {}).get("docs") or []
            if not docs:
                # An invalid id returns HTTP 200 with empty docs + an errors array.
                errs = payload.get("errors") or []
                msg = errs[0].get("message") if errs else "no series returned"
                warnings.warn(
                    f"DBnomics returned no data for alias={alias!r} "
                    f"(series_id={series_id}): {msg}",
                    UserWarning, stacklevel=2,
                )
                continue

            ser = self._extract_series(docs[0])
            if ser is None:
                warnings.warn(
                    f"DBnomics series for alias={alias!r} (series_id={series_id}) "
                    f"had no parseable observations.",
                    UserWarning, stacklevel=2,
                )
                continue
            ser.name = alias
            if start is not None:
                ser = ser.loc[pd.Timestamp(start):]
            if end is not None:
                ser = ser.loc[:pd.Timestamp(end)]
            out[alias] = ser
        return out

    # -------- helpers --------

    @staticmethod
    def _parse_spec(spec: Any) -> str:
        if isinstance(spec, str):
            s = spec.strip()
            parts = s.split("/")
            if len(parts) >= 3 and all(parts):
                return s
        raise ValueError("expected 'PROVIDER/DATASET/SERIES' (at least two '/')")

    @staticmethod
    def _http_get_json(url: str, timeout: int = 60) -> dict:
        req = request.Request(url, headers={
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (compatible; marco-bis/1.0)",
        })
        with request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8", errors="replace"))

    @staticmethod
    def _extract_series(doc: dict) -> pd.Series | None:
        """Build a Series from a DBnomics doc's parallel date/value arrays.

        Prefers ``period_start_day`` (ISO ``YYYY-MM-DD``, so quarterly / annual
        labels resolve to period-start dates); falls back to parsing ``period``.
        """
        values = doc.get("value")
        if not values:
            return None
        psd = doc.get("period_start_day")
        periods = doc.get("period") or []
        raw_dates = psd if (psd and len(psd) == len(values)) else periods

        dates: list[pd.Timestamp] = []
        vals: list[float] = []
        for d, v in zip(raw_dates, values):
            ts = _parse_period(d)
            if ts is None:
                continue
            try:
                fv = float(v) if v not in (None, "NA", "na", ".", "") else float("nan")
            except (ValueError, TypeError):
                fv = float("nan")
            dates.append(ts)
            vals.append(fv)
        if not dates:
            return None
        return pd.Series(vals, index=pd.DatetimeIndex(dates)).sort_index()


def _parse_period(p: Any) -> pd.Timestamp | None:
    """Parse a DBnomics date label. Handles ISO dates, ``YYYY``, ``YYYY-MM``,
    and quarter labels (``1950-Q1`` / ``1950Q1``)."""
    if not isinstance(p, str):
        return None
    p = p.strip()
    try:
        if "Q" in p.upper():
            return pd.Period(p.upper().replace("-", ""), freq="Q").to_timestamp()
        return pd.Timestamp(p)
    except (ValueError, TypeError):
        try:
            return pd.Timestamp(pd.to_datetime(p))
        except (ValueError, TypeError):
            return None


register_fetcher(DbnomicsFetcher.name, DbnomicsFetcher)
