"""Build the G4 country panel staging CSV from FRED.

For each country in {US, UK, JP, DE} and each variable in {ipi, cpi,
ltr, str}, fetch the FRED series listed in config.yaml's fred_series_map
via the public ``fredgraph.csv?id=...`` endpoint. Resample to monthly
(most are already monthly), log-transform per the config, align on the
sample window, and emit a long-format CSV with columns:

    date, unit, ipi, cpi, ltr, str

Re-run this when refreshing data vintages. The replication driver
(replicate.py) reads only the staging CSV.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from urllib import request

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "staging" / "canova_ciccarelli_g4_panel.csv"


def fetch_fred(series_id: str) -> pd.Series:
    """Pull a single series from FRED's public graph CSV endpoint."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    req = request.Request(url, headers={"User-Agent": "marco-bis/1.0"})
    with request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    df = pd.read_csv(io.BytesIO(data))
    df.columns = [c.strip() for c in df.columns]
    date_col = df.columns[0]
    val_col = df.columns[1]
    df[date_col] = pd.to_datetime(df[date_col])
    s = df.set_index(date_col)[val_col].rename(series_id)
    return pd.to_numeric(s, errors="coerce")


def main() -> int:
    cfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    countries = cfg["data"]["countries"]
    var_names = [v["name"] for v in cfg["data"]["variables"]]
    fred_map = cfg["data"]["fred_series_map"]
    transforms = cfg["data"]["transforms"]
    sample_start = pd.Timestamp(cfg["data"]["sample_start"])
    sample_end = pd.Timestamp(cfg["data"]["sample_end"])

    print("=" * 70)
    print("  Canova-Ciccarelli G4 panel — fetching FRED series")
    print("=" * 70)

    rows: list[pd.DataFrame] = []
    for country in countries:
        print(f"\n[{country}]")
        per_country: dict[str, pd.Series] = {}
        for var in var_names:
            sid = fred_map[country][var]
            try:
                s = fetch_fred(sid)
                # Resample to monthly start (most FRED monthly series are already MS)
                s = s.resample("MS").mean()
                per_country[var] = s
                first, last = s.dropna().index.min(), s.dropna().index.max()
                print(f"  {var:<4}  {sid:<25}  {first.date()}..{last.date()}  n={s.dropna().shape[0]}")
            except Exception as exc:
                sys.stderr.write(f"  FAILED to fetch {sid}: {exc}\n")
                raise

        df_c = pd.DataFrame(per_country)
        df_c = df_c.loc[sample_start:sample_end]

        # Apply transforms (currently log for IPI/CPI, level for rates)
        for var, t in transforms.items():
            if t == "log":
                df_c[var] = np.log(df_c[var])
            elif t == "level":
                pass
            else:
                raise ValueError(f"unknown transform {t!r}")

        df_c.insert(0, "unit", country)
        df_c.index.name = "date"
        rows.append(df_c.reset_index())

    panel = pd.concat(rows, ignore_index=True)
    # Drop rows where any variable is NaN
    n_before = len(panel)
    panel = panel.dropna(subset=var_names)
    n_after = len(panel)
    if n_after < n_before:
        print(f"\n  Dropped {n_before - n_after} rows with NaN values.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUTPUT_PATH, index=False, float_format="%.10g")

    print(f"\nWrote {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  Shape: {panel.shape}")
    print(f"  Per-country observation counts:")
    print(panel.groupby("unit").size().to_string())
    print(f"  Date range: {panel['date'].min()} to {panel['date'].max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
