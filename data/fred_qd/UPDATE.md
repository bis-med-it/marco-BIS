# Updating FRED-QD

The `dataset.csv` is not distributed with the framework; on first use,
follow the download steps below once (run them from `data/fred_qd/`).
Record the vintage you downloaded (the file's date or the vintage named
on the page) alongside your copy: results are vintage-dependent and the
loader only records what you tell it.

## Known series caveats

**Reserves (`NONBORRES`, `TOTRESNS`):** unsuitable for pre-1990s
reserves-market analysis without adjustment. The modern FRED series omit
extended credit (the paper-era convention was NBR + extended credit) and
carry unadjusted breaks, namely the 1984 Continental Illinois swings (±13%,
multi-sigma) and the Dec-1990/Apr-1992 reserve-requirement changes.
Replications that pin reserves-market moments (CEE 1996/1999 class) will
diverge from published values for vintage reasons, not methodology;
document the divergence rather than re-tuning, or stage a break-adjusted
historical series.

## Schedule

Update quarterly, shortly after the GDP advance estimate release (typically
late January, April, July, October).

## Manual update

1. Go to https://www.stlouisfed.org/research/economists/mccracken/fred-databases
2. Under "FRED-QD: Quarterly Data", download `current.csv` and save it
   as `data/staging/fred_qd_current.csv` (gitignored).
3. Run the cleaning script below to strip the TWO metadata rows (factor
   flags + transform codes) and write `dataset.csv`,
   `transform_codes.csv` and `factor_flags.csv`. Do NOT save the raw
   `current.csv` as `dataset.csv`, because the metadata rows break
   `lib.load_dataset`.
4. Update `../registry.yaml` with new vintage and date range.

## Cleaning script

```python
import pandas as pd

# The site blocks scripted downloads; download current.csv in a browser.
raw = pd.read_csv("../staging/fred_qd_current.csv")  # the raw download
# Row 0 = factor flags, Row 1 = transform codes, data starts at row 2
factors = raw.iloc[0]
transform_codes = raw.iloc[1]
data = raw.iloc[2:].copy()
# set the index from the parsed dates directly -- assigning datetimes back
# into the string column breaks under pandas >= 3
data = data.set_index(pd.to_datetime(data.iloc[:, 0])).drop(columns=data.columns[0])
data.index.name = 'date'
data = data.apply(pd.to_numeric, errors='coerce')
data.to_csv("dataset.csv")
transform_codes.to_csv("transform_codes.csv")
factors.to_csv("factor_flags.csv")
```

## Notes

- FRED-QD contains ~245 quarterly variables (count varies by vintage).
- Raw CSV has TWO metadata rows: factor flags (row 0) and transform codes (row 1).
- For Sims (1980) and CEE (1999) replications, draw variables directly
  from FRED-QD to ensure quarterly alignment without aggregation.
- Key variable mappings: GDPCTPI (GDP deflator), PCECTPI (PCE price index),
  M1REAL (real M1), GDPC1 (real GDP).
- Transform codes follow the same convention as FRED-MD.
- Reference: McCracken and Ng (2020).
