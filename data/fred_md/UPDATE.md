# Updating FRED-MD

The `dataset.csv` is not distributed with the framework; on first use,
follow the download steps below once (run them from `data/fred_md/`).
Record the vintage you downloaded (the file's date or the vintage named
on the page) alongside your copy: results are vintage-dependent and the
loader only records what you tell it.

## Schedule

Update monthly, shortly after the St. Louis Fed releases the new vintage
(typically within the first week of the month).

## Manual update

1. Go to https://www.stlouisfed.org/research/economists/mccracken/fred-databases
2. Under "FRED-MD: Monthly Data", download `current.csv` and save it
   as `data/staging/fred_md_current.csv` (gitignored).
3. Run the cleaning script below to strip the metadata row and write
   `dataset.csv` + `transform_codes.csv`. Do NOT save the raw
   `current.csv` as `dataset.csv`; its transform-code row breaks
   `lib.load_dataset` (non-date first row -> TypeError on sample
   slicing).
4. Update `../registry.yaml` with new vintage and date range.

## Cleaning script

```python
import pandas as pd

# The site blocks scripted downloads; download current.csv in a browser.
raw = pd.read_csv("../staging/fred_md_current.csv")  # the raw download
transform_codes = raw.iloc[0]
data = raw.iloc[1:].copy()
# set the index from the parsed dates directly -- assigning datetimes back
# into the string column breaks under pandas >= 3
data = data.set_index(pd.to_datetime(data.iloc[:, 0])).drop(columns=data.columns[0])
data.index.name = 'date'
data = data.apply(pd.to_numeric, errors='coerce')
data.to_csv("dataset.csv")
transform_codes.to_csv("transform_codes.csv")
```

## Notes

- FRED-MD contains ~126 monthly variables (count varies by vintage).
- First row of the raw CSV contains transformation codes.
- Transform codes: 1=level, 2=Δ, 3=Δ², 4=log, 5=Δlog, 6=Δ²log, 7=Δ(p/p(-1)-1).
- Reference: McCracken and Ng (2016).
