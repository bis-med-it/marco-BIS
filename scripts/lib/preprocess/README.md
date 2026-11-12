# lib.preprocess

Complex preprocessing operations that don't belong in the data loader (which
is *structural*) or in `lib.transform` (which is *per-column univariate*).
Separating them out lets any downstream model reuse them without duplicating
logic.

## Operations

| Function | Purpose | Methods |
|---|---|---|
| `deflate(nominal, deflator, base_period=None)` | Nominal → real | Divide + optional base rebase |
| `nominalise(real, deflator, base_period=None)` | Real → nominal | Multiply + optional base rebase; inverse of `deflate`, used by `sims_1980_var`, `cee_1996_flow_of_funds`, `cee_1999_monetary_svar`, `glp_2015_prior_selection` |
| `aggregate(df, target_freq, method)` | Frequency aggregation (e.g., M → Q) | `sum` (flows), `mean` (rates), `end`/`start` (stocks), `min`/`max`/`median`/`std`; per-column method dicts |
| `impute(df, method, lags=..., em_iter=...)` | Fill missing values | `ffill`, `linear`, **Kalman smoother** via VAR(p) state-space (reuses [`lib.state_space`](../state_space.py)); optional **EM iteration** (Shumway-Stoffer 1982) for heavy missingness. Covers ragged-edge nowcasting and panel gaps |

**Not included:** frequency disaggregation (Chow-Lin, Denton), seasonal
adjustment, outlier detection and winsorising, standardisation,
year-on-year differences and moving averages. Use seasonally adjusted
source series where available, or call `statsmodels` (X-13, STL) directly;
factor models standardise internally (`lib.factors`, `lib.dfm`).

## Out of scope (already covered elsewhere)

- **Trend-cycle filters**: HP, Hamilton, BN, KMW-BN → live in [`lib.filters`](../filters.py)
- **Per-column univariate transforms** (`log`, `diff`, `log_diff`, …) → live in [`lib.transform`](../data/transforms.py)
- **Identification rotation, IRF computation** → live in [`lib.identification`](../identification.py)

## Design principles

1. **Stateless functions.** Each takes data in, returns data out. No hidden state.
2. **Preserve metadata.** If input is a DataFrame with `df.attrs` (provenance
   from `lib.load_dataset`), preserve it and extend with a `preprocess_applied`
   record so the research log can trace what was done.
3. **Use established packages.** Wrap well-tested implementations
   (`statsmodels`, `scipy`) instead of reimplementing them.

## How this fits into a `replicate.py`

```python
# 1. Load raw data with provenance metadata
raw = lib.load_dataset(source="fred_md", series={...}, start=..., end=...)

# 2. Preprocess (deflate, aggregate, impute); each op returns a Series,
#    so add it to the table as a new column
data = raw.copy()
data["m1"] = lib.preprocess.nominalise(raw["m1_real"], raw["deflator"])
# (more preprocess steps as needed)

# 3. Per-column transforms on the model's variables
data = lib.transform(data[["m1", "gdp"]], {"m1": "log", "gdp": "log_diff_pct"})

# 4. Fit + identify
fit = lib.fit_var(data, lags=4)
irf = lib.cholesky_irf(fit, ...)
```

Pipeline order matters: preprocess applies operations whose result feeds the
transform layer (e.g., nominalise produces nominal M1, which then gets logged).
