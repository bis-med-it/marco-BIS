---
name: data-retrieval
description: >
  Two-mode data management: (1) pre-collected datasets (FRED-MD/QD,
  downloaded once by the user) with rich metadata, and
  (2) runtime collection from APIs for series not in the pre-collected sets.
  Use when the orchestrator needs data, when the user asks for data, or when
  the user refreshes a pre-collected dataset. Also manages API connections, data quality
  checks, and transformation pipelines. Always checks the dataset registry
  before making API calls, because pre-collected data is faster and works
  offline. Stored panels hold raw observations; transforms are applied
  after loading.
---

# Data retrieval

## Two modes

### Mode 1: Pre-collected datasets (preferred)

Verified datasets stored locally with rich metadata. The user refreshes them
with each dataset's `UPDATE.md` recipe. The agent checks these first
before making any API call.

**Why preferred:** no API calls or rate limits, works offline, and the same
stored raw observations give the same results across runs. The source's
transformation codes are stored, not applied; the model applies its own
transforms after loading. Provenance is in the registry and `df.attrs`.

### Mode 2: Runtime collection (fallback)

Fetch from APIs when the needed series is not in any pre-collected dataset.
Also used for one-off analyses, exploring new variables, or economies not
yet covered by pre-collected datasets.

## Priority order

```
1. Pre-collected dataset    (if the variable exists in a verified dataset, use it)
       ↓ if not available
2. Knowledge base           (local data sources specific to the user's economy)
       ↓ if not specified
3. Runtime API call         (FRED, ECB, IMF, BIS, OECD)
       ↓ if no suitable API, or the request fails
4. User-provided file       (CSV, Excel, or other format)
```

### Tier 4: user-staged datasets (`data/staging/`)

Series that exist only behind author archives or logins (Romer-Romer
shocks, Mertens-Ravn AER data, panel assemblies) cannot be fetched and
are **user-staged**: the user places a file at `data/staging/<name>.csv`
and the script loads it via the `csv_file` fetcher. Conventions:

- **Location and naming:** `data/staging/<descriptive_name>.csv`, date
  column first; the consuming script's README documents the exact layout
  and the sourcing route (archive URL, DOI).
- **Registry:** staged files are not in `data/registry.yaml`
  (they are user-supplied inputs, not framework-distributed datasets);
  the consuming script is the source of truth for what is expected.
- **When missing:** the script (or you) must stop cleanly with the
  sourcing instructions. Never substitute, interpolate, or fabricate.
  This is user-staging, **not** cache-on-fetch: the two flows are distinct
  (cache-on-fetch lands automatically from a connector; staging requires
  a human with access).

### Deferred fetches (user declines or postpones a retrieval)

A declined runtime fetch fires nothing on disk, so the acquisition
disposition step does not run. Record the *plan* instead so any later
session can resume: write `deferred_fetch_plan.yaml` in the analysis
folder with: `status: deferred_by_user`, the proposed route(s)
(`lib.fetch_data` call with args), the **provenance of any series ids**
("parametric; verify on the source page before fetching"), the
pre-declared disposition pair (cache target + fetcher tier) to be
re-confirmed at fetch time, an inventory of what is already local, and an
`on_resume` checklist. The research log quotes or references the plan, so
a discarded scaffold must not take the plan with it.

---

## Dataset registry

### Structure

Pre-collected panels and the dataset registry (`data/registry.yaml`) live at the
repo root under `data/`; `lib.fetch_data` registers fetched caches in the
gitignored `data/registry.local.yaml`. Only `connections/` is skill-local config.

```
skills/data-retrieval/
├── SKILL.md               (this file)
└── connections/           API connection documentation
    └── connections.yaml   key variable names, endpoints, rate limits

data/                      (repo root) pre-collected and fetched datasets
├── registry.yaml          the dataset registry
├── fred_md/               FRED-MD monthly US macro panel
│   ├── dataset.csv        the data (not distributed; written by the UPDATE.md recipe)
│   ├── transform_codes.csv  per-series FRED transform codes
│   └── UPDATE.md          one-time download + refresh instructions
├── fred_qd/               FRED-QD quarterly US macro panel
│   ├── dataset.csv
│   ├── transform_codes.csv (+ factor_flags.csv)
│   └── UPDATE.md
├── staging/               raw downloads and user-staged series (gitignored)
└── <id>/                  user-added or cache-on-fetch datasets
    ├── dataset.csv
    └── metadata.yaml      (written by lib.fetch_data for fetched sets)
```

### Registry format

See [`data/registry.yaml`](../../data/registry.yaml) for the format. Each entry
records `id`, `name`, `description`, `source`, `url`, `frequency`, sample
bounds (nominal and `_actual`), `last_updated`, `vintage`,
`transforms_applied: false`, `path`, `variables_actual` (the actual series
count; `variables` is the nominal one), `missing_pct` and `covers`.

### Dataset metadata format

The two shipped panels carry their per-series metadata in
`transform_codes.csv` (+ `factor_flags.csv` for FRED-QD), with dataset-level
provenance in `data/registry.yaml` and `df.attrs` at load time.
**Cache-on-fetch datasets** (written by `lib.fetch_data`) additionally get a
`metadata.yaml` with one block per fetched series:

```yaml
# metadata.yaml for a cache-on-fetch dataset (values illustrative)
dataset_id: "uk_macro"
variables:
  - series_id: "indpro"              # the alias passed to fetch_data
    source_spec: "economy/economicoutputandproductivity/output/timeseries/k222/diop"
    api: "ons"
    endpoint: "https://www.ons.gov.uk"
    fetched_at: "2026-05-19T14:32:11+00:00"
    sample_start: "1992-10-01"
    sample_end: "2026-04-01"
    n_obs: 403
  # ... (one block per fetched series)
last_updated: "2026-05-19T14:32:11+00:00"
```

Units, seasonal adjustment and transform notes are not written by the
fetcher; add them by hand when promoting a dataset.

### How the agent uses pre-collected datasets

When the specification skill requests data:

1. **Check the registry** for a dataset covering the needed variables and sample.
2. **If the dataset file is missing on disk** (registry references it but
   `data/<id>/dataset.csv` doesn't exist, the typical fresh-install scenario for
   FRED-MD / FRED-QD): see "When pre-collected data is missing" below. **Do
   NOT silently fall through to runtime collection or proceed with empty
   data.**
3. **If found:** load directly from `data/<id>/dataset.csv`. No API call.
   Report to the user: "Using pre-collected FRED-MD dataset (vintage 2026-04,
   verified). Variables: INDPRO (industrial production, dlog), CPIAUCSL (CPI,
   dlog), FEDFUNDS (Fed funds rate, level)."
4. **If partially found:** load available variables from the dataset, fetch
   missing variables via runtime API call, and flag the mixed sourcing in the
   research log.
5. **If not in any registered dataset:** fall back to runtime collection
   (Mode 2 above).

### When pre-collected data is missing

For FRED-MD and FRED-QD specifically, the dataset CSVs are **not bundled**
with the framework, so a fresh copy will not have them. Other economies' datasets (a national central bank's,
the ECB's, etc.) follow the same pattern when the user populates them.

If the agent detects a missing FRED-MD or FRED-QD dataset file (the
registry references it but `data/fred_md/dataset.csv` or
`data/fred_qd/dataset.csv` does not exist on disk):

**First, check for recoverable local copies.** A `.bak`, renamed, or
vintage copy in the same folder may make the re-download unnecessary,
but **never silently restore a backup**: integrity-check it against the
registry (row/column counts, date range, the specific series the analysis
needs), report the findings, and propose the restore at a confirmation
gate (restoring writes into `data/`, which is stop-and-ask territory).

If nothing recoverable exists, **stop and prompt the user with the
download instructions below**. Keep the download steps as written (do not
paraphrase URLs or file paths); narrow the prompt to the file(s) the
analysis actually needs, per the note after the block. Do not proceed
until the user confirms placement.

> **FRED data not found.** Your `data/` folder is missing one or both of
> the McCracken (FRED-MD / FRED-QD) panel CSVs. These are not
> distributed with the framework and need a one-time manual download. Both files come from the same
> page; the whole setup takes ~2 minutes.
>
> **Step 1. Open the McCracken page in a browser:**
>
> https://www.stlouisfed.org/research/economists/mccracken/fred-databases
>
> **Step 2. Download FRED-MD (monthly, ~126 series).**
>
> On the page, locate the section **"FRED-MD: Monthly Data"**. Click the
> link to `current.csv` (the most recent vintage). Save the file as:
>
> ```
> data/staging/fred_md_current.csv
> ```
>
> (relative to the framework's repo root)
>
> **Step 3. Download FRED-QD (quarterly, ~245 series).**
>
> On the same page, locate the section **"FRED-QD: Quarterly Data"**.
> Click its `current.csv` link. Save as:
>
> ```
> data/staging/fred_qd_current.csv
> ```
>
> **Step 4. (Required) Run the cleaning recipe.**
>
> The raw `current.csv` files contain metadata rows at the top:
> - FRED-MD: row 1 = transform codes, row 2 onward = data.
> - FRED-QD: row 1 = factor flags, row 2 = transform codes, row 3 onward = data.
>
> The raw download cannot be used directly as `dataset.csv`. Run the
> cleaning recipe in `data/fred_md/UPDATE.md` or `data/fred_qd/UPDATE.md`
> (each is ~10 lines of pandas, run from that folder). It removes the
> metadata rows and writes `dataset.csv` plus the matching metadata
> files (`transform_codes.csv`, and `factor_flags.csv` for FRED-QD).
> If a raw file is loaded by mistake, the loader stops with an error
> pointing to the recipe.
>
> **Why manual?** The St. Louis Fed file server does not serve scripted
> downloads, so the files must be downloaded in a browser.
>
> **Tell me when both files are in place** and I'll continue with your
> analysis.

After the user confirms placement, the agent can re-check the file paths
and proceed with the original analysis request. If only one of the two
files is needed for the requested analysis (e.g., a quarterly-only model
doesn't need FRED-MD), the agent should narrow the prompt to just the
required file rather than asking for both unnecessarily, but always state
explicitly which files are being requested and why.

### Updating pre-collected datasets

Each dataset folder has an `UPDATE.md` with its refresh recipe (see
[`data/fred_md/UPDATE.md`](../../data/fred_md/UPDATE.md); the download step
is manual, see "FRED data not found" above). After a refresh, update
`last_updated` and `vintage` in `data/registry.yaml`, then check:

- [ ] No unexpected missing values in core variables
- [ ] Sample end matches expected date
- [ ] Variable count matches the registry's `variables_actual` (it varies by vintage)
- [ ] Transform codes unchanged from previous vintage
- [ ] Spot-check 3-5 values against the FRED website

### Adding a new economy's dataset

For non-US economies, the user creates a dataset following the same structure:

1. Create `data/[economy_name]/` directory.
2. Collect the data from the local statistical agency or central bank.
3. Write `metadata.yaml` documenting every variable (series ID, source,
   transformation, units, seasonal adjustment).
4. Write `UPDATE.md` with refresh instructions.
5. Add an entry to `registry.yaml`.
6. Verify and confirm.

The agent can help draft the metadata, but the user must verify it.

---

## Runtime collection

### Connection configuration

Per-source API routes (endpoint, API-key variable name, rate limit, fallback
package, documentation link) are in
[`connections/connections.yaml`](connections/connections.yaml). A custom source
is added there as a new block (the file ends with a commented template).

**Note on `connections.yaml` vs the implemented fetchers.** The connection
file documents per-source API routes (including fallback package recipes like
`sdmx`/`fredapi`) for sources the agent may reach ad hoc; it is not a registry
of `lib.fetch_data` fetchers. The implemented fetchers are the six listed
below; in particular, `dbnomics` covers ECB/OECD/IMF/BIS/Eurostat/World Bank
without per-source connections entries.

### Runtime collection workflow

When a variable is not in any pre-collected dataset:

1. **Identify the best source** from connections.yaml (or knowledge base for
   local economies). For US series use `fred`; for euro-area, OECD, IMF, BIS,
   Eurostat, World Bank, or cross-country panels, `dbnomics` reaches them all
   through one no-key API (`api="dbnomics"`, series id `PROVIDER/DATASET/SERIES`).
2. **Check API key availability.** If the source needs a key and it isn't set,
   stop and walk the user through setup (see **"Setting the FRED API key"**
   below). (For FRED, `lib.fetch_data(api="fred", …)` raises a clear, actionable
   error naming `FRED_API_KEY` and the exact steps when the key is missing.)
3. **Agree where the data goes before fetching.** `lib.fetch_data` writes
   the cache (`data/<dataset_id>/`), its metadata and a
   `data/registry.local.yaml` entry as part of the fetch. Unless the storage
   choice is already agreed for this run, confirm it first (Decision A under
   "Data acquisition disposition" below). For scratch-only data, call the
   source's fetcher directly (`from lib.data import get_fetcher`;
   `get_fetcher(api).fetch(series, start, end)`), which returns the series
   without writing to `data/`.
4. **Fetch the series** using the configured package.
5. **Apply transformations** as specified by the user or the model dictionary
   entry.
6. **Present a summary** for user confirmation: variable name, source, sample,
   units, transformation, any missing observations.
7. **Offer to promote** a cached series to a curated pre-collected dataset
   for future use (Decision A, option 3). This is a later upgrade of the
   cache, not the first storage decision.

### Setting the FRED API key

Individual FRED series (`lib.fetch_data(api="fred", …)`) need a free key. The
bulk FRED-MD / FRED-QD panels do **not** (those are staged by manual download,
above).

1. **Get a free key** (~30 seconds): https://fred.stlouisfed.org/docs/api/api_key.html
2. **Put it in the `FRED_API_KEY` environment variable**, whichever route suits
   the user:
   - **Current shell only:** `export FRED_API_KEY=your_key_here` (macOS/Linux) or
     `$env:FRED_API_KEY="your_key_here"` (Windows PowerShell).
   - **Future shells (Windows):** `setx FRED_API_KEY your_key_here`, then open a new shell.
   - **Persistent:** add `FRED_API_KEY=your_key_here` to a project `.env` file
     (if the runner loads one) or to the shell profile (`~/.bashrc` / `~/.zshrc`).
3. **Verify:** `from lib.data import get_fetcher; get_fetcher("fred").fetch({"rgdp": "GDPC1"})`
   (this returns the series without writing to `data/`). If the
   key is missing the fetcher raises a message naming the variable and these
   exact steps; if it's invalid, FRED returns HTTP 400 (surfaced as a warning).

The key lives only in the environment, never in `config.yaml` or any committed
file. `connections.yaml` records the *name* of the variable
(`api_key_env: FRED_API_KEY`), not the key itself.

### Cache-on-fetch: data as a growing asset

When the agent fetches data from a runtime API, the fetched series are
**persisted to disk and registered as a pre-collected dataset** so any future
project can re-use them without an API call. This turns ephemeral fetches into a
durable data asset that grows over time.

`lib.fetch_data` ships with six concrete fetchers: `fred` (individual FRED
series, needs `FRED_API_KEY`), `dbnomics` (ECB / OECD / IMF / BIS / Eurostat /
World Bank, through one no-key API), `ons`, `boe_iadb`, `csv_file`, and
`external_archive`. See
[`scripts/lib/data/fetcher.py`](../../scripts/lib/data/fetcher.py) and
[`scripts/lib/data/fetchers/`](../../scripts/lib/data/fetchers/).

#### Pattern

```python
# First time anywhere in the project: hits the API, saves to disk, registers
raw = lib.fetch_data(
    api="ons",
    series={"indpro": "economy/economicoutputandproductivity/output/timeseries/k222/diop"},  # ONS: URL path string
    dataset_id="uk_macro",                  # destination cache
    start="1992-10", end="2026-04",
)

# Any future project, any time: loads from disk, no API call
raw = lib.load_dataset(source="uk_macro", series={"indpro": "indpro"})
```

`fetch_data` wraps a source-specific fetcher, writes the result into the
per-dataset directory under `data/<dataset_id>/`, updates the per-series
metadata + the registry, and returns the loaded DataFrame.

#### Per-source dataset directory

The layout is the `data/<id>/` folder shown under "Dataset registry" above.
Each cache directory **accumulates columns over time** as new series get
fetched, joined by date in `dataset.csv`. A single directory can be populated
by multiple source APIs (e.g., `uk_macro/` gets columns from both ONS and
BoE). Its `metadata.yaml` records each fetched series as its own block, with
its own `api` and `source_spec`, in the format shown under "Dataset metadata
format" above. `fetch_data` writes no `UPDATE.md`; write one by hand if the
dataset is promoted.

Project B that needs `K222 + IUMABEDR` can re-use `K222` from project A's
fetch (no API call) and only fetch `IUMABEDR` if it's not already there.

#### Cache refresh

The library has no cache-age policy. `refresh="cache_first"` (the default)
reuses a cached series when its alias, `api` and `source_spec` match the
stored provenance; `refresh="force"` re-fetches the requested series; and
`refresh="stale_only"` raises `NotImplementedError`. The agent should:
- Tell the user the cached series' `fetched_at` before reusing it.
- Ask before a forced re-fetch that would overwrite a frozen vintage.
- Keep `fetched_at` in `metadata.yaml` so the run is reproducible.

#### Registry auto-update

After a successful fetch, the dataset directory becomes a fully registered
source. From that point, `lib.load_dataset(source="uk_macro", ...)` works
identically to `fred_md`: same metadata schema, same `df.attrs` contents.

## Data acquisition disposition

Whenever a runtime fetch produces new data (whether via
`lib.fetch_data`, manual paste of a CSV the user provided, a Mode B
paper-supplementary download, or an ad-hoc HTTP call inside a
`replicate.py`), **decide with the user what happens to both the data
and the fetcher recipe**. Settle where the data lives (Decision A) before
the first call that writes to disk, unless the user has already agreed a
storage policy for this run; the fetcher-recipe choice (Decision B) and any
later promotion can follow the fetch. This is the parallel
of the orchestrator's Stage 7 (Disposition & Curation) for the data
layer: without it, datasets accumulate in inconsistent locations and
the framework's "every-decision-has-a-source" promise quietly
degrades.

This step is **mandatory whenever new data would land on disk for
the first time**. Never leave data placement implicit. The user makes two
parallel choices: one about the data file, one about the code that
fetched it.

### Decision A: where does the data live

Five options, mirroring Stage 7's structure:

1. **Scratch / one-off.** Leave the file in `outputs/<run_id>/data/`
   or a tmp location. Not registered, not committed, gitignored. For
   exploratory pulls the user does not expect to reuse. Future runs
   that need the same data will re-fetch.

2. **Cache locally** (default for `lib.fetch_data`). Save to
   `data/<dataset_id>/dataset.csv` per the cache-on-fetch convention.
   Auto-registered in `data/registry.local.yaml` (a gitignored overlay
   merged with `data/registry.yaml` at load); future
   `lib.load_dataset(source=...)` calls return immediately.
   Excluded from version control; lives on this machine only. Most runtime fetches land
   here.

3. **Promote to pre-collected** (the FRED-MD / FRED-QD tier). Same
   placement as cache, *plus*: write a full `UPDATE.md` documenting
   the refresh recipe, fill in `metadata.yaml` source / vintage /
   transform notes, register with explicit "preferred" priority,
   and keep the scaffolding (everything except the CSV itself) under
   version control if you use it. For
   datasets the user expects to keep current and reuse across many
   analyses.

4. **Keep with the framework copy.** For small public-good datasets
   (rule of thumb: under ~5 MB compressed) that the user wants to carry
   with their copy of the framework. Use sparingly: it bloats the tree
   and complicates licence tracking. Most
   data does NOT belong here; canonical *scaffolding* does, the data
   itself usually doesn't.

5. **Discard.** Delete the file. For one-time exploration that didn't
   pan out, or duplicates of data the framework already has. Distinct
   from option 1: "scratch" means "I might want it later, so leave it
   in outputs"; "discard" means "I am done with it, so delete now".

### Decision B: what about the fetcher code

Three options:

1. **Keep ad-hoc in `replicate.py`.** The fetch logic lives inline in
   the replication / application script. Reasonable when the recipe is
   one-paragraph specific to this paper (e.g., download Smith-Jones-
   2024's author archive CSV, decode one known column).

2. **Persist as a connector entry.** Add the recipe to
   `skills/data-retrieval/connections/connections.yaml` per the
   connection-configuration pattern (see "Runtime collection" above).
   The entry is documentation for future sessions (the next agent reads
   it before hunting for the source); it does not by itself become a
   `lib.fetch_data` fetcher. Reasonable when the source is one the user expects
   to hit again (a particular author archive, a particular central-
   bank feed).

3. **Promote to `scripts/lib/data/fetchers/<source>.py`.** Add a
   registered fetcher with full I/O contract, retry handling, error
   normalisation, and `metadata.yaml` writeback. Reasonable when the
   source covers a *family* of datasets (e.g., a new central bank's
   API serving many series), justifying the library-grade
   implementation cost. Adding a fetcher makes the same source available to
   every future replication and application.

### Decision support

Propose a specific (data, fetcher) pair based on the session's
signals. Common patterns:

| Session signal | Recommended (data, fetcher) |
|---|---|
| Ad-hoc one-time CSV from a paper's supplementary materials | (1 scratch *or* 5 discard, 1 ad-hoc) |
| User said "fetch the latest BoE rate and just use it" | (2 cache, 1 ad-hoc) |
| User adopting a new economy's data stack (e.g., setting up the context for a new country) | (3 pre-collected, 2 connector) |
| First call to a new central-bank API; user plans more analyses with it | (2 cache, 3 lib fetcher) |
| Small public dataset the user wants to ship for reproducibility | (4 commit, 1 ad-hoc *or* 2 connector) |
| Failed download or duplicate of existing data | (5 discard, 1 ad-hoc) |

Always present the proposal with reasoning, e.g. (Ruritania is a fictional
economy):

> *"This is the first time we've called the Central Bank of Ruritania's API
> and you've said you'll be doing more analyses of Ruritania. I recommend: **cache
> the dataset** at `data/ruritania_macro/` (option 2) and **promote the
> fetcher to `lib/data/fetchers/`** (option 3) so future Ruritania
> replications get it for free. Alternatives: keep it ad-hoc in this
> replicate.py if this is one-off, or scratch-only if you're still
> evaluating whether the central bank is the right source. Which?"*

Per the orchestrator's design principle 3, never ask "what do you
want to do with this data?" without a proposal first.

### Logging

Record the disposition in the research log alongside the data fetch:

```yaml
data_disposition:
  fetched_at: "2026-06-02T14:21:00Z"
  source: "boe_iadb"
  dataset_id: "uk_macro"
  series: ["IUMABEDR", "LPMAUYN"]
  decision_data: "cache"           # scratch | cache | pre_collected | commit | discard
  decision_fetcher: "lib_fetcher"  # ad_hoc | connector | lib_fetcher
  destination_data: "data/uk_macro/"
  destination_fetcher: "scripts/lib/data/fetchers/boe_iadb.py"
  rationale: "Repeat-use UK source; a lib fetcher makes future replications easier"
  prior_runs_using_this: []
```

Both decisions get a log entry, even "scratch" + "ad-hoc". Future
sessions need to know what state previous work left the data layer
in.

### Stop-and-ask checklist

A natural extension of the orchestrator's stop-and-ask rules. If
about to:

- Silently save fetched data to `data/<id>/` without confirming it's
  the right tier → **stop, run the disposition step**
- Add a new connector or lib fetcher without the user agreeing →
  **stop, ask**
- Discard fetched data without the user confirming → **stop, ask**
- Commit a data file > 5 MB → **stop, flag the repo-bloat trade-off**

### Frequency alignment

When combining variables from different frequencies:

| Situation | Method | Notes |
|-----------|--------|-------|
| Monthly → quarterly (flow variable, e.g., GDP) | Sum or average | check if the source reports flow or stock |
| Monthly → quarterly (stock variable, e.g., M1) | End-of-period | last month of the quarter |
| Monthly → quarterly (rate variable, e.g., interest rate) | Average | arithmetic average of 3 months |
| Quarterly → monthly (disaggregation) | Not recommended | use monthly source if available; otherwise Chow-Lin interpolation |

`lib.preprocess.aggregate` validates bin completeness against the source
calendar: an all-missing or partial bin (missing values, missing rows, or a
truncated boundary period) returns NaN rather than a complete-looking total,
and "end"/"start" use the scheduled endpoint rather than silently
substituting an earlier observation. Pass `allow_partial=True` only to opt
into explicit partial aggregates and last-AVAILABLE endpoint semantics.

**Derived series use the index that built them.** When reversing a
published deflation (e.g. reconstructing nominal M1 from FRED's M1REAL),
multiply by the SAME price index the source used to deflate it (CPIAUCSL
for M1REAL); any other index leaves a relative-price component in the
result. See `lib.preprocess.nominalise`'s docstring; at a lower observation
frequency the reversal is approximate and should be described as such.

Always inform the user of the alignment method used.

### Vintage tracking

For reproducibility, record:

```yaml
data_provenance:
  variable: "INDPRO"
  source: "fred_md pre-collected dataset"   # or "FRED API runtime"
  vintage: "2026-04"
  retrieval_date: "2026-04-11"
  transform_applied: "dlog"
  frequency_original: "monthly"
  frequency_used: "quarterly (averaged)"
  missing_obs_filled: 0
  notes: ""
```

This is sent to the research log for every variable in the analysis.

---

## FRED-MD quick reference

Since FRED-MD is the primary pre-collected dataset, here is a quick reference
for the most commonly used variables in VAR/BVAR analysis. The Transform
column is a common VAR choice, not the official FRED-MD code (for example
CPIAUCSL and M1SL have code 6, UNRATE and FEDFUNDS code 2); the official codes
are in `data/fred_md/transform_codes.csv`.

### Output and income
| Series | Name | Transform | Typical use |
|--------|------|-----------|-------------|
| INDPRO | Industrial Production | dlog | Monthly output proxy |
| RPI | Real Personal Income | dlog | Income measure |
| CUMFNS | Capacity Utilisation | level | Output gap proxy |

### Labour market
| Series | Name | Transform | Typical use |
|--------|------|-----------|-------------|
| UNRATE | Unemployment Rate | level | Labour market slack |
| PAYEMS | Nonfarm Payrolls | dlog | Employment |
| CES0600000008 | Avg Hourly Earnings (goods) | dlog | Wage measure |
| HOUST | Housing Starts | log | Leading indicator |

### Prices
| Series | Name | Transform | Typical use |
|--------|------|-----------|-------------|
| CPIAUCSL | CPI All Items | dlog | Headline inflation |
| CPIULFSL | CPI Less Food | dlog | Core-ish inflation (the panel's ex-food series) |
| PCEPI | PCE Price Index | dlog | Fed's preferred inflation measure |
| WPSFD49207 | PPI Final Demand: Finished Goods | dlog | Producer prices |
| OILPRICEx | Crude Oil Price (WTI) | dlog | Commodity/supply shock proxy |

### Money and credit
| Series | Name | Transform | Typical use |
|--------|------|-----------|-------------|
| M1SL | M1 Money Stock | dlog | Narrow money |
| M2SL | M2 Money Stock | dlog | Broad money |
| TOTRESNS | Total Reserves | dlog | Banking reserves |
| BUSLOANS | Commercial & Industrial Loans | dlog | Credit conditions |

### Interest rates and spreads
| Series | Name | Transform | Typical use |
|--------|------|-----------|-------------|
| FEDFUNDS | Federal Funds Rate | level | Policy rate |
| TB3MS | 3-Month Treasury Bill | level | Short rate |
| GS10 | 10-Year Treasury | level | Long rate |
| BAAFFM | Baa minus Fed Funds Spread | level | Credit spread |
| T10YFFM | Term Spread (10Y minus Fed Funds) | level | Yield curve slope |

### Exchange rates
| Series | Name | Transform | Typical use |
|--------|------|-----------|-------------|
| EXUSUKx | US/UK Exchange Rate | dlog | Bilateral rate |
| EXJPUSx | Japan/US Exchange Rate | dlog | Bilateral rate |
| EXCAUSx | Canada/US Exchange Rate | dlog | Bilateral rate |
| TWEXAFEGSMTHx | Trade-Weighted USD (adv. economies) | dlog | Effective exchange rate |

### Mapping model dictionary entries to FRED-MD

For canonical replications, use the paired `scripts/replications/<id>/config.yaml`
for the exact series, frequency and transforms. Treat any other mapping as an
adaptation to agree with the user.

---

## Error handling

- **API key missing:** inform user with the URL to obtain a free key.
- **API rate limit:** the fetchers do not retry. Wait and retry with increasing
  delays; inform the user if delays exceed 30 seconds.
- **Series discontinued:** flag, search for successor series, suggest alternatives.
- **Missing observations:** report count and location; offer interpolation options
  but recommend against silent interpolation. `lib.load_dataset` REFUSES to
  silently row-drop observations missing INSIDE the requested sample (that
  would corrupt calendar lags and differences): choose a contiguous sample,
  impute via `lib.preprocess.impute`, or pass `dropna=False` and handle the
  gaps explicitly.
- **Ragged recent vintages:** fresh FRED-MD/QD vintages update series at
  different speeds, so the last few periods can have internal holes that
  trip the loader's gap guard. Bound the load (`end=` / a config
  `sample_end`) at the sample the analysis actually uses. The
  gertler_karadi and stock_watson replications show the pattern.
- **Event-based instrument series** (shock series with months that have no
  usable announcement) legitimately contain internal NaNs: fetch with
  `lib.fetch_data(..., dropna=False)` and handle them explicitly. The
  standard convention is zero measured surprise for no-announcement
  periods, which keeps the estimation sample contiguous. Dropping the rows
  instead corrupts lags and removes high-leverage observations.
- **Vintage mismatch:** if the user requests a specific vintage and the pre-collected
  dataset has a different one, warn and record the vintage actually used. No
  connector retrieves historical vintages; the user would have to supply that
  vintage as a staged file.
