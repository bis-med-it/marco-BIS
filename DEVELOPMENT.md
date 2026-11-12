# Development guide

## Architecture philosophy

The framework is built so that each part can change without breaking the
others. Every design decision prioritises adaptability:

1. **Skills are modular.** A skill can be revised or replaced on its own as
   long as the contracts other skills rely on (its inputs, outputs and
   handoffs) still hold. The orchestrator delegates; it does not contain
   estimation logic.
2. **Interface contracts, not implementation details.** Skills communicate through
   structured inputs and outputs. The internal implementation can change freely
   as long as the contract holds.
3. **The orchestrator is a coordinator, not a controller.** It defines the workflow
   stages and routes between skills. Its own domain task is the Stage 5
   economic sensibility check; specification, estimation and presentation
   live in their skills and the library. If the workflow itself needs to change, edit `skills/orchestrator/SKILL.md` (and
   `modes.md` for mode triage).
4. **Knowledge base is separate from logic.** Economic context (priors, data sources,
   conventions) lives in the knowledge base, not hardcoded in estimation skills.
   Changing country context never requires editing a skill.
5. **Progressive complexity.** Optional skills (benchmarking, the advisor
   protocol, consistency-audit) add capability without burdening a standard run.

## Interface contracts

The Python contract is the fit-object API (`VARFit` subclasses,
`parameter_draws()`) and `IRFContainer` (`scripts/lib/irf.py`). These
YAML shapes describe what the agent reports, not Python input/output.
They are a conceptual schema for the orchestrator, the research log and
the results-presentation skill.

For the **mathematical conventions** used by the library (horizon indexing,
normalisations, B_hat layout, ddof choices, etc.), see
[`CONVENTIONS.md`](CONVENTIONS.md). When writing equations from code or
translating a paper's math into a library call, consult that file first.


### Specification record

The specification skill's [output schema](skills/specification/SKILL.md#output)
is the decision record: each substantive choice with its rationale and
knowledge source, logged by the research-log skill. A method may add
model-specific fields (e.g., `prior` for BVAR, `instrument` for LP-IV). The
record is not the run configuration: the estimation script reads its own
`config.yaml` through `lib.Spec` (`scripts/lib/specs.py`).

### Results schema

The agent reports each estimation run in this shape:

```yaml
results:
  specification: echo of the input (for audit trail)
  estimation_summary:
    method: string (human-readable name)
    convergence: dict or null (for iterative methods)
  structural:
    irf: standardised IRF structure (see below)
    fevd: standardised FEVD structure or null
    historical_decomposition: structure or null
  diagnostics: dict (from diagnostics skill, or inline)
  sensibility_flags: list of strings
  charts: list of file paths
  code_archive: list of file paths (scripts used)
```

### Standardised IRF structure

The agent reports IRFs in this shape, regardless of estimation method. In
Python the same content is an `IRFContainer`:

```yaml
irf:
  shock_name:
    variable_name:
      horizon: [0, 1, 2, ..., H]
      median: [values]
      lower_68: [values]
      upper_68: [values]
      lower_90: [values]
      upper_90: [values]
```

This allows the results-presentation skill to generate charts from any estimation
method without knowing which method produced them. It also enables the LP-vs-VAR
comparison charts: both output the same structure, so overlaying is trivial.

### Standardised FEVD structure

```yaml
fevd:
  horizon:
    shock_name:
      variable_name:
        median: float
        lower_68: float
        upper_68: float
```

## How to add a new estimation method

Methodology lives in `scripts/lib/`, not in `skills/`. Adding a new method
follows this checklist:

1. **Add a module to `scripts/lib/`** following the patterns in
   [`scripts/lib/README.md`](scripts/lib/README.md): DAG-respecting imports,
   the fit-object API and `IRFContainer` output (see "Interface
   contracts" above), and a
   `parameter_draws()` method if the fit type needs uncertainty bands. If
   the method is a new identification scheme or restriction style, prefer
   adding a kwarg to the existing module (e.g., `higher_moment_restrictions`
   on `sign_restricted_irf`) over creating a new top-level function.

2. **Add unit tests to `tests/`.** At minimum: instantiation on a small
   synthetic example, the IRF / FEVD output-contract test, and a closed-form
   convention contract in `tests/test_conventions.py` for any new
   mathematical choice (horizon indexing, normalisation, ddof, etc.). For
   methods with stochastic output, add a Monte Carlo coverage test under
   `tests/test_coverage.py` (slow).

3. **Update the model-library table** in
   [`skills/model-selection/SKILL.md`](skills/model-selection/SKILL.md).
   Add one row with model class, best-for, sample needs, identification
   options, and the lib entry point (`lib.<function_name>`).

4. **Pair with a canonical paper if applicable.** Add a YAML entry to
   [`skills/model-dictionary/entries/<id>.yaml`](skills/model-dictionary/entries/),
   and ideally a worked replication under
   [`scripts/replications/<id>/`](scripts/replications/) with `config.yaml`,
   `replicate.py`, `README.md`, and `paper_targets.yaml`. Run
   `lib.compare_to_paper` at the end of replicate.py to validate against
   declared paper headlines.

5. **End-to-end smoke test.** Add the new replication to
   `tests/test_replications_smoke.py` (slow). Confirms the pipeline runs
   under `pytest --runslow` without errors.

6. **Add a behavioural test to [`TEST_SUITE.md`](TEST_SUITE.md).** If the
   new method introduces user-facing behaviour the agent will route to
   (new identification scheme, new prior, new estimator), add a test under
   Tier 5 (or the relevant tier) following the Prompt / Expected / Tests
   format. Skip only for purely internal refactors with no API change.

That's it. The orchestrator does not change; Stage 4 routes directly to
the new lib function via `lib.run` or via the generated `replicate.py`. No
new SKILL.md file is needed; see the three-layer architecture in
[`CLAUDE.md`](CLAUDE.md) §"What this project is".

### Adding a new workflow skill

Distinct from adding a method: if you want to add a new *workflow* skill
(a new decision protocol, output format, or governance pattern alongside
the existing orchestrator / model-selection / specification / diagnostics
/ research-log / etc.), the original SKILL.md authoring pattern still
applies:

1. Create `skills/<workflow-skill-name>/` with a `SKILL.md` containing
   frontmatter (`name`, `description`), purpose, inputs / outputs, and
   when to invoke.
2. Wire it into the orchestrator's "Invoking sub-skills" table.
3. Update CLAUDE.md's file-roles table if developer or user sessions
   depend on the skill.

This path is rarer than adding a method; most extension happens in
`scripts/lib/`.

### Other extension points

- **New data source:** add a fetcher under `scripts/lib/data/fetchers/`
  and note it in the data-retrieval skill.
- **New advisor provider:** add a provider block under `advisor:` in
  `config.yaml`.

## How to modify an existing skill

### Changing implementation details

If you want to change how an estimation method works internally (e.g.,
switch the BVAR sampler from Gibbs to Hamiltonian Monte Carlo):

- Edit the lib module (for a workflow skill, its `SKILL.md`).
- Ensure the fit-object API and `IRFContainer` output (see "Interface
  contracts" above) still hold.

### Changing the interface

If you need to change what a skill accepts or returns (e.g., adding a new field
to the IRF structure):

- This is a breaking change. Update:
  1. The skill itself.
  2. The interface contract in this guide.
  3. Any skill that consumes the changed output (typically results-presentation).
  4. The orchestrator, if the workflow routing is affected.

### Changing the workflow

If you want to add, remove, or reorder orchestrator stages:

- Edit `orchestrator/SKILL.md`.
- The stages are defined in prose, not code.
- If adding a new stage, create a new skill for it and add to the sub-skill table.

### Changing the knowledge base structure

If you need new knowledge base categories:

- Add new template files to `knowledge-base/context/` or create new subdirectories.
- Update the lookup protocol in `knowledge-base/SKILL.md`.

## Versioning

Skill frontmatter carries only `name` and `description`, the two fields
the skill loader uses. Do not add `version`, `last_updated` or `changelog`
fields; per-file metadata of that kind goes stale silently. The framework
version is in `CITATION.cff`. To reproduce published results, use that
release and the same data files.

## Quality standards for skills

### Clarity

- Skills are read by the coding agent, which is capable but literal. Write
  instructions as if for a capable but unfamiliar colleague.
- Use concrete examples, not abstract descriptions.

### Completeness

- Every skill must define: inputs, steps, outputs, sensibility checks, references.
- Edge cases should be addressed: what happens when data is missing, estimation
  fails, or the user's request doesn't fit the skill's scope.

### Conciseness

- Keep each SKILL.md under 500 lines where possible.
- Move detailed reference material to `references/` subdirectories.
- The skill should be readable in one sitting.

### Implementation: use packages, not raw code

- `scripts/lib/` calls `statsmodels`, `scipy` and `numpy` where they cover a
  step (OLS VAR fit, state-space smoothing, information criteria) and
  implements the remaining methods from the papers, pinned by the test suite.
- For a method the library covers, call the library. Do not substitute
  another package or write a new estimator (see `packages:` in `config.yaml`).
- For a method the library does not cover, prefer an established package
  over custom code.
- Record which packages and versions a run used in the research log.

### Testability

- A skill should be testable with the Sims (1980) example or an equivalent
  canonical specification.
- Expected outputs for the test case should be documented (at least qualitatively:
  "output should decline after a monetary shock, with peak effect around 6 quarters").

## Keeping the three layers consistent

The skills, the lib, and the dictionary drift apart silently: a paper rebuild
changes variable names but not the tests that pin them; a method is added but
the orchestrator's capability list isn't; a folder is deleted but a doc still
links to it. Two layers of defence guard against this, and a developer runs
both after any batch of skill / lib / dictionary edits:

1. **Automated (every test run):** `tests/test_doc_consistency.py` mechanically
   checks YAML parse-ability, index↔entries sync, stale module/entry counts,
   `verified.status` validity, broken skill links, dangling `lib.*` references, and
   replication-folder completeness. New *mechanically checkable* invariants go
   here.
2. **On demand (semantic):** the [`consistency-audit`](skills/consistency-audit/SKILL.md)
   skill covers what a regex can't: stage-numbering coherence, propose/confirm
   choreography, doc-vs-practice, orphaned skills. Invoke it with "audit the
   framework" before a release or after a rebuild.

## Provenance and licensing

`scripts/lib/` is an original implementation written from the published
papers and pinned by the test suite; no code was ported from the
Canova-Ferroni BVAR_ toolbox (MATLAB), which served only as an
independent cross-check of numerical conventions. The ECB's
BEAR toolbox served as a methodology reference for named conventions
(block-exogeneity shrinkage lambda5, Villani mean-adjusted Gibbs,
relative-magnitude sign restrictions); no code was ported from it either. VAR-Toolbox
(Cesa-Bianchi) served purely as an independent cross-validation target
during development (record in §"Independent cross-checks" below); the harness is
maintainer-local and never distributed (it requires MATLAB and the
GPL-3.0 toolbox). The FRED-MD /
FRED-QD data files are not redistributed; the repo ships only derived
per-series metadata (transform codes, factor flags) and download
instructions. Source-paper PDFs are likewise user-staged and gitignored
(see `papers/README.md`). The framework itself is licensed under Apache-2.0 (see LICENSE and NOTICE).

## Independent cross-checks

Parts of the library were cross-checked numerically against independent
implementations during development: the frequentist VAR layer (estimation,
recursive, long-run and external-instrument identification, impulse
responses, variance and historical decompositions), and selected components
of the Bayesian VAR and sign-restriction code. Results agreed to numerical
precision where the conventions are the same; convention differences kept
on purpose are documented in place in CONVENTIONS.md. These were one-time
checks, and the check harnesses are not distributed.
`tests/test_differential_statsmodels.py` is the standing differential test
within Python.

## Known gaps

The framework does not implement the following. When one of them would
fit a research question, the agent says so and proposes the closest
available alternative.

**Methods:** VECM / cointegrated VAR and Johansen rank tests (the
specification skill's integration doctrine documents the levels-BVAR
workaround); constant-coefficient BVAR with stochastic volatility
(Carriero-Clark-Marcellino class); structural scenario analysis and soft
conditioning (`conditional_forecast` is hard-conditioning, reduced-form
only); Markov-switching and threshold VARs and GIRFs (state-dependent LP
is the closest alternative); structural-break tests at unknown dates
(Bai-Perron / QLR); bridge equations, MIDAS nowcasting, forecast
combination and encompassing tests; fan charts in `lib.report`; DSGE-VAR.

**Inference and options:** Anderson-Rubin weak-IV-robust sets for LP-IV;
heteroskedasticity-robust first-stage F (Montiel Olea-Pflueger) and the
Mertens-Ravn reliability statistic (squared proxy-shock correlation /
reliability eigenvalues; only the reliability-adjusted shock-size
normalisation is implemented, and only for a single instrument); ARRW
(2018) Algorithm-3 importance weights for zero+sign inference, and zero
restrictions beyond the impact horizon; DFM loading normalisation
(Geweke-Zhou) and Bai-Ng factor selection; a moving-block bootstrap for
the multi-instrument proxy path; Andrews automatic HAC bandwidth for LP;
`lib.run` dispatch beyond var/bvar (panel, LP, LP-IV, FAVAR, DFM, TVP,
MF-VAR fits and several identification methods); priors on
BLP control coefficients and marginal-likelihood λ selection; seasonal
adjustment and the other operations listed in
`scripts/lib/preprocess/README.md`.

**Data and environment:** `requirements.txt` caps `numpy<2`; the
Romer-Romer and Mertens-Ravn replications need staged data that no
recipe downloads (openICPSR 116025); the Auerbach-Gorodnichenko
replication starts in 1959, not the paper's 1947.

**Known limitations of what is implemented:**
- Monte Carlo coverage studies (`tests/test_coverage.py`) do not cover
  long-run identification, proxy-SVAR wild-bootstrap or BVAR posterior
  bands.
- The proxy-SVAR `one_sd` estimand is nonstandard (documented); an
  invertibility-based one-s.d. shock would match the usual object.
- `_apply_sign_normalization` applies only the first {variable: sign}
  rule per shock and ignores later entries.
- Rotation and parameter draws share one RNG seed. No measurable
  distortion was found, and independent streams would change every
  seeded replication, so the convention is kept.

## Extending your own copy

This repository is published as a frozen archive and does not accept pull
requests. The intended path is to extend your own copy:

- Add new estimation methods, dictionary entries or knowledge bases in your
  copy, following the interface contract and the verification standard the
  existing components meet (tests for code, field-by-field checks against
  the source paper for entries).
- The interface contract is the acceptance criterion: a component that
  follows the contract works with the framework.

## File listing

See the project structure section in [README.md](README.md) for the full directory layout.

---

## Established conventions

New work should follow these conventions unless there is a specific
reason to deviate; document any deviation in the replication or
application README.

1. **Test-first validation, never legacy-as-truth.** See [`tests/README.md`](tests/README.md). Default validation choices in order:
   - Theoretical property (e.g., Cholesky IRF at h=0 = chol(Σ))
   - Cross-package (statsmodels reference)
   - Limiting behaviour (e.g., as λ→∞, BVAR → OLS)
   - **Avoid** "matches legacy script" as the primary validation; keep it as a regression baseline only

2. **Replication folder structure**: one folder per paper, each containing:
   - `README.md`: paper citation, spec, expected results, scope vs original
   - `config.yaml`: all spec choices (mirrors the dictionary entry)
   - `paper_targets.yaml`: (recommended) declared paper values for validation; see [`scripts/replications/README.md`](scripts/replications/README.md) §"Paper-target validation"
   - `replicate.py`: thin orchestration script (a few hundred lines at most), all heavy lifting in `lib`
   - `outputs/`: HTML report, numerical snapshot, research log (generated on run; gitignored, so regenerate via `replicate.py`)

3. **Dictionary entries vs replications.** Entries are paper specs; replications are runnable demos. The `verified.status` levels are defined in [`skills/model-dictionary/SKILL.md`](skills/model-dictionary/SKILL.md) ("Verification status"). Entries describe the literature, not the implementation.

4. **Documented substitutions.** Many replications use FRED-MD/QD substitutes for paper-specific data (EBP, ff4 instrument, EM sovereign bonds). Each README explicitly documents the substitutions and what they do/don't preserve. The paper-target validation framework (`lib.compare_to_paper`) makes substitution-induced deviations explicit at build time.

5. **Output of fit objects integrates with identification routines.** `BVARFit`, `SteadyStateBVARFit`, `MFVARFit`, `TVPVARFit`, `PanelVARFit`, `HierarchicalPanelVARFit` all subclass `VARFit`; fits that support draw-based inference override (or, like global-VAR panel fits, inherit) `parameter_draws()` and plug into `cholesky_irf` / `long_run_irf` (`bands="parameter_draws"`) and `sign_restricted_irf` (`parameter_uncertainty=True`). Pooled, mean-group and partial-pooling panel fits are point-only (`bands="none"`); see the model-selection table. New fit types should follow this pattern.

6. **Windows console encoding gotcha.** `cp1252` doesn't render Unicode characters like `→`, `Δ`, `Φ`, `Σ`, `≈`. Replication scripts should use ASCII alternatives in print statements (the actual code can use Unicode internally).

7. **State-space wrapping.** `StateSpaceModel` is a thin wrapper around `statsmodels.tsa.statespace.SimulationSmoother`. Supports 3D time-varying `Z`, 3D time-varying `H`, 2D time-varying `obs_intercept`. NaN observations handled natively.

8. **Three-mode entry triage (orchestrator).** Sessions start in Mode A, B or C and can change mode mid-session; see [`skills/orchestrator/modes.md`](skills/orchestrator/modes.md).

9. **Three source tiers.** `scripts/replications/<id>/` (canonical, public-good, ranked first in Mode A search), `scripts/applications/<id>/` (your past work; surfaced as "you've done something similar"), `outputs/` (scratch, always gitignored, never an orchestrator source; generated model outputs are regenerable, but session records saved there are not and must be copied out before deleting). Mode B and C runs get promoted to `applications/` by default; canonical promotion is rare.

10. **Agent behaviour.** See the behaviour rules in [`CLAUDE.md`](CLAUDE.md).

11. **Data layer separation.** `lib.load_dataset(source, series, start, end, vintage)` returns raw, structurally-clean data plus provenance metadata in `df.attrs` (source, vintage, frequency, url, loaded_at, sample bounds, per-series source_code + recommended_transform from FRED-MD `transform_codes.csv`). `lib.transform(df, transforms)` applies per-column transforms (level / log / diff / log_diff / pct_change / scaled variants) as a separate step, because the transform decision belongs with the model.

12. **Spec / run / diagnostics / sensitivity layer.** The framework's *flexibility* spine on top of the deterministic library. `lib.Spec` is an immutable dataclass mirroring the existing replication-config shape (`data` / `model` / `irf` / `forecast` / `metadata`); `Spec.from_replication(id)` loads a replication's `config.yaml` as a `Spec`; `Spec.update(**overrides)` deep-merges into nested blocks (partial overrides preserve siblings). `lib.run(spec, seed)` executes it when the model class (var, bvar) and identification (cholesky, sign_restrictions, proxy_svar, long_run) are supported, and raises `NotImplementedError` otherwise. Use the replication's `replicate.py` for the rest. It also runs an optional forecast block and returns a `SpecResult`. `lib.diagnostics(fit)` produces a one-button residual + stability + lag-selection report. `lib.sensitivity(base_spec, grid)` is a Cartesian-product hyperparameter sweep over `lib.run`. Together they answer "I want to run paper X with a different λ₁ / lag order / sample" in three lines instead of editing config.yaml + rerunning a 200-line replicate.py.

13. **Convention contracts + paper-target validation**: twin discipline layers that catch drift at build time:
   - [`tests/test_conventions.py`](tests/test_conventions.py) holds closed-form contracts pinning the framework's mathematical conventions (horizon indexing, normalisations, B_hat layout, ddof choices). Any new lib function with a convention choice should add a contract here.
   - [`CONVENTIONS.md`](CONVENTIONS.md) is the narrative reference for equation-writeup conventions; consult before translating code to equations or vice versa.
   - `lib.compare_to_paper(results, paper_targets.yaml)` provides per-replication validation against declared paper values; see [`scripts/replications/README.md`](scripts/replications/README.md) for the pattern.

## Architectural decisions

### Library

- **State-space and Gibbs utilities are roots** shared by MF-VAR, TVP-VAR and DFM. If a feature is needed by more than one downstream module, factor it into a leaf-level utility first.
- **No lazy imports.** `_ensure_psd` lives in `var.py` so that `inference.py` can import it without a cycle. A new lazy import is a smell.
- **`stochastic_volatility.py` is self-contained** and reusable across TVP-VAR-SV, future Cogley-Sargent applications, and any model needing univariate SV. Don't put SV logic inside `tvp.py` even if that's the immediate caller.
- **Primiceri 2005 has three implementations** (`fit_tvp_var`, `fit_tvp_var_sv`, `fit_tvp_var_sv_primiceri`) reflecting a hierarchy of complexity. The existing `primiceri_2005_tvpvar/` replication uses the simplest one. Swapping it to `fit_tvp_var_sv_primiceri` is a config change in `replicate.py`.
- **Schema extensions**: some dictionary entries carry fields outside the schema in `model-dictionary/SKILL.md` (for example `implementation_notes.fred_alignment` and `verified.notes`). They are kept because they hold information the agent needs.
- **VARX support added to `fit_var`** with a centralised `_build_x_row` helper used by three bootstrap call sites (`inference.residual_bootstrap_draws`, `identification._proxy_wild_bootstrap`, `identification._proxy_wild_bootstrap_multi`). The convention contracts enforce this centralisation.
- **Multi-shock RWZ 2010, ADRR 2018 (Type 1 + Type 2), FEV-share, relative-magnitude and AFM 2023 higher-moment restrictions are all composable in `sign_restricted_irf`**: one function, dispatching to a separate `_sign_restricted_irf_multi_shock` for `len(shock_names) > 1`.

### Data layer

- **Runtime fetchers live in `scripts/lib/data/fetchers/`** with cache-on-fetch via `lib.fetch_data(api, series, dataset_id, ...)`. Per-dataset directory under `data/<id>/` accumulates columns across calls (full series cached regardless of user's sample); per-series provenance in `metadata.yaml`; fetched datasets are registered in the gitignored `data/registry.local.yaml`.
- **`load_dataset` vs `fetch_data`**: `load_dataset` reads from the registry (FRED-MD, FRED-QD, etc., pre-cached); `fetch_data` hits live APIs (FRED, DBnomics, ONS, BoE IADB, external archives). Both return DataFrames with provenance in `df.attrs`. `load_dataset` filters at load time; `fetch_data` caches full history and filters via parameters.
- **`lib.preprocess`** holds frequency-aware preprocessing (`deflate`, `nominalise`, `aggregate`, `impute`), distinct from `lib.transform` which is per-column scalar transforms (log, diff, log_diff, pct_change). Trend-cycle filters stay in `lib.filters`.
- **PeriodIndex-vs-Timestamp alignment trap**: FRED-QD uses end-of-quarter Timestamps; `pd.resample("QS").mean()` returns start-of-quarter Timestamps. A naive merge across the two produces all-NaN rows. When mixing FRED-QD with externally-resampled data, convert both to `PeriodIndex('Q')` before merging.

## Working style for agents

- **Report negative findings.** The Rigobon 2003 weak-identification result and the BBE 2005 sign issues with FRED-MD substitutes are real findings; do not dress them up.
- **Run tests before claiming success.** Never claim "validated" without showing the test output.
- **Rank alternatives.** When proposing options, say which you would pick and why.
