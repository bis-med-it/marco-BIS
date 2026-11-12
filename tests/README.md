# Tests

Self-consistency and cross-package validation for `scripts/lib/`. Tests
validate against theory and independent packages, not against earlier
output (see "Why legacy output is never a golden copy" below).

## Validation principles

1. **Theoretical self-consistency**: properties that follow from the math:
   - Cholesky IRF at `h=0` equals `np.linalg.cholesky(Sigma_u)` exactly
   - LP at `h=0` with same controls equals the OLS shock coefficient
   - LP-IV first stage matches a standalone OLS regression
   - At `h=0`, LP-IV of policy variable on its own shock equals 1
   - As `lambda_1 -> infinity`, BVAR posterior mean -> OLS estimate
   - As `lambda_1 -> 0`, BVAR coefficients shrink to `rw_deltas`
2. **Cross-package validation**: `statsmodels.tsa.api.VAR` is independently
   maintained:
   - `fit_var` coefficients match `statsmodels` to 1e-10
   - AIC, BIC match `statsmodels` to floating-point precision
   - `cholesky_irf` point estimate matches `statsmodels.results.irf().orth_irfs`
3. **Property-based checks**: outputs satisfy invariants regardless of input:
   - Confidence bands are ordered: `lower_90 ≤ lower_68 ≤ median ≤ upper_68 ≤ upper_90`
   - Sign-restricted IRFs satisfy their own restrictions at every constrained horizon
   - Parameter draws are centred on the point estimate

## Running

```bash
# pytest is a development dependency, not part of requirements.txt:
python -m pip install pytest

# Fast suite (~1-2 minutes). FRED-dependent tests skip cleanly
# when the data CSVs are absent.
python -m pytest tests/ -q

# Include the slow end-to-end replication smoke runs (about 20 minutes):
python -m pytest tests/ -q --runslow

# The documentation/spec drift guard on its own:
python -m pytest tests/test_doc_consistency.py -q
```

## What's tested

1,000+ tests, one test module per lib module, plus the
cross-cutting layers below (and the differential, textbook-oracle and
higher-moment suites):

| Layer | Files | What it pins |
|---|---|---|
| Per-module unit tests | `test_var.py`, `test_bvar.py`, `test_identification.py`, `test_lp.py`, `test_filters.py`, `test_tvp.py`, … (one per lib module) | API contracts, theoretical properties, limiting behaviour |
| Convention contracts | `test_conventions.py` | The math conventions in `CONVENTIONS.md` (horizon indexing, normalisations, ddof, prior calibrations) as closed-form assertions |
| DGP-recovery contracts | inside the relevant module tests | Parameter-level recovery on known data-generating processes (shape and finiteness tests cannot catch orientation, sign or calibration errors) |
| Coverage studies | `test_coverage.py` (slow) | Monte Carlo band coverage for the inference machinery |
| Doc/spec drift guard | `test_doc_consistency.py` | YAML parse-ability, index↔entries sync, stale counts, broken links, dangling lib references |
| End-to-end smoke | `test_replications_smoke.py` (`--runslow`) | Full replication runs against their `paper_targets.yaml`; a run that completes but reports a failing paper target fails the test (numerical-regression guard) |

## Why legacy output is never a golden copy

The unit and contract tests validate against theory (closed forms,
known DGPs, published numbers), not against the output of a previous
implementation. The reason is methodological: when two implementations
share a bug, a regression test against the older one falsely confirms
the newer one. Those contracts are therefore stated as properties the
correct mathematics must satisfy, with the source (paper equation,
CONVENTIONS.md entry, or analytic limit) named in the test docstring.

The exceptions are explicit regression checks: the replication smoke
tests, whose targets for some replications are the authors' own results
from the same replication (each `paper_targets.yaml` says which), and a
few end-to-end tests that pin a replication's headline number. These
catch drift from an earlier run; they cannot catch an error that run
already contained.

## Adding new tests

When adding a new library module, prefer (in order):
1. A theoretical property that must hold by construction
2. A comparison against `statsmodels` / `scipy` / `numpy` where applicable
3. A limiting-behaviour check (e.g., as a hyperparameter approaches a
   degenerate value, the output should approach a known closed form)

Avoid:
- "Matches output of `legacy_script.py`": that's a regression test, not
  a correctness test
- "Matches published Table 1 of paper X": useful but fragile (paper data
  may not be reproducible from FRED revisions)
