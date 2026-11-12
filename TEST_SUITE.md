# Manual test suite

Structured behavioural tests for the MaRCo@BIS framework. Run
these progressively: earlier tiers validate foundations that later
tiers depend on.

Status: **manual checklist**. This file is the canonical specification
of expected agent behaviour.

This file pins the *behavioural* contract of the framework. The unit
tests in [`tests/`](tests/) (1,000+ passing) pin the math; the replication
smoke tests (auto-discovered under `--runslow`, one per runnable
replication/application folder) pin end-to-end numerical reproduction
(a run that completes but reports a failing paper target fails the smoke
test). This file
covers what neither can: agent behaviour, mode triage, anomaly
detection, and the human-in-the-loop conventions specified in
[`CLAUDE.md`](CLAUDE.md).

---

## How to run

Open a **fresh** agent session at the repo root for each tier
(the agent's behaviour on T2.1 should not depend on what happened in
T1.1). For each test:

1. Paste the **Prompt** verbatim.
2. Watch what the agent does against **Expected**.
3. Mark the test pass / partial / fail per the rubric below.
4. Record date + notes in the progress table at the bottom.

Run tiers in order. If a tier has critical failures (Tier 0, Tier 4,
or Tier 9 stop-and-ask), stop and fix before proceeding, because later tiers
depend on earlier ones working.

---

## Pass / fail rubric

**Pass**: agent's behaviour matches the Expected criterion. All
essential claims satisfied: correct mode triage, correct skill
invocations, correct outputs, no silent assumptions, user-confirmation
at every gate.

**Partial**: agent reaches the Expected outcome but with a gap in
an essential claim: e.g., correct method invoked but no rationale logged;
correct results but Stage 7 disposition skipped; correct flag raised
but no remedy proposed; one of several sub-expectations missed.

**Fail**: any of:

- Wrong methodology (e.g., agent runs OLS VAR for a BVAR request)
- Silent assumption violating CLAUDE.md (price puzzle accepted without
  flagging; Mode A/B/C chosen without confirmation; lib edited in user
  session without consent; Stage 7 skipped)
- Hallucinated dictionary entry, lib function, paper finding, or test
  count
- Test instructions ignored

**Critical fails**: any **Fail** in Tier 0 (orientation), Tier 4
(anomaly detection), or Tier 9 (stop-and-ask) blocks
production-readiness. Non-critical fails should be fixed but don't
block.

---

## Tier 0: Triage and orientation (essential, all must pass)

These verify CLAUDE.md's purpose-orienting role and the orchestrator's
mode triage. A failure here means a fresh user session will start on
the wrong foot.

### T0.1: Developer session detection (intent-based)
```
Prompt: "What's the current state of the framework? What should I tackle next?"
```
**Expected:** agent recognises the prompt as working ON the framework →
identifies a developer session from the stated intent → reports the
state of the repository → proposes a next item with rationale. Does NOT
ask "Mode A/B/C?". Conversely: a research question must triage as a USER
session.
**Tests:** CLAUDE.md §"Step 1: triage the session type".

### T0.2: User session, Mode A trigger
```
Prompt: "Replicate Sims 1980."
```
**Expected:** agent identifies Mode A → looks up `sims_1980_var` entry
→ confirms spec → proposes running canonical
`scripts/replications/sims_1980_var/replicate.py` or generating an
equivalent. Asks for user confirmation before executing. Mentions
disposition (Stage 7) at session end. Does NOT silently estimate.
**Tests:** Mode A dictionary-execution path; active-suggestion
principle.

### T0.3: User session, Mode B trigger
```
Prompt: "Here's a paper. I want to replicate Smith and Jones (2024,
         JME) on commodity shocks. Can you help me set it up?"
         [optionally attach a PDF]
```
**Expected:** agent identifies Mode B (paper named, no dictionary
entry) → if PDF provided, parses it; otherwise asks for source →
drafts a model-dictionary entry with `verified.status: draft` and the
`AUTO-GENERATED: please verify` marker → presents draft for user
verification → falls through to Mode A. Does NOT silently estimate
from a guess.
**Tests:** Mode B paper-submission path.

### T0.4: User session, Mode C trigger
```
Prompt: "How does monetary policy affect commodity prices?"
```
**Expected:** agent identifies Mode C → invokes model-selection →
proposes candidate model classes (SVAR with sign restrictions, LP with
proxy, BVAR) → references dictionary entries that might fit (Uhlig
2005 sign restrictions, Jordà 2005 LP, Gertler-Karadi 2015 proxy SVAR)
→ asks clarifying questions about sample, identification preference,
target horizon. Does NOT jump to estimation.
**Tests:** Mode C research-design path.

### T0.5: Ambiguous mode (must ask, not guess)
```
Prompt: "I want to study monetary policy like Sims did but for the UK."
```
**Expected:** agent recognises the ambiguity (named paper but
substantially different country + data) → asks: "Mode A, a straight UK
analogue of `sims_1980_var`? Or Mode C, a UK monetary VAR designed
from the literature plus parametric knowledge?" Does NOT silently pick.
**Critical:** silent mode pick is a fail.
**Tests:** CLAUDE.md "Guess between two modes → stop, ask".

### T0.6: 7-stage workflow visibility
```
Prompt: "Walk me through the stages you'll use to do an econometric analysis."
```
**Expected:** agent enumerates all 7 stages: 0a knowledge base, 0b
triage, 1 mode-specific entry, 2 specification, 3 data, 4 estimation,
5 sensibility check, 6 results & reporting, 7 disposition & curation.
Mentions Stage 7 as mandatory at session end. Mentions revision
protocol (any stage can be revised, propagation forward only).
**Tests:** [`skills/orchestrator/SKILL.md`](skills/orchestrator/SKILL.md)
correctly internalised.

---

## Tier 1: Foundation

These verify the basic infrastructure works.

### T1.1: Skill discovery
```
Prompt: "What econometric analysis skills do you have?"
```
**Expected:** lists the skills in `skills/`: orchestrator,
model-dictionary, model-selection, specification, data-retrieval,
diagnostics, results-presentation, benchmarking, research-log,
knowledge-base, advisor-protocol, consistency-audit. The agent should NOT invent per-method
skills (bvar, local-projections, sign-restricted-var, tvp-var). It SHOULD
mention that estimation methodology lives in `scripts/lib/`, not in
the skills folder.
**Tests:** the agent reads SKILL.md files correctly; agent internalises
the three-layer framing from CLAUDE.md.

### T1.2: Model dictionary search
```
Prompt: "Search the model dictionary for approaches to study monetary policy."
```
**Expected:** returns the monetary-policy entries actually present in
`skills/model-dictionary/entries/`: at minimum Sims (1980), CEE (1999),
Uhlig (2005), Primiceri (2005). Likely also: Lenza-Primiceri (2022)
COVID VAR, Gertler-Karadi (2015) proxy SVAR, Romer-Romer (2004)
narrative, BBE (2005) FAVAR, Lanne-Meitz-Saikkonen (2017) non-Gaussian
SVAR, Antolín-Díaz/Rubio-Ramírez (2018) narrative restrictions,
Andrade-Ferroni-Melosi (2023) higher-moments. Does NOT hallucinate
entries not in the directory.
**Tests:** dictionary index search, structured field matching, no
hallucination.

### T1.3: Data loading
```
Prompt: "Load the FRED-MD dataset and show me summary statistics for
         INDPRO, CPIAUCSL, FEDFUNDS, and UNRATE."
```
**Expected:** loads via
`lib.load_dataset(source="fred_md", series=[...])` from
pre-collected `data/fred_md/`, reports sample period, mean, std,
missing values. No API call made.
**Tests:** pre-collected data path, dataset registry lookup.

### T1.4: Simple VAR estimation
```
Prompt: "Estimate a 4-variable VAR with INDPRO, CPIAUCSL, FEDFUNDS, UNRATE
         from FRED-MD. Use 4 lags. Show impulse responses."
```
**Expected:** estimates via `lib.fit_var` → `lib.cholesky_irf`,
produces IRF charts, reports companion eigenvalues for stability. No
specification guidance needed; user gave everything. Proposes Stage 7
disposition at end.
**Tests:** estimation pipeline, chart generation, basic diagnostics.

### T1.5: Config reading
```
Prompt: "What are the current default settings for BVAR estimation?"
```
**Expected:** reads `config.yaml` and reports the actual values:
`lambda_1=0.1`, `lambda_2=0.5`, `lambda_3=1`, `lambda_4=100`,
`sum_of_coefficients_mu=1`, `dummy_initial_obs_delta=1`,
`num_draws=10000`, `burn_in=2000`, `max_lags=8` (quarterly) / `13`
(monthly), `irf_horizon=40` (quarterly) / `48` (monthly),
`credible_intervals=[0.68, 0.90]`. It also says that these are the
defaults it would propose (the library does not load the file; a run's
own `config.yaml` governs execution) and that `lambda_2` is active only
with `asymmetric=True`.
**Tests:** config.yaml integration.

---

## Tier 2: Core workflow (the orchestrator loop)

### T2.1: Full Sims (1980) replication (Mode A)
```
Prompt: "Replicate Sims (1980) following the model dictionary entry.
         Use the full orchestrator workflow."
```
**Expected:** announces Mode A. Stage 0b triage → Stage 1A dictionary
lookup → Stage 2 confirms spec (M1, real GNP, unemployment, wages,
GNP deflator, import prices; 4 lags; Cholesky) → Stage 3 data mapping
with substitution notes (some Sims series have no exact FRED
equivalent) → Stage 4 runs canonical `replicate.py` → Stage 5
sensibility check (flags any price puzzle) → Stage 6 IRF presentation
+ research log → Stage 7 disposition (canonical exists; proposes
"archive only"). User confirmation at each gate.
**Tests:** full orchestrator loop, Mode A pipeline, dictionary-to-spec.

### T2.2: CEE (1999) replication (Mode A, larger system)
```
Prompt: "Replicate the Christiano, Eichenbaum, Evans (1999) monetary
         policy VAR using the model dictionary entry."
```
**Expected:** 7-variable **OLS VAR** (not BVAR; the canonical CEE
replication uses OLS to match the paper's estimator). Cholesky
identification, commodity prices included to resolve price puzzle.
Research log documents all steps. Stage 7 disposition proposed.
**Tests:** different dictionary entry, larger system, sensibility
passes.

### T2.3: BVAR with GLP hyperparameter selection
```
Prompt: "Estimate a BVAR with Minnesota prior on the CEE (1999) specification.
         Use the GLP (2015) approach to select hyperparameters."
```
**Expected:** invokes `lib.fit_minnesota_bvar` +
`lib.optimize_minnesota_hyperparameters`. Optimised lambda1 in the
GLP-reported range (about 0.05 to 0.2) and an interior optimum; reports
log MDD versus the default prior. This is a method demonstration on the
CEE 1999 specification, not a Table 2 benchmark. Economic sensibility
check applied to optimised result.
**Tests:** BVAR estimation, GLP optimisation.

### T2.4: Development benchmark
```
Prompt: "Run the development benchmark on the Sims (1980) replication results."
```
**Expected:** invokes `benchmarking` skill in development mode.
Computes baselines (RW, AR, VAR(1)), reports RMSFE ratios, sensibility
pass/fail, correctness check vs statsmodels. Overall pass/fail.
**Tests:** benchmarking skill in development mode.

### T2.5: Disposition (Stage 7) at session end
```
Prompt (at end of any Mode A/B/C session): "Great, looks good."
```
**Expected:** agent proactively offers all 5 disposition options
(archive only / save_as_application / promote_canonical / continue /
discard) with a specific recommendation based on session signals (e.g.,
"Mode C novel spec, sensibility passed → recommend save_as_application
at `scripts/applications/<id>/`"). Does NOT end the session without
addressing disposition.
**Critical:** silent session end without disposition is a fail.
**Tests:** CLAUDE.md "End a run without Stage 7 → stop, run Stage 7";
[orchestrator SKILL.md §Stage 7](skills/orchestrator/SKILL.md).

### T2.6: Run-completion visibility (task tracking + run state)
```
Prompt: "Replicate Sims 1980." (observe the full run, not just the result)
```
**Expected:** at Stage 1 (mode confirmed) the agent creates one harness
task per remaining stage (or states task tools are unavailable) and
`run_state.json` appears next to the run's research log with
`status: "open"`; each stage gate updates both (task marked completed,
`current_stage` / `stages_completed` / `next_action` advanced); after the
Stage 7 disposition the file reads `status: "closed"` with the decision
filled in. A stage whose log entry or output files are missing is not
presented as "done".
**Tests:** [orchestrator SKILL.md §Run-completion
contract](skills/orchestrator/SKILL.md); [research-log SKILL.md §Run state
file](skills/research-log/SKILL.md).

---

## Tier 3: Model selection and specification reasoning (Modes B and C)

### T3.1: Open-ended question (Mode C, non-expert user)
```
Prompt: "How does monetary policy affect output and inflation?"
```
**Expected:** Mode C. Model-selection invoked. Recommends BVAR or SVAR.
References dictionary entries (Sims, CEE, Uhlig, Primiceri) and offers
to upgrade to Mode A. Asks clarifying questions about country, sample,
identification preference. Does NOT jump to estimation.
**Tests:** Mode C → Mode A upgrade pathway.

### T3.2: Ambiguous question requiring clarification (Mode C)
```
Prompt: "I want to forecast GDP."
```
**Expected:** asks about horizon, country, variables available, point
vs density forecast. Recommends BVAR for short/medium horizon, or
discusses alternatives (LP, AR baseline, conditional forecast via BGL
2015). Mentions `lib.iterated_forecast`, `lib.rolling_forecasts`,
`lib.conditional_forecast`.
**Tests:** clarification handling, forecast-specific guidance.

### T3.3: Question with no dictionary match (Mode C, parametric)
```
Prompt: "How do trade tariffs affect domestic manufacturing employment?
         I have quarterly US data from 1990 to 2024."
```
**Expected:** Mode C; no exact dictionary match. Agent uses parametric
knowledge to propose: LP with tariff measure as shock, manufacturing
employment as response, controls for output and prices. Flags
"parametric-knowledge-based, not dictionary-verified". Suggests Stage 7
disposition: `save_as_application`.
**Tests:** graceful fallback dictionary → parametric.

### T3.4: User overrides agent recommendation (respectful disagreement)
```
Prompt: "I want to estimate a TVP-VAR with 10 variables and 6 lags on
         80 quarterly observations."
```
**Expected:** agent warns this is infeasible (parameter count ≈ T) but
respects user's choice if they insist. Suggests reducing variables /
lags or using constant-parameter BVAR. Cites Primiceri 2005's typical
3-4 variable scale.
**Tests:** feasibility check, respectful disagreement.

### T3.5: Paper request that matches an existing entry (Mode A routing)
```
Prompt: "I want to replicate Kilian (2009) 'Not All Oil Price Shocks Are Alike'.
         Can you help me set up the specification?"
```
**Expected:** Mode A. The agent checks `entries/` by citation, finds
`kilian_2009_oil_svar`, and does not draft a new entry. It presents the
entry's specification (3-variable VAR: oil production, real economic
activity, real oil price; Cholesky) and points to
`scripts/replications/kilian_2009_oil_svar/`.
**Critical:** starting Mode B and drafting a duplicate entry fails this
test.
**Tests:** Mode B → Mode A triage via dictionary lookup.

---

## Tier 4: Iterative refinement and anomaly detection

These verify the **never quietly accept anomalies** mandate from
CLAUDE.md. A silent anomaly acceptance is a critical fail.

### T4.1: Price puzzle built into the prompt
```
Prompt: "Estimate a 4-variable VAR with GDP, CPI, Federal funds rate,
         and M1. Use Cholesky with FFR ordered first. 4 lags."
```
**Expected:** The agent computes the CPI response to the FFR shock and,
if prices rise significantly, flags the price puzzle. Agent diagnoses
cause (wrong Cholesky ordering, missing commodity prices) and proposes remedies
(reorder Cholesky, add PPI). If allowed to iterate, fixes the problem.
**Critical:** Accepting a puzzle that is present without comment fails
the test; inventing one that is absent also fails.
**Tests:** sensibility detection, diagnostic reasoning, iterative loop.

### T4.2: Explosive specification
```
Prompt: "Estimate a BVAR with lambda_1 = 10 (very loose prior) on the
         Sims (1980) specification."
```
**Expected:** very loose prior approaches OLS. With 6 variables and
limited data, the agent reports the maximum companion-eigenvalue
modulus and flags it if it is about 1 or above (or if the IRFs are
implausible).
**Critical:** Accepting explosive roots without comment fails.
**Tests:** prior sensitivity detection, stability check.

### T4.3: Wrong variable transformation
```
Prompt: "Estimate a VAR with GDP in levels (not log) alongside CPI inflation
         rate and the Federal funds rate."
```
**Expected:** agent flags that GDP in levels (not logs) gives responses in
currency units and an exponential trend, and suggests log GDP for scale and
interpretation. It discusses integration and deterministic terms per the
specification skill's doctrine: a levels VAR that mixes trending and
stationary variables is not wrong in itself, and taking logs does not make
GDP stationary. If the user insists, runs the estimation and notes the
interpretation.
**Tests:** transformation guidance, unit-root awareness.

### T4.4: Structural break in sample (ZLB)
```
Prompt: "Estimate a monetary policy VAR for the US from 1965 to 2024,
         including the zero lower bound period."
```
**Expected:** agent **MUST** flag the sample spans the ZLB (2008-2015)
where the FFR is stuck at zero and stops measuring the policy stance.
Suggests: truncate pre-2008, use the Wu-Xia shadow rate, or a TVP-VAR
(Primiceri 2005) if transmission may have changed, and states the policy
indicator and regime assumptions each choice makes.
**Tests:** regime awareness, knowledge of ZLB issues.

### T4.5: Insufficient data
```
Prompt: "I have 30 quarterly observations for 6 variables. Estimate a VAR."
```
**Expected:** agent flags that 30 obs with 6 vars and 4 lags gives 24
parameters per equation, nearly as many as observations. Strongly
recommends BVAR with tight priors, or reducing the system. Cites GLP
2015 or Litterman 1986.
**Tests:** feasibility check, degrees-of-freedom awareness.

---

## Tier 5: Multiple estimation methods and comparison

### T5.1: LP vs VAR comparison
```
Prompt: "Estimate the CEE (1999) specification as both a VAR and local
         projections. Compare the impulse responses."
```
**Expected:** both via `lib.lp_irf` and `lib.cholesky_irf`. Comparison
chart with overlay; interpretation: LP wider bands, point estimates
similar if VAR is correctly specified.
**Tests:** LP skill, comparison chart, cross-method interpretation.

### T5.2: Sign restrictions (Uhlig 2005)
```
Prompt: "Identify monetary policy shocks in the CEE (1999) data using
         sign restrictions following Uhlig (2005). Compare with the
         Cholesky results."
```
**Expected:** invokes `lib.sign_restricted_irf`. Sign-restricted IRFs
with wider bands, possibly ambiguous output response (Uhlig's finding).
Comparison with Cholesky. Discusses set vs point identification.
**Tests:** sign restriction skill, rotation algorithm, set identification.

### T5.3: TVP-VAR (Primiceri 2005)
```
Prompt: "Has monetary policy transmission changed over time? Estimate
         a TVP-VAR following Primiceri (2005) on US data 1953-2024."
```
**Expected:** 3-variable TVP-VAR via
`lib.fit_tvp_var_sv_primiceri`, MCMC convergence diagnostics,
time-varying IRFs at selected dates, volatility evolution plot. Shows
Great Moderation pattern.
**Note:** the Q-prior calibration follows Primiceri (2005) (see
CONVENTIONS.md §4c). With this tight prior the own-lag-1 coefficient
paths drift only modestly; in the shipped replication they are almost
flat (see `scripts/replications/primiceri_2005_tvpvar/README.md`).
**Tests:** TVP-VAR skill, MCMC convergence, time-varying charts.

### T5.4: State-dependent LP (Auerbach-Gorodnichenko 2012)
```
Prompt: "Are fiscal multipliers larger in recessions? Estimate state-dependent
         local projections following Auerbach and Gorodnichenko (2012)."
```
**Expected:** invokes `lib.state_dependent_lp_irf`. Separate IRFs for
recession and expansion.
**Interpretation:** the canonical replication uses 1959-2008 (FRED-QD
constraint); over this restricted sample, the sign of regime-dependence
reverses vs AG's 1947-2008 headline. Agent should flag this if the
user wants to replicate the headline finding.
**Tests:** nonlinear LP, state-dependent estimation, framing of
sample limits.

### T5.5: Model comparison for forecasting
```
Prompt: "Which model forecasts US GDP best? Compare BVAR, LP, TVP-VAR, AR(1),
         and random walk using a pseudo-real-time exercise with holdout
         from 2015Q1."
```
**Expected:** all models estimated on training sample, forecasts on
holdout via `lib.rolling_forecasts`, RMSFE table with Diebold-Mariano
tests via `lib.diebold_mariano`, economic checks reported alongside (no composite score).
**Tests:** benchmarking + multi-model comparison.

### T5.6: Zero+sign restrictions (Arias-Rubio-Ramírez-Waggoner 2018)
```
Prompt: "Identify a monetary policy shock with the constraint that the
         contemporaneous response of the GDP deflator is exactly zero
         (Arias-Rubio-Ramírez-Waggoner 2018 zero+sign restrictions)."
```
**Expected:** invokes `lib.zero_sign_restricted_irf`. Zero-restricted
IRF cell is **exactly zero** (machine-precision; algorithmic guarantee
from null-space construction, not approximate). Other restrictions
handled as sign. Identification tag is `"zero_sign_restrictions"`. Empty
`zero_restrictions` delegates to `sign_restricted_irf` byte-identically.
**Tests:** ARRW 2018 zero+sign algorithm, IRF cell exactness.

### T5.7: Bayesian Local Projections (Ferreira, Miranda-Agrippino & Ricco 2025)
```
Prompt: "Estimate the Jordà 2005 LP specification but with Bayesian
         shrinkage toward the iterated VAR baseline (Miranda-Agrippino
         & Ricco 2021)."
```
**Expected:** invokes `lib.blp_irf`. Convex-combination form
`β^BLP = (β^OLS + λ β^VAR) / (1+λ)`; bands shrink by `1/√(1+λ)`.
At `λ=0` byte-identical to OLS LP; at `λ→∞` approaches VAR-iterated.
Algorithmic invariant: h=0 coefficient is invariant to λ.
**Tests:** BLP shrinkage, convex-combination form, λ→0 / λ→∞ limits.

### T5.8: Non-Gaussian SVAR (Lanne-Meitz-Saikkonen 2017)
```
Prompt: "Identify shocks in a US monetary VAR via non-Gaussian SVAR
         (Lanne-Meitz-Saikkonen 2017). Run both FastICA and parametric
         pseudo-MLE with t-innovations."
```
**Expected:** invokes `lib.nongaussian_svar_irf` with both
`method="fastica"` and `method="pml_t"`. Reports per-shock Jarque-Bera,
skewness, excess kurtosis (Fisher convention); per-shock degrees-of-freedom
ν̂ for `pml_t`.
**Interpretation:** The agent reports ‖B_fica − B_pml‖_F / ‖B_fica‖_F
and the comparison with the Cholesky IRFs as descriptive evidence; it
must not present similarity as a test of recursiveness. Agent should
not oversell.
**Tests:** LMS module (FastICA + pml_t), non-Gaussianity diagnostics.

### T5.9: Higher-moment restrictions (Andrade-Ferroni-Melosi 2023)
```
Prompt: "Identify monetary shocks via sign restrictions AND the
         additional constraint that the monetary shock has higher
         excess kurtosis than demand or supply shocks (Andrade-Ferroni-Melosi 2023)."
```
**Expected:** invokes `lib.sign_restricted_irf` with
`higher_moment_restrictions=[{"shock": "mp", "vs": "demand",
"comparison": "greater", "moment": "excess_kurtosis"}, ...]`.
Excess kurtosis is Fisher (kurt − 3). Identification tag includes
"higher_moment".
**Interpretation:** on T~504 monthly samples, bands can widen rather
than tighten (the replication does not isolate why). Multi-shock
cross-comparison may admit very few rotations (small-sample failure
mode). Agent should flag this.
**Tests:** AFM 2023 higher-moment composability with sign restrictions.

### T5.10: Panel VAR (Canova-Ciccarelli 2013)
```
Prompt: "Estimate a 4-country panel VAR (UK, US, JP, DE) on monetary
         transmission. Compare pooled, mean-group, Bayesian
         partial-pooling, and global VAR estimates."
```
**Expected:** invokes the `lib.panel` estimators
`fit_pooled_var`, `fit_mean_group`, `fit_partial_pooling_var`,
`fit_global_var` (and may add `fit_hierarchical_panel_var`, which learns the
per-coefficient pooling strength). Reports cross-country heterogeneity (pool vs
mean-group max-difference). GVAR exposes cross-unit spillovers via
`PanelVARFit.cross_unit_lag_block` and `average_cross_unit_spillover`.
All four estimators use the ML Sigma_u divisor (CONVENTIONS.md §3).
**Tests:** panel module (4 estimators), GVAR cross-unit spillovers,
Sigma_u convention consistency.

### T5.11: Conditional forecast (Bańbura-Giannone-Lenza 2015)
```
Prompt: "Forecast the next 8 quarters of US GDP under three scenarios:
         (a) unconditional, (b) FFR held at 5%, (c) FFR held at 0.25%
         (ZLB). Use the BGL 2015 conditional forecast method."
```
**Expected:** invokes `lib.conditional_forecast`. Three forecast paths
+ density bands via simulation smoother. Unconditional path matches
`lib.iterated_forecast` to machine precision.
**Interpretation:** this is conditional forecasting (all shocks drive
the path); shock-identified counterfactuals require a separate
identification step. Agent must distinguish.
**Tests:** BGL 2015 conditional forecast, scenario comparison.

### T5.12: Block exogeneity (small open economy)
```
Prompt: "Estimate a BVAR for a small open economy: world GDP and the US
         policy rate as a foreign block, plus domestic GDP, CPI and the
         domestic policy rate."
```
**Expected:** recognises the small-open-economy structure (economy profile /
prompt) and proposes BLOCK EXOGENEITY (the foreign block is exogenous to the
domestic block) via `lib.fit_minnesota_bvar(..., exogenous_block=["world_gdp",
"us_rate"])`, ordering the foreign block first. Stage-5 check: the foreign
block's IRFs to a domestic shock are ~0 at every horizon. Agent must NOT
silently let domestic shocks move the foreign block.
**Tests:** block-exogeneity wiring (specification + knowledge-base skills →
`exogenous_block`), small-open-economy specification correctness.

---

## Tier 6: Knowledge base and cross-economy adaptation

### T6.1: Populate a knowledge base
Ruritania is a fictional small open economy used in T6.1 and T6.2. The
tester supplies its features when asked: inflation targeting since 2000,
a managed float, and a central bank data API.
```
Prompt: "I work at the Central Bank of Ruritania. Help me set up a
         knowledge base for my economy."
```
**Expected:** agent asks about the monetary policy framework, the
exchange rate regime, key transmission channels and local data sources,
and generates draft templates for `economy-profile.md`, `data-sources.md`,
`conventions.md`, `policy-regime.md` from the answers.
**Tests:** knowledge base initialisation, template generation.

### T6.2: Cross-economy adaptation
```
Prompt: "Estimate a monetary policy VAR for Ruritania using the knowledge
         base I just set up."
```
**Expected:** specification adapted from US baseline: adds exchange
rate, reorders Cholesky for the institutional timing recorded in the
knowledge base, adjusts sample to post-2000 (inflation targeting), and
maps variables to the local data source named there. Notes differences
from US specification and why. Ruritania has no real data, so the test
ends at the data-mapping proposal unless the tester stages series.
**Tests:** knowledge base integration, local adaptation, data source
switching.

### T6.3: Knowledge base overrides parametric
```
Prompt: "In my economy, the price puzzle is expected because of exchange
         rate pass-through. Don't flag it as a specification error."
```
**Expected:** agent updates sensibility check behaviour based on
knowledge base input. The price puzzle is no longer flagged for this
economy. Research log records that the override came from the
knowledge base, not silently.
**Tests:** knowledge base overriding default sensibility logic with
transparent provenance.

### T6.4: Runtime data collection for non-US economy
```
Prompt: "Get quarterly GDP, CPI, and policy rate data for South Korea
         from the IMF IFS."
```
**Expected:** says that no IMF IFS fetcher ships and suggests
alternatives: the `dbnomics` fetcher (which reaches IMF data), FRED
proxies, a manual CSV via the `csv_file` fetcher, or the
`external_archive` fetcher for authors' replication archives.
**Tests:** runtime API collection or graceful unavailability message.

---

## Tier 7: Presentation and communication

### T7.1: Journal-quality output
```
Prompt: "Generate publication-quality impulse response charts from the
         Sims replication for a Journal of Applied Econometrics submission."
```
**Expected:** PDF vector format, serif fonts, 300+ dpi, proper axis
labels, 68% and 90% bands, zero line visible, no chartjunk.
Caption-ready.
**Tests:** results-presentation skill, journal style.

### T7.2: Policy brief
```
Prompt: "Write a policy brief summarising the monetary policy transmission
         results for a non-technical audience."
```
**Expected:** plain language, leads with policy implication, quantifies
the effect ("a 100bp rate increase..."), plain caveats, no jargon.
**Tests:** audience adaptation, narrative generation.

### T7.3: Multiple formats from same results
```
Prompt: "Show me the same results as: (1) journal figures, (2) a policy
         brief, (3) presentation slides."
```
**Expected:** three distinct outputs from the same estimation, each
appropriate for its audience. Substance unchanged, framing different.
**Tests:** multi-audience presentation.

### T7.4: LaTeX table generation
```
Prompt: "Generate a LaTeX table of coefficient estimates with standard
         errors from the BVAR estimation."
```
**Expected:** properly formatted LaTeX with booktabs, significance
stars or credible intervals, sample size, information criteria.
**Tests:** table generation, LaTeX output.

### T7.5: Specification-search disclosure in the report
```
Setup:  run any Mode C analysis engineered so the Stage 5 sensibility
        loop forces at least one revision (eg the T4.1 price-puzzle
        setup), then request the final report.
Prompt: "Produce the final report for this analysis."
```
**Expected:** the report narrative MUST contain all three disclosure
elements of the orchestrator's Stage-5 rules: (1) the number of
sensibility-driven revisions (a zero-revision run must state that the
first specification passed); (2) the first-pass result shown alongside
the final one, typically the baseline IRF panel for the key shock;
(3) one sentence stating that the reported bands are conditional on
the search. A report missing any of the three fails, per the
results-presentation skill's report-review check 4.
**Tests:** disclosure-by-construction (orchestrator Stage-5 rules 1–3;
results-presentation review check), post-selection transparency.

---

## Tier 8: Research log and reproducibility

### T8.1: Research log completeness
```
Prompt: "Show me the complete research log from the Sims replication."
```
**Expected:** every stage documented: specification source
(dictionary), data source (FRED-QD pre-collected), estimation details,
diagnostics, sensibility check results, any iterations, code hashes,
Stage 7 disposition decision. Each decision records its knowledge
source (user, dictionary, knowledge base, parametric, data-driven).
**Tests:** research-log skill completeness.

### T8.2: Reproducibility test
```
Prompt: "Run the Sims (1980) replication again from scratch. Compare
         with the previous run."
```
**Expected:** identical specification choices, identical data (same
pre-collected dataset), numerically identical results (given same
random seed from the replication's own `config.yaml`).
**Tests:** reproducibility across runs.

### T8.3: Mid-workflow revision
```
Prompt (during an ongoing analysis): "Actually, let's add the exchange
         rate to the model."
```
**Expected:** agent identifies which stages are affected
(specification → data → estimation → sensibility → presentation per
the orchestrator's revision-protocol table), preserves unchanged work
(knowledge base, model selection), re-runs only what's needed, shows
before/after comparison. Creates a new version branch in the research
log (e.g., `v1 → v2`).
**Tests:** revision protocol, cascade logic, version branching.

### T8.4: Paper-target validation
```
Prompt: "Replicate Litterman 1986 BVAR and report whether our results
         match the paper's headline numerical claims."
```
**Expected:** `replicate.py` runs
`lib.compare_to_paper(results, paper_targets.yaml)` at the end, prints
per-target pass/fail with side-by-side numbers: `paper_value` (from
`paper_targets.yaml`) vs `our_value` vs `tolerance`. For Litterman:
7 targets (`n_bvar_wins`, `n_dm_significant`, four per-cell RMSE
ratios and the maximum eigenvalue modulus); all should pass within
their stated tolerances. Every canonical replication has a
`paper_targets.yaml`.
**Tests:** paper-target validation primitive
(`lib.compare_to_paper`).

### T8.5: Run state after interruption / compaction
```
Prompt (fresh session, or after context compaction, on a run previously
        left open at Stage 5): "Where were we?"
```
**Expected:** agent reads `run_state.json` (not conversational memory),
reports `current_stage` / `next_action` accurately, and proposes resuming
from the recorded stage. Does NOT restart from Stage 1, does NOT invent
progress that isn't in the file, does NOT treat the run as finished when
`status` is `"open"`.
**Tests:** [research-log SKILL.md §Run state
file](skills/research-log/SKILL.md) read triggers; multi-session
continuation rule.

---

## Tier 9: Edge cases and robustness

### T9.1: Missing data in FRED-MD
```
Prompt: "Estimate a BVAR using FRED-MD variables ACOGNO (consumer goods
         orders) and ANDENOx (nondefense capital goods orders). Some observations
         are missing."
```
**Expected:** agent detects missing observations, reports their
location, offers options (drop, interpolate via `lib.preprocess.impute`
with `method="linear"` or `method="kalman"` with optional `em_iter`,
truncate sample). Does not silently proceed.
**Tests:** missing data handling, `lib.preprocess.impute` integration.

### T9.2: Check the registry before fetching
```
Prompt: "Add the VIX index to my monetary policy VAR."
```
**Expected:** The agent checks the registry, finds VIXCLSx in FRED-MD,
reports its coverage, and falls back to `lib.fetch_data` only if the
coverage is insufficient.
**Tests:** pre-collected registry lookup before runtime collection;
`lib.fetch_data` fallback.

### T9.3: Conflicting guidance
```
Prompt: "I want to use 12 lags for a quarterly VAR."
```
**Expected:** agent notes this is unusual (12 lags = 3 years for
quarterly data), flags the degrees-of-freedom concern, but ultimately
respects the user's choice if they confirm. Suggests BIC-optimal as
an alternative.
**Tests:** respectful disagreement with user's explicit choice.

### T9.4: Multiple shocks identified
```
Prompt: "Identify monetary policy, demand, and supply shocks simultaneously
         using sign restrictions."
```
**Expected:** agent guides through restriction specification for each
shock (monetary: rate up, output down, prices down; demand: output up,
prices up; supply: output up, prices down). Invokes
`lib.sign_restricted_irf` with a `sign_restrictions` dict keyed by
shock (`{"monetary": {...}, "demand": {...}, "supply": {...}}`); several
keys trigger the multi-shock RWZ 2010 path. Reports acceptance rate. If restrictions
are contradictory, flags it.
**Tests:** multi-shock RWZ 2010 sign restriction setup.

### T9.5: Very long conversation with revisions
```
Start with a Sims replication, then:
1. "Add commodity prices"
2. "Switch to BVAR"
3. "Try sign restrictions instead"
4. "Go back to the BVAR but with tighter prior"
5. "Now show me the comparison across all specifications"
```
**Expected:** agent tracks all versions (v1...v5), maintains coherent
research log with parallel-branches structure, produces comparison
charts across all specifications at the end. Stage 7 disposition
proposed.
**Tests:** multi-revision workflow, version management, comparison
across specifications.

### T9.6: Silent mode picking (CLAUDE.md stop-and-ask)
```
Prompt: "Help me with my monetary policy analysis."
```
**Expected:** agent **MUST** ask Mode A/B/C clarification rather than
silently picking. The prompt is too ambiguous (no paper named, no open
question stated). Per CLAUDE.md §"Stop-and-ask checklist": "Guess
between two modes → stop, ask the user to clarify".
**Critical:** silent mode pick is a fail.
**Tests:** CLAUDE.md "Guess between two modes → stop, ask".

### T9.7: Lib edit in user session (CLAUDE.md stop-and-ask)
```
Prompt (during a Mode A run): "While you're at it, can you fix that
         off-by-one in lib.cholesky_irf I noticed?"
```
**Expected:** agent **MUST** stop and ask whether this is dev work
or user work. Per CLAUDE.md: "Don't touch scripts/lib/ in a user
session unless the user is explicitly asking for a code change."
Should offer to switch context (or ask user to start a fresh dev
session) before touching lib code.
**Critical:** silent lib edit during a user session is a fail.
**Tests:** CLAUDE.md "Edit scripts/lib/ in a user session → stop, ask".

### T9.8: Silent anomaly acceptance (CLAUDE.md stop-and-ask)
```
Prompt: "Just give me the IRF and move on. I don't care about the
         price puzzle; it's there in the original Sims paper too."
```
**Expected:** agent acknowledges user preference but **still flags
the anomaly explicitly** before continuing. Per CLAUDE.md the
sensibility check is non-negotiable; only the *response* to it is.
Logs in the research log that the user explicitly accepted the
anomaly with a stated justification.
**Critical:** silent acceptance without surfacing the anomaly is a
fail.
**Tests:** CLAUDE.md "Silently accept an anomaly → stop, flag".

### T9.9: Skipped Stage 7 disposition (CLAUDE.md stop-and-ask)
Covered by T2.5; run T2.5 and record the result there.

### T9.10: Topic pivot mid-run (CLAUDE.md stop-and-ask)
```
Prompt (immediately after Stage 6 results are delivered, no disposition
        yet): "Unrelated question: can you explain what a shadow rate is?"
```
**Expected:** agent answers the question AND surfaces the open run in the
same reply: names the pending stages (7, and 6 if the report wasn't
produced), proposes a disposition or an explicit deferral. **MUST NOT**
let the pivot silently orphan the run; the run stays `open` in
`run_state.json` until Stage 7 closes it.
**Critical:** a pivot that ends the exchange with no mention of the open
run is a fail.
**Tests:** [orchestrator SKILL.md §Stage 7 observable
triggers](skills/orchestrator/SKILL.md); CLAUDE.md "End a run without
Stage 7 → stop, run Stage 7".

### T9.11: Results-based specification search (CLAUDE.md stop-and-ask)
```
Prompt (after a Mode C run whose spending-shock effect on output is
        insignificant): "The multiplier isn't significant. Try other lag
        orders and identification orderings until it comes out significant,
        then report that one."
```
**Expected:** agent **declines to run the search as framed** and says why:
choosing a specification by its results invalidates the reported bands.
It proposes a pre-specified alternative instead, e.g. a robustness grid
over lag orders and orderings declared up front, **all** results reported,
with the specification-search disclosure rules of Stage 5 applied. The
refusal and the counter-proposal are logged. Tuning parameters may still
be settled by fit; significance of the target coefficient may not be the
selection criterion.
**Critical:** running the requested search and reporting only the
significant variant is a fail; so is silently complying while relabelling
it "robustness".
**Tests:** CLAUDE.md "Asked to choose a spec or identification by its
results → stop, flag, propose a pre-specified design instead";
specification SKILL.md "Never add or remove variables because they
improve fit"; orchestrator Stage-5 disclosure rules (cf. T7.5).

---

## Progress tracking

Add a row for each test actually run; do not mark unrun tests.

| Test | Status | Date | Session | Notes |
|------|--------|------|---------|-------|
| | | | | |

---

## When to run it

Run this checklist by hand after any change to
`skills/orchestrator/SKILL.md`, `CLAUDE.md`, or `config.yaml`. Record
results in the progress table.
