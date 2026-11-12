---
name: model-selection
description: >
  Recommends appropriate econometric models and specifications based on the user's
  research question, data availability, and analytical objectives. The orchestrator
  invokes this at the start of a Mode C analysis (and as a sanity check in Mode A),
  before data retrieval or estimation. Use whenever
  the user describes a research question, policy problem, or analytical goal without
  specifying a model, or when they ask "what model should I use", "how should I analyse
  this", "what's the best approach for", or describe a macro/financial question they
  want to investigate. Also triggers when the user has chosen a model but the skill
  should validate that choice against the research question.
---

# Model selection

## Purpose

Translate a research question into one or more recommended econometric approaches,
with explicit rationale for why each is suitable and what trade-offs are involved.
This skill helps users without deep econometric expertise arrive at an
appropriate specification, and gives experienced users a second opinion on
their choice.

## When this skill runs

1. **At the start of a Mode C analysis** (open research question), in Stage 1C
   before Stage 2 (specification), and as a quick sanity check in Mode A.
   Even if the user has named a model, this skill validates the choice.
2. **On request.** The user asks what approach to use for a given question.
3. **On revision.** If Stage 5 (economic sensibility) fails repeatedly, the orchestrator
   may return here to consider alternative model classes.

## Workflow

### Step 1: Understand the research question

Gather (through conversation or from the user's initial prompt):

1. **Research question in plain language.** What relationship or phenomenon is being
   investigated? Examples:
   - "How does monetary policy affect output and inflation?"
   - "What drives exchange rate fluctuations in small open economies?"
   - "I want to forecast GDP growth for the next four quarters."
   - "What is the effect of fiscal stimulus on unemployment?"

2. **Analytical objective.** Classify into one or more of:
   - **Structural analysis:** identifying causal effects of shocks (→ structural VAR,
     DSGE, local projections)
   - **Forecasting:** predicting future values (→ BVAR with tight priors, factor models,
     machine learning)
   - **Policy simulation:** counterfactual scenarios (→ DSGE, structural VAR with
     identified shocks)
   - **Descriptive dynamics:** understanding co-movement and transmission (→ reduced-form
     VAR, spectral analysis)
   - **Historical decomposition:** attributing past movements to sources (→ structural
     VAR, BVAR)

3. **Data context:**
   - How many variables? (small system 2-4 vs medium 5-10 vs large 10+)
   - What frequency? (monthly, quarterly, annual)
   - How long is the sample? (short < 80 obs, medium 80-200, long > 200)
   - Any known structural breaks in the sample?
   - Cross-sectional or panel dimension?

4. **Constraints and preferences:**
   - Must results be interpretable to policymakers? (favours simpler models)
   - Is there a specific identification strategy in mind?
   - Is Bayesian or frequentist inference preferred?
   - Any institutional conventions or precedent to follow?

### Step 2: Map to candidate models

Based on the answers, select from the model library. Present 1-3 recommendations
ranked by suitability, with clear rationale.

#### Model library

| Model class | Best for | Variables | Sample needs | Identification | Library entry point |
|-------------|----------|-----------|--------------|----------------|---------------------|
| Classical VAR | Descriptive dynamics, Granger causality, baseline | 2-8 | Medium+ | Cholesky, none | `lib.fit_var` |
| Bayesian VAR (Minnesota) | Forecasting, small samples, many variables; block exogeneity for small open economies (`exogenous_block=`) | 2-20+ | Any | Cholesky, sign restrictions | `lib.fit_minnesota_bvar` |
| Bayesian VAR (Sims-Zha) | Structural analysis with informative priors | 2-10 | Any | Cholesky, exclusion | `lib.fit_minnesota_bvar` (Sims-Zha dummy obs) |
| Bayesian VAR (steady-state / mean-adjusted) | Anchor a variable's long-run mean to a known value (e.g. an inflation target) | 2-10 | Any | Cholesky, sign | `lib.fit_steady_state_bvar` (Villani 2009) |
| Structural VAR (short-run) | Causal shock identification | 2-8 | Medium+ | Cholesky, recursive | `lib.fit_var` + `lib.cholesky_irf` |
| Sign-restricted VAR | Theory-consistent identification | 2-10 | Medium+ | Sign / zero+sign / narrative / magnitude / FEV-share / higher-moment | `lib.sign_restricted_irf`, `lib.zero_sign_restricted_irf` |
| Local projections | Robust impulse responses, nonlinearities | 2-8 | Long | IV, Cholesky | `lib.lp_irf`, `lib.lp_iv_irf`, `lib.state_dependent_lp_irf`, `lib.blp_irf` |
| TVP-VAR | Time-varying parameters, regime changes | 2-5 | Long | Cholesky, sign | `lib.fit_tvp_var_sv_primiceri` and variants; IRFs trace a documented final-period-parameter counterfactual; historical decomposition and shock recovery are unsupported for TVP fits (the library rejects them) |
| Smooth-transition VAR (STVAR) | State-dependent dynamics (e.g. recession vs expansion fiscal multipliers, AG 2012) | 2-5 | Long | Recursive within regimes | no native STVAR primitive; approximated by `lib.state_dependent_lp_irf` (Ramey-Zubairy-style state-dependent LP) |
| Structural VAR (long-run) | Permanent vs transitory shocks | 2-4 | Long | Blanchard-Quah | `lib.long_run_irf` |
| Proxy SVAR | External-instrument identification | 2-8 | Medium+ | Single- or multi-instrument | `lib.proxy_svar_irf` |
| Non-Gaussian SVAR | Identification via non-Gaussianity | 2-6 | Medium+ | FastICA / parametric pseudo-MLE | `lib.nongaussian_svar_irf` |
| Panel VAR | Cross-country / cross-unit analysis (pooled, mean-group, partial-pooling, hierarchical, global) | 2-6 per unit | Short OK | Cholesky, sign | `lib.fit_pooled_var`, `lib.fit_mean_group`, `lib.fit_partial_pooling_var`, `lib.fit_hierarchical_panel_var`, `lib.fit_global_var`. Pooled / mean-group / partial-pooling fits are POINT-ESTIMATE ONLY (generic parameter draws and residual bootstrap are rejected; their VARFit face carries one canonical unit's sample); draw-based bands need the hierarchical fit (posterior draws) or the global VAR (a single stacked system) |
| Mixed-frequency VAR | Mixed monthly / quarterly data | 2-10 | Medium+ | Cholesky | `lib.fit_mfvar_bayesian` (quarterly series must be quarter-on-quarter growth rates, not annualised) |
| FAVAR / BDFM | Large information sets via factors | Many | Medium+ | Cholesky, sign | `lib.fit_favar`, `lib.fit_dfm_bayesian` |
| DSGE-VAR | Theory-data bridge, policy simulation | Model-specific | Medium+ | Model-implied | *not implemented* |

For the complete index of lib modules and their dependency DAG, see
[`scripts/lib/README.md`](../../scripts/lib/README.md). Models marked
*not implemented* are not in the library; if recommended, inform the user of
availability and suggest the closest available alternative.

### Step 3: Recommend and explain

For each recommended model, present:

1. **Why this model.** Connect directly to the user's research question.
   - Example: "You want to understand how monetary policy shocks affect output. A
     VAR with Cholesky identification follows the approach of Sims (1980). For your
     relatively short sample, I recommend adding Bayesian shrinkage, drawing on
     Litterman (1986)."

2. **What it assumes.** In plain language, not just technical conditions.
   - Example: "The Cholesky identification assumes that money supply does not respond
     to output within the same quarter. This is debatable; we can revisit if results
     are problematic."

3. **What it cannot do.** State the limitations plainly.
   - Example: "A reduced-form VAR cannot distinguish supply from demand shocks without
     further identification assumptions."

4. **Trade-offs between candidates.** If recommending multiple options:
   - Example: "Local projections and VAR estimate the *same* impulse
     responses in population (Plagborg-Møller & Wolf 2021, ECMA), so the
     choice is a finite-sample bias-variance trade-off, not a choice
     between different questions. Short lag-augmented LP is nearly
     unbiased but noisy at long horizons; the VAR extrapolates its lag
     structure, gaining precision at the cost of lag-truncation bias
     (Li, Plagborg-Møller & Wolf 2024). With your 120-observation sample
     I'd start with the BVAR for precision and use LP as the
     misspecification check at the horizons you care about."

5. **Literature pointers.** Reference the key paper(s) for the recommended approach
   so the user can read further.

### Step 4: Validate user's choice (if they came with one)

If the user already specified a model:

1. Confirm whether it's appropriate for their stated research question.
2. If appropriate, say so briefly and proceed.
3. If potentially problematic, explain the concern and suggest alternatives, but
   respect the user's expertise. Frame it as "you may also want to consider" rather
   than "you should not use".
4. If clearly inappropriate (e.g., reduced-form VAR for causal claims without
   identification), flag this directly but respectfully.

### Step 5: Confirm and hand off

Present the recommendation summary. Wait for the user to confirm or modify.
Once confirmed, return the selected model class and preliminary specification
to the orchestrator. Stage 2 (specification) runs only for a new or adapted
specification; an unchanged Mode A entry follows Stage 1A.

**Log this stage** via the research-log skill, including the research question,
candidates considered, recommendation rationale, and user's decision.

## Knowledge sources

Use the source-priority order in
[`specification/SKILL.md`](../specification/SKILL.md#decision-hierarchy). For
model selection: search the dictionary first; use the knowledge base to adapt
or filter matches (a US specification may need fundamental changes for an
open economy); use parametric knowledge throughout to judge fit, combine
entries and fill gaps; escalate to `advisor-protocol` for a novel method
beyond the agent's confidence.

## Guiding non-expert users

A key design goal is that a user with a policy question but limited econometric
training can reach a sensible specification. The skill should:

- Avoid jargon in the initial conversation. Use plain language first, technical terms
  second (with brief definitions).
- Ask clarifying questions one at a time, not as a long list.
- Provide concrete examples: "This is the approach Sims used in his 1980 paper to
  study how monetary policy affects the economy."
- Default to well-established, widely-used methods rather than the newest techniques
  unless the user's question specifically requires them.
- Be transparent about what the tool can and cannot do. If the best model
  is not implemented, say so and suggest the best available alternative.

## Example dialogues

**Example 1: Non-expert user**
> User: "I want to understand how interest rate changes affect housing prices in my
> country. I have quarterly data for 15 years."
>
> Model selection recommends: Bayesian VAR with 3-4 variables (policy rate, housing
> prices, GDP, CPI), Minnesota prior, Cholesky identification with the
> recursive order confirmed from policy and data timing. Rationale: 60
> quarterly observations is a small sample, so Bayesian shrinkage helps. The question is about transmission of a specific shock, which
> maps to impulse response analysis.

**Example 2: Experienced user with a choice**
> User: "I want to estimate a DSGE-VAR following Del Negro and Schorfheide (2004)."
>
> Model selection validates: appropriate for bridging theory and data. Notes that
> the DSGE-VAR method is not implemented. Suggests BVAR with
> theory-informed priors as an interim approach, and offers to escalate to the
> advisor protocol for guidance on manually implementing the DSGE-VAR methodology.

**Example 3: Ambiguous question**
> User: "I need to forecast inflation."
>
> Model selection asks: forecast horizon? (short-term → BVAR; medium-term → factor
> model or DSGE). Single country or multiple? Point forecast or density forecast?
> Are you comparing against an existing institutional model? Then recommends
> accordingly.
