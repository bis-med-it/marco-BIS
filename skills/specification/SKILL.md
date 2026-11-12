---
name: specification
description: >
  Handles all model specification decisions: variable selection, lag order, prior
  hyperparameters, identification strategy, and sample period. Combines multiple
  knowledge sources (literature, parametric knowledge, user expertise, knowledge base,
  data-driven search) with economic logic checked at every step.
  Use whenever the orchestrator reaches the specification stage, or when the user asks
  "which variables should I include", "how many lags", "what prior", "how to identify
  shocks", or any question about model design. Also triggers when specification
  revisions are needed after failed sensibility checks.
---

# Specification

## The specification problem

Econometric results are only as good as the specification. Variable selection, lag
order, prior hyperparameters, identification, and sample period are all choices that
shape the results fundamentally. Getting them wrong produces technically valid but
economically meaningless output.

The challenge: these choices interact. Adding a variable changes the optimal lag order.
Changing the sample period changes the appropriate prior. Changing identification
changes which variables are needed. The specification skill must handle this as a
coherent decision, not a sequence of independent choices.

## Decision hierarchy

This is the framework's **canonical source-priority order**, the single source
of truth that CLAUDE.md, the orchestrator, and model-selection all reference.
Every specification decision follows it:

```
1. User's explicit choice        (if they know what they want, respect it)
       ↓ if not specified
2. Model dictionary              (literature specifications checked against the papers)
       + combined with
3. Knowledge base                (local context, institutional conventions)
       + combined with
4. Parametric knowledge          (active throughout: adapts, combines, fills gaps)
       ↓ if data-driven tuning is needed
5. Data-driven search            (grid search, cross-validation, information criteria)
       ↓ always subject to
6. Economic logic check          (at every level; results that violate economic
                                  theory are flagged and investigated, never
                                  silently accepted or silently discarded)
```

**Important:** levels 2-4 are not a strict if-then cascade. They work in
parallel. The dictionary provides a verified anchor. The knowledge base
adapts it to local context. Parametric knowledge reasons about whether the
combination makes sense, suggests modifications, fills gaps, and flags
potential problems. The agent is always thinking, not just looking up.

For example: the dictionary says CEE (1999) used 7 variables with 4 lags.
The knowledge base says the sample has only 80 observations. Parametric
knowledge reasons: "7 variables × 4 lags = 28 parameters per equation,
which is too many for 80 observations. I recommend reducing to 4-5
variables or tightening the prior." This reasoning is not in the dictionary
or the knowledge base; it comes from the agent's understanding of
econometrics.

### The model dictionary's role

The model dictionary (see `../model-dictionary/SKILL.md`) contains entries
extracted from canonical papers and checked against them. Each entry documents the exact variable
set, lag order, identification, prior, and expected results. When a dictionary
entry matches the user's research question:

- **Start from its specification.** Use it as a template, not a mandate.
- **Adapt for local context.** The knowledge base and parametric reasoning
  jointly determine what needs to change.
- **Use its expected results for sensibility checking.** The dictionary specifies
  what the impulse responses should look like, but the agent should also reason
  about whether those expectations apply in the current context.
- **Go beyond it when needed.** If the user's question doesn't fit any entry
  exactly, the agent combines insights from multiple entries, draws on parametric
  knowledge, and proposes a novel specification with clear rationale.

### How this works in practice

**Example: choosing variables for a monetary policy VAR**

1. User says: "I want to study monetary policy transmission." (no variable list)
2. Knowledge base says: "In our economy, the exchange rate is a key transmission
   channel" (from economy-profile.md).
3. Literature says: Sims (1980) used M1, GNP, unemployment, wages, prices, import
   prices. Christiano, Eichenbaum, Evans (1999) used output, prices, commodity
   prices, Fed funds rate, non-borrowed reserves, total reserves, M1.
4. Parametric knowledge says: a minimal monetary policy VAR needs at least output,
   prices, and a policy rate. Common additions: money aggregate, exchange rate,
   commodity prices.
5. Data-driven: not applicable for variable selection (see below).
6. Economic logic: variables must form a coherent system where the identified shock
   has a clear economic interpretation.

**Result presented to user:**

> "For monetary policy transmission in your economy (small open, inflation targeting),
> I recommend starting with:
> - Policy rate (your instrument, from knowledge base)
> - Real GDP (output measure)
> - CPI (your inflation target variable, from knowledge base)
> - Nominal effective exchange rate (key transmission channel in your economy,
>   from knowledge base)
>
> Optional additions depending on your focus:
> - Money aggregate (M2, if you want to study the money channel)
> - Commodity price index (to control for supply shocks, following CEE 1999)
>
> This specification follows Sims (1980) in spirit but is adapted to your economy's
> structure. The exchange rate is essential given your open economy profile.
> Shall I proceed with this, or would you like to modify?"

## Variable selection

### Principles

1. **Theory first, data never.** Variables are selected because economic theory says
   they belong in the system, not because they improve statistical fit. Data-driven
   variable selection (stepwise, LASSO on the variable set) is not appropriate for
   structural VAR analysis: it optimises prediction, not causal identification.

2. **Parsimony matters.** Every additional variable costs degrees of freedom
   (an n-variable VAR has n² slope coefficients per lag; adding one variable
   adds 2n+1 per lag). VARs with more than 8-10 variables require
   Bayesian shrinkage or factor augmentation. Recommend the smallest system that
   answers the research question.

3. **Omitted variable bias is real.** But so is overfitting. The trade-off depends
   on sample size, frequency, and the question asked.

### Decision process

#### Step 1: Core variables (mandatory)

Determined by the research question and identification strategy:

| Research question | Core variables (minimum) |
|-------------------|--------------------------|
| Monetary policy transmission | Output, prices, policy rate |
| Fiscal policy effects | Output, prices, government spending, tax revenue |
| Exchange rate dynamics | Output, prices, interest rate, exchange rate |
| Oil price shocks | Output, prices, oil price, interest rate |
| Financial conditions | Output, prices, credit, asset prices, interest rate |

Source: literature + parametric knowledge.

#### Step 2: Context-specific additions

From the knowledge base:

- Open economy → add exchange rate (if not already in core)
- Commodity exporter → add commodity price
- Dollarised economy → add foreign interest rate, foreign money
- Financial development → add credit spread or financial conditions index
- Inflation targeter → use CPI matching the target definition

When these additions introduce a **foreign block** (world demand, foreign policy
rate, commodity prices) for a small open economy, also treat that block as
exogenous to the domestic block (see "Step 5: Block exogeneity" under Prior
hyperparameter selection below).

#### Step 3: User modifications

Present the proposed variable list. The user may:
- Add variables they consider essential for their specific question.
- Remove variables they consider irrelevant.
- Substitute proxies (e.g., industrial production instead of GDP if monthly).

Accept all user modifications without argument, but flag if a modification creates
a potential identification problem or omitted variable concern.

#### Step 4: Feasibility check

Verify the proposed system is estimable:
- Count: n variables × p lags × n equations = total parameters.
- Compare against sample size T. Rule of thumb: T > 3 × (total parameters per equation).
- If infeasible: recommend reducing variables, reducing lags, or switching to BVAR
  with tight priors (or FAVAR if the variable list is large).

Present this check to the user before proceeding.

## Transformation & integration treatment

The single most consequential transform decision for a macro VAR is how to
treat unit roots and shared stochastic trends. It is decided HERE, as part
of the specification, never silently inherited from a data file.

### Doctrine

1. **Structural VARs: default to (log-)LEVELS, with unit-root-aware
   priors.** The Sims-Stock-Watson (1990) tradition: levels estimation
   gives consistent coefficient estimates whether or not unit roots /
   cointegration are present, and preserves long-run relationships that
   differencing destroys. In Bayesian estimation, discipline the levels
   system with the sum-of-coefficients (``mu``) and
   dummy-initial-observation (``delta``) priors; that is exactly what
   those dummies exist for.
2. **Differencing is for forecasting-oriented and factor models**, or
   for when the user explicitly wants growth-rate dynamics. Differencing a
   cointegrated system discards the error-correction terms and distorts
   long-horizon responses.
3. **FRED-MD/QD `tcodes` are factor-model transforms; never auto-apply
   them to a structural VAR.** The McCracken-Ng transformation codes are
   designed to make each series stationary for principal-components
   factor extraction. The data layer surfaces them as
   ``recommended_transform``; for structural work treat them as a
   stationarity *reference*, not a recommendation. Applying them
   wholesale silently differences away every long-run relationship in
   the system.
4. **Cointegration: flag, don't fake.** When the system plausibly shares
   stochastic trends (e.g. consumption-income, money-prices-output,
   term-structure spreads), say so explicitly. The framework has
   **no VECM estimator and no Johansen test** (a known gap). The
   menu to present: (a) levels VAR with SoC/DIO priors (captures
   the levels information without imposing the rank), (b) differenced
   VAR with the long-run caveat stated, (c) defer to external VECM
   tooling. Never present a differenced system as if it answered a
   long-run question.

### Decision output

Record, per variable: the transform applied, the assumed integration
order, and (system-wide) whether cointegration was considered and what
was chosen, with the knowledge source, like every other specification
decision.

## Lag order selection

### Decision process

#### Step 1: Literature or user default

If the user or literature specifies a lag order, use it. Common conventions:
- Quarterly data: 4 lags (one year) is standard starting point.
- Monthly data: 12 lags (one year) or 6 lags (common in practice).
- Annual data: 1-2 lags.

#### Step 2: Information criteria search

If lag order is not pre-specified, estimate the classical VAR at lag orders
1 through `max_lags` (from config.yaml) and compute:

| Criterion | Formula | Tendency |
|-----------|---------|----------|
| AIC | -2L/T + 2k/T | Tends to overfit (more lags) |
| BIC (Schwarz) | -2L/T + k·ln(T)/T | Conservative (fewer lags) |
| HQ (Hannan-Quinn) | -2L/T + 2k·ln(ln(T))/T | Between AIC and BIC |

Present all three. Default recommendation: BIC (conservative, avoids overfitting).
Note if AIC and BIC disagree substantially; this indicates sensitivity.

#### Step 3: Economic logic check

Information criteria can recommend lag orders that don't make economic sense:

- **Too few lags (p=1 for quarterly data):** check the residuals for
  remaining autocorrelation and report how the results change with a longer
  lag order. A late peak in the expected responses is not by itself a reason
  for more lags: a VAR(1) can produce a response that peaks after several
  quarters.

- **Too many lags:** may capture noise rather than economic dynamics, and waste
  degrees of freedom. Flag if the selected lag order uses more than 60% of
  available degrees of freedom.

- **Residual autocorrelation at selected lag:** if residuals at the BIC-selected
  lag show significant autocorrelation (Ljung-Box or Breusch-Godfrey), increase
  the lag order until autocorrelation is resolved. This is a statistical
  requirement, not a preference.

#### Step 4: Present the recommendation

> "Information criteria suggest:
> - AIC: 4 lags
> - BIC: 2 lags
> - HQ: 3 lags
>
> I recommend 4 lags. While BIC prefers 2, the Ljung-Box test at 2 lags shows
> residual autocorrelation in the output equation (p=0.03). At 4 lags, residuals
> are clean. Additionally, 4 lags covers one full year, which is standard for
> quarterly macro VARs and allows the model to capture seasonal and policy
> transmission dynamics.
>
> Degrees of freedom check: 6 variables × 4 lags = 24 parameters per equation,
> with T=120 observations. Adequate.
>
> Shall I proceed with 4 lags?"

## Prior hyperparameter selection (Bayesian models)

This is where data-driven search becomes valuable, but only within economically
sensible bounds.

### Decision process

#### Step 1: Literature defaults

Start with well-established values from the literature:

| Hyperparameter | Literature default | Source |
|----------------|--------------------|--------|
| lambda_1 (tightness) | 0.1 | Litterman (1986), common in practice |
| lambda_2 (cross-variable) | 0.5 | Kadiyala and Karlsson (1997); active only with `asymmetric=True`, a no-op under the default symmetric prior |
| lambda_3 (lag decay) | 1 | Litterman (1986) |
| mu (sum-of-coefficients) | 1 | Doan, Litterman, Sims (1984) |
| delta (dummy initial obs) | 1 | Sims (1993) |

#### Step 2: Knowledge base adjustments

The knowledge base may specify preferred values:
- "We typically use lambda_1 = 0.2 in our department" → use 0.2.
- "Our variables are well-behaved, loose priors work" → increase lambda_1.

#### Step 3: Grid search with a stability screen

If the user wants data-driven hyperparameter selection, run a grid search. If
initial results fail sensibility checks, a grid search is a class-3 proposal
(see "Iterative revision after failed sensibility checks" below), not an
automatic step:

**Grid definition** (default symmetric natural-conjugate prior):

```yaml
grid:
  lambda_1: [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
  lambda_3: [1, 2]
  mu: [0.5, 1, 2, 5]
  delta: [0.5, 1, 2, 5]
```

`lambda_2` is left out: under the symmetric prior it has no effect, so every
value would tie and a reported "best `lambda_2`" would be meaningless. It
becomes active only with `asymmetric=True`, which treats `Sigma_u` as a fixed
plug-in; the library defines no marginal likelihood for that mode
(`log_marginal_data_density` raises), so tune `lambda_2` there by
out-of-sample forecast fit, not by marginal likelihood. Switching to the
asymmetric prior changes the estimator: propose it as a specification
choice, never as a way to activate one hyperparameter.

**Evaluation criteria: statistical fit, with economic checks reported:**

For each grid point, estimate the model and evaluate:

1. **Statistical fit:**
   - Marginal likelihood (if computable in closed form for the conjugate case).
   - Out-of-sample forecast RMSE (rolling window or pseudo-real-time).
     **No look-ahead:** when the selected hyperparameters will be judged by
     an OOS exercise, tune them on the initial training window only (or
     re-tune inside each origin). Hyperparameters optimised on the full
     sample have already seen the evaluation observations, and the "OOS"
     gains they show are leakage, not forecasting skill.
   - Log predictive score (for density forecast evaluation).

2. **Economic checks (reported, not scored):**
   - Do impulse responses have the expected signs at impact?
   - Do responses settle within reasonable horizons (die out if transitory, flatten if permanent)?
   - Is the price puzzle present? (If the user's knowledge base says it shouldn't be)
   - Is the sacrifice ratio plausible (output loss per percentage point of
     disinflation)?

3. **Score:**

   ```
   score(theta) = statistical_fit(theta)    if stable(theta) and estimated(theta)
                = -infinity                  otherwise
   ```

   The grid screens only for stability (explosive roots) and failed
   estimation. Economic checks (impact signs, price puzzle) are computed and
   reported for each grid point but do not enter the score. A prior change
   motivated by them is a class-3 proposal, run as a labelled sensitivity
   analysis.

   **Disclosure:** the orchestrator's Stage 5 disclosure rules apply. Report
   how many combinations the stability screen excluded.

**Present the grid search results** (illustrative, hypothetical numbers):

> "Grid search over 192 hyperparameter combinations:
> - 14 were excluded because they produce explosive roots.
> - Best marginal likelihood among the remaining combinations: lambda_1=0.1,
>   lambda_3=1, mu=1, delta=1 (the literature default).
> - Economic checks for each grid point (impact signs, price puzzle) are
>   reported in the research log.
>
> I recommend lambda_1=0.1. Full grid results are in the research log."

#### Step 4: Marginal likelihood-based selection (advanced)

For formal hyperparameter selection, maximise the marginal data density:

    p(Y | lambda) = integral p(Y | B, Sigma, lambda) p(B, Sigma | lambda) dB dSigma

This is available in closed form for the conjugate Minnesota prior (Giannone,
Lenza, Primiceri, 2015). For non-conjugate priors the shipped conjugate MDD
calculation does not apply: report the MDD as unavailable unless a separately
validated estimator suited to that model is supplied, and do not use the plain
harmonic-mean estimator.

**Implausible responses:** if the marginal-likelihood-optimal hyperparameters
give implausible responses, report this and propose an alternative as a
sensitivity analysis.

#### Step 5: Block exogeneity (small open economies)

If the knowledge base's economy profile is a **small open economy**, the foreign
block (world demand, foreign policy rate, commodity prices) should be treated as
**exogenous** to the domestic block: domestic shocks must not move foreign
variables. This is a specification-correctness matter, not a refinement: without
it the estimated VAR lets a domestic shock feed back into the world economy,
which is economically wrong. Propose it whenever the profile signals a small open
economy and the variable set includes a foreign block:

> "Your economy profile is a small open economy, so I'll make the foreign block
> (world GDP, US policy rate) exogenous, so domestic shocks won't feed back into
> it. This is the standard block-exogeneity restriction."

Implement it through the Minnesota BVAR's per-equation prior:

```python
fit = lib.fit_minnesota_bvar(
    data, lags=p,
    exogenous_block=["world_gdp", "us_rate"],   # the foreign block
)
```

This shrinks the foreign equations' coefficients on every domestic variable's
lags toward zero (``prior.lambda_bex``, default 0.001). It uses the asymmetric
(per-equation) Minnesota path automatically, because the symmetric natural-conjugate
prior cannot encode a per-equation exclusion. **Constraint:** the asymmetric
path does not support observation weighting, so ``exogenous_block`` cannot be
combined with Lenza-Primiceri ``obs_scale`` (the library rejects the
combination); if a small-open-economy spec also needs COVID-style
down-weighting, choose one lever and handle the other differently (e.g.
shrinkage, or a sample sensitivity run). Order the foreign block FIRST in
the variable list so a recursive identification also keeps the contemporaneous
foreign response to a domestic shock at zero. Stage-5 sensibility check: confirm
the foreign block's IRFs to domestic shocks are ~0 at every horizon.

## Identification strategy

### Decision process

Identification is fundamentally a theoretical choice, not a data-driven one.
The data cannot tell you which shock is "monetary policy" without assumptions.

#### Step 1: Research question determines the approach

| Research question | Natural identification | Why |
|-------------------|------------------------|-----|
| "Effect of monetary policy on X" | Cholesky (recursive) or narrative | Clear causal question with timing assumptions |
| "What drives output fluctuations?" | Sign restrictions | Multiple shocks, theory gives signs |
| "Has monetary transmission changed?" | TVP-VAR with recursive | Time variation, not identification, is the focus |
| "Effect of specific historical shocks" | Narrative | Known events provide identification |
| "Robust impulse responses" | LP-IV with external instrument | Robust to VAR misspecification |

#### Step 2: Knowledge base informs details

The knowledge base specifies:
- **Variable ordering for Cholesky:** set by the information and timing
  assumptions, not statistical convenience. In the CEE (1999) benchmark, slow
  variables (output, prices) precede the policy rate and fast financial
  variables (reserves, money) follow it. Other designs and other economies
  need their own rationale, which the knowledge base should record.
- **Expected signs for sign restrictions:** what does economic theory predict
  in this economy? A contractionary monetary shock should raise the policy rate,
  lower output, and lower prices, but exchange rate and asset price responses
  depend on the economy's structure.

#### Step 3: Literature precedent

If the user's question closely matches a published study, adopt that study's
identification as the starting point. This is the safest choice for publication:
referees understand established identification strategies.

#### Step 4: User choice

Some users have strong views on identification, perhaps from a specific
theoretical model or from their own prior research. Respect these. If the
chosen identification is non-standard, the advisor protocol can verify
the mathematical implementation.

#### Step 5: Robustness

Regardless of the primary identification:
- **Always offer a robustness check** with at least one alternative identification.
- If using Cholesky: check sensitivity to variable ordering.
- If using sign restrictions: check sensitivity to restriction horizon.
- If using LP-IV: report first-stage F-statistic.
- If using a proxy SVAR: report the first-stage F, and use
  `bands="moving_block_bootstrap"` (Jentsch and Lunsford 2019) for a single
  instrument. The library default, `wild_bootstrap`, is asymptotically
  invalid for proxy SVARs and is kept only to compare with legacy results;
  do not present its bands as valid inference. The multi-instrument path
  has no moving-block option, so present its bands as illustrative only or
  report point responses.

## Sample period selection

### Decision process

#### Step 1: Knowledge base and regime awareness

The knowledge base (policy-regime.md) specifies known structural breaks.
These determine the maximum sensible sample:

- Inflation targeting adoption → start from adoption date.
- Exchange rate regime change → do not pool fixed and floating periods.
- Major crisis (GFC, pandemic) → either include with awareness or exclude
  as a robustness check.

#### Step 2: Literature guidance

If replicating a published study, use the original sample period (or extend
to present if updating the study).

#### Step 3: Trade-off: length vs homogeneity

- Longer samples give more precise estimates but risk spanning structural breaks.
- Shorter samples ensure regime homogeneity but reduce degrees of freedom.
- Bayesian methods help with short samples (informative priors compensate for
  fewer observations).

Present this trade-off to the user:

> "Your economy adopted inflation targeting in 2000Q1. Using data from 2000Q1
> to 2024Q4 gives 100 quarterly observations, which is tight for a 6-variable VAR with
> 4 lags (24 parameters per equation). Options:
> 1. Use 2000Q1-2024Q4 with BVAR and informative priors (recommended).
> 2. Extend to 1990Q1 for more observations, but this pools two policy regimes.
> 3. Use 2000Q1-2024Q4 but reduce to 4 variables for more degrees of freedom.
>
> I recommend option 1. The Bayesian prior compensates for the short sample,
> and regime homogeneity ensures the estimated relationships are meaningful
> for current policy."

#### Step 4: Sensitivity check

Always report results for at least one alternative sample period:
- Excluding the pandemic (pre-2020).
- Extended sample (if regime break is debatable).
- Sub-samples around known breaks.

## Iterative revision after failed sensibility checks

When the Stage 5 sensibility check fails, route by the classes in the
orchestrator's
[Stage 5](../orchestrator/SKILL.md#stage-5-economic-sensibility-check). The
remedies below (adding variables, reordering, re-weighting, tightening
priors) are class-3 changes: propose them, wait for confirmation, and run an
accepted revision as a labelled sensitivity exercise alongside the kept
baseline. The specification skill should:

1. **Diagnose which specification choice is likely responsible.**
   - Price puzzle → consider adding commodity prices, reordering Cholesky,
     tightening priors on price response.
   - Explosive responses → reduce lag order, tighten overall prior. Also
     check for **outlier leverage**: a handful of extreme observations
     (2020 pandemic quarters, 9/11 reserves, 2007-08 money-market spikes)
     can drive a small-sample system explosive on their own. Date the
     largest residuals before blaming the lag order. Remedies that keep
     the sample: Lenza-Primiceri observation weighting
     (`fit_minnesota_bvar(..., obs_scale=...)`, see the
     `lenza_primiceri_2022_covid_var` entry; symmetric prior path only,
     so it cannot be combined with `exogenous_block` or an asymmetric
     prior) or shrinkage; document whichever is used as a revision.
   - Implausible magnitudes → check variable units and transformations.
   - Non-convergence → simplify the model (fewer variables, fewer lags).

2. **Propose a specific revision**, not a vague suggestion.
   - Bad: "Consider changing the specification."
   - Good: "Adding a commodity price index often resolves the price puzzle
     (Sims 1992). I propose adding the World Bank commodity price index as
     the 5th variable, ordered first in the Cholesky decomposition."

3. **Estimate the accepted revision and present the comparison.**
   - Show old vs new impulse responses side by side.
   - Note what changed and why.
   - Log both versions in the research log.

4. **Respect the iteration limit.** The `max_auto_iterations` budget (from
   config.yaml, default 5) covers only the automatic classes
   (implementation/data fixes, pre-specified statistical rules). When it is
   exhausted, or when every remaining remedy is a gated economic change,
   stop and present all attempts to the user with a summary of what was
   tried and what failed. The human decides next steps.

## What is NOT data-driven

Some choices should never be made by optimising a statistical criterion:

1. **Variable selection.** Never add or remove variables because they improve fit.
   Variables are in the model because economic theory says they belong.
2. **Identification.** Never choose between Cholesky and sign restrictions based
   on which gives "better" results. Identification is a theoretical commitment.
   **Distinguish this from a legitimate identification revision.** When a
   Stage-5 anomaly *points to a failure of the identifying assumption
   itself*, switching schemes can be the remedy the diagnosis calls for,
   not mining. A price puzzle that survives the standard remedies may
   indicate that the recursive timing assumption mislabels the policy
   shock. A first-stage F below 10 signals a weak instrument: standard
   inference is unreliable and weak-instrument-robust inference is
   needed. Neither proves that identification has failed. Three
   conditions make a switch legitimate, and all
   three go in the research log: (i) the trigger is a *diagnosed
   assumption failure*, stated in economic terms, not "the new scheme
   gives nicer results"; (ii) the new scheme is chosen for its
   assumptions (and leaves the contested response unrestricted where
   possible, e.g. Uhlig-style restrictions keep output agnostic); and
   (iii) the report discloses the full revision path, shows the
   first-pass results alongside the final ones, and notes that bands are
   conditional on the search.
3. **Shock labelling.** Never relabel a shock because the responses don't match
   expectations. If the shock produces unexpected signs, check data,
   specification and identification. Do not relabel the shock. An
   unexpected sign that survives these checks is reported as a finding.
4. **Sensibility thresholds.** Never loosen the definition of "economically
   sensible" to make results pass. If results are implausible, diagnose the
   cause under the Stage-5 routing and report it.

## Output

The specification skill returns:

```yaml
specification:
  variables:
    endogenous: [{name, source, transform, rationale, knowledge_source}]
    exogenous: [{name, source, transform, rationale, knowledge_source}]
  lag_order:
    selected: integer
    method: "user" | "bic" | "aic" | "literature" | "residual_diagnostics"
    rationale: string
    alternatives_considered: [{order, criterion_value, notes}]
  prior: (if Bayesian)
    type: string
    hyperparameters: dict
    method: "literature_default" | "knowledge_base" | "grid_search" | "marginal_likelihood" | "user"
    rationale: string
    grid_search_results: dict or null
  identification:
    method: string
    details: dict (ordering, restrictions, instrument, etc.)
    rationale: string
    knowledge_source: "user" | "dictionary" | "knowledge_base" | "parametric" | "data_driven"
  sample:
    start: date
    end: date
    rationale: string
    regime_notes: string
    robustness_alternatives: [{start, end, reason}]
  feasibility:
    total_parameters_per_equation: integer
    observations: integer
    ratio: float
    adequate: bool
  decision_sources:
    summary: "X of Y decisions from the dictionary, Z from the knowledge base, W from the user"
```

Each substantive decision carries its rationale and knowledge source, for
transparency and research-log compatibility. This is the decision record, not
the run configuration: the estimation script reads its own `config.yaml`
(through `lib.Spec`).

## References

- Sims, C.A. (1980). Macroeconomics and Reality. Econometrica, 48(1), 1-48.
- Christiano, L.J., Eichenbaum, M., Evans, C.L. (1999). Monetary Policy Shocks:
  What Have We Learned and to What End? Handbook of Macroeconomics, Vol. 1A, 65-148.
- Giannone, D., Lenza, M., Primiceri, G.E. (2015). Prior Selection for Vector
  Autoregressions. Review of Economics and Statistics, 97(2), 436-451.
- Litterman, R.B. (1986). Forecasting with Bayesian Vector Autoregressions.
  Journal of Business & Economic Statistics, 4(1), 25-38.
- Sims, C.A. (1992). Interpreting the Macroeconomic Time Series Facts: The Effects
  of Monetary Policy. European Economic Review, 36(5), 975-1000.
