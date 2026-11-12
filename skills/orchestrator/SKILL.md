---
name: econometric-orchestrator
description: >
  Orchestrates iterative econometric analysis using an agentic, human-in-the-loop workflow.
  Use this skill whenever the user wants to run a VAR, BVAR, or structural econometric model,
  asks for macroeconomic time-series analysis, has a research question and needs help
  choosing the right model, or submits a paper to replicate. Also triggers when the user says
  "run an econometric analysis", "estimate a VAR", "replicate Sims", "impulse responses",
  "what model should I use", "replicate this paper", or any request involving macro modelling
  with iterative refinement. Handles three entry modes: dictionary-driven (deterministic
  replication of a verified spec), paper submission (parse a new paper into a draft entry,
  then replicate), and research design (a spec designed from a research question using the model's
  parametric knowledge). This is the entry point that coordinates all sub-skills (model selection,
  data retrieval, estimation via scripts/lib/, diagnostics, research log).
---

# Econometric analysis orchestrator

This skill manages an iterative, human-in-the-loop agentic loop for econometric analysis.
It coordinates sub-skills and checks each step for economic sensibility
before proceeding.

## Design principles

1. **Everything is editable, always.** Any decision at any stage can be revised at
   any time. The framework never locks the user into a path. Revisions propagate
   forward automatically, and the research log tracks every version.
2. **Specific knowledge overrides parametric.** Every decision draws on specific
   sources before generic ones: in priority order, the user, the model
   dictionary, then the knowledge base (local context: economy profile, data
   sources, modelling conventions). Fall back to parametric knowledge only for
   gaps. The full cascade is defined once in the `specification` skill; always
   make the source of each decision transparent.
3. **Active suggestions, human confirmation.** At every choice point (model
   class, variables, lags, priors, identification, sample period, sensibility
   remedies, mode triage, data substitutions), propose a specific recommendation
   with rationale drawn from (in priority order) the dictionary, knowledge
   base, and parametric / economic-theory knowledge. The user confirms,
   overrides, or asks for alternatives. Never proceed silently; never ask a
   blank "what do you want?" question without offering a proposal first. The
   agent is a dialogue partner, not a form to fill out.
4. **Economic sensibility first.** Every intermediate output is evaluated against economic
   theory before being accepted. Flag anomalies (price puzzles, explosive roots, implausible
   magnitudes) and propose remedies.
5. **Iterative refinement.** If results fail sensibility checks, revise and
   re-estimate, but automatically only within Stage 5's class rules
   (implementation/data fixes and pre-specified statistical rules);
   economically substantive changes are gated proposals, run as labelled
   sensitivity exercises alongside the kept baseline.
6. **Tech-stack agnostic, package-first.** Generate code in whichever language
   the user prefers (Python, R, Julia, MATLAB). Always use the best available
   library or package for the task; do not reimplement standard algorithms
   from scratch. Only write custom code when no reliable package exists for
   the specific method needed.
7. **Fully logged.** Every decision, iteration, and rationale is recorded via the
   research-log sub-skill for reproducibility.
8. **One file of recommended defaults.** The starting proposals for tuneable
   settings (prior hyperparameters, draw counts, chart styles, data source
   preferences) come from `config.yaml` at the project root. The library does
   not load that file: write the values a run uses into the run's own
   `config.yaml` (and pass them as explicit arguments), because that file and
   those arguments are what govern execution. Editing the root file changes
   future proposals, not existing scripts; canonical replications keep their
   declared settings unless a deviation is approved and logged. Fields marked
   `[lib constant]` there are informational. Override order: config.yaml <
   knowledge base < runtime instruction (user always wins).
   (The one intentional exception: API key variable names / endpoints / rate limits live in
   `skills/data-retrieval/connections/connections.yaml`, because they are
   credentials and connection plumbing, not behaviour.)

## Three modes of entry

A session can begin in one of three modes. The orchestrator's first job (Stage 0b
below) is to identify which mode the request is in and route accordingly:

| Mode | When | What the orchestrator does |
|---|---|---|
| **A: Dictionary execution** | User names a paper / method with an entry in [`model-dictionary/entries/`](../model-dictionary/entries/); provides no novel content | Look up entry, run or generate `replicate.py`, skip spec design |
| **B: Paper submission** | User provides a paper (PDF / URL / citation) with no existing entry | Parse paper → draft entry (`verified.status: draft`) → user verifies → fall through to Mode A |
| **C: Research design** | User asks an open question with no named method | Run `model-selection`; if dictionary match → upgrade to A; else design from parametric knowledge |

If the request is ambiguous (e.g., it names a paper AND asks an open
question), ask before proceeding. Do not guess the mode silently. A wrong mode
choice wastes the user's time on the wrong work.

All three modes converge on the same execution pipeline (stages 2-7, with Stage 2 skipped in a deviation-free Mode A); only the
path *into* the specification differs. See [`modes.md`](modes.md) for full path
descriptions, transition examples, and a deeper reference.

### Source tiers

Mode A's lookup searches two source tiers (the scratch folder is listed for completeness but is not a source):

| Tier | Folder | Role | Gitignored? |
|---|---|---|---|
| **canonical** | [`scripts/replications/<id>/`](../../scripts/replications/) | Canonical replications, checked against the paper; public-good gallery | Scaffolding committed; `outputs/` ignored |
| **application** | [`scripts/applications/<id>/`](../../scripts/applications/) | User-specific past analyses (variants, adaptations, exploratory) | Scaffolding committed; `outputs/` ignored |
| **scratch** | `outputs/` (repo root) | Transient files from the current run | Always ignored; disposable |

Mode A ranks canonical above application but surfaces both. Successful Mode B
and Mode C runs get promoted to one of these tiers: applications by default;
canonical only after rigorous verification (`verified.status:
checked_against_paper` or higher in the dictionary entry).

`outputs/` at the repo root is **not** a source for the orchestrator. It is a
scratch space. If it disappears, no script is lost, because the script lives
in the relevant `scripts/<tier>/<id>/` folder. Generated model outputs are
regenerable, but session records (`session_research_log.json`,
`research_log.md`, `run_state.json`, any saved plan) are not: copy them out
before deleting any `outputs/` folder.

### Mode transitions

Modes are not one-way. A session can move between them mid-flow: **C → A**
(model-selection finds a clean dictionary match), **B → C** (paper extraction
fails or method is too novel), **A → C** (user deviates too far from the entry),
**C → B** (user produces a paper mid-design). Always announce the transition
explicitly and log it. See [`modes.md`](modes.md#mode-transitions-extended) for
worked examples.

## Revision protocol

Research is non-linear. The user will want to go back, change things, and see what
happens. The framework must make this effortless, not punishing.

### Core rule: any stage, any time

The user can revise any previous decision at any point in the workflow:

- "Actually, let's add the exchange rate to the model" → revises stage 2, re-runs 3-6.
- "Try Minnesota prior with tighter lambda" → revises stage 4 only, re-runs 4-6.
- "Can we see the same results as a policy brief instead?" → revises stage 6 only.
- "Let's switch to sign restrictions instead of Cholesky" → revises stages 2 and 4-6.
- "I want to start the sample from 2000 instead" → revises stage 3, re-runs 3-6.
- "Let me update the economy profile" → revises stage 0a, may affect all subsequent.

### How revisions work

1. **Identify what changed.** Parse the user's request to determine which stage(s)
   are affected.

2. **Determine the propagation scope.** Changes cascade forward but never backward:

   | Changed stage | Must re-run |
   |---------------|-------------|
   | 0a (knowledge base) | Potentially all; check which downstream decisions relied on the changed context |
   | 1 (model selection) | 2, 3, 4, 5, 6 |
   | 2 (specification) | 3 (if variables changed), 4, 5, 6 |
   | 3 (data) | 4, 5, 6 |
   | 4 (estimation) | 5, 6 |
   | 5 (sensibility) | 6 (if re-estimation triggered) |
   | 6 (presentation) | 6 only (re-render) |

3. **Preserve what hasn't changed.** If the user only changes the prior, don't
   re-fetch the data. If they only change the audience format, don't re-estimate.
   Re-use cached results from unchanged stages.

4. **Create a new version.** Every revision adds a version to the research
   log's `version_history` (schema in
   [`research-log/SKILL.md`](../research-log/SKILL.md)). Previous versions are
   never deleted, so the user can compare versions at any time.

5. **Present the diff.** When re-running after a revision, show the user what
   changed relative to the previous version:

   > "**Revision v1 → v2:** Added nominal effective exchange rate to the VAR.
   > Lag order unchanged (4). Cholesky ordering updated: exchange rate placed
   > after the policy rate.
   >
   > **Impact on results:** The price puzzle is resolved: prices now decline
   > after a monetary tightening. Output response is slightly smaller (-0.25%
   > vs -0.31% at peak). Exchange rate appreciates on impact, consistent with
   > UIP."

### What is editable

Everything: the knowledge base (stage 0a), the model class (stage 1), the
specification (variables, ordering, lags, identification, prior, sample), the
data (source, transforms, frequency, a user CSV), estimation settings (draws,
burn-in, convergence criteria, library or language), the sensibility checks
and their thresholds (for example, keeping a price puzzle the user justifies
on theoretical grounds), and every part of the presentation (audience, style,
language, narrative, components, format).

### Parallel specifications

The user may want to run multiple specifications simultaneously and compare:

- "Run this with 4 lags and 8 lags and show me both."
- "Estimate with Minnesota and Sims-Zha priors."
- "Give me BVAR and LP results side by side."

Keep each specification as a parallel branch in the research log (schema in
[`research-log/SKILL.md`](../research-log/SKILL.md)). `VARReport` can add a
baseline-versus-alternative comparison chart.

### Revision of skills themselves

The SKILL.md files and the root `config.yaml` are plain text. The user can
edit them, or ask the agent to, for example: "Change the default overall prior
tightness (lambda_1) in `config.yaml` to 0.2." Changes apply from the next
invocation. Adding methods to `scripts/lib/` is developer work (see
[DEVELOPMENT.md](../../DEVELOPMENT.md#how-to-add-a-new-estimation-method)).
The research log records the script path and a content hash for each
analysis.

### Graceful handling of mid-workflow changes

When the user interrupts the workflow to make a change:

1. **Acknowledge the change immediately.** Don't insist on completing the current
   stage first.
2. **Save current progress.** Whatever has been computed so far is preserved.
3. **Identify the minimum re-work.** Only re-run what's affected by the change.
4. **Confirm the plan.** Tell the user: "You've changed X. This means I need to
   re-run stages Y and Z. Stages A and B are unaffected. Shall I proceed?"
5. **Execute and present the diff.** Show what changed in the results.

## Workflow stages

### Run-completion contract

A run can go quiet before Stages 6-7 (report, disposition) when the user
pauses, pivots to another topic, or the context is compacted. Three
mechanisms keep every run visibly open until Stage 7 closes it:

1. **Durable run state.** From Stage 1 onward, maintain `run_state.json`
   next to the run's research log. Its schema and lifecycle are owned by the
   research-log skill
   ([`research-log/SKILL.md`](../research-log/SKILL.md#run-state-file-run_statejson)).
   Update it at every stage gate, in the same breath as the stage's log
   entry. It is the anchor that survives context compaction: on any resumed
   or compacted session, read it before acting.

2. **Harness task tracking.** At Stage 1 (once the mode is confirmed), if
   the harness offers task-tracking tools, create one task per remaining stage
   ("Stage 2: specification", … "Stage 7: disposition") and update each to
   in-progress / completed as the run crosses its gates. Pending stages then
   stay visible to agent and user independently of the conversation text.
   Without task tools, `run_state.json` alone carries the state.

3. **Definitions of done.** A stage is not complete until its outputs
   exist *and* its research-log entry is appended. Reporting is part of the
   work, not an epilogue:

   | Stage | Not done until |
   |---|---|
   | 1-3 | user confirmed the decision + log entry appended |
   | 4 | outputs written to the co-located `outputs/` + diagnostics run + log entry |
   | 5 | sensibility verdict presented (pass / flagged anomalies) + log entry |
   | 6 | `report.html` + narrative delivered + log entry |
   | 7 | disposition + tool review logged; `run_state.json` closed |

**Before concluding any reply that ends a piece of work** (results delivered,
question answered, error resolved): check the run state. If a run is `open`
at a stage below 7, either proceed to the next stage or state explicitly what
remains and propose when to do it. An explicit, logged deferral is fine;
silence is not.

### Stage 0a: Knowledge base initialisation (one-time)

Before any analysis, check whether a knowledge base exists for this context.

1. If `skills/knowledge-base/context/` is populated, load it and summarise the
   key parameters to the user for confirmation.
2. If empty (first use), ask the user to describe their economy and institutional
   context. Generate draft templates and present for review. This only happens once.
   **Fast path:** when the request is a standard-US / pre-collected-panel
   analysis (a canonical replication, or a Mode C question the dictionary
   already covers for the US), propose *skipping* the interview ("no local
   knowledge base needed for this run; the dictionary supplies the US
   context") and proceed once the user agrees. Do not force the onboarding
   interview on a context-free run. **Non-US one-offs:** the same fast path
   applies: propose proceeding with the dictionary anchor plus parametric
   local knowledge, *clearly labelled as parametric* in every source
   attribution, and offer to set up an economy profile only if the user
   expects recurring work on that economy.
3. If the user provides papers or references, process them into the knowledge base
   via the knowledge-base skill.

All subsequent stages consult the knowledge base before falling back to parametric
knowledge. The research log records which knowledge source informed each decision.

### Stage 0b: Intake & triage (per request)

Classify every request into Mode A, B, or C per the
[Three modes of entry](#three-modes-of-entry) section above. The triage is a
one-paragraph reasoning step:

1. **Scan the request** for the signals in the mode table.
2. **State the mode** to the user before proceeding: *"This looks like Mode A:
   you've named CEE (1999), which is in the dictionary."*
3. **If the signals are mixed or ambiguous, ask.** Do not silently pick a mode.

**Log the triage decision** in the research log, including the reasoning and any
dictionary entry that was matched. If the mode changes mid-session (see
[Mode transitions](#mode-transitions)), log that too.

### Stage 1: Mode-specific entry flow

The work in this stage depends on the mode determined in Stage 0b. Each path
produces the same output for downstream stages: a confirmed specification.

#### Stage 1A: Dictionary execution

The dictionary entry already names the model class, variables, identification,
and sample. The orchestrator's job here is:

1. Read the entry from
   [`skills/model-dictionary/entries/<id>.yaml`](../model-dictionary/entries/).
2. If [`scripts/replications/<id>/`](../../scripts/replications/) already exists,
   summarise its `config.yaml` and `README.md` to the user. Confirm any spec
   overrides the user wants (different sample, different data substitutions).
3. If no replication folder exists, sketch the equivalent: which
   [`scripts/lib/`](../../scripts/lib/) functions will be called (e.g.,
   `fit_var` → `cholesky_irf` → `VARReport`), what data is needed, what
   substitutions the FRED-MD/QD panels can supply.

**Wait for user confirmation** of the spec and any deviations before proceeding
to data collection. Then jump to Stage 2 only if the user requested deviations;
otherwise jump straight to Stage 3.

#### Stage 1B: Paper submission

The orchestrator must convert the paper into a verified dictionary entry before
estimation.

0. **Duplicate check first.** Identify the paper (open the file far enough
   to read the title page; never trust the filename) and search the
   dictionary for it: citation fields, `verified.pdf_location`, and ids.
   If a verified entry already exists, this is **Mode A, not B**: say so
   and route there instead of drafting a duplicate.
1. **Verify the staged file**, then extract the paper's spec. Integrity
   first: the file opens, the page count is plausible, the title page
   matches the user's citation. Report what you found before parsing.
   Use the `pdf` skill for PDFs, web fetch for URLs, or work directly
   from the user-provided text. **Image-only scans** (no text layer,
   common for pre-2000 JSTOR PDFs): render pages to PNG with a PDF renderer
   available in your environment (about 150 dpi for prose; re-render the tables, footnotes
   and data appendix that the specification depends on at ~300 dpi) and read the images;
   mind the scan-page vs journal-page offset when citing pages; flag any
   value you cannot read with confidence rather than guessing, and offer
   the user an OCR re-staging alternative.
2. Delegate to the `model-dictionary` skill to draft an entry following its
   [schema](../model-dictionary/SKILL.md#entry-schema). Save as a YAML under
   [`entries/<id>.yaml`](../model-dictionary/entries/) with an
   `AUTO-GENERATED: please verify` note. Verification status
   (`draft | checked_against_paper | signed_off`): an entry drafted
   **field-by-field from the staged PDF with a page reference on every
   field that defines the specification** is `checked_against_paper`, because drafting and
   checking coincide. An entry drafted from memory, an abstract, or
   secondary sources is `draft` until checked against the paper; see
   [DEVELOPMENT.md](../../DEVELOPMENT.md).
3. Present the draft to the user. Highlight: choice of identification, sample
   bounds, variable proxies, prior hyperparameters, any judgment calls the paper
   made implicitly. Make ambiguities visible; do not paper over them.
4. If the method is not implemented in [`scripts/lib/`](../../scripts/lib/),
   invoke `advisor-protocol` (if enabled) to derive the algorithm. Save the verified
   derivation as a reference document so it can become a new `scripts/lib/`
   function later.
5. After the user verifies the entry, **fall through to Stage 1A** with that
   entry as the spec source.

**Wait for user verification** of the draft entry before proceeding.

#### Stage 1C: Research design

Delegate to the `model-selection` sub-skill.

1. Understand the research question and analytical objective.
2. Assess data context (number of variables, sample length, frequency).
3. **Search the dictionary** for matching entries via the `model-dictionary`
   skill. If one or more strong matches exist, propose upgrading to Mode A
   with that entry, and announce the transition explicitly.
4. If no clean match exists, recommend 1-3 candidate model classes from
   parametric knowledge with rationale.
5. **Wait for user confirmation** before proceeding to specification design.

For a Sims (1980)-style request initiated as an open research question ("how
does MP affect output?"), this stage finds `sims_1980_var` and
`cee_1999_monetary_svar` in the dictionary and proposes upgrading to Mode A.

**Log this stage** via the research-log skill, including the mode and the
dictionary match (if any).

### Stage 2: Specification

**Mode applicability.** In Mode A, this stage is mostly a confirmation step:
the dictionary entry already determines everything. In Mode B, the dictionary
draft created in Stage 1B is the input; spec decisions become "verify or
override the draft." Mode C is where most spec design happens.

Delegate to the `specification` sub-skill, which owns the framework's
**canonical decision hierarchy** (defined once there; see
[`specification/SKILL.md`](../specification/SKILL.md#decision-hierarchy)):

> **user ▸ model dictionary ▸ knowledge base ▸ parametric knowledge ▸
> data-driven search**, with **economic logic** checked at every level.

Levels 2–4 *combine* rather than strictly cascade. The dictionary anchors,
the knowledge base adapts it to local context, and parametric reasoning fills
the gaps; the order is only the tie-breaker when two sources conflict on the same
field (in Mode A the dictionary anchor dominates for faithfulness; in Mode C
the knowledge base adapts it to the local economy). Data-driven search is
confined to hyperparameters / lag order, never variable selection or
identification.

The specification skill determines:
- **Variable selection:** theory-first, informed by knowledge base and literature.
  Never data-driven: variables are in the model because economics says they belong.
- **Lag order:** information criteria search (AIC, BIC, HQ) with economic logic
  override (e.g., residual autocorrelation, plausible transmission lags).
- **Prior hyperparameters:** literature defaults, optionally refined by grid search
  that screens only for stability and failed estimation. Economic checks (impact
  signs, price puzzle) are reported diagnostics: they trigger investigation and
  class-3 proposals (see Stage 5), not automatic rejection.
- **Identification:** purely theoretical choice, informed by the research question,
  knowledge base, and literature. Never selected by data fit.
- **Sample period:** determined by regime awareness (from knowledge base) and the
  trade-off between sample length and structural homogeneity.

Every decision includes its rationale and knowledge source (user, knowledge base,
literature, parametric, or data-driven) for full transparency.

In a dictionary run, read the specification from the entry (for example
`entries/sims_1980_var.yaml`) rather than recalling it, and summarise
variables, sample, deterministics, identification and lags.

Present the specification summary to the user and wait for confirmation before proceeding.

**Log this stage** via the research-log skill.

### Stage 3: Data collection

Delegate to the `data-retrieval` sub-skill:

1. Map each variable to a data source (default: FRED API).
2. **If the fetch will write new data to disk for the first time (a new dataset, a
   new source, or a manually pasted CSV), settle where it lives before fetching**,
   following data-retrieval's [runtime collection workflow](../data-retrieval/SKILL.md#runtime-collection-workflow)
   and [data-acquisition disposition](../data-retrieval/SKILL.md#data-acquisition-disposition)
   (scratch / cache / pre-collected / commit / discard). Skip when the storage
   policy is already agreed for this run or the dataset is already registered.
3. Retrieve the series programmatically.
4. Present a summary table: variable name, source series ID, frequency, sample coverage,
   units, seasonal adjustment status.
5. Propose transformations (log levels, first differences, seasonal adjustment) with
   economic rationale.
6. **Wait for user confirmation** of variable mapping and transformations.
7. Settle what happens to the fetcher recipe (ad-hoc / connector / lib fetcher) and
   any promotion of a cache to a pre-collected dataset. This is the data-layer
   analogue of Stage 7's code disposition; all these decisions get logged.

A canonical replication's variable mapping lives in its `config.yaml` (for
example `scripts/replications/sims_1980_var/config.yaml`). Some original
series have no exact FRED equivalent: identify the closest match, explain the
discrepancy, and let the user decide.

**Log this stage**, including both data-acquisition disposition choices (data tier
and fetcher tier) when steps 2 and 7 fire.

### Stage 4: Estimation

Estimation flows through [`scripts/lib/`](../../scripts/lib/README.md), the
Python library that wraps `statsmodels`, `scipy`, `numpy` and `plotly` and
adds the macroeconometric pieces those packages do not expose cleanly. Its
README is the authoritative module inventory.

There are no per-method skills: each method's conceptual documentation
lives in its `scripts/lib/` module docstring (and CONVENTIONS.md for the
pinned math conventions). The orchestrator generates a thin script that
loads config, calls library functions, and saves outputs.

1. **Generate a thin `replicate.py`-style script** (~150 lines) following the
   [conventions in `scripts/replications/`](../../scripts/replications/README.md).
   The script: reads its settings from its own folder's `config.yaml` (not the
   root file), loads data via
   `lib.load_dataset`, applies per-column transforms via `lib.transform`, calls
   `scripts/lib/` estimation functions, writes outputs to the script's
   **co-located** `outputs/` subfolder (`scripts/<tier>/<id>/outputs/`,
   excluded from version control; this is what every shipped replication does, and the repo-root `outputs/`
   is scratch only). Default working locations:
   - **Mode A**: use the existing script at
     [`scripts/replications/<id>/replicate.py`](../../scripts/replications/)
     (canonical) or [`scripts/applications/<id>/replicate.py`](../../scripts/applications/)
     (application). No new script needed unless the user requested deviations.
   - **Mode A with user deviations** (different sample, variables, economy):
     a variant is no longer canonical, so create a NEW folder at
     `scripts/applications/<descriptive_id>/` and leave the canonical
     replication untouched (same rule as Modes B/C below; see the source-tier
     table and `scripts/applications/README.md`).
   - **Modes B and C**: write the new script to
     `scripts/applications/<id>/replicate.py` as the working location. The
     **final tier** (stay in applications, promote to canonical, or be
     discarded) is decided in Stage 7 (Disposition & curation).
2. Estimate the model with the confirmed specification.
3. Report the outputs the method produces. For VAR-type fits these are
   coefficient summaries, posterior distributions (if Bayesian), lag length
   criteria (AIC, BIC, HQ) and stability (eigenvalues of the companion
   matrix); other methods report their own (e.g. filtered series, factor
   loadings).
4. Run the `diagnostics` sub-skill where it supports the fit; otherwise state
   which checks do not apply.

**Do not present impulse responses yet.** First confirm the model is well-specified.

**Log this stage**, including the exact lib functions called and their key
arguments (number of draws, lag order, identification scheme) so the run can
be reproduced from the log alone.

### Stage 5: Economic sensibility check

This is the critical gate. **Division of labour with Stage 4:** the
`diagnostics` sub-skill (Stage 4) owns *statistical* adequacy (residual
autocorrelation, ARCH, normality, companion-matrix stability). Stage 5
*consumes* those diagnostics (it does not recompute them) and adds the
*economic* judgement below. Evaluate:

1. **Stability.** Report the maximum root and classify the dynamics as stationary (below 1),
   unit-root (at 1 within sampling error) or explosive (above 1). Apply the
   declared model-class and replication criteria consistently, treat
   near-unit-root sampling uncertainty as a disclosed diagnostic, and do not
   certify stability from finite-horizon plots.
2. **Residual diagnostics.** No severe autocorrelation, heteroskedasticity broadly
   acceptable, normality a soft requirement.
3. **Sign and magnitude of key responses.** For a monetary policy shock (Sims example):
   - Does output decline? (expected)
   - Does the price level eventually decline? (if not, flag the "price puzzle")
   - Is the unemployment response consistent with Okun's law in magnitude?
   - **Convergence, conditioned on shock type.** Responses to *transitory*
     shocks should die out at reasonable horizons (24-40 quarters).
     Responses to *permanent* shocks (long-run-identified shocks such as
     Blanchard-Quah supply, or level responses in unit-root systems)
     must **flatten to a finite limit**, not vanish; demanding die-out
     there fails correct models by construction (the framework's own BQ
     replication is the canonical example).
4. **Forecast error variance decomposition.** Are the shares economically plausible?

If checks fail, diagnose the CAUSE first, then route by class:

1. **Implementation or data errors** (wrong transform, misaligned sample,
   mislabelled series, a coding slip): fix and **return to stage 4**
   automatically. These are corrections, not specification search.
2. **Pre-specified statistical rules** (restoring stability by the
   documented default, lag order by the stated criterion): apply and
   re-run automatically, logging the rule invoked. Classes 1-2 share a
   budget of **5 automatic iterations**; after that, present all
   attempts and ask the user for guidance.
3. **Economically motivated substantive changes** (variable set,
   ordering, sample period, prior settings: anything that would move
   results toward expected signs): **never automatic**. Explain which
   check failed and why it matters economically, propose the remedy as
   a gated choice (per the diagnostics skill's propose → confirm
   protocol), and run accepted alternatives as labelled sensitivity
   exercises ALONGSIDE the baseline. The first-pass result is kept and
   reported, never silently replaced.
4. **Unexpected but clean results** (signs contrary to priors with no
   error found): report and interpret; this may be the finding rather
   than a defect. Specifications are never chosen by their results
   (CLAUDE.md stop-and-ask).

If checks pass:
- Present results with interpretation.
- **Wait for user confirmation** before finalising.

#### Specification search: the cost of this gate, and its disclosure rules

Iterating the specification *in the direction of expected signs* until
checks pass is, by construction, **specification search** (pre-testing /
the garden of forking paths): the final result cannot falsify the priors
embedded in the checks, and conventional confidence or credible bands no
longer have their nominal coverage after such a search. The gate is still
worth having (silently shipping a price puzzle is worse), but its cost
must be visible, not just logged. Therefore:

1. **Always disclose the search in the final report.** The Stage 6
   narrative must state the number of sensibility-driven revisions
   (including zero: "the first specification passed") and summarise what
   changed at each step (the research log's version history is the
   source).
2. **Show the first-pass result alongside the final one** whenever any
   sensibility-driven revision occurred, typically the baseline IRF panel
   for the key shock, so the reader sees what the search changed.
3. **State that intervals are conditional on the search.** One sentence
   in the narrative: reported bands do not account for the specification
   revisions that preceded them.
4. The existing hard rules remain binding: never relabel shocks to fit
   the story, never loosen a threshold to pass a check (see the
   specification skill), and the accept-the-anomaly path (keeping a price
   puzzle *with a theoretical justification*) is always available and
   often the more appropriate disposition.

**Log this stage, including any failed iterations and the reasoning.**

### Stage 6: Results and reporting

Delegate to the `results-presentation` sub-skill:

1. **Generate an interactive HTML report** as the primary output using
   `scripts/lib/report.py` (`VARReport` class), saved as `report.html` in the
   run's co-located `outputs/` folder (`scripts/<tier>/<id>/outputs/`).
   Choose sections by results-presentation's method-appropriate section
   rules. Every report has the specification summary, the narrative
   interpretation and the research log. Mark unavailable or inapplicable
   diagnostics, structural quantities and inference explicitly; do not fill
   their place with made-up sections.
2. **PDF charts and LaTeX tables are optional, on request**, for journal
   submission or print workflows. The shipped replications produce the
   HTML report + JSON outputs only; do not treat PDF/LaTeX as a default
   deliverable.
3. Determine the target audience (journal, working paper, policy brief,
   presentation, internal), matching the `results-presentation` audience enum.
4. Consult the knowledge base for house style and formatting conventions.
5. Write a narrative interpretation calibrated to the audience.
6. Run the communication sensibility check: does the narrative accurately reflect
   the uncertainty in the results?
7. Save the session log (`session_research_log.json`) and its Markdown summary
   (`research_log.md`) to the output directory (see the research-log skill).

The estimation script (saved to `scripts/applications/<id>/` by default in
Stage 4) should include the HTML report generation as its final section. All
outputs go to the script's co-located `outputs/` subfolder
(`scripts/<tier>/<id>/outputs/`), which is gitignored: scripts are versioned,
and generated model outputs are regenerable. The session records written
there (`session_research_log.json`, `research_log.md`, `run_state.json`, any
saved plan) are not regenerable and must be copied out before `outputs/` is
deleted. (The repo-root `outputs/` is scratch for ad-hoc runs not yet tied to
a tier.)

Present everything to the user for final review, then proceed to Stage 7 to
confirm disposition.

### Stage 7: Disposition & curation

After Stage 6 produces results and reports, **explicitly ask the user how to
handle the code that was just produced**. This is the governance step that
keeps working files from accumulating without a final home.

This stage is **mandatory at the end of every session**, including failed runs.
Never leave disposition implicit. That is how `outputs/`, working scripts, and
half-finished applications pile up without a way to tell apart real research
from experiments.

**Observable triggers.** "End of session" is not observable, because no user ever
announces it. Propose disposition the moment any of these concrete signals
appears, instead of waiting for an ending that never gets declared:

| Signal | Behaviour |
|---|---|
| User accepts the Stage 6 results ("thanks", "looks good", "that's what I needed") with no revision request | Propose disposition now |
| User pivots to an unrelated topic while a run is open | Surface the open run and propose disposition before (or alongside) the new topic; a pivot must not silently orphan the run |
| User asks a wrap-up-flavoured question (where the files are, how to cite, "anything else?") | Fold the disposition proposal into the answer |
| The run stalls unresolvably (sensibility loop exhausted, data unavailable, user unresponsive to a gate) | Failed runs get dispositions too, usually **discard** or **continue** with a `next_steps` note |

If the user is still mid-flow (asked for one more revision), disposition
waits, but the run stays `open` in `run_state.json`, and the
[run-completion contract](#run-completion-contract) keeps the pending stages
visible until Stage 7 closes them.

#### Five options

1. **Archive only.** Keep outputs in the run's co-located `scripts/<tier>/<id>/outputs/`. No script promotion. For
   successful but one-off analyses you might revisit. The next session sees
   the outputs but no committed script; if you return, decide then.

2. **Save as application** (default for Modes B and C with passing checks).
   Keep / promote the run to
   [`scripts/applications/<id>/`](../../scripts/applications/) with:
   - `README.md`: paper / question, what was done, key findings, caveats
   - `config.yaml`: full spec choices (mirrors the dictionary entry if one exists)
   - `replicate.py`: the runnable script
   the script's `outputs/` stays gitignored. If a dictionary entry exists, link
   the application from the entry and leave its `verified.status` unchanged:
   approving an application does not check the entry against its paper.
   Record the approval in the session log. A new entry starts as `draft`;
   the model-dictionary skill defines when it moves up.

3. **Promote to canonical.** Move to
   [`scripts/replications/<id>/`](../../scripts/replications/) and create or
   update the corresponding dictionary entry (`verified.status: checked_against_paper` or higher).
   Reserved for paper-faithful, public-good work that has been rigorously
   verified, typically after explicit replication of a published paper with
   results matching to within reasonable tolerance. **Rare.**

4. **Continue iterating.** Leave everything in its current working location.
   For active multi-session development. Record a `next_steps` note in the
   research log so the next session knows where to pick up.

5. **Discard.** For failed runs (sensibility checks rejected unresolvably),
   exact duplicates of canonical work, or dead ends. First copy the session
   records (`session_research_log.json`, `research_log.md`, `run_state.json`,
   any saved plan) out of `scripts/<tier>/<id>/outputs/`, because they are
   not regenerable. Then delete that `outputs/` folder and any unsaved
   working script. For a **Mode A canonical run**, delete nothing (its
   generated outputs are regenerable, so leave them in place) and record the
   disposition in the log. For a plan-only session (e.g. a deferred fetch),
   quote or reference the plan in the log before deleting any scaffolding.

#### Decision support

Propose a specific option based on the session's mode and outcome:

| Session signal | Recommended disposition |
|---|---|
| Mode A canonical run; results match expected | **Archive only** (canonical already exists) |
| Mode B paper-submission run; sensibility checks passed | **Save as application** |
| Mode C research-design run; sensibility checks passed; novel spec | **Save as application** |
| Mode C run with results closely matching a published paper | **Promote to canonical** (rare) |
| Any mode; sensibility checks failed unresolvably | **Discard** or **continue iterating** |
| Any mode; user flagged the run as framework testing | **Discard** |

Always present the proposal with reasoning, e.g.:

> *"Sensibility checks passed; this is a Mode C run with a novel
> commodity-prices spec. I recommend **save as application** at
> `scripts/applications/eu_commodity_spillover/`. Alternatively:
> archive only / promote to canonical / continue / discard. Which?"*

The user confirms or picks a different option. Per design principle 3, never
ask blank "what do you want to do with this?"; always propose with rationale.

#### Logging

The disposition is recorded in the research log:

```yaml
disposition:
  decision: "save_as_application"   # archive | save_as_application | promote_canonical | continue | discard
  decided_at: "2026-05-19T17:42:00Z"
  destination: "scripts/applications/eu_commodity_spillover/"
  rationale: "Mode C novel spec; sensibility passed; reusable for EU spillover work"
  dictionary_entry_updated: false
  prior_runs: []                    # any superseded runs under the same <id>
```

**Log this stage.** Even "archive only" and "discard" decisions get an entry,
because future sessions need to know what happened to past work and why.

#### Tool review (sub-step)

The code disposition covers the script. Sessions also write inline helper
functions, and some are reusable. Review them so later sessions do not
rewrite or hand-copy the same logic.

After the code-disposition decision is logged, **automatically enumerate
helper functions** written in this session's script(s) and propose a
disposition for each. **Default is "keep ad-hoc"**; promotion to lib
should be the user's explicit opt-in.

##### Which functions count

Include any function defined in the session's
`replicate.py` / `scripts/applications/<id>/*.py` that:

- Is at least ~5 lines of substantive code (skip trivial lambdas / one-liners).
- Has at least one well-typed parameter (not `def main():` style entry points).
- Isn't already imported from `lib` or another framework module.

Skip:

- One-line wrappers around a lib function.
- Functions whose body trivially reduces to a single library call.
- Closures / decorators specific to this script's control flow.

##### Three options per function

1. **Keep ad-hoc.** The function stays inline; no further action.
   Default for functions whose signatures reference replication-specific
   names (e.g., variable names hardcoded from the paper), or whose logic
   is tightly coupled to this paper's data structure.

2. **Flag for promotion** (the "speculation queue"). The function name,
   signature, location, and a proposed lib home get appended to
   `skills/research-log/tools_to_promote.yaml` (see
   [`research-log/SKILL.md`](../research-log/SKILL.md#tool-promotion-queue)
   for the schema and consolidation rules). No immediate change to lib.
   The flag persists across sessions; when a future session writes a
   similar function, the agent surfaces the prior candidate and proposes
   consolidation. For functions that look generic but haven't yet
   appeared in a second caller.

3. **Promote to lib now.** Add the function to the appropriate existing
   `scripts/lib/` module (or create a new one if no home fits;
   a `lib/utils.py` can be created as the fallback for tools that don't
   fit a single method module). Requires: at minimum a smoke test, fit the
   framework's docstring conventions, and check the DAG hasn't been
   violated. Reserved for functions used more than once in *this*
   session, or where the user explicitly requests immediate promotion.
   Defers to option 2 when in doubt, because promotion incurs immediate
   maintenance debt.

##### Decision signals

For each candidate, surface to the user:

| Signal | Suggests |
|---|---|
| Function called > 1 time in this session | Promote now |
| Function signature references replication-specific names | Keep ad-hoc |
| Function matches a queued candidate from a prior session | Promote (consolidation) |
| Function is generic but hasn't been seen before | Flag for promotion |
| Function < 10 LOC and a trivial wrapper | Keep ad-hoc (under threshold) |
| Function obviously belongs in an existing lib module (e.g., dating, irf) | Flag, with `proposed_lib_home` filled |
| Session disposition is **discard** | Keep ad-hoc; don't queue candidates from a script being deleted; note the observation in the research log and re-flag if the pattern reappears |

Propose a specific decision for each candidate with reasoning, for example:
*"`cumulative_multiplier(beta_g, beta_gdp, h, y_g)`: 12 lines, used twice in
this session (expansion and recession). I recommend promote now, to
`scripts/lib/lp.py`."*

Per design principle 3, never auto-promote without confirmation. Even
"promote" recommendations wait for the user's yes before any lib edit
happens.

##### Logging

```yaml
tool_review:
  reviewed_at: "2026-06-02T16:35:00Z"
  candidates:
    - function: extract_cholesky_shock
      location: scripts/replications/jorda_2005_lp/replicate.py
      loc: 18
      reused_in_session: false
      replication_specific_names: false
      proposed_lib_home: scripts/lib/var.py
      decision: flag_for_promotion
      rationale: "Generic helper; second caller"
    - function: cumulative_multiplier
      location: scripts/replications/auerbach_gorodnichenko_2012/replicate.py
      loc: 12
      reused_in_session: true
      replication_specific_names: false
      proposed_lib_home: scripts/lib/lp.py
      decision: promote
      rationale: "Used 2x this session; generic LP multiplier formula"
```

##### Stop-and-ask additions

- Promote a tool to lib without at least a smoke test → **stop, write the test first**
- Promote a function whose signature has replication-specific names → **stop, generalise the API first**
- Skip the tool-review sub-step when closing a run → **stop, run it**
- Auto-promote without user confirmation → **stop, the default is "keep ad-hoc", not "promote"**

## Invoking sub-skills

The sub-skills below are the workflow + knowledge layer (orchestration,
dictionary lookup, knowledge base, decision protocols, output, research log).
Estimation methodology lives in
[`scripts/lib/`](../../scripts/lib/README.md), not in a per-method skill:
the orchestrator routes Stage 4 directly to the relevant `lib.<fn>` call.

Read the relevant SKILL.md before delegating:

| Task | Skill / library location | Used in mode(s) |
|---|---|---|
| Knowledge base | [`../knowledge-base/SKILL.md`](../knowledge-base/SKILL.md) | All |
| Model dictionary | [`../model-dictionary/SKILL.md`](../model-dictionary/SKILL.md) | All (lookup in A, draft in B, match in C) |
| Paper parsing (PDF) | `pdf` (plugin skill) | B |
| Model selection | [`../model-selection/SKILL.md`](../model-selection/SKILL.md) | C (and as a sanity check in A) |
| Specification | [`../specification/SKILL.md`](../specification/SKILL.md) | All |
| Data retrieval | [`../data-retrieval/SKILL.md`](../data-retrieval/SKILL.md) | All |
| Estimation (all methods) | [`../../scripts/lib/README.md`](../../scripts/lib/README.md) | All (called by the generated `replicate.py`) |
| Diagnostics | [`../diagnostics/SKILL.md`](../diagnostics/SKILL.md) | All (where the method supports it) |
| Results presentation | [`../results-presentation/SKILL.md`](../results-presentation/SKILL.md) | All (uses `lib.VARReport`) |
| Benchmarking | [`../benchmarking/SKILL.md`](../benchmarking/SKILL.md) | On request (forecasting / model comparison / paper prep); optional, not a stage gate |
| Research log | [`../research-log/SKILL.md`](../research-log/SKILL.md) | All |
| Advisor (complex maths) | [`../advisor-protocol/SKILL.md`](../advisor-protocol/SKILL.md) | B (novel methods); occasionally C |

(The consistency-audit skill is left out of this table: it is a
developer-session maintenance pass over the framework itself, not a stage of
the research loop; see its SKILL.md.)

### When to invoke the advisor

At any stage, if mathematical content cannot be translated into code with
confidence (for example, a novel prior or identification scheme in a
user-provided paper), read
[`advisor-protocol/SKILL.md`](../advisor-protocol/SKILL.md) and follow its
escalation procedure. The advisor is off unless `advisor.enabled` is true in
`config.yaml`; if it is off, tell the user and continue without it. Save the
verified output as a reference document so the derivation is not escalated
again.

## Extending the framework

To add methods, dictionary entries or model classes, see
[DEVELOPMENT.md](../../DEVELOPMENT.md#how-to-add-a-new-estimation-method).
New personal applications go under `scripts/applications/<id>/` through
Stage 7.

## Error handling

- If data retrieval fails (API down, series discontinued), inform the user and suggest
  alternatives.
- If estimation fails to converge, report diagnostics and suggest prior/specification
  changes.
- Never silently swallow errors. Every failure is logged and surfaced.
