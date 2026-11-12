---
name: model-dictionary
description: >
  A registry of econometric modelling approaches extracted from the
  literature, as structured specifications checked against the source papers,
  with verification provenance recorded per entry. Each entry documents the
  key specification decisions (variables, prior, identification, sample) in
  structured fields
  that the agent can search without reading or interpreting original papers.
  Use this skill before model selection and specification: the agent browses
  the dictionary to find the most relevant approach for the user's research
  question, then proposes it. Indexed lookup reduces repeated extraction
  from papers at runtime; entries remain subject to error and review.
  Triggers when any skill needs literature-based guidance, or when the user
  asks "what does the literature suggest", "which paper should I follow",
  "what specification has been used for this question", or provides a new
  paper to add to the dictionary.
---

# Model dictionary

## Purpose

The model dictionary is the framework's institutional memory of the
econometrics literature. Instead of the agent reading papers at runtime
and risking misinterpretation, each entry records the essential modelling
decisions from one paper in a structured format. Entries are drafted,
checked field by field against the source paper, and remain open to review.

**The dictionary is a starting point, not a constraint.** The agent's
parametric knowledge remains active at every step, suggesting adaptations,
identifying gaps in dictionary entries, proposing novel combinations,
reasoning about specifications that no single paper covers, and connecting
insights across entries.

## How dictionary and parametric knowledge work together

They are complementary, not sequential:

| Situation | Dictionary provides | Parametric knowledge provides |
|-----------|--------------------|-----------------------------|
| Canonical question (monetary policy transmission) | Verified specification from CEE (1999) | "Given your short sample, you might tighten the prior beyond what CEE used" |
| Novel question (climate risk + monetary policy) | No exact match, but related entries | Creative combination: "Start with a standard monetary VAR but add a climate risk variable, following the logic of commodity price controls in Sims (1992)" |
| Adapting to a different economy | The US specification as a template | "For your small open economy, the exchange rate channel dominates, so reorder the Cholesky and consider adding the foreign interest rate" |
| Prior selection | GLP (2015) optimal hyperparameters as a reference | "Your sample is half the length of GLP's, so you likely need tighter priors than their optimum suggests" |
| Unexpected results | Expected results for comparison | "The sign reversal at horizon 8 isn't in the dictionary's expected results, but it's consistent with the exchange rate overshooting literature. This might not be a specification error" |
| Sensibility check fails | Known remedies (add commodity prices for price puzzle) | "The standard remedy isn't working here. This could be due to your economy's commodity-exporter structure, where commodity prices are endogenous to domestic policy" |

The dictionary reduces the risk of the agent misremembering what Sims (1980)
actually specified. Parametric knowledge prevents the agent from blindly
applying Sims (1980) to a context where it doesn't fit.

## What the dictionary does NOT do

- It does not replace the agent's reasoning. The agent should always think
  about whether a dictionary entry is appropriate, not just look up and apply.
- It does not limit the agent to documented approaches. If the user's question
  has no dictionary match, the agent uses parametric knowledge to propose a
  specification and flags it for the user's review.
- It does not prevent the agent from suggesting modifications to dictionary
  entries. "CEE (1999) uses 7 variables, but for your 80-observation sample
  I'd recommend dropping reserves and using 5 variables instead" is exactly
  the kind of reasoning the agent should provide.
- It does not make the framework deterministic. Two runs with different user
  contexts should produce different adaptations of the same dictionary entry.

## Architecture

```
model-dictionary/
├── SKILL.md                  (this file)
├── entries/                  one YAML file per modelling approach
│   ├── sims_1980_var.yaml
│   ├── litterman_1986_bvar.yaml
│   ├── cee_1999_monetary_svar.yaml
│   ├── glp_2015_prior_selection.yaml
│   ├── bgr_2010_large_bvar.yaml
│   ├── uhlig_2005_sign_restrictions.yaml
│   ├── jorda_2005_lp.yaml
│   ├── primiceri_2005_tvpvar.yaml
│   └── ...
└── index.yaml                searchable index (hand-maintained; kept in sync by tests/test_doc_consistency.py)
```

## Entry schema

Every dictionary entry follows the same structured format. Fields are
designed for search and matching, not for human reading pleasure. (Entries
may add optional sections beyond this core schema, e.g.
`implementation_notes` with a `replication:` pointer and data-alignment
notes, or method-specific blocks like `construction_methodology`; see
DEVELOPMENT.md §"Established conventions".)

```yaml
# --- Identification ---
id: "<author_year_topic>"           # unique, snake_case
citation:
  authors: "<authors>"
  year: <year>
  title: "<title>"
  journal: "<journal or book>"
  volume: "<volume>"
  pages: "<page range>"

# --- What question does this approach answer? ---
research_questions:
  - "<research question>"
tags:
  - <tag>

# --- Model specification ---
model:
  class: "<model class>"              # var | bvar | structural_var | sign_var | lp | tvp_var | stvar | panel_var | favar | dfm | mfvar | filter | shock_series
  prior: <null | prior name>          # null for classical; "minnesota", "sims_zha", etc.
  variables:
    endogenous:
      - name: "<variable name>"
        proxy: "<series used to measure it>"
        transform: "<log | level | diff>"
        role: "<role in the model>"
    exogenous: []
  lags: <lag order>
  frequency: "<monthly | quarterly | ...>"

# --- Identification ---
identification:
  # method is a CONTROLLED VOCABULARY (enforced by tests/test_doc_consistency.py):
  #   cholesky | long_run | sign_restrictions | narrative_sign_restrictions |
  #   sign_restrictions_plus_higher_moments | external_instrument |
  #   heteroskedasticity | non_gaussian_independent_components |
  #   generalised_variance_decomposition | none | varies
  # Nuance goes in a same-line comment or the rationale, not in the value.
  method: "<identification method>"
  ordering:                           # or the restrictions, as the method needs
    - "<variable name>"
  rationale: >
    <why the identification holds, with a paper page reference>

# --- Sample ---
sample:
  start: "<start period>"
  end: "<end period>"
  notes: "<paper page for the sample; any truncation as an explicit deviation>"

# --- Expected results (for sensibility checking) ---
expected_results:
  <shock name>:
    <variable>: "<expected response, with paper page>"
  notes: >
    <context for the expected results>

# --- When to recommend this approach ---
recommend_when:
  - "<condition>"
not_recommended_when:
  - "<condition>"
alternative_approaches:
  - id: "<other entry id>"
    reason: "<when to prefer it>"

# --- Verification status ---
verified:
  status: "<draft | checked_against_paper | signed_off>"   # see "Verification status" below
  date: "<YYYY-MM-DD>"
  against: "<what the entry was checked against>"
  pdf_location: "papers/<id>.pdf"    # local filename at verification time; PDFs are NOT distributed (see papers/README.md)
  notes: "<corrections made, with paper page>"
```

For a worked example, see [`entries/cee_1999_monetary_svar.yaml`](entries/cee_1999_monetary_svar.yaml).

### Field provenance: the paper's claims vs the framework's editorial content

An entry mixes two kinds of content, and the distinction must stay
visible, because it is what makes an entry defensible if the original author
reads it:

| Provenance | Fields | Standard |
|---|---|---|
| **Attributed to the paper** | `citation`, `model` (variables, lags, frequency, deterministics), `identification` (method, ordering/restrictions, rationale, `headline_finding`), `sample`, `expected_results` | Every claim the entry relies on carries a page / table / figure / equation reference to the source paper. Never attribute to the paper anything you cannot point to in it; quoted phrases must be near-verbatim. |
| **Framework editorial content** | `research_questions`, `tags`, `recommend_when`, `not_recommended_when`, `alternative_approaches`, `implementation_notes` | Our judgment about when and how to use the approach. NOT statements by, or endorsed by, the paper's authors; phrase accordingly. |

Data substitutions and sample truncations relative to the paper live in
the entry's `sample.notes` / `implementation_notes` and the paired
replication's README, always as explicit deviations, never silently
presented as the paper's own spec.

`scripts/tools/make_author_dossiers.py` renders each entry into an
author-facing dossier organised along exactly this split (what we
attribute to the paper, what we reproduce, what we changed, what is
ours). The working defensibility test: *would you send the dossier to
the paper's author?* The drift guard enforces the mechanical half
(page-reference presence on `checked_against_paper` entries).

## How the agent uses the dictionary

### Step 1: Parse the research question

Extract key terms from the user's request:
- Topic: "monetary policy", "fiscal policy", "exchange rate", etc.
- Objective: "transmission", "forecasting", "identification", etc.
- Context: "US", "small open economy", "emerging market", etc.

### Step 2: Search the dictionary

Match against the index's `questions` and `tags` fields to make a
shortlist. Then read the full entries on the shortlist (including
`recommend_when` and `not_recommended_when`) to judge fit: context,
recommendations and any implementation differences.

```
User: "How does monetary policy affect output and inflation?"

Ranked matches:
  1. cee_1999_monetary_svar       (exact match on research question)
  2. sims_1980_var                (foundational, same question)
  3. uhlig_2005_sign_restrictions (same question, different identification)
  4. primiceri_2005_tvpvar        (same question, time-varying)
```

### Step 3: Filter by context

Apply `not_recommended_when` filters based on the knowledge base:
- If the economy is open → filter out closed-economy-only entries.
- If the sample is short → filter out entries requiring long samples.
- If the user prefers sign restrictions → filter out Cholesky-only entries.

### Step 4: Present recommendations

Show the top 1-3 matches with their full specification, citing the
dictionary entry (not the original paper: the entry IS the verified
information source).

> "For your question about monetary policy transmission, the literature
> suggests three approaches:
>
> 1. **Christiano, Eichenbaum, Evans (1999):** 7-variable structural VAR
>    with recursive identification. Includes commodity prices to resolve
>    the price puzzle. [Verified against original paper]
>
> 2. **Sims (1980):** 6-variable VAR, the foundational approach. Simpler
>    but may exhibit the price puzzle.
>
> 3. **Uhlig (2005):** sign restrictions instead of Cholesky ordering.
>    More agnostic but produces wider confidence bands.
>
> Given your economy profile (US, long sample, interest in transmission),
> I recommend starting with CEE (1999). Shall I set up this specification?"

### Step 5: Adopt the specification

If the user confirms, use the entry as the reference for the paper's
specification. Entries describe the paper; a paired replication may use a
different estimator, sample or data. Read the replication's `config.yaml`
and `README.md`, note where it departs from the paper, and confirm the
executable specification through the orchestrator's Stage 1A. For a new or
adapted implementation, pass the entry and the open choices to the
specification skill, which adjusts for local context via the knowledge base
(e.g., local data sources, variable proxies).

## Adding new entries

### From a paper the user provides

1. The user uploads or references a paper.
2. The agent extracts the modelling choices into a draft entry.
3. The agent checks the draft field by field against the paper, adds page
   references, and sets `verified.status` to the level reached.
4. The user reviews the draft. On confirmation, the entry is saved to
   `entries/` and `index.yaml` is updated.

### From the agent's parametric knowledge

The agent can draft entries based on its training knowledge, but these
must be marked as `verified.status: "draft"` and then checked field-by-field
against the source PDF before being treated as authoritative (promoting
them to `checked_against_paper`; a sign-off promotes further to
`signed_off`).

```yaml
verified:
  status: "draft"                   # FLAGGED: not yet checked against the paper
  date: "2026-04-11"
  against: "parametric knowledge"
  notes: "AUTO-GENERATED: please verify against original paper"
```

**Verification status**:
`draft` means drafted but not yet checked against the paper.
`checked_against_paper` means checked field by field against the source
PDF, with page references. `signed_off` means the entry has also been
confirmed by the user. The model-selection and specification skills should
prefer `checked_against_paper` and `signed_off` entries over `draft`
entries. Shipped entries are `checked_against_paper`. Verification is best
effort, not a guarantee. Where an entry matters to your analysis, check the
substantive fields against the paper yourself and log suspected errors.

### From a completed analysis

A completed analysis can be recorded as a new entry in the user's own copy,
through the same draft-and-check steps above.

## Relationship to other skills

| Skill | How it uses the dictionary |
|-------|--------------------------|
| Model selection | Searches `research_questions` and `tags` to recommend model classes |
| Specification | Starts from the entry's `model`, `identification`, `sample` for a new or adapted specification |
| Sensibility check | Compares results against `expected_results` |
| Benchmarking | Uses dictionary entries as alternative specifications for comparison |
| Knowledge base | Dictionary entries override parametric knowledge for documented papers |
| Advisor protocol | Uses entries as references for review; when to escalate is set by the [advisor protocol](../advisor-protocol/SKILL.md) and its enabled setting |

## Quality over quantity

Prefer a few entries checked against their papers over many unchecked ones.
An entry is treated as authoritative only after it is checked against its
source paper.

## Index generation

The index is a flat file listing all entries with their searchable fields,
hand-maintained and drift-guarded by tests (`tests/test_doc_consistency.py`).
See [`index.yaml`](index.yaml) for its format.

The agent loads only the index for search, then reads the full entry only
for matched results. This keeps lookup fast even as the dictionary grows.
