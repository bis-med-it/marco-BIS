---
name: knowledge-base
description: >
  Manages the interplay between parametric knowledge (the agent's built-in econometric
  training) and specific knowledge (local economic structure, institutional data sources,
  country-specific conventions, user-provided literature). Every other skill in the
  framework consults the knowledge base before making decisions. Use this skill when
  setting up a new analysis context, when the user provides local information or papers,
  or when any skill needs to decide whether to apply general or context-specific logic.
  Also triggers when the user says "use our data", "in my country", "our central bank",
  or provides institutional context.
---

# Knowledge base

## The problem

The agent has two sources of knowledge, and they can conflict:

1. **Parametric knowledge.** What the model learned during training: textbook
   econometrics, canonical papers, standard practices. This is broad but generic,
   and skewed towards US/EU macro.

2. **Specific knowledge.** What the user provides: their country's economic
   structure, institutional data sources, policy regime, modelling conventions,
   relevant local literature. This is narrow but authoritative for the context.

Without explicit orchestration, the agent defaults to parametric knowledge and
may make choices that are technically correct but economically wrong for the
user's context. For example:

- Recommending FRED series when the user's country has its own statistical office.
- Assuming flexible exchange rates when the country has a peg.
- Setting a random-walk prior mean (`rw_deltas` entry of 1) for a variable that is
  stationary in the local economy.
- Choosing a Cholesky ordering based on US institutional timing when the user's
  central bank operates differently.

## Design principle: specific overrides parametric, parametric fills gaps

The full source order (user, dictionary, knowledge base, parametric,
data-driven) is defined in
[`specification/SKILL.md`](../specification/SKILL.md#decision-hierarchy).
Between the knowledge base and parametric knowledge:

1. **If the knowledge base has specific guidance, use it.**
2. **If not, fall back to parametric knowledge.**
3. **If uncertain, ask the user.**
4. **Always make the source of each decision transparent.**

## Knowledge base structure

The knowledge base is a set of documents the user populates, organised by domain.
Each document is a markdown file in the `knowledge-base/` directory.

```
knowledge-base/
├── SKILL.md                 (this file)
├── context/
│   ├── economy-profile.md   country/economy structural description
│   ├── data-sources.md      local data APIs, databases, file locations
│   ├── conventions.md       institutional modelling conventions
│   └── policy-regime.md     monetary/fiscal framework, exchange rate regime
├── literature/
│   ├── paper-summaries/     user-verified summaries of relevant papers
│   └── methodology-notes/   specific methodological guidance from papers
└── priors/
    ├── variable-priors.md   prior beliefs about variable behaviour
    └── identification.md    preferred identification strategies with rationale
```

(Only an empty `context/` folder ships with the repo; its files and the
`literature/` and `priors/` folders are created as the user adds
material.)

### economy-profile.md

The user describes their economy. This informs model selection, variable choice,
and sensibility checks. Template:

```markdown
# Economy profile

## Country/region
[e.g., Euro area, Chile]

## Economic structure
- Open/closed economy: [small open economy / large closed / ...]
- Exchange rate regime: [float / managed float / peg / currency board]
- Capital account: [open / partially open / closed]
- Dominant sectors: [manufacturing, commodities, services, tourism]
- Key trading partners: [...]

## Monetary policy framework
- Instrument: [policy rate / reserve requirements / exchange rate target / ...]
- Target: [inflation targeting / monetary aggregate / exchange rate / dual mandate]
- Transmission channels: [interest rate / credit / exchange rate / asset price]
- Policy meeting frequency: [monthly / 6-weekly / quarterly]
- Decision-to-implementation lag: [immediate / ...]

## Fiscal framework
- Fiscal rule: [yes/no, type]
- Government debt level: [approximate]
- Fiscal year: [calendar / April-March / October-September]

## Known structural features
- Dollarisation: [none / partial / full]
- Informal economy: [estimated share]
- Commodity dependence: [which commodities, share of exports/GDP]
- Seasonal patterns: [agriculture, tourism, fiscal cycles]

## Historical structural breaks
- [Date]: [event and economic significance]
- [Date]: [event]
```

### data-sources.md

Maps conceptual variables to local data sources. This overrides the data-retrieval
skill's default (FRED). Template:

```markdown
# Local data sources

## Primary statistical agency
- Name: [e.g., the national central bank, INE Chile, Eurostat]
- API: [endpoint if available]
- Access method: [API / bulk download / manual / internal database]

## Variable mapping

| Concept           | Local source    | Series ID  | Freq | Notes              |
|-------------------|-----------------|------------|------|--------------------|
| GDP (real)        | [agency]        | [id]       | Q    | [base year, SA]    |
| CPI               | [agency]        | [id]       | M    | [basket details]   |
| Policy rate       | [central bank]  | [id]       | D/M  | [which rate]       |
| Money supply      | [central bank]  | [id]       | M    | [M1/M2/M3]        |
| Exchange rate     | [central bank]  | [id]       | D    | [bilateral/NEER]   |
| Unemployment      | [agency]        | [id]       | M/Q  | [survey details]   |
| ...               | ...             | ...        | ...  | ...                |

## International data
- [Which international sources are also relevant: BIS, IMF IFS, etc.]

## Data quirks
- [e.g., GDP is revised heavily in the first release]
- [e.g., unemployment survey changed methodology in 2012]
- [e.g., CPI basket was rebased in 2019]
```

### conventions.md

Institutional modelling practices. Template:

```markdown
# Modelling conventions

## Preferred software/language
[e.g., MATLAB is standard in our research department / we use R / Python]

## Standard VAR specification
- Typical variable set: [what variables are commonly included]
- Typical lag order: [e.g., 4 for quarterly, 12 for monthly]
- Preferred identification: [e.g., Cholesky, slow variables before the policy rate, fast financial variables after]
- Standard sample period: [e.g., 2000Q1 onwards due to regime change]

## Prior conventions (if Bayesian)
- [e.g., we typically use Minnesota priors with lambda_1 = 0.2]
- [e.g., we treat the exchange rate as I(1) but CPI as trend-stationary]

## Reporting standards
- [e.g., always report 68% and 90% bands]
- [e.g., IRF horizon = 20 quarters]
- [e.g., charts must follow house style: specific colours, fonts]

## Peer review expectations
- [e.g., reviewers expect robustness to lag order and sample period]
- [e.g., sign restrictions are preferred over Cholesky in our department]
```

### policy-regime.md

Describes the policy framework in enough detail for the agent to make sensible
identification and sensibility-check decisions. Template:

```markdown
# Policy regime

## Current regime
- Framework: [e.g., flexible inflation targeting since 2000]
- Instrument: [e.g., 1-day bilateral repurchase rate]
- Target: [e.g., headline CPI inflation 1-3% with 2.5% midpoint]
- Communication: [e.g., forward guidance via policy statement]

## Historical regimes
- [Date range]: [regime description]
- [Date range]: [regime description]

## Implications for modelling
- [e.g., pre-2000 data should not be pooled with post-2000 due to regime change]
- [e.g., the exchange rate was effectively fixed before 1997, so models including
  the exchange rate should start after the float]
- [e.g., unconventional policy measures since 2020 may affect interest rate
  transmission]
```

### literature/ directory

User-provided paper summaries and methodology notes. These are verified by the
user before the agent treats them as authoritative.

Each paper summary should include:
- Citation
- Key contribution (1-2 sentences)
- Model/method used
- Key findings relevant to the user's context
- Any methodological choices the agent should adopt

## How skills consult the knowledge base

Every skill follows this lookup protocol:

### 1. Check for specific guidance first

Before making any decision (variable choice, prior setting, identification,
data source, sensibility threshold), the skill checks the knowledge base:

```
decision_needed: "Which data source for GDP?"

1. Read knowledge-base/context/data-sources.md
   → Found: "GDP (real) from the national statistical office, series <series id>"
   → Use this. Do not default to FRED.

2. If not found in data-sources.md:
   → Fall back to parametric knowledge (FRED, standard series)
   → Inform user: "No local GDP source specified in your knowledge base.
     Defaulting to [FRED/IMF]. You can add your local source to
     knowledge-base/context/data-sources.md."
```

### 2. Check for conflicts

Sometimes parametric knowledge and specific knowledge point in different
directions. Examples:

| Decision | Parametric says | Specific says | Resolution |
|----------|----------------|---------------|------------|
| Variable order | Slow variables, then the policy instrument, then fast variables (e.g. CEE 1999 timing) | Policy rate last (our CB meets after data release) | **Specific wins** |
| GDP prior | Random walk (`rw_deltas` = 1) | Trend-stationary (our GDP is mean-reverting) | **Specific wins** |
| Sample start | 1960Q1 (maximise observations) | 2000Q1 (regime change) | **Specific wins** |
| IRF sign check | Price puzzle is a red flag | Price puzzle expected due to exchange rate pass-through | **Specific wins, but flag for discussion** |
| Lag order | 4 lags for quarterly data | No guidance | **Parametric fills the gap** |

### 3. Make the source transparent

In every decision presented to the user, note whether it came from specific
or parametric knowledge:

> "I'm using the national statistical office's real GDP series [from your data sources].
> For the import price index, I'm defaulting to IMF IFS series
> [parametric; no local source specified]. Would you like to add a local source?"

This transparency lets the user catch errors and progressively fill gaps in
the knowledge base.

### 4. Escalate when uncertain

If the agent encounters a situation where:
- Parametric and specific knowledge clearly conflict, and
- The specific guidance is ambiguous or incomplete

Then ask the user rather than guessing. Frame it as:

> "Your economy profile notes a managed float exchange rate regime, but the
> identification strategy in your conventions assumes a freely floating rate.
> Which should I follow for this analysis?"

## Populating the knowledge base

### Initial setup

The orchestrator's Stage 0a runs the setup, including the fast path that
skips the interview when a run needs no local context (see
[`orchestrator/SKILL.md`](../orchestrator/SKILL.md#stage-0a-knowledge-base-initialisation-one-time)).
Draft each document from the templates above, present it for review, and save
the confirmed version. The knowledge base persists across sessions.

### Incremental updates

As the user works with the framework:

- New papers can be added to `literature/` at any time.
- The agent can suggest updates: "Your analysis used a data source not in your
  knowledge base. Would you like me to add it?"
- After each completed analysis, offer to update conventions based on choices
  made during the session.

### From uploaded papers

When the user provides a paper (PDF or reference):

1. The agent extracts methodological choices relevant to the knowledge base.
2. Presents a summary for user verification.
3. On confirmation, saves it to `literature/paper-summaries/` and updates
   relevant context files if the paper contains local economic information.

This is distinct from the advisor protocol: the advisor handles mathematical
derivation; the knowledge base handles contextual and methodological choices.

## Integration with other skills

| Skill | What it looks up | Knowledge base source |
|-------|-----------------|----------------------|
| Model selection | Economic structure, regime, conventions | economy-profile, policy-regime, conventions |
| Data retrieval | Local data sources, variable mapping | data-sources |
| Specification: BVAR priors | Prior beliefs, variable treatment (I(1) vs I(0)) | variable-priors, conventions |
| Specification: block exogeneity | Foreign block to treat as exogenous (small open economy) | economy-profile |
| Specification: sign restrictions | Expected shock signs in this economy | identification, literature |
| Specification: TVP-VAR | Known structural break dates | economy-profile, policy-regime |
| Specification: local projections | Instruments relevant to this economy | literature, identification |
| Diagnostics | Sensibility benchmarks for this economy | economy-profile, conventions |
| Advisor protocol | Papers to reference for derivations | literature/methodology-notes |
| Research log | Document which knowledge source informed each decision | all |

## Example: a Sims (1980)-style VAR for a small open economy

Without a knowledge base, the agent uses FRED data, the paper's Cholesky
ordering and US macro intuition for the sensibility checks. With one, the same
workflow uses local data (data-sources.md), adds the exchange rate and an
exogenous foreign block (economy-profile.md; see the specification skill's
block-exogeneity step), reorders the Cholesky to local policy timing
(conventions.md), treats the exchange rate as I(1) and the policy rate as I(0)
(variable-priors.md), starts the sample at inflation-targeting adoption
(policy-regime.md), and may expect a price puzzle through exchange rate
pass-through (economy-profile.md, literature/).
