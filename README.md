# MaRCo@BIS: the MAcroeconometric Research Companion

**A skill framework that turns an AI coding agent into a research companion
for macroeconometric analysis.** Released as a public good alongside the
BIS working paper *"MaRCo@BIS: AI-agentic econometric analysis for central
bankers"* (Lombardi, Park and Perez-Cruz, BIS Working Papers, no 1395,
2026). You open Claude Code
in this repo, ask a research question, and the agent walks you through a
7-stage workflow with human-in-the-loop confirmation at every important
decision: model selection, variables, lags, identification, sample,
sensibility checks, reporting.
Each choice states its source (you, a dictionary entry, the knowledge base,
or the literature), and the agent is instructed to flag anomalies (price
puzzle, explosive root, weak instrument) before they reach your final results.

The framework has three parts: [skills](skills/) (the workflow and domain
knowledge, as plain Markdown), a [reference Python library](scripts/lib/)
covered by more than 1,000 tests, and a
[dictionary](skills/model-dictionary/entries/) of canonical paper
specifications paired with worked replications.

## Quick start

1. Install [Claude Code](https://claude.com/claude-code).
2. Unzip the distributed archive into a folder and `cd` into it.
3. `pip install -r requirements.txt` (Python ≥ 3.10, tested on 3.11;
   tested on Windows and Ubuntu; not tested on macOS). This is
   needed to *run* estimations; the skill layer alone needs nothing.
   For reproducibility, version pins
   are retained in [requirements-reproduction.txt](requirements-reproduction.txt).
4. Download the data CSVs once: see `data/fred_md/UPDATE.md` and
   `data/fred_qd/UPDATE.md` (the CSVs are not distributed). Run
   the cleaning script in each UPDATE.md, because the raw FRED
   download needs its metadata rows stripped before the loader can
   read it.
5. Open Claude Code in the repo. Say one of:
   - **Mode A:** *"Replicate the Sims (1980) six-variable VAR using FRED data."*
   - **Mode B:** *"Here's a PDF. Replicate Smith and Jones (2024) on monetary policy."*
   - **Mode C:** *"How does monetary policy affect output and inflation?"*

The agent triages your request to the right mode and walks you through each
stage, generating code and results as it goes. You confirm or override at
each checkpoint; no silent decisions.

To verify an installation end-to-end, first `pip install pytest` (a
development dependency, not in `requirements.txt`); then
`python -m pytest tests -q` runs the fast suite (~2 minutes; data-dependent tests skip cleanly until the FRED
panels are downloaded), and
`python -m pytest tests/test_replications_smoke.py --runslow` runs every
replication against its paper targets (about 20 minutes): numbers read from the
paper where it reports directly comparable ones, and otherwise the authors'
own results from the same replication; see [`tests/README.md`](tests/README.md).

## Three ways to use it

The framework supports three entry modes, each with its own workflow:

| Mode | When | What the agent does |
|---|---|---|
| **A: Dictionary execution** | You name a paper / method already in the model dictionary | Look up the verified entry, run the existing replication, report results |
| **B: Paper submission** | You provide a paper (PDF / URL / citation) that's not yet in the dictionary | Parse the paper, draft a dictionary entry, ask you to verify the spec, then execute |
| **C: Research design** | You ask an open question with no specific method in mind | Search the dictionary for matches; if none, design a specification from parametric knowledge and economic theory, with every decision's source flagged |

Modes can transition mid-session (a Mode B that turns into a Mode A once
the dictionary entry is verified, for instance) and the agent always
announces transitions explicitly.

See [`skills/orchestrator/SKILL.md`](skills/orchestrator/SKILL.md) and
[`skills/orchestrator/modes.md`](skills/orchestrator/modes.md) for the full
mode-triage protocol.

## How a session looks

```
You ask a question
  │
  ▼
0a. Knowledge base: load saved economy context (offers setup if none, or you can proceed without it)
0b. Triage: Mode A / B / C
  │
  ▼
1. Model selection / entry retrieval / paper parsing  ✋ confirm
2. Specification (vars, lags, priors, identification) ✋ confirm
3. Data collection                                    ✋ confirm variable mapping
4. Estimation: run the existing script or a new thin one, report (no confirm)
5. Sensibility checks: price puzzle? stable? plausible?
   if fails: error fixes / pre-specified rules re-run automatically (max 5);
   economic respecification                           ✋ confirm (runs as labelled sensitivity)
6. Results & reporting: interactive HTML report (PDF charts, LaTeX tables on request)
7. Disposition & curation                             ✋ archive / save / promote / continue / discard
```

Any stage can be revised at any time. Say "add commodity prices" mid-analysis
and the framework re-runs only the affected stages, preserving unchanged work
and logging the revision in the research log.

## What gets created

| What | Where | Kept or regenerable? |
|---|---|:---:|
| Estimation scripts | `scripts/applications/<id>/` (default) or `scripts/replications/<id>/` (canonical only) | Kept |
| Interactive HTML reports | `scripts/<tier>/<id>/outputs/report.html` | Regenerable |
| Numerical snapshots | `scripts/<tier>/<id>/outputs/numerical_results.json` | Regenerable |
| Run-level research logs | `scripts/<tier>/<id>/outputs/research_log.json` | Regenerable |
| Session records: the agent's decisions, gates, dispositions and continuation state | `scripts/<tier>/<id>/outputs/session_research_log.json` (+ `research_log.md`, written at completion) and `run_state.json` in the same folder | **Kept (not regenerable)** |
| Dataset CSVs: pre-collected panels (e.g. FRED-MD/QD), runtime-fetched caches, user-staged series | `data/<id>/dataset.csv` | Regenerable (re-download, re-fetch or re-stage; see each dataset's `UPDATE.md`) |

To start fresh: delete the generated files in the script's `outputs/` folder
and re-run, but first copy out the session records above
(`session_research_log.json`, `research_log.md`, `run_state.json`). They record
what was decided in conversation and cannot be rebuilt by re-running a script.
`outputs/` is excluded from version control, so these files are not committed
either unless you copy them elsewhere. The numerical outputs regenerate from
the same code, settings, random seed (the script's own `config.yaml`) and data
files.

## Replications

A public-good gallery in
[`scripts/replications/`](scripts/replications/README.md): **one canonical
replication per dictionary entry**, each in its own folder with a README, config,
runnable script, and paper_targets.yaml validation (outputs are
regenerable: re-run `replicate.py` to rebuild them, typically in
under a minute each; a few replications need paper-specific staged
data, documented in their folder READMEs). The gallery's own
[README](scripts/replications/README.md) lists every worked example.

The model dictionary at
[`skills/model-dictionary/entries/`](skills/model-dictionary/entries/) holds
the full set of recognised specifications. In your own copy, Mode B adds
new dictionary entries and Stage 7 can promote an application to a
canonical replication.

## Data

The data layer (`data/<id>/`) is source-agnostic: each dataset is a folder
with a `dataset.csv`, per-series metadata, and a registry entry, whatever
the origin. Datasets arrive three ways:

- **Pre-collected panels:** curated, documented in an `UPDATE.md`,
  refreshed manually. The shipped examples are the standard US research
  panels: **FRED-MD** (monthly) and **FRED-QD** (quarterly), 1959 to
  the latest vintage (CSVs not distributed; download once via their
  `UPDATE.md`).
- **Runtime-fetched caches:** any series reachable through the pluggable
  fetchers ([`scripts/lib/data/fetchers/`](scripts/lib/data/fetchers/)):
  `fred` (individual FRED series; needs a free `FRED_API_KEY`; see
  `skills/data-retrieval/SKILL.md` for 2-minute setup), `dbnomics`
  (ECB / OECD / IMF / BIS / Eurostat / World Bank through one no-key
  API), `ons`, `boe_iadb`, `csv_file`, `external_archive`. Fetched once,
  cached locally, auto-registered.
- **User-staged series:** data with no stable public URL (author
  archives, internal series) dropped into `data/staging/` per the
  conventions in the replication READMEs that need them.

The pre-collected CSVs need no API key at all. New economies add their own
datasets through the same three routes; nothing in the layer is specific
to FRED or to US data.

Data licensing: no bulk data is redistributed with the repo. The connectors
and replication scripts retrieve data from third-party sources and keep
local copies, and each source's terms of use apply to you, including any
limits on storing or redistributing the data or using it with AI systems.
This product uses the FRED® API but is not endorsed or certified by the
Federal Reserve Bank of St. Louis. By retrieving FRED data with this
framework you agree to the
[FRED® API Terms of Use](https://fred.stlouisfed.org/docs/api/terms_of_use.html)
and the [FRED® Services Terms of Use](https://fred.stlouisfed.org/legal/).

## Skill inventory

The skills under [`skills/`](skills/) cover workflow and knowledge. Estimation methodology lives separately in
[`scripts/lib/`](scripts/lib/); see "Estimation methods" below.

| Stage | Skill | Description |
|-------|-------|-------------|
| 0 | [knowledge-base](skills/knowledge-base/SKILL.md) | Economy profiles, conventions, prior beliefs; loaded at intake, consulted throughout |
| 0 | *(orchestrator)* | Intake and triage into mode A / B / C |
| 1 | [model-selection](skills/model-selection/SKILL.md) | Research question → methodology mapping |
| 1 | [model-dictionary](skills/model-dictionary/SKILL.md) | Paper specifications: lookup, provenance and entry curation |
| 2 | [specification](skills/specification/SKILL.md) | Variables, lags, priors, identification, sample |
| 3 | [data-retrieval](skills/data-retrieval/SKILL.md) | FRED-MD/QD datasets + runtime API fetch |
| 4 | *(routes to `scripts/lib/`)* | Estimation methods live in the library, not a per-method skill |
| 4 | [diagnostics](skills/diagnostics/SKILL.md) | Residual tests and model adequacy checks (the statistical gate) |
| 5 | *(orchestrator)* | Economic sensibility check (signs, magnitudes, stability); consumes the Stage 4 diagnostics |
| 6 | [results-presentation](skills/results-presentation/SKILL.md) | Charts, tables and HTML reports |
| 7 | *(orchestrator)* | Disposition & curation: archive, save as application, promote, continue, discard |
| n/a | [orchestrator](skills/orchestrator/SKILL.md) | Main agentic loop coordinating all stages |
| n/a | [benchmarking](skills/benchmarking/SKILL.md) | On-request evaluation against baselines, competitors and published results; not a stage gate |
| n/a | [consistency-audit](skills/consistency-audit/SKILL.md) | On-demand drift audit across skills, lib, and dictionary |
| n/a | [advisor-protocol](skills/advisor-protocol/SKILL.md) | Optional review of difficult derivations by another model; off by default |
| n/a | [research-log](skills/research-log/SKILL.md) | Session decisions, versions, disposition and continuation state |

## Estimation methods

Estimation lives in [`scripts/lib/`](scripts/lib/README.md), a Python library
covering:

- **VAR / BVAR:** classical VAR, Minnesota / Sims-Zha priors, dummy
  observations, GLP hyperparameter optimisation, steady-state / mean-adjusted
  prior (Villani 2009), block exogeneity for small open economies,
  multi-instrument proxy SVAR
- **Local projections:** linear LP, LP-IV, state-dependent LP
  (Auerbach-Gorodnichenko 2012 / Ramey-Zubairy 2018), Bayesian LP
  (Ferreira, Miranda-Agrippino & Ricco 2025)
- **Identification:** Cholesky, sign restrictions (basic / multi-shock RWZ
  2010 / narrative ADRR 2018 Type 1 + Type 2 / magnitude / FEV-share /
  higher-moment AFM 2023), zero+sign (ARRW 2018), long-run (Blanchard-Quah),
  heteroskedasticity (Rigobon 2003), non-Gaussian SVAR (Lanne-Meitz-Saikkonen
  2017 FastICA + parametric pseudo-MLE)
- **Time-varying / panel / mixed-frequency:** TVP-VAR with stochastic
  volatility (Primiceri 2005), Lenza-Primiceri 2022 COVID-VAR, panel VAR
  (pooled / mean-group / Bayesian partial-pooling / hierarchical / stacked cross-country), MF-VAR
  (Schorfheide-Song 2015), FAVAR (BBE 2005), BDFM (Banbura-Modugno 2014)
- **Forecasting / decomposition:** iterated forecast, rolling-window OOS,
  Diebold-Mariano, conditional forecast (BGL 2015), historical decomposition,
  structural FEVD, connectedness (Diebold-Yilmaz 2012)
- **Filters / dating:** HP filter, Hamilton 2018 filter, Beveridge-Nelson,
  Kamber-Morley-Wong 2018 BNF, simplified Bry-Boschan / Harding-Pagan dating
- **Preprocessing / data:** deflate / nominalise, aggregate (mixed-freq),
  impute (linear / Kalman / EM via Shumway-Stoffer), FRED-MD/QD loaders,
  pluggable fetchers (FRED, DBnomics, ONS, BoE IADB, external archives)
- **Validation:** `compare_to_paper` (paper-target validation), spec /
  run / diagnostics / sensitivity layer for hyperparameter sweeps

For non-Python target languages (R / Julia / Matlab), the agent generates
code following the same methodology, using the lib module docstrings as
the canonical specification.

## Project structure

```
marco-bis/
├── README.md                       this file
├── CLAUDE.md                       agent operating instructions
├── DEVELOPMENT.md                  architecture + how to add new methods
├── CONVENTIONS.md                  math conventions used in scripts/lib/
├── TEST_SUITE.md                   manual checklist of agent behaviour
├── config.yaml                     recommended defaults the agent reads (not loaded by lib)
├── requirements.txt                Python dependencies
├── CITATION.cff, LICENSE, NOTICE, SECURITY.md
│
├── skills/                         workflow + knowledge, one folder per skill
│   ├── orchestrator/               main agentic loop (7-stage workflow) + mode triage
│   ├── knowledge-base/             economy profiles (context/ ships empty; you fill it)
│   ├── model-dictionary/           index.yaml + entries/ (canonical paper specifications)
│   └── <other skills>/SKILL.md     decision protocols, output, research log
│
├── scripts/
│   ├── lib/                        reference Python library (estimation, data, preprocessing)
│   ├── replications/               worked replications, one folder per dictionary entry
│   ├── applications/               your own analyses (ships empty)
│   └── tools/                      maintenance tools (author dossiers)
│
├── data/
│   ├── registry.yaml               dataset registry
│   ├── fred_md/                    UPDATE.md + transform codes; dataset.csv is written by the UPDATE.md recipe
│   ├── fred_qd/                    UPDATE.md + transform codes and factor flags; same
│   ├── staging/                    raw downloads and staged series (you create it; gitignored)
│   └── <id>/                       runtime-fetched caches (created by lib.fetch_data)
│
├── papers/                         source PDFs you stage for Mode B (PDFs are gitignored)
├── tests/                          over 1,000 unit tests + slow coverage/smoke tests
└── outputs/                        repo-root scratch (created on first use); per-run results live in scripts/<tier>/<id>/outputs/
```

See [DEVELOPMENT.md](DEVELOPMENT.md) for architecture principles, the
standardised IRF/FEVD output contract, and how to add new estimation skills
or canonical replications.

## For framework developers

If you're working ON the framework (not WITH it), see:

- [`CLAUDE.md`](CLAUDE.md): orientation for the agent when invoked in this repo
- [`DEVELOPMENT.md`](DEVELOPMENT.md): architecture + interface contracts
- [`CONVENTIONS.md`](CONVENTIONS.md): math conventions pinned by tests
- [`scripts/lib/README.md`](scripts/lib/README.md): Python library modules
  and dependency DAG
- [`tests/README.md`](tests/README.md): test discipline and what each test layer pins

## Privacy and data flow: read before using with internal data

Agent sessions send your prompts and any file contents the agent
reads (including data files you point it at) to the model provider's
API for processing (Anthropic's, for Claude Code). Before using the
framework with **confidential internal series or unpublished
analysis**, clear it with your institution's policy, including which
deployment route and data-retention terms it has approved. The pre-collected FRED data and the replication gallery are
public and raise no such concern. The optional advisor protocol
(`skills/advisor-protocol/`) lets the agent send mathematical questions to a
second model, through Anthropic's API by default or another provider set in
`config.yaml`. It is off until you set `advisor.enabled: true` there. When
on, it is meant to send only equations and derivations, never data; this is
an instruction to the agent, not a technical filter (see its SKILL.md).

## Validation status and intended use

This is a research tool: outputs are inputs to expert judgement,
not official statistics or policy advice, and every run is designed to be
reviewed (propose → confirm gates, sensibility checks, research logs).
Verification is best-effort; errors may remain despite it.
What stands behind the numbers:

- the reference library's math conventions are pinned by a 1,000+ test
  suite, including parameter-recovery contracts on known DGPs and Monte
  Carlo coverage studies of the recursive VAR and local-projection bands;
- the canonical replications validate against per-paper
  `paper_targets.yaml` contracts. Targets are the paper's own numbers
  where it reports directly comparable ones; for most replications some
  or all targets are the authors' own results from the same replication,
  which catch regressions but do not by themselves establish agreement
  with the paper (the file header or each target's `location` field says which);
- each dictionary entry was checked field by field against the original
  paper PDF;
- known differences from the papers (sample gaps, method demonstrations)
  are documented in each replication's README;
- parts of the library were compared numerically with independent MATLAB
  toolboxes during development: the frequentist VAR layer, and selected
  components of the Bayesian VAR and sign-restriction code. These were
  one-time checks. The record and the convention differences are in
  [`DEVELOPMENT.md`](DEVELOPMENT.md) §"Independent cross-checks" and
  `CONVENTIONS.md`;
- standard inference critiques for set-identified and LP methods are
  documented in `CONVENTIONS.md` §4 and the relevant docstrings.

The evidence is not the same for every method. The VAR layer (OLS
estimation, recursive, long-run and single-instrument proxy identification,
impulse responses and decompositions) and the HP and band-pass filters are
compared with independent implementations; the Minnesota BVAR and the
sign-restriction sampler were compared once during development; all other
methods, including the time-varying, mixed-frequency, factor, panel and
non-Gaussian models, rest on the library's own tests on simulated data.
Results from this last group deserve more caution: where a result matters,
check it against another implementation. The
[replication index](scripts/replications/README.md) labels each replication
as following the paper's specification, using the paper's system with a
modified method, or demonstrating the method only.

## Dissemination

The framework ships as a self-contained archive with the working paper.
The public repository is a frozen copy of the same release. Installation
is in the Quick start above.

## How to cite

See [`CITATION.cff`](CITATION.cff) (GitHub renders a "Cite this repository"
button from it).

## Licence

[Apache-2.0](LICENSE). Copyright © 2026, Bank for International
Settlements. See [`NOTICE`](NOTICE) and [`SECURITY.md`](SECURITY.md)
(vulnerability reporting).

## Disclaimers

The repository content is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. Nothing in the License shall constitute, or be construed as constituting a waiver, express or implied, of any privileges, immunities and/or exemptions enjoyed by the BIS. See the License for more details regarding permissions and limitations. Copyright © 2026, Bank for International Settlements.

This repository contains sample code. The code was developed for the purposes of an experimental project. The content in this repository does not constitute regulatory, financial, legal or any other professional advice and should not be acted on as such. BIS shall not be liable for any damage or loss of any kind howsoever caused as a result of the use of the information contained or referenced in this repository.

This repository is not maintained. No contributions are expected and issues or pull requests are likely to remain unresolved.

The views expressed in this repository are those of the authors and do not necessarily reflect the views of the BIS or its member central banks. The authors acknowledge the use of artificial intelligence in research assistance and editorial refinement. Responsibility for all errors remains with the authors. References to individual firms and projects are for illustrative purposes and should not be construed as any statement on the soundness or the legal or regulatory status of any firm or project.
