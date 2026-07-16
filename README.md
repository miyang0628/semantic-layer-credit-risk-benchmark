# Semantic Layer Architecture for Credit Rating AI

An experimental comparison of Semantic Layer and Text-to-SQL approaches for
LLM-based data querying in high-security financial domains, including a
payload-level verification of the Semantic Layer's security claim and a
corrected implementation developed in response to that verification.

---

## Overview

This repository contains the code, data, and results for a benchmark study
examining whether a **Semantic Layer** can simultaneously improve query
accuracy and reduce sensitive information exposure compared to the
conventional **Text-to-SQL** approach, in the context of credit rating AI
systems.

An initial Semantic Layer implementation (**v1**) defines concept-level
abstractions but also includes, in the same LLM-facing prompt, a mapping
reference from concept names to physical SQL. Exhaustive verification of the
complete prompt payload — not a schema-level approximation — found that this
mapping reference reintroduces all 15 sensitive column identifiers the
concept layer is nominally designed to hide, so v1 achieves **no reduction**
in schema-level exposure relative to Text-to-SQL. A corrected implementation
(**v2**) withholds this mapping from the LLM entirely, applying it via a local
compiler after the API response is returned, and achieves the originally
intended 100% reduction, verified exhaustively across the full benchmark.

The central finding is not that a Semantic Layer resolves the
security-efficiency dilemma unconditionally, but that whether it does so
depends on an implementation detail invisible to a schema-level exposure
analysis, and that accuracy parity between the corrected architecture and
Text-to-SQL holds on a benchmark designed alongside the Semantic Layer
specification but does not extend uniformly to an externally-authored
question set evaluated against the same schema. We characterize this as a
**selective encapsulation** problem rather than a general-purpose
security-efficiency mediation effect.

---

## Research Questions

- **RQ0.** Does a given Semantic Layer implementation actually withhold
  sensitive schema information from the LLM, verified over the complete
  prompt payload rather than a single component in isolation?
- **RQ1.** Does the Semantic Layer improve data query accuracy compared to
  Text-to-SQL in the credit rating domain, and does this hold across
  benchmarks and models?
- **RQ2.** Does a correctly-implemented Semantic Layer structurally reduce
  sensitive information exposure to the LLM compared to Text-to-SQL?

---

## Key Findings

### Exposure verification (RQ0)

| Condition | Sensitive columns exposed (of 15) | Internal codes exposed (of 9) |
|---|---|---|
| Text-to-SQL | 15/15 | 0/9 |
| Semantic Layer v1 (initial) | 15/15 — **0% reduction** | 0/9 |
| Semantic Layer v2 (corrected) | 0/15 — **100% reduction** | 0/9 |

v1's mapping reference and physical table/column listing, present in the same
LLM-facing prompt as the concept definitions, reintroduce the raw identifiers
the concept layer is meant to hide. v2 withholds this mapping from the LLM
entirely; a local compiler applies it after the API response is returned.

### Accuracy (RQ1)

| Phase | Dataset | v1 Accuracy | v2 Accuracy | SQL Accuracy | Note |
|---|---|---|---|---|---|
| Phase 1 | ACME Insurance (n=2,585/condition) | 75.4% | — | 62.9% | Separate reproduction study; not re-run under v1/v2 |
| Phase 2 | BIRD Financial (n=106 questions) | 42.5% | **31.1%** | 38.7% | v1-vs-v2 gap attributed principally to compiler limitations on an externally-authored question set (see below) |
| Phase 3 | Credit Rating (n=60 questions × 5 iter.) | 38.3% | 38.7% | 39.0% | v1 vs. v2: paired $t\approx0.00$, $p=1.00$; TOST-equivalent at ±10pp |
| Phase 3 (GPT-4o) | Credit Rating, v2 only | — | 42.0% | — | vs. GPT-4o-mini v2: $t=0.83$, $p=.41$, not significant |

- **Category-level (Phase 3, v1 vs. v2)**: v2 advantage in card risk (+17.1pp)
  and default risk (+14.0pp); v1 advantage in regional risk (-12.7pp) and
  transaction behavior (-11.8pp), attributed to multi-hop join-path
  resolution and date/measure-handling limitations in the local compiler.
- **Failure taxonomy (Phase 3, v2, GPT-4o-mini)**: correct 38.7%, compile
  failure 21.0%, execution failure 27.7%, timeout 1.0%, wrong result 11.7%.
- **BIRD Phase 2 gap (v1 vs. v2)**: manual reclassification of the dominant
  compile-error subtype found the majority of the 79 flagged records to be
  genuine measure-in-WHERE violations rather than compiler false positives;
  misclassified records account for at most ~3.4% of the full record set and
  do not materially change the reported accuracy gap.
- **Component ablation (18-question stratified subset)**: removing few-shot
  examples significantly reduces accuracy (Δ=-13.3pp, p=.042) and roughly
  doubles the compile-error rate (25.6%→52.2%); removing natural-language
  descriptions has no significant effect (Δ=+3.3pp, p=.68).
- **Cross-model comparison**: the *join hint unresolved* compile-error
  subtype occurs at an identical rate (28 records) under GPT-4o-mini and
  GPT-4o, consistent with this being a compiler-level rather than
  model-level limitation.
- **Difficulty effect**: accuracy declines monotonically with difficulty
  across phases and conditions; we distinguish this conventional difficulty
  gradient from genuine Jagged Frontier–style irregularity (adjacent,
  similarly-difficult tasks producing sharply divergent outcomes), evidence
  for which appears in the Phase 1 simple/multi-hop reversal, the Phase 3
  category-level v1-vs-v2 reversals, and the cross-model category reversals
  above, rather than in the difficulty-stratified decline alone.

---

## Repository Structure

```
semantic-layer-credit-risk-benchmark/
│
├── 📄 utils.py                                   # Shared functions (LLM query, evaluation, exposure measurement)
├── 📄 semantic_compiler.py                       # v2 local compiler: concept references → physical SQL
├── 📄 evidence_sanitizer.py                      # Payload-level exposure verification utility
│
├── 📄 00_fetch_phase1_data.ipynb                 # Phase 1: fetch dbt-labs benchmark data
├── 📄 01_fetch_bird_data.ipynb                   # Phase 2: fetch BIRD Dev Set
├── 📄 02_coverage_analysis.ipynb                  # Semantic Layer cube coverage vs. BIRD schema
├── 📄 03_pilot_run.ipynb                          # Prompt-engineering pilot experiments
├── 📄 04_full_experiment.ipynb                    # Phase 3 main experiment (v2, GPT-4o-mini)
├── 📄 05a_exposure_final_check.ipynb              # RQ0: full-payload exposure verification (v1 vs. v2)
├── 📄 05b_full_run_diagnostic.ipynb               # Diagnostic: measure-in-WHERE false-positive investigation
├── 📄 05c_verify_run_used_sanitizer.ipynb         # Self-audit: confirms exposure claims match actual API payloads
├── 📄 06_statistical_reanalysis.ipynb             # Question-level paired reanalysis, TOST equivalence, failure taxonomy
├── 📄 07_multi_model_experiment.ipynb             # Phase 3 experiment on GPT-4o
├── 📄 08_multi_model_comparison.ipynb             # GPT-4o vs. GPT-4o-mini statistical comparison
├── 📄 09_latency_analysis.ipynb                   # Full latency distribution, paired tests, tail latency
├── 📄 10_bird_phase2_v2_experiment.ipynb          # Phase 2 experiment under v2 architecture
├── 📄 11_independent_gold_sql_validation.ipynb    # Stratified gold-SQL clarity check (18 of 60 questions)
├── 📄 12_prompt_ablation_experiment.ipynb         # Few-shot / description component ablation (v2)
├── 📄 all_figures.ipynb                           # Publication-quality figures
│
├── 📄 financial_semantic_layer.yaml              # Semantic Layer specification (BIRD financial DB)
│
├── 📄 credit_rating_questions_all.json           # Credit rating question set (60 questions, CR01–CR60)
│
├── 📄 results_phase3_v2.csv                      # Phase 3, v1 (original implementation), 600 records
├── 📄 results_phase3_v2_arch.csv                 # Phase 3, v2 (corrected implementation), 300 records
├── 📄 results_phase3_v2_gpt4o.csv                # Phase 3, v2, GPT-4o, 300 records
├── 📄 results_phase2_bird_v2_arch.csv            # Phase 2, v2, 530 records
├── 📄 results_ablation.csv                        # Component ablation results (18-question subset)
├── 📄 gold_sql_independent_validation.csv        # Gold SQL clarity-check comparison records
│
├── 🖼️ fig01_architecture_comparison.png          # Text-to-SQL vs. SL v1 vs. SL v2 architecture
├── 🖼️ fig02_experimental_design_overview.png     # Overall experimental design (5-level overview)
├── 🖼️ fig03_overall_accuracy.png                 # Overall accuracy, all phases and conditions
├── 🖼️ fig04_difficulty_accuracy.png              # Accuracy by difficulty level
├── 🖼️ fig05_category_accuracy.png                # Accuracy by category, v1 vs. v2 (Phase 3)
├── 🖼️ fig06_exposure_comparison.png              # Sensitive information exposure, verified over full payload
├── 🖼️ fig07_iteration_stability.png              # Accuracy across 5 iterations, v1 vs. v2
├── 🖼️ fig08_model_comparison.png                 # Category-level accuracy, GPT-4o-mini vs. GPT-4o
│
├── 📁 dbt-llm-sl-bench/                          # Phase 1 dataset (dbt-labs benchmark)
├── 📁 dev/                                       # BIRD Dev Set (Li et al., 2023)
│
└── 📄 semantic_bench_requirements.txt            # Python dependencies
```

> **Note on provenance:** exposure and accuracy figures reported above were
> verified over the complete LLM-facing API payload for each condition (see
> `05a_exposure_final_check.ipynb`, `05c_verify_run_used_sanitizer.ipynb`),
> not approximated from a single prompt component in isolation. Raw,
> per-question, per-iteration outcomes for every condition, phase, and model
> reported here — including generated SQL, compile/execution error messages,
> and full API request payloads — are included in the CSV files above rather
> than summarized only in aggregate.

---

## Experimental Design

### Phase 1 — ACME Insurance (External Reproduction Study)
- **Dataset**: dbt-labs/dbt-llm-sl-bench benchmark database
- **Scale**: 11 questions × 24 models × 20 iterations = 5,170 records
- **Role**: External benchmark re-analysis, reported for context; not re-run
  under the v1/v2 distinction developed for Phases 2–3, since that
  distinction was motivated by a payload-level check specific to this
  study's own implementation and has no bearing on this pre-existing
  benchmark's reported figures.

### Phase 2 — BIRD Financial (External Validity)
- **Dataset**: BIRD Benchmark Dev Set, financial domain (Li et al., 2023)
- **Scale**: 106 questions, evaluated under Text-to-SQL, v1, and v2
- **Coverage note**: the Semantic Layer specification's 7 cubes cover 6 of
  BIRD's 8 financial-domain tables; the `order` table has no corresponding
  cube. 4 of 106 questions reference `order` or a table combination not
  jointly covered by a single cube and are expected to fail under v2 for
  this reason; retained in the analysis rather than excluded.
- **Role**: External validity on a standard NL2SQL benchmark, and a test of
  whether the corrected (v2) architecture's accuracy parity with v1
  generalizes to a question set the Semantic Layer specification was not
  designed alongside.

### Phase 3 — Credit Rating Domain (Main Experiment)
- **Dataset**: Custom-designed credit rating question set (60 questions, 6
  categories), authored solely by this study
- **Scale**: 60 questions × 3 conditions (Text-to-SQL, v1, v2) × 5
  iterations; a subset also evaluated under GPT-4o
- **Categories**: default_risk, regional_risk, transaction_behavior,
  loan_portfolio, client_profile, card_risk
- **Role**: Domain-specific benchmark — first credit rating NL2SQL benchmark
  proposed. The Semantic Layer specification was designed concurrently with
  this question set, which we note as relevant context for interpreting the
  accuracy parity observed here relative to the larger gap observed on
  Phase 2.

---

## Methods

Three query generation conditions are compared:

**Text-to-SQL (Baseline)**
- LLM receives raw DDL schema
- All internal column names (A2–A16), Czech-language codes, and status codes
  are directly exposed

**Semantic Layer v1 (Initial Implementation)**
- LLM receives abstracted concept definitions (cube names, dimension/measure
  descriptions)
- The LLM-facing prompt *also* includes a SQL mapping reference and physical
  table/column listing, needed for the LLM to compose valid joins
- Verified over the complete prompt payload, this implementation does **not**
  achieve a reduction in sensitive column exposure relative to Text-to-SQL

**Semantic Layer v2 (Corrected Implementation)**
- LLM receives only concept names, descriptions, and concept-level join
  relationships — no mapping reference or physical schema information
- A local compiler, executed after the API response and never exposed to the
  LLM, resolves concept references to physical SQL
- Verified over the complete prompt payload, this implementation achieves a
  100% reduction in sensitive column exposure

**Evaluation metric**: Execution Accuracy (EX) — result-set level match,
consistent with the BIRD benchmark standard. All condition comparisons are
conducted at the question level (5 iterations aggregated to a single
per-question accuracy value before any significance test), treating
conditions on a shared question set as paired rather than independent
observations.

---

## Statistical Validation

| Test | Purpose | Key Result |
|---|---|---|
| Paired $t$-test / Wilcoxon | Question-level v1-vs-v2 comparison (Phase 3) | $t\approx0.00$, $p=1.00$; Wilcoxon $p=.948$ |
| Question-level bootstrap CI | Paired accuracy difference (5,000 resamples) | 95% CI $[-0.103, +0.098]$ |
| TOST equivalence test | Formal equivalence vs. non-significance | Equivalent at ±10pp/±15pp; not at ±5pp |
| Chi-square | Phase-level accuracy comparison (retained from original design) | Phase 1: p<.001; Phase 2 (v1 vs. SQL): n.s. |
| One-way ANOVA | Difficulty level effect | F > 40, p < .001 (v1 and Text-to-SQL) |
| Paired ablation $t$-test | Few-shot / description component contribution | Few-shot: p=.042; description: p=.68 |

---

## Setup

### Requirements

```bash
pip install -r semantic_bench_requirements.txt
```

Notebooks using SQL-structure comparison (e.g., the measure-in-WHERE
reclassification in `05b_full_run_diagnostic.ipynb`) additionally require
`sqlglot` and `pyyaml`:

```bash
pip install sqlglot pyyaml
```

### Environment Variables

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_api_key_here
```

### Data

The BIRD Dev Set is required for Phase 2. Download from
[https://bird-bench.github.io](https://bird-bench.github.io) and place the
extracted folder at:

```
./dev/dev_20240627/
```

The dbt-labs benchmark database is required for Phase 1. Clone from
[https://github.com/dbt-labs/dbt-llm-sl-bench](https://github.com/dbt-labs/dbt-llm-sl-bench)
and place at:

```
./dbt-llm-sl-bench/
```

### Execution Order

```
00_fetch_phase1_data.ipynb
01_fetch_bird_data.ipynb
02_coverage_analysis.ipynb
03_pilot_run.ipynb
04_full_experiment.ipynb
05a_exposure_final_check.ipynb
05b_full_run_diagnostic.ipynb
05c_verify_run_used_sanitizer.ipynb
06_statistical_reanalysis.ipynb
07_multi_model_experiment.ipynb
08_multi_model_comparison.ipynb
09_latency_analysis.ipynb
10_bird_phase2_v2_experiment.ipynb
11_independent_gold_sql_validation.ipynb
12_prompt_ablation_experiment.ipynb
all_figures.ipynb
```

---

## Theoretical Framework

This study draws on:

- **Information Hiding** (Parnas, 1972): The Semantic Layer is intended to
  encapsulate internal schema implementation, exposing only abstract business
  concepts to the LLM. This study finds that whether this is genuinely
  achieved depends on whether the concept-to-SQL mapping itself is withheld
  from the LLM-facing prompt — an implementation-level distinction not
  visible from the concept layer's design alone.
- **Selective Encapsulation** (this study): abstraction improves accuracy
  where the underlying raw encoding is opaque and can degrade it where the
  raw schema is already transparent; full information hiding preserves
  accuracy in aggregate only when the specification is designed with the
  target query workload in mind.
- **Jagged Frontier** (Dell'Acqua et al., 2023): LLM performance varies
  non-uniformly across task types and complexity levels. We distinguish a
  conventional monotonic difficulty gradient from genuine jaggedness
  (irregular reversals among nominally comparable tasks) and report evidence
  of the latter specifically in category-level and cross-model reversals.

---

## Known Limitations

- Evaluated on two same-family models (GPT-4o-mini, GPT-4o); not verified
  beyond this pair or on models fine-tuned specifically for SQL generation.
- Phases 2 and 3 share the same underlying database schema (BIRD financial);
  generalizability to other proprietary schemas is untested.
- Exposure measurement is scoped to schema-structural tokens within the
  LLM-facing prompt; query-value and output-level exposure are not captured.
- The component ablation was conducted only on the v2 architecture and was
  not repeated symmetrically on Text-to-SQL.
- The gold-SQL clarity check (`11_independent_gold_sql_validation.ipynb`)
  covers a stratified 18-question sample re-derived by the benchmark's
  single author, not independent third-party validation of the full
  60-question benchmark.
- Latency comparisons across conditions and models were run in separate
  sessions rather than interleaved, and are reported descriptively rather
  than causally.

---

## Citation

This repository is associated with a manuscript currently under review.
Citation information will be updated upon publication.

---

## License

This project is released for academic reproducibility purposes. The BIRD
benchmark dataset is subject to its original license. The dbt-labs benchmark
is subject to its original license.
