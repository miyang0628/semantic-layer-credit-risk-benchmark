# Semantic Layer as a Security-Efficiency Mediator in Credit Rating AI

An experimental comparison of Semantic Layer and Text-to-SQL approaches for LLM-based data querying in high-security financial domains.

---

## Overview

This repository contains the code, data, and results for a benchmark study examining whether a **Semantic Layer** can simultaneously improve query accuracy and reduce sensitive information exposure compared to the conventional **Text-to-SQL** approach, in the context of credit rating AI systems.

The central argument is that the Semantic Layer acts as a **Security-Efficiency Mediator**: it encapsulates internal schema details (raw column names, domain-specific codes, business logic) from the LLM while maintaining or improving query accuracy — resolving the security-efficiency dilemma inherent in AI adoption at regulated financial institutions.

---

## Research Questions

- **RQ1.** Does the Semantic Layer improve data query accuracy compared to Text-to-SQL in the credit rating domain?
- **RQ2.** Does the Semantic Layer structurally reduce sensitive information exposure to the LLM compared to Text-to-SQL?
- **RQ3.** What moderating role does the Semantic Layer play in the relationship between security and efficiency?

---

## Key Findings

| Phase | Dataset | SL Accuracy | SQL Accuracy | Δ | Significance |
|---|---|---|---|---|---|
| Phase 1 | ACME Insurance (n=2,585) | 75.4% | 62.9% | +12.5%p | *** |
| Phase 2 | BIRD Financial (n=106) | 42.5% | 38.7% | +3.8%p | n.s. |
| Phase 3 | Credit Rating (n=300) | 38.3% | 39.0% | -0.7%p | n.s. |

- **Default risk category**: SL 40.0% vs SQL 20.0% (Cohen's d = +0.447)
- **Sensitive column exposure**: SQL exposes 15 raw columns (A2–A16); SL exposes **0** (100% structural elimination)
- **Latency**: SL is consistently faster or equivalent to SQL across all phases
- **Jagged Frontier**: Accuracy declines significantly with difficulty for both methods (ANOVA F > 40, p < .001)

---

## Repository Structure

```
semantic-layer-credit-risk-benchmark/
│
├── 📄 utils.py                                  # Shared functions (LLM query, evaluation, exposure measurement)
├── 📄 phase1_analysis.ipynb                     # Phase 1: ACME Insurance benchmark re-analysis
├── 📄 bird_financial_experiment.ipynb           # Phase 2: BIRD Financial domain experiment
├── 📄 credit_rating_benchmark_experiment.ipynb  # Phase 3: Credit rating domain experiment (main)
├── 📄 statistical_validation.ipynb              # Full statistical validation (all phases)
├── 📄 paper_figures.ipynb                       # Publication-quality figures (DPI 600)
│
├── 📄 financial_semantic_layer.yaml             # Semantic Layer definition (BIRD financial DB)
│
├── 📄 credit_rating_questions.json              # Credit rating question set (CR01–CR30)
├── 📄 credit_rating_questions_add.json          # Credit rating question set (CR31–CR60)
├── 📄 credit_rating_questions_all.json          # Merged question set (60 questions)
│
├── 📄 results_phase2_bird_v2.csv                # Phase 2 experiment results (212 records)
├── 📄 results_phase3_v2.csv                     # Phase 3 experiment results (600 records)
├── 📄 results_phase2_exposure_v2.csv            # RQ2 sensitive information exposure results
├── 📄 results_statistical_accuracy.csv          # Chi-square test results
├── 📄 results_statistical_anova.csv             # ANOVA results
├── 📄 results_statistical_effectsize.csv        # Cohen's d effect sizes
├── 📄 results_statistical_bootstrap_ci.csv      # Bootstrap 95% confidence intervals
├── 📄 results_rq2_exposure.csv                  # RQ2 exposure summary
│
├── 🖼️ figure1_overall_accuracy.png              # Figure 1: Overall accuracy (all phases)
├── 🖼️ figure2_difficulty_accuracy.png           # Figure 2: Accuracy by difficulty (Jagged Frontier)
├── 🖼️ figure3_category_accuracy.png             # Figure 3: Accuracy by category (Phase 3)
├── 🖼️ figure4_exposure.png                      # Figure 4: Sensitive information exposure (RQ2)
├── 🖼️ figure5_iteration_stability.png           # Figure 5: Iteration stability (Phase 3)
│
├── 📁 dbt-llm-sl-bench/                         # Phase 1 dataset (dbt-labs benchmark)
├── 📁 dev/                                      # BIRD Dev Set (Li et al., 2023)
│
└── 📄 semantic_bench_requirements.txt           # Python dependencies
```

---

## Experimental Design

### Phase 1 — ACME Insurance (Pilot)
- **Dataset**: dbt-labs/dbt-llm-sl-bench benchmark database
- **Scale**: 11 questions × 24 models × 20 iterations = 5,170 records
- **Role**: Prior work replication and baseline establishment

### Phase 2 — BIRD Financial (External Validity)
- **Dataset**: BIRD Benchmark Dev Set, financial domain (Li et al., 2023)
- **Scale**: 106 questions × 2 methods = 212 records
- **Role**: External validity on a standard NL2SQL benchmark

### Phase 3 — Credit Rating Domain (Main Experiment)
- **Dataset**: Custom-designed credit rating question set (60 questions, 6 categories)
- **Scale**: 60 questions × 2 methods × 5 iterations = 600 records
- **Categories**: default_risk, regional_risk, transaction_behavior, loan_portfolio, client_profile, card_risk
- **Role**: Domain-specific benchmark — first credit rating NL2SQL benchmark proposed

---

## Methods

Two query generation methods are compared:

**Text-to-SQL (Baseline)**
- LLM receives raw DDL schema
- All internal column names (A2–A16), Czech-language codes, and status codes are directly exposed
- Standard NL2SQL approach

**Semantic Layer**
- LLM receives abstracted concept definitions (cube names, dimension/measure descriptions)
- Raw schema details hidden via Information Hiding principle (Parnas, 1972)
- SQL mapping resolved internally without exposing sensitive tokens

**Evaluation metric**: Execution Accuracy (EX) — result-set level exact match, consistent with BIRD benchmark standard.

---

## Statistical Validation

| Test | Purpose | Key Result |
|---|---|---|
| Chi-square | Overall accuracy difference | Phase 1: p<.001; Phase 2–3: n.s. |
| McNemar's | Paired question-level comparison | n.s. across all phases |
| One-way ANOVA | Difficulty level effect | F > 40, p < .001 (both methods) |
| Kruskal-Wallis | Non-parametric difficulty test | Consistent with ANOVA |
| Bootstrap CI | 95% confidence intervals (n=5,000) | Overlapping CIs in Phase 2–3 |
| Cohen's d | Effect size | default_risk: d = +0.447 |

---

## Setup

### Requirements

```bash
pip install -r semantic_bench_requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_api_key_here
```

### Data

The BIRD Dev Set is required for Phase 2. Download from [https://bird-bench.github.io](https://bird-bench.github.io) and place the extracted folder at:

```
./dev/dev_20240627/
```

The dbt-labs benchmark database is required for Phase 1. Clone from [https://github.com/dbt-labs/dbt-llm-sl-bench](https://github.com/dbt-labs/dbt-llm-sl-bench) and place at:

```
./dbt-llm-sl-bench/
```

### Execution Order

```
1. phase1_analysis.ipynb
2. bird_financial_experiment.ipynb
3. credit_rating_benchmark_experiment.ipynb
4. statistical_validation.ipynb
5. paper_figures.ipynb
```

---

## Theoretical Framework

This study draws on:

- **Information Hiding** (Parnas, 1972): The Semantic Layer encapsulates internal schema implementation, exposing only abstract business concepts to the LLM
- **Jagged Frontier** (Dell'Acqua et al., 2023): LLM performance varies non-uniformly across task types and complexity levels
- **TOE Framework** (Tornatzky & Fleischer, 1990): Technology-Organization-Environment model for AI adoption context

---

## Citation

This repository is associated with an anonymous manuscript currently under review. Citation information will be updated upon publication.

---

## License

This project is released for academic reproducibility purposes. The BIRD benchmark dataset is subject to its original license. The dbt-labs benchmark is subject to its original license.
