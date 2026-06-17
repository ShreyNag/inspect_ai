# Multilingual Knowledge Evaluation (Module 1)

## Overview

This evaluation measures the multilingual knowledge and reasoning capabilities of Large Language Models (LLMs) across multiple Indian languages using a translated subset of the MMLU (Massive Multitask Language Understanding) benchmark.

The benchmark evaluates whether model performance degrades when the same knowledge questions are presented in different Indian languages.

---

## Objective

The primary objective is to assess:

* Multilingual knowledge retention
* Cross-lingual reasoning consistency
* Performance degradation across languages
* Robustness of LLMs on translated educational and factual questions

---

## Dataset

Base Dataset: MMLU (Massive Multitask Language Understanding)

Domains include:

* Mathematics
* Science
* Social Sciences
* Humanities
* Professional Subjects

Each question contains:

* Question text
* Four answer choices (A–D)
* Single correct answer

---

## Languages Evaluated

* English (en)
* Hindi (hi)
* Tamil (ta)
* Telugu (te)
* Bengali (bn)
* Marathi (mr)
* Gujarati (gu)

Questions and textual answer options are translated while preserving answer labels (A–D), numeric values, equations, and metadata.

---

## Evaluation Procedure

1. Load multilingual dataset.
2. Generate model response using INSPECT AI.
3. Extract predicted answer choice.
4. Compare prediction with ground-truth answer.
5. Aggregate results by language and subject.

---

## Metrics

### Overall Accuracy

Percentage of correctly answered questions across all languages.

### Language-wise Accuracy

Accuracy computed separately for each language.

### Subject-wise Accuracy

Accuracy computed separately for each MMLU subject category.

---

## Model

Current Development Model:

* Ollama
* Llama 3.2

---

## Outputs

Generated artifacts:

* multilingual_summary.csv
* multilingual_language_report.csv
* multilingual_subject_report.csv
* benchmark_report.md

---

## Limitations

* Current development runs use a reduced subset of MMLU.
* Translation quality may influence performance.
* Results depend on translation fidelity across languages.


.\garak-env\Scripts\Activate.ps1

inspect eval india_evals/module1/multilingual/task.py@multilingual --model ollama/llama3.2

inspect view

python -m india_evals.module1.scripts.analyze_multilingual