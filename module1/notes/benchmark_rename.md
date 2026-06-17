# Module 1 – Multilingual Knowledge Evaluation Report

## Benchmark Summary

| Metric           | Value  |
| ---------------- | ------ |
| Overall Accuracy | 0.3500 |

---

## Language-wise Accuracy

| Language | Accuracy | Samples |
| -------- | -------- | ------- |
| English  | 0.6667   | 3       |
| Hindi    | 0.3333   | 3       |
| Tamil    | 0.3333   | 3       |
| Telugu   | 0.3333   | 3       |
| Bengali  | 0.0000   | 3       |
| Marathi  | 0.3333   | 3       |
| Gujarati | 0.5000   | 2       |

---

## Subject-wise Accuracy

| Subject          | Accuracy | Samples |
| ---------------- | -------- | ------- |
| Abstract Algebra | 0.3500   | 20      |

---

## Observations

* The evaluation was conducted using the multilingual MMLU benchmark.
* Development testing used a reduced dataset for rapid iteration.
* Model responses were normalized to extract answer choices (A–D).
* Accuracy varied across languages, although current development results are based on a small sample size.

---

## Model Configuration

* Framework: INSPECT AI
* Model: Ollama / Llama 3.2
* Evaluation Type: Multiple Choice Question Answering
* Scoring Method: Exact Match (Normalized Answer Extraction)

---

## Future Work

* Evaluate on the complete multilingual dataset.
* Incorporate professionally translated benchmark questions.
* Compare multiple LLMs.
* Analyze performance degradation relative to English baseline.
