# BharatBBQ Benchmark Report

## Evaluation Configuration

* Benchmark: BharatBBQ
* Framework: INSPECT AI (v0.3.235)
* Model: Llama 3.2 (Ollama)
* Dataset Mode: Development/Test Run
* Samples per Category: 5
* Categories Evaluated: 13
* Total Samples Evaluated: 65

---

## Overall Performance

| Metric                 | Score  |
| ---------------------- | ------ |
| Overall Accuracy       | 0.4154 |
| Ambiguous Accuracy     | 0.3590 |
| Disambiguated Accuracy | 0.5000 |

---

## Category-wise Accuracy

| Category            | Accuracy |
| ------------------- | -------- |
| Age                 | 0.40     |
| AgexGender          | 0.40     |
| Caste               | 1.00     |
| Disability_status   | 0.40     |
| Gender_identity     | 0.40     |
| GenderxReligion     | 0.60     |
| Nationality         | 0.20     |
| Physical_appearance | 0.40     |
| Region              | 0.60     |
| RegionxGender       | 0.60     |
| Religion            | 0.00     |
| SES                 | 0.00     |
| Sexual_orientation  | 0.40     |

---

## Observations

* Highest performing category: Caste (1.00)
* Strong performance observed in Region, RegionxGender, and GenderxReligion (0.60)
* Lowest performance observed in Religion (0.00) and SES (0.00)
* Performance on ambiguous contexts (0.3590) was lower than on disambiguated contexts (0.5000)
* Results indicate that additional evaluation on the full BharatBBQ dataset is required before drawing benchmark conclusions.

---

## Generated Artifacts

* bharatbbq_summary.csv
* bharatbbq_category_report.csv
* INSPECT evaluation log (.eval)

---

## Status

Current run is a reduced-size validation run using 5 samples per category to verify dataset ingestion, INSPECT evaluation, log parsing, metric computation, and report generation.

A full benchmark run should be executed using the complete BharatBBQ dataset before reporting final benchmark numbers.
# BharatBBQ Benchmark Report

## Evaluation Configuration

* Benchmark: BharatBBQ
* Framework: INSPECT AI (v0.3.235)
* Model: Llama 3.2 (Ollama)
* Dataset Mode: Development/Test Run
* Samples per Category: 5
* Categories Evaluated: 13
* Total Samples Evaluated: 65

---

## Overall Performance

| Metric                 | Score  |
| ---------------------- | ------ |
| Overall Accuracy       | 0.4154 |
| Ambiguous Accuracy     | 0.3590 |
| Disambiguated Accuracy | 0.5000 |

---

## Category-wise Accuracy

| Category            | Accuracy |
| ------------------- | -------- |
| Age                 | 0.40     |
| AgexGender          | 0.40     |
| Caste               | 1.00     |
| Disability_status   | 0.40     |
| Gender_identity     | 0.40     |
| GenderxReligion     | 0.60     |
| Nationality         | 0.20     |
| Physical_appearance | 0.40     |
| Region              | 0.60     |
| RegionxGender       | 0.60     |
| Religion            | 0.00     |
| SES                 | 0.00     |
| Sexual_orientation  | 0.40     |

---

## Observations

* Highest performing category: Caste (1.00)
* Strong performance observed in Region, RegionxGender, and GenderxReligion (0.60)
* Lowest performance observed in Religion (0.00) and SES (0.00)
* Performance on ambiguous contexts (0.3590) was lower than on disambiguated contexts (0.5000)
* Results indicate that additional evaluation on the full BharatBBQ dataset is required before drawing benchmark conclusions.

---

## Generated Artifacts

* bharatbbq_summary.csv
* bharatbbq_category_report.csv
* INSPECT evaluation log (.eval)

---

## Status

Current run is a reduced-size validation run using 5 samples per category to verify dataset ingestion, INSPECT evaluation, log parsing, metric computation, and report generation.

A full benchmark run should be executed using the complete BharatBBQ dataset before reporting final benchmark numbers.


inspect eval india_evals/module2/bias/task.py@bharatbbq --model ollama/llama3.2
python -m india_evals.module2.scripts.analyze_bharatbbq