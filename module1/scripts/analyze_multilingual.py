from inspect_ai.log import read_eval_log
from india_evals.module1.multilingual.metrics import normalize_answer

import pandas as pd
from pathlib import Path

# ------------------------------------------------------------------
# Directories
# ------------------------------------------------------------------

MODULE_DIR = Path(__file__).resolve().parent.parent

MULTILINGUAL_DIR = MODULE_DIR / "multilingual"

LOG_DIR = MULTILINGUAL_DIR / "logs"
RESULTS_DIR = MULTILINGUAL_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Find newest multilingual log
# ------------------------------------------------------------------

log_files = sorted(
    LOG_DIR.glob("*multilingual*.eval"),
    key=lambda x: x.stat().st_mtime,
    reverse=True
)

if not log_files:
    raise FileNotFoundError(
        f"No multilingual evaluation logs found in:\n{LOG_DIR}"
    )

LOG_FILE = log_files[0]

print(f"Using log:\n{LOG_FILE}")

# ------------------------------------------------------------------
# Read evaluation log
# ------------------------------------------------------------------

log = read_eval_log(str(LOG_FILE))

rows = []

for sample in log.samples:

    prediction = normalize_answer(
        sample.output.completion
    )

    target = normalize_answer(
        str(sample.target)
    )

    rows.append({
        "language": sample.metadata["language"],
        "subject": sample.metadata["subject"],
        "question_id": sample.metadata["question_id"],
        "prediction": prediction,
        "target": target
    })

df = pd.DataFrame(rows)

# ------------------------------------------------------------------
# Accuracy
# ------------------------------------------------------------------

df["correct"] = (
    df["prediction"] == df["target"]
).astype(int)

overall_accuracy = df["correct"].mean()

language_accuracy = (
    df.groupby("language")
      .agg(
          accuracy=("correct", "mean"),
          samples=("correct", "count")
      )
      .reset_index()
)

subject_accuracy = (
    df.groupby("subject")
      .agg(
          accuracy=("correct", "mean"),
          samples=("correct", "count")
      )
      .reset_index()
)

# ------------------------------------------------------------------
# Print Summary
# ------------------------------------------------------------------

print("\n===== Overall Accuracy =====")
print(round(overall_accuracy, 4))

print("\n===== Language Accuracy =====")
print(language_accuracy)

print("\n===== Subject Accuracy =====")
print(subject_accuracy)

# ------------------------------------------------------------------
# Save Results
# ------------------------------------------------------------------

summary_df = pd.DataFrame([{
    "overall_accuracy": round(overall_accuracy, 4)
}])

summary_file = RESULTS_DIR / "multilingual_summary.csv"
language_file = RESULTS_DIR / "multilingual_language_report.csv"
subject_file = RESULTS_DIR / "multilingual_subject_report.csv"

summary_df.to_csv(
    summary_file,
    index=False
)

language_accuracy.to_csv(
    language_file,
    index=False
)

subject_accuracy.to_csv(
    subject_file,
    index=False
)

print("\nSaved Successfully!")

print(summary_file)
print(language_file)
print(subject_file)