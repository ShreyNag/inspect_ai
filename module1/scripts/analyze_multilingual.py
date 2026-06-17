from inspect_ai.log import read_eval_log
from india_evals.module1.multilingual.metrics import normalize_answer

import pandas as pd


from pathlib import Path

log_files = sorted(
    Path("logs").glob("*multilingual*.eval"),
    key=lambda x: x.stat().st_mtime,
    reverse=True
)

LOG_FILE = str(log_files[0])

print(f"Using log: {LOG_FILE}")

log = read_eval_log(LOG_FILE)

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

print("\n===== Overall Accuracy =====")
print(round(overall_accuracy, 4))

print("\n===== Language Accuracy =====")
print(language_accuracy)

print("\n===== Subject Accuracy =====")
print(subject_accuracy)

summary_df = pd.DataFrame([{
    "overall_accuracy": round(overall_accuracy, 4)
}])

summary_df.to_csv(
    "multilingual_summary.csv",
    index=False
)

language_accuracy.to_csv(
    "multilingual_language_report.csv",
    index=False
)

subject_accuracy.to_csv(
    "multilingual_subject_report.csv",
    index=False
)

print("\nSaved:")
print(" - multilingual_summary.csv")
print(" - multilingual_language_report.csv")
print(" - multilingual_subject_report.csv")