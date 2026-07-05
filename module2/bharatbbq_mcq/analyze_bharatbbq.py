from inspect_ai.log import read_eval_log
from india_evals.module2.bharatbbq_mcq.metrics import compute_all_bias_metrics
import pandas as pd
from pathlib import Path
import sys


# ------------------------------------------------------------------
# Directories
# ------------------------------------------------------------------

MODULE_DIR = Path(__file__).resolve().parent.parent

BHARATBBQ_DIR = MODULE_DIR / "bharatbbq_mcq"

LOG_DIR = BHARATBBQ_DIR / "logs"
RESULTS_DIR = BHARATBBQ_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Load newest evaluation log
# ------------------------------------------------------------------

log_files = sorted(
    LOG_DIR.glob("*bharatbbq*.eval"),
    key=lambda x: x.stat().st_mtime,
    reverse=True
)

if not log_files:
    raise FileNotFoundError(
        f"No BharatBBQ evaluation log files found in:\n{LOG_DIR}"
    )

LOG_FILE = log_files[0]

print(f"Using log:\n{LOG_FILE}")

LETTER_TO_OPTION = {
    "A": 1,
    "B": 2,
    "C": 3
}

log = read_eval_log(str(LOG_FILE))

rows = []

samples = getattr(log, "samples", []) or []

for sample in samples:

    # ---------------- Prediction ----------------

    prediction_text = sample.output.completion.strip().upper()

    prediction_letter = None

    for letter in ["A", "B", "C"]:
        if prediction_text.startswith(letter):
            prediction_letter = letter
            break

    prediction = (
        LETTER_TO_OPTION[prediction_letter]
        if prediction_letter
        else -1
    )

    # ---------------- Target ----------------

    target_clean = str(sample.target).strip().upper()

    target_letter = None

    for letter in ["A", "B", "C"]:
        if letter in target_clean:
            target_letter = letter
            break

    target_val = (
        LETTER_TO_OPTION[target_letter]
        if target_letter
        else -1
    )

    rows.append({
        "category": sample.metadata.get("category", "unknown"),
        "context_type": sample.metadata.get("context_type", "unknown"),
        "label": sample.metadata.get("label", "unknown"),
        "target_numeric": sample.metadata.get("target_numeric"),
        "pairing": sample.metadata.get("pairing"),
        "question_polarity": sample.metadata.get("question_polarity"),
        "qid": sample.metadata.get("qid"),
        "prediction": prediction,
        "target": target_val
    })

print(f"Total samples parsed from log: {len(rows)}")

if not rows:
    print("\n[ERROR] Zero samples found inside the evaluation log.")
    sys.exit(1)

df = pd.DataFrame(rows)

# ------------------------------------------------------------------
# Accuracy
# ------------------------------------------------------------------

df["correct"] = (
    df["prediction"] == df["target"]
).astype(int)

overall_accuracy = df["correct"].mean()

amb_accuracy = (
    df[df["context_type"] == "amb"]["correct"]
    .mean()
)

disamb_accuracy = (
    df[df["context_type"] == "disamb"]["correct"]
    .mean()
)

category_accuracy = (
    df.groupby("category")
      .agg(
          accuracy=("correct", "mean"),
          samples=("correct", "count")
      )
      .reset_index()
)

# ------------------------------------------------------------------
# Bias Metrics
# ------------------------------------------------------------------

bias_metrics = compute_all_bias_metrics(df)

print("\n===== BharatBBQ Summary =====")
print(f"Overall Accuracy: {overall_accuracy:.4f}")
print(f"Ambiguous Accuracy: {amb_accuracy:.4f}")
print(f"Disambiguated Accuracy: {disamb_accuracy:.4f}")

print("\n===== Bias Metrics =====")

for metric, value in bias_metrics.items():
    print(f"{metric}: {value}")

print("\n===== Category Accuracy =====")

print(category_accuracy.to_string(index=False))

# ------------------------------------------------------------------
# Save Results
# ------------------------------------------------------------------

summary_df = pd.DataFrame([{
    "overall_accuracy": round(overall_accuracy, 4),
    "ambiguous_accuracy": round(amb_accuracy, 4),
    "disambiguated_accuracy": round(disamb_accuracy, 4),
    "bias_score_ambiguous": bias_metrics.get("BSA", 0),
    "bias_score_disambiguated": bias_metrics.get("BSD", 0),
    "stereotypical_bias_score_ambiguous": bias_metrics.get("SBSA", 0),
    "stereotypical_bias_score_disambiguated": bias_metrics.get("SBSD", 0)
}])

summary_file = RESULTS_DIR / "bharatbbq_summary.csv"
category_file = RESULTS_DIR / "bharatbbq_category_report.csv"

summary_df.to_csv(
    summary_file,
    index=False
)

category_accuracy.to_csv(
    category_file,
    index=False
)

print("\nSaved Successfully!")

print(summary_file)
print(category_file)