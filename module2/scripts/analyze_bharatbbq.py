from inspect_ai.log import read_eval_log
from india_evals.module2.bias.metrics import compute_all_bias_metrics
import pandas as pd


LOG_FILE = "logs/2026-06-15T16-24-48-00-00_bharatbbq_g8AsfBZhKUn9ydToRnqJvz.eval"

LETTER_TO_OPTION = {
    "A": 1,
    "B": 2,
    "C": 3
}


log = read_eval_log(LOG_FILE)

rows = []

for sample in log.samples:

    prediction_text = sample.output.completion.strip().upper()

    prediction_letter = None

    for letter in ["A", "B", "C"]:
        if prediction_text.startswith(letter):
            prediction_letter = letter
            break

    if prediction_letter is None:
        prediction = -1
    else:
        prediction = LETTER_TO_OPTION[prediction_letter]

    target_letter = (
        str(sample.target)
        .strip()
        .replace(".", "")
        .upper()
    )

    rows.append({
        "category": sample.metadata["category"],
        "context_type": sample.metadata["context_type"],
        "label": sample.metadata["label"],
        "target_numeric": sample.metadata["target_numeric"],
        "pairing": sample.metadata["pairing"],
        "question_polarity": sample.metadata["question_polarity"],
        "qid": sample.metadata["qid"],
        "prediction": prediction,
        "target": LETTER_TO_OPTION[target_letter]
    })

df = pd.DataFrame(rows)

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

summary_df = pd.DataFrame([{
    "overall_accuracy": round(overall_accuracy, 4),
    "ambiguous_accuracy": round(amb_accuracy, 4),
    "disambiguated_accuracy": round(disamb_accuracy, 4),
    "bias_score_ambiguous": bias_metrics["BSA"],
    "bias_score_disambiguated": bias_metrics["BSD"],
    "stereotypical_bias_score_ambiguous": bias_metrics["SBSA"],
    "stereotypical_bias_score_disambiguated": bias_metrics["SBSD"]
}])

summary_df.to_csv(
    "bharatbbq_summary.csv",
    index=False
)

category_accuracy.to_csv(
    "bharatbbq_category_report.csv",
    index=False
)

print("\nSaved:")
print(" - bharatbbq_summary.csv")
print(" - bharatbbq_category_report.csv")