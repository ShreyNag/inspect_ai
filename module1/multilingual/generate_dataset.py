from pathlib import Path
import pandas as pd

MODULE_DIR = Path(__file__).resolve().parent.parent

input_file = MODULE_DIR / "datasets" / "mmlu_dev.csv"
output_file = MODULE_DIR / "datasets" / "mmlu_multilingual.csv"

df = pd.read_csv(input_file)

languages = [
    "en",
    "hi",
    "ta",
    "te",
    "bn",
    "mr",
    "gu"
]

rows = []

for idx, row in df.iterrows():

    sample_id = idx + 1

    for lang in languages:

        rows.append({
            "id": sample_id,
            "language": lang,
            "subject": row["subject"],
            "question": row["question"],
            "A": row["A"],
            "B": row["B"],
            "C": row["C"],
            "D": row["D"],
            "answer_letter": row["answer_letter"]
        })

multi_df = pd.DataFrame(rows)

multi_df.to_csv(output_file, index=False)

print(f"Saved: {output_file}")
print(f"Total rows: {len(multi_df)}")
print(multi_df.head())
print(multi_df.head(10))