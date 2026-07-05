import pandas as pd
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate
from inspect_ai.scorer import match
import os
from pathlib import Path

TARGET_MAP = {
    1: "A",
    2: "B",
    3: "C"
}


def build_prompt(row):

    return f"""
Context:
{row['Context']}

Question:
{row['Question']}

Options:

A. {row['Option1']}
B. {row['Option2']}
C. {row['Option3']}

Answer with only A, B, or C.
"""


def load_bharatbbq():

    samples = []

    # project root = meity/
    module_dir = Path(__file__).resolve().parent.parent

    dataset_dir = (
        module_dir
        / "datasets"
        / "bharatbbqmcq"
    )

    print(f"Dataset directory: {dataset_dir}")
    print(f"Exists: {dataset_dir.exists()}")

    csv_files = list(dataset_dir.glob("*.csv"))

    print(f"CSV files found: {len(csv_files)}")

    for csv_file in csv_files:

        print(f"Loading {csv_file.name}")

        df = pd.read_csv(csv_file)

        # Development run
        df = df.head(5)

        # Full benchmark later:
        # df = df

        for _, row in df.iterrows():

            samples.append(
                Sample(
                    input=build_prompt(row),

                    target=TARGET_MAP[int(row["Target"])],

                    metadata={
                        "category": row.get("Category"),
                        "context_type": row.get("Context_type"),
                        "source_file": csv_file.name,

                        "label": row.get("Label"),
                        "target_numeric": row.get("Target"),
                        "pairing": row.get("Pairing"),
                        "question_polarity": row.get("Question_polarity"),

                        "qid": row.get("Qid"),
                        "proper_noun": row.get("Proper_Noun")
                    }
                )
            )

    print(f"Total samples loaded: {len(samples)}")

    return samples

# ------------------------------------------------------------------
# Store Inspect AI logs inside module2/bharatbbq_mcq/logs
# ------------------------------------------------------------------

MODULE_DIR = Path(__file__).resolve().parent

LOG_DIR = MODULE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

os.environ["INSPECT_LOG_DIR"] = str(LOG_DIR)

@task
def bharatbbq():

    return Task(
        dataset=load_bharatbbq(),
        solver=generate(),
        scorer=match()
    )