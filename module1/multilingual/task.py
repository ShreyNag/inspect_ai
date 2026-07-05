from pathlib import Path
import pandas as pd

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate

from metrics import mmlu_accuracy


def load_samples():

    module_dir = Path(__file__).resolve().parent.parent

    csv_path = (
        module_dir
        / "datasets"
        / "mmlu_translated.csv"
    )

    df = pd.read_csv(csv_path)

    samples = []

    for _, row in df.head(14).iterrows():
    #for _, row in df.iterrows():

        prompt = f"""
Answer the following multiple-choice question.

Question:
{row['question']}

A. {row['A']}
B. {row['B']}
C. {row['C']}
D. {row['D']}

Respond with ONLY one capital letter.

Valid responses:
A
B
C
D

Do not explain your answer.
Do not write any additional text.
"""

        samples.append(
            Sample(
                input=prompt,
                target=row["answer_letter"],
                metadata={
                    "language": row["language"],
                    "subject": row["subject"],
                    "question_id": int(row["id"])
                }
            )
        )

    return samples


@task
def multilingual():

    return Task(
        dataset=load_samples(),
        solver=generate(),
        scorer=mmlu_accuracy()
    )