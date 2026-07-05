import json
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate
from inspect_ai.scorer import includes


def build_prompt(item):

    context = item.get("context", "")

    question = item["question"]

    return f"""
Context:
{context}

Question:
{question}

Provide a thoughtful answer.
"""


def load_bharatbbq_reasoning():

    samples = []

    module_dir = Path(__file__).resolve().parent.parent

    dataset_file = (
    module_dir
    / "datasets"
    / "bharatbbq_reasoning"
    / "260_ques.json"
)

    with open(
        dataset_file,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)
    #reducing data size    
    data = data[:5]    

    for item in data:

        samples.append(
            Sample(
                input=build_prompt(item),

                target="",

                metadata={
                    "id":
                        item["id"],

                    "category":
                        item["category"],

                    "subcategory":
                        item["subcategory"],

                    "rubric":
                        item["rubric"],

                    "target_group":
                        item["metadata"].get(
                            "target_group"
                        ),

                    "question_type":
                        item["metadata"].get(
                            "question_type"
                        ),

                    "evaluation_type":
                        item["metadata"].get(
                            "evaluation_type"
                        ),

                    "intersectional":
                        item["metadata"].get(
                            "intersectional"
                        )
                }
            )
        )

    print(
        f"Loaded {len(samples)} samples"
    )

    return samples


@task
def bharatbbq_reasoning():

    return Task(
        dataset=load_bharatbbq_reasoning(),
        solver=generate(),
        scorer=includes("")
    )