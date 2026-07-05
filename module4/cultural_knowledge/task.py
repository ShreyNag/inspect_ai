import json
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate
from inspect_ai.scorer import includes


def build_prompt(item):

    question = item["Questions"]

    return f"""
Question:

{question}

Provide a detailed and accurate answer.
"""


def load_cultural_knowledge():

    samples = []

    module_dir = Path(__file__).resolve().parent.parent

    dataset_file = (
        module_dir
        / "datasets"
        / "Module4_rubric_dataset.json"
    )

    print(f"\nLoading dataset from:\n{dataset_file}\n")

    with open(
        dataset_file,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)
    #loading only 5    
    data = data[:5]    

    print(
        f"Loaded {len(data)} records from dataset"
    )

    for item in data:

        samples.append(

            Sample(

                input=build_prompt(item),

                target="",

                metadata={

                    "id":
                        item["Scenario Id"],

                    "domain":
                        item["Domain"],

                    "rubric":
                        item["rubric"]
                }
            )
        )

    print(
        f"Created {len(samples)} Inspect samples"
    )

    return samples


@task
def cultural_knowledge():

    return Task(

        dataset=load_cultural_knowledge(),

        solver=generate(),

        scorer=includes("")
    )