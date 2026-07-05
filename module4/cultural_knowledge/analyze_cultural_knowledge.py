import json
import asyncio
import pandas as pd

from pathlib import Path

from inspect_ai.log import read_eval_log
from inspect_ai.model import get_model

from metrics import (
    overall_score,
    domain_scores
)


LOG_DIR = (
    Path(__file__).resolve().parent
    / "logs"
)


def latest_log():

    logs = sorted(
        LOG_DIR.glob("*.eval"),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )

    if not logs:

        raise FileNotFoundError(
            f"No evaluation logs found in {LOG_DIR}"
        )

    latest = logs[0]

    print("\n===================================")
    print("USING LOG FILE")
    print("===================================")
    print(latest)
    print("===================================\n")

    return str(latest)


def build_judge_prompt(
    question,
    answer,
    rubric
):

    rubric_text = ""

    for i, criterion in enumerate(
        rubric,
        start=1
    ):
        rubric_text += (
            f"{i}. {criterion}\n"
        )

    return f"""
You are an evaluation engine.

Question:

{question}

Model Answer:

{answer}

Rubric:

{rubric_text}

Evaluate the answer against EVERY rubric criterion.

For each criterion determine whether the answer satisfies it.

Return ONLY valid JSON.

Required format:

{{
  "results": [
    {{
      "criterion": "criterion text",
      "passed": true
    }}
  ]
}}

Rules:
- Return valid JSON only
- No markdown
- No code fences
- No explanations
- One result per rubric criterion
"""


async def judge_sample(
    judge,
    sample
):

    prompt = build_judge_prompt(
        sample.input,
        sample.output.completion,
        sample.metadata["rubric"]
    )

    response = await judge.generate(
        prompt
    )

    text = response.completion.strip()

    try:

        result = json.loads(text)

    except Exception:

        print(
            f"\nJSON Parse Failed for "
            f"{sample.metadata['id']}"
        )

        print("\nJudge Output:")
        print(text)

        return None

    if "results" not in result:

        print(
            f"\nMissing results key for "
            f"{sample.metadata['id']}"
        )

        return None

    results = result["results"]

    if len(results) == 0:

        return None

    passed = sum(
        1
        for r in results
        if r.get("passed", False)
    )

    score = passed / len(results)

    return {
        "sample_score": score,
        "results": results
    }


async def main():

    LOG_FILE = latest_log()

    log = read_eval_log(
        LOG_FILE
    )

    print(
        f"Loaded {len(log.samples)} samples\n"
    )

    judge = get_model(
        "ollama/llama3.2"
    )

    print(
        "Judge model loaded: ollama/llama3.2\n"
    )

    rows = []

    criteria_rows = []

    total_samples = len(log.samples)

    for idx, sample in enumerate(
        log.samples,
        start=1
    ):

        print(
            f"[{idx}/{total_samples}] "
            f"{sample.metadata['id']} | "
            f"{sample.metadata['domain']}"
        )

        judged = await judge_sample(
            judge,
            sample
        )

        if judged is None:
            continue

        rows.append({

            "id":
                sample.metadata["id"],

            "domain":
                sample.metadata["domain"],

            "score":
                judged["sample_score"]
        })

        for result in judged["results"]:

            passed_val = result.get(
                "passed"
            )

            if passed_val is None:
                passed_val = False

            criteria_rows.append({

                "id":
                    sample.metadata["id"],

                "domain":
                    sample.metadata["domain"],

                "criterion":
                    result.get(
                        "criterion",
                        "Unknown Criterion"
                    ),

                "passed":
                    int(passed_val)
            })

    if len(rows) == 0:

        print(
            "\nNo valid judged samples found."
        )

        return

    df = pd.DataFrame(rows)

    overall = overall_score(df)

    domain_report = domain_scores(df)

    results_dir = (
        Path(__file__).resolve().parent
        / "results"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    summary_df = pd.DataFrame([{
        "overall_score":
            round(overall, 4),

        "samples":
            len(df)
    }])

    summary_df.to_csv(
        results_dir
        / "cultural_knowledge_summary.csv",
        index=False
    )

    domain_report.to_csv(
        results_dir
        / "cultural_knowledge_domain_report.csv",
        index=False
    )

    df.to_csv(
        results_dir
        / "cultural_knowledge_sample_report.csv",
        index=False
    )

    print("\n===================================")
    print(
        f"Evaluation Complete! Overall Score: {overall:.4f}"
    )
    print(
        f"All reports successfully generated inside: {results_dir}"
    )
    print("===================================\n")


if __name__ == "__main__":

    asyncio.run(main())