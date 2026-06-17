import re

from inspect_ai.scorer import Score, scorer
from inspect_ai.scorer import Target


def normalize_answer(text: str) -> str:

    if not text:
        return ""

    text = str(text).strip().upper()

    match = re.search(r"\b([ABCD])\b", text)

    if match:
        return match.group(1)

    return ""


@scorer(metrics=[])
def mmlu_accuracy():

    async def score(state, target: Target):

        prediction = normalize_answer(
            state.output.completion
        )

        expected = normalize_answer(
            target.text
        )

        correct = prediction == expected

        return Score(
            value=1 if correct else 0,
            answer=prediction,
            explanation=f"predicted={prediction}, expected={expected}"
        )

    return score