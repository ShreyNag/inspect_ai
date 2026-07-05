import pandas as pd


def overall_score(df):

    return round(
        df["score"].mean(),
        4
    )


def category_scores(df):

    return (
        df.groupby("category")
        .agg(
            score=("score", "mean"),
            samples=("score", "count")
        )
        .reset_index()
    )


def subcategory_scores(df):

    return (
        df.groupby(
            ["category", "subcategory"]
        )
        .agg(
            score=("score", "mean"),
            samples=("score", "count")
        )
        .reset_index()
    )


def rubric_scores(criteria_rows):

    df = pd.DataFrame(
        criteria_rows
    )

    return (
        df.groupby("criterion")
        .agg(
            score=("passed", "mean"),
            count=("passed", "count")
        )
        .reset_index()
    )