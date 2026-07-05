import pandas as pd


def overall_score(df):

    return round(
        df["score"].mean(),
        4
    )


def domain_scores(df):

    return (
        df.groupby("domain")
        .agg(
            score=("score", "mean"),
            samples=("score", "count")
        )
        .reset_index()
    )

