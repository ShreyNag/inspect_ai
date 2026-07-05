from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------
# Directories
# -------------------------------------------------------

MODULE_DIR = Path(__file__).resolve().parent

RESULTS_DIR = MODULE_DIR / "results"
PLOTS_DIR = MODULE_DIR / "plots"

PLOTS_DIR.mkdir(exist_ok=True)

# -------------------------------------------------------
# Load Results
# -------------------------------------------------------

summary = pd.read_csv(
    RESULTS_DIR / "bharatbbq_reasoning_summary.csv"
)

category = pd.read_csv(
    RESULTS_DIR / "bharatbbq_reasoning_category_report.csv"
)

subcategory = pd.read_csv(
    RESULTS_DIR / "bharatbbq_reasoning_subcategory_report.csv"
)

# -------------------------------------------------------
# Plot 1 : Overall Score
# -------------------------------------------------------

overall_score = summary.loc[0, "overall_score"]

plt.figure(figsize=(4, 5))

bars = plt.bar(
    ["Overall"],
    [overall_score]
)

plt.ylim(0, 1)

plt.ylabel("Score")
plt.title("Overall BharatBBQ Reasoning Score")

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.02,
        f"{height:.2f}",
        ha="center",
        fontsize=11
    )

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "overall_score.png",
    dpi=300
)

plt.close()

# -------------------------------------------------------
# Plot 2 : Category Scores
# -------------------------------------------------------

category = category.sort_values(
    by="score",
    ascending=True
)

plt.figure(figsize=(9, 6))

bars = plt.barh(
    category["category"],
    category["score"]
)

plt.xlim(0, 1)

plt.xlabel("Average Score")
plt.title("Average Score by Category")

for bar in bars:
    width = bar.get_width()
    plt.text(
        width + 0.01,
        bar.get_y() + bar.get_height()/2,
        f"{width:.2f}",
        va="center",
        fontsize=9
    )

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "category_scores.png",
    dpi=300
)

plt.close()

# -------------------------------------------------------
# Plot 3 : Top Subcategories
# -------------------------------------------------------

subcategory = subcategory.sort_values(
    by="score",
    ascending=False
)

plt.figure(figsize=(12, 7))

bars = plt.bar(
    subcategory["subcategory"],
    subcategory["score"]
)

plt.ylim(0, 1)

plt.ylabel("Average Score")
plt.title("Average Score by Subcategory")

plt.xticks(
    rotation=60,
    ha="right"
)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.015,
        f"{height:.2f}",
        ha="center",
        fontsize=8
    )

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "subcategory_scores.png",
    dpi=300
)

plt.close()

print("\nPlots saved successfully!\n")

for plot in sorted(PLOTS_DIR.glob("*.png")):
    print(plot)