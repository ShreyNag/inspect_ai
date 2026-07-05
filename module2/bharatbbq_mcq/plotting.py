from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------------
# Directories
# --------------------------------------------------------

MODULE_DIR = Path(__file__).resolve().parent

RESULTS_DIR = MODULE_DIR / "results"
PLOTS_DIR = MODULE_DIR / "plots"

PLOTS_DIR.mkdir(exist_ok=True)

# --------------------------------------------------------
# Load CSV files
# --------------------------------------------------------

summary = pd.read_csv(
    RESULTS_DIR / "bharatbbq_summary.csv"
)

category = pd.read_csv(
    RESULTS_DIR / "bharatbbq_category_report.csv"
)

# --------------------------------------------------------
# Plot 1 : Category Accuracy
# --------------------------------------------------------

plt.figure(figsize=(10,6))

plt.bar(
    category["category"],
    category["accuracy"]
)

plt.title("Accuracy by Bias Category")
plt.xlabel("Category")
plt.ylabel("Accuracy")
plt.xticks(rotation=45, ha="right")
plt.ylim(0,1)
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "category_accuracy.png",
    dpi=300
)

plt.close()

# --------------------------------------------------------
# Plot 2 : Overall Accuracy
# --------------------------------------------------------

overall = summary.loc[0, "overall_accuracy"]

plt.figure(figsize=(4,5))

plt.bar(
    ["Overall"],
    [overall]
)

plt.ylim(0,1)

plt.ylabel("Accuracy")

plt.title("Overall Accuracy")

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "overall_accuracy.png",
    dpi=300
)

plt.close()

# --------------------------------------------------------
# Plot 3 : Bias Metrics
# --------------------------------------------------------

metrics = [
    "bias_score_ambiguous",
    "bias_score_disambiguated",
    "stereotypical_bias_score_ambiguous",
    "stereotypical_bias_score_disambiguated"
]

values = [
    summary.loc[0, m]
    for m in metrics
]

labels = [
    "BSA",
    "BSD",
    "SBSA",
    "SBSD"
]

plt.figure(figsize=(8,5))

plt.bar(
    labels,
    values
)

plt.title("Bias Metrics")

plt.ylabel("Score")

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "bias_metrics.png",
    dpi=300
)

plt.close()

print("\nPlots saved successfully!\n")

for file in PLOTS_DIR.glob("*.png"):
    print(file)