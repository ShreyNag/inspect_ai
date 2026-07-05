from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------
# Directories
# -------------------------------------------------------

MODULE_DIR = Path(__file__).resolve().parent.parent

MULTILINGUAL_DIR = MODULE_DIR / "multilingual"

RESULTS_DIR = MULTILINGUAL_DIR / "results"
PLOTS_DIR = MULTILINGUAL_DIR / "plots"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------
# Load Results
# -------------------------------------------------------

df = pd.read_csv(
    RESULTS_DIR / "multilingual_language_report.csv"
)

df = df.sort_values("accuracy")

# -------------------------------------------------------
# Plot
# -------------------------------------------------------

plt.figure(figsize=(8,5))

bars = plt.barh(
    df["language"],
    df["accuracy"]
)

plt.xlabel("Accuracy")
plt.ylabel("Language")
plt.title("Language-wise Accuracy")

plt.xlim(0,1)

for bar in bars:
    width = bar.get_width()

    plt.text(
        width + 0.01,
        bar.get_y() + bar.get_height()/2,
        f"{width:.2f}",
        va="center",
        fontsize=10
    )

plt.tight_layout()

plot_path = PLOTS_DIR / "language_accuracy.png"

plt.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nPlot saved successfully!\n")
print(plot_path)