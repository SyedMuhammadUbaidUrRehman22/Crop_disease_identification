from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "outputs" / "final_model_comparison.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_PNG = OUTPUT_DIR / "model_comparison.png"


def main():
    if not INPUT_FILE.exists():
        print(f"Input file not found: {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE)

    required_cols = ["model", "accuracy", "precision", "recall", "f1_score", "auc_roc"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing required columns: {missing_cols}")
        return

    # Sort by F1-score for cleaner presentation
    df = df.sort_values(by="f1_score", ascending=False).reset_index(drop=True)

    models = df["model"].tolist()
    x = range(len(models))
    bar_width = 0.16

    plt.figure(figsize=(12, 7))

    plt.bar([i - 2 * bar_width for i in x], df["accuracy"], width=bar_width, label="Accuracy")
    plt.bar([i - 1 * bar_width for i in x], df["precision"], width=bar_width, label="Precision")
    plt.bar([i for i in x], df["recall"], width=bar_width, label="Recall")
    plt.bar([i + 1 * bar_width for i in x], df["f1_score"], width=bar_width, label="F1-score")
    plt.bar([i + 2 * bar_width for i in x], df["auc_roc"], width=bar_width, label="AUC-ROC")

    plt.xticks(list(x), models, rotation=30, ha="right")
    plt.ylim(0.0, 1.05)
    plt.ylabel("Score")
    plt.xlabel("Model")
    plt.title("Comparative Performance of Trained Models")
    plt.legend()
    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close()

    print("Model comparison plot saved to:")
    print(OUTPUT_PNG)


if __name__ == "__main__":
    main()