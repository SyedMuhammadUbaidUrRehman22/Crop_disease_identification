from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = BASE_DIR / "outputs" / "logs"
OUTPUT_FILE = BASE_DIR / "outputs" / "final_model_comparison.csv"


def main():
    rows = []

    if not LOGS_DIR.exists():
        print(f"Logs directory not found: {LOGS_DIR}")
        return

    for model_dir in sorted([d for d in LOGS_DIR.iterdir() if d.is_dir()]):
        metrics_file = model_dir / "test_metrics.csv"

        if not metrics_file.exists():
            print(f"Skipping {model_dir.name}: test_metrics.csv not found")
            continue

        try:
            df = pd.read_csv(metrics_file)
            if df.empty:
                print(f"Skipping {model_dir.name}: metrics file is empty")
                continue

            record = df.iloc[0].to_dict()
            record["model"] = model_dir.name
            rows.append(record)

        except Exception as e:
            print(f"Error reading {metrics_file}: {e}")

    if not rows:
        print("No model metrics found.")
        return

    results_df = pd.DataFrame(rows)

    desired_order = ["model", "accuracy", "precision", "recall", "f1_score", "auc_roc"]
    existing_cols = [c for c in desired_order if c in results_df.columns]
    remaining_cols = [c for c in results_df.columns if c not in existing_cols]
    results_df = results_df[existing_cols + remaining_cols]

    if "accuracy" in results_df.columns:
        results_df = results_df.sort_values(by="accuracy", ascending=False)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_FILE, index=False)

    print("\nFinal Comparison Table:")
    print(results_df.to_string(index=False))

    print(f"\nSaved comparison CSV to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()