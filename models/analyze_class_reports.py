from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = BASE_DIR / "outputs" / "logs"
OUTPUT_FILE = BASE_DIR / "outputs" / "class_report_summary.csv"


def parse_classification_report(report_path: Path, model_name: str):
    rows = []

    with open(report_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        if not line:
            continue

        if line.startswith("precision") or line.startswith("accuracy"):
            continue

        if line.startswith("macro avg") or line.startswith("weighted avg"):
            continue

        parts = line.split()

        if len(parts) < 5:
            continue

        try:
            support = int(parts[-1])
            f1_score = float(parts[-2])
            recall = float(parts[-3])
            precision = float(parts[-4])
            class_name = " ".join(parts[:-4])

            rows.append({
                "model": model_name,
                "class_name": class_name,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "support": support,
            })
        except ValueError:
            continue

    return rows


def main():
    all_rows = []

    if not LOGS_DIR.exists():
        print(f"Logs directory not found: {LOGS_DIR}")
        return

    for model_dir in sorted([d for d in LOGS_DIR.iterdir() if d.is_dir()]):
        report_path = model_dir / "classification_report.txt"

        if not report_path.exists():
            print(f"Skipping {model_dir.name}: classification_report.txt not found")
            continue

        rows = parse_classification_report(report_path, model_dir.name)
        all_rows.extend(rows)

    if not all_rows:
        print("No class-wise classification reports found.")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved full class-wise summary to: {OUTPUT_FILE}\n")

    print("Best and worst classes by model:\n")
    for model_name, group in df.groupby("model"):
        best_row = group.sort_values(by="f1_score", ascending=False).iloc[0]
        worst_row = group.sort_values(by="f1_score", ascending=True).iloc[0]

        print(f"Model: {model_name}")
        print(
            f"  Best class : {best_row['class_name']} "
            f"(F1={best_row['f1_score']:.4f}, "
            f"Precision={best_row['precision']:.4f}, "
            f"Recall={best_row['recall']:.4f}, "
            f"Support={int(best_row['support'])})"
        )
        print(
            f"  Worst class: {worst_row['class_name']} "
            f"(F1={worst_row['f1_score']:.4f}, "
            f"Precision={worst_row['precision']:.4f}, "
            f"Recall={worst_row['recall']:.4f}, "
            f"Support={int(worst_row['support'])})"
        )
        print()


if __name__ == "__main__":
    main()