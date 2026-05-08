import argparse
import sys
from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from data_scripts.preprocessing import create_datasets, create_dataloaders
from models.model_factory import create_model
from models.metrics import (
    get_predictions,
    calculate_metrics,
    print_metrics,
    generate_classification_report,
    save_classification_report,
    save_metrics_csv,
    plot_confusion_matrix,
    plot_multiclass_roc_curve,
)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def evaluate_model(
    model_name: str,
    checkpoint_path: Path,
    batch_size: int,
    dropout: float,
):
    device = get_device()
    print(f"Using device: {device}")

    train_dir = BASE_DIR / "data" / "train"
    val_dir = BASE_DIR / "data" / "val"
    test_dir = BASE_DIR / "data" / "test"

    output_dir = BASE_DIR / "outputs"
    figures_dir = output_dir / "figures"
    logs_dir = output_dir / "logs"

    figures_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    print("Creating datasets...")
    train_dataset, val_dataset, test_dataset = create_datasets(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
    )

    print("Creating dataloaders...")
    _, _, test_loader = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
    )

    checkpoint = load_checkpoint(checkpoint_path, device)

    checkpoint_model_name = checkpoint.get("model_name", model_name)
    num_classes = checkpoint.get("num_classes", len(test_dataset.classes))
    class_names = checkpoint.get("class_names", test_dataset.classes)
    checkpoint_dropout = checkpoint.get("dropout", dropout)

    print(f"Checkpoint model: {checkpoint_model_name}")
    print(f"Number of classes: {num_classes}")

    model = create_model(
        model_name=checkpoint_model_name,
        num_classes=num_classes,
        dropout=checkpoint_dropout,
        pretrained=False,
        freeze_features=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    print("Running inference on test set...")
    y_true, y_pred, y_prob = get_predictions(
        model=model,
        dataloader=test_loader,
        device=device,
    )

    metrics = calculate_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        average="weighted",
    )

    print_metrics(metrics)

    metrics_path = logs_dir / f"{checkpoint_model_name}_test_metrics.csv"
    save_metrics_csv(metrics, metrics_path)

    report = generate_classification_report(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
    )

    print("\nClassification Report")
    print("-" * 50)
    print(report)

    report_path = logs_dir / f"{checkpoint_model_name}_classification_report.txt"
    save_classification_report(report, report_path)

    confusion_matrix_path = figures_dir / f"{checkpoint_model_name}_confusion_matrix.png"
    plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        output_path=confusion_matrix_path,
        normalize=False,
    )

    normalized_confusion_matrix_path = figures_dir / f"{checkpoint_model_name}_confusion_matrix_normalized.png"
    plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        output_path=normalized_confusion_matrix_path,
        normalize=True,
    )

    roc_path = figures_dir / f"{checkpoint_model_name}_roc_curve.png"
    plot_multiclass_roc_curve(
        y_true=y_true,
        y_prob=y_prob,
        class_names=class_names,
        output_path=roc_path,
    )

    print("\nEvaluation complete.")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Classification report saved to: {report_path}")
    print(f"Figures saved to: {figures_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate trained crop disease detection model."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="efficientnet_b0",
        choices=[
            "mobilenetv2",
            "resnet18",
            "efficientnet_b0",
            "resnet50",
            "efficientnet_b3",
            "densenet201",
        ],
        help="Model architecture.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/weights/efficientnet_b0_best.pth",
        help="Path to saved model checkpoint.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation.",
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout value used in the model head.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    checkpoint_path = BASE_DIR / args.checkpoint

    evaluate_model(
        model_name=args.model,
        checkpoint_path=checkpoint_path,
        batch_size=args.batch_size,
        dropout=args.dropout,
    )


if __name__ == "__main__":
    main()