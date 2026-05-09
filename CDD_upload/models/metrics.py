from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # safe for Colab / headless environments

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize


def get_predictions(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs inference on a dataloader and returns:
        y_true: true labels, shape (N,)
        y_pred: predicted labels, shape (N,)
        y_prob: predicted probabilities, shape (N, C)
    """
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    return y_true, y_pred, y_prob


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    average: str = "weighted",
) -> Dict[str, float]:
    """
    Calculates accuracy, precision, recall, F1-score, and optional AUC-ROC.
    """
    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=average,
        zero_division=0,
    )

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

    if y_prob is not None:
        try:
            num_classes = y_prob.shape[1]
            y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

            auc_roc = roc_auc_score(
                y_true_bin,
                y_prob,
                average=average,
                multi_class="ovr",
            )
            metrics["auc_roc"] = auc_roc

        except Exception as e:
            print(f"AUC-ROC could not be calculated: {e}")
            metrics["auc_roc"] = float("nan")

    return metrics


def print_metrics(metrics: Dict[str, float]) -> None:
    """
    Prints metrics in a readable format.
    """
    print("\nEvaluation Metrics")
    print("-" * 30)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> str:
    """
    Generates a detailed class-wise classification report.
    """
    return classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )


def save_classification_report(
    report: str,
    output_path: Path,
) -> None:
    """
    Saves the classification report as a text file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Classification report saved to: {output_path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_path: Optional[Path] = None,
    normalize: bool = False,
    figsize: Tuple[int, int] = (12, 10),
) -> None:
    """
    Plots and optionally saves a confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)

    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix saved to: {output_path}")

    plt.close()


def plot_training_curves(
    history: Dict[str, List[float]],
    output_dir: Optional[Path] = None,
    model_name: str = "model",
) -> None:
    """
    Plots training and validation loss/accuracy curves.

    Saves to:
        output_dir/model_name/loss_curve.png
        output_dir/model_name/accuracy_curve.png
    """
    if output_dir is not None:
        output_dir = output_dir / model_name
        output_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Training Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training and Validation Loss - {model_name}")
    plt.legend()
    plt.tight_layout()

    if output_dir:
        loss_path = output_dir / "loss_curve.png"
        plt.savefig(loss_path, dpi=300, bbox_inches="tight")
        print(f"Loss curve saved to: {loss_path}")

    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], label="Training Accuracy")
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Training and Validation Accuracy - {model_name}")
    plt.legend()
    plt.tight_layout()

    if output_dir:
        acc_path = output_dir / "accuracy_curve.png"
        plt.savefig(acc_path, dpi=300, bbox_inches="tight")
        print(f"Accuracy curve saved to: {acc_path}")

    plt.close()


def plot_multiclass_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> None:
    """
    Plots one-vs-rest ROC curves for multi-class classification.
    """
    num_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

    plt.figure(figsize=figsize)

    for class_idx in range(num_classes):
        try:
            fpr, tpr, _ = roc_curve(y_true_bin[:, class_idx], y_prob[:, class_idx])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{class_names[class_idx]} AUC={roc_auc:.2f}")
        except Exception:
            continue

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multi-Class ROC Curve")
    plt.legend(loc="lower right", fontsize=7)
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"ROC curve saved to: {output_path}")

    plt.close()


def save_metrics_csv(
    metrics: Dict[str, float],
    output_path: Path,
) -> None:
    """
    Saves metrics dictionary to a CSV file.
    """
    import pandas as pd

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([metrics])
    df.to_csv(output_path, index=False)

    print(f"Metrics saved to: {output_path}")


if __name__ == "__main__":
    print("metrics.py loaded successfully.")
    print("Use this module inside train.py and evaluate.py.")