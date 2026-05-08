import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Allow imports from project root
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from data_scripts.preprocessing import create_datasets, create_dataloaders
from models.model_factory import create_model
from models.metrics import plot_training_curves


def get_device() -> torch.device:
    """
    Selects GPU if available, otherwise CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_class_weights(train_dataset) -> torch.Tensor:
    """
    Computes inverse-frequency class weights for imbalanced datasets.

    This gives higher importance to underrepresented classes during training.
    """

    class_counts = [0] * len(train_dataset.classes)

    for _, label in train_dataset.samples:
        class_counts[label] += 1

    total_samples = sum(class_counts)
    num_classes = len(class_counts)

    weights = []
    for count in class_counts:
        if count == 0:
            weights.append(0.0)
        else:
            weights.append(total_samples / (num_classes * count))

    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion,
    optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Trains the model for one epoch.
    """

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        progress_bar.set_postfix({
            "loss": loss.item(),
            "acc": correct / total if total > 0 else 0,
        })

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def validate_one_epoch(
    model: nn.Module,
    dataloader,
    criterion,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluates the model on validation data.
    """

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc="Validation", leave=False)

    with torch.no_grad():
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            progress_bar.set_postfix({
                "loss": loss.item(),
                "acc": correct / total if total > 0 else 0,
            })

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def save_history_csv(history: Dict[str, List[float]], output_path: Path) -> None:
    """
    Saves training history to CSV.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])

        for i in range(len(history["train_loss"])):
            writer.writerow([
                i + 1,
                history["train_loss"][i],
                history["val_loss"][i],
                history["train_acc"][i],
                history["val_acc"][i],
            ])

    print(f"Training history saved to: {output_path}")


def save_class_mapping(class_names: List[str], output_path: Path) -> None:
    """
    Saves class index mapping.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for idx, class_name in enumerate(class_names):
            f.write(f"{idx},{class_name}\n")

    print(f"Class mapping saved to: {output_path}")


def train_model(
    model_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    dropout: float,
    weight_decay: float,
    pretrained: bool,
    freeze_features: bool,
    use_class_weights: bool,
) -> None:
    """
    Full training pipeline.
    """

    device = get_device()
    print(f"Using device: {device}")

    train_dir = BASE_DIR / "data" / "train"
    val_dir = BASE_DIR / "data" / "val"
    test_dir = BASE_DIR / "data" / "test"

    output_dir = BASE_DIR / "outputs"
    weights_dir = output_dir / "weights"
    figures_dir = output_dir / "figures"
    logs_dir = output_dir / "logs"

    weights_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    print("Creating datasets...")
    train_dataset, val_dataset, test_dataset = create_datasets(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
    )

    print("Creating dataloaders...")
    train_loader, val_loader, _ = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
    )

    class_names = train_dataset.classes
    num_classes = len(class_names)

    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    save_class_mapping(
        class_names,
        logs_dir / f"{model_name}_class_mapping.csv",
    )

    print(f"Creating model: {model_name}")
    model = create_model(
        model_name=model_name,
        num_classes=num_classes,
        dropout=dropout,
        pretrained=pretrained,
        freeze_features=freeze_features,
    )

    model = model.to(device)

    if use_class_weights:
        print("Using inverse-frequency class weights.")
        class_weights = compute_class_weights(train_dataset).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_val_acc = 0.0
    best_model_path = weights_dir / f"{model_name}_best.pth"

    print("\nStarting training...\n")

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        print("-" * 40)

        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_acc = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            checkpoint = {
                "model_name": model_name,
                "model_state_dict": model.state_dict(),
                "num_classes": num_classes,
                "class_names": class_names,
                "dropout": dropout,
                "pretrained": pretrained,
                "freeze_features": freeze_features,
                "best_val_acc": best_val_acc,
            }

            torch.save(checkpoint, best_model_path)
            print(f"Saved best model to: {best_model_path}")

        print()

    history_path = logs_dir / f"{model_name}_training_history.csv"
    save_history_csv(history, history_path)

    plot_training_curves(
        history,
        output_dir=figures_dir,
    )

    print("\nTraining complete.")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best model saved at: {best_model_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CNN model for crop disease detection."
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
        help="Model architecture to train.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate.",
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout probability.",
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for Adam optimizer.",
    )

    parser.add_argument(
        "--no_pretrained",
        action="store_true",
        help="Disable ImageNet pretrained weights.",
    )

    parser.add_argument(
        "--freeze_features",
        action="store_true",
        help="Freeze CNN backbone and train only classifier head.",
    )

    parser.add_argument(
        "--no_class_weights",
        action="store_true",
        help="Disable class-weighted CrossEntropyLoss.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    train_model(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        pretrained=not args.no_pretrained,
        freeze_features=args.freeze_features,
        use_class_weights=not args.no_class_weights,
    )


if __name__ == "__main__":
    main()