import argparse
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import datasets, transforms

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from models.model_factory import create_model


IMAGE_SIZE = 224


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def get_display_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    ])


def load_checkpoint(checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location=device)


def load_model_from_checkpoint(
    checkpoint_path: Path,
    fallback_model_name: str,
    fallback_dropout: float,
    device: torch.device,
):
    checkpoint = load_checkpoint(checkpoint_path, device)

    model_name = checkpoint.get("model_name", fallback_model_name)
    num_classes = checkpoint["num_classes"]
    class_names = checkpoint["class_names"]
    dropout = checkpoint.get("dropout", fallback_dropout)

    model = create_model(
        model_name=model_name,
        num_classes=num_classes,
        dropout=dropout,
        pretrained=False,
        freeze_features=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, class_names, model_name


def predict_image(model, image_tensor, device):
    with torch.no_grad():
        outputs = model(image_tensor.unsqueeze(0).to(device))
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
    return pred_idx


def collect_samples(dataset, model, class_names, device, num_samples=6, prefer_misclassified=False):
    """
    Collect sample indices from the test dataset.
    If prefer_misclassified=True, tries to collect wrong predictions first.
    """
    candidates = []

    for idx, (path, true_idx) in enumerate(dataset.samples):
        try:
            img = Image.open(path).convert("RGB")
            eval_tensor = get_eval_transform()(img)
            pred_idx = predict_image(model, eval_tensor, device)

            is_correct = pred_idx == true_idx
            candidates.append({
                "path": path,
                "true_idx": true_idx,
                "pred_idx": pred_idx,
                "is_correct": is_correct,
            })
        except Exception as e:
            print(f"Skipping {path}: {e}")

    if prefer_misclassified:
        wrong = [c for c in candidates if not c["is_correct"]]
        correct = [c for c in candidates if c["is_correct"]]
        selected = wrong[:num_samples]
        if len(selected) < num_samples:
            selected.extend(correct[: num_samples - len(selected)])
    else:
        random.shuffle(candidates)
        selected = candidates[:num_samples]

    return selected


def collect_balanced_samples(dataset, model, class_names, device, num_samples=6):
    """
    Try to return a balanced mix of correct and incorrect predictions.
    """
    candidates = []

    for idx, (path, true_idx) in enumerate(dataset.samples):
        try:
            img = Image.open(path).convert("RGB")
            eval_tensor = get_eval_transform()(img)
            pred_idx = predict_image(model, eval_tensor, device)

            is_correct = pred_idx == true_idx
            candidates.append({
                "path": path,
                "true_idx": true_idx,
                "pred_idx": pred_idx,
                "is_correct": is_correct,
            })
        except Exception as e:
            print(f"Skipping {path}: {e}")

    wrong = [c for c in candidates if not c["is_correct"]]
    correct = [c for c in candidates if c["is_correct"]]

    random.shuffle(wrong)
    random.shuffle(correct)

    selected = []
    half = num_samples // 2

    selected.extend(wrong[:half])
    selected.extend(correct[:num_samples - len(selected)])

    if len(selected) < num_samples:
        remaining = [c for c in candidates if c not in selected]
        random.shuffle(remaining)
        selected.extend(remaining[: num_samples - len(selected)])

    return selected[:num_samples]


def plot_samples(samples, class_names, output_path: Path, title: str):
    n = len(samples)
    cols = 3
    rows = int(np.ceil(n / cols))

    display_transform = get_display_transform()

    plt.figure(figsize=(15, 5 * rows))

    for i, sample in enumerate(samples, start=1):
        img = Image.open(sample["path"]).convert("RGB")
        img = display_transform(img)

        true_label = class_names[sample["true_idx"]]
        pred_label = class_names[sample["pred_idx"]]

        plt.subplot(rows, cols, i)
        plt.imshow(img)
        plt.axis("off")

        color = "green" if sample["is_correct"] else "red"
        plt.title(
            f"True: {true_label}\nPred: {pred_label}",
            color=color,
            fontsize=10
        )

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Sample prediction figure saved to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate sample prediction figure from trained model."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="densenet201",
        choices=[
            "mobilenetv2",
            "resnet18",
            "efficientnet_b0",
            "resnet50",
            "densenet201",
            "efficientnet_b3",
        ],
        help="Model name."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/weights/densenet201/best.pth",
        help="Path to model checkpoint."
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=6,
        help="Number of samples to include in the figure."
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="balanced",
        choices=["balanced", "misclassified", "random"],
        help="Sample selection mode."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device = get_device()
    print(f"Using device: {device}")

    checkpoint_path = BASE_DIR / args.checkpoint
    test_dir = BASE_DIR / "data" / "test"

    model, class_names, resolved_model_name = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        fallback_model_name=args.model,
        fallback_dropout=0.3,
        device=device,
    )

    dataset = datasets.ImageFolder(
        root=str(test_dir),
        transform=None,
    )

    if dataset.classes != class_names:
        print("Warning: Dataset class order and checkpoint class order may differ.")
        print("Using checkpoint class names for titles.")

    if args.mode == "misclassified":
        samples = collect_samples(
            dataset=dataset,
            model=model,
            class_names=class_names,
            device=device,
            num_samples=args.num_samples,
            prefer_misclassified=True,
        )
        figure_title = f"Sample Misclassifications - {resolved_model_name}"

    elif args.mode == "random":
        samples = collect_samples(
            dataset=dataset,
            model=model,
            class_names=class_names,
            device=device,
            num_samples=args.num_samples,
            prefer_misclassified=False,
        )
        figure_title = f"Random Sample Predictions - {resolved_model_name}"

    else:
        samples = collect_balanced_samples(
            dataset=dataset,
            model=model,
            class_names=class_names,
            device=device,
            num_samples=args.num_samples,
        )
        figure_title = f"Sample Predictions - {resolved_model_name}"

    output_path = BASE_DIR / "outputs" / "figures" / resolved_model_name / "sample_predictions.png"
    plot_samples(
        samples=samples,
        class_names=class_names,
        output_path=output_path,
        title=figure_title,
    )


if __name__ == "__main__":
    main()