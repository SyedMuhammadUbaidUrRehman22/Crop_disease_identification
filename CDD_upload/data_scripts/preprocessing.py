from pathlib import Path
from typing import Tuple, Dict

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0  # keep 0 for Windows stability


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns training and evaluation transforms.

    The selected paper resizes images to 224x224 and uses ImageNet normalization.
    Training transforms include augmentation to improve generalization.
    """

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    return train_transform, eval_transform


def create_datasets(
    train_dir: Path,
    val_dir: Path,
    test_dir: Path
) -> Tuple[datasets.ImageFolder, datasets.ImageFolder, datasets.ImageFolder]:
    """
    Creates ImageFolder datasets from train, validation, and test folders.

    Expected folder structure:
    data/
    ├── train/
    │   ├── apple_leaf_healthy/
    │   ├── apple_leaf_rust/
    │   └── ...
    ├── val/
    └── test/
    """

    train_transform, eval_transform = get_transforms()

    train_dataset = datasets.ImageFolder(
        root=str(train_dir),
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        root=str(val_dir),
        transform=eval_transform
    )

    test_dataset = datasets.ImageFolder(
        root=str(test_dir),
        transform=eval_transform
    )

    return train_dataset, val_dataset, test_dataset


def create_dataloaders(
    train_dataset: datasets.ImageFolder,
    val_dataset: datasets.ImageFolder,
    test_dataset: datasets.ImageFolder,
    batch_size: int = BATCH_SIZE
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates PyTorch DataLoaders.
    """

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    return train_loader, val_loader, test_loader


def get_class_info(dataset: datasets.ImageFolder) -> Dict[int, str]:
    """
    Returns class index to class name mapping.
    """

    return {idx: class_name for class_name, idx in dataset.class_to_idx.items()}


def main():
    base_dir = Path(__file__).resolve().parent.parent

    train_dir = base_dir / "data" / "train"
    val_dir = base_dir / "data" / "val"
    test_dir = base_dir / "data" / "test"

    print("Creating PyTorch datasets...")
    train_dataset, val_dataset, test_dataset = create_datasets(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir
    )

    print("Creating PyTorch dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset
    )

    class_info = get_class_info(train_dataset)

    print("\nDataset summary:")
    print(f"Train images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"Test images: {len(test_dataset)}")
    print(f"Number of classes: {len(class_info)}")

    print("\nClass mapping:")
    for idx, class_name in class_info.items():
        print(f"{idx}: {class_name}")

    print("\nTesting one batch...")
    images, labels = next(iter(train_loader))

    print(f"Image batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")
    print("Preprocessing pipeline is working correctly.")


if __name__ == "__main__":
    main()