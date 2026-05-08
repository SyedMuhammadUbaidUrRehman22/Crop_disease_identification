import os
import hashlib
from pathlib import Path
from PIL import Image, UnidentifiedImageError

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def file_hash(filepath: Path, chunk_size: int = 8192) -> str:
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def remove_corrupted_images(root_dir: Path):
    removed = []

    for path in root_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
            try:
                with Image.open(path) as img:
                    img.verify()
            except (UnidentifiedImageError, OSError, IOError):
                path.unlink(missing_ok=True)
                removed.append(str(path))

    print(f"Removed {len(removed)} corrupted images.")
    return removed


def remove_duplicates(root_dir: Path):
    seen_hashes = {}
    duplicates = []

    for path in root_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
            try:
                h = file_hash(path)
                if h in seen_hashes:
                    path.unlink(missing_ok=True)
                    duplicates.append(str(path))
                else:
                    seen_hashes[h] = str(path)
            except Exception as e:
                print(f"Error processing {path}: {e}")

    print(f"Removed {len(duplicates)} duplicate images.")
    return duplicates


def filter_small_images(root_dir: Path, min_width: int = 100, min_height: int = 100):
    removed = []

    for path in root_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
            try:
                with Image.open(path) as img:
                    width, height = img.size
                    if width < min_width or height < min_height:
                        path.unlink(missing_ok=True)
                        removed.append(str(path))
            except Exception as e:
                print(f"Error checking image {path}: {e}")

    print(f"Removed {len(removed)} very small images.")
    return removed


def main():
    base_dir = Path(__file__).resolve().parent.parent
    target_dirs = [
        base_dir / "data" / "combined_dataset",
        base_dir / "data" / "train",
        base_dir / "data" / "val",
        base_dir / "data" / "test",
    ]

    for dataset_path in target_dirs:
        if not dataset_path.exists():
            print(f"Skipping missing folder: {dataset_path}")
            continue

        print(f"\nCleaning dataset: {dataset_path}")
        print("Step 1: Removing corrupted images...")
        remove_corrupted_images(dataset_path)

        print("Step 2: Removing duplicate images...")
        remove_duplicates(dataset_path)

        print("Step 3: Removing very small images...")
        filter_small_images(dataset_path, min_width=100, min_height=100)

    print("\nData cleaning completed successfully.")


if __name__ == "__main__":
    main()