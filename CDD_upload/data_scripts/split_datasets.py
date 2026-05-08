import shutil
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SOURCE_DIR = BASE_DIR / "data" / "combined_dataset"
TRAIN_DIR = BASE_DIR / "data" / "train"
VAL_DIR = BASE_DIR / "data" / "val"
TEST_DIR = BASE_DIR / "data" / "test"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def clear_split_dirs():
    for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)


def get_images(class_dir: Path):
    return [f for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS]


def copy_files(files, destination: Path):
    ensure_dir(destination)
    for file in files:
        shutil.copy2(file, destination / file.name)


def split_class(class_name: str, class_dir: Path):
    images = get_images(class_dir)
    random.shuffle(images)

    n = len(images)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    train_files = images[:train_end]
    val_files = images[train_end:val_end]
    test_files = images[val_end:]

    copy_files(train_files, TRAIN_DIR / class_name)
    copy_files(val_files, VAL_DIR / class_name)
    copy_files(test_files, TEST_DIR / class_name)

    print(f"{class_name}: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")


def main():
    random.seed(SEED)

    if not SOURCE_DIR.exists():
        print(f"Source directory not found: {SOURCE_DIR}")
        return

    clear_split_dirs()

    class_dirs = [d for d in SOURCE_DIR.iterdir() if d.is_dir()]
    if not class_dirs:
        print("No class folders found in combined_dataset.")
        return

    print("Splitting dataset...\n")
    for class_dir in sorted(class_dirs):
        split_class(class_dir.name, class_dir)

    print("\nDone. Dataset split created in:")
    print(TRAIN_DIR)
    print(VAL_DIR)
    print(TEST_DIR)


if __name__ == "__main__":
    main()