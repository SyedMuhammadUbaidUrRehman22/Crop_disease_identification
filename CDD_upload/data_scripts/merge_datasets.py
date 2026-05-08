import os
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

PLANTDOC_TRAIN = BASE_DIR / "data" / "plantdoc" / "train"
PLANTDOC_TEST = BASE_DIR / "data" / "plantdoc" / "test"

WEB_ROOT = BASE_DIR / "data" / "web_sourced" / "Plant Leaf Disease Detection Using Deep Learning - A Multi-Dataset Approach (Web sourced Dataset)"
WEB_TRAIN = WEB_ROOT / "train"
WEB_TEST = WEB_ROOT / "test"

COMBINED_ROOT = BASE_DIR / "data" / "combined_dataset"

PLANTDOC_MAP = {
    "Apple_leaf": "apple_leaf_healthy",
    "Apple_rust_leaf": "apple_leaf_rust",
    "Apple_Scab_Leaf": "apple_leaf_scab",
    "Corn_Gray_leaf_spot": "corn_grey_leaf_spot",
    "Corn_leaf_blight": "corn_leaf_blight",
    "Corn_rust_leaf": "corn_leaf_rust",
    "Potato_leaf_late_blight": "potato_leaf_blight",
    "Tomato_leaf_bacterial_spot": "tomato_leaf_bacterial_spot",
    "Tomato_Early_blight_leaf": "tomato_leaf_early_blight",
    "Tomato_leaf": "tomato_leaf_healthy",
    "Tomato_leaf_late_blight": "tomato_leaf_late_blight",
    "Tomato_mold_leaf": "tomato_leaf_mould",
    "Tomato_Septoria_leaf_spot": "tomato_septoria_leaf_spot",
}

WEB_MAP = {
    "apple leaf healthy": "apple_leaf_healthy",
    "apple leaf rust": "apple_leaf_rust",
    "apple leaf scab": "apple_leaf_scab",
    "corn gray leaf spot": "corn_grey_leaf_spot",
    "corn leaf": "corn_leaf_healthy",
    "corn leaf blight": "corn_leaf_blight",
    "corn leaf rust": "corn_leaf_rust",
    "potato leaf blight": "potato_leaf_blight",
    "potato leafroll virus": "potato_leafroll_virus",
    "tomato leaf bacterial spot": "tomato_leaf_bacterial_spot",
    "tomato leaf early blight": "tomato_leaf_early_blight",
    "tomato leaf healthy": "tomato_leaf_healthy",
    "tomato leaf late blight": "tomato_leaf_late_blight",
    "tomato leaf mold": "tomato_leaf_mould",
    "tomato leaf powdery mildew": "tomato_leaf_powdery_mildew",
    "tomato septoria leaf spot": "tomato_septoria_leaf_spot",
}

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def copy_images(src_dir: Path, dst_dir: Path, prefix: str):
    if not src_dir.exists():
        print(f"Missing folder: {src_dir}")
        return 0

    ensure_dir(dst_dir)
    count = 0

    for file in src_dir.iterdir():
        if file.is_file() and file.suffix.lower() in VALID_EXTENSIONS:
            new_name = f"{prefix}_{count:05d}{file.suffix.lower()}"
            shutil.copy2(file, dst_dir / new_name)
            count += 1

    print(f"Copied {count} images from {src_dir} -> {dst_dir}")
    return count


def merge_dataset(split_dirs, class_map, source_prefix):
    summary = {}

    for split_name, split_path in split_dirs.items():
        print(f"\nChecking split: {split_name} -> {split_path}")
        if not split_path.exists():
            print(f"Warning: split path does not exist: {split_path}")
            continue

        for folder_name, standard_name in class_map.items():
            src_class_dir = split_path / folder_name
            dst_class_dir = COMBINED_ROOT / standard_name

            copied = copy_images(
                src_class_dir,
                dst_class_dir,
                prefix=f"{source_prefix}_{split_name}"
            )

            summary.setdefault(standard_name, 0)
            summary[standard_name] += copied

    return summary


def main():
    print("Base directory:", BASE_DIR)
    print("PlantDoc train:", PLANTDOC_TRAIN)
    print("PlantDoc test:", PLANTDOC_TEST)
    print("Web train:", WEB_TRAIN)
    print("Web test:", WEB_TEST)
    print("Combined root:", COMBINED_ROOT)

    ensure_dir(COMBINED_ROOT)

    print("\nMerging PlantDoc...")
    plantdoc_summary = merge_dataset(
        {"train": PLANTDOC_TRAIN, "test": PLANTDOC_TEST},
        PLANTDOC_MAP,
        "plantdoc"
    )

    print("\nMerging web-sourced dataset...")
    web_summary = merge_dataset(
        {"train": WEB_TRAIN, "test": WEB_TEST},
        WEB_MAP,
        "web"
    )

    all_classes = sorted(set(plantdoc_summary.keys()) | set(web_summary.keys()))

    print("\nCombined dataset summary:")
    for cls in all_classes:
        pd_count = plantdoc_summary.get(cls, 0)
        web_count = web_summary.get(cls, 0)
        total = pd_count + web_count
        print(f"{cls}: PlantDoc={pd_count}, Web={web_count}, Total={total}")

    print("\nDone. Combined dataset created at:", COMBINED_ROOT)


if __name__ == "__main__":
    main()