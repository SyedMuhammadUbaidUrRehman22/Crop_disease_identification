import os
from datasets import load_dataset

SAVE_DIR = "data/plantvillage"

def main():
    dataset = load_dataset("mohanty/PlantVillage", "color")
    train_data = dataset["train"]

    label_names = train_data.features["label"].names
    os.makedirs(SAVE_DIR, exist_ok=True)

    for idx, sample in enumerate(train_data):
        image = sample["image"]
        label = sample["label"]
        class_name = label_names[label].replace(" ", "_").replace(",", "").replace("/", "_")

        class_dir = os.path.join(SAVE_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)

        image_path = os.path.join(class_dir, f"img_{idx}.jpg")
        image.save(image_path)

        if idx % 500 == 0:
            print(f"Saved {idx} images...")

    print("PlantVillage dataset saved successfully.")

if __name__ == "__main__":
    main()