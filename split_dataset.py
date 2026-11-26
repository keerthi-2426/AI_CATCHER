import os
import shutil
import random
from pathlib import Path

# Parameters
SOURCE_DIRS = {
    "violence": r"e:\AI_Catcher\violence",
    "non_violence": r"e:\AI_Catcher\non_violence"
}
DATASET_DIR = r"e:\AI_Catcher\dataset"
TRAIN_RATIO = 0.8

def main():
    print(f"Creating dataset structure at {DATASET_DIR}...")
    # Create dataset directories
    for split in ["train", "val"]:
        for class_name in SOURCE_DIRS.keys():
            os.makedirs(os.path.join(DATASET_DIR, split, class_name), exist_ok=True)
            
    for class_name, source_dir in SOURCE_DIRS.items():
        print(f"Processing {class_name}...")
        
        # Get all image paths
        all_images = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            all_images.extend(list(Path(source_dir).glob(ext)))
            
        print(f"Found {len(all_images)} images for {class_name}.")
        
        if not all_images:
            continue
            
        # Shuffle randomly
        random.seed(42)
        random.shuffle(all_images)
        
        train_count = int(len(all_images) * TRAIN_RATIO)
        train_images = all_images[:train_count]
        val_images = all_images[train_count:]
        
        # Moved to copying to preserve originals
        for i, img_path in enumerate(train_images):
            dest = os.path.join(DATASET_DIR, "train", class_name, img_path.name)
            shutil.copy2(img_path, dest)
            if i % 1000 == 0 and i > 0:
                print(f"Copied {i} train images...")
                
        for i, img_path in enumerate(val_images):
            dest = os.path.join(DATASET_DIR, "val", class_name, img_path.name)
            shutil.copy2(img_path, dest)
            if i % 1000 == 0 and i > 0:
                print(f"Copied {i} val images...")
                
        print(f"Class {class_name}: {len(train_images)} train, {len(val_images)} val.")

    print("Dataset splitting complete!")

if __name__ == "__main__":
    main()
