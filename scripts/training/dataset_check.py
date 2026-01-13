"""
AI Sort Bin - Dataset Integrity Checker

Execution:
    python scripts/training/dataset_check.py

Must be run from the repository root directory.
Validates dataset structure and image readability against config labels.
"""
import os
import sys
from pathlib import Path
from PIL import Image

from scripts.common.config import get_labels, validate_labels_against_dataset

# Resolve repo root from scripts/training/*.py → parents[2] is repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = REPO_ROOT / "dataset" / "raw"

VALID_EXTENSIONS = [".jpg", ".jpeg", ".png"]


def check_dataset_structure():
    print("=== Dataset Structure Check ===")

    # Load expected labels from config
    expected_labels = get_labels()
    print(f"Expected labels (from config): {expected_labels}")

    if not DATASET_PATH.is_dir():
        sys.exit(
            f"[ERROR] Dataset path not found: {DATASET_PATH}\n\n"
            "Expected structure:\n"
            f"  {DATASET_PATH}/\n" +
            "\n".join(f"    {label}/" for label in expected_labels) +
            "\n\nCreate these directories and add images to each class folder."
        )

    # Get actual folders (excluding hidden files)
    actual_folders = sorted([
        f for f in os.listdir(DATASET_PATH)
        if os.path.isdir(DATASET_PATH / f) and not f.startswith(".")
    ])
    print(f"Found folders: {actual_folders}")

    # Validate labels match config
    is_valid, message = validate_labels_against_dataset(actual_folders)

    if not is_valid:
        missing_in_dataset = set(expected_labels) - set(actual_folders)
        extra_in_dataset = set(actual_folders) - set(expected_labels)

        error_msg = "[ERROR] Dataset folders do not match config labels!\n\n"
        error_msg += f"Config labels: {expected_labels}\n"
        error_msg += f"Dataset folders: {actual_folders}\n\n"

        if missing_in_dataset:
            error_msg += f"MISSING folders (create these):\n"
            for label in sorted(missing_in_dataset):
                error_msg += f"  mkdir -p {DATASET_PATH / label}\n"
            error_msg += "\n"

        if extra_in_dataset:
            error_msg += f"EXTRA folders (not in config, consider removing or updating config):\n"
            for label in sorted(extra_in_dataset):
                error_msg += f"  {DATASET_PATH / label}\n"
            error_msg += "\n"

        error_msg += "To fix:\n"
        error_msg += "  1. Create missing folders with images, OR\n"
        error_msg += "  2. Update dataset/dataset_config.json labels to match your folders\n"

        sys.exit(error_msg)

    print("\n[OK] Dataset folders match config labels exactly")

    # Check each class folder
    total_images = 0
    for cls in actual_folders:
        cls_path = DATASET_PATH / cls

        images = [
            f for f in os.listdir(cls_path)
            if not f.startswith(".") and os.path.isfile(cls_path / f)
        ]
        print(f"\nClass '{cls}' - {len(images)} files")

        class_valid = 0
        for img in images:
            img_path = cls_path / img
            ext = os.path.splitext(img)[-1].lower()

            # Check extension
            if ext not in VALID_EXTENSIONS:
                print(f"  [WARN] Invalid extension: {img}")
                continue

            # Check if file is readable
            try:
                with Image.open(img_path) as im:
                    im.verify()
            except Exception as e:
                print(f"  [CORRUPT] {img} ({e})")
                continue

            # Try reopening normally to get size
            with Image.open(img_path) as im:
                width, height = im.size

            print(f"  [OK] {img} - {width}x{height}")
            class_valid += 1

        total_images += class_valid

        if class_valid == 0:
            print(f"  [WARN] No valid images in '{cls}' folder!")

    print(f"\n=== Summary ===")
    print(f"Total valid images: {total_images}")
    print(f"Classes: {len(actual_folders)}")

    if total_images == 0:
        print("\n[WARN] No images found! Add images to class folders before training.")


if __name__ == "__main__":
    check_dataset_structure()
