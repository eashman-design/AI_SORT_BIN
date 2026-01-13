"""
AI Sort Bin - Image Preprocessing

Execution:
    python scripts/training/preprocess.py

Must be run from the repository root directory.
Resizes raw images to 224x224 for model training.
Validates that dataset folders match config labels.
"""
import os
import sys
from pathlib import Path
from PIL import Image

from scripts.common.config import get_labels

# =====================
# CONFIG
# =====================
TEST_MODE = True  # change to False when you're ready for real processing
TARGET_SIZE = (224, 224)

# Resolve repo root from scripts/training/*.py → parents[2] is repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "dataset" / "raw"

if TEST_MODE:
    OUT_DIR = REPO_ROOT / "dataset" / "processed_test"
else:
    OUT_DIR = REPO_ROOT / "dataset" / "processed"

# =====================
# FUNCTIONS
# =====================

def ensure_dir(path):
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True)


def preprocess_image(img_path, out_path):
    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
        img.save(out_path, format="JPEG", quality=95)
    except Exception as e:
        print(f"[ERROR] Processing {img_path}: {e}")


def preprocess_dataset():
    # Load expected labels from config
    expected_labels = get_labels()
    print(f"Expected labels (from config): {expected_labels}")

    if not RAW_DIR.is_dir():
        sys.exit(
            f"[ERROR] Raw dataset directory not found: {RAW_DIR}\n\n"
            "Expected structure:\n"
            f"  {RAW_DIR}/\n" +
            "\n".join(f"    {label}/" for label in expected_labels) +
            "\n\nCreate these directories and add images to each class folder."
        )

    ensure_dir(OUT_DIR)

    # Get actual folders
    actual_folders = sorted([
        f for f in os.listdir(RAW_DIR)
        if (RAW_DIR / f).is_dir() and not f.startswith(".")
    ])

    print("=== Preprocessing Images ===")
    print(f"Writing to: {OUT_DIR}")
    print(f"Found folders: {actual_folders}")

    # Validate against config
    missing = set(expected_labels) - set(actual_folders)
    extra = set(actual_folders) - set(expected_labels)

    if missing:
        print(f"\n[WARN] Missing folders for labels: {sorted(missing)}")
        print("These classes will have no preprocessed images.")
        print("Create folders with:")
        for label in sorted(missing):
            print(f"  mkdir -p {RAW_DIR / label}")

    if extra:
        print(f"\n[WARN] Extra folders not in config: {sorted(extra)}")
        print("These will be skipped. Update config if you want to include them.")

    # Process only folders that are in the expected labels
    processed_count = 0
    for cls in expected_labels:
        cls_raw = RAW_DIR / cls

        if not cls_raw.is_dir():
            print(f"\n[SKIP] {cls}/ - folder not found")
            continue

        cls_out = OUT_DIR / cls
        ensure_dir(cls_out)

        images = [
            f for f in os.listdir(cls_raw)
            if not f.startswith(".") and (cls_raw / f).is_file()
        ]

        print(f"\nClass '{cls}' - {len(images)} files")

        for img_name in images:
            raw_path = cls_raw / img_name
            out_name = Path(img_name).stem + ".jpg"
            out_path = cls_out / out_name

            preprocess_image(raw_path, out_path)
            print(f"  [OK] {img_name} -> {out_name}")
            processed_count += 1

    print(f"\n=== Done ===")
    print(f"Processed {processed_count} images")


if __name__ == "__main__":
    preprocess_dataset()
