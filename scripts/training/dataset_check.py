import os
from pathlib import Path
from PIL import Image

# Resolve repo root from scripts/training/*.py → parents[2] is repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = REPO_ROOT / "dataset" / "raw"

VALID_EXTENSIONS = [".jpg", ".jpeg", ".png"]

def check_dataset_structure():
    print("=== Dataset Structure Check ===")

    if not os.path.exists(DATASET_PATH):
        print(f"❌ ERROR: Dataset path not found: {DATASET_PATH}")
        return

    classes = sorted(os.listdir(DATASET_PATH))
    print(f"Found classes: {classes}")

    for cls in classes:
        cls_path = os.path.join(DATASET_PATH, cls)

        if not os.path.isdir(cls_path):
            print(f"❌ ERROR: {cls_path} is not a directory!")
            continue

        images = os.listdir(cls_path)
        print(f"\nClass '{cls}' → {len(images)} images")

        for img in images:
            if img.startswith("."):
                continue

            img_path = os.path.join(cls_path, img)
            ext = os.path.splitext(img)[-1].lower()

            # Check extension
            if ext not in VALID_EXTENSIONS:
                print(f"⚠️ WARNING: Invalid extension in {img_path}")
                continue

            # Check if a file is readable
            try:
                with Image.open(img_path) as im:
                    im.verify()
            except Exception as e:
                print(f"❌ CORRUPT: {img_path} ({e})")
                continue

            # Try reopening normally to get size
            with Image.open(img_path) as im:
                width, height = im.size

            print(f"  ✔ {img} — {width}x{height}")

if __name__ == "__main__":
    check_dataset_structure()
