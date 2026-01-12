import os
from pathlib import Path
from PIL import Image

# =====================
# CONFIG
# =====================
TEST_MODE = True  # ← change to False when you're ready for real processing
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
    if not os.path.exists(path):
        os.makedirs(path)

def preprocess_image(img_path, out_path):
    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
        img.save(out_path, format="JPEG", quality=95)
    except Exception as e:
        print(f"❌ Error processing {img_path}: {e}")

def preprocess_dataset():
    ensure_dir(OUT_DIR)

    classes = os.listdir(RAW_DIR)
    print("=== Preprocessing Images ===")
    print(f"Writing to: {OUT_DIR}")
    print(f"Found classes: {classes}")

    for cls in classes:
        cls_raw = os.path.join(RAW_DIR, cls)

        if not os.path.isdir(cls_raw):
            continue

        cls_out = os.path.join(OUT_DIR, cls)
        ensure_dir(cls_out)

        for img_name in os.listdir(cls_raw):
            # Skip hidden files or invalid files
            if img_name.startswith("."):
                continue
            raw_path = os.path.join(cls_raw, img_name)

            if not os.path.isfile(raw_path):
                continue

            out_name = os.path.splitext(img_name)[0] + ".jpg"
            out_path = os.path.join(cls_out, out_name)

            preprocess_image(raw_path, out_path)
            print(f"  ✔ Processed: {img_name} → {out_name}")

if __name__ == "__main__":
    preprocess_dataset()
