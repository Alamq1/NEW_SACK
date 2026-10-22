"""
RICE BAG 10 - V5
POLYGON -> YOLO BBOX CONVERTER

Original dataset is NEVER modified.

Input:
    data/Rice bag 10/
        train/images
        train/labels
        valid/images
        valid/labels

Output:
    data/Rice bag 10 V5/
        train/images
        train/labels
        valid/images
        valid/labels
        data.yaml

YOLO segmentation label:
    class x1 y1 x2 y2 x3 y3 ...

YOLO detection label:
    class x_center y_center width height
"""

from pathlib import Path
import shutil
import yaml


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

SOURCE = ROOT / "data" / "Rice bag 10"
DEST = ROOT / "data" / "Rice bag 10 V5"

CLASS_NAME = "Rice-bag-10"
CLASS_ID = 0


# ============================================================
# HELPERS
# ============================================================

def polygon_to_bbox(values):
    """
    Convert normalized polygon coordinates to normalized YOLO bbox.
    values = [x1, y1, x2, y2, ...]
    """

    if len(values) < 6 or len(values) % 2 != 0:
        return None

    xs = values[0::2]
    ys = values[1::2]

    # Keep normalized coordinates inside image bounds
    xs = [max(0.0, min(1.0, x)) for x in xs]
    ys = [max(0.0, min(1.0, y)) for y in ys]

    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)

    width = xmax - xmin
    height = ymax - ymin

    if width <= 0 or height <= 0:
        return None

    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0

    return CLASS_ID, x_center, y_center, width, height


def convert_label(src_label, dst_label):
    converted = []
    skipped = 0

    with open(src_label, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()

        if not parts:
            continue

        try:
            class_id = int(float(parts[0]))
            coords = [float(x) for x in parts[1:]]
        except ValueError:
            skipped += 1
            continue

        bbox = polygon_to_bbox(coords)

        if bbox is None:
            skipped += 1
            continue

        cid, xc, yc, w, h = bbox

        converted.append(
            f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n"
        )

    with open(dst_label, "w", encoding="utf-8") as f:
        f.writelines(converted)

    return len(converted), skipped


# ============================================================
# VALIDATE SOURCE
# ============================================================

print("=" * 70)
print("RICE BAG 10 - V5 POLYGON → BBOX CONVERTER")
print("=" * 70)

print(f"Source : {SOURCE}")
print(f"Output : {DEST}")

if not SOURCE.exists():
    raise FileNotFoundError(
        f"Source dataset not found:\n{SOURCE}"
    )

if not (SOURCE / "data.yaml").exists():
    raise FileNotFoundError(
        f"Source data.yaml not found:\n{SOURCE / 'data.yaml'}"
    )


# ============================================================
# CREATE DESTINATION
# ============================================================

if DEST.exists():
    print("\nOutput folder already exists.")
    print("Existing V5 converted dataset will be replaced.")

    shutil.rmtree(DEST)

for split in ["train", "valid"]:
    (DEST / split / "images").mkdir(parents=True, exist_ok=True)
    (DEST / split / "labels").mkdir(parents=True, exist_ok=True)


# ============================================================
# CONVERT
# ============================================================

total_images = 0
total_labels = 0
total_skipped = 0

for split in ["train", "valid"]:

    src_images = SOURCE / split / "images"
    src_labels = SOURCE / split / "labels"

    dst_images = DEST / split / "images"
    dst_labels = DEST / split / "labels"

    if not src_images.exists():
        print(f"\nWARNING: {src_images} does not exist")
        continue

    if not src_labels.exists():
        print(f"\nWARNING: {src_labels} does not exist")
        continue

    print(f"\n{'-' * 70}")
    print(f"CONVERTING: {split.upper()}")
    print(f"{'-' * 70}")

    image_files = [
        p for p in src_images.iterdir()
        if p.is_file()
    ]

    split_images = 0
    split_labels = 0
    split_skipped = 0

    for image_path in image_files:

        # Copy image unchanged
        shutil.copy2(
            image_path,
            dst_images / image_path.name
        )

        total_images += 1
        split_images += 1

        label_path = src_labels / f"{image_path.stem}.txt"
        dst_label_path = dst_labels / f"{image_path.stem}.txt"

        # If an image has no label, create empty label file
        if not label_path.exists():
            dst_label_path.write_text(
                "",
                encoding="utf-8"
            )
            continue

        converted, skipped = convert_label(
            label_path,
            dst_label_path
        )

        total_labels += converted
        total_skipped += skipped

        split_labels += converted
        split_skipped += skipped

    print(f"Images converted : {split_images}")
    print(f"BBoxes created   : {split_labels}")
    print(f"Labels skipped   : {split_skipped}")


# ============================================================
# CREATE DATA.YAML
# ============================================================

data_yaml = {
    "train": "../train/images",
    "val": "../valid/images",
    "test": "../test/images",
    "nc": 1,
    "names": [CLASS_NAME],
}

with open(
    DEST / "data.yaml",
    "w",
    encoding="utf-8"
) as f:
    yaml.safe_dump(
        data_yaml,
        f,
        sort_keys=False
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CONVERSION COMPLETE")
print("=" * 70)

print(f"Source dataset:")
print(f"  {SOURCE}")

print(f"\nNew V5 dataset:")
print(f"  {DEST}")

print(f"\nTotal images copied : {total_images}")
print(f"Total BBoxes created: {total_labels}")
print(f"Invalid/skipped     : {total_skipped}")

print("\nOriginal dataset was NOT modified.")

print("\nNext:")
print("1. Verify converted BBoxes visually.")
print("2. If correct, train V5 detection model.")
print("=" * 70)
