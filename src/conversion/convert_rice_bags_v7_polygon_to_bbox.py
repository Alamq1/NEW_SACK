from pathlib import Path
import shutil


# ============================================================
# RICE BAG 10 - V7
# POLYGON -> BBOX CONVERTER
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_DATASET = PROJECT_ROOT / "data" / "Rice Bag 10 V7"
OUTPUT_DATASET = PROJECT_ROOT / "data" / "Rice Bag 10 V7 BBOX"


print("=" * 70)
print("RICE BAG 10 - V7 POLYGON -> BBOX CONVERTER")
print("=" * 70)

print(f"Source : {SOURCE_DATASET}")
print(f"Output : {OUTPUT_DATASET}")


# ============================================================
# CHECK SOURCE
# ============================================================

if not SOURCE_DATASET.exists():
    raise FileNotFoundError(
        f"\nSource dataset not found:\n{SOURCE_DATASET}"
    )


# ============================================================
# CONVERSION FUNCTION
# ============================================================

def polygon_to_bbox(label_line):

    values = label_line.strip().split()

    if len(values) < 7:
        return None

    try:
        class_id = int(values[0])

        coordinates = list(
            map(float, values[1:])
        )

    except ValueError:
        return None

    # Polygon must contain x,y pairs
    if len(coordinates) < 6 or len(coordinates) % 2 != 0:
        return None

    xs = coordinates[0::2]
    ys = coordinates[1::2]

    # Bounding rectangle
    xmin = max(0.0, min(xs))
    xmax = min(1.0, max(xs))

    ymin = max(0.0, min(ys))
    ymax = min(1.0, max(ys))

    width = xmax - xmin
    height = ymax - ymin

    if width <= 0 or height <= 0:
        return None

    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2

    return (
        f"{class_id} "
        f"{x_center:.6f} "
        f"{y_center:.6f} "
        f"{width:.6f} "
        f"{height:.6f}"
    )


# ============================================================
# SPLITS
# ============================================================

splits = [
    "train",
    "valid",
    "test"
]

total_images = 0
total_boxes = 0
total_skipped = 0


# ============================================================
# PROCESS DATASET
# ============================================================

for split in splits:

    source_images = SOURCE_DATASET / split / "images"
    source_labels = SOURCE_DATASET / split / "labels"

    output_images = OUTPUT_DATASET / split / "images"
    output_labels = OUTPUT_DATASET / split / "labels"

    print("\n" + "-" * 70)
    print(f"CONVERTING: {split.upper()}")
    print("-" * 70)

    if not source_images.exists():

        print(f"No images directory: {source_images}")
        continue

    output_images.mkdir(
        parents=True,
        exist_ok=True
    )

    output_labels.mkdir(
        parents=True,
        exist_ok=True
    )

    split_images = 0
    split_boxes = 0
    split_skipped = 0

    image_files = []

    for extension in [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.JPG",
        "*.JPEG",
        "*.PNG"
    ]:

        image_files.extend(
            source_images.glob(extension)
        )

    for image_path in sorted(image_files):

        split_images += 1
        total_images += 1

        # ----------------------------------------------------
        # COPY IMAGE
        # ----------------------------------------------------

        destination_image = (
            output_images / image_path.name
        )

        shutil.copy2(
            image_path,
            destination_image
        )

        # ----------------------------------------------------
        # FIND LABEL
        # ----------------------------------------------------

        label_path = (
            source_labels
            / f"{image_path.stem}.txt"
        )

        output_label = (
            output_labels
            / f"{image_path.stem}.txt"
        )

        # No label file
        if not label_path.exists():

            output_label.write_text("")

            continue

        # ----------------------------------------------------
        # CONVERT LABELS
        # ----------------------------------------------------

        converted_boxes = []

        lines = label_path.read_text(
            encoding="utf-8"
        ).splitlines()

        for line in lines:

            if not line.strip():
                continue

            bbox = polygon_to_bbox(line)

            if bbox is None:

                split_skipped += 1
                total_skipped += 1

                continue

            converted_boxes.append(bbox)

        # ----------------------------------------------------
        # WRITE NEW LABEL
        # ----------------------------------------------------

        output_label.write_text(
            "\n".join(converted_boxes),
            encoding="utf-8"
        )

        split_boxes += len(
            converted_boxes
        )

        total_boxes += len(
            converted_boxes
        )

    print(f"Images converted : {split_images}")
    print(f"BBoxes created   : {split_boxes}")
    print(f"Labels skipped   : {split_skipped}")


# ============================================================
# CREATE NEW DATA.YAML
# ============================================================

OUTPUT_DATASET.mkdir(
    parents=True,
    exist_ok=True
)

yaml_content = """train: ../train/images
val: ../valid/images
test: ../test/images

nc: 1
names:
- Rice-bag-10
"""

yaml_path = OUTPUT_DATASET / "data.yaml"

yaml_path.write_text(
    yaml_content,
    encoding="utf-8"
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("V7 CONVERSION COMPLETE")
print("=" * 70)

print(f"Source dataset:")
print(f"  {SOURCE_DATASET}")

print("\nNew V7 BBOX dataset:")
print(f"  {OUTPUT_DATASET}")

print(f"\nTotal images copied : {total_images}")
print(f"Total BBoxes created: {total_boxes}")
print(f"Invalid/skipped     : {total_skipped}")

print(f"\nNew data.yaml:")
print(f"  {yaml_path}")

print("\nOriginal dataset was NOT modified.")

print("\nNext step:")
print("1. Verify BBoxes visually.")
print("2. Then train V7 detection model.")
print("=" * 70)