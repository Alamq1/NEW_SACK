from pathlib import Path
import shutil
import cv2


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_DATASET = PROJECT_ROOT / "data" / "Rice Bag 10"
OUTPUT_DATASET = PROJECT_ROOT / "data" / "Rice Bag 10_v9_converted"

SPLITS = ["train", "valid", "test"]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def polygon_to_bbox(points):
    """
    Convert polygon points into bounding box.

    points:
        [(x1, y1), (x2, y2), ...]

    Returns:
        xmin, ymin, xmax, ymax
    """

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    xmin = min(xs)
    ymin = min(ys)
    xmax = max(xs)
    ymax = max(ys)

    return xmin, ymin, xmax, ymax


def convert_label_file(label_path, image_width, image_height, output_label_path):
    """
    Convert YOLO segmentation polygon labels into
    YOLO detection bounding-box labels.

    YOLO segmentation format:

    class_id x1 y1 x2 y2 x3 y3 ...

    Coordinates are normalized.

    Output:

    class_id center_x center_y width height
    """

    converted_lines = []

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line_number, line in enumerate(lines, start=1):

        line = line.strip()

        if not line:
            continue

        values = line.split()

        # Need at least:
        # class_id + 3 polygon points
        if len(values) < 7:
            print(
                f"WARNING: Skipping invalid line "
                f"{label_path} line {line_number}"
            )
            continue

        try:
            class_id = int(float(values[0]))

            coordinates = list(map(float, values[1:]))

        except ValueError:
            print(
                f"WARNING: Invalid numbers in "
                f"{label_path} line {line_number}"
            )
            continue

        # Polygon needs x,y pairs
        if len(coordinates) % 2 != 0:
            print(
                f"WARNING: Odd number of coordinates in "
                f"{label_path} line {line_number}"
            )
            continue

        points = []

        for i in range(0, len(coordinates), 2):

            x = coordinates[i]
            y = coordinates[i + 1]

            # Polygon coordinates are normalized
            pixel_x = x * image_width
            pixel_y = y * image_height

            points.append((pixel_x, pixel_y))

        if len(points) < 3:
            print(
                f"WARNING: Polygon has less than 3 points in "
                f"{label_path} line {line_number}"
            )
            continue

        # Polygon -> Bounding Box
        xmin, ymin, xmax, ymax = polygon_to_bbox(points)

        # Clamp values
        xmin = max(0, min(xmin, image_width))
        xmax = max(0, min(xmax, image_width))

        ymin = max(0, min(ymin, image_height))
        ymax = max(0, min(ymax, image_height))

        # Convert pixel bbox -> YOLO normalized bbox

        center_x = ((xmin + xmax) / 2) / image_width
        center_y = ((ymin + ymax) / 2) / image_height

        width = (xmax - xmin) / image_width
        height = (ymax - ymin) / image_height

        # Safety clamp
        center_x = max(0.0, min(center_x, 1.0))
        center_y = max(0.0, min(center_y, 1.0))

        width = max(0.0, min(width, 1.0))
        height = max(0.0, min(height, 1.0))

        converted_lines.append(
            f"{class_id} "
            f"{center_x:.6f} "
            f"{center_y:.6f} "
            f"{width:.6f} "
            f"{height:.6f}\n"
        )

    output_label_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_label_path, "w") as f:
        f.writelines(converted_lines)


# ============================================================
# DATASET CONVERSION
# ============================================================

def convert_split(split):

    source_images = SOURCE_DATASET / split / "images"
    source_labels = SOURCE_DATASET / split / "labels"

    output_images = OUTPUT_DATASET / split / "images"
    output_labels = OUTPUT_DATASET / split / "labels"

    output_images.mkdir(
        parents=True,
        exist_ok=True
    )

    output_labels.mkdir(
        parents=True,
        exist_ok=True
    )

    if not source_images.exists():
        print(f"\nWARNING: Images folder not found:")
        print(source_images)
        return

    if not source_labels.exists():
        print(f"\nWARNING: Labels folder not found:")
        print(source_labels)
        return

    image_extensions = [
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    ]

    images = [
        p for p in source_images.iterdir()
        if p.suffix.lower() in image_extensions
    ]

    print("\n" + "=" * 60)
    print(f"CONVERTING SPLIT: {split}")
    print("=" * 60)

    converted_count = 0
    skipped_count = 0

    for image_path in images:

        label_path = source_labels / f"{image_path.stem}.txt"

        output_image_path = (
            output_images / image_path.name
        )

        output_label_path = (
            output_labels / f"{image_path.stem}.txt"
        )

        # Read image
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"WARNING: Cannot read image: {image_path}")
            skipped_count += 1
            continue

        image_height, image_width = image.shape[:2]

        # Copy image
        shutil.copy2(
            image_path,
            output_image_path
        )

        # Missing label
        if not label_path.exists():

            print(
                f"WARNING: Label missing for "
                f"{image_path.name}"
            )

            # Create empty label file
            output_label_path.touch()

            skipped_count += 1
            continue

        # Convert polygon -> bbox
        convert_label_file(
            label_path,
            image_width,
            image_height,
            output_label_path
        )

        converted_count += 1

    print(f"\nImages found     : {len(images)}")
    print(f"Converted        : {converted_count}")
    print(f"Skipped          : {skipped_count}")


# ============================================================
# DATA.YAML
# ============================================================

def create_data_yaml():

    yaml_path = OUTPUT_DATASET / "data.yaml"

    yaml_content = f"""path: {OUTPUT_DATASET.as_posix()}
train: train/images
val: valid/images
test: test/images

names:
  0: rice_bag
"""

    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print("\nCreated:")
    print(yaml_path)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("RICE BAG 10 - V9 POLYGON TO BBOX CONVERTER")
    print("=" * 60)

    print(f"\nSource dataset:")
    print(SOURCE_DATASET)

    print(f"\nOutput dataset:")
    print(OUTPUT_DATASET)

    # Create output directory
    OUTPUT_DATASET.mkdir(
        parents=True,
        exist_ok=True
    )

    # Convert train / valid / test
    for split in SPLITS:
        convert_split(split)

    # Create data.yaml
    create_data_yaml()

    print("\n" + "=" * 60)
    print("V9 CONVERSION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()