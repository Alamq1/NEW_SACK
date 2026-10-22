from pathlib import Path
from PIL import Image
from collections import Counter
import yaml


# ============================================================
# SACK KEYPOINT DATASET AUDIT
# ============================================================
#
# Purpose:
#   1. Check dataset structure
#   2. Check data.yaml
#   3. Detect annotation format
#   4. Validate bounding boxes
#   5. Detect possible keypoint annotations
#   6. Check missing/orphan labels
#   7. Check image sizes
#   8. Give a final dataset health summary
#
# IMPORTANT:
#   We DO NOT modify any dataset files.
#   This script is READ-ONLY.
# ============================================================


# ============================================================
# CONFIG
# ============================================================

DATASET_DIR = Path("data/Sack_keypoint.yolov8")

EXPECTED_SPLITS = [
    "train",
    "valid",
    "test"
]


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# FORMAT DETECTION
# ============================================================

def detect_annotation_format(parts):

    if len(parts) == 5:
        return "YOLO DETECTION"

    elif len(parts) > 5:
        return "POSSIBLE KEYPOINT / POSE"

    else:
        return "INVALID"


# ============================================================
# CHECK BBOX
# ============================================================

def validate_bbox(xc, yc, w, h):

    errors = []

    if not (0 <= xc <= 1):
        errors.append("x_center_out_of_range")

    if not (0 <= yc <= 1):
        errors.append("y_center_out_of_range")

    if not (0 < w <= 1):
        errors.append("width_out_of_range")

    if not (0 < h <= 1):
        errors.append("height_out_of_range")

    return errors


# ============================================================
# INSPECT ONE SPLIT
# ============================================================

def inspect_split(split_name):

    image_dir = DATASET_DIR / split_name / "images"
    label_dir = DATASET_DIR / split_name / "labels"

    print("\n")
    print("=" * 70)
    print(f"{split_name.upper()} DATASET")
    print("=" * 70)

    # --------------------------------------------------------
    # Directory checks
    # --------------------------------------------------------

    if not image_dir.exists():

        print(f"ERROR: Image directory missing:")
        print(f"      {image_dir}")

        return

    if not label_dir.exists():

        print(f"ERROR: Label directory missing:")
        print(f"      {label_dir}")

        return

    # --------------------------------------------------------
    # Find images
    # --------------------------------------------------------

    images = [
        p for p in image_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    labels = list(label_dir.glob("*.txt"))

    # --------------------------------------------------------
    # Match images and labels
    # --------------------------------------------------------

    image_names = {
        p.stem
        for p in images
    }

    label_names = {
        p.stem
        for p in labels
    }

    missing_labels = image_names - label_names

    orphan_labels = label_names - image_names

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    total_objects = 0

    invalid_lines = 0

    invalid_bbox_values = 0

    invalid_class_ids = 0

    empty_label_files = 0

    detection_annotations = 0

    possible_keypoint_annotations = 0

    image_read_errors = 0

    class_counter = Counter()

    format_counter = Counter()

    bbox_errors = Counter()

    # --------------------------------------------------------
    # Inspect labels
    # --------------------------------------------------------

    for label_file in labels:

        try:

            with open(label_file, "r") as f:

                lines = [
                    line.strip()
                    for line in f
                    if line.strip()
                ]

        except Exception as e:

            print(
                f"Could not read label file: "
                f"{label_file} -> {e}"
            )

            continue

        # Empty annotation file
        if not lines:

            empty_label_files += 1

            continue

        for line in lines:

            parts = line.split()

            annotation_format = detect_annotation_format(parts)

            format_counter[annotation_format] += 1

            # ------------------------------------------------
            # Invalid
            # ------------------------------------------------

            if len(parts) < 5:

                invalid_lines += 1

                continue

            # ------------------------------------------------
            # Parse first five values
            # ------------------------------------------------

            try:

                cls = float(parts[0])
                xc = float(parts[1])
                yc = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])

            except ValueError:

                invalid_lines += 1

                continue

            total_objects += 1

            # ------------------------------------------------
            # Class ID
            # ------------------------------------------------

            if cls < 0 or cls != int(cls):

                invalid_class_ids += 1

            else:

                class_counter[int(cls)] += 1

            # ------------------------------------------------
            # BBOX
            # ------------------------------------------------

            errors = validate_bbox(
                xc,
                yc,
                w,
                h
            )

            if errors:

                invalid_bbox_values += 1

                for error in errors:

                    bbox_errors[error] += 1

            # ------------------------------------------------
            # Format
            # ------------------------------------------------

            if len(parts) == 5:

                detection_annotations += 1

            elif len(parts) > 5:

                possible_keypoint_annotations += 1

    # ========================================================
    # IMAGE ANALYSIS
    # ========================================================

    image_sizes = Counter()

    for image in images:

        try:

            with Image.open(image) as im:

                image_sizes[im.size] += 1

        except Exception as e:

            image_read_errors += 1

            print(
                f"Could not read image: "
                f"{image.name} -> {e}"
            )

    # ========================================================
    # RESULTS
    # ========================================================

    print(f"\nImages                  : {len(images)}")

    print(f"Labels                  : {len(labels)}")

    print(f"Objects                 : {total_objects}")

    print(f"Missing labels          : {len(missing_labels)}")

    print(f"Orphan labels           : {len(orphan_labels)}")

    print(f"Empty label files       : {empty_label_files}")

    print(f"Invalid annotation lines: {invalid_lines}")

    print(f"Invalid class IDs       : {invalid_class_ids}")

    print(f"Invalid bbox values     : {invalid_bbox_values}")

    print(f"Image read errors       : {image_read_errors}")

    # ========================================================
    # ANNOTATION FORMAT
    # ========================================================

    print("\nAnnotation formats:")

    for fmt, count in format_counter.items():

        print(
            f"  {fmt:<30} : {count}"
        )

    # ========================================================
    # CLASSES
    # ========================================================

    print("\nClass distribution:")

    if class_counter:

        for cls, count in sorted(class_counter.items()):

            print(
                f"  Class {cls:<5} : {count}"
            )

    else:

        print("  No valid classes found.")

    # ========================================================
    # IMAGE SIZES
    # ========================================================

    print("\nImage sizes:")

    if image_sizes:

        for size, count in image_sizes.items():

            print(
                f"  {size} : {count}"
            )

    else:

        print("  No readable images.")

    # ========================================================
    # BBOX ERRORS
    # ========================================================

    if bbox_errors:

        print("\nBBox problems:")

        for error, count in bbox_errors.items():

            print(
                f"  {error:<25} : {count}"
            )

    # ========================================================
    # WARNINGS
    # ========================================================

    if missing_labels:

        print("\nWARNING:")
        print(
            f"{len(missing_labels)} image(s) "
            f"do not have labels."
        )

    if orphan_labels:

        print("\nWARNING:")
        print(
            f"{len(orphan_labels)} label file(s) "
            f"do not have matching images."
        )

    if invalid_bbox_values:

        print("\nWARNING:")
        print(
            "Invalid bounding boxes detected."
        )

    if possible_keypoint_annotations:

        print("\nINFO:")
        print(
            "Annotations contain more than 5 values."
        )

        print(
            "Possible keypoint/pose format detected."
        )


# ============================================================
# DATA.YAML
# ============================================================

def inspect_yaml():

    yaml_file = DATASET_DIR / "data.yaml"

    print("\n")
    print("=" * 70)
    print("DATA.YAML")
    print("=" * 70)

    if not yaml_file.exists():

        print("ERROR: data.yaml not found!")

        return None

    try:

        with open(yaml_file, "r") as f:

            data = yaml.safe_load(f)

    except Exception as e:

        print(
            f"Could not read data.yaml: {e}"
        )

        return None

    print(
        f"Path   : {data.get('path')}"
    )

    print(
        f"Train  : {data.get('train')}"
    )

    print(
        f"Val    : {data.get('val')}"
    )

    print(
        f"Test   : {data.get('test')}"
    )

    print(
        f"Classes: {data.get('nc')}"
    )

    print(
        f"Names  : {data.get('names')}"
    )

    return data


# ============================================================
# FINAL SUMMARY
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("          RICE SACK KEYPOINT DATASET AUDIT")
    print("=" * 70)

    print(
        f"\nDataset directory:"
        f"\n{DATASET_DIR}"
    )

    # --------------------------------------------------------
    # Dataset directory
    # --------------------------------------------------------

    if not DATASET_DIR.exists():

        print("\nERROR:")
        print(
            "Dataset directory does not exist."
        )

        print(
            f"Expected:\n{DATASET_DIR}"
        )

        return

    # --------------------------------------------------------
    # YAML
    # --------------------------------------------------------

    yaml_data = inspect_yaml()

    # --------------------------------------------------------
    # Splits
    # --------------------------------------------------------

    for split in EXPECTED_SPLITS:

        split_dir = DATASET_DIR / split

        if split_dir.exists():

            inspect_split(split)

        else:

            print(
                f"\nWARNING: {split}/ directory missing."
            )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n")
    print("=" * 70)
    print("                    AUDIT COMPLETE")
    print("=" * 70)

    print(
        "\nIMPORTANT:"
    )

    print(
        "Do NOT start training until we inspect "
        "this audit output."
    )

    print(
        "\nThe audit does not modify your dataset."
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()