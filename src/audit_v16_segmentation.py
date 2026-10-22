#!/usr/bin/env python3
"""
V16 Rice Bag Segmentation Dataset Audit
=======================================

Dataset:
    data/yolo_v16_data

Purpose:
    Deep audit of YOLO instance-segmentation annotations before training.

Checks:
    1. Dataset structure
    2. Image <-> label matching
    3. Empty / unreadable labels
    4. YOLO segmentation vs bbox format
    5. Polygon point count
    6. Odd coordinate counts
    7. Non-numeric values
    8. Coordinates outside [0, 1]
    9. Degenerate / near-zero-area polygons
    10. Tiny polygons
    11. Duplicate / near-duplicate polygons
    12. Per-image annotation distribution
    13. Per-split statistics
    14. Class distribution
    15. Final PASS / WARNING / FAIL verdict

IMPORTANT:
    This script DOES NOT modify any dataset files.
"""

from pathlib import Path
from collections import Counter, defaultdict
import math
import statistics
import sys

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow is required.")
    print("Install with:")
    print("    pip install pillow")
    sys.exit(1)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "data" / "yolo_v16_data"

SPLITS = ["train", "valid", "test"]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

# Minimum number of polygon points.
# A valid polygon needs at least 3 points.
MIN_POLYGON_POINTS = 3

# Very small polygon area threshold.
# Area here is calculated in normalized coordinates.
TINY_AREA_THRESHOLD = 0.00001

# Near duplicate IoU threshold.
NEAR_DUPLICATE_IOU = 0.98

# Number of polygon coordinates expected at minimum:
# 3 points = x,y x,y x,y = 6 numbers + class ID
MIN_SEGMENTATION_VALUES = 7


# ============================================================
# COLORS / TERMINAL OUTPUT
# ============================================================

RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"


def ok(text):
    print(f"{GREEN}[PASS]{RESET} {text}")


def warn(text):
    print(f"{YELLOW}[WARN]{RESET} {text}")


def fail(text):
    print(f"{RED}[FAIL]{RESET} {text}")


def info(text):
    print(f"{CYAN}[INFO]{RESET} {text}")


def section(title):
    print()
    print("=" * 80)
    print(f" {title}")
    print("=" * 80)


# ============================================================
# GEOMETRY HELPERS
# ============================================================

def polygon_area(points):
    """
    Shoelace formula.

    points:
        [(x1, y1), (x2, y2), ...]

    Returns:
        absolute normalized polygon area
    """

    if len(points) < 3:
        return 0.0

    area = 0.0

    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]

        area += (x1 * y2) - (x2 * y1)

    return abs(area) / 2.0


def polygon_bbox(points):
    """
    Return normalized bbox:
        xmin, ymin, xmax, ymax
    """

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    return (
        min(xs),
        min(ys),
        max(xs),
        max(ys),
    )


def bbox_iou(box1, box2):
    """
    IoU of two bounding boxes.

    box:
        xmin, ymin, xmax, ymax
    """

    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)

    inter_area = inter_w * inter_h

    area1 = max(0.0, x1_max - x1_min) * max(
        0.0, y1_max - y1_min
    )

    area2 = max(0.0, x2_max - x2_min) * max(
        0.0, y2_max - y2_min
    )

    union = area1 + area2 - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union


def points_to_bbox(points):
    return polygon_bbox(points)


# ============================================================
# DATA STRUCTURES
# ============================================================

class AuditStats:

    def __init__(self, split):
        self.split = split

        self.image_count = 0
        self.label_count = 0

        self.total_annotations = 0

        self.empty_labels = []
        self.invalid_lines = []
        self.bbox_format_lines = []
        self.segmentation_format_lines = []

        self.non_numeric = []
        self.odd_coordinate_count = []
        self.invalid_class_ids = []

        self.out_of_range_coords = []

        self.too_few_points = []
        self.tiny_polygons = []
        self.zero_area_polygons = []

        self.duplicate_polygons = []
        self.near_duplicate_polygons = []

        self.image_without_label = []
        self.label_without_image = []

        self.unreadable_images = []

        self.class_counts = Counter()
        self.bags_per_image = []

        self.polygon_point_counts = []
        self.polygon_areas = []

        self.image_sizes = {}

        self.annotations_by_image = defaultdict(list)

    @property
    def problems(self):
        return (
            len(self.invalid_lines)
            + len(self.non_numeric)
            + len(self.odd_coordinate_count)
            + len(self.invalid_class_ids)
            + len(self.out_of_range_coords)
            + len(self.too_few_points)
            + len(self.zero_area_polygons)
            + len(self.image_without_label)
            + len(self.label_without_image)
            + len(self.unreadable_images)
        )


# ============================================================
# DATASET STRUCTURE
# ============================================================

def check_dataset_structure():

    section("1. DATASET STRUCTURE")

    if not DATASET_ROOT.exists():
        fail(f"Dataset does not exist: {DATASET_ROOT}")
        return False

    ok(f"Dataset found: {DATASET_ROOT}")

    all_ok = True

    for split in SPLITS:

        split_dir = DATASET_ROOT / split
        image_dir = split_dir / "images"
        label_dir = split_dir / "labels"

        print()
        print(f"Split: {split}")

        if not split_dir.exists():
            fail(f"Missing split directory: {split_dir}")
            all_ok = False
            continue

        if not image_dir.exists():
            fail(f"Missing image directory: {image_dir}")
            all_ok = False
        else:
            ok(f"Images directory exists")

        if not label_dir.exists():
            fail(f"Missing label directory: {label_dir}")
            all_ok = False
        else:
            ok(f"Labels directory exists")

    yaml_file = DATASET_ROOT / "data.yaml"

    print()

    if yaml_file.exists():
        ok(f"data.yaml found")
    else:
        warn("data.yaml not found")

    return all_ok


# ============================================================
# YAML INSPECTION
# ============================================================

def inspect_yaml():

    section("2. DATA.YAML INSPECTION")

    yaml_file = DATASET_ROOT / "data.yaml"

    if not yaml_file.exists():
        warn("data.yaml missing. Skipping YAML inspection.")
        return

    try:
        content = yaml_file.read_text(
            encoding="utf-8",
            errors="replace",
        )

        print(content)

        names_line = None

        for line in content.splitlines():
            stripped = line.strip()

            if stripped.startswith("names:"):
                names_line = stripped
                break

        if names_line:
            ok("Class names entry found")
        else:
            warn("Could not automatically detect 'names:' entry")

    except Exception as e:
        fail(f"Could not read data.yaml: {e}")


# ============================================================
# IMAGE DISCOVERY
# ============================================================

def get_images(image_dir):

    return sorted(
        [
            p
            for p in image_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def get_labels(label_dir):

    return sorted(
        [
            p
            for p in label_dir.glob("*.txt")
            if p.is_file()
        ]
    )


# ============================================================
# IMAGE VALIDATION
# ============================================================

def validate_image(image_path, stats):

    try:

        with Image.open(image_path) as img:

            width, height = img.size

            if width <= 0 or height <= 0:
                stats.unreadable_images.append(
                    (str(image_path), "invalid dimensions")
                )
                return None

            # Force actual image loading.
            img.load()

            stats.image_sizes[image_path.stem] = (
                width,
                height,
            )

            return width, height

    except Exception as e:

        stats.unreadable_images.append(
            (str(image_path), str(e))
        )

        return None


# ============================================================
# LABEL PARSING
# ============================================================

def parse_label_line(
    line,
    label_path,
    line_number,
    stats,
):

    stripped = line.strip()

    if not stripped:
        return None

    parts = stripped.split()

    # --------------------------------------------------------
    # Detect obvious bbox format
    # --------------------------------------------------------

    if len(parts) == 5:
        stats.bbox_format_lines.append(
            (
                str(label_path),
                line_number,
                stripped,
            )
        )

        return None

    # --------------------------------------------------------
    # Minimum segmentation line
    #
    # class + x,y x,y x,y
    #
    # = 1 + 6 = 7 values
    # --------------------------------------------------------

    if len(parts) < MIN_SEGMENTATION_VALUES:

        stats.invalid_lines.append(
            (
                str(label_path),
                line_number,
                stripped,
                f"Only {len(parts)} values",
            )
        )

        return None

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    try:

        values = [float(x) for x in parts]

    except ValueError:

        stats.non_numeric.append(
            (
                str(label_path),
                line_number,
                stripped,
            )
        )

        return None

    # --------------------------------------------------------
    # Class ID
    # --------------------------------------------------------

    class_id_raw = values[0]

    if not math.isfinite(class_id_raw):

        stats.invalid_class_ids.append(
            (
                str(label_path),
                line_number,
                class_id_raw,
            )
        )

        return None

    if class_id_raw < 0 or int(class_id_raw) != class_id_raw:

        stats.invalid_class_ids.append(
            (
                str(label_path),
                line_number,
                class_id_raw,
            )
        )

        return None

    class_id = int(class_id_raw)

    stats.class_counts[class_id] += 1

    # --------------------------------------------------------
    # Coordinates
    # --------------------------------------------------------

    coords = values[1:]

    if len(coords) % 2 != 0:

        stats.odd_coordinate_count.append(
            (
                str(label_path),
                line_number,
                len(coords),
            )
        )

        return None

    points = []

    has_invalid_coord = False

    for i in range(0, len(coords), 2):

        x = coords[i]
        y = coords[i + 1]

        if not math.isfinite(x) or not math.isfinite(y):

            has_invalid_coord = True
            break

        if x < 0 or x > 1 or y < 0 or y > 1:

            stats.out_of_range_coords.append(
                (
                    str(label_path),
                    line_number,
                    x,
                    y,
                )
            )

            has_invalid_coord = True
            break

        points.append((x, y))

    if has_invalid_coord:
        return None

    # --------------------------------------------------------
    # Minimum polygon points
    # --------------------------------------------------------

    if len(points) < MIN_POLYGON_POINTS:

        stats.too_few_points.append(
            (
                str(label_path),
                line_number,
                len(points),
            )
        )

        return None

    # --------------------------------------------------------
    # Polygon area
    # --------------------------------------------------------

    area = polygon_area(points)

    stats.polygon_point_counts.append(len(points))
    stats.polygon_areas.append(area)

    if area <= 0:

        stats.zero_area_polygons.append(
            (
                str(label_path),
                line_number,
                area,
            )
        )

    elif area < TINY_AREA_THRESHOLD:

        stats.tiny_polygons.append(
            (
                str(label_path),
                line_number,
                area,
            )
        )

    stats.segmentation_format_lines.append(
        (
            str(label_path),
            line_number,
        )
    )

    return {
        "class_id": class_id,
        "points": points,
        "area": area,
    }


# ============================================================
# DUPLICATE CHECK
# ============================================================

def check_duplicates(stats):

    for image_name, annotations in stats.annotations_by_image.items():

        if len(annotations) < 2:
            continue

        for i in range(len(annotations)):

            box_i = points_to_bbox(
                annotations[i]["points"]
            )

            for j in range(i + 1, len(annotations)):

                box_j = points_to_bbox(
                    annotations[j]["points"]
                )

                iou = bbox_iou(box_i, box_j)

                if iou >= 1.0 - 1e-9:

                    stats.duplicate_polygons.append(
                        (
                            image_name,
                            i,
                            j,
                            iou,
                        )
                    )

                elif iou >= NEAR_DUPLICATE_IOU:

                    stats.near_duplicate_polygons.append(
                        (
                            image_name,
                            i,
                            j,
                            iou,
                        )
                    )


# ============================================================
# SPLIT AUDIT
# ============================================================

def audit_split(split):

    section(f"3. AUDITING SPLIT: {split.upper()}")

    stats = AuditStats(split)

    image_dir = DATASET_ROOT / split / "images"
    label_dir = DATASET_ROOT / split / "labels"

    if not image_dir.exists() or not label_dir.exists():
        fail(f"Missing images/labels directory for {split}")
        return stats

    images = get_images(image_dir)
    labels = get_labels(label_dir)

    stats.image_count = len(images)
    stats.label_count = len(labels)

    info(f"Images : {len(images)}")
    info(f"Labels : {len(labels)}")

    image_map = {
        p.stem: p
        for p in images
    }

    label_map = {
        p.stem: p
        for p in labels
    }

    # --------------------------------------------------------
    # Image -> label matching
    # --------------------------------------------------------

    image_names = set(image_map.keys())
    label_names = set(label_map.keys())

    missing_labels = sorted(
        image_names - label_names
    )

    orphan_labels = sorted(
        label_names - image_names
    )

    stats.image_without_label.extend(
        missing_labels
    )

    stats.label_without_image.extend(
        orphan_labels
    )

    if missing_labels:
        fail(
            f"{len(missing_labels)} image(s) have no label file"
        )

        for name in missing_labels[:20]:
            print(f"    - {name}")

    else:
        ok("Every image has a matching label file")

    if orphan_labels:
        fail(
            f"{len(orphan_labels)} label(s) have no matching image"
        )

        for name in orphan_labels[:20]:
            print(f"    - {name}")

    else:
        ok("Every label has a matching image")

    # --------------------------------------------------------
    # Validate images
    # --------------------------------------------------------

    for image_path in images:

        validate_image(
            image_path,
            stats,
        )

    if stats.unreadable_images:

        fail(
            f"{len(stats.unreadable_images)} unreadable image(s)"
        )

        for item in stats.unreadable_images[:20]:
            print(f"    - {item[0]} -> {item[1]}")

    else:
        ok("All images are readable")

    # --------------------------------------------------------
    # Parse labels
    # --------------------------------------------------------

    for label_path in labels:

        try:

            content = label_path.read_text(
                encoding="utf-8",
                errors="replace",
            )

        except Exception as e:

            stats.invalid_lines.append(
                (
                    str(label_path),
                    0,
                    "",
                    f"Cannot read label: {e}",
                )
            )

            continue

        lines = content.splitlines()

        if not any(line.strip() for line in lines):

            stats.empty_labels.append(
                str(label_path)
            )

            continue

        image_annotation_count = 0

        for line_number, line in enumerate(
            lines,
            start=1,
        ):

            parsed = parse_label_line(
                line,
                label_path,
                line_number,
                stats,
            )

            if parsed is not None:

                image_annotation_count += 1

                stats.annotations_by_image[
                    label_path.stem
                ].append(parsed)

        stats.total_annotations += image_annotation_count

        stats.bags_per_image.append(
            image_annotation_count
        )

    # --------------------------------------------------------
    # Duplicate detection
    # --------------------------------------------------------

    check_duplicates(stats)

    # --------------------------------------------------------
    # Split summary
    # --------------------------------------------------------

    print()
    print("-" * 80)
    print(f"{split.upper()} SUMMARY")
    print("-" * 80)

    print(
        f"Images                    : {stats.image_count}"
    )

    print(
        f"Label files               : {stats.label_count}"
    )

    print(
        f"Total valid annotations   : {stats.total_annotations}"
    )

    print(
        f"Empty label files         : {len(stats.empty_labels)}"
    )

    print(
        f"BBox-format lines         : {len(stats.bbox_format_lines)}"
    )

    print(
        f"Segmentation lines        : {len(stats.segmentation_format_lines)}"
    )

    print(
        f"Invalid lines             : {len(stats.invalid_lines)}"
    )

    print(
        f"Non-numeric lines         : {len(stats.non_numeric)}"
    )

    print(
        f"Odd coordinate lines      : {len(stats.odd_coordinate_count)}"
    )

    print(
        f"Invalid class IDs         : {len(stats.invalid_class_ids)}"
    )

    print(
        f"Out-of-range coordinates  : {len(stats.out_of_range_coords)}"
    )

    print(
        f"Too few polygon points    : {len(stats.too_few_points)}"
    )

    print(
        f"Zero-area polygons        : {len(stats.zero_area_polygons)}"
    )

    print(
        f"Tiny polygons             : {len(stats.tiny_polygons)}"
    )

    print(
        f"Exact duplicate polygons  : {len(stats.duplicate_polygons)}"
    )

    print(
        f"Near duplicate polygons   : {len(stats.near_duplicate_polygons)}"
    )

    # --------------------------------------------------------
    # Format verdict
    # --------------------------------------------------------

    if stats.bbox_format_lines:

        fail(
            "BBox-format annotations detected!"
        )

    elif stats.segmentation_format_lines:

        ok(
            "Annotations contain YOLO polygon segmentation data"
        )

    else:

        warn(
            "No valid segmentation annotations detected"
        )

    return stats


# ============================================================
# STATISTICS
# ============================================================

def print_statistics(all_stats):

    section("4. DATASET STATISTICS")

    total_images = sum(
        s.image_count
        for s in all_stats
    )

    total_labels = sum(
        s.label_count
        for s in all_stats
    )

    total_annotations = sum(
        s.total_annotations
        for s in all_stats
    )

    print(f"Total images        : {total_images}")
    print(f"Total label files   : {total_labels}")
    print(f"Total annotations   : {total_annotations}")

    print()

    print(
        f"{'Split':<10}"
        f"{'Images':>10}"
        f"{'Labels':>10}"
        f"{'Objects':>12}"
        f"{'Avg Obj/Img':>15}"
    )

    print("-" * 60)

    for stats in all_stats:

        if stats.image_count:

            avg = (
                stats.total_annotations
                / stats.image_count
            )

        else:
            avg = 0.0

        print(
            f"{stats.split:<10}"
            f"{stats.image_count:>10}"
            f"{stats.label_count:>10}"
            f"{stats.total_annotations:>12}"
            f"{avg:>15.2f}"
        )

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    print()
    print("CLASS DISTRIBUTION")
    print("-" * 40)

    total_classes = Counter()

    for stats in all_stats:
        total_classes.update(
            stats.class_counts
        )

    if total_classes:

        for class_id, count in sorted(
            total_classes.items()
        ):

            print(
                f"Class {class_id:<5}: {count} annotations"
            )

    else:

        warn("No valid class annotations found")

    # --------------------------------------------------------
    # Polygon statistics
    # --------------------------------------------------------

    all_point_counts = []

    all_areas = []

    for stats in all_stats:

        all_point_counts.extend(
            stats.polygon_point_counts
        )

        all_areas.extend(
            stats.polygon_areas
        )

    if all_point_counts:

        print()
        print("POLYGON GEOMETRY STATISTICS")
        print("-" * 40)

        print(
            f"Min polygon points : {min(all_point_counts)}"
        )

        print(
            f"Max polygon points : {max(all_point_counts)}"
        )

        print(
            f"Mean polygon points: "
            f"{statistics.mean(all_point_counts):.2f}"
        )

        print(
            f"Median points      : "
            f"{statistics.median(all_point_counts):.2f}"
        )

    if all_areas:

        print()
        print("NORMALIZED POLYGON AREA")
        print("-" * 40)

        print(
            f"Minimum : {min(all_areas):.8f}"
        )

        print(
            f"Maximum : {max(all_areas):.8f}"
        )

        print(
            f"Mean    : {statistics.mean(all_areas):.8f}"
        )

        print(
            f"Median  : {statistics.median(all_areas):.8f}"
        )


# ============================================================
# DETAILED PROBLEM REPORT
# ============================================================

def print_problem_report(all_stats):

    section("5. DETAILED PROBLEM REPORT")

    found_problem = False

    # --------------------------------------------------------
    # Empty labels
    # --------------------------------------------------------

    for stats in all_stats:

        if stats.empty_labels:

            found_problem = True

            warn(
                f"{stats.split}: "
                f"{len(stats.empty_labels)} empty label file(s)"
            )

            for item in stats.empty_labels[:20]:
                print(f"    {item}")

    # --------------------------------------------------------
    # BBox lines
    # --------------------------------------------------------

    for stats in all_stats:

        if stats.bbox_format_lines:

            found_problem = True

            fail(
                f"{stats.split}: "
                f"{len(stats.bbox_format_lines)} bbox-format line(s)"
            )

            for item in stats.bbox_format_lines[:20]:

                print(
                    f"    {item[0]}:"
                    f"{item[1]} -> {item[2]}"
                )

    # --------------------------------------------------------
    # Invalid lines
    # --------------------------------------------------------

    for stats in all_stats:

        if stats.invalid_lines:

            found_problem = True

            fail(
                f"{stats.split}: "
                f"{len(stats.invalid_lines)} invalid line(s)"
            )

            for item in stats.invalid_lines[:20]:

                print(
                    f"    {item[0]}:"
                    f"{item[1]} -> {item[3]}"
                )

    # --------------------------------------------------------
    # Non-numeric
    # --------------------------------------------------------

    for stats in all_stats:

        if stats.non_numeric:

            found_problem = True

            fail(
                f"{stats.split}: "
                f"{len(stats.non_numeric)} non-numeric line(s)"
            )

    # --------------------------------------------------------
    # Odd coordinates
    # --------------------------------------------------------

    for stats in all_stats:

        if stats.odd_coordinate_count:

            found_problem = True

            fail(
                f"{stats.split}: "
                f"{len(stats.odd_coordinate_count)} odd-coordinate line(s)"
            )

    # --------------------------------------------------------
    # Invalid class IDs
    # --------------------------------------------------------

    for stats in all_stats:

        if stats.invalid_class_ids:

            found_problem = True

            fail(
                f"{stats.split}: "
                f"{len(stats.invalid_class_ids)} invalid class ID(s)"
            )

    # --------------------------------------------------------
    # Out-of-range coordinates
    # --------------------------------------------------------

    for stats in all_stats:

        if stats.out_of_range_coords:

            found_problem = True

            fail(
                f"{stats.split}: "
                f"{len(stats.out_of_range_coords)} out-of-range coordinate issue(s)"
            )

            for item in stats.out_of_range_coords[:20]:

                print(
                    f"    {item[0]}:"
                    f"{item[1]} -> x={item[2]}, y={item[3]}"
                )

    # --------------------------------------------------------
    # Too few points
    # --------------------------------------------------------

    for stats in all_stats:

        if stats.too_few_points:

            found_problem = True

            fail(
                f"{stats.split}: "
                f"{len(stats.too_few_points)} polygon(s) with <3 points"
            )

    # --------------------------------------------------------
    # Zero area
    # --------------------------------------------------------

    for stats in all_stats:

        if stats.zero_area_polygons:

            found_problem = True

            fail(
                f"{stats.split}: "
                f"{len(stats.zero_area_polygons)} zero-area polygon(s)"
            )

    # --------------------------------------------------------
    # Tiny polygons
    # --------------------------------------------------------

    for stats in all_stats:

        if stats.tiny_polygons:

            found_problem = True

            warn(
                f"{stats.split}: "
                f"{len(stats.tiny_polygons)} tiny polygon(s)"
            )

            for item in stats.tiny_polygons[:10]:

                print(
                    f"    {item[0]}:"
                    f"{item[1]} -> area={item[2]:.8f}"
                )

    # --------------------------------------------------------
    # Duplicate polygons
    # --------------------------------------------------------

    for stats in all_stats:

        if stats.duplicate_polygons:

            found_problem = True

            fail(
                f"{stats.split}: "
                f"{len(stats.duplicate_polygons)} exact duplicate polygon pair(s)"
            )

            for item in stats.duplicate_polygons[:20]:

                print(
                    f"    Image={item[0]} "
                    f"polygon {item[1]} vs {item[2]} "
                    f"IoU={item[3]:.4f}"
                )

    # --------------------------------------------------------
    # Near duplicates
    # --------------------------------------------------------

    for stats in all_stats:

        if stats.near_duplicate_polygons:

            found_problem = True

            warn(
                f"{stats.split}: "
                f"{len(stats.near_duplicate_polygons)} near-duplicate polygon pair(s)"
            )

            for item in stats.near_duplicate_polygons[:20]:

                print(
                    f"    Image={item[0]} "
                    f"polygon {item[1]} vs {item[2]} "
                    f"IoU={item[3]:.4f}"
                )

    # --------------------------------------------------------
    # Missing files
    # --------------------------------------------------------

    for stats in all_stats:

        if stats.image_without_label:

            found_problem = True

            fail(
                f"{stats.split}: "
                f"{len(stats.image_without_label)} image(s) without labels"
            )

        if stats.label_without_image:

            found_problem = True

            fail(
                f"{stats.split}: "
                f"{len(stats.label_without_image)} label(s) without images"
            )

    # --------------------------------------------------------
    # Unreadable images
    # --------------------------------------------------------

    for stats in all_stats:

        if stats.unreadable_images:

            found_problem = True

            fail(
                f"{stats.split}: "
                f"{len(stats.unreadable_images)} unreadable image(s)"
            )

    if not found_problem:

        ok("No structural annotation problems detected.")


# ============================================================
# FINAL VERDICT
# ============================================================

def final_verdict(all_stats):

    section("6. FINAL V16 DATASET VERDICT")

    total_bbox = sum(
        len(s.bbox_format_lines)
        for s in all_stats
    )

    total_invalid = sum(
        len(s.invalid_lines)
        + len(s.non_numeric)
        + len(s.odd_coordinate_count)
        + len(s.invalid_class_ids)
        + len(s.out_of_range_coords)
        + len(s.too_few_points)
        + len(s.zero_area_polygons)
        for s in all_stats
    )

    total_duplicates = sum(
        len(s.duplicate_polygons)
        for s in all_stats
    )

    total_near_duplicates = sum(
        len(s.near_duplicate_polygons)
        for s in all_stats
    )

    total_images = sum(
        s.image_count
        for s in all_stats
    )

    total_annotations = sum(
        s.total_annotations
        for s in all_stats
    )

    print()

    print(f"Total images              : {total_images}")
    print(f"Total valid annotations   : {total_annotations}")
    print(f"BBox-format annotations   : {total_bbox}")
    print(f"Critical annotation errors: {total_invalid}")
    print(f"Exact duplicate polygons  : {total_duplicates}")
    print(f"Near duplicate polygons   : {total_near_duplicates}")

    print()
    print("-" * 80)

    # --------------------------------------------------------
    # FAIL conditions
    # --------------------------------------------------------

    if total_bbox > 0:

        fail(
            "FINAL VERDICT: FAIL"
        )

        print()
        print(
            "Reason: bbox-format annotations were detected."
        )

        print(
            "Do NOT start segmentation training yet."
        )

        return False

    if total_invalid > 0:

        fail(
            "FINAL VERDICT: FAIL"
        )

        print()
        print(
            "Reason: critical annotation errors were detected."
        )

        print(
            "Fix the dataset before training."
        )

        return False

    # --------------------------------------------------------
    # WARNING conditions
    # --------------------------------------------------------

    if total_duplicates > 0:

        warn(
            "Exact duplicate polygons exist."
        )

        print(
            "Dataset should be cleaned before final training."
        )

        return False

    if total_near_duplicates > 0:

        warn(
            "Near-duplicate polygons exist."
        )

        print(
            "Review them visually before final training."
        )

    # --------------------------------------------------------
    # PASS
    # --------------------------------------------------------

    ok(
        "FINAL VERDICT: DATASET STRUCTURE + LABEL FORMAT PASS"
    )

    print()
    print(
        "The dataset is suitable to proceed toward YOLO26l-Seg training,"
    )

    print(
        "subject to reviewing any WARNINGs above."
    )

    print()
    print(
        "IMPORTANT: A format PASS does NOT guarantee high model accuracy."
    )

    print(
        "Model performance will still depend on dataset size,"
    )

    print(
        "scene diversity, polygon quality, occlusion, lighting,"
    )

    print(
        "camera angle, and train/validation distribution."
    )

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("          V16 RICE BAG SEGMENTATION DATASET AUDIT")
    print("=" * 80)
    print()
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Dataset      : {DATASET_ROOT}")
    print()

    # --------------------------------------------------------
    # Structure
    # --------------------------------------------------------

    structure_ok = check_dataset_structure()

    if not structure_ok:

        print()
        fail(
            "Dataset structure is incomplete."
        )

        print(
            "Fix the structure before continuing."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # YAML
    # --------------------------------------------------------

    inspect_yaml()

    # --------------------------------------------------------
    # Split audits
    # --------------------------------------------------------

    all_stats = []

    for split in SPLITS:

        stats = audit_split(split)

        all_stats.append(stats)

    # --------------------------------------------------------
    # Global statistics
    # --------------------------------------------------------

    print_statistics(all_stats)

    # --------------------------------------------------------
    # Detailed problems
    # --------------------------------------------------------

    print_problem_report(all_stats)

    # --------------------------------------------------------
    # Final verdict
    # --------------------------------------------------------

    passed = final_verdict(all_stats)

    print()
    print("=" * 80)
    print("                    AUDIT COMPLETE")
    print("=" * 80)
    print()

    if passed:

        print(
            f"{GREEN}{BOLD}"
            "V16 AUDIT COMPLETED SUCCESSFULLY"
            f"{RESET}"
        )

    else:

        print(
            f"{RED}{BOLD}"
            "V16 AUDIT FOUND ISSUES — DO NOT TRAIN YET"
            f"{RESET}"
        )

    print()

    # Do not force non-zero exit code for warnings.
    # Only structure failures above exit immediately.


if __name__ == "__main__":
    main()