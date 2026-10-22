from pathlib import Path
from collections import Counter

ROOT = Path("data/Rice bag 11.yolo26")

VALID_CLASSES = {0, 1, 2}
CLASS_NAMES = {
    0: "Rice-bag-10",
    1: "other",
    2: "pile",
}

print("=" * 70)
print("RICE BAG SEGMENTATION DATASET AUDIT")
print("=" * 70)
print(f"Dataset: {ROOT.resolve()}")

total_images = 0
total_labels = 0
total_instances = 0

for split in ["train", "valid", "test"]:

    img_dir = ROOT / split / "images"
    lbl_dir = ROOT / split / "labels"

    print("\n" + "-" * 70)
    print(f"{split.upper()}")
    print("-" * 70)

    if not img_dir.exists():
        print(f"WARNING: Image directory missing: {img_dir}")
        continue

    if not lbl_dir.exists():
        print(f"WARNING: Label directory missing: {lbl_dir}")
        continue

    image_files = [
        p for p in img_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ]

    label_files = list(lbl_dir.glob("*.txt"))

    print(f"Images : {len(image_files)}")
    print(f"Labels : {len(label_files)}")

    total_images += len(image_files)
    total_labels += len(label_files)

    image_stems = {p.stem for p in image_files}
    label_stems = {p.stem for p in label_files}

    missing_labels = image_stems - label_stems
    orphan_labels = label_stems - image_stems

    if missing_labels:
        print(f"WARNING: Images without labels: {len(missing_labels)}")
        for x in list(missing_labels)[:10]:
            print(f"  - {x}")

    if orphan_labels:
        print(f"WARNING: Labels without images: {len(orphan_labels)}")
        for x in list(orphan_labels)[:10]:
            print(f"  - {x}")

    class_counts = Counter()

    malformed = 0
    invalid_class = 0
    invalid_coordinates = 0
    too_few_points = 0
    empty_labels = 0

    for label_file in label_files:

        lines = label_file.read_text().splitlines()

        if not lines:
            empty_labels += 1
            continue

        for line_no, line in enumerate(lines, start=1):

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            # Segmentation requires:
            # class_id + at least 3 (x,y) points
            if len(parts) < 7:
                malformed += 1
                continue

            try:
                class_id = int(parts[0])
            except ValueError:
                malformed += 1
                continue

            if class_id not in VALID_CLASSES:
                invalid_class += 1
                continue

            class_counts[class_id] += 1
            total_instances += 1

            coordinates = parts[1:]

            if len(coordinates) % 2 != 0:
                malformed += 1
                continue

            try:
                coords = [float(x) for x in coordinates]
            except ValueError:
                malformed += 1
                continue

            # Minimum 3 polygon points
            if len(coords) < 6:
                too_few_points += 1
                continue

            # YOLO normalized coordinates must be [0, 1]
            if any(x < 0 or x > 1 for x in coords):
                invalid_coordinates += 1

    print("\nClass distribution:")

    for class_id in sorted(VALID_CLASSES):
        print(
            f"  {class_id} = {CLASS_NAMES[class_id]:15s} : "
            f"{class_counts[class_id]}"
        )

    print("\nLabel quality:")

    print(f"  Empty label files       : {empty_labels}")
    print(f"  Malformed annotations   : {malformed}")
    print(f"  Invalid class IDs       : {invalid_class}")
    print(f"  Invalid coordinates     : {invalid_coordinates}")
    print(f"  Too few polygon points  : {too_few_points}")

print("\n" + "=" * 70)
print("TOTAL")
print("=" * 70)

print(f"Total images     : {total_images}")
print(f"Total label files: {total_labels}")
print(f"Total instances  : {total_instances}")

print("\nAudit completed.")
print("=" * 70)