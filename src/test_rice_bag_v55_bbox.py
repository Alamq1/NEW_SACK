from pathlib import Path
import cv2
import random


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT / "data" / "Rice bag 10 V5"

OUTPUT = ROOT / "outputs" / "rice_bag_10_v5_bbox_visualization"

OUTPUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

NUM_IMAGES = 5


# ============================================================
# READ YOLO BBOX LABEL
# ============================================================

def read_labels(label_path, img_width, img_height):

    boxes = []

    if not label_path.exists():
        return boxes

    with open(label_path, "r") as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            class_id = int(parts[0])

            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            # YOLO normalized -> pixel coordinates

            x1 = int(
                (x_center - width / 2) * img_width
            )

            y1 = int(
                (y_center - height / 2) * img_height
            )

            x2 = int(
                (x_center + width / 2) * img_width
            )

            y2 = int(
                (y_center + height / 2) * img_height
            )

            boxes.append(
                (
                    x1,
                    y1,
                    x2,
                    y2
                )
            )

    return boxes


# ============================================================
# PROCESS SPLIT
# ============================================================

def process_split(split):

    image_dir = DATASET / split / "images"
    label_dir = DATASET / split / "labels"

    if not image_dir.exists():

        print(
            f"Missing: {image_dir}"
        )

        return

    images = list(image_dir.glob("*"))

    if not images:

        print(
            f"No images found in {image_dir}"
        )

        return

    # Random images
    random.shuffle(images)

    images = images[:NUM_IMAGES]

    output_dir = OUTPUT / split
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"\n{'=' * 60}"
    )

    print(
        f"{split.upper()} VISUALIZATION"
    )

    print(
        f"{'=' * 60}"
    )

    for image_path in images:

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            continue

        h, w = image.shape[:2]

        label_path = (
            label_dir
            / f"{image_path.stem}.txt"
        )

        boxes = read_labels(
            label_path,
            w,
            h
        )

        # Draw BBoxes

        for i, (
            x1,
            y1,
            x2,
            y2
        ) in enumerate(boxes, start=1):

            # Clamp coordinates

            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Bag number

            cv2.putText(
                image,
                str(i),
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )

        # Total

        cv2.putText(
            image,
            f"TOTAL BAGS: {len(boxes)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        output_path = (
            output_dir
            / f"visual_{image_path.name}"
        )

        cv2.imwrite(
            str(output_path),
            image
        )

        print(
            f"{image_path.name} -> "
            f"{len(boxes)} boxes"
        )


# ============================================================
# MAIN
# ============================================================

print("=" * 60)
print("RICE BAG 10 - V5 BBOX VISUALIZATION")
print("=" * 60)

print(
    f"\nDataset: {DATASET}"
)

print(
    f"Output : {OUTPUT}"
)


if not DATASET.exists():

    raise FileNotFoundError(
        f"\nDataset not found:\n{DATASET}"
    )


process_split("train")
process_split("valid")


print(
    "\n" + "=" * 60
)

print("VISUALIZATION COMPLETE")

print("=" * 60)

print(
    f"\nOpen this folder:"
)

print(
    OUTPUT
)