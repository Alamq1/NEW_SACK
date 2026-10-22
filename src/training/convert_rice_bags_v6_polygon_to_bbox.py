from pathlib import Path
import shutil
import yaml


ROOT = Path(__file__).resolve().parents[2]

SOURCE = ROOT / "data" / "Rice Bag 10"
DEST = ROOT / "data" / "Rice bag 10 V6"

CLASS_ID = 0
CLASS_NAME = "Rice-bag-10"


def polygon_to_bbox(values):

    if len(values) < 6 or len(values) % 2 != 0:
        return None

    xs = values[0::2]
    ys = values[1::2]

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

    xc = (xmin + xmax) / 2
    yc = (ymin + ymax) / 2

    return CLASS_ID, xc, yc, width, height


def convert_label(src, dst):

    converted = 0
    skipped = 0

    with open(src, "r", encoding="utf-8") as f:
        lines = f.readlines()

    output = []

    for line in lines:

        parts = line.strip().split()

        if not parts:
            continue

        try:
            coords = [float(x) for x in parts[1:]]
        except ValueError:
            skipped += 1
            continue

        bbox = polygon_to_bbox(coords)

        if bbox is None:
            skipped += 1
            continue

        cid, xc, yc, w, h = bbox

        output.append(
            f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n"
        )

        converted += 1

    with open(dst, "w", encoding="utf-8") as f:
        f.writelines(output)

    return converted, skipped


print("=" * 70)
print("RICE BAG 10 - V6 POLYGON → BBOX CONVERTER")
print("=" * 70)

print(f"Source : {SOURCE}")
print(f"Output : {DEST}")


if not SOURCE.exists():
    raise FileNotFoundError(f"Source dataset not found:\n{SOURCE}")

if DEST.exists():
    print("\nRemoving previous V6 dataset...")
    shutil.rmtree(DEST)


for split in ["train", "valid"]:

    (DEST / split / "images").mkdir(
        parents=True,
        exist_ok=True
    )

    (DEST / split / "labels").mkdir(
        parents=True,
        exist_ok=True
    )


total_images = 0
total_boxes = 0
total_skipped = 0


for split in ["train", "valid"]:

    src_images = SOURCE / split / "images"
    src_labels = SOURCE / split / "labels"

    dst_images = DEST / split / "images"
    dst_labels = DEST / split / "labels"

    print("\n" + "-" * 70)
    print(f"CONVERTING {split.upper()}")
    print("-" * 70)

    image_files = list(src_images.iterdir())

    split_boxes = 0

    for image in image_files:

        if not image.is_file():
            continue

        shutil.copy2(
            image,
            dst_images / image.name
        )

        total_images += 1

        label = src_labels / f"{image.stem}.txt"
        output_label = dst_labels / f"{image.stem}.txt"

        if not label.exists():

            output_label.write_text(
                "",
                encoding="utf-8"
            )

            continue

        boxes, skipped = convert_label(
            label,
            output_label
        )

        split_boxes += boxes
        total_boxes += boxes
        total_skipped += skipped

    print(f"Images : {len(list(dst_images.iterdir()))}")
    print(f"BBoxes : {split_boxes}")


data_yaml = {
    "train": "../train/images",
    "val": "../valid/images",
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


print("\n" + "=" * 70)
print("V6 CONVERSION COMPLETE")
print("=" * 70)

print(f"Total images : {total_images}")
print(f"Total BBoxes : {total_boxes}")
print(f"Skipped      : {total_skipped}")
print(f"Dataset      : {DEST}")