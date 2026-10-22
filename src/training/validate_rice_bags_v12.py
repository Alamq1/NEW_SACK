#!/usr/bin/env python3
"""
Rice Bag Dataset Validator - V12
Read-only validation for YOLO instance-segmentation datasets.
NEVER converts, deletes, rewrites, or modifies annotations.
"""

from pathlib import Path
import argparse
import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def find_dataset_root(path: Path) -> Path:
    path = path.resolve()
    if (path / "data.yaml").exists():
        return path
    candidates = list(path.rglob("data.yaml"))
    if len(candidates) == 1:
        return candidates[0].parent
    raise FileNotFoundError("Could not uniquely locate data.yaml")

def read_yaml(root: Path):
    with open(root / "data.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def resolve_split(root: Path, value):
    p = Path(str(value))
    # Resolve relative paths relative to dataset YAML location first.
    return (root / p).resolve() if not p.is_absolute() else p

def image_stem_map(folder: Path):
    if not folder.exists():
        return {}
    return {p.stem: p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS}

def validate_labels(images_dir: Path, labels_dir: Path):
    images = image_stem_map(images_dir)
    labels = {p.stem: p for p in labels_dir.rglob("*.txt")} if labels_dir.exists() else {}

    missing_labels = sorted(set(images) - set(labels))
    orphan_labels = sorted(set(labels) - set(images))
    bbox_records = []
    polygon_records = 0
    malformed = []

    for stem, lp in labels.items():
        with open(lp, "r", encoding="utf-8", errors="replace") as f:
            for line_no, raw in enumerate(f, 1):
                line = raw.strip()
                if not line:
                    continue
                vals = line.split()
                try:
                    nums = [float(x) for x in vals]
                except ValueError:
                    malformed.append((lp, line_no, line))
                    continue

                # YOLO detection = 5 values.
                if len(nums) == 5:
                    bbox_records.append((lp, line_no, line))
                # YOLO segmentation = class + >=3 coordinate pairs.
                elif len(nums) >= 7 and (len(nums) - 1) % 2 == 0:
                    polygon_records += 1
                else:
                    malformed.append((lp, line_no, line))

    return images, labels, missing_labels, orphan_labels, polygon_records, bbox_records, malformed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=".", help="Dataset root or directory containing data.yaml")
    args = ap.parse_args()

    root = find_dataset_root(Path(args.dataset))
    cfg = read_yaml(root)

    print("=" * 72)
    print("RICE BAG DATASET VALIDATION - V12")
    print("=" * 72)
    print(f"Dataset root : {root}")
    print(f"YAML         : {root / 'data.yaml'}")
    print()

    names = cfg.get("names", {})
    print(f"Classes      : {names}")
    print(f"nc           : {cfg.get('nc', 'not specified')}")
    print()

    if isinstance(names, dict):
        class_ids = set(map(int, names.keys()))
    elif isinstance(names, list):
        class_ids = set(range(len(names)))
    else:
        class_ids = set()

    failed = False

    for split in ("train", "val", "test"):
        raw = cfg.get(split)
        if raw is None:
            print(f"[{split}] NOT DEFINED")
            continue

        img_dir = resolve_split(root, raw)
        # Standard YOLO layout: .../images -> sibling .../labels
        label_dir = img_dir.parent / "labels"

        print(f"[{split}]")
        print(f"  images : {img_dir}")
        print(f"  labels : {label_dir}")

        if not img_dir.exists():
            print("  STATUS : MISSING IMAGE DIRECTORY")
            failed = True
            continue

        images, labels, missing, orphan, poly, boxes, bad = validate_labels(img_dir, label_dir)

        print(f"  images             : {len(images)}")
        print(f"  label files        : {len(labels)}")
        print(f"  polygon records    : {poly}")
        print(f"  BBox records       : {len(boxes)}")
        print(f"  malformed records  : {len(bad)}")
        print(f"  images w/o labels  : {len(missing)}")
        print(f"  orphan labels      : {len(orphan)}")

        if boxes:
            failed = True
            print("  BBOX RECORDS FOUND (training must stop):")
            for p, n, line in boxes:
                print(f"    {p}:{n} -> {line}")

        if bad:
            failed = True
            print("  MALFORMED RECORDS FOUND:")
            for p, n, line in bad[:20]:
                print(f"    {p}:{n} -> {line}")

        if missing or orphan:
            failed = True

        print()

    print("=" * 72)
    if failed:
        print("VALIDATION FAILED")
        print("No annotations were modified. Fix the reported dataset issues first.")
        raise SystemExit(2)
    print("VALIDATION PASSED")
    print("Dataset structure and label syntax are safe to proceed to training.")
    print("=" * 72)

if __name__ == "__main__":
    main()
