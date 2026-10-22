from pathlib import Path
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data" / "Rice Bag 10" / "data.yaml"
OUTPUT_DIR = ROOT / "outputs" / "rice_bag_10_v5_seg"
RUN_NAME = "rice_bag_10_v5_seg"

print("=" * 70)
print("RICE BAG 10 - V5 INITIAL TRAINING")
print("ORIGINAL POLYGON LABELS -> YOLO SEGMENTATION")
print("=" * 70)

print(f"Project Root : {ROOT}")
print(f"Dataset      : {DATASET}")
print(f"Output       : {OUTPUT_DIR / RUN_NAME}")

DEVICE = 0 if torch.cuda.is_available() else "cpu"

if torch.cuda.is_available():
    print(f"GPU          : {torch.cuda.get_device_name(0)}")
    print(f"CUDA         : {torch.version.cuda}")
else:
    print("WARNING: CUDA not available; using CPU.")

if not DATASET.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATASET}"
    )

dataset_root = DATASET.parent

required = [
    dataset_root / "train" / "images",
    dataset_root / "train" / "labels",
    dataset_root / "valid" / "images",
    dataset_root / "valid" / "labels",
]

for p in required:
    if not p.exists():
        raise FileNotFoundError(f"Required dataset path missing: {p}")

print("\n✓ Dataset structure OK")

print("\nLoading yolo11m-seg.pt...")
model = YOLO("yolo11m-seg.pt")

print("\nStarting V5 training for EXACTLY 100 epochs...")

model.train(
    data=str(DATASET),
    epochs=100,
    imgsz=1280,
    batch=8,
    device=DEVICE,
    workers=4,

    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    warmup_epochs=3.0,
    weight_decay=0.0005,

    val=True,
    patience=100,

    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.3,
    degrees=5.0,
    translate=0.05,
    scale=0.3,
    fliplr=0.5,
    flipud=0.0,
    mosaic=0.3,

    project=str(OUTPUT_DIR),
    name=RUN_NAME,
    exist_ok=True,
    save=True,
    save_period=10,
    plots=True,

    seed=42,
    deterministic=True,
    overlap_mask=True,
    mask_ratio=4,
    verbose=True,
)

run_dir = OUTPUT_DIR / RUN_NAME

print("\n" + "=" * 70)
print("V5 TRAINING COMPLETE")
print("=" * 70)
print(f"Best model: {run_dir / 'weights' / 'best.pt'}")
print(f"Last model: {run_dir / 'weights' / 'last.pt'}")
print("\nSTOP HERE.")
print("Next step will be the V5 polygon -> BBox converter.")
