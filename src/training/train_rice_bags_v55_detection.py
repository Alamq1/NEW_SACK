from pathlib import Path
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]

DATASET = ROOT / "data" / "Rice bag 10 V5" / "data.yaml"

OUTPUT_DIR = ROOT / "outputs"
RUN_NAME = "rice_bag_10_v55_detection"

DEVICE = 0 if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("RICE BAG 10 - V5 DETECTION TRAINING")
print("=" * 70)

print(f"Dataset : {DATASET}")
print(f"Output  : {OUTPUT_DIR / RUN_NAME}")
print(f"Device  : {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU     : {torch.cuda.get_device_name(0)}")

if not DATASET.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET}"
    )

print("\n✓ Converted V5 dataset found")

# ============================================================
# LOAD DETECTION MODEL
# ============================================================

print("\nLoading YOLO11m detection model...")

model = YOLO("yolo11m.pt")

# ============================================================
# TRAIN
# ============================================================

print("\nStarting V5 detection training...")
print("Epochs: 100")

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

    warmup_epochs=3,

    weight_decay=0.0005,

    # Validation
    val=True,

    patience=100,

    # Augmentation
    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.3,

    degrees=5.0,

    translate=0.05,

    scale=0.3,

    fliplr=0.5,

    flipud=0.0,

    mosaic=0.3,

    # Output
    project=str(OUTPUT_DIR),

    name=RUN_NAME,

    exist_ok=True,

    save=True,

    save_period=10,

    plots=True,

    # Reproducibility
    seed=42,

    deterministic=True,

    # Detection
    max_det=1000,

    verbose=True,
)

# ============================================================
# FINAL
# ============================================================

RUN_DIR = OUTPUT_DIR / RUN_NAME

BEST = RUN_DIR / "weights" / "best.pt"

LAST = RUN_DIR / "weights" / "last.pt"

print("\n" + "=" * 70)
print("V5 DETECTION TRAINING COMPLETE")
print("=" * 70)

print(f"\nBest model:")
print(BEST)

print(f"\nLast model:")
print(LAST)

print("\nNow run the Flask testing UI:")
print("python src/test_rice_bag_v5_ui.py")