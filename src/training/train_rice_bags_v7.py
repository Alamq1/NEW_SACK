from pathlib import Path
import torch
from ultralytics import YOLO


# ============================================================
# RICE BAG 10 - V7 DETECTION TRAINING
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATASET = (
    ROOT
    / "data"
    / "Rice bag 10 V7 BBOX"
    / "data.yaml"
)

OUTPUT_DIR = ROOT / "outputs"

RUN_NAME = "rice_bag_10_v7_detection"


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("RICE BAG 10 - V7 DETECTION TRAINING")
print("=" * 70)

print(f"Dataset : {DATASET}")
print(f"Output  : {OUTPUT_DIR / RUN_NAME}")


# ============================================================
# GPU
# ============================================================

if torch.cuda.is_available():

    DEVICE = 0

    print(f"GPU     : {torch.cuda.get_device_name(0)}")
    print(f"CUDA    : {torch.version.cuda}")

else:

    DEVICE = "cpu"

    print("WARNING: CUDA not available. Using CPU.")


# ============================================================
# DATASET CHECK
# ============================================================

if not DATASET.exists():

    raise FileNotFoundError(
        f"\nDataset not found:\n{DATASET}"
    )

print("\n✓ V7 BBOX dataset found")


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_NAME = "yolo11m.pt"

print(f"\nLoading model: {MODEL_NAME}")

model = YOLO(MODEL_NAME)


# ============================================================
# TRAINING
# ============================================================

results = model.train(

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    data=str(DATASET),

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    epochs=100,

    imgsz=1280,

    batch=8,

    device=DEVICE,

    workers=4,

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer="AdamW",

    lr0=0.001,

    lrf=0.01,

    warmup_epochs=3,

    weight_decay=0.0005,

    # --------------------------------------------------------
    # AUGMENTATION
    # --------------------------------------------------------

    hsv_h=0.015,

    hsv_s=0.4,

    hsv_v=0.3,

    degrees=3.0,

    translate=0.05,

    scale=0.2,

    shear=0.0,

    perspective=0.0,

    fliplr=0.5,

    flipud=0.0,

    mosaic=0.3,

    mixup=0.0,

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    val=True,

    patience=30,

    # --------------------------------------------------------
    # DETECTION
    # --------------------------------------------------------

    conf=0.25,

    iou=0.50,

    max_det=1000,

    single_cls=True,

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    project=str(OUTPUT_DIR),

    name=RUN_NAME,

    exist_ok=True,

    save=True,

    save_period=10,

    plots=True,

    # --------------------------------------------------------
    # REPRODUCIBILITY
    # --------------------------------------------------------

    seed=42,

    deterministic=True,

    verbose=True,
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("V7 TRAINING COMPLETE")
print("=" * 70)

RUN_DIR = OUTPUT_DIR / RUN_NAME

print(f"\nTraining output:")
print(RUN_DIR)

print("\nBest model:")
print(
    RUN_DIR
    / "weights"
    / "best.pt"
)

print("\nLast model:")
print(
    RUN_DIR
    / "weights"
    / "last.pt"
)

print("\nNext:")
print("1. Check validation metrics")
print("2. Check validation predictions")
print("3. Test on unseen images")
print("4. Build Flask UI")

print("=" * 70)