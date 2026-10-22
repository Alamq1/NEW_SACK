from ultralytics import YOLO
from pathlib import Path
import torch


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_YAML = (
    PROJECT_ROOT
    / "data"
    / "Rice Bag 10"
    / "data.yaml"
)

# Object Detection model
MODEL_NAME = "yolo26l.pt"

OUTPUT_DIR = PROJECT_ROOT / "outputs"


# ============================================================
# GPU CHECK
# ============================================================

print("=" * 60)
print("RICE BAG DETECTION V2 TRAINING")
print("=" * 60)

print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
else:
    raise RuntimeError("CUDA GPU is not available!")


print(f"\nDataset: {DATA_YAML}")
print(f"Model: {MODEL_NAME}")


# ============================================================
# LOAD MODEL
# ============================================================

model = YOLO(MODEL_NAME)


# ============================================================
# TRAIN
# ============================================================

results = model.train(

    # Dataset
    data=str(DATA_YAML),

    # Training
    epochs=100,
    imgsz=960,
    batch=8,

    # GPU
    device=0,
    workers=4,

    # Optimizer
    optimizer="auto",

    # Output
    project=str(OUTPUT_DIR),
    name="rice_bag_detection_v2",
    exist_ok=True,

    # Validation
    val=True,

    # Saving
    save=True,
    save_period=20,

    # Training control
    patience=50,
    seed=42,

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
    mixup=0.0,

    verbose=True
)


print("\n" + "=" * 60)
print("V2 TRAINING COMPLETED")
print("=" * 60)