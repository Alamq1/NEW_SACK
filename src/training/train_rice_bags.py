from ultralytics import YOLO
import torch
from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_YAML = (
    PROJECT_ROOT
    / "data"
    / "Rice Bag dataset"
    / "data.yaml"
)

MODEL_PATH = PROJECT_ROOT / "yolo11x-seg.pt"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# ============================================================
# GPU CHECK
# ============================================================

print("=" * 60)
print("RICE BAG SEGMENTATION TRAINING")
print("=" * 60)

print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
else:
    raise RuntimeError("CUDA GPU is not available!")

print(f"\nDataset: {DATA_YAML}")
print(f"Model: {MODEL_PATH}")

# ============================================================
# LOAD MODEL
# ============================================================

model = YOLO(str(MODEL_PATH))

# ============================================================
# TRAIN
# ============================================================

results = model.train(
    data=str(DATA_YAML),

    epochs=100,
    imgsz=640,
    batch=8,

    device=0,
    workers=4,

    optimizer="auto",

    project=str(OUTPUT_DIR),
    name="rice_bag_segmentation",
    exist_ok=True,

    val=True,

    save=True,
    save_period=10,

    seed=42,
    verbose=True,
)

print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)