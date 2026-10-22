from ultralytics import YOLO
from pathlib import Path
import torch

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_YAML = PROJECT_ROOT / "data" / "yolo_v16_data" / "data.yaml"

MODEL = "yolo26l-seg.pt"

# New run after annotation cleanup
RUN_NAME = "rice_bags_v16_yolo26l_seg_clean"


# ============================================================
# INFORMATION
# ============================================================

print("=" * 80)
print("        V16 YOLO26L SEGMENTATION TRAINING - CLEAN DATASET")
print("=" * 80)

print(f"\nDataset : {DATA_YAML}")
print(f"Model   : {MODEL}")
print(f"Run     : {RUN_NAME}")

print("\nHardware")
print("-" * 40)

print(f"CUDA    : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU     : {torch.cuda.get_device_name(0)}")
    print(f"CUDA    : {torch.version.cuda}")

print("-" * 40)


# ============================================================
# VALIDATION
# ============================================================

if not DATA_YAML.exists():
    raise FileNotFoundError(
        f"Dataset YAML not found:\n{DATA_YAML}"
    )

if not Path(MODEL).exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading YOLO26l-Seg model...")

model = YOLO(MODEL)

print("Model loaded successfully.")


# ============================================================
# TRAINING
# ============================================================

print("\nStarting training...")

results = model.train(

    # Dataset
    data=str(DATA_YAML),

    # Training
    epochs=100,
    imgsz=1280,
    batch=8,

    # GPU
    device=0,
    workers=8,

    # Output
    project=str(PROJECT_ROOT / "runs" / "segment"),
    name=RUN_NAME,

    # Model
    pretrained=True,

    # Optimizer
    optimizer="auto",

    # Early stopping
    patience=20,

    # Performance
    amp=True,
    cache=False,

    # Plots / metrics
    plots=True,

    verbose=True,
)


# ============================================================
# COMPLETION
# ============================================================

RUN_DIR = (
    PROJECT_ROOT
    / "runs"
    / "segment"
    / RUN_NAME
)

WEIGHTS_DIR = RUN_DIR / "weights"


print("\n" + "=" * 80)
print("              TRAINING COMPLETED")
print("=" * 80)

print(f"\nRun directory:")
print(RUN_DIR)

print("\nWeights directory:")
print(WEIGHTS_DIR)

print("\nBest model:")
print(WEIGHTS_DIR / "best.pt")

print("\nLast model:")
print(WEIGHTS_DIR / "last.pt")

print("\n" + "=" * 80)