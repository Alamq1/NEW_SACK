from pathlib import Path
from ultralytics import YOLO
import torch


# ============================================================
# V15 KEYPOINT DETECTION TRAINING
# ============================================================

# Project root:
# /home/aiteam/NEW_SACK_COUNT
ROOT = Path(__file__).resolve().parents[2]

# Actual V15 dataset
DATASET = (
    ROOT
    / "data"
    / "v_15_keypoints_ready"
    / "v_15_keypoints_ready"
)

DATA = DATASET / "data.yaml"

# Model
MODEL = "yolo26l-pose.pt"

# Training output
PROJECT = ROOT / "runs" / "pose"
RUN_NAME = "v_15_keypoints"


# ============================================================
# INFORMATION
# ============================================================

print("=" * 70)
print("             V15 KEYPOINT DETECTION TRAINING")
print("=" * 70)

print(f"\nProject Root : {ROOT}")
print(f"Dataset      : {DATASET}")
print(f"Data YAML    : {DATA}")
print(f"Model        : {MODEL}")


# ============================================================
# CHECK DATASET
# ============================================================

print("\nChecking dataset...")

if not DATA.exists():
    raise FileNotFoundError(
        f"\nERROR: data.yaml not found!\n"
        f"Expected location:\n{DATA}\n"
    )

train_images = DATASET / "images" / "train"
val_images = DATASET / "images" / "val"

train_labels = DATASET / "labels" / "train"
val_labels = DATASET / "labels" / "val"

required_paths = [
    train_images,
    val_images,
    train_labels,
    val_labels,
]

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(
            f"\nERROR: Required path not found:\n{path}"
        )

print("Dataset structure: OK")


# ============================================================
# HARDWARE
# ============================================================

print("\nHardware")
print("-" * 70)

print("CUDA Available :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU            :", torch.cuda.get_device_name(0))
    print("CUDA Version   :", torch.version.cuda)

print("-" * 70)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading YOLO26L Pose model...")

model = YOLO(MODEL)

print("Model loaded successfully.")


# ============================================================
# TRAINING
# ============================================================

print("\nStarting V15 training...\n")

model.train(
    data=str(DATA),

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------
    epochs=60,
    imgsz=1280,
    batch=8,

    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------
    device=0,
    workers=4,

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------
    amp=True,

    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------
    patience=20,

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------
    project=str(PROJECT),
    name=RUN_NAME,
    exist_ok=False,

    # --------------------------------------------------------
    # Checkpoints
    # --------------------------------------------------------
    save=True,
    save_period=10,

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------
    plots=True,

    verbose=True,
)


# ============================================================
# TRAINING COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("                 V15 TRAINING COMPLETED")
print("=" * 70)

print("\nBest Model:")
print(PROJECT / RUN_NAME / "weights" / "best.pt")

print("\nLast Model:")
print(PROJECT / RUN_NAME / "weights" / "last.pt")

print("\nTraining results:")
print(PROJECT / RUN_NAME)