# ============================================================
# FINAL C-BELT SACK COUNTING
# YOLO26L INSTANCE SEGMENTATION TRAINING
# ============================================================

from pathlib import Path
import torch
from ultralytics import YOLO


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_PATH = "yolo26l-seg.pt"

DATASET_YAML = "data/Final_C_belt_yolo26/data.yaml"

# Ultralytics automatically creates this directory
PROJECT_DIR = "runs/segment"
RUN_NAME = "final_c_belt_sack_yolo26l_seg_v1"

EPOCHS = 150
IMAGE_SIZE = 1280

# -1 = automatically determine suitable batch size
BATCH_SIZE = -1

DEVICE = 0
WORKERS = 8

PATIENCE = 30
SEED = 42


# ============================================================
# 2. PATHS
# ============================================================

model_path = Path(MODEL_PATH)
dataset_path = Path(DATASET_YAML)

output_dir = Path(PROJECT_DIR) / RUN_NAME
weights_dir = output_dir / "weights"


# ============================================================
# 3. GPU CHECK
# ============================================================

print("\n" + "=" * 70)
print("FINAL C-BELT SACK — YOLO26L-SEG TRAINING")
print("=" * 70)

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU is not available. Training stopped."
    )

gpu_name = torch.cuda.get_device_name(0)
gpu_memory = (
    torch.cuda.get_device_properties(0).total_memory
    / (1024 ** 3)
)

print(f"GPU             : {gpu_name}")
print(f"GPU VRAM        : {gpu_memory:.2f} GB")


# ============================================================
# 4. VERIFY REQUIRED FILES
# ============================================================

print("\n" + "=" * 70)
print("VERIFYING FILES")
print("=" * 70)

if not model_path.exists():
    raise FileNotFoundError(
        f"Model file not found:\n{model_path.resolve()}"
    )

if not dataset_path.exists():
    raise FileNotFoundError(
        f"Dataset YAML not found:\n{dataset_path.resolve()}"
    )

print(f"Model           : {model_path.resolve()}")
print(f"Dataset         : {dataset_path.resolve()}")


# ============================================================
# 5. DISPLAY TRAINING CONFIGURATION
# ============================================================

print("\n" + "=" * 70)
print("TRAINING CONFIGURATION")
print("=" * 70)

print(f"Model           : {MODEL_PATH}")
print(f"Dataset         : {DATASET_YAML}")
print(f"Epochs          : {EPOCHS}")
print(f"Image size      : {IMAGE_SIZE}")
print(f"Batch size      : {BATCH_SIZE}")
print(f"Device          : GPU {DEVICE}")
print(f"Workers         : {WORKERS}")
print(f"Patience        : {PATIENCE}")
print(f"Seed            : {SEED}")

print("\nOutput directory:")
print(output_dir.resolve())


# ============================================================
# 6. LOAD PRETRAINED YOLO26L-SEG
# ============================================================

print("\n" + "=" * 70)
print("LOADING YOLO26L-SEG")
print("=" * 70)

model = YOLO(MODEL_PATH)

print("✅ YOLO26L-Seg loaded successfully.")


# ============================================================
# 7. START TRAINING
# ============================================================

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)

results = model.train(

    # Dataset
    data=DATASET_YAML,

    # Training
    epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,

    # GPU
    device=DEVICE,
    workers=WORKERS,

    # Pretrained
    pretrained=True,

    # Validation
    val=True,

    # Early stopping
    patience=PATIENCE,

    # Checkpoints
    save=True,
    save_period=10,

    # Output
    project=PROJECT_DIR,
    name=RUN_NAME,
    exist_ok=False,

    # Metrics / plots
    plots=True,

    # Reproducibility
    seed=SEED,
    deterministic=True,

    # Augmentation
    degrees=5.0,
    translate=0.10,
    scale=0.50,
    fliplr=0.5,

    # Mosaic augmentation
    mosaic=1.0,

    # Disable mosaic near end
    close_mosaic=10,

    # Do not cache entire dataset in RAM automatically
    cache=False,

    # Detailed logging
    verbose=True,
)


# ============================================================
# 8. VERIFY OUTPUTS
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print(f"Output directory:\n{output_dir.resolve()}")

best_model = weights_dir / "best.pt"
last_model = weights_dir / "last.pt"

print("\nCheckpoint verification:")

if best_model.exists():
    print(f"✅ best.pt : {best_model.resolve()}")
else:
    print("❌ best.pt NOT FOUND")

if last_model.exists():
    print(f"✅ last.pt : {last_model.resolve()}")
else:
    print("❌ last.pt NOT FOUND")


# ============================================================
# 9. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL TRAINING OUTPUT")
print("=" * 70)

print(
    f"""
Run folder:
{output_dir.resolve()}

Best model:
{best_model.resolve()}

Last model:
{last_model.resolve()}
"""
)

if best_model.exists():
    print("✅ BEST MODEL READY FOR PHASE 3")
else:
    print("⚠️ Training finished but best.pt was not found.")