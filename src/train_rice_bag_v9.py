"""
RICE BAG V9 - YOLO11M DETECTION TRAINING

Trains YOLO11m detection model on V9 converted BBox dataset.

Training parameters:
- Model: YOLO11m (detection)
- Epochs: 100
- Image size: 640
- Batch: 16
- Device: GPU if available, else CPU
- Dataset: data/Rice Bag 10_v9_converted/data.yaml

Output:
- runs/rice_bag_10_v9/
- models/rice_bag_10_v9/rice_bag_10_v9_best.pt
"""

from pathlib import Path
import torch
from ultralytics import YOLO
import yaml
import shutil


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATASET_YAML = ROOT / "data" / "Rice Bag 10_v9_converted" / "data.yaml"
RUNS_DIR = ROOT / "runs"
MODELS_DIR = ROOT / "models"
RUN_NAME = "rice_bag_10_v9"
MODEL_NAME = "rice_bag_10_v9_best.pt"


# ============================================================
# SETUP
# ============================================================

print("=" * 70)
print("RICE BAG V9 - YOLO11M DETECTION TRAINING")
print("=" * 70)

# Verify dataset
if not DATASET_YAML.exists():
    raise FileNotFoundError(f"Dataset YAML not found: {DATASET_YAML}")

with open(DATASET_YAML) as f:
    dataset_config = yaml.safe_load(f)

print(f"\nDataset: {DATASET_YAML}")
print(f"✓ data.yaml found")

# Parse dataset config
dataset_path = Path(dataset_config["path"])
train_path = dataset_path / dataset_config["train"]
val_path = dataset_path / dataset_config["val"]
test_path = dataset_path / dataset_config.get("test", "test/images")

# Count images
train_images = len(list((dataset_path / "train" / "images").glob("*")))
valid_images = len(list((dataset_path / "valid" / "images").glob("*")))
test_images = len(list((dataset_path / "test" / "images").glob("*"))) if (dataset_path / "test" / "images").exists() else 0

# Get GPU info
DEVICE = 0 if torch.cuda.is_available() else "cpu"
GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
CUDA_VERSION = torch.version.cuda if torch.cuda.is_available() else "N/A"

# Print pre-training report
print("\n" + "-" * 70)
print("PRE-TRAINING VERIFICATION")
print("-" * 70)

print(f"\nDataset:")
print(f"  Path              : {dataset_path}")
print(f"  Train images      : {train_images}")
print(f"  Validation images : {valid_images}")
print(f"  Test images       : {test_images}")

print(f"\nClasses:")
for class_id, class_name in dataset_config.get("names", {}).items():
    print(f"  {class_id}: {class_name}")

print(f"\nEnvironment:")
print(f"  Device            : {DEVICE}")
print(f"  GPU               : {GPU_NAME}")
print(f"  CUDA              : {CUDA_VERSION}")
print(f"  PyTorch           : {torch.__version__}")

print(f"\nTraining Configuration:")
print(f"  Model             : YOLO11m (detection)")
print(f"  Epochs            : 100")
print(f"  Image size        : 640")
print(f"  Batch size        : 16")
print(f"  Optimizer         : AdamW")
print(f"  Learning rate     : 0.0001")
print(f"  Output directory  : {RUNS_DIR / RUN_NAME}")

print("\n" + "-" * 70)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading YOLO11m detection model...")
model = YOLO("yolo11m.pt")

print(f"✓ Model loaded")
print(f"  Task: {model.task}")
print(f"  Model: {model.model_name}")


# ============================================================
# VERIFY TASK
# ============================================================

if model.task != "detect":
    print(f"\n✗ ERROR: Model task is '{model.task}' but must be 'detect'")
    print("Do NOT train segmentation models for detection tasks!")
    exit(1)

print(f"✓ Task is 'detect' (correct)")


# ============================================================
# TRAIN
# ============================================================

print("\n" + "-" * 70)
print("STARTING TRAINING")
print("-" * 70)
print()

results = model.train(
    data=str(DATASET_YAML),
    epochs=100,
    imgsz=640,
    batch=16,
    device=DEVICE,
    workers=4,
    
    # Optimizer
    optimizer="AdamW",
    lr0=0.0001,
    lrf=0.0001,
    warmup_epochs=3,
    weight_decay=0.0005,
    
    # Validation
    val=True,
    patience=15,
    
    # Augmentation (reduced for cleaner training)
    hsv_h=0.010,
    hsv_s=0.4,
    hsv_v=0.25,
    degrees=3.0,
    translate=0.05,
    scale=0.25,
    fliplr=0.5,
    flipud=0.0,
    mosaic=0.2,
    
    # Output
    project=str(RUNS_DIR),
    name=RUN_NAME,
    exist_ok=True,
    save=True,
    save_period=10,
    plots=True,
    
    # Reproducibility
    seed=42,
    deterministic=True,
    
    # Detection
    max_det=500,
    
    verbose=True,
)


# ============================================================
# POST-TRAINING
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

# Paths
run_dir = RUNS_DIR / RUN_NAME
best_pt = run_dir / "weights" / "best.pt"
last_pt = run_dir / "weights" / "last.pt"

print(f"\nTraining output:")
print(f"  {run_dir}")

if best_pt.exists():
    print(f"\n✓ Best model: {best_pt}")
else:
    print(f"\n✗ Best model not found: {best_pt}")

if last_pt.exists():
    print(f"✓ Last model: {last_pt}")
else:
    print(f"✗ Last model not found: {last_pt}")

# Copy best model to models directory
model_output_dir = MODELS_DIR / RUN_NAME
model_output_dir.mkdir(parents=True, exist_ok=True)
model_output_path = model_output_dir / MODEL_NAME

if best_pt.exists():
    shutil.copy2(best_pt, model_output_path)
    print(f"\n✓ Copied best model to: {model_output_path}")
else:
    print(f"\n✗ Could not copy best model")

print("\n" + "=" * 70)
print("NEXT STEPS")
print("=" * 70)
print(f"\n1. Test the model:")
print(f"   python src/test_rice_bag_v9.py")
print(f"\n2. Run Flask UI:")
print(f"   python src/test_rice_bag_v9_ui.py")
print("\n" + "=" * 70)