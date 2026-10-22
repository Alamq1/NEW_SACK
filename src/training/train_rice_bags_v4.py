"""
RICE BAG DETECTION V4 - TRAINING WITH CORRECTED LABELS

V4 uses corrected BBox labels converted from polygon annotations.

Main fixes:
✓ Correct project root path
✓ Correct dataset path
✓ BBox detection labels
✓ 1280 image size for dense/small bags
✓ AdamW optimizer
✓ 200 epochs
✓ Early stopping
✓ Validation monitoring
✓ Training plots
"""

from pathlib import Path
import json
import torch
import pandas as pd

from ultralytics import YOLO


# ============================================================
# PROJECT PATHS
# ============================================================

# train_rice_bags_v4.py
#       ↓
# warehouse_bag_poc/
# ├── data/
# ├── outputs/
# └── src/
#     └── training/
#
# parents[2] = warehouse_bag_poc

ROOT = Path(__file__).resolve().parents[2]

DATASET = ROOT / "data" / "Rice Bag 10" / "data.yaml"

OUTPUT_DIR = ROOT / "outputs"

RUN_NAME = "rice_bag_detection_v4"


print(
    f"""
{'=' * 70}
RICE BAG DETECTION - V4 TRAINING
CORRECTED LABELS (Polygon → BBox Conversion)
{'=' * 70}
"""
)


# ============================================================
# HARDWARE CHECK
# ============================================================

DEVICE = 0 if torch.cuda.is_available() else "cpu"

print(f"Device : {DEVICE}")

if torch.cuda.is_available():

    print(
        f"GPU    : "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"CUDA   : "
        f"{torch.version.cuda}"
    )

else:

    print(
        "WARNING: GPU not available, using CPU (SLOW)"
    )


print(f"Root   : {ROOT}")
print(f"Dataset: {DATASET}")
print(
    f"Output : "
    f"{OUTPUT_DIR / RUN_NAME}"
)


# ============================================================
# DATASET VALIDATION
# ============================================================

print(
    f"\n{'=' * 70}"
)

print(
    "CHECKING DATASET"
)

print(
    f"{'=' * 70}"
)


if not DATASET.exists():

    raise FileNotFoundError(
        f"\nDataset not found:\n{DATASET}\n\n"
        f"Expected location:\n"
        f"{ROOT / 'data' / 'Rice Bag 10'}"
    )


print(
    f"✓ Dataset found:"
)

print(
    f"  {DATASET}"
)


# ============================================================
# CHECK DATASET FOLDERS
# ============================================================

DATASET_ROOT = DATASET.parent

TRAIN_IMAGES = (
    DATASET_ROOT / "train" / "images"
)

TRAIN_LABELS = (
    DATASET_ROOT / "train" / "labels"
)

VALID_IMAGES = (
    DATASET_ROOT / "valid" / "images"
)

VALID_LABELS = (
    DATASET_ROOT / "valid" / "labels"
)


print("\nDataset structure:")

print(
    f"Train images : "
    f"{TRAIN_IMAGES.exists()}"
)

print(
    f"Train labels : "
    f"{TRAIN_LABELS.exists()}"
)

print(
    f"Valid images : "
    f"{VALID_IMAGES.exists()}"
)

print(
    f"Valid labels : "
    f"{VALID_LABELS.exists()}"
)


if not TRAIN_IMAGES.exists():

    raise FileNotFoundError(
        f"Train images folder missing:\n"
        f"{TRAIN_IMAGES}"
    )


if not TRAIN_LABELS.exists():

    raise FileNotFoundError(
        f"Train labels folder missing:\n"
        f"{TRAIN_LABELS}"
    )


if not VALID_IMAGES.exists():

    raise FileNotFoundError(
        f"Validation images folder missing:\n"
        f"{VALID_IMAGES}"
    )


if not VALID_LABELS.exists():

    raise FileNotFoundError(
        f"Validation labels folder missing:\n"
        f"{VALID_LABELS}"
    )


# ============================================================
# COUNT FILES
# ============================================================

train_images = list(
    TRAIN_IMAGES.glob("*")
)

train_labels = list(
    TRAIN_LABELS.glob("*.txt")
)

valid_images = list(
    VALID_IMAGES.glob("*")
)

valid_labels = list(
    VALID_LABELS.glob("*.txt")
)


print(
    "\nDataset counts:"
)

print(
    f"Train images : {len(train_images)}"
)

print(
    f"Train labels : {len(train_labels)}"
)

print(
    f"Valid images : {len(valid_images)}"
)

print(
    f"Valid labels : {len(valid_labels)}"
)


# ============================================================
# LOAD MODEL
# ============================================================

print(
    f"\n{'=' * 70}"
)

print(
    "LOADING MODEL"
)

print(
    f"{'=' * 70}"
)

print(
    "Model: yolo11m.pt"
)


model = YOLO(
    "yolo11m.pt"
)


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

TRAIN_CONFIG = {

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    "data": str(DATASET),

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    "epochs": 200,

    "imgsz": 1280,

    "batch": 8,

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    "device": DEVICE,

    "workers": 4,

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    "optimizer": "AdamW",

    "lr0": 0.01,

    "lrf": 0.01,

    "warmup_epochs": 3.0,

    # --------------------------------------------------------
    # REGULARIZATION
    # --------------------------------------------------------

    "weight_decay": 0.0005,

    "dropout": 0.0,

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    "val": True,

    "patience": 30,

    # --------------------------------------------------------
    # AUGMENTATION
    # --------------------------------------------------------

    "hsv_h": 0.015,

    "hsv_s": 0.5,

    "hsv_v": 0.3,

    "degrees": 5.0,

    "translate": 0.05,

    "scale": 0.3,

    "fliplr": 0.5,

    "flipud": 0.1,

    "mosaic": 1.0,

    # --------------------------------------------------------
    # DETECTION
    # --------------------------------------------------------

    "conf": None,

    "iou": 0.7,

    "max_det": 300,

    "single_cls": False,

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    "project": str(OUTPUT_DIR),

    "name": RUN_NAME,

    "exist_ok": True,

    "save": True,

    "save_period": 10,

    "plots": True,

    # --------------------------------------------------------
    # REPRODUCIBILITY
    # --------------------------------------------------------

    "seed": 42,

    "deterministic": True,
}


# ============================================================
# PRINT CONFIG
# ============================================================

print(
    f"\n{'=' * 70}"
)

print(
    "TRAINING CONFIGURATION"
)

print(
    f"{'=' * 70}"
)

for key, value in TRAIN_CONFIG.items():

    print(
        f"{key:20}: {value}"
    )


# ============================================================
# START TRAINING
# ============================================================

print(
    f"\n{'=' * 70}"
)

print(
    "STARTING V4 TRAINING"
)

print(
    f"{'=' * 70}"
)

print(
    "\nTraining will run for up to 200 epochs."
)

print(
    "Early stopping patience: 30 epochs."
)

print()


results = model.train(
    **TRAIN_CONFIG
)


# ============================================================
# TRAINING COMPLETE
# ============================================================

print(
    f"\n{'=' * 70}"
)

print(
    "V4 TRAINING COMPLETE"
)

print(
    f"{'=' * 70}"
)


RUN_DIR = (
    OUTPUT_DIR
    / RUN_NAME
)


BEST_MODEL = (
    RUN_DIR
    / "weights"
    / "best.pt"
)


LAST_MODEL = (
    RUN_DIR
    / "weights"
    / "last.pt"
)


print(
    f"\nBest model:"
)

print(
    f"  {BEST_MODEL}"
)


print(
    f"\nLast model:"
)

print(
    f"  {LAST_MODEL}"
)


# ============================================================
# RESULTS ANALYSIS
# ============================================================

RESULTS_FILE = (
    RUN_DIR
    / "results.csv"
)


if RESULTS_FILE.exists():

    print(
        f"\n{'=' * 70}"
    )

    print(
        "VALIDATION RESULTS"
    )

    print(
        f"{'=' * 70}"
    )


    df = pd.read_csv(
        RESULTS_FILE
    )


    metric_column = (
        "metrics/mAP50(B)"
    )


    if metric_column in df.columns:

        # Ignore NaN values
        valid_df = df.dropna(
            subset=[metric_column]
        )


        if not valid_df.empty:

            best_idx = (
                valid_df[
                    metric_column
                ].idxmax()
            )


            best_row = (
                valid_df.loc[
                    best_idx
                ]
            )


            print(
                "\nBest Epoch Metrics:"
            )


            print(
                f"Epoch:"
                f" "
                f"{int(best_row['epoch']) + 1}"
            )


            print(
                f"Precision:"
                f" "
                f"{best_row['metrics/precision(B)']:.4f}"
            )


            print(
                f"Recall:"
                f" "
                f"{best_row['metrics/recall(B)']:.4f}"
            )


            print(
                f"mAP50:"
                f" "
                f"{best_row['metrics/mAP50(B)']:.4f}"
            )


            print(
                f"mAP50-95:"
                f" "
                f"{best_row['metrics/mAP50-95(B)']:.4f}"
            )


            if (
                best_row[
                    "metrics/mAP50(B)"
                ] > 0.1
            ):

                print(
                    "\n✅ Model is learning."
                )

            else:

                print(
                    "\n⚠️ Metrics are still low."
                )

                print(
                    "Check validation predictions "
                    "and label quality."
                )


# ============================================================
# VALIDATION VISUALIZATIONS
# ============================================================

print(
    f"\n{'=' * 70}"
)

print(
    "VALIDATION VISUALIZATIONS"
)

print(
    f"{'=' * 70}"
)


possible_files = [

    RUN_DIR / "val_batch0_labels.jpg",

    RUN_DIR / "val_batch0_pred.jpg",

    RUN_DIR / "val_batch1_labels.jpg",

    RUN_DIR / "val_batch1_pred.jpg",

]


for file in possible_files:

    if file.exists():

        print(
            f"✓ {file}"
        )


# ============================================================
# SAVE CONFIG
# ============================================================

CONFIG_FILE = (
    RUN_DIR
    / "training_config_v4.json"
)


with open(
    CONFIG_FILE,
    "w"
) as f:

    json.dump(
        TRAIN_CONFIG,
        f,
        indent=2
    )


print(
    f"\nConfig saved:"
)

print(
    f"  {CONFIG_FILE}"
)


# ============================================================
# FINAL
# ============================================================

print(
    f"\n{'=' * 70}"
)

print(
    "DONE"
)

print(
    f"{'=' * 70}"
)

print(
    "\nNext step:"
)

print(
    "Use best.pt for testing."
)

print(
    "\nDO NOT change parameters yet."
)

print(
    "First inspect:"
)

print(
    "1. results.png"
)

print(
    "2. val_batch0_labels.jpg"
)

print(
    "3. val_batch0_pred.jpg"
)

print(
    "4. Precision / Recall / mAP"
)

print(
    "5. Test on a training image"
)

print(
    "\nThen we diagnose the actual problem."
)