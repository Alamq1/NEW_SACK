from ultralytics import YOLO
from pathlib import Path
import torch


# ============================================================
# SACK KEYPOINT PROJECT
# ============================================================
# IMPORTANT:
# Dataset/project is named "keypoint", but the exported
# annotations are YOLO bounding-box annotations.
#
# Therefore we train a DETECTION model.
#
# The testing UI will later convert each detected bbox
# center into a clean keypoint/dot.
# ============================================================


# ============================================================
# PATHS
# ============================================================

DATASET_YAML = "data/Sack_keypoint.yolov8/data.yaml"

PROJECT_DIR = "runs"

RUN_NAME = "sack_keypoint"


# ============================================================
# MODEL
# ============================================================

# Start from pretrained weights.
# X model gives us a strong feature extractor.

MODEL_NAME = "yolo11x.pt"


# ============================================================
# TRAINING SETTINGS
# ============================================================

EPOCHS = 150

IMAGE_SIZE = 640

BATCH_SIZE = 16

DEVICE = 0

WORKERS = 8


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 75)
    print("        RICE SACK KEYPOINT PROJECT - DETECTION TRAINING")
    print("=" * 75)

    print(f"Dataset : {DATASET_YAML}")
    print(f"Model   : {MODEL_NAME}")
    print(f"Epochs  : {EPOCHS}")
    print(f"Image   : {IMAGE_SIZE}")
    print(f"Batch   : {BATCH_SIZE}")
    print(f"Device  : GPU {DEVICE}")

    print("-" * 75)

    # --------------------------------------------------------
    # GPU CHECK
    # --------------------------------------------------------

    if torch.cuda.is_available():

        print("CUDA              : AVAILABLE")
        print(
            "GPU               :",
            torch.cuda.get_device_name(DEVICE)
        )

    else:

        print("WARNING: CUDA NOT AVAILABLE")

    print("-" * 75)

    # --------------------------------------------------------
    # LOAD PRETRAINED MODEL
    # --------------------------------------------------------

    print("\nLoading pretrained YOLO model...")

    model = YOLO(MODEL_NAME)

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print("\nStarting training...\n")

    results = model.train(

        # Dataset
        data=DATASET_YAML,

        # Training duration
        epochs=EPOCHS,

        # Dataset is already 640x640
        imgsz=IMAGE_SIZE,

        # GPU batch
        batch=BATCH_SIZE,

        device=DEVICE,

        workers=WORKERS,

        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        project=PROJECT_DIR,

        name=RUN_NAME,

        exist_ok=True,

        # ----------------------------------------------------
        # OPTIMIZER
        # ----------------------------------------------------

        optimizer="AdamW",

        lr0=0.001,

        lrf=0.01,

        weight_decay=0.0005,

        warmup_epochs=5,

        warmup_momentum=0.8,

        warmup_bias_lr=0.1,

        # ----------------------------------------------------
        # AUGMENTATION
        # ----------------------------------------------------
        # Controlled augmentation because dataset is small.

        hsv_h=0.015,

        hsv_s=0.5,

        hsv_v=0.4,

        degrees=5.0,

        translate=0.05,

        scale=0.5,

        shear=2.0,

        perspective=0.0,

        flipud=0.0,

        fliplr=0.5,

        # Mosaic helps object detection,
        # but we don't want extreme synthetic scenes.

        mosaic=0.5,

        mixup=0.0,

        copy_paste=0.0,

        # ----------------------------------------------------
        # TRAINING CONTROL
        # ----------------------------------------------------

        patience=30,

        pretrained=True,

        amp=True,

        cache="disk",

        verbose=True,

        plots=True,

        save=True,

        save_period=10,

        # ----------------------------------------------------
        # REPRODUCIBILITY
        # ----------------------------------------------------

        seed=42,

        deterministic=True,

    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n")
    print("=" * 75)
    print("                 TRAINING COMPLETE")
    print("=" * 75)

    best_model = Path(
        PROJECT_DIR
    ) / RUN_NAME / "weights" / "best.pt"

    last_model = Path(
        PROJECT_DIR
    ) / RUN_NAME / "weights" / "last.pt"

    print("\nBEST MODEL:")
    print(best_model)

    print("\nLAST MODEL:")
    print(last_model)

    print("\nIMPORTANT:")
    print("Use best.pt for testing.")
    print("=" * 75)


if __name__ == "__main__":
    main()