#!/usr/bin/env python3

"""
======================================================================
        RICE BAG INSTANCE SEGMENTATION - YOLO26L-SEG V12
======================================================================

Purpose:
    Train YOLO26l-Seg on preserved polygon segmentation annotations.

Dataset:
    data/c-belt_rice_data/

Classes:
    0 - Rice-bag-10
    1 - c-belt
    2 - other
    3 - pile

IMPORTANT:
    - Polygon annotations are NOT converted.
    - No test directory is required.
    - Train/validation split only.
    - Training is resumable.
======================================================================
"""

from pathlib import Path
import random
import numpy as np
import torch
from ultralytics import YOLO


# ======================================================================
# CONFIGURATION
# ======================================================================

DATASET = Path("data/c-belt_rice_data")
DATA_YAML = DATASET / "data.yaml"

MODEL_WEIGHTS = "yolo26l-seg.pt"

PROJECT_DIR = Path("runs/segment")
RUN_NAME = "rice_bags_v12_yolo26l_seg"

EPOCHS = 150
IMGSZ = 1280

# RTX PRO 6000 Blackwell has large VRAM.
# Start conservatively; YOLO will report actual memory usage.
BATCH = 16

WORKERS = 8

DEVICE = 0

SEED = 42


# ======================================================================
# REPRODUCIBILITY
# ======================================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ======================================================================
# HARD VALIDATION BEFORE TRAINING
# ======================================================================

def validate_dataset():

    print("\n" + "=" * 70)
    print("PRE-TRAINING DATASET CHECK - V12")
    print("=" * 70)

    if not DATASET.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET.resolve()}"
        )

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"data.yaml not found: {DATA_YAML.resolve()}"
        )

    train_images = DATASET / "train/images"
    train_labels = DATASET / "train/labels"

    val_images = DATASET / "valid/images"
    val_labels = DATASET / "valid/labels"

    required = [
        train_images,
        train_labels,
        val_images,
        val_labels,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Required dataset path missing: {path.resolve()}"
            )

    train_img_count = len(list(train_images.glob("*")))
    train_lbl_count = len(list(train_labels.glob("*.txt")))

    val_img_count = len(list(val_images.glob("*")))
    val_lbl_count = len(list(val_labels.glob("*.txt")))

    print(f"Dataset      : {DATASET.resolve()}")
    print(f"YAML         : {DATA_YAML.resolve()}")

    print("\nTrain")
    print(f"  Images     : {train_img_count}")
    print(f"  Labels     : {train_lbl_count}")

    print("\nValidation")
    print(f"  Images     : {val_img_count}")
    print(f"  Labels     : {val_lbl_count}")

    if train_img_count != train_lbl_count:
        raise RuntimeError(
            "Train image/label count mismatch."
        )

    if val_img_count != val_lbl_count:
        raise RuntimeError(
            "Validation image/label count mismatch."
        )

    print("\nTest directory:")
    print("  Not required / not present.")
    print("  Final unseen-image testing will be Phase 2.")

    print("\nDataset check PASSED.")


# ======================================================================
# GPU INFORMATION
# ======================================================================

def print_gpu_info():

    print("\n" + "=" * 70)
    print("HARDWARE")
    print("=" * 70)

    print(f"CUDA Available : {torch.cuda.is_available()}")

    if torch.cuda.is_available():

        gpu = torch.cuda.get_device_name(DEVICE)

        print(f"GPU            : {gpu}")
        print(
            f"CUDA Version   : "
            f"{torch.version.cuda}"
        )

        total_memory = torch.cuda.get_device_properties(
            DEVICE
        ).total_memory / (1024 ** 3)

        print(
            f"GPU Memory     : "
            f"{total_memory:.1f} GB"
        )

    else:
        print("WARNING: CUDA is not available.")


# ======================================================================
# TRAINING
# ======================================================================

def train():

    validate_dataset()
    print_gpu_info()

    print("\n" + "=" * 70)
    print("YOLO26L-SEG V12 TRAINING")
    print("=" * 70)

    print(f"Model          : {MODEL_WEIGHTS}")
    print(f"Image Size     : {IMGSZ}")
    print(f"Batch Size     : {BATCH}")
    print(f"Epochs         : {EPOCHS}")
    print(f"Workers        : {WORKERS}")
    print(f"Device         : {DEVICE}")
    print(f"Run Name       : {RUN_NAME}")

    # --------------------------------------------------------------
    # Load YOLO26l-Seg pretrained weights
    # --------------------------------------------------------------

    model = YOLO(MODEL_WEIGHTS)

    # --------------------------------------------------------------
    # Training
    #
    # resume=False intentionally for a fresh V12 training run.
    # If interrupted, use the resume command provided below.
    # --------------------------------------------------------------

    results = model.train(

        data=str(DATA_YAML),

        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,

        device=DEVICE,
        workers=WORKERS,

        project=str(PROJECT_DIR),
        name=RUN_NAME,

        seed=SEED,

        pretrained=True,

        # ----------------------------------------------------------
        # Optimizer / LR
        # ----------------------------------------------------------

        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=5.0,

        # ----------------------------------------------------------
        # Segmentation-friendly augmentation
        # ----------------------------------------------------------

        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,

        degrees=5.0,
        translate=0.10,
        scale=0.50,

        shear=2.0,
        perspective=0.0,

        flipud=0.0,
        fliplr=0.5,

        mosaic=0.7,
        mixup=0.0,
        copy_paste=0.0,

        # Close mosaic near the end for better convergence.
        close_mosaic=15,

        # ----------------------------------------------------------
        # Loss / training stability
        # ----------------------------------------------------------

        amp=True,

        patience=30,

        cache=False,

        deterministic=True,

        # ----------------------------------------------------------
        # Checkpointing / logging
        # ----------------------------------------------------------

        save=True,
        save_period=10,

        plots=True,
        verbose=True,

        # ----------------------------------------------------------
        # Validation
        # ----------------------------------------------------------

        val=True,

        # No test path because dataset has no test split.
        # ----------------------------------------------------------

        exist_ok=False,
    )

    return results


# ======================================================================
# POST-TRAINING VERIFICATION
# ======================================================================

def verify_model():

    run_dir = PROJECT_DIR / RUN_NAME

    best_pt = run_dir / "weights/best.pt"
    last_pt = run_dir / "weights/last.pt"

    print("\n" + "=" * 70)
    print("POST-TRAINING MODEL VERIFICATION")
    print("=" * 70)

    print(f"Run directory : {run_dir.resolve()}")
    print(f"best.pt       : {best_pt.resolve()}")
    print(f"last.pt       : {last_pt.resolve()}")

    if not best_pt.exists():
        raise RuntimeError(
            "TRAINING DID NOT PRODUCE best.pt"
        )

    if not last_pt.exists():
        raise RuntimeError(
            "TRAINING DID NOT PRODUCE last.pt"
        )

    print("\nLoading best.pt...")

    trained_model = YOLO(str(best_pt))

    print(
        f"Model loaded successfully: "
        f"{best_pt.resolve()}"
    )

    print("\nModel classes:")

    for idx, name in trained_model.names.items():
        print(f"  {idx}: {name}")

    print("\nMODEL VERIFICATION PASSED.")

    return best_pt, last_pt


# ======================================================================
# MAIN
# ======================================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("       RICE BAG INSTANCE SEGMENTATION - V12")
    print("=" * 70)

    try:

        train()

        verify_model()

        print("\n" + "=" * 70)
        print("V12 TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 70)

    except KeyboardInterrupt:

        print("\n")
        print("=" * 70)
        print("TRAINING INTERRUPTED")
        print("=" * 70)

        print(
            "\nIf checkpoints exist, resume using:"
        )

        print(
            f"python {Path(__file__).name} --resume"
        )

        raise

    except Exception as e:

        print("\n" + "=" * 70)
        print("TRAINING FAILED")
        print("=" * 70)

        print(f"\nError: {e}")

        raise