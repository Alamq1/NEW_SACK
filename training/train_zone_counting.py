from pathlib import Path
import torch
from ultralytics import YOLO


# ============================================================
# ZONE COUNTING — YOLO26L SEGMENTATION TRAINING
# Version: v1_1280
# ============================================================


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path("/home/aiteam/ZONE_COUNTING")

DATASET_YAML = (
    PROJECT_ROOT
    / "Zone_Counting_dataset"
    / "data.yaml"
)

RUN_PROJECT = PROJECT_ROOT / "runs" / "segment"

RUN_NAME = "zone_counting_yolo26l_seg_v1_1280"


# ============================================================
# 2. MODEL
# ============================================================

MODEL_NAME = "yolo26l-seg.pt"


# ============================================================
# 3. TRAINING CONFIGURATION
# ============================================================

EPOCHS = 150

# High-resolution training for better segmentation/detail
IMAGE_SIZE = 1280

# Start with 8 on the RTX PRO 6000.
# If CUDA OOM occurs, change ONLY this to 4.
BATCH_SIZE = 8

DEVICE = 0

WORKERS = 8

SEED = 42


# ============================================================
# 4. MAIN
# ============================================================

def main():

    print("\n" + "=" * 75)
    print("ZONE COUNTING — YOLO26L SEGMENTATION TRAINING")
    print("=" * 75)

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    print("\n[1] ENVIRONMENT")
    print("-" * 75)

    print(f"PyTorch        : {torch.__version__}")
    print(f"CUDA available : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. "
            "This training script is configured for GPU training."
        )

    gpu_name = torch.cuda.get_device_name(DEVICE)

    gpu_memory = (
        torch.cuda.get_device_properties(DEVICE).total_memory
        / (1024 ** 3)
    )

    print(f"GPU            : {gpu_name}")
    print(f"GPU VRAM       : {gpu_memory:.2f} GB")


    # --------------------------------------------------------
    # Dataset validation
    # --------------------------------------------------------

    print("\n[2] DATASET")
    print("-" * 75)

    if not DATASET_YAML.exists():
        raise FileNotFoundError(
            f"\nDataset YAML not found:\n{DATASET_YAML}"
        )

    print(f"Dataset YAML   : {DATASET_YAML}")

    print("\nDataset configuration:")

    with open(DATASET_YAML, "r", encoding="utf-8") as f:
        print(f.read())


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\n[3] MODEL")
    print("-" * 75)

    print(f"Model          : {MODEL_NAME}")
    print("Task           : Instance Segmentation")
    print("Pretrained     : YES")

    model = YOLO(MODEL_NAME)

    print("\nModel loaded successfully.")


    # --------------------------------------------------------
    # Training configuration
    # --------------------------------------------------------

    print("\n[4] TRAINING CONFIGURATION")
    print("-" * 75)

    print(f"Epochs         : {EPOCHS}")
    print(f"Image size     : {IMAGE_SIZE}")
    print(f"Batch size     : {BATCH_SIZE}")
    print(f"Device         : GPU {DEVICE}")
    print(f"Workers        : {WORKERS}")
    print(f"Seed           : {SEED}")

    print(f"\nRun project    : {RUN_PROJECT}")
    print(f"Run name       : {RUN_NAME}")


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\n[5] STARTING TRAINING")
    print("=" * 75)

    try:

        results = model.train(

            # ------------------------------------------------
            # Dataset
            # ------------------------------------------------

            data=str(DATASET_YAML),

            # ------------------------------------------------
            # Core training
            # ------------------------------------------------

            epochs=EPOCHS,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,

            # ------------------------------------------------
            # GPU
            # ------------------------------------------------

            device=DEVICE,
            workers=WORKERS,

            # ------------------------------------------------
            # Experiment
            # ------------------------------------------------

            project=str(RUN_PROJECT),
            name=RUN_NAME,

            # ------------------------------------------------
            # Validation
            # ------------------------------------------------

            val=True,

            # ------------------------------------------------
            # Checkpoints
            # ------------------------------------------------

            save=True,
            save_period=10,

            # ------------------------------------------------
            # Optimization
            # ------------------------------------------------

            optimizer="auto",

            # ------------------------------------------------
            # Early stopping
            # ------------------------------------------------

            patience=40,

            # ------------------------------------------------
            # Reproducibility
            # ------------------------------------------------

            seed=SEED,

            # ------------------------------------------------
            # Moderate augmentation
            #
            # Dataset is relatively small, so we don't use
            # aggressive augmentation.
            # ------------------------------------------------

            degrees=3.0,
            translate=0.05,
            scale=0.30,

            shear=0.0,
            perspective=0.0,

            fliplr=0.5,
            flipud=0.0,

            mosaic=0.5,
            mixup=0.0,

            # ------------------------------------------------
            # Output / plots
            # ------------------------------------------------

            plots=True,
            verbose=True,
        )

    except RuntimeError as e:

        error_text = str(e).lower()

        if "out of memory" in error_text or "cuda" in error_text:

            print("\n" + "!" * 75)
            print("CUDA / GPU MEMORY ERROR")
            print("!" * 75)

            print(
                "\nIf this is a CUDA Out-Of-Memory error,"
                " change:"
            )

            print("\n    BATCH_SIZE = 8")

            print("\nto:")

            print("\n    BATCH_SIZE = 4")

            print("\nKeep IMAGE_SIZE = 1280.")

        raise


    # --------------------------------------------------------
    # Training completed
    # --------------------------------------------------------

    run_dir = RUN_PROJECT / RUN_NAME

    best_model = run_dir / "weights" / "best.pt"
    last_model = run_dir / "weights" / "last.pt"
    results_csv = run_dir / "results.csv"

    print("\n\n" + "=" * 75)
    print("TRAINING COMPLETED")
    print("=" * 75)

    print(f"\nRun directory:")
    print(run_dir)

    print(f"\nBest model:")
    print(best_model)

    print(f"\nLast model:")
    print(last_model)

    print(f"\nResults:")
    print(results_csv)

    print("\n" + "=" * 75)

    if best_model.exists():
        print("STATUS: BEST MODEL FOUND ✓")
    else:
        print("WARNING: best.pt was not found.")

    print("=" * 75)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()