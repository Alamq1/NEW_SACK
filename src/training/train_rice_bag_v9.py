from pathlib import Path
from ultralytics import YOLO
import torch


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_YAML = (
    PROJECT_ROOT
    / "data"
    / "Rice Bag 10_v9_converted"
    / "data.yaml"
)

RUNS_DIR = PROJECT_ROOT / "runs"

MODEL_OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "rice_bag_10_v9"
)

MODEL_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TRAINING CONFIG
# ============================================================

BASE_MODEL = "yolo11m.pt"

EPOCHS = 100

IMAGE_SIZE = 640

BATCH_SIZE = 16

WORKERS = 8


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RICE BAG 10 - V9 TRAINING")
    print("=" * 70)

    print(f"\nProject root : {PROJECT_ROOT}")
    print(f"Dataset      : {DATA_YAML}")
    print(f"Epochs       : {EPOCHS}")
    print(f"Image size   : {IMAGE_SIZE}")
    print(f"Batch size   : {BATCH_SIZE}")

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if torch.cuda.is_available():

        device = 0

        print("\nCUDA AVAILABLE")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    else:

        device = "cpu"

        print("\nWARNING: CUDA NOT AVAILABLE")
        print("Training will run on CPU.")

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading base model...")

    model = YOLO(BASE_MODEL)

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\nStarting V9 training...")

    results = model.train(

        data=str(DATA_YAML),

        epochs=EPOCHS,

        imgsz=IMAGE_SIZE,

        batch=BATCH_SIZE,

        workers=WORKERS,

        device=device,

        project=str(RUNS_DIR),

        name="rice_bag_10_v9",

        exist_ok=True,

        pretrained=True,

        optimizer="auto",

        patience=30,

        save=True,

        plots=True,

        verbose=True
    )

    # --------------------------------------------------------
    # Best model
    # --------------------------------------------------------

    best_model = (
        RUNS_DIR
        / "rice_bag_10_v9"
        / "weights"
        / "best.pt"
    )

    last_model = (
        RUNS_DIR
        / "rice_bag_10_v9"
        / "weights"
        / "last.pt"
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    print(f"\nBest model:")
    print(best_model)

    print(f"\nLast model:")
    print(last_model)

    # --------------------------------------------------------
    # Copy best model to models/
    # --------------------------------------------------------

    if best_model.exists():

        import shutil

        destination = (
            MODEL_OUTPUT_DIR
            / "rice_bag_10_v9_best.pt"
        )

        shutil.copy2(
            best_model,
            destination
        )

        print("\nBest model copied to:")
        print(destination)

    print("\nV9 training finished successfully.")


if __name__ == "__main__":
    main()