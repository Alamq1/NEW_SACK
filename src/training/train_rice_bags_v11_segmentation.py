# from pathlib import Path
# from ultralytics import YOLO
# import torch
# import re
# import shutil


# # ============================================================
# # RICE BAG INSTANCE SEGMENTATION - V11
# # ============================================================
# #
# # Change ONLY:
# #
# #     PRETRAINED_MODEL = "yolo11x-seg.pt"
# #
# # Examples:
# #     yolo11n-seg.pt
# #     yolo11s-seg.pt
# #     yolo11m-seg.pt
# #     yolo11l-seg.pt
# #     yolo11x-seg.pt
# #
# # Original pretrained .pt is NEVER modified.
# #
# # ============================================================


# # ------------------------------------------------------------
# # DATASET
# # ------------------------------------------------------------

# DATASET_YAML = "data/Rice bag 11.yolo26/data.yaml"


# # ------------------------------------------------------------
# # MODEL
# # ------------------------------------------------------------

# PRETRAINED_MODEL = "yolo11x-seg.pt"


# # ------------------------------------------------------------
# # TRAINING
# # ------------------------------------------------------------

# EPOCHS = 200
# IMAGE_SIZE = 1280
# BATCH_SIZE = 8
# DEVICE = 0
# WORKERS = 8


# # ------------------------------------------------------------
# # OUTPUT STRUCTURE
# # ------------------------------------------------------------

# RUNS_DIR = Path("runs/segment")

# MODELS_DIR = Path("models/rice_bag_v11")


# # ------------------------------------------------------------
# # EXTRACT MODEL NAME
# # ------------------------------------------------------------

# def get_model_name(model_path):

#     name = Path(model_path).stem

#     name = re.sub(r"-seg$", "", name)
#     name = re.sub(r"_seg$", "", name)

#     return name


# MODEL_NAME = get_model_name(PRETRAINED_MODEL)


# # ------------------------------------------------------------
# # MODEL-SPECIFIC NAMES
# # ------------------------------------------------------------

# RUN_NAME = f"rice_bag_11_{MODEL_NAME}_segmentation"

# RUN_DIR = RUNS_DIR / RUN_NAME

# MODEL_SAVE_DIR = MODELS_DIR / MODEL_NAME

# LAST_CHECKPOINT = RUN_DIR / "weights" / "last.pt"


# # ------------------------------------------------------------
# # FINAL MODEL NAMES
# # ------------------------------------------------------------

# BEST_MODEL_NAME = f"best_{MODEL_NAME}_seg.pt"
# LAST_MODEL_NAME = f"last_{MODEL_NAME}_seg.pt"

# BEST_MODEL_PATH = MODEL_SAVE_DIR / BEST_MODEL_NAME
# LAST_MODEL_PATH = MODEL_SAVE_DIR / LAST_MODEL_NAME


# # ------------------------------------------------------------
# # PRINT CONFIG
# # ------------------------------------------------------------

# print("=" * 80)
# print("RICE BAG INSTANCE SEGMENTATION - V11")
# print("=" * 80)

# print(f"\nPretrained model:")
# print(f"  {PRETRAINED_MODEL}")

# print(f"\nModel name:")
# print(f"  {MODEL_NAME}")

# print(f"\nTraining run:")
# print(f"  {RUN_NAME}")

# print(f"\nRun directory:")
# print(f"  {RUN_DIR.resolve()}")

# print(f"\nPermanent model directory:")
# print(f"  {MODEL_SAVE_DIR.resolve()}")


# # ------------------------------------------------------------
# # ENVIRONMENT
# # ------------------------------------------------------------

# print("\n" + "=" * 80)
# print("ENVIRONMENT")
# print("=" * 80)

# print(f"PyTorch        : {torch.__version__}")
# print(f"CUDA available : {torch.cuda.is_available()}")

# if not torch.cuda.is_available():
#     raise RuntimeError(
#         "CUDA GPU is not available."
#     )

# print(
#     f"GPU            : "
#     f"{torch.cuda.get_device_name(0)}"
# )

# print(
#     f"GPU count      : "
#     f"{torch.cuda.device_count()}"
# )


# # ------------------------------------------------------------
# # DATASET CHECK
# # ------------------------------------------------------------

# if not Path(DATASET_YAML).exists():

#     raise FileNotFoundError(
#         f"\nDataset YAML not found:\n"
#         f"{Path(DATASET_YAML).resolve()}"
#     )


# print("\n" + "=" * 80)
# print("DATASET")
# print("=" * 80)

# print(
#     f"Dataset YAML:\n"
#     f"  {Path(DATASET_YAML).resolve()}"
# )

# print("\nClasses:")
# print("  0 -> Rice-bag-10")
# print("  1 -> other")
# print("  2 -> pile")

# print("\nAnnotation:")
# print("  YOLO Polygon / Instance Segmentation")

# print("\nPolygon -> BBox:")
# print("  DISABLED")


# # ------------------------------------------------------------
# # CREATE DIRECTORIES
# # ------------------------------------------------------------

# MODEL_SAVE_DIR.mkdir(
#     parents=True,
#     exist_ok=True
# )


# # ------------------------------------------------------------
# # RESUME LOGIC
# # ------------------------------------------------------------

# if LAST_CHECKPOINT.exists():

#     print("\n" + "=" * 80)
#     print("CHECKPOINT FOUND")
#     print("=" * 80)

#     print(
#         f"Last checkpoint:\n"
#         f"  {LAST_CHECKPOINT.resolve()}"
#     )

#     print("\nResuming previous training...")

#     model = YOLO(
#         str(LAST_CHECKPOINT)
#     )

#     RESUME = True

# else:

#     print("\n" + "=" * 80)
#     print("STARTING NEW TRAINING")
#     print("=" * 80)

#     print(
#         f"Loading pretrained model:\n"
#         f"  {PRETRAINED_MODEL}"
#     )

#     model = YOLO(
#         PRETRAINED_MODEL
#     )

#     RESUME = False


# # ------------------------------------------------------------
# # TRAINING CONFIG
# # ------------------------------------------------------------

# print("\n" + "=" * 80)
# print("TRAINING CONFIGURATION")
# print("=" * 80)

# print(f"Model          : {PRETRAINED_MODEL}")
# print(f"Epochs         : {EPOCHS}")
# print(f"Image size     : {IMAGE_SIZE}")
# print(f"Batch size     : {BATCH_SIZE}")
# print(f"GPU            : {DEVICE}")
# print(f"Workers        : {WORKERS}")
# print("Early stopping : DISABLED")
# print(f"Resume         : {RESUME}")


# # ------------------------------------------------------------
# # TRAIN
# # ------------------------------------------------------------

# results = model.train(

#     data=DATASET_YAML,

#     epochs=EPOCHS,

#     imgsz=IMAGE_SIZE,

#     batch=BATCH_SIZE,

#     device=DEVICE,

#     workers=WORKERS,

#     pretrained=True,

#     optimizer="auto",

#     lr0=0.01,

#     lrf=0.01,

#     warmup_epochs=5,

#     warmup_momentum=0.8,

#     warmup_bias_lr=0.1,

#     # --------------------------------------------------------
#     # NO EARLY STOPPING
#     # --------------------------------------------------------

#     patience=0,

#     # --------------------------------------------------------
#     # SEGMENTATION
#     # --------------------------------------------------------

#     overlap_mask=True,

#     mask_ratio=4,

#     # --------------------------------------------------------
#     # AUGMENTATION
#     # --------------------------------------------------------

#     degrees=5.0,

#     translate=0.10,

#     scale=0.50,

#     shear=0.0,

#     perspective=0.0,

#     fliplr=0.5,

#     flipud=0.0,

#     hsv_h=0.015,

#     hsv_s=0.40,

#     hsv_v=0.30,

#     mosaic=0.8,

#     close_mosaic=15,

#     copy_paste=0.10,

#     # --------------------------------------------------------
#     # VALIDATION
#     # --------------------------------------------------------

#     val=True,

#     # --------------------------------------------------------
#     # CHECKPOINTS
#     # --------------------------------------------------------

#     save=True,

#     save_period=5,

#     # --------------------------------------------------------
#     # PLOTS
#     # --------------------------------------------------------

#     plots=True,

#     verbose=True,

#     # --------------------------------------------------------
#     # OUTPUT
#     # --------------------------------------------------------

#     project=str(RUNS_DIR),

#     name=RUN_NAME,

#     # --------------------------------------------------------
#     # PERFORMANCE
#     # --------------------------------------------------------

#     amp=True,

#     cache=True,

#     # --------------------------------------------------------
#     # REPRODUCIBILITY
#     # --------------------------------------------------------

#     seed=42,

#     # --------------------------------------------------------
#     # MULTI CLASS
#     # --------------------------------------------------------

#     single_cls=False,

#     # --------------------------------------------------------
#     # MANY SACKS PER IMAGE
#     # --------------------------------------------------------

#     max_det=1000,

#     rect=False,

#     # --------------------------------------------------------
#     # RESUME
#     # --------------------------------------------------------

#     resume=RESUME,
# )


# # ============================================================
# # COPY FINAL MODELS
# # ============================================================

# print("\n" + "=" * 80)
# print("SAVING MODEL-SPECIFIC WEIGHTS")
# print("=" * 80)


# SOURCE_BEST = RUN_DIR / "weights" / "best.pt"
# SOURCE_LAST = RUN_DIR / "weights" / "last.pt"


# # ------------------------------------------------------------
# # BEST MODEL
# # ------------------------------------------------------------

# if SOURCE_BEST.exists():

#     shutil.copy2(
#         SOURCE_BEST,
#         BEST_MODEL_PATH
#     )

#     print(
#         f"\nBest model saved as:\n"
#         f"  {BEST_MODEL_PATH.resolve()}"
#     )

# else:

#     print(
#         "\nWARNING: best.pt was not found."
#     )


# # ------------------------------------------------------------
# # LAST MODEL
# # ------------------------------------------------------------

# if SOURCE_LAST.exists():

#     shutil.copy2(
#         SOURCE_LAST,
#         LAST_MODEL_PATH
#     )

#     print(
#         f"\nLast model saved as:\n"
#         f"  {LAST_MODEL_PATH.resolve()}"
#     )

# else:

#     print(
#         "\nWARNING: last.pt was not found."
#     )


# # ============================================================
# # COMPLETE
# # ============================================================

# print("\n" + "=" * 80)
# print("V11 SEGMENTATION TRAINING COMPLETE")
# print("=" * 80)

# print("\nOriginal pretrained model:")
# print(f"  {PRETRAINED_MODEL}")

# print("\nTraining results:")
# print(f"  {RUN_DIR.resolve()}")

# print("\nPermanent BEST model:")
# print(f"  {BEST_MODEL_PATH.resolve()}")

# print("\nPermanent LAST model:")
# print(f"  {LAST_MODEL_PATH.resolve()}")

# print("\nFor final testing use:")
# print(f"  {BEST_MODEL_PATH.resolve()}")

# print("=" * 80)


# ============================================================================
# BACKUP 2 - PREVIOUS ACTIVE VERSION (yolo26l-seg, fixed batch=4, no
# resume, output under runs/segment/<name>/weights/). Restore the same
# way as backup 1: remove the leading "# " from each line below.
# ============================================================================
#
# from pathlib import Path
# from ultralytics import YOLO
# import torch
# import shutil
# import re
#
#
# # ============================================================
# # RICE BAG INSTANCE SEGMENTATION - V11
# # ============================================================
# #
# # PURPOSE:
# #   Train Rice Bag dataset directly on YOLO Polygon
# #   / Instance Segmentation annotations.
# #
# # IMPORTANT:
# #   - NO Polygon -> BBox conversion
# #   - 3 classes:
# #         0 = Rice-bag-10
# #         1 = other
# #         2 = pile
# #
# #   - Fresh training
# #   - NO resume from old failed V11 run
# #   - Original pretrained .pt is preserved
# #   - Best/Last trained weights are copied with model name
# #
# # ============================================================
#
#
# # ============================================================
# # 1. DATASET
# # ============================================================
#
# DATASET_YAML = "data/Rice bag 11.yolo26/data.yaml"
#
#
# # ============================================================
# # 2. PRETRAINED MODEL
# # ============================================================
# #
# # Change ONLY this line when testing another segmentation model.
# #
# # Examples:
# #
# #   yolo11n-seg.pt
# #   yolo11s-seg.pt
# #   yolo11m-seg.pt
# #   yolo11l-seg.pt
# #   yolo11x-seg.pt
# #
# # ============================================================
#
# PRETRAINED_MODEL = "yolo26l-seg.pt"
#
#
# # ============================================================
# # 3. TRAINING CONFIGURATION
# # ============================================================
#
# EPOCHS = 150
#
# IMAGE_SIZE = 1280
#
# # Batch reduced from 8 -> 4 because previous training
# # experienced CUDA OOM during segmentation mask loss.
# BATCH_SIZE = 4
#
# DEVICE = 0
#
# WORKERS = 8
#
#
# # ============================================================
# # 4. OUTPUT DIRECTORIES
# # ============================================================
#
# # Clean Ultralytics training output
# RUNS_DIR = Path("runs/segment")
#
# # Permanent model storage
# MODELS_DIR = Path("models/rice_bag_v11")
#
#
# # ============================================================
# # 5. AUTOMATIC MODEL NAME
# # ============================================================
#
# def get_model_name(model_path):
#
#     name = Path(model_path).stem
#
#     # Remove -seg / _seg from model name
#     name = re.sub(r"-seg$", "", name)
#     name = re.sub(r"_seg$", "", name)
#
#     return name
#
#
# MODEL_NAME = get_model_name(PRETRAINED_MODEL)
#
#
# # ============================================================
# # 6. MODEL-SPECIFIC RUN NAME
# # ============================================================
#
# RUN_NAME = (
#     f"rice_bag_11_{MODEL_NAME}_segmentation"
# )
#
# RUN_DIR = RUNS_DIR / RUN_NAME
#
#
# # ============================================================
# # 7. PERMANENT MODEL DIRECTORY
# # ============================================================
#
# MODEL_SAVE_DIR = (
#     MODELS_DIR / MODEL_NAME
# )
#
# MODEL_SAVE_DIR.mkdir(
#     parents=True,
#     exist_ok=True
# )
#
#
# # ============================================================
# # 8. FINAL MODEL FILE NAMES
# # ============================================================
#
# PRETRAINED_SAVE_PATH = (
#     MODEL_SAVE_DIR /
#     f"pretrained_{MODEL_NAME}_seg.pt"
# )
#
# BEST_MODEL_PATH = (
#     MODEL_SAVE_DIR /
#     f"best_{MODEL_NAME}_seg.pt"
# )
#
# LAST_MODEL_PATH = (
#     MODEL_SAVE_DIR /
#     f"last_{MODEL_NAME}_seg.pt"
# )
#
#
# # ============================================================
# # 9. PRINT CONFIGURATION
# # ============================================================
#
# print("\n" + "=" * 80)
# print("RICE BAG INSTANCE SEGMENTATION - V11")
# print("=" * 80)
#
# print("\nMODEL")
# print("-" * 80)
#
# print(f"Pretrained model : {PRETRAINED_MODEL}")
# print(f"Model name       : {MODEL_NAME}")
#
# print("\nDATASET")
# print("-" * 80)
#
# print(
#     f"Dataset YAML     : "
#     f"{Path(DATASET_YAML).resolve()}"
# )
#
# print("Annotation       : YOLO Polygon / Segmentation")
# print("BBox conversion  : DISABLED")
#
# print("\nCLASSES")
# print("-" * 80)
#
# print("0 -> Rice-bag-10")
# print("1 -> other")
# print("2 -> pile")
#
# print("\nTRAINING")
# print("-" * 80)
#
# print(f"Epochs           : {EPOCHS}")
# print(f"Image size       : {IMAGE_SIZE}")
# print(f"Batch size       : {BATCH_SIZE}")
# print(f"Device            : GPU {DEVICE}")
# print(f"Workers           : {WORKERS}")
# print("Early stopping    : DISABLED")
# print("Resume            : DISABLED")
#
# print("\nOUTPUT")
# print("-" * 80)
#
# print(
#     f"Run directory:\n"
#     f"  {RUN_DIR.resolve()}"
# )
#
# print(
#     f"\nPermanent model directory:\n"
#     f"  {MODEL_SAVE_DIR.resolve()}"
# )
#
#
# # ============================================================
# # 10. CHECK DATASET
# # ============================================================
#
# if not Path(DATASET_YAML).exists():
#
#     raise FileNotFoundError(
#         "\nDataset YAML not found:\n"
#         f"{Path(DATASET_YAML).resolve()}"
#     )
#
#
# # ============================================================
# # 11. GPU CHECK
# # ============================================================
#
# print("\n" + "=" * 80)
# print("GPU CHECK")
# print("=" * 80)
#
# print(
#     f"\nCUDA available : "
#     f"{torch.cuda.is_available()}"
# )
#
# if not torch.cuda.is_available():
#
#     raise RuntimeError(
#         "\nCUDA GPU is not available."
#     )
#
#
# print(
#     f"GPU             : "
#     f"{torch.cuda.get_device_name(0)}"
# )
#
# print(
#     f"GPU count       : "
#     f"{torch.cuda.device_count()}"
# )
#
#
# # ============================================================
# # 12. FRESH TRAINING
# # ============================================================
# #
# # IMPORTANT:
# #
# # We intentionally DO NOT check last.pt.
# #
# # This means:
# #
# #   old failed training -> NOT resumed
# #
# # Training starts from:
# #
# #   yolo11x-seg.pt
# #
# # ============================================================
#
# print("\n" + "=" * 80)
# print("STARTING FRESH TRAINING")
# print("=" * 80)
#
# print(
#     f"\nLoading pretrained model:\n"
#     f"  {PRETRAINED_MODEL}"
# )
#
# model = YOLO(
#     PRETRAINED_MODEL
# )
#
#
# # ============================================================
# # 13. PRESERVE ORIGINAL PRETRAINED MODEL
# # ============================================================
#
# print("\n" + "=" * 80)
# print("PRESERVING PRETRAINED MODEL")
# print("=" * 80)
#
#
# # Ultralytics may download the model automatically.
# # model.ckpt_path gives the actual checkpoint path when available.
#
# source_pretrained = None
#
# if getattr(model, "ckpt_path", None):
#
#     source_pretrained = Path(
#         model.ckpt_path
#     )
#
# else:
#
#     source_pretrained = Path(
#         PRETRAINED_MODEL
#     )
#
#
# if source_pretrained.exists():
#
#     # Avoid copying file onto itself
#     try:
#
#         same_file = (
#             source_pretrained.resolve()
#             == PRETRAINED_SAVE_PATH.resolve()
#         )
#
#     except Exception:
#
#         same_file = False
#
#
#     if not same_file:
#
#         shutil.copy2(
#             source_pretrained,
#             PRETRAINED_SAVE_PATH
#         )
#
#         print(
#             "\nOriginal pretrained model preserved:"
#         )
#
#         print(
#             f"  {PRETRAINED_SAVE_PATH.resolve()}"
#         )
#
#     else:
#
#         print(
#             "\nPretrained model already exists at:"
#         )
#
#         print(
#             f"  {PRETRAINED_SAVE_PATH.resolve()}"
#         )
#
# else:
#
#     print(
#         "\nWARNING:"
#         "\nCould not locate pretrained .pt file "
#         "for permanent copy."
#     )
#
#
# # ============================================================
# # 14. TRAINING
# # ============================================================
#
# print("\n" + "=" * 80)
# print("TRAINING START")
# print("=" * 80)
#
# print("\nStarting fresh 150-epoch segmentation training...\n")
#
#
# results = model.train(
#
#     # --------------------------------------------------------
#     # DATASET
#     # --------------------------------------------------------
#
#     data=DATASET_YAML,
#
#     # --------------------------------------------------------
#     # BASIC TRAINING
#     # --------------------------------------------------------
#
#     epochs=EPOCHS,
#
#     imgsz=IMAGE_SIZE,
#
#     batch=BATCH_SIZE,
#
#     device=DEVICE,
#
#     workers=WORKERS,
#
#     # --------------------------------------------------------
#     # OPTIMIZER
#     # --------------------------------------------------------
#
#     optimizer="auto",
#
#     lr0=0.01,
#
#     lrf=0.01,
#
#     # --------------------------------------------------------
#     # WARMUP
#     # --------------------------------------------------------
#
#     warmup_epochs=5,
#
#     warmup_momentum=0.8,
#
#     warmup_bias_lr=0.1,
#
#     # --------------------------------------------------------
#     # EARLY STOPPING
#     # --------------------------------------------------------
#     #
#     # patience=0 means don't stop early.
#     #
#     # Training will attempt all 150 epochs.
#     #
#     # --------------------------------------------------------
#
#     patience=0,
#
#     # --------------------------------------------------------
#     # SEGMENTATION
#     # --------------------------------------------------------
#
#     overlap_mask=True,
#
#     mask_ratio=4,
#
#     # --------------------------------------------------------
#     # AUGMENTATION
#     # --------------------------------------------------------
#
#     degrees=5.0,
#
#     translate=0.10,
#
#     scale=0.50,
#
#     shear=0.0,
#
#     perspective=0.0,
#
#     fliplr=0.5,
#
#     flipud=0.0,
#
#     hsv_h=0.015,
#
#     hsv_s=0.40,
#
#     hsv_v=0.30,
#
#     mosaic=0.8,
#
#     close_mosaic=15,
#
#     copy_paste=0.10,
#
#     # --------------------------------------------------------
#     # VALIDATION
#     # --------------------------------------------------------
#
#     val=True,
#
#     # --------------------------------------------------------
#     # SAVE
#     # --------------------------------------------------------
#
#     save=True,
#
#     # Save checkpoint every 5 epochs
#     save_period=5,
#
#     # --------------------------------------------------------
#     # PLOTS
#     # --------------------------------------------------------
#
#     plots=True,
#
#     verbose=True,
#
#     # --------------------------------------------------------
#     # OUTPUT
#     # --------------------------------------------------------
#
#     project=str(RUNS_DIR),
#
#     name=RUN_NAME,
#
#     exist_ok=True,
#
#     # --------------------------------------------------------
#     # PERFORMANCE
#     # --------------------------------------------------------
#
#     amp=True,
#
#     cache=True,
#
#     # --------------------------------------------------------
#     # REPRODUCIBILITY
#     # --------------------------------------------------------
#
#     seed=42,
#
#     # --------------------------------------------------------
#     # MULTI-CLASS
#     # --------------------------------------------------------
#
#     single_cls=False,
#
#     # --------------------------------------------------------
#     # MANY BAGS PER IMAGE
#     # --------------------------------------------------------
#
#     max_det=1000,
#
#     rect=False,
#
#     # --------------------------------------------------------
#     # IMPORTANT
#     # --------------------------------------------------------
#     #
#     # NO resume=True
#     #
#     # This is intentionally a fresh training.
#     #
#     # --------------------------------------------------------
#
#     resume=False,
# )
#
#
# # ============================================================
# # 15. TRAINING OUTPUT PATHS
# # ============================================================
#
# SOURCE_BEST = (
#     RUN_DIR /
#     "weights" /
#     "best.pt"
# )
#
# SOURCE_LAST = (
#     RUN_DIR /
#     "weights" /
#     "last.pt"
# )
#
#
# # ============================================================
# # 16. COPY BEST MODEL
# # ============================================================
#
# print("\n" + "=" * 80)
# print("SAVING FINAL MODEL WEIGHTS")
# print("=" * 80)
#
#
# if SOURCE_BEST.exists():
#
#     shutil.copy2(
#         SOURCE_BEST,
#         BEST_MODEL_PATH
#     )
#
#     print(
#         "\nBEST MODEL:"
#     )
#
#     print(
#         f"  {BEST_MODEL_PATH.resolve()}"
#     )
#
# else:
#
#     print(
#         "\nWARNING: best.pt not found."
#     )
#
#
# # ============================================================
# # 17. COPY LAST MODEL
# # ============================================================
#
# if SOURCE_LAST.exists():
#
#     shutil.copy2(
#         SOURCE_LAST,
#         LAST_MODEL_PATH
#     )
#
#     print(
#         "\nLAST MODEL:"
#     )
#
#     print(
#         f"  {LAST_MODEL_PATH.resolve()}"
#     )
#
# else:
#
#     print(
#         "\nWARNING: last.pt not found."
#     )
#
#
# # ============================================================
# # 18. FINAL SUMMARY
# # ============================================================
#
# print("\n" + "=" * 80)
# print("V11 SEGMENTATION TRAINING COMPLETE")
# print("=" * 80)
#
# print("\nPretrained model:")
# print(
#     f"  {PRETRAINED_SAVE_PATH.resolve()}"
# )
#
# print("\nTraining run:")
# print(
#     f"  {RUN_DIR.resolve()}"
# )
#
# print("\nBest model:")
# print(
#     f"  {BEST_MODEL_PATH.resolve()}"
# )
#
# print("\nLast model:")
# print(
#     f"  {LAST_MODEL_PATH.resolve()}"
# )
#
# print("\nFinal testing should use:")
# print(
#     f"  {BEST_MODEL_PATH.resolve()}"
# )
#
# print("\n" + "=" * 80)
# print("DONE")
# print("=" * 80)
#
# ============================================================================
# END OF BACKUP 2
# ============================================================================


# ============================================================
# RICE BAG INSTANCE SEGMENTATION - V11 (resumable, autobatch)
# ============================================================
#
# PURPOSE:
#   Train on the REAL polygon/segmentation dataset
#   (data/Rice bag 11.yolo26) -- 3 classes, variable-length
#   polygon labels verified directly against the label files
#   before writing this script. NO polygon -> bbox conversion
#   anywhere in this pipeline.
#
# CHANGES FROM BACKUP 2:
#   - Verifies yolo26l-seg.pt is actually a segmentation model
#     before training starts (fails fast with a clear error if not).
#   - Resume support restored: if a checkpoint already exists at
#     the exact output path below, training resumes from it
#     instead of silently starting over.
#   - batch=-1 (Ultralytics AutoBatch) instead of a hardcoded
#     value -- this machine's GPU is shared with other jobs, so
#     the safe batch size changes run to run; AutoBatch measures
#     currently-free VRAM at start time instead of guessing.
#   - Output forced to the exact path requested:
#         runs/rice_bag_v11_segmentation/weights/best.pt
#         runs/rice_bag_v11_segmentation/weights/last.pt
#     (previous version nested under runs/segment/<model-name>/,
#     which the test UI could not find; the run name here doubles
#     as the search key the test UI now filters on).
#
# ============================================================


from pathlib import Path
from ultralytics import YOLO
import torch
import shutil


# ============================================================
# 1. DATASET (verified: real polygon labels, 3 classes)
# ============================================================

DATASET_YAML = "data/Rice bag 11.yolo26/data.yaml"


# ============================================================
# 2. PRETRAINED MODEL
# ============================================================

PRETRAINED_MODEL = "yolo26l-seg.pt"


# ============================================================
# 3. OUTPUT (exact path requested)
# ============================================================
#
# NOTE: Ultralytics does not reliably honor project="runs" as a
# literal final directory for segmentation runs on this Ultralytics
# version -- it nests an extra task-name folder in regardless
# (observed: runs/segment/runs/rice_bag_v11_segmentation/...).
# So we don't trust ANY single fixed path for finding checkpoints;
# find_checkpoint() below searches for it instead. BEST_MODEL_PATH /
# LAST_MODEL_PATH remain the literal, guaranteed location: after
# training we copy the real weights there ourselves.
#
# ============================================================

RUNS_DIR = Path("runs")
RUN_NAME = "rice_bag_v11_segmentation"
RUN_DIR = RUNS_DIR / RUN_NAME

BEST_MODEL_PATH = RUN_DIR / "weights" / "best.pt"
LAST_MODEL_PATH = RUN_DIR / "weights" / "last.pt"


def find_checkpoint(filename):
    """Search anywhere under runs/ for filename (best.pt/last.pt)
    belonging to this run, regardless of how Ultralytics actually
    nested the output directory. Returns the most recently modified
    match, or None."""

    candidates = [
        p for p in RUNS_DIR.rglob(filename)
        if RUN_NAME in str(p)
    ]

    if not candidates:
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


# ============================================================
# 4. TRAINING CONFIGURATION
# ============================================================

EPOCHS = 150

# Reduced from 1280 -> 960 after a real mid-training CUDA OOM at
# epoch 62/150 (see history below). Segmentation mask-loss memory
# scales with imgsz^2 * instance_count, and these images have
# hundreds of overlapping sack instances -- 960 cuts that mask
# memory to ~56% of what 1280 needed, on top of the batch reduction
# below.
IMAGE_SIZE = 960

# CHANGED from -1 (AutoBatch) after a real crash: AutoBatch measured
# free VRAM ONCE at startup (94.4GB free -> picked batch=5), but this
# GPU is shared with other long-running jobs (a ~49GB VLLM engine
# among them) whose usage grows over the run. By epoch 62 only
# 1.56GB was free and training OOM'd inside the segmentation loss.
# AutoBatch has no way to react to that drift mid-run, so we use a
# small fixed batch with real headroom instead.
BATCH_SIZE = 2

DEVICE = 0

WORKERS = 8


# ============================================================
# 5. PRINT CONFIGURATION
# ============================================================

print("\n" + "=" * 80)
print("RICE BAG INSTANCE SEGMENTATION - V11 (resumable)")
print("=" * 80)

print("\nMODEL")
print("-" * 80)
print(f"Pretrained model : {PRETRAINED_MODEL}")

print("\nDATASET")
print("-" * 80)
print(f"Dataset YAML     : {Path(DATASET_YAML).resolve()}")
print("Annotation       : YOLO Polygon / Segmentation")
print("BBox conversion  : DISABLED")

print("\nCLASSES")
print("-" * 80)
print("0 -> Rice-bag-10")
print("1 -> other")
print("2 -> pile")

print("\nTRAINING")
print("-" * 80)
print(f"Epochs           : {EPOCHS}")
print(f"Image size       : {IMAGE_SIZE}")
print(f"Batch size       : {'AutoBatch' if BATCH_SIZE == -1 else BATCH_SIZE}")
print(f"Device           : GPU {DEVICE}")
print(f"Workers          : {WORKERS}")

print("\nOUTPUT")
print("-" * 80)
print(f"Run directory:\n  {RUN_DIR.resolve()}")
print(f"\nBest model will be saved to:\n  {BEST_MODEL_PATH.resolve()}")
print(f"\nLast model will be saved to:\n  {LAST_MODEL_PATH.resolve()}")


# ============================================================
# 6. CHECK DATASET
# ============================================================

if not Path(DATASET_YAML).exists():
    raise FileNotFoundError(
        f"\nDataset YAML not found:\n{Path(DATASET_YAML).resolve()}"
    )


# ============================================================
# 7. GPU CHECK
# ============================================================

print("\n" + "=" * 80)
print("GPU CHECK")
print("=" * 80)

print(f"\nCUDA available : {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise RuntimeError("\nCUDA GPU is not available.")

print(f"GPU             : {torch.cuda.get_device_name(0)}")
print(f"GPU count       : {torch.cuda.device_count()}")

free_mem, total_mem = torch.cuda.mem_get_info(0)
print(
    f"GPU memory free : {free_mem / 1024**3:.1f} GB "
    f"/ {total_mem / 1024**3:.1f} GB total"
)


# ============================================================
# 8. LOAD MODEL + RESUME CHECK
# ============================================================
#
# If a checkpoint already exists ANYWHERE under runs/ for this run
# name (searched via find_checkpoint, not a single guessed path --
# see note in section 3), resume from it. Ultralytics restores the
# original run's config/optimizer state automatically on resume.
# Otherwise start fresh from the pretrained segmentation weights.
#
# ============================================================

found_last = find_checkpoint("last.pt")
RESUME = found_last is not None

if RESUME:

    print("\n" + "=" * 80)
    print("CHECKPOINT FOUND - RESUMING TRAINING")
    print("=" * 80)
    print(f"\nResuming from:\n  {found_last.resolve()}")

    model = YOLO(str(found_last))

else:

    print("\n" + "=" * 80)
    print("STARTING FRESH TRAINING")
    print("=" * 80)
    print(f"\nLoading pretrained model:\n  {PRETRAINED_MODEL}")

    model = YOLO(PRETRAINED_MODEL)


# ============================================================
# 9. VERIFY THE MODEL ACTUALLY SUPPORTS SEGMENTATION
# ============================================================
#
# Fail fast and clearly rather than silently training a detection
# model on segmentation labels (or vice versa).
#
# ============================================================

print("\n" + "=" * 80)
print("VERIFYING MODEL TASK")
print("=" * 80)

print(f"\nModel task: {model.task}")

if model.task != "segment":
    raise RuntimeError(
        f"\n{PRETRAINED_MODEL if not RESUME else found_last} is a "
        f"'{model.task}' model, not a segmentation model. "
        "Instance segmentation requires a '-seg' checkpoint "
        "(e.g. yolo26l-seg.pt), not a plain detection weight file."
    )

print("Confirmed: this is a segmentation model (mask head present).")


# ============================================================
# 10. TRAIN
# ============================================================

print("\n" + "=" * 80)
print("TRAINING START")
print("=" * 80)

if RESUME:

    # On resume, Ultralytics reloads the original run's full config
    # (dataset, augmentation, batch, etc.) from the checkpoint itself.
    # Passing new values here would be ignored with a warning, so we
    # only pass resume=True.
    results = model.train(resume=True)

else:

    results = model.train(

        # ----------------------------------------------------
        # DATASET
        # ----------------------------------------------------

        data=DATASET_YAML,

        # ----------------------------------------------------
        # BASIC TRAINING
        # ----------------------------------------------------

        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,

        # ----------------------------------------------------
        # OPTIMIZER
        # ----------------------------------------------------

        optimizer="auto",
        lr0=0.01,
        lrf=0.01,

        # ----------------------------------------------------
        # WARMUP
        # ----------------------------------------------------

        warmup_epochs=5,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------
        #
        # Validation set is only 3 images -- mAP on that few
        # images is noisy epoch-to-epoch, so early stopping on it
        # risks quitting on an unlucky epoch rather than genuine
        # convergence. Disabled on purpose; epochs are cheap on
        # this GPU relative to the dataset size.
        #
        # ----------------------------------------------------

        patience=0,

        # ----------------------------------------------------
        # SEGMENTATION
        # ----------------------------------------------------

        overlap_mask=True,
        mask_ratio=4,

        # ----------------------------------------------------
        # AUGMENTATION
        # (copy_paste + mosaic specifically help the model learn
        # occluded/touching-instance boundaries, which is the
        # main failure mode we're targeting)
        # ----------------------------------------------------

        degrees=5.0,
        translate=0.10,
        scale=0.50,
        shear=0.0,
        perspective=0.0,
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.015,
        hsv_s=0.40,
        hsv_v=0.30,
        mosaic=0.8,
        close_mosaic=15,
        copy_paste=0.10,

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        val=True,

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        # save_period disabled (-1): this server's shared disk is
        # essentially always near-full from other unrelated projects,
        # and periodic full epoch snapshots (12 of them at ~190MB each
        # last run) directly caused an OSError mid-training. best.pt
        # and last.pt (updated every epoch) are all we actually need.
        save=True,
        save_period=-1,

        # ----------------------------------------------------
        # PLOTS
        # ----------------------------------------------------

        plots=True,
        verbose=True,

        # ----------------------------------------------------
        # OUTPUT (exact path requested)
        # ----------------------------------------------------

        project=str(RUNS_DIR),
        name=RUN_NAME,
        exist_ok=True,

        # ----------------------------------------------------
        # PERFORMANCE
        # ----------------------------------------------------

        amp=True,
        cache=True,

        # ----------------------------------------------------
        # REPRODUCIBILITY
        # ----------------------------------------------------

        seed=42,

        # ----------------------------------------------------
        # MULTI-CLASS
        # ----------------------------------------------------

        single_cls=False,

        # ----------------------------------------------------
        # MANY OVERLAPPING BAGS PER IMAGE
        # ----------------------------------------------------

        max_det=1000,
        rect=False,

        resume=False,
    )


# ============================================================
# 11. COPY REAL WEIGHTS TO THE EXACT REQUESTED PATH
# ============================================================
#
# Ultralytics may have written best.pt/last.pt somewhere other than
# BEST_MODEL_PATH/LAST_MODEL_PATH (see note in section 3). Find the
# real files and copy them to the exact path requested, so that
# path is guaranteed valid regardless of Ultralytics' internal
# directory quirks.
#
# ============================================================

print("\n" + "=" * 80)
print("LOCATING TRAINED WEIGHTS")
print("=" * 80)

real_best = find_checkpoint("best.pt")
real_last = find_checkpoint("last.pt")

BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

if real_best and real_best.resolve() != BEST_MODEL_PATH.resolve():
    print(f"\nFound best.pt at:\n  {real_best.resolve()}")
    shutil.copy2(real_best, BEST_MODEL_PATH)
    print(f"Copied to requested path:\n  {BEST_MODEL_PATH.resolve()}")

if real_last and real_last.resolve() != LAST_MODEL_PATH.resolve():
    print(f"\nFound last.pt at:\n  {real_last.resolve()}")
    shutil.copy2(real_last, LAST_MODEL_PATH)
    print(f"Copied to requested path:\n  {LAST_MODEL_PATH.resolve()}")


# ============================================================
# 12. VALIDATE THE BEST MODEL
# ============================================================

print("\n" + "=" * 80)
print("VALIDATING BEST MODEL")
print("=" * 80)

if BEST_MODEL_PATH.exists():

    val_model = YOLO(str(BEST_MODEL_PATH))

    if val_model.task != "segment":
        raise RuntimeError(
            f"\nbest.pt at {BEST_MODEL_PATH.resolve()} loaded as task "
            f"'{val_model.task}', not 'segment'. Refusing to report success."
        )

    metrics = val_model.val(data=DATASET_YAML, imgsz=IMAGE_SIZE, device=DEVICE)

    print("\nValidation (best.pt):")
    print(f"  Box   mAP50    : {metrics.box.map50:.4f}")
    print(f"  Box   mAP50-95 : {metrics.box.map:.4f}")
    print(f"  Mask  mAP50    : {metrics.seg.map50:.4f}")
    print(f"  Mask  mAP50-95 : {metrics.seg.map:.4f}")

else:

    raise FileNotFoundError(
        f"\nbest.pt was not found anywhere under {RUNS_DIR.resolve()} "
        f"for run '{RUN_NAME}' after training. Training did not "
        "complete successfully."
    )


# ============================================================
# 12. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("V11 SEGMENTATION TRAINING COMPLETE")
print("=" * 80)

print("\nBest model:")
print(f"  {BEST_MODEL_PATH.resolve()}")
print(f"  exists: {BEST_MODEL_PATH.exists()}")

print("\nLast model:")
print(f"  {LAST_MODEL_PATH.resolve()}")
print(f"  exists: {LAST_MODEL_PATH.exists()}")

print("\nFinal testing should use:")
print(f"  {BEST_MODEL_PATH.resolve()}")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)

