"""
RICE BAG V9 - TEST & INFERENCE

Tests the trained V9 model on test images and generates annotated output.

Loads: models/rice_bag_10_v9/rice_bag_10_v9_best.pt
Tests: data/Rice Bag 10_v9_converted/test/images
Output: runs/rice_bag_10_v9/test_results/
"""

from pathlib import Path
import cv2
from ultralytics import YOLO
import numpy as np
import torch


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "rice_bag_10_v9" / "rice_bag_10_v9_best.pt"
DATASET_ROOT = ROOT / "data" / "Rice Bag 10_v9_converted"
TEST_IMAGES_DIR = DATASET_ROOT / "test" / "images"
RESULTS_DIR = ROOT / "runs" / "rice_bag_10_v9" / "test_results"


# ============================================================
# SETUP
# ============================================================

print("=" * 70)
print("RICE BAG V9 - TEST & INFERENCE")
print("=" * 70)

# Verify paths
if not MODEL_PATH.exists():
    print(f"\n✗ Model not found: {MODEL_PATH}")
    print("Please run training first:")
    print("  python src/train_rice_bag_v9.py")
    exit(1)

if not TEST_IMAGES_DIR.exists():
    print(f"\n✗ Test images directory not found: {TEST_IMAGES_DIR}")
    exit(1)

print(f"\nModel: {MODEL_PATH}")
print(f"Test images: {TEST_IMAGES_DIR}")

# Create output directory
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
print(f"Output: {RESULTS_DIR}")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")
model = YOLO(str(MODEL_PATH))
print(f"✓ Model loaded")
print(f"  Task: {model.task}")


# ============================================================
# INFERENCE SETTINGS
# ============================================================

CONF_THRESHOLD = 0.50  # Confidence threshold
IOU_THRESHOLD = 0.45   # NMS IOU threshold
MAX_DETECTIONS = 500   # Maximum detections per image
DEVICE = 0 if torch.cuda.is_available() else "cpu"


# ============================================================
# RUN INFERENCE
# ============================================================

print("\n" + "-" * 70)
print("RUNNING INFERENCE")
print("-" * 70)

# Get test images
test_images = sorted(TEST_IMAGES_DIR.glob("*"))
print(f"\nFound {len(test_images)} test images")

if len(test_images) == 0:
    print("\nNo test images found. Inference skipped.")
    print("Note: Dataset has no test split.")
    exit(0)

# Process each image
total_detections = 0
results_summary = []

for img_path in test_images:
    print(f"\nProcessing: {img_path.name}")
    
    # Load image
    image = cv2.imread(str(img_path))
    if image is None:
        print(f"  ✗ Failed to load image")
        continue
    
    h, w, _ = image.shape
    
    # Inference
    results = model.predict(
        source=str(img_path),
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        max_det=MAX_DETECTIONS,
        device=DEVICE,
        verbose=False,
    )
    
    result = results[0]
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        num_detections = 0
        confidences = []
        total_detections += num_detections
        results_summary.append({
            "image": img_path.name,
            "detections": num_detections,
            "confidences": confidences,
        })
        print(f"  Detections: {num_detections}")
        continue

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    num_detections = len(boxes)
    total_detections += num_detections

    print(f"  Detections: {num_detections}")

    # Get confidence values
    confidences = confs.tolist()
    avg_conf = np.mean(confidences)
    print(f"  Avg confidence: {avg_conf:.4f}")
    print(f"  Min confidence: {min(confidences):.4f}")
    print(f"  Max confidence: {max(confidences):.4f}")

    # Store result
    results_summary.append({
        "image": img_path.name,
        "detections": num_detections,
        "confidences": confidences,
    })

    # Draw annotations
    annotated_image = image.copy()

    for i in range(len(boxes)):
        x1, y1, x2, y2 = map(int, xyxy[i])
        conf = float(confs[i])
        label = f"Bag {i + 1} ({conf:.2f})"

        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated_image,
            label,
            (x1, max(y1 - 5, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    # Add total detection count at top
    total_label = f"Total: {num_detections} bags"
    cv2.putText(
        annotated_image,
        total_label,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2
    )

    # Save annotated image
    output_path = RESULTS_DIR / f"{img_path.stem}_annotated.jpg"
    cv2.imwrite(str(output_path), annotated_image)
    print(f"  ✓ Saved: {output_path.name}")


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("INFERENCE COMPLETE")
print("=" * 70)

print(f"\nTotal images processed: {len(results_summary)}")
print(f"Total detections: {total_detections}")

if len(results_summary) > 0:
    avg_detections_per_image = total_detections / len(results_summary)
    print(f"Average detections per image: {avg_detections_per_image:.1f}")

print(f"\nDetailed results:")
print(f"{'-' * 70}")

for result in results_summary:
    print(f"\n{result['image']}:")
    print(f"  Detections: {result['detections']}")
    if result['confidences']:
        avg_conf = np.mean(result['confidences'])
        print(f"  Avg confidence: {avg_conf:.4f}")

print(f"\n{'-' * 70}")
print(f"Results saved to: {RESULTS_DIR}")
print("=" * 70)
