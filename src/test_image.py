from ultralytics import YOLO
import cv2
import os

# -----------------------------
# Configuration
# -----------------------------
IMAGE_PATH = "data/images/test.jpg"
OUTPUT_DIR = "outputs/predictions"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Load YOLO Segmentation Model
# -----------------------------
print("Loading YOLO segmentation model...")

model = YOLO("yolo11x-seg.pt")

# -----------------------------
# Run Prediction
# -----------------------------
print("Running prediction...")

results = model.predict(
    source=IMAGE_PATH,
    conf=0.25,
    imgsz=1280,
    device=0
)
print("\n===== DETECTION RESULTS =====")

for result in results:
    if result.boxes is None or len(result.boxes) == 0:
        print("No objects detected.")
    else:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])

            print(
                f"Class: {class_name} | "
                f"Confidence: {confidence:.2f}"
            )

print("=============================\n")

# -----------------------------
# Save Annotated Output
# -----------------------------
annotated_image = results[0].plot()

output_path = os.path.join(
    OUTPUT_DIR,
    "prediction.jpg"
)

cv2.imwrite(output_path, annotated_image)

print("\n==============================")
print("Prediction Completed!")
print(f"Output saved to: {output_path}")
print("==============================\n")