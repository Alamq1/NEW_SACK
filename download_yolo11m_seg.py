from ultralytics import YOLO
import os

print("=" * 70)
print("DOWNLOADING YOLO11M-SEG.PT MODEL")
print("=" * 70)

print("\n⏳ Downloading model...")
model = YOLO('yolo11m-seg.pt')

print("\n✓ Model downloaded successfully!")
print(f"\nModel Path: {model.model_name}")
print("\nNow you can run: python src/test_rice_bag_v5_bbox.py")
