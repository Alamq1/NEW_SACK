# # from ultralytics import YOLO
# # from pathlib import Path
# # import cv2

# # # ============================================================
# # # PATHS
# # # ============================================================

# # PROJECT_ROOT = Path(__file__).resolve().parents[2]

# # MODEL_PATH = (
# #     PROJECT_ROOT
# #     / "outputs"
# #     / "rice_bag_segmentation"
# #     / "weights"
# #     / "best.pt"
# # )

# # IMAGE_PATH = PROJECT_ROOT / "data" / "images" / "test.jpg"

# # OUTPUT_DIR = PROJECT_ROOT / "outputs" / "predictions"
# # OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# # OUTPUT_IMAGE = OUTPUT_DIR / "test_prediction.jpg"


# # # ============================================================
# # # LOAD MODEL
# # # ============================================================

# # print("=" * 60)
# # print("RICE BAG MODEL TESTING")
# # print("=" * 60)

# # print(f"Model: {MODEL_PATH}")
# # print(f"Image: {IMAGE_PATH}")

# # model = YOLO(str(MODEL_PATH))


# # # ============================================================
# # # PREDICTION
# # # ============================================================

# # results = model.predict(
# #     source=str(IMAGE_PATH),
# #     conf=0.25,
# #     imgsz=640,
# #     device=0,
# #     verbose=True
# # )


# # # ============================================================
# # # RESULTS
# # # ============================================================

# # result = results[0]

# # if result.boxes is not None:
# #     bag_count = len(result.boxes)
# # else:
# #     bag_count = 0

# # print("\n" + "=" * 60)
# # print(f"DETECTED RICE BAGS: {bag_count}")
# # print("=" * 60)


# # # ============================================================
# # # SAVE RESULT
# # # ============================================================

# # annotated_image = result.plot()

# # cv2.imwrite(str(OUTPUT_IMAGE), annotated_image)

# # print(f"\nPrediction saved at:")
# # print(OUTPUT_IMAGE)
# from ultralytics import YOLO
# from pathlib import Path
# import cv2


# # ============================================================
# # PATHS
# # ============================================================

# PROJECT_ROOT = Path(__file__).resolve().parents[2]

# MODEL_PATH = (
#     PROJECT_ROOT
#     / "outputs"
#     / "rice_bag_segmentation"
#     / "weights"
#     / "best.pt"
# )

# IMAGE_PATH = PROJECT_ROOT / "data" / "images" / "test.jpg"

# OUTPUT_DIR = PROJECT_ROOT / "outputs" / "predictions"
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# OUTPUT_IMAGE = OUTPUT_DIR / "test_prediction.jpg"


# # ============================================================
# # CONFIGURATION
# # ============================================================

# BAG_WEIGHT_KG = 50


# # ============================================================
# # LOAD MODEL
# # ============================================================

# model = YOLO(str(MODEL_PATH))


# # ============================================================
# # PREDICT
# # ============================================================

# results = model.predict(
#     source=str(IMAGE_PATH),
#     conf=0.25,
#     imgsz=640,
#     device=0,
#     verbose=False
# )

# result = results[0]

# # Original image copy
# output_image = result.orig_img.copy()


# # ============================================================
# # DETECTED BAGS
# # ============================================================

# bag_count = 0

# if result.boxes is not None:

#     boxes = result.boxes.xyxy.cpu().numpy()

#     for box in boxes:

#         x1, y1, x2, y2 = map(int, box)

#         bag_count += 1

#         # Draw bounding box
#         cv2.rectangle(
#             output_image,
#             (x1, y1),
#             (x2, y2),
#             (0, 255, 0),
#             2
#         )

#         # Only BAG NUMBER
#         label = f"BAG {bag_count}"

#         cv2.putText(
#             output_image,
#             label,
#             (x1, max(30, y1 - 10)),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.6,
#             (0, 255, 0),
#             2
#         )


# # ============================================================
# # CALCULATE TOTAL WEIGHT
# # ============================================================

# total_weight = bag_count * BAG_WEIGHT_KG


# # ============================================================
# # SHOW TOTALS ON IMAGE
# # ============================================================

# cv2.putText(
#     output_image,
#     f"TOTAL BAGS: {bag_count}",
#     (20, 40),
#     cv2.FONT_HERSHEY_SIMPLEX,
#     1,
#     (255, 255, 255),
#     2
# )

# cv2.putText(
#     output_image,
#     f"TOTAL WEIGHT: {total_weight} KG",
#     (20, 80),
#     cv2.FONT_HERSHEY_SIMPLEX,
#     1,
#     (255, 255, 255),
#     2
# )


# # ============================================================
# # SAVE RESULT
# # ============================================================

# cv2.imwrite(str(OUTPUT_IMAGE), output_image)


# # ============================================================
# # TERMINAL OUTPUT
# # ============================================================

# print("\n" + "=" * 50)
# print("RICE BAG INVENTORY RESULT")
# print("=" * 50)

# print(f"Total Bags Detected : {bag_count}")
# print(f"Weight Per Bag      : {BAG_WEIGHT_KG} KG")
# print(f"Total Weight        : {total_weight} KG")
# print("=" * 50)

# print(f"\nSaved: {OUTPUT_IMAGE}")


#################################################################################################################################################################
#################################################################################################################################################################


from ultralytics import YOLO
from pathlib import Path
import cv2


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "rice_bag_segmentation"
    / "weights"
    / "best.pt"
)

IMAGE_PATH = PROJECT_ROOT / "data" / "images" / "test.jpg"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_IMAGE = OUTPUT_DIR / "test_prediction.jpg"


# ============================================================
# CONFIGURATION
# ============================================================

BAG_WEIGHT_KG = 50


# ============================================================
# LOAD MODEL
# ============================================================

model = YOLO(str(MODEL_PATH))


# ============================================================
# PREDICT
# ============================================================

results = model.predict(
    source=str(IMAGE_PATH),
    conf=0.10,
    imgsz=960, #640
    iou=0.7, # just to see if it helps with overlapping bags
    device=0,
    verbose=False
)

result = results[0]

# Original clean image
output_image = result.orig_img.copy()


# ============================================================
# DETECTED BAGS
# ============================================================

bag_count = 0

if result.boxes is not None:

    boxes = result.boxes.xyxy.cpu().numpy()

    for box in boxes:

        x1, y1, x2, y2 = map(int, box)

        bag_count += 1

        # ----------------------------------------------------
        # CALCULATE CENTER OF DETECTED BAG
        # ----------------------------------------------------

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2


        # ----------------------------------------------------
        # DRAW ONLY CENTER POINT
        # ----------------------------------------------------

        cv2.circle(
            output_image,
            (center_x, center_y),
            8,          # Point radius
            (0, 255, 0),
            -1          # Filled circle
        )


        # ----------------------------------------------------
        # BAG NUMBER
        # ----------------------------------------------------

        label = f"BAG {bag_count}"

        cv2.putText(
            output_image,
            label,
            (center_x - 25, center_y - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )


# ============================================================
# CALCULATE TOTAL WEIGHT
# ============================================================

total_weight = bag_count * BAG_WEIGHT_KG


# ============================================================
# SUMMARY PANEL
# ============================================================

cv2.putText(
    output_image,
    f"TOTAL BAGS: {bag_count}",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255, 255, 255),
    2
)

cv2.putText(
    output_image,
    f"TOTAL WEIGHT: {total_weight} KG",
    (20, 80),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255, 255, 255),
    2
)


# ============================================================
# SAVE OUTPUT
# ============================================================

cv2.imwrite(str(OUTPUT_IMAGE), output_image)


# ============================================================
# TERMINAL OUTPUT
# ============================================================

print("\n" + "=" * 50)
print("RICE BAG INVENTORY RESULT")
print("=" * 50)

print(f"Total Bags Detected : {bag_count}")
print(f"Weight Per Bag      : {BAG_WEIGHT_KG} KG")
print(f"Total Weight        : {total_weight} KG")
print("=" * 50)

print(f"\nSaved: {OUTPUT_IMAGE}")