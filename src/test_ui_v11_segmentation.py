"""
RICE BAG V11 SEGMENTATION - PILE ESTIMATOR (Length / Breadth / Height)

New, standalone file. Does NOT modify test_rice_bag_v11_segmentation_ui.py
or warehouse_pile_estimator_ui.py in any way -- it only reuses ideas from
both:

  - Detection engine: the V11 instance-segmentation model
    (runs/rice_bag_v11_segmentation/weights/best.pt) plus the exact same
    two-stage mask-based duplicate dedup (Stage A: mask pixel-overlap,
    Stage B: mask-centroid distance clustering) added to
    test_rice_bag_v11_segmentation_ui.py.

  - Interaction model: the click-2-points Length / Breadth / Height
    reference-line tool from warehouse_pile_estimator_ui.py (draw a line
    across a row/column of the pile; Total = Length x Breadth x Height).

Why the L/B/H numbers here should be BETTER than the box-based estimator:
segmentation gives an actual pixel mask per bag, not just a rectangle. So:
  - The bag "center" used for drawing/labels is the mask's true centroid,
    not a box midpoint -- accurate even for bags photographed at an angle
    or partly hidden behind a neighbour.
  - The bag "size" used elsewhere is a tight mask-hugging bounding box,
    not the coarser YOLO box -- a rotated or irregular sack no longer
    inflates the size estimate the way a box around it would.

UI pass 1: restyled to match the dark slate/KPI-card dashboard look of
test_rice_bag_v11_segmentation_ui.py (header/brand, dropzone upload,
settings chips, KPI cards, panel/details-grid) instead of the old teal-on-
navy canvas UI.

UI/logic pass 2 (this pass):
  - Confidence / IoU are adjustable sliders again (like
    test_rice_bag_v11_segmentation_ui.py), sent with each /detect call --
    the previous pass had turned these into fixed read-only chips, which
    is reverted here.
  - The L/B/H line no longer matches bags by "is this bag's point within
    some pixel tolerance of the line" -- it now checks, per bag, whether
    the drawn line segment actually TOUCHES that bag's real segmentation
    mask (its polygon outline). A bag only counts on an axis if the line
    genuinely crosses its mask, same as looking at the picture and
    counting how many bags the line passes through. Because of this the
    old "line tolerance (px)" control no longer means anything and has
    been removed along with it.

Only this file was changed for either pass -- the other two UI files are
untouched.

Run: python src/test_ui_v11_segmentation.py
Access: http://localhost:5007
"""

from pathlib import Path
import base64
import statistics

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request, render_template_string
from ultralytics import YOLO


# ============================================================
# MODEL AUTO DISCOVERY (same search logic as test_rice_bag_v11_segmentation_ui.py)
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"


def find_model():
    best_models = [
        p for p in RUNS_DIR.rglob("best.pt")
        if "rice_bag_v11_segmentation" in str(p)
    ]
    if best_models:
        best_models.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return best_models[0]

    last_models = [
        p for p in RUNS_DIR.rglob("last.pt")
        if "rice_bag_v11_segmentation" in str(p)
    ]
    if last_models:
        last_models.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return last_models[0]

    raise FileNotFoundError("No V11 segmentation checkpoint found under runs/.")


MODEL_PATH = find_model()
print(f"\nLoading V11 segmentation model: {MODEL_PATH}")
model = YOLO(str(MODEL_PATH))
print(f"Model loaded. Task: {model.task}")

if model.task != "segment":
    raise RuntimeError(
        f"Loaded model's task is '{model.task}', not 'segment' -- "
        "this UI requires the instance-segmentation model."
    )

CLASS_NAMES = {0: "Rice-bag-10", 1: "other", 2: "pile"}

CONF_THRESHOLD = 0.25         # default; adjustable per-request via the confidence slider
IOU_THRESHOLD = 0.65          # default; adjustable per-request via the IoU slider -- loose model-level NMS, Stage A/B below clean up duplicates precisely using masks
MAX_DETECTIONS = 3000
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# Stage A / Stage B dedup settings (same tuned values as
# test_rice_bag_v11_segmentation_ui.py)
DEDUP_MASK_IOU_THRESHOLD = 0.45
DEDUP_MASK_CONTAINMENT_THRESHOLD = 0.70
CENTER_DIST_FACTOR = 0.45

DOT_RADIUS = 3

# Max polygon points sent per bag for the client-side line/mask-touch test
# -- approxPolyDP already simplifies the contour; this is just a hard cap
# so an unusually noisy mask can't blow up the JSON payload.
MAX_POLYGON_POINTS = 40

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


# ============================================================
# STAGE A + STAGE B - MASK-BASED DUPLICATE DEDUP
# (identical logic to test_rice_bag_v11_segmentation_ui.py)
# ============================================================

def _mask_iou_and_containment(mask_a, mask_b):
    inter = np.logical_and(mask_a, mask_b).sum()
    if inter == 0:
        return 0.0, 0.0
    area_a = mask_a.sum()
    area_b = mask_b.sum()
    union = area_a + area_b - inter
    iou = inter / union if union > 0 else 0.0
    containment = inter / max(1, min(area_a, area_b))
    return float(iou), float(containment)


def deduplicate_instances(low_res_masks, boxes_xyxy, confs, cls_ids,
                           iou_thresh=DEDUP_MASK_IOU_THRESHOLD,
                           containment_thresh=DEDUP_MASK_CONTAINMENT_THRESHOLD):
    n = len(low_res_masks)
    if n == 0:
        return []

    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        bi = boxes_xyxy[i]
        for j in range(i + 1, n):
            if cls_ids[i] != cls_ids[j]:
                continue
            bj = boxes_xyxy[j]
            if bi[2] <= bj[0] or bj[2] <= bi[0] or bi[3] <= bj[1] or bj[3] <= bi[1]:
                continue
            iou, containment = _mask_iou_and_containment(low_res_masks[i], low_res_masks[j])
            if iou >= iou_thresh or containment >= containment_thresh:
                union(i, j)

    clusters = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    keep = [max(members, key=lambda idx: confs[idx]) for members in clusters.values()]
    return sorted(keep)


def cluster_center_points(centers, sizes, cls_ids, confs, factor=CENTER_DIST_FACTOR):
    n = len(centers)
    if n == 0:
        return []

    typical_size = statistics.median(sizes) if sizes else 0.0
    threshold = typical_size * factor

    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        for j in range(i + 1, n):
            if cls_ids[i] != cls_ids[j]:
                continue
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            if (dx * dx + dy * dy) ** 0.5 <= threshold:
                union(i, j)

    clusters = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    keep_local = [max(members, key=lambda idx: confs[idx]) for members in clusters.values()]
    return sorted(keep_local)


# ============================================================
# STAGE 1 - DETECTION (V11 segmentation model + Stage A/B dedup),
# returns mask-derived (tight, centroid-accurate) bag entries, each
# carrying its simplified mask polygon for the client-side L/B/H
# line-touches-mask test.
# ============================================================

def run_detection(image_bgr, conf=None, iou=None):
    """Run V11 segmentation, dedup duplicate bag instances (mask-based),
    and draw each surviving bag as a center-point dot on top of a light
    mask overlay. Returns (annotated_image, sack_boxes, avg_conf,
    raw_count, duplicates_removed).

    conf/iou default to the module-level CONF_THRESHOLD/IOU_THRESHOLD but
    can be overridden per-request (the UI's confidence/IoU sliders)."""

    conf = CONF_THRESHOLD if conf is None else conf
    iou = IOU_THRESHOLD if iou is None else iou

    h, w = image_bgr.shape[:2]

    results = model.predict(
        source=image_bgr,
        imgsz=1280,
        conf=conf,
        iou=iou,
        max_det=MAX_DETECTIONS,
        device=DEVICE,
        verbose=False,
    )

    result = results[0]
    annotated = image_bgr.copy()

    if result.boxes is not None:
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    else:
        classes = np.array([], dtype=int)
        confidences = np.array([])
        boxes_xyxy = np.zeros((0, 4))

    masks = result.masks.data.cpu().numpy() if result.masks is not None else None

    raw_count = int((classes == 0).sum())

    # --- Stage A + Stage B dedup (mirrors test_rice_bag_v11_segmentation_ui.py) ---
    if len(classes) > 0 and masks is not None:
        low_res_masks_bool = masks > 0.5

        stage_a_survivors = deduplicate_instances(
            low_res_masks_bool, boxes_xyxy, confidences, classes
        )

        mh, mw = low_res_masks_bool.shape[1:]
        centers = []
        sizes = []

        for idx in stage_a_survivors:
            ys, xs = np.where(low_res_masks_bool[idx])
            if len(xs) == 0:
                bx = boxes_xyxy[idx]
                cx_full = (bx[0] + bx[2]) / 2
                cy_full = (bx[1] + bx[3]) / 2
            else:
                cx_full = float(np.mean(xs)) * (w / mw)
                cy_full = float(np.mean(ys)) * (h / mh)
            centers.append((cx_full, cy_full))

            bx = boxes_xyxy[idx]
            box_area = max(1.0, (bx[2] - bx[0]) * (bx[3] - bx[1]))
            sizes.append(float(np.sqrt(box_area)))

        stage_b_keep_local = cluster_center_points(
            centers, sizes,
            [classes[idx] for idx in stage_a_survivors],
            [confidences[idx] for idx in stage_a_survivors],
        )

        final_indices = sorted(stage_a_survivors[k] for k in stage_b_keep_local)
    else:
        final_indices = list(range(len(classes)))

    # --- draw: mask overlay accumulated once (avoids the addWeighted
    # compounding-darkening bug), dots drawn after the single blend ---
    overlay_accum = np.zeros_like(annotated)
    draw_items = []
    sack_boxes = []
    sack_id = 0

    for i in final_indices:
        cls_id = int(classes[i])
        conf_i = float(confidences[i])

        if masks is None or i >= len(masks):
            continue

        mask_full = cv2.resize(masks[i], (w, h), interpolation=cv2.INTER_NEAREST)
        binary_mask = (mask_full > 0.5).astype(np.uint8)
        mask_bool = binary_mask.astype(bool)

        if cls_id == 0:
            overlay_accum[:, :, 1][mask_bool] = 255
        elif cls_id == 1:
            overlay_accum[:, :, 2][mask_bool] = 255
        elif cls_id == 2:
            overlay_accum[:, :, 0][mask_bool] = 255

        ys, xs = np.where(binary_mask > 0)
        if len(xs) == 0:
            continue

        # Mask-derived tight bounding box -- hugs the actual bag shape,
        # a better length/breadth reference than the raw YOLO box.
        x1m, x2m = float(xs.min()), float(xs.max())
        y1m, y2m = float(ys.min()), float(ys.max())

        M = cv2.moments(binary_mask)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            cx = float(np.mean(xs))
            cy = float(np.mean(ys))

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if cls_id == 0:
            sack_id += 1

            # Simplified mask polygon (list of [x, y] ints, full-image
            # pixel coordinates) sent to the client so the L/B/H line
            # tool can test actual line-touches-mask intersection instead
            # of a point/tolerance approximation.
            polygon = []
            if contours:
                largest = max(contours, key=cv2.contourArea)
                epsilon = max(1.5, 0.01 * cv2.arcLength(largest, True))
                approx = cv2.approxPolyDP(largest, epsilon, True)
                points = approx.reshape(-1, 2)
                if len(points) > MAX_POLYGON_POINTS:
                    step = max(1, len(points) // MAX_POLYGON_POINTS)
                    points = points[::step]
                polygon = [[int(px), int(py)] for px, py in points]

            sack_boxes.append({
                "id": sack_id,
                "x1": x1m, "y1": y1m, "x2": x2m, "y2": y2m,
                "cx": cx, "cy": cy,
                "w": x2m - x1m, "h": y2m - y1m,
                "conf": conf_i,
                "polygon": polygon,
            })
            label = f"{sack_id} ({conf_i:.2f})"
            draw_items.append(("bag", contours, int(cx), int(cy), label))
        else:
            draw_items.append(("other", contours, int(cx), int(cy), None))

    annotated = cv2.addWeighted(annotated, 0.72, overlay_accum, 0.28, 0)

    for kind, contours, cx, cy, label in draw_items:
        cv2.drawContours(annotated, contours, -1, (255, 255, 255), 1)

        if kind == "bag":
            cv2.circle(annotated, (cx, cy), DOT_RADIUS, (255, 255, 255), -1)
            cv2.circle(annotated, (cx, cy), DOT_RADIUS + 2, (0, 0, 0), 1)
            cv2.putText(
                annotated, label,
                (cx + DOT_RADIUS + 3, cy - DOT_RADIUS),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA
            )

    avg_conf = float(np.mean([b["conf"] for b in sack_boxes])) if sack_boxes else 0.0
    duplicates_removed = raw_count - len(sack_boxes)

    cv2.putText(
        annotated,
        f"Total: {len(sack_boxes)} bags | Avg: {avg_conf:.3f} | Dupes removed: {duplicates_removed}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
    )

    return annotated, sack_boxes, avg_conf, raw_count, duplicates_removed


# ============================================================
# STAGE 2 (auto hint only) - HEIGHT ROW CLUSTERING
# (same algorithm as warehouse_pile_estimator_ui.py, now fed
# mask-derived centroids/heights instead of box ones)
# ============================================================

def estimate_height_rows(boxes):
    if len(boxes) < 2:
        return {"count": len(boxes), "confidence": 0.0}

    ys = sorted(b["cy"] for b in boxes)
    heights = [b["h"] for b in boxes if b["h"] > 0]
    typical_h = statistics.median(heights) if heights else 20.0
    gap_threshold = max(typical_h * 0.6, 5.0)

    clusters = [[ys[0]]]
    for y in ys[1:]:
        if y - clusters[-1][-1] > gap_threshold:
            clusters.append([y])
        else:
            clusters[-1].append(y)

    row_count = len(clusters)
    centers = [statistics.mean(c) for c in clusters]

    if len(centers) >= 3:
        gaps = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
        mean_gap = statistics.mean(gaps)
        stdev_gap = statistics.pstdev(gaps)
        consistency = 1.0 - min(1.0, stdev_gap / mean_gap) if mean_gap > 0 else 0.0
    elif len(centers) == 2:
        consistency = 0.5
    else:
        consistency = 0.2

    confidence = round(max(0.0, min(1.0, consistency)), 2)
    return {"count": row_count, "confidence": confidence}


# ============================================================
# HTML
# (dashboard visual language matches test_rice_bag_v11_segmentation_ui.py:
# same header/brand, dropzone, settings chips, KPI cards, panel/details-grid
# system. The interactive canvas + Length/Breadth/Height line tool is
# layered into that same design system.)
# ============================================================

HTML = r"""
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>
Rice Bag AI Vision | V11 Pile Estimator
</title>


<style>

/* =========================================================
   GLOBAL
   ========================================================= */

* {
    box-sizing: border-box;
}


body {

    margin: 0;

    background: #0b1120;

    color: #f8fafc;

    font-family:
        Inter,
        ui-sans-serif,
        system-ui,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
}


button,
input {

    font-family: inherit;
}


button {

    border: 0;

    padding: 11px 16px;

    border-radius: 10px;

    cursor: pointer;

    font-weight: 700;

    font-size: 13px;

    background: #2563eb;

    color: white;
}


button.secondary {

    background: #1e293b;

    color: #cbd5e1;
}


button.danger {

    background: #7f1d1d;

    color: #fecaca;
}


button:disabled {

    opacity: 0.4;

    cursor: not-allowed;
}


/* =========================================================
   HEADER
   ========================================================= */

.header {

    background: #111827;

    border-bottom:
        1px solid #1f2937;

    padding:
        18px 30px;

    display: flex;

    align-items: center;

    justify-content: space-between;

    gap: 20px;

}


.brand {

    display: flex;

    align-items: center;

    gap: 14px;

}


.logo {

    width: 46px;

    height: 46px;

    border-radius: 12px;

    background: #2563eb;

    display: flex;

    align-items: center;

    justify-content: center;

    font-size: 23px;

}


.brand h1 {

    margin: 0;

    font-size: 21px;

}


.brand p {

    margin: 3px 0 0;

    color: #94a3b8;

    font-size: 13px;

}


.status {

    display: flex;

    align-items: center;

    gap: 8px;

    padding:
        8px 13px;

    border-radius: 999px;

    background: #0f172a;

    border:
        1px solid #1e293b;

    font-size: 13px;

    color: #cbd5e1;

}


.status-dot {

    width: 8px;

    height: 8px;

    border-radius: 50%;

    background: #22c55e;

}


/* =========================================================
   MAIN
   ========================================================= */

.container {

    max-width: 1600px;

    margin: auto;

    padding: 25px 30px 40px;

}


/* =========================================================
   UPLOAD
   ========================================================= */

.upload-card {

    background: #111827;

    border:
        1px solid #1f2937;

    border-radius: 16px;

    padding: 24px;

    margin-bottom: 20px;

}


.drop-zone {

    border:
        2px dashed #334155;

    border-radius: 14px;

    padding: 35px 20px;

    text-align: center;

    cursor: pointer;

    transition: 0.2s;

}


.drop-zone:hover {

    border-color: #60a5fa;

    background: #0f172a;

}


.drop-zone.dragover {

    border-color: #60a5fa;

    background: #172554;

}


.upload-icon {

    font-size: 36px;

    margin-bottom: 8px;

}


.drop-zone h3 {

    margin: 5px 0;

}


.drop-zone p {

    color: #94a3b8;

    margin: 7px 0 15px;

    font-size: 14px;

}


.file-name {

    color: #60a5fa;

    font-size: 13px;

    margin-top: 10px;

}


input[type=file] {

    display: none;

}


.run-button {

    width: 100%;

    margin-top: 16px;

    padding: 13px;

    font-size: 15px;

}


.upload-buttons {

    display: flex;

    gap: 10px;

    margin-top: 16px;

}


.upload-buttons .run-button {

    margin-top: 0;

    flex: 1;
}


.status-banner {

    margin-top: 14px;

    padding: 12px 14px;

    border-radius: 10px;

    background: #0b1220;

    border:
        1px solid #1e293b;

    color: #cbd5e1;

    font-size: 13px;

}


/* =========================================================
   SETTINGS
   ========================================================= */

.settings {

    margin-top: 18px;

    display: grid;

    grid-template-columns:
        repeat(3, 1fr);

    gap: 15px;

}


.setting {

    background: #0b1220;

    border:
        1px solid #1e293b;

    border-radius: 10px;

    padding: 14px;

}


.setting-title {

    font-size: 12px;

    color: #94a3b8;

    margin-bottom: 8px;

}


.setting-value {

    font-size: 18px;

    font-weight: 700;

}


.setting-note {

    color: #64748b;

    font-size: 11px;

    margin-top: 6px;

}


input[type=range] {

    width: 100%;

    margin-top: 8px;

}


/* =========================================================
   KPI CARDS
   ========================================================= */

.kpis {

    display: grid;

    grid-template-columns:
        repeat(5, 1fr);

    gap: 14px;

    margin-bottom: 20px;

}


.kpi {

    background: #111827;

    border:
        1px solid #1f2937;

    border-radius: 14px;

    padding: 18px;

}


.kpi-top {

    color: #94a3b8;

    font-size: 12px;

    text-transform: uppercase;

    letter-spacing: 0.06em;

}


.kpi-number {

    font-size: 28px;

    font-weight: 800;

    margin-top: 8px;

}


.kpi-sub {

    color: #64748b;

    font-size: 11px;

    margin-top: 4px;

}


/* =========================================================
   CANVAS / DETECTION IMAGE
   ========================================================= */

.image-card {

    background: #111827;

    border:
        1px solid #1f2937;

    border-radius: 16px;

    overflow: hidden;

    margin-bottom: 20px;

}


.card-header {

    padding:
        15px 18px;

    border-bottom:
        1px solid #1f2937;

    display: flex;

    justify-content: space-between;

    align-items: center;

}


.card-header h2 {

    margin: 0;

    font-size: 15px;

}


.badge {

    font-size: 11px;

    padding:
        5px 9px;

    border-radius: 999px;

    background: #1e293b;

    color: #cbd5e1;

}


.image-container {

    padding: 15px;

    background: #080d18;

    text-align: center;

    overflow: auto;

}


.image-container canvas {

    max-width: 100%;

    max-height: 720px;

    border-radius: 10px;

    cursor: crosshair;

}


/* =========================================================
   DETAILS / PANELS
   ========================================================= */

.details-grid {

    display: grid;

    grid-template-columns:
        1.3fr 1fr;

    gap: 20px;

}


.panel {

    background: #111827;

    border:
        1px solid #1f2937;

    border-radius: 16px;

    overflow: hidden;

}


.panel-title {

    padding:
        16px 18px;

    border-bottom:
        1px solid #1f2937;

    font-size: 15px;

    font-weight: 700;

}


.panel-body {

    padding: 18px;

}


.locked {

    opacity: 0.45;

    pointer-events: none;

}


.instructions {

    background: #0b1220;

    border:
        1px solid #1e293b;

    border-radius: 12px;

    padding: 14px;

    line-height: 1.6;

    margin-bottom: 16px;

    font-size: 13px;

    color: #cbd5e1;

}


/* =========================================================
   AXIS MEASUREMENT CARDS
   ========================================================= */

.measurement {

    background: #0b1220;

    border:
        1px solid #1e293b;

    border-radius: 12px;

    padding: 14px;

    margin-bottom: 12px;

}


.measurement.active {

    border-color: #2563eb;
}


.measurement h3 {

    margin: 0 0 10px;

    font-size: 14px;

    display: flex;

    align-items: center;

}


.swatch {

    display: inline-block;

    width: 10px;

    height: 10px;

    border-radius: 3px;

    margin-right: 8px;

}


.axis-row {

    display: flex;

    gap: 10px;

    align-items: center;

    flex-wrap: wrap;

    margin-top: 8px;

}


.count-value {

    margin-top: 8px;

    font-weight: 700;

    color: #38bdf8;

    font-size: 13px;

}


/* =========================================================
   RESULTS PANEL
   ========================================================= */

.info-row {

    display: flex;

    justify-content: space-between;

    gap: 15px;

    padding:
        10px 0;

    border-bottom:
        1px solid #1e293b;

    font-size: 13px;

}


.info-row:last-child {

    border-bottom: none;
}


.info-label {

    color: #64748b;
}


.info-value {

    color: #e2e8f0;

    font-weight: 700;
}


.info-value.big {

    font-size: 22px;

    color: #22c55e;
}


/* =========================================================
   ERROR / WARNINGS
   ========================================================= */

.error {

    background: #450a0a;

    border:
        1px solid #7f1d1d;

    color: #fecaca;

    border-radius: 12px;

    padding: 15px;

    margin-bottom: 20px;

    white-space: pre-wrap;

}


.warnings {

    margin-top: 14px;

    padding: 12px 14px;

    border-radius: 10px;

    background: rgba(217, 119, 6, 0.12);

    border:
        1px solid rgba(217, 119, 6, 0.4);

    color: #fed7aa;

    font-size: 13px;

    line-height: 1.6;

}


/* =========================================================
   FOOTER
   ========================================================= */

.footer {

    text-align: center;

    color: #475569;

    font-size: 12px;

    padding-top: 25px;

}


/* =========================================================
   RESPONSIVE
   ========================================================= */

@media(max-width: 1200px) {

    .kpis {

        grid-template-columns:
            repeat(3, 1fr);

    }

}


@media(max-width: 900px) {

    .details-grid {

        grid-template-columns: 1fr;

    }

    .settings {

        grid-template-columns:
            repeat(2, 1fr);

    }

}


@media(max-width: 600px) {

    .container {

        padding: 15px;

    }

    .header {

        padding: 15px;

    }

    .status {

        display: none;

    }

    .kpis {

        grid-template-columns:
            repeat(2, 1fr);

    }

    .settings {

        grid-template-columns: 1fr;

    }

}

</style>

</head>


<body>


<!-- ======================================================
     HEADER
     ====================================================== -->

<div class="header">

    <div class="brand">

        <div class="logo">
            ◉
        </div>

        <div>

            <h1>
                Rice Bag AI Vision
            </h1>

            <p>
                V11 Segmentation • Pile Estimator (L / B / H) • Warehouse POC
            </p>

        </div>

    </div>


    <div class="status">

        <span class="status-dot"></span>

        Model Loaded

    </div>

</div>


<div class="container">


<!-- ======================================================
     UPLOAD
     ====================================================== -->

<div class="upload-card">

<div class="drop-zone"
     id="dropZone">

    <div class="upload-icon">
        📷
    </div>

    <h3>
        Upload Warehouse Image
    </h3>

    <p>
        Drag & drop an image here or click to browse
    </p>

    <strong>
        Select Image
    </strong>

    <div class="file-name"
         id="fileName">
    </div>

</div>


<input
    type="file"
    id="imageInput"
    accept="image/*"
>


<!-- ======================================================
     SETTINGS (confidence / IoU sliders + static info)
     ====================================================== -->

<div class="settings">


<div class="setting">

    <div class="setting-title">
        Confidence Threshold
    </div>

    <div class="setting-value">
        <span id="confValue">
            {{ confidence }}
        </span>
    </div>

    <input
        type="range"
        id="confidence"
        min="0.10"
        max="0.90"
        step="0.05"
        value="{{ confidence }}"
    >

</div>


<div class="setting">

    <div class="setting-title">
        IoU Threshold
    </div>

    <div class="setting-value">
        <span id="iouValue">
            {{ iou }}
        </span>
    </div>

    <input
        type="range"
        id="iou"
        min="0.20"
        max="0.90"
        step="0.05"
        value="{{ iou }}"
    >

</div>


<div class="setting">

    <div class="setting-title">
        Inference Resolution
    </div>

    <div class="setting-value">
        1280 × 1280
    </div>

    <div class="setting-note">
        Segmentation optimized
    </div>

</div>


</div>


<div class="upload-buttons">

    <button
        class="run-button"
        id="detectBtn"
        disabled
        onclick="detectSacks()"
    >
        🚀 1. Detect Sacks
    </button>

    <button
        class="secondary run-button"
        onclick="resetAll()"
    >
        Reset
    </button>

</div>


<div id="status" class="status-banner">
    Upload an image to begin.
</div>

</div>


<!-- ======================================================
     KPI DASHBOARD
     ====================================================== -->

<div class="kpis">


<div class="kpi">

    <div class="kpi-top">
        Detected Sacks
    </div>

    <div class="kpi-number" id="resultDetected">
        --
    </div>

    <div class="kpi-sub">
        After duplicate dedup
    </div>

</div>


<div class="kpi">

    <div class="kpi-top">
        Raw Detections
    </div>

    <div class="kpi-number" id="resultRaw">
        --
    </div>

    <div class="kpi-sub">
        Before dedup
    </div>

</div>


<div class="kpi">

    <div class="kpi-top">
        Duplicates Removed
    </div>

    <div class="kpi-number" id="resultDuplicates">
        --
    </div>

    <div class="kpi-sub">
        Mask-based Stage A + B
    </div>

</div>


<div class="kpi">

    <div class="kpi-top">
        Avg Confidence
    </div>

    <div class="kpi-number" id="resultAvgConf">
        --
    </div>

    <div class="kpi-sub">
        Mean detection confidence
    </div>

</div>


<div class="kpi">

    <div class="kpi-top">
        Auto Height Hint
    </div>

    <div class="kpi-number" id="resultAutoHeight">
        --
    </div>

    <div class="kpi-sub">
        Row clustering estimate
    </div>

</div>


</div>


<!-- ======================================================
     DETECTION CANVAS
     ====================================================== -->

<div class="image-card">

    <div class="card-header">

        <h2>
            Detection + Axis Lines
        </h2>

        <span class="badge">
            INTERACTIVE
        </span>

    </div>


    <div class="image-container">

        <canvas id="canvas"></canvas>

    </div>

</div>


<!-- ======================================================
     AXIS MEASUREMENT + RESULTS
     ====================================================== -->

<div class="details-grid">


<!-- AXIS MEASUREMENT -->

<div class="panel">

    <div class="panel-title">
        📏 Length / Breadth / Height
    </div>


    <div class="panel-body">

        <div id="axisSection" class="locked">

            <div class="instructions">
                Select an axis below, then click 2 points on the detected image along that axis.
                Any bag whose actual mask the drawn line passes through counts as matched on that axis
                &mdash; not a nearby point, the real segmented shape.
                Total sacks = Length count &times; Breadth count &times; Height count.
            </div>

            <div id="lengthBox" class="measurement">
                <h3><span class="swatch" style="background:#00ff88;"></span>Length</h3>
                <div class="axis-row">
                    <button class="secondary axis-btn" disabled onclick="selectMeasurement('length')">Select Length</button>
                </div>
                <div id="lengthCountValue" class="count-value">--</div>
            </div>

            <div id="breadthBox" class="measurement">
                <h3><span class="swatch" style="background:#00aaff;"></span>Breadth</h3>
                <div class="axis-row">
                    <button class="secondary axis-btn" disabled onclick="selectMeasurement('breadth')">Select Breadth</button>
                </div>
                <div id="breadthCountValue" class="count-value">--</div>
            </div>

            <div id="heightBox" class="measurement">
                <h3><span class="swatch" style="background:#ffcc00;"></span>Height</h3>
                <div class="axis-row">
                    <button class="secondary axis-btn" disabled onclick="selectMeasurement('height')">Select Height</button>
                </div>
                <div id="heightCountValue" class="count-value">--</div>
            </div>

            <button class="secondary axis-btn" disabled onclick="clearCurrentLine()">Clear Current Line</button>

        </div>

    </div>

</div>


<!-- RESULTS -->

<div class="panel">

    <div class="panel-title">
        📊 Results
    </div>


    <div class="panel-body">

        <div class="info-row"><span class="info-label">Sacks Along Length</span><span class="info-value" id="resultSacksLength">--</span></div>
        <div class="info-row"><span class="info-label">Sacks Along Breadth</span><span class="info-value" id="resultSacksBreadth">--</span></div>
        <div class="info-row"><span class="info-label">Sacks Along Height</span><span class="info-value" id="resultSacksHeight">--</span></div>
        <div class="info-row"><span class="info-label">Total Estimated Sacks (L&times;B&times;H)</span><span class="info-value big" id="resultTotal">--</span></div>

        <div id="warnings" class="warnings" style="display:none;"></div>

    </div>

</div>


</div>


<div class="footer">

    Rice Bag AI Vision • V11 Segmentation Pile Estimator

</div>


</div>


<script>


const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

let image = new Image();
let imageLoaded = false;
let detecting = false;

let detectedBoxes = [];
let autoHeight = null;

let currentMeasurement = null;

let lines = { length: [], breadth: [], height: [] };
let axisCounts = { length: null, breadth: null, height: null };
let matchedBoxIds = { length: [], breadth: [], height: [] };

const AXIS_COLORS = { length: "#00ff88", breadth: "#00aaff", height: "#ffcc00" };
const AXIS_LABELS = { length: "LENGTH", breadth: "BREADTH", height: "HEIGHT" };


function setStatus(text) {
    document.getElementById("status").innerText = text;
}

function confidenceLabel(c) {
    if (c >= 0.7) return "High";
    if (c >= 0.4) return "Medium";
    return "Low";
}

function resetResultFields() {
    document.getElementById("resultDetected").innerText = "--";
    document.getElementById("resultRaw").innerText = "--";
    document.getElementById("resultDuplicates").innerText = "--";
    document.getElementById("resultAvgConf").innerText = "--";
    document.getElementById("resultAutoHeight").innerText = "--";
    document.getElementById("resultSacksLength").innerText = "--";
    document.getElementById("resultSacksBreadth").innerText = "--";
    document.getElementById("resultSacksHeight").innerText = "--";
    document.getElementById("resultTotal").innerText = "--";

    ["length", "breadth", "height"].forEach(type => {
        document.getElementById(type + "CountValue").innerText = "--";
    });

    document.getElementById("warnings").style.display = "none";
    document.getElementById("warnings").innerHTML = "";
}


// ------------------------------------------------------------
// CONFIDENCE / IOU SLIDERS
// ------------------------------------------------------------

const confSlider = document.getElementById("confidence");
const confValue = document.getElementById("confValue");

confSlider.addEventListener("input", () => {
    confValue.textContent = confSlider.value;
});

const iouSlider = document.getElementById("iou");
const iouValue = document.getElementById("iouValue");

iouSlider.addEventListener("input", () => {
    iouValue.textContent = iouSlider.value;
});


// ------------------------------------------------------------
// FILE INPUT / DRAG & DROP
// ------------------------------------------------------------

const dropZone = document.getElementById("dropZone");
const imageInput = document.getElementById("imageInput");
const fileNameEl = document.getElementById("fileName");

dropZone.addEventListener("click", () => imageInput.click());

dropZone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragover");

    if (event.dataTransfer.files.length > 0) {
        imageInput.files = event.dataTransfer.files;
        handleFileSelected();
    }
});

imageInput.addEventListener("change", handleFileSelected);

function handleFileSelected() {
    const file = imageInput.files[0];
    if (!file) return;

    fileNameEl.textContent = "Selected: " + file.name;

    const reader = new FileReader();

    reader.onload = function (event) {
        image.onload = function () {
            imageLoaded = true;

            canvas.width = image.naturalWidth;
            canvas.height = image.naturalHeight;

            detectedBoxes = [];
            autoHeight = null;
            lines = { length: [], breadth: [], height: [] };
            axisCounts = { length: null, breadth: null, height: null };
            matchedBoxIds = { length: [], breadth: [], height: [] };
            currentMeasurement = null;

            redraw();
            resetResultFields();

            document.querySelectorAll(".axis-btn").forEach(b => b.disabled = true);
            document.getElementById("axisSection").classList.add("locked");
            document.querySelectorAll(".measurement").forEach(el => el.classList.remove("active"));

            document.getElementById("detectBtn").disabled = false;

            setStatus('Image loaded. Click "1. Detect Sacks" to run V11 segmentation first.');
        };
        image.src = event.target.result;
    };

    reader.readAsDataURL(file);
}


// ------------------------------------------------------------
// STAGE 1 - DETECTION
// ------------------------------------------------------------

async function detectSacks() {

    if (!imageLoaded) { setStatus("Upload an image first."); return; }

    const file = imageInput.files[0];
    if (!file) { setStatus("Upload an image first."); return; }

    if (detecting) return;
    detecting = true;

    const btn = document.getElementById("detectBtn");
    btn.disabled = true;
    setStatus("Running V11 segmentation...");

    const form = new FormData();
    form.append("image", file);
    form.append("confidence", confSlider.value);
    form.append("iou", iouSlider.value);

    try {
        const response = await fetch("/detect", { method: "POST", body: form });
        const data = await response.json();

        if (!data.success) {
            setStatus(data.message || "Detection failed.");
            detecting = false;
            btn.disabled = false;
            return;
        }

        detectedBoxes = data.boxes;
        autoHeight = data.auto_height;

        lines = { length: [], breadth: [], height: [] };
        axisCounts = { length: null, breadth: null, height: null };
        matchedBoxIds = { length: [], breadth: [], height: [] };

        image.onload = function () {
            canvas.width = image.naturalWidth;
            canvas.height = image.naturalHeight;
            redraw();
        };
        image.src = "data:image/jpeg;base64," + data.image_base64;

        document.getElementById("resultDetected").innerText = data.detected_sacks;
        document.getElementById("resultRaw").innerText = data.raw_detections;
        document.getElementById("resultDuplicates").innerText = data.duplicates_removed;
        document.getElementById("resultAvgConf").innerText = (data.avg_confidence * 100).toFixed(1) + "%";

        if (data.detected_sacks > 0) {
            document.getElementById("resultAutoHeight").innerText =
                autoHeight.count + " rows (" + confidenceLabel(autoHeight.confidence) + ")";
        } else {
            document.getElementById("resultAutoHeight").innerText = "No sacks";
        }

        if (data.detected_sacks > 0) {
            document.querySelectorAll(".axis-btn").forEach(b => b.disabled = false);
            document.getElementById("axisSection").classList.remove("locked");

            currentMeasurement = "length";
            document.querySelectorAll(".measurement").forEach(el => el.classList.remove("active"));
            document.getElementById("lengthBox").classList.add("active");

            setStatus("Detected " + data.detected_sacks + " sacks (" + data.duplicates_removed +
                " duplicates removed). Now select Length and click 2 points across the pile.");
        } else {
            setStatus("No sacks detected in this image. Try a clearer photo, or lower the confidence threshold.");
        }

        updateTotal();
        updateWarnings();

    } catch (err) {
        setStatus("Detection error: " + err.message);
    }

    detecting = false;
    btn.disabled = false;
}


// ------------------------------------------------------------
// STAGE 2 - AXIS SELECTION & LINE-TOUCHES-MASK COUNTING
// ------------------------------------------------------------

function selectMeasurement(type) {
    if (!detectedBoxes.length) {
        setStatus('Run "1. Detect Sacks" first.');
        return;
    }

    currentMeasurement = type;

    document.querySelectorAll(".measurement").forEach(el => el.classList.remove("active"));
    document.getElementById(type + "Box").classList.add("active");

    setStatus("Now click 2 points across the pile's " + AXIS_LABELS[type].toLowerCase() + ".");
}

// --- geometry: does the drawn line segment actually touch this bag's mask? ---

function orientation(a, b, c) {
    const val = (b.y - a.y) * (c.x - b.x) - (b.x - a.x) * (c.y - b.y);
    if (Math.abs(val) < 1e-9) return 0;
    return val > 0 ? 1 : 2;
}

function onSegment(a, b, c) {
    return Math.min(a.x, c.x) <= b.x && b.x <= Math.max(a.x, c.x) &&
           Math.min(a.y, c.y) <= b.y && b.y <= Math.max(a.y, c.y);
}

function segmentsIntersect(p1, p2, p3, p4) {
    const o1 = orientation(p1, p2, p3);
    const o2 = orientation(p1, p2, p4);
    const o3 = orientation(p3, p4, p1);
    const o4 = orientation(p3, p4, p2);

    if (o1 !== o2 && o3 !== o4) return true;

    if (o1 === 0 && onSegment(p1, p3, p2)) return true;
    if (o2 === 0 && onSegment(p1, p4, p2)) return true;
    if (o3 === 0 && onSegment(p3, p1, p4)) return true;
    if (o4 === 0 && onSegment(p3, p2, p4)) return true;

    return false;
}

function pointInPolygon(pt, poly) {
    let inside = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
        const xi = poly[i][0], yi = poly[i][1];
        const xj = poly[j][0], yj = poly[j][1];
        const intersect = ((yi > pt.y) !== (yj > pt.y)) &&
            (pt.x < (xj - xi) * (pt.y - yi) / (yj - yi) + xi);
        if (intersect) inside = !inside;
    }
    return inside;
}

// A bag counts on this axis only if the drawn segment genuinely crosses
// its mask polygon (or starts/ends inside it) -- not just "near" it.
function lineTouchesMask(p1, p2, polygon) {
    if (!polygon || polygon.length < 3) return false;

    for (let i = 0; i < polygon.length; i++) {
        const a = { x: polygon[i][0], y: polygon[i][1] };
        const b = { x: polygon[(i + 1) % polygon.length][0], y: polygon[(i + 1) % polygon.length][1] };
        if (segmentsIntersect(p1, p2, a, b)) return true;
    }

    if (pointInPolygon(p1, polygon) || pointInPolygon(p2, polygon)) return true;

    return false;
}

function countAlongLine(p1, p2) {
    // Extend the clicked segment slightly past both ends so a mask right
    // at the pile's visible edge isn't missed just because the click
    // landed a few px short of the bag.
    const dx = p2.x - p1.x, dy = p2.y - p1.y;
    const len = Math.hypot(dx, dy);
    if (len < 1) return { count: 0, ids: [] };

    const ux = dx / len, uy = dy / len;
    const margin = len * 0.08;
    const ep1 = { x: p1.x - ux * margin, y: p1.y - uy * margin };
    const ep2 = { x: p2.x + ux * margin, y: p2.y + uy * margin };

    const ids = [];
    detectedBoxes.forEach(b => {
        if (lineTouchesMask(ep1, ep2, b.polygon)) {
            ids.push(b.id);
        }
    });

    return { count: ids.length, ids: ids };
}

function recomputeAxis(type) {
    const pts = lines[type];
    if (pts.length !== 2) return;

    const { count, ids } = countAlongLine(pts[0], pts[1]);

    axisCounts[type] = count;
    matchedBoxIds[type] = ids;

    document.getElementById(type + "CountValue").innerText = count + " sacks matched";

    redraw();
    updateTotal();
    updateWarnings();
}

function clearCurrentLine() {
    if (!currentMeasurement) {
        setStatus("Select Length, Breadth or Height first.");
        return;
    }

    lines[currentMeasurement] = [];
    axisCounts[currentMeasurement] = null;
    matchedBoxIds[currentMeasurement] = [];
    document.getElementById(currentMeasurement + "CountValue").innerText = "--";

    redraw();
    updateTotal();
    updateWarnings();

    setStatus(AXIS_LABELS[currentMeasurement] + " line cleared.");
}

canvas.addEventListener("click", function (event) {
    if (!imageLoaded) return;

    if (!detectedBoxes.length) {
        setStatus('Run "1. Detect Sacks" first before drawing reference lines.');
        return;
    }

    if (!currentMeasurement) {
        setStatus("Select Length, Breadth or Height first.");
        return;
    }

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const x = (event.clientX - rect.left) * scaleX;
    const y = (event.clientY - rect.top) * scaleY;

    const pts = lines[currentMeasurement];

    if (pts.length >= 2) {
        pts.length = 0;
        axisCounts[currentMeasurement] = null;
        matchedBoxIds[currentMeasurement] = [];
    }

    pts.push({ x: x, y: y });

    if (pts.length === 2) {
        recomputeAxis(currentMeasurement);
        setStatus(AXIS_LABELS[currentMeasurement] + ": line complete, " +
            axisCounts[currentMeasurement] + " sacks matched.");
    } else {
        redraw();
        setStatus(AXIS_LABELS[currentMeasurement] + ": first point selected. Click second point.");
    }
});


// ------------------------------------------------------------
// TOTALS & WARNINGS
// ------------------------------------------------------------

function updateTotal() {
    const l = axisCounts.length, b = axisCounts.breadth, h = axisCounts.height;

    document.getElementById("resultSacksLength").innerText = (l === null ? "--" : l);
    document.getElementById("resultSacksBreadth").innerText = (b === null ? "--" : b);
    document.getElementById("resultSacksHeight").innerText = (h === null ? "--" : h);

    const totalEl = document.getElementById("resultTotal");
    if (l && b && h) {
        totalEl.innerText = (l * b * h) + " sacks";
    } else {
        totalEl.innerText = "--";
    }
}

function updateWarnings() {
    const warnings = [];

    ["length", "breadth", "height"].forEach(type => {
        const c = axisCounts[type];
        if (lines[type].length === 2) {
            if (c === 0) {
                warnings.push(AXIS_LABELS[type] + ": the line doesn't pass through any detected bag's mask. Redraw it across a row/column of bags.");
            } else if (c === 1 && detectedBoxes.length > 8) {
                warnings.push(AXIS_LABELS[type] + ": only 1 sack matched - this count looks unreliable. Redraw the line so it clearly crosses more than one bag.");
            }
        }
    });

    if (detectedBoxes.length > 0 && detectedBoxes.length < 3) {
        warnings.push("Very few sacks were detected overall - axis counts and the total will not be reliable.");
    }

    if (autoHeight && lines.height.length === 2 && axisCounts.height !== null) {
        if (Math.abs(axisCounts.height - autoHeight.count) > Math.max(2, autoHeight.count * 0.4)) {
            warnings.push("Height line count (" + axisCounts.height + ") disagrees a lot with the automatic height hint (" +
                autoHeight.count + "). Re-check the line placement.");
        }
    }

    const box = document.getElementById("warnings");
    if (warnings.length) {
        box.style.display = "block";
        box.innerHTML = "&#9888; " + warnings.join("<br>&#9888; ");
    } else {
        box.style.display = "none";
        box.innerHTML = "";
    }
}


// ------------------------------------------------------------
// DRAWING
// ------------------------------------------------------------

function drawAxisLine(type) {
    const pts = lines[type];
    if (!pts.length) return;

    const color = AXIS_COLORS[type];

    pts.forEach((p, index) => {
        ctx.beginPath();
        ctx.arc(p.x, p.y, 8, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();

        ctx.fillStyle = "white";
        ctx.font = "bold 16px Arial";
        ctx.fillText(AXIS_LABELS[type] + " " + (index + 1), p.x + 10, p.y - 10);
    });

    if (pts.length === 2) {
        const p1 = pts[0], p2 = pts[1];

        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.stroke();

        (matchedBoxIds[type] || []).forEach(id => {
            const b = detectedBoxes.find(bb => bb.id === id);
            if (!b) return;
            ctx.beginPath();
            ctx.arc(b.cx, b.cy, 7, 0, Math.PI * 2);
            ctx.globalAlpha = 0.85;
            ctx.fillStyle = color;
            ctx.fill();
            ctx.globalAlpha = 1.0;
        });

        const midX = (p1.x + p2.x) / 2, midY = (p1.y + p2.y) / 2;
        ctx.fillStyle = color;
        ctx.font = "bold 20px Arial";
        const cnt = axisCounts[type];
        ctx.fillText(AXIS_LABELS[type] + ": " + (cnt === null ? "?" : cnt) + " sacks", midX, midY - 16);
    }
}

function redraw() {
    if (!imageLoaded) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

    drawAxisLine("length");
    drawAxisLine("breadth");
    drawAxisLine("height");
}


// ------------------------------------------------------------
// RESET
// ------------------------------------------------------------

function resetAll() {
    imageInput.value = "";
    fileNameEl.textContent = "";

    imageLoaded = false;
    detecting = false;
    detectedBoxes = [];
    autoHeight = null;
    lines = { length: [], breadth: [], height: [] };
    axisCounts = { length: null, breadth: null, height: null };
    matchedBoxIds = { length: [], breadth: [], height: [] };
    currentMeasurement = null;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    canvas.width = 0;
    canvas.height = 0;

    resetResultFields();

    document.querySelectorAll(".axis-btn").forEach(b => b.disabled = true);
    document.getElementById("axisSection").classList.add("locked");
    document.querySelectorAll(".measurement").forEach(el => el.classList.remove("active"));
    document.getElementById("detectBtn").disabled = true;

    setStatus("Upload an image to begin.");
}

</script>


</body>

</html>
"""


@app.route("/")
def index():
    return render_template_string(
        HTML,
        confidence=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
    )


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "V11 Segmentation Pile Estimator",
        "device": str(DEVICE),
        "model": str(MODEL_PATH),
    })


@app.route("/detect", methods=["POST"])
def detect():
    try:
        image_file = request.files.get("image")
        if image_file is None or image_file.filename == "":
            return jsonify({"success": False, "message": "Please upload an image first."})

        image_bytes = image_file.read()
        if len(image_bytes) == 0:
            return jsonify({"success": False, "message": "Uploaded image is empty."})

        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image_bgr is None:
            return jsonify({"success": False, "message": "The uploaded file is not a valid image."})

        conf = float(request.form.get("confidence", CONF_THRESHOLD))
        iou = float(request.form.get("iou", IOU_THRESHOLD))

        annotated, boxes, avg_conf, raw_count, duplicates_removed = run_detection(
            image_bgr, conf=conf, iou=iou
        )
        auto_height = estimate_height_rows(boxes)

        is_success, buffer = cv2.imencode(".jpg", annotated)
        if not is_success:
            return jsonify({"success": False, "message": "Failed to create annotated image."})

        return jsonify({
            "success": True,
            "detected_sacks": len(boxes),
            "raw_detections": raw_count,
            "duplicates_removed": duplicates_removed,
            "avg_confidence": avg_conf,
            "confidence": conf,
            "iou": iou,
            "boxes": boxes,
            "auto_height": auto_height,
            "image_base64": base64.b64encode(buffer.tobytes()).decode("utf-8"),
        })

    except Exception as exc:
        return jsonify({"success": False, "message": f"Runtime error: {exc}"})


if __name__ == "__main__":
    print("=" * 60)
    print("V11 SEGMENTATION PILE ESTIMATOR")
    print("=" * 60)
    print(f"Model : {MODEL_PATH}")
    print(f"Device: {DEVICE}")
    print(f"Default confidence: {CONF_THRESHOLD}  Default IoU: {IOU_THRESHOLD}")
    print("Local URL: http://127.0.0.1:5007")
    print("Network URL: http://0.0.0.0:5007")
    print("=" * 60)

    app.run(host="0.0.0.0", port=5007, debug=False)
