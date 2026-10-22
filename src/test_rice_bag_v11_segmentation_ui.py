"""
RICE BAG V11 - PROFESSIONAL INSTANCE SEGMENTATION TESTING UI

Bugs fixed from the originally pasted version:

  1. find_model() filtered on "rice_bag_11", which does not match this
     model's actual saved path ("rice_bag_v11_segmentation") -- would
     have raised FileNotFoundError.
  2. device=0 hardcoded (GPU-only) -- crashes on this CPU-only machine.
     Now auto-detects device like the other test UIs in this project.
  3. The mask overlay blend (cv2.addWeighted) ran INSIDE the per-instance
     loop -- with a few hundred detections that decays the image to
     solid black (0.72^N -> ~0). Now accumulated once and blended once
     after the loop, respecting the show_masks/show_points/show_labels
     toggles.

Fine-tuning pass (duplicate bags / missed bags), same idea as the
two-stage dedup already proven in test_rice_bag_v9_ui.py, adapted to use
real masks instead of boxes:

  4. Same-bag duplicate instances (repetitive sack texture triggering two
     confident detections on one physical bag) are now merged in two
     stages -- Stage A on actual mask pixel overlap, Stage B on mask
     centroid distance for the rarer case where two duplicate masks barely
     touch. See the "DUPLICATE-INSTANCE POST-PROCESSING" section below for
     the full explanation.
  5. The model's own NMS was tight enough (iou=0.50) to also suppress
     genuinely distinct, tightly-touching neighbour bags -- that's the
     "bags choot rahe hain" symptom. Default iou raised to 0.65 (looser
     NMS keeps more real detections) and max_det raised 1000 -> 3000 for
     dense piles; the new mask-based Stage A/B below safely cleans up the
     extra duplicates this now lets through.
"""

from flask import Flask, request, render_template_string
from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
import base64
import time
import torch
import statistics


# ============================================================
# RICE BAG V11
# PROFESSIONAL INSTANCE SEGMENTATION TESTING UI
# ============================================================

app = Flask(__name__)

PROJECT_ROOT = Path.cwd()
RUNS_DIR = PROJECT_ROOT / "runs"

DEVICE = 0 if torch.cuda.is_available() else "cpu"


# ============================================================
# DUPLICATE-INSTANCE DEDUP SETTINGS
#
# Stage A: two instances whose actual MASKS overlap heavily are almost
# certainly the same physical bag (pixel overlap is the real ground truth
# for segmentation -- much more reliable than box IoU, which can also be
# high for two genuinely different, tightly-touching adjacent bags).
#
# Stage B: catches the rarer case where a repetitive fold/stitch pattern
# on one sack triggers two confident detections on two different
# sub-regions of that SAME bag -- their masks barely overlap (so Stage A
# doesn't merge them) but their centroids sit right next to each other.
# Same idea as CENTER_DIST_FACTOR in test_rice_bag_v9_ui.py, just driven
# off mask centroids/box size instead of box centers.
# ============================================================

DEDUP_MASK_IOU_THRESHOLD = 0.45
DEDUP_MASK_CONTAINMENT_THRESHOLD = 0.70
CENTER_DIST_FACTOR = 0.45


# ============================================================
# MODEL AUTO DISCOVERY
# ============================================================

def find_model():

    print("\n" + "=" * 80)
    print("SEARCHING FOR V11 SEGMENTATION MODEL")
    print("=" * 80)

    # --------------------------------------------------------
    # Search best.pt first
    #
    # FIX: filter changed from "rice_bag_11" to
    # "rice_bag_v11_segmentation" -- that's the actual run name
    # this model is trained/saved under (verified against the
    # real output path), the old filter matches nothing.
    # --------------------------------------------------------

    best_models = [
        p for p in RUNS_DIR.rglob("best.pt")
        if "rice_bag_v11_segmentation" in str(p)
    ]

    if best_models:

        best_models.sort(
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        model_path = best_models[0]

        print("\nBEST MODEL FOUND:")
        print(model_path.resolve())

        return model_path


    # --------------------------------------------------------
    # Fallback to last.pt
    # --------------------------------------------------------

    last_models = [
        p for p in RUNS_DIR.rglob("last.pt")
        if "rice_bag_v11_segmentation" in str(p)
    ]

    if last_models:

        last_models.sort(
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        model_path = last_models[0]

        print("\nBEST MODEL NOT FOUND.")
        print("USING LAST CHECKPOINT:")
        print(model_path.resolve())

        return model_path


    # --------------------------------------------------------
    # Nothing found
    # --------------------------------------------------------

    print("\nNO V11 MODEL FOUND.")

    print("\nAvailable .pt files:")

    for p in RUNS_DIR.rglob("*.pt"):
        print(p.resolve())

    raise FileNotFoundError(
        "\nNo V11 segmentation checkpoint found."
    )


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = find_model()

print("\n" + "=" * 80)
print("LOADING MODEL")
print("=" * 80)

print(
    f"\nModel:\n{MODEL_PATH.resolve()}"
)

model = YOLO(str(MODEL_PATH))

print("\nModel loaded successfully.")


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = {
    0: "Rice-bag-10",
    1: "other",
    2: "pile"
}


# ============================================================
# DUPLICATE-INSTANCE POST-PROCESSING
# ============================================================

def _mask_iou_and_containment(mask_a, mask_b):
    """IoU and containment between two boolean masks (same shape)."""

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
    """Stage A: union-find merge of same-class instances whose actual
    (low-res, pre-resize) masks overlap heavily. A cheap box-overlap
    pre-filter skips pairs that can't possibly touch before paying for the
    real per-pixel mask comparison. Keeps the highest-confidence instance
    per merged cluster.
    """

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

            # cheap reject: boxes don't even overlap
            if bi[2] <= bj[0] or bj[2] <= bi[0] or bi[3] <= bj[1] or bj[3] <= bi[1]:
                continue

            iou, containment = _mask_iou_and_containment(
                low_res_masks[i], low_res_masks[j]
            )

            if iou >= iou_thresh or containment >= containment_thresh:
                union(i, j)

    clusters = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    keep = [max(members, key=lambda idx: confs[idx]) for members in clusters.values()]

    return sorted(keep)


def cluster_center_points(centers, sizes, cls_ids, confs, factor=CENTER_DIST_FACTOR):
    """Stage B: union-find merge of same-class instances (already survivors
    of Stage A) whose centroids are closer than `factor` times the typical
    instance size. Operates on, and returns, LOCAL indices into the lists
    passed in. Keeps the highest-confidence instance per merged cluster.
    """

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
            dist = (dx * dx + dy * dy) ** 0.5

            if dist <= threshold:
                union(i, j)

    clusters = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    keep_local = [max(members, key=lambda idx: confs[idx]) for members in clusters.values()]

    return sorted(keep_local)


# ============================================================
# HTML
# ============================================================

HTML = r"""
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>
Rice Bag AI Vision | V11
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

    border: none;

    border-radius: 10px;

    background: #2563eb;

    color: white;

    font-size: 15px;

    font-weight: 700;

    cursor: pointer;

}


.run-button:hover {

    background: #1d4ed8;

}


/* =========================================================
   SETTINGS
   ========================================================= */

.settings {

    margin-top: 18px;

    display: grid;

    grid-template-columns:
        repeat(4, 1fr);

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


input[type=range] {

    width: 100%;

    margin-top: 8px;

}


.check-row {

    display: flex;

    gap: 14px;

    flex-wrap: wrap;

    margin-top: 10px;

}


.check {

    display: flex;

    gap: 6px;

    align-items: center;

    font-size: 13px;

    color: #cbd5e1;

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

    font-size: 30px;

    font-weight: 800;

    margin-top: 8px;

}


.kpi-sub {

    color: #64748b;

    font-size: 11px;

    margin-top: 4px;

}


/* =========================================================
   IMAGE GRID
   ========================================================= */

.image-grid {

    display: grid;

    grid-template-columns:
        1fr 1fr;

    gap: 20px;

    margin-bottom: 20px;

}


.image-card {

    background: #111827;

    border:
        1px solid #1f2937;

    border-radius: 16px;

    overflow: hidden;

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

}


.image-container img {

    width: 100%;

    max-height: 700px;

    object-fit: contain;

    border-radius: 10px;

}


/* =========================================================
   DETAILS
   ========================================================= */

.details-grid {

    display: grid;

    grid-template-columns:
        1fr 1fr;

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


.model-info {

    padding: 18px;

}


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

    text-align: right;

    word-break: break-word;

}


/* =========================================================
   TABLE
   ========================================================= */

.table-wrapper {

    overflow-x: auto;

}


table {

    width: 100%;

    border-collapse: collapse;

    font-size: 13px;

}


th {

    text-align: left;

    color: #64748b;

    font-weight: 600;

    padding: 12px 16px;

    border-bottom:
        1px solid #1f2937;

}


td {

    padding: 12px 16px;

    border-bottom:
        1px solid #1e2937;

}


.conf {

    font-weight: 700;

}


.empty {

    padding: 30px;

    text-align: center;

    color: #64748b;

}


/* =========================================================
   ERROR
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

    .image-grid,
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
                V11 • Instance Segmentation • Warehouse POC
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
     ERROR
     ====================================================== -->

{% if error %}

<div class="error">

    {{ error }}

</div>

{% endif %}


<!-- ======================================================
     UPLOAD
     ====================================================== -->

<div class="upload-card">

<form method="POST"
      enctype="multipart/form-data">


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
    name="image"
    accept="image/*"
    required
>


<!-- ======================================================
     SETTINGS
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
        name="confidence"
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
        name="iou"
        id="iou"
        min="0.20"
        max="0.90"
        step="0.05"
        value="{{ iou }}"
    >

</div>


<div class="setting">

    <div class="setting-title">
        Visualization
    </div>

    <div class="check-row">

        <label class="check">

            <input
                type="checkbox"
                name="show_masks"
                checked
            >

            Masks

        </label>


        <label class="check">

            <input
                type="checkbox"
                name="show_points"
                checked
            >

            Points

        </label>


        <label class="check">

            <input
                type="checkbox"
                name="show_labels"
                checked
            >

            Labels

        </label>

    </div>

</div>


<div class="setting">

    <div class="setting-title">
        Inference Resolution
    </div>

    <div class="setting-value">
        1280 × 1280
    </div>

    <div
        style="
        color:#64748b;
        font-size:11px;
        margin-top:6px;
        "
    >
        Segmentation optimized
    </div>

</div>


</div>


<button
    class="run-button"
    type="submit"
>

    🚀 Run AI Segmentation

</button>


</form>

</div>


{% if result %}


<!-- ======================================================
     KPI DASHBOARD
     ====================================================== -->

<div class="kpis">


<div class="kpi">

    <div class="kpi-top">
        Rice Bags
    </div>

    <div class="kpi-number">
        {{ total_bags }}
    </div>

    <div class="kpi-sub">
        Individual bag instances
    </div>

</div>


<div class="kpi">

    <div class="kpi-top">
        Other
    </div>

    <div class="kpi-number">
        {{ total_other }}
    </div>

    <div class="kpi-sub">
        Non-bag detections
    </div>

</div>


<div class="kpi">

    <div class="kpi-top">
        Pile
    </div>

    <div class="kpi-number">
        {{ total_pile }}
    </div>

    <div class="kpi-sub">
        Pile instances
    </div>

</div>


<div class="kpi">

    <div class="kpi-top">
        Confidence
    </div>

    <div class="kpi-number">
        {{ avg_confidence }}%
    </div>

    <div class="kpi-sub">
        Average detection confidence
    </div>

</div>


<div class="kpi">

    <div class="kpi-top">
        Inference
    </div>

    <div class="kpi-number">
        {{ inference_ms }} ms
    </div>

    <div class="kpi-sub">
        {{ fps }} FPS equivalent
    </div>

</div>


</div>


<!-- ======================================================
     ORIGINAL / RESULT
     ====================================================== -->

<div class="image-grid">


<div class="image-card">

    <div class="card-header">

        <h2>
            Original Image
        </h2>

        <span class="badge">
            INPUT
        </span>

    </div>


    <div class="image-container">

        <img
            src="data:image/jpeg;base64,{{ original }}"
        >

    </div>

</div>


<div class="image-card">

    <div class="card-header">

        <h2>
            AI Segmentation
        </h2>

        <span class="badge">
            OUTPUT
        </span>

    </div>


    <div class="image-container">

        <img
            src="data:image/jpeg;base64,{{ result }}"
        >

    </div>

</div>


</div>


<!-- ======================================================
     DETAILS
     ====================================================== -->

<div class="details-grid">


<!-- MODEL INFO -->

<div class="panel">

    <div class="panel-title">
        🧠 Model Information
    </div>


    <div class="model-info">


        <div class="info-row">

            <span class="info-label">
                Model
            </span>

            <span class="info-value">
                {{ model_name }}
            </span>

        </div>


        <div class="info-row">

            <span class="info-label">
                Architecture
            </span>

            <span class="info-value">
                YOLO Segmentation
            </span>

        </div>


        <div class="info-row">

            <span class="info-label">
                Annotation
            </span>

            <span class="info-value">
                Polygon / Instance Mask
            </span>

        </div>


        <div class="info-row">

            <span class="info-label">
                Input
            </span>

            <span class="info-value">
                {{ image_size }}
            </span>

        </div>


        <div class="info-row">

            <span class="info-label">
                Confidence
            </span>

            <span class="info-value">
                {{ confidence }}
            </span>

        </div>


        <div class="info-row">

            <span class="info-label">
                IoU
            </span>

            <span class="info-value">
                {{ iou }}
            </span>

        </div>


    </div>

</div>


<!-- DETECTION TABLE -->

<div class="panel">

    <div class="panel-title">
        📋 Detection Details
    </div>


    <div class="table-wrapper">

    {% if detections %}

    <table>

        <thead>

        <tr>

            <th>
                ID
            </th>

            <th>
                Class
            </th>

            <th>
                Confidence
            </th>

        </tr>

        </thead>


        <tbody>

        {% for d in detections %}

        <tr>

            <td>
                {{ d.id }}
            </td>

            <td>
                {{ d.class_name }}
            </td>

            <td class="conf">
                {{ d.confidence }}
            </td>

        </tr>

        {% endfor %}

        </tbody>

    </table>

    {% else %}

    <div class="empty">
        No detections found.
    </div>

    {% endif %}

    </div>

</div>


</div>


{% endif %}


<div class="footer">

    Rice Bag AI Vision • V11 Segmentation Testing

</div>


</div>


<script>


// ==========================================================
// FILE INPUT
// ==========================================================

const dropZone =
    document.getElementById("dropZone");

const imageInput =
    document.getElementById("imageInput");

const fileName =
    document.getElementById("fileName");


dropZone.addEventListener(
    "click",
    () => imageInput.click()
);


imageInput.addEventListener(
    "change",
    () => {

        if (imageInput.files.length > 0) {

            fileName.textContent =
                "Selected: " +
                imageInput.files[0].name;

        }

    }
);


// ==========================================================
// DRAG & DROP
// ==========================================================

dropZone.addEventListener(
    "dragover",
    (event) => {

        event.preventDefault();

        dropZone.classList.add(
            "dragover"
        );

    }
);


dropZone.addEventListener(
    "dragleave",
    () => {

        dropZone.classList.remove(
            "dragover"
        );

    }
);


dropZone.addEventListener(
    "drop",
    (event) => {

        event.preventDefault();

        dropZone.classList.remove(
            "dragover"
        );


        if (
            event.dataTransfer.files.length > 0
        ) {

            imageInput.files =
                event.dataTransfer.files;

            fileName.textContent =
                "Selected: " +
                event.dataTransfer.files[0].name;

        }

    }
);


// ==========================================================
// CONFIDENCE SLIDER
// ==========================================================

const confSlider =
    document.getElementById("confidence");

const confValue =
    document.getElementById("confValue");


confSlider.addEventListener(
    "input",
    () => {

        confValue.textContent =
            confSlider.value;

    }
);


// ==========================================================
// IOU SLIDER
// ==========================================================

const iouSlider =
    document.getElementById("iou");

const iouValue =
    document.getElementById("iouValue");


iouSlider.addEventListener(
    "input",
    () => {

        iouValue.textContent =
            iouSlider.value;

    }
);

</script>


</body>

</html>
"""


# ============================================================
# IMAGE ENCODING
# ============================================================

def encode_image(image):

    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            90
        ]
    )

    if not success:
        raise ValueError(
            "Could not encode image."
        )

    return base64.b64encode(
        encoded.tobytes()
    ).decode("utf-8")


# ============================================================
# ROUTE
# ============================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
def index():

    # --------------------------------------------------------
    # Defaults
    # --------------------------------------------------------

    result_image = None

    original_image = None

    total_bags = 0

    total_other = 0

    total_pile = 0

    total_objects = 0

    avg_confidence = 0

    inference_ms = 0

    fps = 0

    image_size = "-"

    detections = []

    error = None

    confidence = 0.25

    iou = 0.65


    # ========================================================
    # POST
    # ========================================================

    if request.method == "POST":

        try:

            # ------------------------------------------------
            # SETTINGS
            # ------------------------------------------------

            confidence = float(
                request.form.get(
                    "confidence",
                    0.25
                )
            )

            iou = float(
                request.form.get(
                    "iou",
                    0.65
                )
            )


            show_masks = (
                request.form.get(
                    "show_masks"
                )
                is not None
            )

            show_points = (
                request.form.get(
                    "show_points"
                )
                is not None
            )

            show_labels = (
                request.form.get(
                    "show_labels"
                )
                is not None
            )


            # ------------------------------------------------
            # FILE
            # ------------------------------------------------

            file = request.files.get(
                "image"
            )


            if file is None:

                raise ValueError(
                    "Please select an image."
                )


            # ------------------------------------------------
            # READ IMAGE
            # ------------------------------------------------

            file_bytes = np.frombuffer(
                file.read(),
                np.uint8
            )


            image = cv2.imdecode(
                file_bytes,
                cv2.IMREAD_COLOR
            )


            if image is None:

                raise ValueError(
                    "Could not read uploaded image."
                )


            h, w = image.shape[:2]

            image_size = (
                f"{w} × {h}"
            )


            # ------------------------------------------------
            # ORIGINAL
            # ------------------------------------------------

            original_image = encode_image(
                image
            )


            # ------------------------------------------------
            # INFERENCE
            #
            # FIX: device=0 -> device=DEVICE (auto-detected).
            #
            # iou raised (default 0.50 -> 0.65) and max_det raised
            # (1000 -> 3000): the model's own NMS was suppressing
            # genuinely distinct, tightly-touching bags as if they were
            # duplicates. It's fine to loosen it now -- the mask-based
            # Stage A/B dedup below cleans up the real duplicates this
            # lets through, using actual pixel overlap instead of boxes.
            # ------------------------------------------------

            start_time = time.perf_counter()


            results = model.predict(

                source=image,

                imgsz=1280,

                conf=confidence,

                iou=iou,

                max_det=3000,

                device=DEVICE,

                verbose=False
            )


            elapsed = (
                time.perf_counter()
                - start_time
            )


            inference_ms = round(
                elapsed * 1000,
                1
            )


            if elapsed > 0:

                fps = round(
                    1 / elapsed,
                    2
                )


            result = results[0]


            # ------------------------------------------------
            # OUTPUT
            # ------------------------------------------------

            output = image.copy()


            # ------------------------------------------------
            # BOX DATA
            # ------------------------------------------------

            if result.boxes is not None:

                classes = (
                    result.boxes.cls
                    .cpu()
                    .numpy()
                    .astype(int)
                )

                confidences = (
                    result.boxes.conf
                    .cpu()
                    .numpy()
                )

                boxes_xyxy = (
                    result.boxes.xyxy
                    .cpu()
                    .numpy()
                )

            else:

                classes = []

                confidences = []

                boxes_xyxy = np.zeros((0, 4))


            # ------------------------------------------------
            # MASK DATA
            # ------------------------------------------------

            if result.masks is not None:

                masks = (
                    result.masks.data
                    .cpu()
                    .numpy()
                )

            else:

                masks = None


            # ------------------------------------------------
            # CONFIDENCE
            # ------------------------------------------------

            if len(confidences) > 0:

                avg_confidence = round(
                    float(
                        np.mean(confidences)
                    ) * 100,
                    1
                )


            # ------------------------------------------------
            # DUPLICATE-INSTANCE DEDUP (Stage A + Stage B)
            #
            # Runs on the RAW low-res masks (fast, correct) before any
            # per-instance drawing work happens, so everything below
            # this point only ever iterates over already-deduped,
            # final survivors.
            # ------------------------------------------------

            if len(classes) > 0 and masks is not None:

                low_res_masks_bool = masks > 0.5

                stage_a_survivors = deduplicate_instances(
                    low_res_masks_bool,
                    boxes_xyxy,
                    confidences,
                    classes,
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
                    centers,
                    sizes,
                    [classes[idx] for idx in stage_a_survivors],
                    [confidences[idx] for idx in stage_a_survivors],
                )

                final_indices = sorted(
                    stage_a_survivors[k] for k in stage_b_keep_local
                )

            else:

                final_indices = list(range(len(classes)))


            # FIX: accumulate all instances' mask colors into ONE
            # overlay and blend it onto the image a single time
            # after the loop, instead of calling cv2.addWeighted
            # once PER instance (which re-darkens the already
            # blended image every iteration -- with a few hundred
            # detections that decays to solid black). Contour /
            # point / label drawing is likewise deferred to after
            # the blend so it isn't drawn onto an image that keeps
            # getting darkened underneath it.
            overlay_accum = np.zeros_like(output)
            draw_items = []

            bag_number = 0


            # =================================================
            # PROCESS EACH FINAL (DEDUPED) INSTANCE
            # =================================================

            for i in final_indices:

                cls_id = classes[i]

                conf = float(
                    confidences[i]
                )


                # ------------------------------------------------
                # CLASS COUNT
                # ------------------------------------------------

                if cls_id == 0:

                    bag_number += 1

                    total_bags += 1


                elif cls_id == 1:

                    total_other += 1


                elif cls_id == 2:

                    total_pile += 1


                # ------------------------------------------------
                # DETECTION TABLE
                # ------------------------------------------------

                if cls_id == 0:

                    display_name = (
                        f"Bag {bag_number}"
                    )

                    detection_id = (
                        bag_number
                    )

                else:

                    display_name = (
                        CLASS_NAMES.get(
                            cls_id,
                            "Unknown"
                        )
                    )

                    detection_id = (
                        i + 1
                    )


                detections.append({

                    "id": detection_id,

                    "class_name": display_name,

                    "confidence": (
                        f"{conf * 100:.1f}%"
                    )

                })


                # ------------------------------------------------
                # MASK
                # ------------------------------------------------

                if (
                    masks is None
                    or i >= len(masks)
                ):

                    continue


                mask = masks[i]


                mask = cv2.resize(

                    mask,

                    (w, h),

                    interpolation=cv2.INTER_NEAREST
                )


                binary_mask = (
                    mask > 0.5
                ).astype(np.uint8)


                # =================================================
                # MASK OVERLAY (accumulate only; blended once
                # after the loop -- see overlay_accum above)
                # =================================================

                if show_masks:

                    mask_bool = binary_mask.astype(bool)

                    if cls_id == 0:

                        # Rice bag
                        overlay_accum[:, :, 1][mask_bool] = 255


                    elif cls_id == 1:

                        # Other
                        overlay_accum[:, :, 2][mask_bool] = 255


                    elif cls_id == 2:

                        # Pile
                        overlay_accum[:, :, 0][mask_bool] = 255


                # =================================================
                # MASK CONTOUR (drawn after the blend, see below)
                # =================================================

                contours, _ = cv2.findContours(

                    binary_mask,

                    cv2.RETR_EXTERNAL,

                    cv2.CHAIN_APPROX_SIMPLE
                )


                # =================================================
                # CENTER POINT
                # =================================================

                M = cv2.moments(
                    binary_mask
                )


                if M["m00"] != 0:

                    cx = int(
                        M["m10"] /
                        M["m00"]
                    )

                    cy = int(
                        M["m01"] /
                        M["m00"]
                    )

                else:

                    ys, xs = np.where(
                        binary_mask > 0
                    )


                    if len(xs) == 0:

                        continue


                    cx = int(
                        np.mean(xs)
                    )

                    cy = int(
                        np.mean(ys)
                    )


                # =================================================
                # LABEL TEXT
                # =================================================

                if cls_id == 0:

                    label = (
                        f"Bag {bag_number} "
                        f"{conf:.2f}"
                    )

                else:

                    label = (
                        f"{CLASS_NAMES.get(cls_id, 'Unknown')} "
                        f"{conf:.2f}"
                    )

                draw_items.append(
                    (contours, cx, cy, label)
                )


            # =================================================
            # BLEND MASKS ONTO IMAGE (once, not per-instance)
            # =================================================

            if show_masks:

                output = cv2.addWeighted(
                    output,
                    0.72,
                    overlay_accum,
                    0.28,
                    0
                )


            # =================================================
            # DRAW CONTOURS / CENTER POINTS / LABELS ON TOP OF
            # THE BLENDED IMAGE
            # =================================================

            for contours, cx, cy, label in draw_items:

                if show_masks:

                    cv2.drawContours(
                        output,
                        contours,
                        -1,
                        (255, 255, 255),
                        2
                    )


                if show_points:

                    cv2.circle(
                        output,
                        (cx, cy),
                        7,
                        (255, 255, 255),
                        -1
                    )

                    cv2.circle(
                        output,
                        (cx, cy),
                        11,
                        (0, 0, 0),
                        2
                    )


                if show_labels:

                    cv2.putText(
                        output,
                        label,
                        (cx + 12, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )


            # =================================================
            # TOTAL OBJECTS
            # =================================================

            total_objects = (
                total_bags
                + total_other
                + total_pile
            )


            # =================================================
            # ENCODE RESULT
            # =================================================

            result_image = encode_image(
                output
            )


        except Exception as e:

            error = (
                f"{type(e).__name__}: {str(e)}"
            )


    # ========================================================
    # RENDER
    # ========================================================

    return render_template_string(

        HTML,

        result=result_image,

        original=original_image,

        total_bags=total_bags,

        total_other=total_other,

        total_pile=total_pile,

        total_objects=total_objects,

        avg_confidence=avg_confidence,

        inference_ms=inference_ms,

        fps=fps,

        image_size=image_size,

        detections=detections,

        error=error,

        confidence=confidence,

        iou=iou,

        model_name=str(
            MODEL_PATH.resolve()
        )
    )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 80)

    print(
        "RICE BAG AI VISION - V11 SEGMENTATION UI"
    )

    print("=" * 80)

    print("\nLoaded model:")

    print(
        MODEL_PATH.resolve()
    )

    print("\nOpen:")

    print(
        "http://127.0.0.1:5000"
    )

    print("\nRemote PC:")

    print(
        "http://<SERVER-IP>:5000"
    )

    print("\nPress CTRL+C to stop.")

    print("=" * 80)


    app.run(

        host="0.0.0.0",

        port=5000,

        debug=False,

        threaded=True
    )
