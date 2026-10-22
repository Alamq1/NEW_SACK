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
# CONFIGURATION
# ============================================================

MODEL_PATH = "runs/detect/runs/sack_keypoint/weights/best.pt"

IMAGE_SIZE = 640

DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.50

# Ultralytics caps detections per image at 300 by default -- that's why
# every image was reporting exactly 300 regardless of how many sacks were
# actually present. Raise it well above what any single pile photo should
# realistically contain.
MAX_DETECTIONS = 3000

DEVICE = 0 if torch.cuda.is_available() else "cpu"

# --- Duplicate-detection post-processing (same approach as
# src/test_rice_bag_v9_ui.py) -- this model has no NMS-level cleanup of
# its own, so without this step the same physical sack fires multiple
# overlapping detections (fold lines / repetitive texture), massively
# inflating the count. Stage A merges boxes that overlap or are mostly
# contained in each other; Stage B then merges dots that are still close
# together relative to sack size even when their boxes barely overlapped.
DEDUP_IOU_THRESHOLD = 0.50
DEDUP_CONTAINMENT_THRESHOLD = 0.75
DEDUP_MODE = "merge"
CENTER_DIST_FACTOR = 0.35


# ============================================================
# STAGE A - BOX-LEVEL DEDUP
# ============================================================

def _box_area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _iou_and_containment(box_a, box_b):
    ix1, iy1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    ix2, iy2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0, 0.0
    area_a, area_b = _box_area(box_a), _box_area(box_b)
    union = area_a + area_b - inter
    iou = inter / union if union > 0 else 0.0
    containment = inter / max(1e-6, min(area_a, area_b))
    return iou, containment


def deduplicate_boxes(xyxy, confs,
                       iou_thresh=DEDUP_IOU_THRESHOLD,
                       containment_thresh=DEDUP_CONTAINMENT_THRESHOLD,
                       mode=DEDUP_MODE):
    n = len(xyxy)
    if n == 0:
        return np.zeros((0, 4)), np.zeros((0,))

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
            iou, containment = _iou_and_containment(xyxy[i], xyxy[j])
            if iou >= iou_thresh or containment >= containment_thresh:
                union(i, j)

    clusters = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    merged_boxes = []
    merged_confs = []

    for members in clusters.values():
        if mode == "suppress" or len(members) == 1:
            best = max(members, key=lambda idx: confs[idx])
            merged_boxes.append(xyxy[best])
            merged_confs.append(confs[best])
            continue

        member_confs = np.array([confs[m] for m in members], dtype=np.float64)
        weights = member_confs / member_confs.sum()
        coords = np.array([xyxy[m] for m in members], dtype=np.float64)
        fused = (coords * weights[:, None]).sum(axis=0)

        merged_boxes.append(fused)
        merged_confs.append(float(member_confs.max()))

    return np.array(merged_boxes), np.array(merged_confs)


# ============================================================
# STAGE B - CENTER-POINT DISTANCE CLUSTERING
# ============================================================

def cluster_center_points(boxes, distance_factor=CENTER_DIST_FACTOR):
    n = len(boxes)
    if n <= 1:
        return boxes

    widths = [b["w"] for b in boxes if b["w"] > 0]
    heights = [b["h"] for b in boxes if b["h"] > 0]
    median_w = statistics.median(widths) if widths else 20.0
    median_h = statistics.median(heights) if heights else 20.0
    radius = distance_factor * (median_w + median_h) / 2.0

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
            dx = boxes[i]["cx"] - boxes[j]["cx"]
            dy = boxes[i]["cy"] - boxes[j]["cy"]
            if (dx * dx + dy * dy) ** 0.5 <= radius:
                union(i, j)

    clusters = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    fused_boxes = []

    for members in clusters.values():
        if len(members) == 1:
            fused_boxes.append(boxes[members[0]])
            continue

        member_confs = np.array([boxes[m]["conf"] for m in members], dtype=np.float64)
        weights = member_confs / member_confs.sum()

        cx = sum(boxes[m]["cx"] * w for m, w in zip(members, weights))
        cy = sum(boxes[m]["cy"] * w for m, w in zip(members, weights))
        x1 = sum(boxes[m]["x1"] * w for m, w in zip(members, weights))
        y1 = sum(boxes[m]["y1"] * w for m, w in zip(members, weights))
        x2 = sum(boxes[m]["x2"] * w for m, w in zip(members, weights))
        y2 = sum(boxes[m]["y2"] * w for m, w in zip(members, weights))

        fused_boxes.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "cx": cx, "cy": cy,
            "w": x2 - x1, "h": y2 - y1,
            "conf": float(member_confs.max()),
        })

    return fused_boxes


def get_final_sack_points(result):
    """Run Stage A + Stage B on a YOLO result and return a clean list of
    one dict per PHYSICAL sack: {x1,y1,x2,y2,cx,cy,w,h,conf}."""

    boxes_obj = result.boxes
    if boxes_obj is None or len(boxes_obj) == 0:
        return []

    xyxy = boxes_obj.xyxy.cpu().numpy()
    confs = boxes_obj.conf.cpu().numpy()

    xyxy, confs = deduplicate_boxes(xyxy, confs)

    stage_a_boxes = []
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = map(float, xyxy[i])
        conf = float(confs[i])
        stage_a_boxes.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "cx": (x1 + x2) / 2.0, "cy": (y1 + y2) / 2.0,
            "w": x2 - x1, "h": y2 - y1,
            "conf": conf,
        })

    return cluster_center_points(stage_a_boxes)


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024


# ============================================================
# LOAD MODEL ONCE
# ============================================================

print("\n" + "=" * 70)
print("        SACK KEYPOINT TESTING UI")
print("=" * 70)

print(f"Model: {MODEL_PATH}")

if not Path(MODEL_PATH).exists():

    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}"
    )


model = YOLO(MODEL_PATH)

print("Model loaded successfully.")

print("=" * 70)


# ============================================================
# HTML
# ============================================================

HTML = r"""
<!DOCTYPE html>

<html>

<head>

    <title>Sack Keypoint Detection</title>

    <meta name="viewport"
          content="width=device-width, initial-scale=1">

    <style>

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: #f4f6f8;
            color: #222;
        }

        .header {
            background: #111827;
            color: white;
            padding: 20px 30px;
        }

        .header h1 {
            margin: 0;
            font-size: 25px;
        }

        .header p {
            margin: 6px 0 0;
            color: #cbd5e1;
        }

        .container {
            max-width: 1400px;
            margin: 25px auto;
            padding: 0 20px;
        }

        .panel {
            background: white;
            border-radius: 14px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 3px 15px rgba(0,0,0,0.08);
        }

        .controls {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            align-items: end;
        }

        .control {
            display: flex;
            flex-direction: column;
            gap: 7px;
        }

        label {
            font-size: 13px;
            font-weight: bold;
        }

        input[type="number"] {
            width: 120px;
            padding: 10px;
            border: 1px solid #d1d5db;
            border-radius: 8px;
        }

        input[type="file"] {
            padding: 10px 0;
        }

        button {
            border: none;
            border-radius: 8px;
            padding: 11px 22px;
            cursor: pointer;
            font-weight: bold;
            background: #2563eb;
            color: white;
        }

        button:hover {
            opacity: 0.9;
        }

        .result-grid {
            display: grid;
            grid-template-columns: 1fr 320px;
            gap: 20px;
        }

        .image-container {
            background: #111;
            border-radius: 12px;
            padding: 10px;
            text-align: center;
        }

        .image-container img {
            max-width: 100%;
            max-height: 750px;
            object-fit: contain;
            border-radius: 8px;
        }

        .stats {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }

        .stat {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 18px;
        }

        .stat-title {
            font-size: 13px;
            color: #64748b;
        }

        .stat-value {
            font-size: 32px;
            font-weight: bold;
            margin-top: 5px;
        }

        .success {
            color: #16a34a;
        }

        .warning {
            color: #dc2626;
        }

        .info {
            color: #2563eb;
        }

        .filename {
            margin-top: 10px;
            color: #64748b;
            word-break: break-all;
        }

        @media(max-width: 900px) {

            .result-grid {
                grid-template-columns: 1fr;
            }

        }

    </style>

</head>


<body>


<div class="header">

    <h1>Rice Sack Keypoint Detection</h1>

    <p>
        YOLO26L • Sack_keypoint • Best Model
    </p>

</div>


<div class="container">


    <!-- ================================================= -->
    <!-- UPLOAD PANEL -->
    <!-- ================================================= -->

    <div class="panel">

        <form method="POST"
              enctype="multipart/form-data">

            <div class="controls">


                <div class="control">

                    <label>
                        Upload Image
                    </label>

                    <input
                        type="file"
                        name="image"
                        accept="image/*"
                        required>

                </div>


                <div class="control">

                    <label>
                        Confidence
                    </label>

                    <input
                        type="number"
                        name="conf"
                        value="{{ conf }}"
                        min="0.01"
                        max="0.99"
                        step="0.01">

                </div>


                <div class="control">

                    <label>
                        IoU
                    </label>

                    <input
                        type="number"
                        name="iou"
                        value="{{ iou }}"
                        min="0.01"
                        max="0.99"
                        step="0.01">

                </div>


                <div class="control">

                    <button type="submit">
                        Detect Sacks
                    </button>

                </div>


            </div>

        </form>

    </div>


    {% if image %}


    <!-- ================================================= -->
    <!-- RESULT -->
    <!-- ================================================= -->

    <div class="result-grid">


        <div class="panel">

            <h2>
                Detection Result
            </h2>

            <div class="image-container">

                <img src="data:image/jpeg;base64,{{ image }}">

            </div>

            <div class="filename">

                {{ filename }}

            </div>

        </div>


        <!-- ================================================= -->
        <!-- STATISTICS -->
        <!-- ================================================= -->

        <div class="stats">


            <div class="stat">

                <div class="stat-title">
                    TOTAL SACKS DETECTED
                </div>

                <div class="stat-value success">
                    {{ count }}
                </div>

            </div>


            <div class="stat">

                <div class="stat-title">
                    CONFIDENCE
                </div>

                <div class="stat-value info">
                    {{ conf }}
                </div>

            </div>


            <div class="stat">

                <div class="stat-title">
                    IoU
                </div>

                <div class="stat-value info">
                    {{ iou }}
                </div>

            </div>


            <div class="stat">

                <div class="stat-title">
                    INFERENCE TIME
                </div>

                <div class="stat-value">
                    {{ inference }} ms
                </div>

            </div>


        </div>


    </div>


    {% endif %}


</div>


</body>

</html>
"""


# ============================================================
# DRAW CLEAN CENTER POINTS
# ============================================================

def draw_center_points(image, points):
    """points: list of dicts from get_final_sack_points -- already
    deduped, so this just draws, it doesn't decide what counts as a
    sack."""

    output = image.copy()

    for index, point in enumerate(points):

        # ----------------------------------------------------
        # CENTER POINT
        # ----------------------------------------------------

        cx = int(point["cx"])
        cy = int(point["cy"])

        # ----------------------------------------------------
        # DRAW CENTER
        # ----------------------------------------------------

        cv2.circle(
            output,
            (cx, cy),
            7,
            (0, 255, 0),
            -1
        )

        # ----------------------------------------------------
        # WHITE OUTLINE
        # ----------------------------------------------------

        cv2.circle(
            output,
            (cx, cy),
            9,
            (255, 255, 255),
            2
        )

        # ----------------------------------------------------
        # SACK NUMBER
        # ----------------------------------------------------

        number = str(index + 1)

        cv2.putText(
            output,
            number,
            (cx + 10, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    return output


# ============================================================
# ENCODE IMAGE
# ============================================================

def encode_image(image):

    success, buffer = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 95]
    )

    if not success:
        return None

    return base64.b64encode(
        buffer
    ).decode("utf-8")


# ============================================================
# ROUTE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def index():

    image_base64 = None

    count = 0

    filename = ""

    inference_time = 0

    conf = DEFAULT_CONF

    iou = DEFAULT_IOU


    # ========================================================
    # POST
    # ========================================================

    if request.method == "POST":

        uploaded_file = request.files.get("image")

        # ----------------------------------------------------
        # Validate file
        # ----------------------------------------------------

        if uploaded_file is None:

            return render_template_string(
                HTML,
                image=None,
                count=0,
                filename="",
                inference=0,
                conf=conf,
                iou=iou
            )

        # ----------------------------------------------------
        # Parameters
        # ----------------------------------------------------

        try:

            conf = float(
                request.form.get(
                    "conf",
                    DEFAULT_CONF
                )
            )

        except:

            conf = DEFAULT_CONF


        try:

            iou = float(
                request.form.get(
                    "iou",
                    DEFAULT_IOU
                )
            )

        except:

            iou = DEFAULT_IOU


        conf = max(
            0.01,
            min(0.99, conf)
        )

        iou = max(
            0.01,
            min(0.99, iou)
        )


        # ----------------------------------------------------
        # Read uploaded image
        # ----------------------------------------------------

        file_bytes = uploaded_file.read()

        np_array = np.frombuffer(
            file_bytes,
            np.uint8
        )

        image = cv2.imdecode(
            np_array,
            cv2.IMREAD_COLOR
        )


        if image is None:

            return "Invalid image", 400


        filename = uploaded_file.filename


        # ====================================================
        # INFERENCE
        # ====================================================

        start = time.perf_counter()


        results = model.predict(

            source=image,

            imgsz=IMAGE_SIZE,

            conf=conf,

            iou=iou,

            max_det=MAX_DETECTIONS,

            device=DEVICE,

            verbose=False

        )


        inference_time = (
            time.perf_counter() - start
        ) * 1000


        result = results[0]


        # ====================================================
        # DEDUP: collapse multi-box-per-sack detections down to
        # one point per PHYSICAL sack (Stage A box merge + Stage B
        # center-point clustering) before counting or drawing.
        # ====================================================

        sack_points = get_final_sack_points(result)

        count = len(sack_points)


        # ====================================================
        # DRAW ONLY CENTER POINTS
        # ====================================================

        output = draw_center_points(
            image,
            sack_points
        )


        # ====================================================
        # ENCODE
        # ====================================================

        image_base64 = encode_image(
            output
        )


    # ========================================================
    # HTML RESPONSE
    # ========================================================

    return render_template_string(

        HTML,

        image=image_base64,

        count=count,

        filename=filename,

        inference=round(
            inference_time,
            2
        ),

        conf=conf,

        iou=iou

    )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("SACK KEYPOINT TESTING UI STARTED")
    print("=" * 70)

    print(
        "Open in browser:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print("=" * 70)

    app.run(

        host="0.0.0.0",

        port=5011,

        debug=False

    )