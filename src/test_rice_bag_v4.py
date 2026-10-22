"""
RICE BAG V4 - IMAGE TESTING UI

Features:
- Upload any image from browser
- Uses V4 best.pt
- YOLO detection
- No bounding boxes
- No confidence displayed
- Center point for every detected bag
- Bag number displayed
- Total bag count
- Total estimated weight
- Downloadable result image
"""

from pathlib import Path
from flask import Flask, request, render_template_string, send_file
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import os


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT
    / "outputs"
    / "rice_bag_detection_v4"
    / "weights"
    / "best.pt"
)

UPLOAD_DIR = ROOT / "test_uploads_v4"

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

# Confidence threshold
CONF_THRESHOLD = 0.25

# IoU threshold
IOU_THRESHOLD = 0.50

# Maximum detections
MAX_DETECTIONS = 1000

# Estimated weight of one rice bag
# CHANGE THIS if actual bag weight is different.
BAG_WEIGHT_KG = 50


# ============================================================
# CHECK MODEL
# ============================================================

print("=" * 70)
print("RICE BAG DETECTION V4 - TEST UI")
print("=" * 70)

print(f"Model path:")
print(MODEL_PATH)

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"\nV4 model not found!\n\n"
        f"Expected:\n{MODEL_PATH}\n\n"
        f"Make sure V4 training completed successfully."
    )


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading V4 model...")

model = YOLO(
    str(MODEL_PATH)
)

print("✓ V4 model loaded")


# ============================================================
# FLASK
# ============================================================

app = Flask(
    __name__
)


# ============================================================
# HTML
# ============================================================

HTML = """
<!DOCTYPE html>

<html>

<head>

<title>Rice Bag V4 Tester</title>

<style>

body {
    background: #111;
    color: white;
    font-family: Arial, sans-serif;
    text-align: center;
    padding: 30px;
}

.container {
    max-width: 1100px;
    margin: auto;
}

h1 {
    margin-bottom: 10px;
}

.subtitle {
    color: #aaa;
    margin-bottom: 30px;
}

.upload-box {
    border: 2px dashed #555;
    padding: 30px;
    border-radius: 15px;
    margin-bottom: 25px;
}

input[type=file] {
    margin: 15px;
}

button {
    background: #00c853;
    color: white;
    border: none;
    padding: 12px 25px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 16px;
}

button:hover {
    background: #00a844;
}

.stats {
    display: flex;
    justify-content: center;
    gap: 20px;
    margin: 25px 0;
    flex-wrap: wrap;
}

.card {
    background: #222;
    padding: 20px 35px;
    border-radius: 12px;
    min-width: 160px;
}

.card h2 {
    margin: 0;
    font-size: 30px;
}

.card p {
    color: #aaa;
    margin-bottom: 0;
}

.result {
    margin-top: 30px;
}

.result img {
    max-width: 100%;
    border-radius: 10px;
    border: 1px solid #444;
}

.download {
    display: inline-block;
    margin-top: 20px;
    padding: 12px 20px;
    background: #333;
    color: white;
    text-decoration: none;
    border-radius: 8px;
}

</style>

</head>


<body>

<div class="container">

<h1>🌾 Rice Bag Detection — V4</h1>

<div class="subtitle">
YOLO11m • Center Point Detection • No Bounding Boxes
</div>


<div class="upload-box">

<form
method="POST"
action="/predict"
enctype="multipart/form-data"
>

<input
type="file"
name="image"
accept="image/*"
required
>

<br>

<button type="submit">
Upload & Test Image
</button>

</form>

</div>


{% if result %}

<div class="stats">

<div class="card">

<h2>{{ total_bags }}</h2>

<p>Total Bags</p>

</div>


<div class="card">

<h2>{{ total_weight }} KG</h2>

<p>Estimated Weight</p>

</div>


<div class="card">

<h2>{{ confidence_count }}</h2>

<p>Detections</p>

</div>

</div>


<div class="result">

<h2>Detection Result</h2>

<img
src="/result/{{ result }}"
>


<br>

<a
class="download"
href="/result/{{ result }}"
download
>
Download Result
</a>

</div>

{% endif %}

</div>

</body>

</html>
"""


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template_string(
        HTML,
        result=None
    )


# ============================================================
# PREDICT
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    if "image" not in request.files:

        return "No image uploaded", 400


    file = request.files["image"]


    if file.filename == "":

        return "No image selected", 400


    # --------------------------------------------------------
    # Save uploaded image
    # --------------------------------------------------------

    input_path = (
        UPLOAD_DIR
        / "input.jpg"
    )

    file.save(
        str(input_path)
    )


    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    image = cv2.imread(
        str(input_path)
    )


    if image is None:

        return "Could not read image", 400


    # --------------------------------------------------------
    # YOLO INFERENCE
    # --------------------------------------------------------

    print("\nRunning V4 inference...")

    results = model.predict(

        source=image,

        imgsz=1280,

        conf=CONF_THRESHOLD,

        iou=IOU_THRESHOLD,

        max_det=MAX_DETECTIONS,

        verbose=False,

    )


    result = results[0]


    # --------------------------------------------------------
    # GET DETECTION CENTERS
    # --------------------------------------------------------

    centers = []


    if result.boxes is not None:

        boxes = result.boxes.xyxy.cpu().numpy()

        confidences = (
            result.boxes.conf
            .cpu()
            .numpy()
        )


        for box, conf in zip(
            boxes,
            confidences
        ):

            x1, y1, x2, y2 = box

            center_x = int(
                (x1 + x2) / 2
            )

            center_y = int(
                (y1 + y2) / 2
            )


            centers.append(
                (
                    center_x,
                    center_y,
                    float(conf)
                )
            )


    # --------------------------------------------------------
    # SORT CENTERS
    # --------------------------------------------------------

    # Sort top → bottom and left → right
    centers.sort(
        key=lambda p: (
            p[1],
            p[0]
        )
    )


    # --------------------------------------------------------
    # DRAW ONLY POINTS
    # --------------------------------------------------------

    output = image.copy()


    for index, (
        center_x,
        center_y,
        conf
    ) in enumerate(
        centers,
        start=1
    ):


        # ----------------------------------------------------
        # POINT
        # ----------------------------------------------------

        cv2.circle(

            output,

            (
                center_x,
                center_y
            ),

            5,

            (0, 255, 0),

            -1

        )


        # ----------------------------------------------------
        # BAG NUMBER
        # ----------------------------------------------------

        cv2.putText(

            output,

            str(index),

            (
                center_x + 7,
                center_y - 7
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.45,

            (0, 255, 0),

            1,

            cv2.LINE_AA

        )


    # --------------------------------------------------------
    # TOTALS
    # --------------------------------------------------------

    total_bags = len(
        centers
    )


    total_weight = (
        total_bags
        * BAG_WEIGHT_KG
    )


    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    cv2.rectangle(

        output,

        (10, 10),

        (360, 100),

        (0, 0, 0),

        -1

    )


    cv2.putText(
        

        output,

        f"TOTAL BAGS: {total_bags}",

        (20, 45),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        (255, 255, 255),

        2,

        cv2.LINE_AA

    )


    cv2.putText(

        output,

        f"TOTAL WEIGHT: {total_weight} KG",

        (20, 80),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (255, 255, 255),

        2,

        cv2.LINE_AA

    )


    # --------------------------------------------------------
    # SAVE OUTPUT
    # --------------------------------------------------------

    output_path = (
        UPLOAD_DIR
        / "result.jpg"
    )


    cv2.imwrite(

        str(output_path),

        output

    )


    print(
        f"Detected bags: {total_bags}"
    )

    print(
        f"Estimated weight: {total_weight} KG"
    )

    print(
        f"Result: {output_path}"
    )


    # --------------------------------------------------------
    # SHOW RESULT
    # --------------------------------------------------------

    return render_template_string(

        HTML,

        result="result.jpg",

        total_bags=total_bags,

        total_weight=total_weight,

        confidence_count=total_bags

    )


# ============================================================
# RESULT
# ============================================================

@app.route(
    "/result/<filename>"
)
def result_file(filename):

    path = (
        UPLOAD_DIR
        / filename
    )


    if not path.exists():

        return "Result not found", 404


    return send_file(
        str(path)
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print(
        "\n" + "=" * 70
    )

    print(
        "FLASK SERVER STARTED"
    )

    print(
        "=" * 70
    )

    print(
        "\nOpen in browser:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print(
        "\nFor network access:"
    )

    print(
        "http://0.0.0.0:5000"
    )

    print(
        "\nPress CTRL+C to stop."
    )


    app.run(

        host="0.0.0.0",

        port=5000,

        debug=False

    )