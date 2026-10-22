from pathlib import Path
from flask import Flask, request, render_template_string, send_file
from ultralytics import YOLO
import cv2
import uuid
import time


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT
    / "outputs"
    / "rice_bag_10_v6_detection"
    / "weights"
    / "best.pt"
)

UPLOAD_DIR = ROOT / "data" / "test_uploads_v6"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DETECTION SETTINGS
# ============================================================

# Increased from 0.25 → 0.50
# Removes many low-confidence false detections.
CONF_THRESHOLD = 0.50

# Lower IoU → stronger duplicate suppression
IOU_THRESHOLD = 0.30

IMAGE_SIZE = 1280

# Don't artificially allow thousands of detections.
MAX_DETECTIONS = 300

# Your current assumption:
# 1 rice bag = 50 kg
BAG_WEIGHT_KG = 50


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("RICE BAG 10 - V6 UNSEEN IMAGE TESTING")
print("=" * 70)

print(f"Model: {MODEL_PATH}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"""
V6 model not found:

{MODEL_PATH}

Train V6 first.
"""
    )

model = YOLO(str(MODEL_PATH))

print("✓ V6 model loaded")
print(f"Classes: {model.names}")


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


# ============================================================
# HTML
# ============================================================

HTML = """

<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<title>Rice Bag V6 Detection</title>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    background: #0f1115;
    color: white;
    font-family: Arial, sans-serif;
}

.container {
    max-width: 1250px;
    margin: auto;
    padding: 35px 20px;
}

.header {
    text-align: center;
    margin-bottom: 30px;
}

.header h1 {
    margin-bottom: 8px;
}

.header p {
    color: #9ca3af;
}

.upload-box {
    background: #181b21;
    border: 1px solid #2b3038;
    border-radius: 16px;
    padding: 30px;
    text-align: center;
}

input[type=file] {
    margin: 15px;
    padding: 10px;
}

button {
    background: #16a34a;
    color: white;
    border: none;
    padding: 13px 30px;
    border-radius: 8px;
    font-size: 16px;
    cursor: pointer;
}

button:hover {
    opacity: 0.85;
}

.stats {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 15px;
    margin: 25px 0;
}

.card {
    background: #181b21;
    border: 1px solid #2b3038;
    border-radius: 14px;
    padding: 22px;
    text-align: center;
}

.number {
    font-size: 32px;
    font-weight: bold;
    margin-bottom: 8px;
}

.label {
    color: #9ca3af;
}

.result {
    background: #181b21;
    border: 1px solid #2b3038;
    border-radius: 16px;
    padding: 20px;
    text-align: center;
}

.result img {
    max-width: 100%;
    height: auto;
    border-radius: 10px;
}

.error {
    margin-top: 20px;
    color: #ef4444;
    text-align: center;
}

.info {
    margin-top: 15px;
    color: #9ca3af;
    font-size: 13px;
}

@media(max-width: 700px) {

    .stats {
        grid-template-columns: 1fr;
    }

}

</style>

</head>


<body>

<div class="container">


<div class="header">

<h1>🌾 Rice Bag Detection — V6</h1>

<p>Upload an unseen image to test the trained model</p>

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
accept="image/jpeg,image/png,image/jpg"
required
>

<br>

<button type="submit">
Upload & Detect
</button>

</form>

<div class="info">

Confidence: 0.50 |
IoU: 0.30 |
Image Size: 1280

</div>

</div>


{% if error %}

<div class="error">

{{ error }}

</div>

{% endif %}


{% if result %}

<div class="stats">


<div class="card">

<div class="number">
{{ total_bags }}
</div>

<div class="label">
Total Bags Detected
</div>

</div>


<div class="card">

<div class="number">
{{ total_weight }} KG
</div>

<div class="label">
Estimated Total Weight
</div>

</div>


<div class="card">

<div class="number">
{{ processing_time }} s
</div>

<div class="label">
Inference Time
</div>

</div>


</div>


<div class="result">

<h2>Detection Result</h2>

<img src="/result/{{ result }}">

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
        result=None,
        error=None
    )


# ============================================================
# PREDICTION
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    if "image" not in request.files:

        return render_template_string(
            HTML,
            result=None,
            error="No image selected."
        )


    uploaded = request.files["image"]


    if uploaded.filename == "":

        return render_template_string(
            HTML,
            result=None,
            error="Please select an image."
        )


    # --------------------------------------------------------
    # SAVE INPUT
    # --------------------------------------------------------

    unique_id = uuid.uuid4().hex

    input_path = (
        UPLOAD_DIR
        / f"{unique_id}_input.jpg"
    )

    output_path = (
        UPLOAD_DIR
        / f"{unique_id}_result.jpg"
    )

    uploaded.save(str(input_path))


    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    image = cv2.imread(
        str(input_path)
    )


    if image is None:

        return render_template_string(
            HTML,
            result=None,
            error="Could not read image."
        )


    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    start = time.perf_counter()

    results = model.predict(

        source=image,

        imgsz=IMAGE_SIZE,

        conf=CONF_THRESHOLD,

        iou=IOU_THRESHOLD,

        max_det=MAX_DETECTIONS,

        device=0,

        verbose=False

    )

    elapsed = (
        time.perf_counter()
        - start
    )


    result = results[0]


    # --------------------------------------------------------
    # EXTRACT BOXES
    # --------------------------------------------------------

    detections = []


    if result.boxes is not None:

        boxes = (
            result.boxes.xyxy
            .cpu()
            .numpy()
        )

        confidences = (
            result.boxes.conf
            .cpu()
            .numpy()
        )


        for box, confidence in zip(
            boxes,
            confidences
        ):

            x1, y1, x2, y2 = box

            detections.append(
                {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "confidence": float(confidence)
                }
            )


    # --------------------------------------------------------
    # SORT DETECTIONS
    # Top → Bottom
    # Left → Right
    # --------------------------------------------------------

    detections.sort(
        key=lambda d: (
            d["y1"],
            d["x1"]
        )
    )


    # --------------------------------------------------------
    # DRAW DETECTIONS
    # --------------------------------------------------------

    output = image.copy()


    for index, detection in enumerate(
        detections,
        start=1
    ):

        x1 = int(detection["x1"])
        y1 = int(detection["y1"])

        x2 = int(detection["x2"])
        y2 = int(detection["y2"])


        # Bounding box

        cv2.rectangle(

            output,

            (x1, y1),

            (x2, y2),

            (0, 255, 0),

            2

        )


        # ----------------------------------------------------
        # BAG NUMBER ONLY
        # No confidence displayed
        # ----------------------------------------------------

        label = f"Bag {index}"


        # Background for text

        (tw, th), _ = cv2.getTextSize(

            label,

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            2

        )


        text_y = max(
            th + 5,
            y1 - 5
        )


        cv2.rectangle(

            output,

            (x1, text_y - th - 5),

            (x1 + tw + 6, text_y + 3),

            (0, 0, 0),

            -1

        )


        cv2.putText(

            output,

            label,

            (x1 + 3, text_y),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (255, 255, 255),

            2,

            cv2.LINE_AA

        )


    # --------------------------------------------------------
    # TOTALS
    # --------------------------------------------------------

    total_bags = len(detections)

    total_weight = (
        total_bags
        * BAG_WEIGHT_KG
    )


    # --------------------------------------------------------
    # TOP SUMMARY
    # --------------------------------------------------------

    header_height = 105

    cv2.rectangle(

        output,

        (0, 0),

        (500, header_height),

        (0, 0, 0),

        -1

    )


    cv2.putText(

        output,

        f"TOTAL BAGS: {total_bags}",

        (20, 42),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.9,

        (255, 255, 255),

        2,

        cv2.LINE_AA

    )


    cv2.putText(

        output,

        f"TOTAL WEIGHT: {total_weight} KG",

        (20, 82),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.72,

        (255, 255, 255),

        2,

        cv2.LINE_AA

    )


    # --------------------------------------------------------
    # SAVE RESULT
    # --------------------------------------------------------

    cv2.imwrite(
        str(output_path),
        output
    )


    print()
    print("=" * 50)
    print(f"Detected Bags : {total_bags}")
    print(f"Total Weight  : {total_weight} KG")
    print(f"Confidence    : {CONF_THRESHOLD}")
    print(f"IoU           : {IOU_THRESHOLD}")
    print(f"Inference     : {elapsed:.3f}s")
    print("=" * 50)


    # --------------------------------------------------------
    # RETURN UI
    # --------------------------------------------------------

    return render_template_string(

        HTML,

        result=output_path.name,

        total_bags=total_bags,

        total_weight=total_weight,

        processing_time=f"{elapsed:.3f}",

        error=None

    )


# ============================================================
# RESULT IMAGE
# ============================================================

@app.route(
    "/result/<filename>"
)
def result(filename):

    path = UPLOAD_DIR / filename


    if not path.exists():

        return "Result not found", 404


    return send_file(
        str(path)
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("V6 FLASK SERVER")
    print("=" * 70)

    print()
    print("Local:")
    print("http://127.0.0.1:5000")

    print()
    print("GPU machine:")
    print("http://192.168.2.240:5000")

    print()
    print("Upload an UNSEEN image from the browser.")
    print()

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )