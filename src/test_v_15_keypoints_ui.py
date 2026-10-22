from pathlib import Path
from flask import Flask, request, render_template_string, send_from_directory
from ultralytics import YOLO
import torch
import cv2
import uuid


# ============================================================
# V15 KEYPOINT SACK DETECTION - WEB UI
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT
    / "runs"
    / "pose"
    / "v_15_keypoints-3"
    / "weights"
    / "best.pt"
)

UPLOAD_DIR = ROOT / "runs" / "pose" / "v15_ui_uploads"
RESULT_DIR = ROOT / "runs" / "pose" / "v15_ui_results"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 1280
CONF = 0.25

app = Flask(__name__)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("        V15 KEYPOINT SACK DETECTION WEB UI")
print("=" * 70)

print("\nCUDA Available :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU            :", torch.cuda.get_device_name(0))
    print("CUDA Version   :", torch.version.cuda)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}"
    )

print("\nLoading model...")
model = YOLO(str(MODEL_PATH))
print("Model loaded successfully.")


# ============================================================
# HTML
# ============================================================

HTML = """
<!DOCTYPE html>
<html>

<head>

<title>V15 Sack Keypoint Detection</title>

<style>

body {
    font-family: Arial, sans-serif;
    background: #f4f4f4;
    margin: 0;
    padding: 30px;
    text-align: center;
}

h1 {
    margin-bottom: 25px;
}

.container {
    max-width: 1200px;
    margin: auto;
    background: white;
    padding: 30px;
    border-radius: 12px;
}

input[type=file] {
    margin: 15px;
}

button {
    padding: 12px 25px;
    font-size: 16px;
    cursor: pointer;
}

.count {
    font-size: 24px;
    font-weight: bold;
    margin: 25px;
}

.result {
    max-width: 100%;
    max-height: 650px;
}

.info {
    margin-top: 20px;
    font-size: 16px;
}

</style>

</head>

<body>

<div class="container">

<h1>V15 — Sack Keypoint Detection</h1>

<form action="/predict" method="post" enctype="multipart/form-data">

<input
    type="file"
    name="file"
    accept="image/*"
    required
>

<br>

<button type="submit">
Detect Sacks
</button>

</form>

{% if result_image %}

<div class="count">
Detected Sacks: {{ count }}
</div>

<img
    class="result"
    src="{{ url_for('result_file', filename=result_image) }}"
>

<div class="info">
Confidence Threshold: {{ conf }}
<br>
Image Size: {{ imgsz }}
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
        result_image=None,
        count=0,
        conf=CONF,
        imgsz=IMG_SIZE
    )


# ============================================================
# PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    file = request.files.get("file")

    if file is None or file.filename == "":
        return "No image selected."

    # --------------------------------------------------------
    # Save uploaded image
    # --------------------------------------------------------

    unique_id = uuid.uuid4().hex

    input_name = f"{unique_id}_{file.filename}"

    input_path = UPLOAD_DIR / input_name

    file.save(input_path)

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    frame = cv2.imread(str(input_path))

    if frame is None:
        return "Unable to read image."

    # --------------------------------------------------------
    # YOLO Prediction
    # --------------------------------------------------------

    results = model.predict(
        source=frame,
        imgsz=IMG_SIZE,
        conf=CONF,
        device=0,
        verbose=False
    )

    result = results[0]

    # --------------------------------------------------------
    # Count sacks
    # --------------------------------------------------------

    count = 0

    if result.boxes is not None:
        count = len(result.boxes)

    # --------------------------------------------------------
    # Draw YOLO results
    # --------------------------------------------------------

    annotated = result.plot(
        boxes=True,
        labels=True,
        conf=True,
        kpt_line=False
    )

    # --------------------------------------------------------
    # Draw keypoints manually
    # --------------------------------------------------------

    if result.keypoints is not None:

        keypoints = result.keypoints.xy

        for sack_points in keypoints:

            for point in sack_points:

                x = int(point[0])
                y = int(point[1])

                # Center keypoint
                cv2.circle(
                    annotated,
                    (x, y),
                    8,
                    (0, 255, 0),
                    -1
                )

                # Outer ring
                cv2.circle(
                    annotated,
                    (x, y),
                    12,
                    (255, 255, 255),
                    2
                )

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    output_name = f"{unique_id}_result.jpg"

    output_path = RESULT_DIR / output_name

    cv2.imwrite(
        str(output_path),
        annotated
    )

    return render_template_string(
        HTML,
        result_image=output_name,
        count=count,
        conf=CONF,
        imgsz=IMG_SIZE
    )


# ============================================================
# RESULT FILE
# ============================================================

@app.route("/results/<filename>")
def result_file(filename):

    return send_from_directory(
        RESULT_DIR,
        filename
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("WEB UI STARTED")
    print("=" * 70)

    print("\nOpen from your Windows/browser:")
    print("http://192.168.2.240:5001")

    print("\nServer listening on:")
    print("0.0.0.0:5001")

    print("=" * 70)

    app.run(
        host="0.0.0.0",
        port=5111,
        debug=False,
        threaded=True
    )