from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, request, render_template_string, send_from_directory
from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "segment"
    / "rice_bags_v16_yolo26l_seg_clean"
    / "weights"
    / "best.pt"
)

UPLOAD_DIR = PROJECT_ROOT / "outputs" / "v16_clean_ui" / "uploads"
RESULT_DIR = PROJECT_ROOT / "outputs" / "v16_clean_ui" / "results"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

CONF = 0.25
IMGSZ = 1280
DEVICE = 0

HOST = "0.0.0.0"
PORT = 51112


# ============================================================
# HARDWARE
# ============================================================

print("=" * 80)
print("       V16 CLEAN YOLO26L SEGMENTATION TEST UI")
print("=" * 80)

print(f"Model : {MODEL_PATH}")
print(f"CUDA  : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU   : {torch.cuda.get_device_name(0)}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

model = YOLO(str(MODEL_PATH))

print(f"Task  : {model.task}")
print(f"Names : {model.names}")

print("=" * 80)


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

<title>V16 Rice Bag Segmentation</title>

<style>

body {
    font-family: Arial, sans-serif;
    background: #f2f2f2;
    margin: 0;
    padding: 30px;
}

.container {
    max-width: 1200px;
    margin: auto;
    background: white;
    padding: 30px;
    border-radius: 12px;
}

h1 {
    text-align: center;
}

.upload {
    text-align: center;
    padding: 25px;
}

button {
    padding: 12px 30px;
    font-size: 16px;
    cursor: pointer;
}

.count {
    text-align: center;
    font-size: 30px;
    font-weight: bold;
    margin: 25px;
}

.image-container {
    text-align: center;
}

.result-image {
    max-width: 100%;
    height: auto;
    border-radius: 8px;
}

.info {
    text-align: center;
    margin-top: 15px;
    color: #555;
}

</style>

</head>


<body>

<div class="container">

<h1>Rice Bag Segmentation — V16 Clean</h1>

<div class="upload">

<form method="POST" enctype="multipart/form-data">

<input
    type="file"
    name="image"
    accept="image/*"
    required
>

<br><br>

<button type="submit">
    Run Segmentation
</button>

</form>

</div>


{% if result %}

<div class="count">
    Rice Bags Detected: {{ count }}
</div>

<div class="image-container">

<img
    class="result-image"
    src="{{ url_for('result_file', filename=result) }}"
>

</div>

<div class="info">
    YOLO26L-Seg | Confidence: {{ conf }} | Image Size: {{ imgsz }}
</div>

{% endif %}

</div>

</body>

</html>
"""


# ============================================================
# INFERENCE
# ============================================================

def run_inference(input_path, output_path):

    print("\n" + "=" * 80)
    print("NEW IMAGE")
    print("=" * 80)

    print(f"Input : {input_path}")
    print(f"CONF  : {CONF}")
    print(f"IMGSZ : {IMGSZ}")

    image = cv2.imread(str(input_path))

    if image is None:
        raise RuntimeError(
            f"Cannot read image:\n{input_path}"
        )

    h, w = image.shape[:2]

    print(f"Size  : {w} x {h}")

    # --------------------------------------------------------
    # SAME INFERENCE AS CLI
    # --------------------------------------------------------

    results = model.predict(
        source=str(input_path),
        conf=CONF,
        imgsz=IMGSZ,
        device=DEVICE,
        verbose=True,
    )

    result = results[0]

    # --------------------------------------------------------
    # DEBUG: BOXES
    # --------------------------------------------------------

    if result.boxes is not None:

        box_count = len(result.boxes)

        print(f"BOX DETECTIONS : {box_count}")

        if box_count > 0:

            confs = (
                result.boxes.conf
                .detach()
                .cpu()
                .numpy()
            )

            print(
                "CONFIDENCES    :",
                np.round(confs, 3)
            )

    else:

        box_count = 0

        print("BOX DETECTIONS : 0")

    # --------------------------------------------------------
    # DEBUG: MASKS
    # --------------------------------------------------------

    if result.masks is not None:

        mask_count = len(result.masks)

        print(f"MASK DETECTIONS : {mask_count}")

    else:

        mask_count = 0

        print("MASK DETECTIONS : 0")

    # --------------------------------------------------------
    # IMPORTANT:
    # USE BOX COUNT AS DETECTION COUNT
    # --------------------------------------------------------

    count = box_count

    # --------------------------------------------------------
    # DRAW MASKS
    # --------------------------------------------------------

    if result.masks is not None:

        masks = result.masks.data.detach().cpu().numpy()

        for mask in masks:

            binary = (
                mask > 0.5
            ).astype(np.uint8)

            binary = cv2.resize(
                binary,
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            )

            # ------------------------------------------------
            # CONTOURS
            # ------------------------------------------------

            contours, _ = cv2.findContours(
                binary,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            if contours:

                cv2.drawContours(
                    image,
                    contours,
                    -1,
                    (0, 255, 0),
                    2,
                )

            # ------------------------------------------------
            # CENTER DOT
            # ------------------------------------------------

            ys, xs = np.where(binary > 0)

            if len(xs) > 0:

                cx = int(np.mean(xs))
                cy = int(np.mean(ys))

                cv2.circle(
                    image,
                    (cx, cy),
                    6,
                    (0, 0, 255),
                    -1,
                )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    if not cv2.imwrite(
        str(output_path),
        image
    ):
        raise RuntimeError(
            f"Could not save:\n{output_path}"
        )

    print(f"FINAL COUNT    : {count}")
    print(f"Saved          : {output_path}")
    print("=" * 80)

    return count


# ============================================================
# HOME
# ============================================================

@app.route("/", methods=["GET", "POST"])
def index():

    result_filename = None
    count = 0

    if request.method == "POST":

        file = request.files.get("image")

        if file is None or file.filename == "":
            return render_template_string(
                HTML,
                result=None,
                count=0,
                conf=CONF,
                imgsz=IMGSZ,
            )

        filename = Path(file.filename).name

        input_path = (
            UPLOAD_DIR / filename
        )

        file.save(str(input_path))

        output_filename = (
            f"result_{input_path.stem}.jpg"
        )

        output_path = (
            RESULT_DIR / output_filename
        )

        try:

            count = run_inference(
                input_path,
                output_path,
            )

            result_filename = output_filename

        except Exception as e:

            print("\nERROR:")
            print(e)

            return (
                f"""
                <h2>Inference Error</h2>
                <pre>{e}</pre>
                """,
                500,
            )

    return render_template_string(
        HTML,
        result=result_filename,
        count=count,
        conf=CONF,
        imgsz=IMGSZ,
    )


# ============================================================
# RESULT FILE
# ============================================================

@app.route("/results/<filename>")
def result_file(filename):

    return send_from_directory(
        RESULT_DIR,
        filename,
    )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 80)
    print("SERVER STARTED")
    print("=" * 80)

    print(
        f"\nOpen:"
        f"\nhttp://192.168.1.240:{PORT}"
    )

    print("\nPress CTRL+C to stop.")

    app.run(
        host=HOST,
        port=PORT,
        debug=False,
        threaded=True,
    )
