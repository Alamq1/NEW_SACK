from flask import Flask, request, render_template_string, send_file
from ultralytics import YOLO

from pathlib import Path
from werkzeug.utils import secure_filename

import cv2
import torch
import uuid


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "rice_bag_segmentation"
    / "weights"
    / "best.pt"
)

UPLOAD_DIR = PROJECT_ROOT / "data" / "test_uploads"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ui_predictions"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIG
# ============================================================

BAG_WEIGHT_KG = 50

DEVICE = 0 if torch.cuda.is_available() else "cpu"


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("LOADING RICE BAG MODEL")
print("=" * 60)
print(f"Model : {MODEL_PATH}")
print(f"Device: {DEVICE}")

model = YOLO(str(MODEL_PATH))

print("MODEL LOADED SUCCESSFULLY")
print("=" * 60)


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# HTML UI
# ============================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>

<head>
    <title>Rice Bag Inventory Tester</title>

    <style>

        body {
            font-family: Arial, sans-serif;
            background: #111;
            color: white;
            margin: 0;
            padding: 30px;
        }

        h1 {
            text-align: center;
        }

        .container {
            max-width: 1200px;
            margin: auto;
        }

        .panel {
            background: #1d1d1d;
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 25px;
        }

        input[type=file] {
            margin: 15px 0;
        }

        input[type=range] {
            width: 300px;
        }

        button {
            padding: 12px 25px;
            font-size: 16px;
            cursor: pointer;
        }

        .results {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }

        .image-box {
            flex: 1;
            min-width: 400px;
        }

        img {
            width: 100%;
            border-radius: 10px;
        }

        .stats {
            background: #222;
            padding: 20px;
            border-radius: 10px;
            font-size: 20px;
            line-height: 1.8;
            margin-bottom: 20px;
        }

    </style>

</head>


<body>

<div class="container">

    <h1>Rice Bag Inventory Tester</h1>


    <div class="panel">

        <form method="POST"
              enctype="multipart/form-data">

            <h3>Upload Image</h3>

            <input
                type="file"
                name="image"
                accept="image/*"
                required
            >

            <br><br>


            <h3>
                Confidence:
                <span id="conf_value">0.10</span>
            </h3>

            <input
                type="range"
                name="confidence"
                min="0.05"
                max="0.90"
                step="0.05"
                value="0.10"
                oninput="conf_value.innerText=this.value"
            >


            <h3>
                IoU:
                <span id="iou_value">0.70</span>
            </h3>

            <input
                type="range"
                name="iou"
                min="0.10"
                max="0.90"
                step="0.05"
                value="0.70"
                oninput="iou_value.innerText=this.value"
            >

            <br><br>

            <button type="submit">
                TEST IMAGE
            </button>

        </form>

    </div>


    {% if result_image %}

    <div class="stats">

        <b>Total Bags Detected:</b>
        {{ bag_count }}

        <br>

        <b>Weight Per Bag:</b>
        {{ bag_weight }} KG

        <br>

        <b>Total Weight:</b>
        {{ total_weight }} KG

        <br>

        <b>Confidence Used:</b>
        {{ confidence }}

        <br>

        <b>IoU Used:</b>
        {{ iou }}

    </div>


    <div class="results">

        <div class="image-box">

            <h2>Original Image</h2>

            <img src="/original/{{ image_id }}">

        </div>


        <div class="image-box">

            <h2>Detection Result</h2>

            <img src="/result/{{ image_id }}">

        </div>

    </div>

    {% endif %}

</div>

</body>

</html>
"""


# ============================================================
# IMAGE ROUTES
# ============================================================

@app.route("/original/<image_id>")
def get_original(image_id):

    image_path = UPLOAD_DIR / f"{image_id}.jpg"

    return send_file(
        image_path,
        mimetype="image/jpeg"
    )


@app.route("/result/<image_id>")
def get_result(image_id):

    image_path = OUTPUT_DIR / f"{image_id}.jpg"

    return send_file(
        image_path,
        mimetype="image/jpeg"
    )


# ============================================================
# MAIN ROUTE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":

        # ----------------------------------------------------
        # GET UPLOADED IMAGE
        # ----------------------------------------------------

        uploaded_file = request.files["image"]

        image_id = str(uuid.uuid4())

        input_path = UPLOAD_DIR / f"{image_id}.jpg"
        output_path = OUTPUT_DIR / f"{image_id}.jpg"

        uploaded_file.save(input_path)


        # ----------------------------------------------------
        # GET UI SETTINGS
        # ----------------------------------------------------

        confidence = float(
            request.form.get("confidence", 0.10)
        )

        iou = float(
            request.form.get("iou", 0.70)
        )


        # ----------------------------------------------------
        # LOAD IMAGE
        # ----------------------------------------------------

        image = cv2.imread(str(input_path))

        if image is None:

            return "ERROR: Could not read uploaded image."


        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        results = model.predict(

            source=image,

            conf=confidence,

            iou=iou,

            imgsz=960,

            device=DEVICE,

            verbose=False
        )


        result = results[0]

        output_image = image.copy()


        # ----------------------------------------------------
        # GET DETECTIONS
        # ----------------------------------------------------

        bag_count = 0

        if result.boxes is not None:

            boxes = (
                result.boxes
                .xyxy
                .cpu()
                .numpy()
            )


            # --------------------------------------------
            # SORT DETECTIONS
            #
            # TOP TO BOTTOM
            # LEFT TO RIGHT
            # --------------------------------------------

            boxes = sorted(

                boxes,

                key=lambda box: (
                    (box[1] + box[3]) / 2,
                    (box[0] + box[2]) / 2
                )

            )


            # --------------------------------------------
            # DRAW POINTS
            # --------------------------------------------

            for box in boxes:

                x1, y1, x2, y2 = map(
                    int,
                    box
                )


                # Center point

                center_x = (
                    x1 + x2
                ) // 2

                center_y = (
                    y1 + y2
                ) // 2


                bag_count += 1


                # ----------------------------------------
                # CENTER POINT
                # ----------------------------------------

                cv2.circle(

                    output_image,

                    (
                        center_x,
                        center_y
                    ),

                    7,

                    (0, 255, 0),

                    -1
                )


                # ----------------------------------------
                # BAG NUMBER
                # ----------------------------------------

                label = str(
                    bag_count
                )


                cv2.putText(

                    output_image,

                    label,

                    (
                        center_x + 10,
                        center_y - 10
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.7,

                    (0, 255, 0),

                    2
                )


        # ----------------------------------------------------
        # CALCULATE WEIGHT
        # ----------------------------------------------------

        total_weight = (
            bag_count *
            BAG_WEIGHT_KG
        )


        # ----------------------------------------------------
        # ADD SUMMARY
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # SAVE OUTPUT
        # ----------------------------------------------------

        cv2.imwrite(
            str(output_path),
            output_image
        )


        # ----------------------------------------------------
        # RENDER RESULT
        # ----------------------------------------------------

        return render_template_string(

            HTML_TEMPLATE,

            result_image=True,

            image_id=image_id,

            bag_count=bag_count,

            bag_weight=BAG_WEIGHT_KG,

            total_weight=total_weight,

            confidence=confidence,

            iou=iou
        )


    # ========================================================
    # FIRST PAGE
    # ========================================================

    return render_template_string(
        HTML_TEMPLATE,
        result_image=False
    )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=50002,

        debug=False
    )