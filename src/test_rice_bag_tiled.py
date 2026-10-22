from pathlib import Path
import cv2
import torch
import numpy as np

from flask import (
    Flask,
    request,
    render_template_string,
    send_file,
    redirect,
    url_for
)

from ultralytics import YOLO


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "outputs" / "tiled_predictions"
UPLOAD_DIR = ROOT / "test_uploads"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# MODEL
# ============================================================

# First preference = V3
V3_MODEL = (
    ROOT
    / "outputs"
    / "rice_bag_detection_v3"
    / "weights"
    / "best.pt"
)

# Fallback = V2
V2_MODEL = (
    ROOT
    / "outputs"
    / "rice_bag_detection_v2"
    / "weights"
    / "best.pt"
)


if V3_MODEL.exists():

    MODEL_PATH = V3_MODEL

elif V2_MODEL.exists():

    MODEL_PATH = V2_MODEL

else:

    raise FileNotFoundError(
        "\nNo trained model found.\n"
        "Expected either:\n"
        f"{V3_MODEL}\n"
        f"{V2_MODEL}\n"
    )


# ============================================================
# HARDWARE
# ============================================================

DEVICE = (
    0
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# DETECTION SETTINGS
# ============================================================

CONFIDENCE = 0.50

IOU = 0.45

TILE_SIZE = 1280

OVERLAP = 0.25

MAX_DET_PER_TILE = 150


# ============================================================
# BAG WEIGHT
# ============================================================

# Change this if actual bag weight is different.
BAG_WEIGHT_KG = 50


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("RICE BAG TILED TESTING UI")
print("=" * 70)

print(f"Model : {MODEL_PATH}")
print(f"Device: {DEVICE}")

if torch.cuda.is_available():

    print(
        "GPU   :",
        torch.cuda.get_device_name(0)
    )

print("=" * 70)

model = YOLO(
    str(MODEL_PATH)
)


# ============================================================
# HTML
# ============================================================

HTML = """
<!DOCTYPE html>

<html>

<head>

    <title>Rice Bag Detection</title>

    <meta name="viewport"
          content="width=device-width, initial-scale=1">

    <style>

        body {
            margin: 0;
            padding: 40px;
            background: #111;
            color: white;
            font-family: Arial, sans-serif;
            text-align: center;
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
            padding: 35px;
            border-radius: 15px;
            margin-bottom: 30px;
        }

        input[type=file] {
            margin: 20px;
            padding: 10px;
        }

        button {
            padding: 12px 30px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
        }

        .result {
            margin-top: 30px;
        }

        .stats {
            display: flex;
            justify-content: center;
            gap: 20px;
            flex-wrap: wrap;
            margin: 25px 0;
        }

        .card {
            background: #222;
            border-radius: 12px;
            padding: 20px 35px;
            min-width: 180px;
        }

        .number {
            font-size: 32px;
            font-weight: bold;
        }

        .label {
            color: #aaa;
            margin-top: 5px;
        }

        img {
            max-width: 100%;
            border-radius: 12px;
            margin-top: 20px;
        }

        .model {
            color: #888;
            font-size: 13px;
            margin-top: 20px;
        }

    </style>

</head>


<body>

<div class="container">

    <h1>Rice Bag Detection</h1>

    <div class="subtitle">
        Upload a warehouse image and detect individual rice bags
    </div>


    <div class="upload-box">

        <form
            action="/predict"
            method="POST"
            enctype="multipart/form-data"
        >

            <input
                type="file"
                name="image"
                accept=".jpg,.jpeg,.png"
                required
            >

            <br>

            <button type="submit">
                Detect Rice Bags
            </button>

        </form>

    </div>


    {% if result %}

    <div class="result">

        <div class="stats">

            <div class="card">

                <div class="number">
                    {{ count }}
                </div>

                <div class="label">
                    TOTAL BAGS
                </div>

            </div>


            <div class="card">

                <div class="number">
                    {{ weight }} KG
                </div>

                <div class="label">
                    TOTAL WEIGHT
                </div>

            </div>

        </div>


        <h2>Detection Result</h2>

        <img
            src="{{ url_for('result_image') }}"
        >


        <div class="model">

            Model:
            {{ model_name }}

            <br>

            Detection:
            Tiled inference + NMS

        </div>

    </div>

    {% endif %}

</div>

</body>

</html>
"""


# ============================================================
# GENERATE TILES
# ============================================================

def generate_tiles(
    image,
    tile_size,
    overlap
):

    height, width = image.shape[:2]

    step = int(
        tile_size * (1 - overlap)
    )

    y_positions = list(
        range(
            0,
            max(height - tile_size, 0) + 1,
            step
        )
    )

    x_positions = list(
        range(
            0,
            max(width - tile_size, 0) + 1,
            step
        )
    )


    if not y_positions:

        y_positions = [0]

    if not x_positions:

        x_positions = [0]


    last_y = max(
        height - tile_size,
        0
    )

    last_x = max(
        width - tile_size,
        0
    )


    if y_positions[-1] != last_y:

        y_positions.append(last_y)


    if x_positions[-1] != last_x:

        x_positions.append(last_x)


    tiles = []


    for y in y_positions:

        for x in x_positions:

            x2 = min(
                x + tile_size,
                width
            )

            y2 = min(
                y + tile_size,
                height
            )

            tile = image[
                y:y2,
                x:x2
            ]

            tiles.append(
                (
                    tile,
                    x,
                    y
                )
            )


    return tiles


# ============================================================
# DETECT IMAGE
# ============================================================

def detect_image(image_path):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        raise RuntimeError(
            "Could not read uploaded image."
        )


    all_boxes = []

    all_scores = []


    # --------------------------------------------------------
    # CREATE TILES
    # --------------------------------------------------------

    tiles = generate_tiles(
        image,
        TILE_SIZE,
        OVERLAP
    )


    print()
    print(
        f"Image size: "
        f"{image.shape[1]} x {image.shape[0]}"
    )

    print(
        f"Number of tiles: {len(tiles)}"
    )


    # --------------------------------------------------------
    # RUN YOLO ON EVERY TILE
    # --------------------------------------------------------

    for tile_number, (
        tile,
        offset_x,
        offset_y
    ) in enumerate(tiles):

        print(
            f"Tile "
            f"{tile_number + 1}/"
            f"{len(tiles)}"
        )


        results = model.predict(

            source=tile,

            imgsz=TILE_SIZE,

            conf=CONFIDENCE,

            iou=IOU,

            max_det=MAX_DET_PER_TILE,

            device=DEVICE,

            verbose=False
        )


        result = results[0]


        if result.boxes is None:

            continue


        boxes = (
            result
            .boxes
            .xyxy
            .cpu()
            .numpy()
        )


        scores = (
            result
            .boxes
            .conf
            .cpu()
            .numpy()
        )


        # ----------------------------------------------------
        # MOVE BOXES BACK TO ORIGINAL IMAGE
        # ----------------------------------------------------

        for box, score in zip(
            boxes,
            scores
        ):

            x1, y1, x2, y2 = box


            x1 += offset_x
            x2 += offset_x

            y1 += offset_y
            y2 += offset_y


            all_boxes.append(
                [
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2)
                ]
            )


            all_scores.append(
                float(score)
            )


    # --------------------------------------------------------
    # NO DETECTIONS
    # --------------------------------------------------------

    if not all_boxes:

        return image, 0


    # --------------------------------------------------------
    # NMS
    # --------------------------------------------------------

    nms_boxes = []


    for box in all_boxes:

        x1, y1, x2, y2 = box

        nms_boxes.append(

            [
                float(x1),
                float(y1),
                float(x2 - x1),
                float(y2 - y1)
            ]

        )


    indices = cv2.dnn.NMSBoxes(

        nms_boxes,

        all_scores,

        CONFIDENCE,

        IOU

    )


    if len(indices) == 0:

        return image, 0


    indices = np.array(
        indices
    ).reshape(-1)


    # --------------------------------------------------------
    # DRAW ONLY CENTER POINTS
    # --------------------------------------------------------

    output = image.copy()


    final_count = 0


    for idx in indices:

        x1, y1, x2, y2 = (
            all_boxes[idx]
        )


        # Center of detected bag

        cx = int(
            (x1 + x2) / 2
        )

        cy = int(
            (y1 + y2) / 2
        )


        final_count += 1


        # ----------------------------------------------------
        # GREEN CENTER POINT
        # ----------------------------------------------------

        cv2.circle(

            output,

            (cx, cy),

            5,

            (0, 255, 0),

            -1

        )


        # ----------------------------------------------------
        # BAG NUMBER
        # ----------------------------------------------------

        cv2.putText(

            output,

            str(final_count),

            (
                cx + 7,
                cy - 7
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.5,

            (0, 255, 0),

            2,

            cv2.LINE_AA

        )


    # --------------------------------------------------------
    # TOTAL WEIGHT
    # --------------------------------------------------------

    total_weight = (
        final_count
        * BAG_WEIGHT_KG
    )


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    cv2.rectangle(

        output,

        (10, 10),

        (430, 100),

        (0, 0, 0),

        -1

    )


    cv2.putText(

        output,

        f"TOTAL BAGS: {final_count}",

        (25, 48),

        cv2.FONT_HERSHEY_SIMPLEX,

        1,

        (255, 255, 255),

        2,

        cv2.LINE_AA

    )


    cv2.putText(

        output,

        f"TOTAL WEIGHT: {total_weight} KG",

        (25, 85),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.85,

        (255, 255, 255),

        2,

        cv2.LINE_AA

    )


    return output, final_count


# ============================================================
# HOME
# ============================================================

@app.route(
    "/",
    methods=["GET"]
)
def index():

    return render_template_string(
        HTML,
        result=False
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

        return redirect(
            url_for("index")
        )


    file = request.files["image"]


    if file.filename == "":

        return redirect(
            url_for("index")
        )


    # --------------------------------------------------------
    # SAVE UPLOAD
    # --------------------------------------------------------

    filename = (
        "uploaded_image"
        + Path(file.filename).suffix
    )


    input_path = (
        UPLOAD_DIR
        / filename
    )


    file.save(
        str(input_path)
    )


    # --------------------------------------------------------
    # DETECT
    # --------------------------------------------------------

    output, count = detect_image(
        input_path
    )


    # --------------------------------------------------------
    # SAVE RESULT
    # --------------------------------------------------------

    output_path = (
        OUTPUT_DIR
        / "latest_result.jpg"
    )


    cv2.imwrite(
        str(output_path),
        output
    )


    weight = (
        count
        * BAG_WEIGHT_KG
    )


    return render_template_string(

        HTML,

        result=True,

        count=count,

        weight=weight,

        model_name=MODEL_PATH.name

    )


# ============================================================
# RESULT IMAGE
# ============================================================

@app.route(
    "/result"
)
def result_image():

    result_path = (
        OUTPUT_DIR
        / "latest_result.jpg"
    )


    return send_file(
        str(result_path)
    )


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("SERVER STARTED")
    print("=" * 70)

    print(
        "Open in browser:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print(
        "or"
    )

    print(
        "http://<GPU-IP>:5000"
    )

    print("=" * 70)


    app.run(

        host="0.0.0.0",

        port=5000,

        debug=False

    )