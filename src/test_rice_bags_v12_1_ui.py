from pathlib import Path
import uuid

import cv2
import numpy as np

from flask import Flask, request, render_template_string, send_file
from ultralytics import YOLO


# ============================================================
# YOLO26l-Seg V12.1
# RICE BAG SEGMENTATION TESTING UI
# ============================================================

print("=" * 75)
print("        RICE BAG YOLO26L-SEG V12.1 TESTING UI")
print("=" * 75)


# ============================================================
# PROJECT ROOT
# ============================================================

# Current file:
# /home/aiteam/NEW_SACK_COUNT/src/test_rice_bags_v12_1_ui.py
#
# parents[0] = /home/aiteam/NEW_SACK_COUNT/src
# parents[1] = /home/aiteam/NEW_SACK_COUNT   <-- PROJECT ROOT

BASE_DIR = Path(__file__).resolve().parents[1]


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = (
    BASE_DIR
    / "runs"
    / "segment"
    / "runs"
    / "segment"
    / "rice_bags_v12_yolo26l_seg"
    / "weights"
    / "best.pt"
).resolve()


# ============================================================
# OUTPUT PATH
# ============================================================

OUTPUT_DIR = (
    BASE_DIR
    / "runs"
    / "testing"
    / "v12_1_outputs"
).resolve()

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIG
# ============================================================

PORT = 5012

# YOLO classes:
# 0 = Rice-bag-10
# 1 = c-belt
# 2 = other
# 3 = pile

RICE_BAG_CLASS_ID = 0

CONF_THRESHOLD = 0.25


# ============================================================
# PATH VERIFICATION
# ============================================================

print(f"Project root : {BASE_DIR}")
print(f"Model path   : {MODEL_PATH}")
print(f"Output path  : {OUTPUT_DIR}")
print(f"Port         : {PORT}")
print(f"Confidence   : {CONF_THRESHOLD}")
print()


if not MODEL_PATH.exists():

    print("ERROR: best.pt NOT FOUND")
    print()
    print("Expected:")
    print(MODEL_PATH)
    print()

    raise FileNotFoundError(
        f"YOLO26l-Seg best.pt not found:\n{MODEL_PATH}"
    )


print("best.pt found successfully.")


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("Loading YOLO26l-Seg...")

model = YOLO(str(MODEL_PATH))

print("Model loaded successfully.")
print(f"Model classes : {model.names}")
print()

print(
    f"Target class  : "
    f"{RICE_BAG_CLASS_ID} -> "
    f"{model.names[RICE_BAG_CLASS_ID]}"
)

print("=" * 75)


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# HTML
# ============================================================

HTML = """
<!DOCTYPE html>

<html>

<head>

    <title>Rice Bag Segmentation V12.1</title>

    <meta name="viewport"
          content="width=device-width, initial-scale=1.0">

    <style>

        body {
            font-family: Arial, sans-serif;
            background: #f4f4f4;
            margin: 0;
            padding: 30px;
            text-align: center;
        }

        h1 {
            margin-bottom: 5px;
        }

        .subtitle {
            color: #666;
            margin-bottom: 25px;
        }

        .upload-box {
            background: white;
            padding: 25px;
            border-radius: 12px;
            max-width: 750px;
            margin: auto;
            box-shadow: 0 2px 12px rgba(0,0,0,0.12);
        }

        input[type=file] {
            margin: 15px;
        }

        button {
            padding: 11px 28px;
            font-size: 16px;
            cursor: pointer;
            border-radius: 7px;
            border: none;
            background: #222;
            color: white;
        }

        button:hover {
            opacity: 0.85;
        }

        .result {
            margin-top: 30px;
        }

        .count {
            font-size: 30px;
            font-weight: bold;
            margin: 20px;
        }

        .info {
            margin: 15px auto 25px auto;
            max-width: 750px;
            background: white;
            padding: 18px;
            border-radius: 10px;
            line-height: 1.7;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }

        .image-container {
            background: white;
            padding: 15px;
            max-width: 1400px;
            margin: auto;
            border-radius: 12px;
        }

        img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
        }

        .error {
            margin: 20px auto;
            max-width: 700px;
            padding: 15px;
            background: #ffe5e5;
            color: #990000;
            border-radius: 8px;
        }

    </style>

</head>


<body>

    <h1>Rice Bag Segmentation - V12.1</h1>

    <div class="subtitle">
        YOLO26l-Seg | Rice-bag-10 Only
    </div>


    <div class="upload-box">

        <form
            method="POST"
            enctype="multipart/form-data"
        >

            <input
                type="file"
                name="image"
                accept=".jpg,.jpeg,.png,.webp"
                required
            >

            <br>

            <button type="submit">
                Test Image
            </button>

        </form>

    </div>


    {% if error %}

        <div class="error">
            {{ error }}
        </div>

    {% endif %}


    {% if result_image %}

        <div class="result">

            <div class="count">
                Total Rice Bags Detected: {{ count }}
            </div>


            <div class="info">

                <b>Model:</b>
                YOLO26l-Seg
                <br>

                <b>Target Class:</b>
                Rice-bag-10
                <br>

                <b>Confidence:</b>
                {{ confidence }}

            </div>


            <div class="image-container">

                <img
                    src="/outputs/{{ result_image }}"
                    alt="Rice Bag Segmentation Result"
                >

            </div>

        </div>

    {% endif %}

</body>

</html>
"""


# ============================================================
# SEGMENT RICE BAGS
# ============================================================

def segment_rice_bags(image):

    """
    YOLO26l-Seg inference.

    Only Rice-bag-10 is counted.

    Display:
        - segmentation mask
        - mask boundary
        - center point
        - bag number

    Bounding boxes are NOT displayed.
    """

    results = model.predict(
        source=image,
        conf=CONF_THRESHOLD,
        verbose=False,
        retina_masks=True
    )

    result = results[0]

    output = image.copy()

    bag_count = 0


    # ========================================================
    # NO DETECTIONS
    # ========================================================

    if (
        result.boxes is None
        or result.masks is None
        or len(result.boxes) == 0
    ):

        return output, 0


    # ========================================================
    # GET RESULTS
    # ========================================================

    masks = result.masks.data.cpu().numpy()

    classes = result.boxes.cls.cpu().numpy()

    confidences = result.boxes.conf.cpu().numpy()


    height, width = image.shape[:2]


    # ========================================================
    # PROCESS EACH INSTANCE
    # ========================================================

    for mask, cls_id, confidence in zip(
        masks,
        classes,
        confidences
    ):

        # ----------------------------------------------------
        # ONLY RICE BAG
        # ----------------------------------------------------

        if int(cls_id) != RICE_BAG_CLASS_ID:
            continue


        # ----------------------------------------------------
        # RESIZE MASK
        # ----------------------------------------------------

        mask = cv2.resize(
            mask.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST
        )


        mask_binary = mask > 0


        if not np.any(mask_binary):
            continue


        # ----------------------------------------------------
        # COUNT
        # ----------------------------------------------------

        bag_count += 1


        # ----------------------------------------------------
        # MASK OVERLAY
        # ----------------------------------------------------

        overlay = output.copy()

        overlay[mask_binary] = (
            0,
            255,
            0
        )

        output = cv2.addWeighted(
            output,
            0.65,
            overlay,
            0.35,
            0
        )


        # ----------------------------------------------------
        # MASK CONTOUR
        # ----------------------------------------------------

        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )


        if not contours:
            continue


        contour = max(
            contours,
            key=cv2.contourArea
        )


        # ----------------------------------------------------
        # MASK BOUNDARY
        # ----------------------------------------------------

        cv2.polylines(
            output,
            [contour],
            True,
            (0, 255, 0),
            2
        )


        # ----------------------------------------------------
        # CENTER POINT
        # ----------------------------------------------------

        moments = cv2.moments(contour)


        if moments["m00"] != 0:

            cx = int(
                moments["m10"] /
                moments["m00"]
            )

            cy = int(
                moments["m01"] /
                moments["m00"]
            )

        else:

            x, y, bw, bh = cv2.boundingRect(
                contour
            )

            cx = x + bw // 2
            cy = y + bh // 2


        # ----------------------------------------------------
        # CENTER POINT
        # ----------------------------------------------------

        cv2.circle(
            output,
            (cx, cy),
            6,
            (0, 0, 255),
            -1
        )


        # ----------------------------------------------------
        # BAG NUMBER
        # ----------------------------------------------------

        label = f"Bag {bag_count}"


        cv2.putText(
            output,
            label,
            (cx + 8, cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            3,
            cv2.LINE_AA
        )

        cv2.putText(
            output,
            label,
            (cx + 8, cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )


    return output, bag_count


# ============================================================
# HOME / IMAGE UPLOAD
# ============================================================

@app.route("/", methods=["GET", "POST"])
def index():

    result_image = None

    count = 0

    error = None


    # ========================================================
    # POST
    # ========================================================

    if request.method == "POST":

        file = request.files.get("image")


        if file is None or file.filename == "":

            error = "Please select an image."

            return render_template_string(
                HTML,
                result_image=None,
                count=0,
                confidence=CONF_THRESHOLD,
                error=error
            )


        try:

            # ------------------------------------------------
            # READ IMAGE
            # ------------------------------------------------

            file_bytes = file.read()

            image_array = np.frombuffer(
                file_bytes,
                dtype=np.uint8
            )

            image = cv2.imdecode(
                image_array,
                cv2.IMREAD_COLOR
            )


            if image is None:

                raise ValueError(
                    "Unable to decode uploaded image."
                )


            print()
            print("-" * 75)
            print("NEW IMAGE")
            print(f"Original filename : {file.filename}")
            print(
                f"Resolution        : "
                f"{image.shape[1]} x {image.shape[0]}"
            )


            # ------------------------------------------------
            # INFERENCE
            # ------------------------------------------------

            output, count = segment_rice_bags(
                image
            )


            print(
                f"Rice bags detected: {count}"
            )


            # ------------------------------------------------
            # SAVE RESULT
            # ------------------------------------------------

            filename = (
                f"result_"
                f"{uuid.uuid4().hex[:12]}"
                f".jpg"
            )

            output_path = (
                OUTPUT_DIR / filename
            )


            success = cv2.imwrite(
                str(output_path),
                output
            )


            if not success:

                raise RuntimeError(
                    "Failed to save result image."
                )


            # ------------------------------------------------
            # VERIFY OUTPUT
            # ------------------------------------------------

            if not output_path.exists():

                raise RuntimeError(
                    "Result image does not exist "
                    "after saving."
                )


            print(
                f"Result saved      : {output_path}"
            )

            print(
                f"File size         : "
                f"{output_path.stat().st_size / 1024:.2f} KB"
            )

            print("-" * 75)


            result_image = filename


        except Exception as exc:

            print()
            print("ERROR:")
            print(exc)

            error = str(exc)


    # ========================================================
    # RETURN PAGE
    # ========================================================

    return render_template_string(
        HTML,
        result_image=result_image,
        count=count,
        confidence=CONF_THRESHOLD,
        error=error
    )


# ============================================================
# SERVE RESULT IMAGE
# ============================================================

@app.route("/outputs/<filename>")
def outputs(filename):

    # Prevent path traversal
    safe_filename = Path(filename).name

    file_path = (
        OUTPUT_DIR / safe_filename
    ).resolve()


    # --------------------------------------------------------
    # Security check
    # --------------------------------------------------------

    if OUTPUT_DIR not in file_path.parents:

        return "Invalid file path", 403


    # --------------------------------------------------------
    # File check
    # --------------------------------------------------------

    if not file_path.exists():

        print(
            f"ERROR: Image not found:\n"
            f"{file_path}"
        )

        return "Output image not found", 404


    # --------------------------------------------------------
    # SEND IMAGE
    # --------------------------------------------------------

    return send_file(
        str(file_path),
        mimetype="image/jpeg"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return {
        "status": "ok",
        "model": "YOLO26l-Seg",
        "target_class": "Rice-bag-10",
        "confidence": CONF_THRESHOLD
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 75)
    print("                    UI STARTED")
    print("=" * 75)

    print(f"Project root : {BASE_DIR}")
    print(f"Model        : {MODEL_PATH}")
    print(f"Output dir   : {OUTPUT_DIR}")
    print(f"Port         : {PORT}")
    print(f"Local URL    : http://localhost:{PORT}")
    print(
        f"Network URL  : "
        f"http://192.168.2.240:{PORT}"
    )
    print(f"Confidence   : {CONF_THRESHOLD}")
    print("Target       : Rice-bag-10 ONLY")
    print("Masks        : ENABLED")
    print("Center point : ENABLED")
    print("Bag numbers  : ENABLED")
    print("Bounding box : DISABLED")

    print("=" * 75)
    print()


    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=False
    )