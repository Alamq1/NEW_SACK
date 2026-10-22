from pathlib import Path
import uuid

import cv2
import torch
from flask import Flask, render_template_string, request, send_file
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ============================================================
# ACTUAL TRAINING RUN LOCATION
# ============================================================
#
# Actual structure:
#
# NEW_SACK_COUNT/
# └── runs/
#     └── segment/
#         └── runs/
#             └── segment/
#                 └── final_c_belt_sack_yolo26l_seg_v1/
#                     └── weights/
#                         ├── best.pt
#                         └── last.pt
#
# ============================================================

RUN_ROOT = (
    PROJECT_ROOT
    / "runs"
    / "segment"
    / "runs"
    / "segment"
    / "final_c_belt_sack_yolo26l_seg_v1"
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

UPLOAD_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "final_c_belt_seg_ui"
)

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DEFAULT INFERENCE SETTINGS
# ============================================================

DEFAULT_CONF = 0.35
DEFAULT_IOU = 0.50

IMAGE_SIZE = 1280

HOST = "0.0.0.0"
PORT = 5002


# ============================================================
# FIND BEST.PT
# ============================================================

def find_best_model():

    print("\n" + "=" * 70)
    print("SEARCHING FOR TRAINED BEST.PT")
    print("=" * 70)

    print(f"Expected training run:")
    print(RUN_ROOT)

    # --------------------------------------------------------
    # First check the exact expected path
    # --------------------------------------------------------

    exact_best = RUN_ROOT / "weights" / "best.pt"

    if exact_best.exists():

        size_mb = exact_best.stat().st_size / (
            1024 * 1024
        )

        print("\n✅ BEST MODEL FOUND")
        print(f"Path : {exact_best}")
        print(f"Size : {size_mb:.2f} MB")

        print("=" * 70)

        return exact_best

    # --------------------------------------------------------
    # If exact path doesn't exist, search recursively
    # --------------------------------------------------------

    if RUN_ROOT.exists():

        print(
            "\n⚠️ best.pt not found at expected weights path."
        )

        print("Searching recursively inside training run...")

        candidates = list(
            RUN_ROOT.rglob("best.pt")
        )

    else:

        print(
            "\n⚠️ Expected training directory does not exist."
        )

        candidates = []

    # --------------------------------------------------------
    # If still not found, search entire runs directory
    # --------------------------------------------------------

    if not candidates:

        global_runs = (
            PROJECT_ROOT
            / "runs"
        )

        print(
            "\nSearching entire runs directory..."
        )

        if global_runs.exists():

            candidates = list(
                global_runs.rglob("best.pt")
            )

    # --------------------------------------------------------
    # Nothing found
    # --------------------------------------------------------

    if not candidates:

        raise FileNotFoundError(
            "\n❌ best.pt could not be found.\n\n"
            "Run this command manually:\n\n"
            f"find '{PROJECT_ROOT}/runs' "
            "-name 'best.pt' -type f"
        )

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    candidates = list(
        dict.fromkeys(candidates)
    )

    # --------------------------------------------------------
    # Sort newest first
    # --------------------------------------------------------

    candidates.sort(
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    print("\nFound model candidates:")

    for i, path in enumerate(
        candidates,
        start=1
    ):

        size_mb = path.stat().st_size / (
            1024 * 1024
        )

        print(
            f"{i}. {path} "
            f"({size_mb:.2f} MB)"
        )

    # --------------------------------------------------------
    # Select newest
    # --------------------------------------------------------

    selected = candidates[0]

    print("\n" + "-" * 70)

    print(
        f"✅ Using BEST model:\n{selected}"
    )

    print("-" * 70)

    return selected


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():

    DEVICE = 0

    DEVICE_NAME = torch.cuda.get_device_name(0)

else:

    DEVICE = "cpu"

    DEVICE_NAME = "CPU"


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = find_best_model()

print("\nLoading YOLO26L-Seg model...")

model = YOLO(
    str(MODEL_PATH)
)

print(f"Device : {DEVICE_NAME}")
print(f"Model  : {MODEL_PATH}")
print()


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

    <title>C-Belt Sack Segmentation Test</title>

    <style>

        body {
            font-family: Arial, sans-serif;
            background: #f4f4f4;
            margin: 0;
            padding: 30px;
        }

        .container {
            max-width: 1100px;
            margin: auto;
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 3px 15px rgba(0,0,0,0.12);
        }

        h1 {
            margin-top: 0;
        }

        .model-info {
            background: #f0f0f0;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-size: 14px;
            word-break: break-all;
        }

        .controls {
            display: flex;
            gap: 20px;
            align-items: center;
            flex-wrap: wrap;
            margin-bottom: 20px;
        }

        input[type=file] {
            padding: 8px;
        }

        input[type=number] {
            width: 80px;
            padding: 7px;
        }

        button {
            padding: 10px 18px;
            border: none;
            border-radius: 7px;
            cursor: pointer;
            font-weight: bold;
        }

        button:hover {
            opacity: 0.85;
        }

        .result {
            margin-top: 25px;
        }

        .count {
            font-size: 26px;
            font-weight: bold;
            margin-bottom: 15px;
        }

        img {
            max-width: 100%;
            border-radius: 8px;
            border: 1px solid #ddd;
        }

        video {
            width: 100%;
            max-height: 700px;
            border-radius: 8px;
            background: black;
        }

        .error {
            background: #ffe5e5;
            padding: 15px;
            border-radius: 8px;
            color: #990000;
        }

        .success {
            background: #e8f7e8;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }

        .note {
            font-size: 13px;
            color: #666;
            margin-top: 10px;
        }

    </style>

</head>


<body>


<div class="container">


    <h1>
        🟢 C-Belt Sack Segmentation Test UI
    </h1>


    <div class="model-info">

        <b>Model:</b>
        {{ model_path }}

        <br><br>

        <b>Device:</b>
        {{ device }}

    </div>


    <form
        method="POST"
        enctype="multipart/form-data"
    >


        <div class="controls">


            <div>

                <b>Upload Image / Video</b>

                <br>

                <input
                    type="file"
                    name="media"
                    accept="image/*,video/*"
                    required
                >

            </div>


            <div>

                <b>Confidence</b>

                <br>

                <input
                    type="number"
                    name="conf"
                    min="0.05"
                    max="0.95"
                    step="0.05"
                    value="{{ conf }}"
                >

            </div>


            <div>

                <b>IoU</b>

                <br>

                <input
                    type="number"
                    name="iou"
                    min="0.10"
                    max="0.95"
                    step="0.05"
                    value="{{ iou }}"
                >

            </div>


            <div>

                <br>

                <button type="submit">
                    Run Segmentation
                </button>

            </div>


        </div>


    </form>


    {% if error %}

        <div class="error">

            <b>Error:</b>

            <br><br>

            {{ error }}

        </div>

    {% endif %}


    {% if result_type == "image" %}

        <div class="result">


            <div class="count">

                Detected Sacks:
                {{ count }}

            </div>


            <img
                src="{{ result_url }}"
            >


            <div class="note">

                Instance segmentation masks are displayed.
                Bounding boxes, labels and confidence values
                are hidden for clean visualization.

            </div>


        </div>

    {% endif %}


    {% if result_type == "video" %}

        <div class="result">


            <div class="success">

                <b>Video processing completed.</b>

            </div>


            <video controls>

                <source
                    src="{{ result_url }}"
                    type="video/mp4"
                >

                Your browser does not support
                video playback.

            </video>


            <div class="note">

                Segmentation masks are rendered frame-by-frame.

            </div>


        </div>

    {% endif %}


</div>


</body>

</html>
"""


# ============================================================
# IMAGE PROCESSING
# ============================================================

def process_image(
    input_path,
    output_path,
    conf,
    iou
):

    print("\nRunning image inference...")

    results = model.predict(

        source=str(input_path),

        conf=conf,

        iou=iou,

        imgsz=IMAGE_SIZE,

        device=DEVICE,

        verbose=False,

    )

    result = results[0]


    # --------------------------------------------------------
    # Count segmentation instances
    # --------------------------------------------------------

    if result.masks is not None:

        count = len(
            result.masks.data
        )

    else:

        count = 0


    # --------------------------------------------------------
    # Clean segmentation visualization
    # --------------------------------------------------------

    plotted = result.plot(

        boxes=False,

        labels=False,

        conf=False,

        masks=True,

    )


    cv2.imwrite(
        str(output_path),
        plotted
    )


    print(
        f"Detected sacks: {count}"
    )

    print(
        f"Saved result: {output_path}"
    )


    return count


# ============================================================
# VIDEO PROCESSING
# ============================================================

def process_video(
    input_path,
    output_path,
    conf,
    iou
):

    print("\nStarting video processing...")

    cap = cv2.VideoCapture(
        str(input_path)
    )


    if not cap.isOpened():

        raise RuntimeError(
            "Could not open uploaded video."
        )


    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )


    if fps <= 0:

        fps = 25.0


    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )


    writer = cv2.VideoWriter(

        str(output_path),

        fourcc,

        fps,

        (width, height),

    )


    if not writer.isOpened():

        cap.release()

        raise RuntimeError(
            "Could not create output video."
        )


    frame_count = 0


    while True:

        ret, frame = cap.read()


        if not ret:

            break


        results = model.predict(

            source=frame,

            conf=conf,

            iou=iou,

            imgsz=IMAGE_SIZE,

            device=DEVICE,

            verbose=False,

        )


        result = results[0]


        plotted = result.plot(

            boxes=False,

            labels=False,

            conf=False,

            masks=True,

        )


        writer.write(
            plotted
        )


        frame_count += 1


        if frame_count % 50 == 0:

            print(
                f"Processed "
                f"{frame_count} frames..."
            )


    cap.release()

    writer.release()


    print(
        f"\nVideo processing completed."
    )

    print(
        f"Total frames: {frame_count}"
    )

    print(
        f"Output: {output_path}"
    )


# ============================================================
# MAIN ROUTE
# ============================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)

def index():

    error = None

    result_type = None

    result_url = None

    count = None


    conf = DEFAULT_CONF

    iou = DEFAULT_IOU


    if request.method == "POST":

        try:

            file = request.files.get(
                "media"
            )


            if (
                file is None
                or file.filename == ""
            ):

                raise ValueError(
                    "Please select an image or video."
                )


            conf = float(
                request.form.get(
                    "conf",
                    DEFAULT_CONF
                )
            )


            iou = float(
                request.form.get(
                    "iou",
                    DEFAULT_IOU
                )
            )


            if not 0.01 <= conf <= 0.99:

                raise ValueError(
                    "Confidence must be between "
                    "0.01 and 0.99."
                )


            if not 0.01 <= iou <= 0.99:

                raise ValueError(
                    "IoU must be between "
                    "0.01 and 0.99."
                )


            extension = Path(
                file.filename
            ).suffix.lower()


            unique_id = uuid.uuid4().hex[:10]


            # ------------------------------------------------
            # Save input
            # ------------------------------------------------

            input_path = (
                UPLOAD_DIR
                / f"input_{unique_id}{extension}"
            )


            file.save(
                input_path
            )


            # ------------------------------------------------
            # Supported formats
            # ------------------------------------------------

            image_extensions = {

                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".webp",

            }


            video_extensions = {

                ".mp4",
                ".avi",
                ".mov",
                ".mkv",
                ".webm",

            }


            # =================================================
            # IMAGE
            # =================================================

            if extension in image_extensions:


                output_path = (
                    UPLOAD_DIR
                    / f"result_{unique_id}.jpg"
                )


                count = process_image(

                    input_path,

                    output_path,

                    conf,

                    iou,

                )


                result_type = "image"

                result_url = (
                    f"/result/"
                    f"{output_path.name}"
                )


            # =================================================
            # VIDEO
            # =================================================

            elif extension in video_extensions:


                output_path = (
                    UPLOAD_DIR
                    / f"result_{unique_id}.mp4"
                )


                process_video(

                    input_path,

                    output_path,

                    conf,

                    iou,

                )


                result_type = "video"

                result_url = (
                    f"/result/"
                    f"{output_path.name}"
                )


            else:

                raise ValueError(

                    f"Unsupported file type: "
                    f"{extension}"

                )


        except Exception as e:

            error = str(e)


    return render_template_string(

        HTML,

        model_path=str(
            MODEL_PATH
        ),

        device=DEVICE_NAME,

        result_type=result_type,

        result_url=result_url,

        count=count,

        error=error,

        conf=conf,

        iou=iou,

    )


# ============================================================
# SERVE RESULT FILES
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

        return (
            "File not found",
            404
        )


    return send_file(
        path
    )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)

    print(
        "C-BELT SACK YOLO26L-SEG TEST UI"
    )

    print("=" * 70)

    print(
        f"Model : {MODEL_PATH}"
    )

    print(
        f"Device: {DEVICE_NAME}"
    )

    print(
        f"URL   : http://localhost:{PORT}"
    )

    print("=" * 70)

    print(
        "\nOpen the URL in your browser."
    )

    print()


    app.run(

        host=HOST,

        port=PORT,

        debug=False,

        threaded=True,

    )