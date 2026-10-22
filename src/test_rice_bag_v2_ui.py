# from flask import Flask, request, render_template_string, send_file
# from ultralytics import YOLO
# from pathlib import Path
# import cv2
# import torch
# import uuid


# # ============================================================
# # PROJECT PATHS
# # ============================================================

# PROJECT_ROOT = Path(__file__).resolve().parents[1]

# MODEL_PATH = (
#     PROJECT_ROOT
#     / "outputs"
#     / "rice_bag_detection_v2"
#     / "weights"
#     / "best.pt"
# )

# UPLOAD_DIR = PROJECT_ROOT / "data" / "test_uploads_v2"

# OUTPUT_DIR = (
#     PROJECT_ROOT
#     / "outputs"
#     / "ui_predictions_v2"
# )

# UPLOAD_DIR.mkdir(
#     parents=True,
#     exist_ok=True
# )

# OUTPUT_DIR.mkdir(
#     parents=True,
#     exist_ok=True
# )


# # ============================================================
# # CONFIGURATION
# # ============================================================

# BAG_WEIGHT_KG = 50

# DEVICE = 0 if torch.cuda.is_available() else "cpu"

# IMAGE_SIZE = 960


# # ============================================================
# # LOAD MODEL
# # ============================================================

# print("=" * 70)
# print("RICE BAG V2 - INVENTORY TEST UI")
# print("=" * 70)

# print(f"Model Path: {MODEL_PATH}")
# print(f"Device    : {DEVICE}")

# if not MODEL_PATH.exists():

#     raise FileNotFoundError(
#         f"\nModel not found:\n{MODEL_PATH}\n"
#         "\nMake sure V2 training has completed and best.pt exists."
#     )


# model = YOLO(str(MODEL_PATH))

# print("\nModel loaded successfully.")
# print("=" * 70)


# # ============================================================
# # FLASK APP
# # ============================================================

# app = Flask(__name__)


# # ============================================================
# # HTML TEMPLATE
# # ============================================================

# HTML_TEMPLATE = """

# <!DOCTYPE html>

# <html>

# <head>

#     <title>Rice Bag V2 Tester</title>

#     <style>

#         body {
#             font-family: Arial, sans-serif;
#             background: #111;
#             color: white;
#             margin: 0;
#             padding: 30px;
#         }


#         .container {
#             max-width: 1400px;
#             margin: auto;
#         }


#         h1 {
#             text-align: center;
#         }


#         .panel {

#             background: #1e1e1e;

#             padding: 25px;

#             border-radius: 15px;

#             margin-bottom: 25px;
#         }


#         input[type=file] {

#             padding: 10px;

#             margin-bottom: 20px;
#         }


#         input[type=range] {

#             width: 350px;
#         }


#         button {

#             padding: 12px 30px;

#             font-size: 16px;

#             cursor: pointer;

#             border-radius: 8px;

#             border: none;
#         }


#         .results {

#             display: flex;

#             gap: 20px;

#             flex-wrap: wrap;
#         }


#         .image-box {

#             flex: 1;

#             min-width: 450px;
#         }


#         img {

#             width: 100%;

#             border-radius: 10px;
#         }


#         .stats {

#             background: #1e1e1e;

#             padding: 25px;

#             border-radius: 15px;

#             font-size: 20px;

#             line-height: 2;
#         }

#     </style>

# </head>


# <body>

# <div class="container">

#     <h1>Rice Bag Detection V2</h1>


#     <!-- ==================================================== -->
#     <!-- UPLOAD PANEL -->
#     <!-- ==================================================== -->

#     <div class="panel">

#         <form
#             method="POST"
#             enctype="multipart/form-data"
#         >

#             <h3>Upload Rice Bag Image</h3>

#             <input
#                 type="file"
#                 name="image"
#                 accept="image/*"
#                 required
#             >

#             <br>


#             <h3>

#                 Confidence:
#                 <span id="conf_value">
#                     {{ confidence if confidence else "0.25" }}
#                 </span>

#             </h3>


#             <input
#                 type="range"

#                 name="confidence"

#                 min="0.05"
#                 max="0.90"
#                 step="0.05"

#                 value="{{ confidence if confidence else "0.25" }}"

#                 oninput="conf_value.innerText=this.value"
#             >


#             <h3>

#                 IoU:
#                 <span id="iou_value">
#                     {{ iou if iou else "0.70" }}
#                 </span>

#             </h3>


#             <input
#                 type="range"

#                 name="iou"

#                 min="0.10"
#                 max="0.90"
#                 step="0.05"

#                 value="{{ iou if iou else "0.70" }}"

#                 oninput="iou_value.innerText=this.value"
#             >


#             <br><br>


#             <button type="submit">

#                 TEST IMAGE

#             </button>

#         </form>

#     </div>


#     <!-- ==================================================== -->
#     <!-- RESULTS -->
#     <!-- ==================================================== -->

#     {% if result_image %}


#     <div class="stats">

#         <b>Total Bags Detected:</b>
#         {{ bag_count }}

#         <br>

#         <b>Weight Per Bag:</b>
#         {{ bag_weight }} KG

#         <br>

#         <b>Total Weight:</b>
#         {{ total_weight }} KG

#         <br>

#         <b>Confidence:</b>
#         {{ confidence }}

#         <br>

#         <b>IoU:</b>
#         {{ iou }}

#     </div>


#     <br>


#     <div class="results">


#         <div class="image-box">

#             <h2>Original Image</h2>

#             <img src="/original/{{ image_id }}">

#         </div>


#         <div class="image-box">

#             <h2>Detection Result</h2>

#             <img src="/result/{{ image_id }}">

#         </div>


#     </div>


#     {% endif %}


# </div>


# </body>

# </html>

# """


# # ============================================================
# # GET ORIGINAL IMAGE
# # ============================================================

# @app.route("/original/<image_id>")
# def get_original(image_id):

#     image_path = (
#         UPLOAD_DIR
#         / f"{image_id}.jpg"
#     )

#     return send_file(
#         image_path,
#         mimetype="image/jpeg"
#     )


# # ============================================================
# # GET RESULT IMAGE
# # ============================================================

# @app.route("/result/<image_id>")
# def get_result(image_id):

#     image_path = (
#         OUTPUT_DIR
#         / f"{image_id}.jpg"
#     )

#     return send_file(
#         image_path,
#         mimetype="image/jpeg"
#     )


# # ============================================================
# # MAIN PAGE
# # ============================================================

# @app.route(
#     "/",
#     methods=["GET", "POST"]
# )
# def home():

#     # ========================================================
#     # POST - IMAGE UPLOADED
#     # ========================================================

#     if request.method == "POST":

#         # ----------------------------------------------------
#         # GET FILE
#         # ----------------------------------------------------

#         uploaded_file = request.files.get("image")

#         if (
#             uploaded_file is None
#             or uploaded_file.filename == ""
#         ):

#             return "Please upload an image."


#         # ----------------------------------------------------
#         # UNIQUE ID
#         # ----------------------------------------------------

#         image_id = str(
#             uuid.uuid4()
#         )


#         input_path = (
#             UPLOAD_DIR
#             / f"{image_id}.jpg"
#         )


#         output_path = (
#             OUTPUT_DIR
#             / f"{image_id}.jpg"
#         )


#         # ----------------------------------------------------
#         # SAVE UPLOADED FILE
#         # ----------------------------------------------------

#         uploaded_file.save(
#             str(input_path)
#         )


#         # ----------------------------------------------------
#         # GET SETTINGS
#         # ----------------------------------------------------

#         confidence = float(
#             request.form.get(
#                 "confidence",
#                 0.25
#             )
#         )


#         iou = float(
#             request.form.get(
#                 "iou",
#                 0.70
#             )
#         )


#         # ----------------------------------------------------
#         # LOAD IMAGE
#         # ----------------------------------------------------

#         image = cv2.imread(
#             str(input_path)
#         )


#         if image is None:

#             return (
#                 "ERROR: Could not read "
#                 "uploaded image."
#             )


#         # ----------------------------------------------------
#         # RUN YOLO
#         # ----------------------------------------------------

#         results = model.predict(

#             source=image,

#             conf=confidence,

#             iou=iou,

#             imgsz=IMAGE_SIZE,

#             device=DEVICE,

#             verbose=False,

#             max_det=1000
#         )


#         result = results[0]


#         # ----------------------------------------------------
#         # CLEAN IMAGE FOR OUTPUT
#         # ----------------------------------------------------

#         output_image = image.copy()


#         # ----------------------------------------------------
#         # GET DETECTED BOXES
#         # ----------------------------------------------------

#         bag_count = 0


#         if (
#             result.boxes is not None
#             and len(result.boxes) > 0
#         ):


#             boxes = (
#                 result.boxes
#                 .xyxy
#                 .cpu()
#                 .numpy()
#             )


#             # ------------------------------------------------
#             # SORT DETECTIONS
#             #
#             # TOP → BOTTOM
#             # LEFT → RIGHT
#             # ------------------------------------------------

#             boxes = sorted(

#                 boxes,

#                 key=lambda box: (

#                     (box[1] + box[3]) / 2,

#                     (box[0] + box[2]) / 2
#                 )
#             )


#             # ------------------------------------------------
#             # DRAW CENTER POINTS
#             # ------------------------------------------------

#             for box in boxes:


#                 x1, y1, x2, y2 = map(
#                     int,
#                     box
#                 )


#                 # Center of detection

#                 center_x = (
#                     x1 + x2
#                 ) // 2


#                 center_y = (
#                     y1 + y2
#                 ) // 2


#                 # Increase count

#                 bag_count += 1


#                 # Draw center point

#                 cv2.circle(

#                     output_image,

#                     (
#                         center_x,
#                         center_y
#                     ),

#                     6,

#                     (0, 255, 0),

#                     -1
#                 )


#                 # Bag number

#                 label = str(
#                     bag_count
#                 )


#                 cv2.putText(

#                     output_image,

#                     label,

#                     (
#                         center_x + 8,
#                         center_y - 8
#                     ),

#                     cv2.FONT_HERSHEY_SIMPLEX,

#                     0.6,

#                     (0, 255, 0),

#                     2
#                 )


#         # ====================================================
#         # CALCULATE TOTAL WEIGHT
#         # ====================================================

#         total_weight = (
#             bag_count
#             * BAG_WEIGHT_KG
#         )


#         # ====================================================
#         # SUMMARY
#         # ====================================================

#         cv2.putText(

#             output_image,

#             f"TOTAL BAGS: {bag_count}",

#             (20, 40),

#             cv2.FONT_HERSHEY_SIMPLEX,

#             1,

#             (255, 255, 255),

#             2
#         )


#         cv2.putText(

#             output_image,

#             f"TOTAL WEIGHT: {total_weight} KG",

#             (20, 80),

#             cv2.FONT_HERSHEY_SIMPLEX,

#             1,

#             (255, 255, 255),

#             2
#         )


#         # ====================================================
#         # SAVE OUTPUT
#         # ====================================================

#         cv2.imwrite(

#             str(output_path),

#             output_image
#         )


#         # ====================================================
#         # RETURN RESULTS
#         # ====================================================

#         return render_template_string(

#             HTML_TEMPLATE,

#             result_image=True,

#             image_id=image_id,

#             bag_count=bag_count,

#             bag_weight=BAG_WEIGHT_KG,

#             total_weight=total_weight,

#             confidence=confidence,

#             iou=iou
#         )


#     # ========================================================
#     # FIRST PAGE
#     # ========================================================

#     return render_template_string(

#         HTML_TEMPLATE,

#         result_image=False,

#         confidence=0.25,

#         iou=0.70
#     )


# # ============================================================
# # START SERVER
# # ============================================================

# if __name__ == "__main__":

#     print("\nStarting Flask server...")

#     print(
#         "Open in browser:"
#     )

#     print(
#         "http://192.168.2.240:5003"
#     )


#     app.run(

#         host="0.0.0.0",

#         port=5003,

#         debug=False
#     )

from flask import Flask, request, render_template_string, send_file
from ultralytics import YOLO
from pathlib import Path
import cv2
import torch
import uuid


# ============================================================
# PROJECT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "rice_bag_detection_v2"
    / "weights"
    / "best.pt"
)


# ============================================================
# DIRECTORIES
# ============================================================

UPLOAD_DIR = (
    PROJECT_ROOT
    / "data"
    / "test_uploads_v2"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "ui_predictions_v2"
)


UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIG
# ============================================================

BAG_WEIGHT_KG = 50

IMAGE_SIZE = 960

DEVICE = (
    0
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("RICE BAG V2 TESTING UI")
print("=" * 70)

print(f"Model : {MODEL_PATH}")
print(f"Device: {DEVICE}")


if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}"
    )


model = YOLO(
    str(MODEL_PATH)
)


print("Model loaded successfully.")

print("=" * 70)


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

<title>Rice Bag Detection V2</title>

<style>

body {

    font-family: Arial;

    background: #111;

    color: white;

    margin: 0;

    padding: 30px;

}


.container {

    max-width: 1400px;

    margin: auto;

}


.panel {

    background: #1e1e1e;

    padding: 25px;

    border-radius: 15px;

    margin-bottom: 25px;

}


input[type=file] {

    margin: 15px 0;

}


input[type=range] {

    width: 400px;

}


button {

    padding: 12px 30px;

    font-size: 16px;

    border: none;

    border-radius: 8px;

    cursor: pointer;

}


.stats {

    background: #1e1e1e;

    padding: 25px;

    border-radius: 15px;

    font-size: 20px;

    line-height: 2;

}


.results {

    display: flex;

    gap: 20px;

    margin-top: 25px;

}


.image-box {

    width: 50%;

}


img {

    width: 100%;

    border-radius: 10px;

}


</style>

</head>


<body>


<div class="container">


<h1>🌾 Rice Bag Detection V2</h1>


<div class="panel">


<form
method="POST"
enctype="multipart/form-data"
>


<h3>Upload Image</h3>


<input
type="file"
name="image"
accept="image/*"
required
>


<br>


<h3>

Confidence:

<span id="conf_value">

{{ confidence }}

</span>

</h3>


<input
type="range"

name="confidence"

min="0.20"

max="0.80"

step="0.05"

value="{{ confidence }}"

oninput="conf_value.innerText=this.value"
>


<br>


<h3>

IoU:

<span id="iou_value">

{{ iou }}

</span>

</h3>


<input
type="range"

name="iou"

min="0.30"

max="0.70"

step="0.05"

value="{{ iou }}"

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


<b>Total Bags:</b>

{{ bag_count }}


<br>


<b>Weight Per Bag:</b>

{{ bag_weight }} KG


<br>


<b>Total Weight:</b>

{{ total_weight }} KG


<br>


<b>Confidence:</b>

{{ confidence }}


<br>


<b>IoU:</b>

{{ iou }}


</div>


<div class="results">


<div class="image-box">

<h2>Original</h2>

<img src="/original/{{ image_id }}">

</div>


<div class="image-box">

<h2>Detection</h2>

<img src="/result/{{ image_id }}">

</div>


</div>


{% endif %}


</div>


</body>

</html>

"""


# ============================================================
# ORIGINAL IMAGE
# ============================================================

@app.route(
    "/original/<image_id>"
)
def original(image_id):

    path = (
        UPLOAD_DIR
        / f"{image_id}.jpg"
    )

    return send_file(
        path,
        mimetype="image/jpeg"
    )


# ============================================================
# RESULT IMAGE
# ============================================================

@app.route(
    "/result/<image_id>"
)
def result(image_id):

    path = (
        OUTPUT_DIR
        / f"{image_id}.jpg"
    )

    return send_file(
        path,
        mimetype="image/jpeg"
    )


# ============================================================
# HOME
# ============================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
def home():


    # ========================================================
    # GET
    # ========================================================

    if request.method == "GET":

        return render_template_string(

            HTML,

            result_image=False,

            confidence=0.45,

            iou=0.45

        )


    # ========================================================
    # IMAGE
    # ========================================================

    uploaded = request.files.get(
        "image"
    )


    if (
        uploaded is None
        or uploaded.filename == ""
    ):

        return "Please upload an image."


    # ========================================================
    # UNIQUE ID
    # ========================================================

    image_id = str(
        uuid.uuid4()
    )


    input_path = (
        UPLOAD_DIR
        / f"{image_id}.jpg"
    )


    output_path = (
        OUTPUT_DIR
        / f"{image_id}.jpg"
    )


    uploaded.save(
        str(input_path)
    )


    # ========================================================
    # PARAMETERS
    # ========================================================

    confidence = float(
        request.form.get(
            "confidence",
            0.45
        )
    )


    iou = float(
        request.form.get(
            "iou",
            0.45
        )
    )


    # ========================================================
    # READ IMAGE
    # ========================================================

    image = cv2.imread(
        str(input_path)
    )


    if image is None:

        return "Could not read image."


    # ========================================================
    # YOLO PREDICTION
    # ========================================================

    results = model.predict(

        source=image,

        conf=confidence,

        iou=iou,

        imgsz=IMAGE_SIZE,

        device=DEVICE,

        max_det=300,

        verbose=False

    )


    result = results[0]


    output = image.copy()


    # ========================================================
    # DETECTIONS
    # ========================================================

    bag_count = 0


    if (
        result.boxes is not None
        and len(result.boxes) > 0
    ):


        boxes = (
            result.boxes
            .xyxy
            .cpu()
            .numpy()
        )


        # ----------------------------------------------------
        # SORT
        # ----------------------------------------------------

        boxes = sorted(

            boxes,

            key=lambda box: (

                (box[1] + box[3]) / 2,

                (box[0] + box[2]) / 2

            )

        )


        # ----------------------------------------------------
        # CENTER POINTS ONLY
        # ----------------------------------------------------

        for box in boxes:


            x1, y1, x2, y2 = map(
                int,
                box
            )


            center_x = (
                x1 + x2
            ) // 2


            center_y = (
                y1 + y2
            ) // 2


            bag_count += 1


            # ----------------------------------------------
            # POINT
            # ----------------------------------------------

            cv2.circle(

                output,

                (
                    center_x,
                    center_y
                ),

                6,

                (0, 255, 0),

                -1

            )


            # ----------------------------------------------
            # NUMBER
            # ----------------------------------------------

            cv2.putText(

                output,

                str(bag_count),

                (
                    center_x + 8,
                    center_y - 8
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.55,

                (0, 255, 0),

                2

            )


    # ========================================================
    # WEIGHT
    # ========================================================

    total_weight = (
        bag_count
        * BAG_WEIGHT_KG
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    cv2.putText(

        output,

        f"TOTAL BAGS: {bag_count}",

        (20, 40),

        cv2.FONT_HERSHEY_SIMPLEX,

        1,

        (255, 255, 255),

        2

    )


    cv2.putText(

        output,

        f"TOTAL WEIGHT: {total_weight} KG",

        (20, 80),

        cv2.FONT_HERSHEY_SIMPLEX,

        1,

        (255, 255, 255),

        2

    )


    # ========================================================
    # SAVE
    # ========================================================

    cv2.imwrite(

        str(output_path),

        output

    )


    # ========================================================
    # RETURN
    # ========================================================

    return render_template_string(

        HTML,

        result_image=True,

        image_id=image_id,

        bag_count=bag_count,

        bag_weight=BAG_WEIGHT_KG,

        total_weight=total_weight,

        confidence=confidence,

        iou=iou

    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    print(
        "\nOpen:"
    )

    print(
        "http://192.168.2.240:5003"
    )

    app.run(

        host="0.0.0.0",

        port=5005,

        debug=False

    )