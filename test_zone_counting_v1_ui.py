# from flask import Flask, request, render_template_string, send_from_directory
# from ultralytics import YOLO
# from pathlib import Path
# import uuid
# import cv2
# import numpy as np


# # ============================================================
# # ZONE COUNTING — TEST UI
# # YOLO26L-Seg v1_1280
# # ============================================================

# BASE_DIR = Path("/home/aiteam/ZONE_COUNTING")

# MODEL_PATH = (
#     BASE_DIR
#     / "runs"
#     / "segment"
#     / "zone_counting_yolo26l_seg_v1_1280"
#     / "weights"
#     / "best.pt"
# )

# UPLOAD_DIR = BASE_DIR / "test_uploads"
# OUTPUT_DIR = BASE_DIR / "test_outputs"

# UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# # ============================================================
# # MODEL
# # ============================================================

# print("=" * 70)
# print("ZONE COUNTING — TEST UI")
# print("=" * 70)

# print(f"Model: {MODEL_PATH}")

# if not MODEL_PATH.exists():
#     raise FileNotFoundError(
#         f"\nBEST MODEL NOT FOUND:\n{MODEL_PATH}\n"
#         "\nTrain the model first."
#     )

# model = YOLO(str(MODEL_PATH))

# print("Model loaded successfully.")


# # ============================================================
# # FLASK
# # ============================================================

# app = Flask(__name__)


# # ============================================================
# # HTML
# # ============================================================

# HTML = """
# <!DOCTYPE html>

# <html>

# <head>

#     <title>Zone Counting - YOLO26L Segmentation</title>

#     <style>

#         body {
#             font-family: Arial, sans-serif;
#             background: #f4f4f4;
#             margin: 0;
#             padding: 30px;
#         }

#         .container {
#             max-width: 1100px;
#             margin: auto;
#             background: white;
#             padding: 30px;
#             border-radius: 12px;
#             box-shadow: 0 3px 15px rgba(0,0,0,0.15);
#         }

#         h1 {
#             margin-top: 0;
#         }

#         .model {
#             background: #eeeeee;
#             padding: 12px;
#             border-radius: 8px;
#             margin-bottom: 25px;
#             word-break: break-all;
#         }

#         input {
#             margin: 15px 0;
#         }

#         button {
#             padding: 10px 20px;
#             border: none;
#             border-radius: 7px;
#             cursor: pointer;
#             font-size: 16px;
#         }

#         .result {
#             margin-top: 30px;
#         }

#         img {
#             max-width: 100%;
#             border-radius: 8px;
#             margin-top: 15px;
#         }

#         .count {
#             font-size: 28px;
#             font-weight: bold;
#             margin-top: 15px;
#         }

#         .confidence {
#             margin-top: 10px;
#         }

#     </style>

# </head>


# <body>

# <div class="container">

#     <h1>Zone Counting — YOLO26L Segmentation</h1>

#     <div class="model">
#         <b>Model:</b> {{ model_path }}
#     </div>

#     <form method="POST" enctype="multipart/form-data">

#         <input
#             type="file"
#             name="file"
#             accept="image/*"
#             required
#         >

#         <br>

#         <button type="submit">
#             Test Image
#         </button>

#     </form>


#     {% if result %}

#     <div class="result">

#         <hr>

#         <h2>Prediction Result</h2>

#         <div class="count">
#             Detected Objects: {{ count }}
#         </div>

#         <div class="confidence">
#             Average Confidence: {{ confidence }}
#         </div>

#         <img src="{{ url_for('output_file', filename=output_file) }}">

#     </div>

#     {% endif %}

# </div>

# </body>

# </html>
# """


# # ============================================================
# # IMAGE PROCESSING
# # ============================================================

# def run_inference(image_path, output_path):

#     image = cv2.imread(str(image_path))

#     if image is None:
#         raise ValueError("Unable to read uploaded image.")

#     # --------------------------------------------------------
#     # Inference
#     # --------------------------------------------------------

#     results = model.predict(
#         source=image,
#         imgsz=1280,
#         conf=0.25,
#         iou=0.50,
#         device=0,
#         verbose=False
#     )

#     result = results[0]

#     # --------------------------------------------------------
#     # Detection count
#     # --------------------------------------------------------

#     if result.boxes is None:
#         count = 0
#         confidences = []

#     else:
#         count = len(result.boxes)

#         if result.boxes.conf is not None:
#             confidences = (
#                 result.boxes.conf.detach()
#                 .cpu()
#                 .numpy()
#                 .tolist()
#             )
#         else:
#             confidences = []

#     # --------------------------------------------------------
#     # Draw segmentation masks
#     # --------------------------------------------------------

#     output = image.copy()

#     if result.masks is not None:

#         masks = result.masks.data.detach().cpu().numpy()

#         for mask in masks:

#             mask = cv2.resize(
#                 mask,
#                 (output.shape[1], output.shape[0]),
#                 interpolation=cv2.INTER_NEAREST
#             )

#             mask_binary = mask > 0.5

#             # Mask overlay
#             overlay = output.copy()

#             overlay[mask_binary] = (
#                 overlay[mask_binary] * 0.45
#                 + np.array([0, 255, 0]) * 0.55
#             ).astype(np.uint8)

#             output = overlay

#             # Mask contour
#             contours, _ = cv2.findContours(
#                 mask_binary.astype(np.uint8),
#                 cv2.RETR_EXTERNAL,
#                 cv2.CHAIN_APPROX_SIMPLE
#             )

#             cv2.drawContours(
#                 output,
#                 contours,
#                 -1,
#                 (0, 255, 0),
#                 2
#             )

#     # --------------------------------------------------------
#     # Count display
#     # --------------------------------------------------------

#     cv2.rectangle(
#         output,
#         (15, 15),
#         (350, 70),
#         (0, 0, 0),
#         -1
#     )

#     cv2.putText(
#         output,
#         f"Objects: {count}",
#         (30, 52),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         1.0,
#         (255, 255, 255),
#         2,
#         cv2.LINE_AA
#     )

#     # --------------------------------------------------------
#     # Save
#     # --------------------------------------------------------

#     cv2.imwrite(str(output_path), output)

#     # --------------------------------------------------------
#     # Confidence
#     # --------------------------------------------------------

#     if confidences:
#         avg_conf = sum(confidences) / len(confidences)
#         confidence_text = f"{avg_conf:.3f}"
#     else:
#         confidence_text = "N/A"

#     return count, confidence_text


# # ============================================================
# # ROUTES
# # ============================================================

# @app.route("/", methods=["GET", "POST"])
# def index():

#     if request.method == "POST":

#         uploaded_file = request.files.get("file")

#         if uploaded_file is None:
#             return "No file uploaded."

#         if uploaded_file.filename == "":
#             return "No file selected."

#         # Unique filename
#         uid = uuid.uuid4().hex

#         original_name = Path(uploaded_file.filename).name

#         input_name = f"{uid}_{original_name}"
#         output_name = f"{uid}_result.jpg"

#         input_path = UPLOAD_DIR / input_name
#         output_path = OUTPUT_DIR / output_name

#         uploaded_file.save(input_path)

#         try:

#             count, confidence = run_inference(
#                 input_path,
#                 output_path
#             )

#         except Exception as e:

#             return f"""
#             <h2>Inference Error</h2>
#             <pre>{e}</pre>
#             """

#         return render_template_string(
#             HTML,
#             model_path=str(MODEL_PATH),
#             result=True,
#             count=count,
#             confidence=confidence,
#             output_file=output_name
#         )

#     return render_template_string(
#         HTML,
#         model_path=str(MODEL_PATH),
#         result=False
#     )


# # ============================================================
# # OUTPUT FILE ROUTE
# # ============================================================

# @app.route("/outputs/<filename>")
# def output_file(filename):

#     return send_from_directory(
#         OUTPUT_DIR,
#         filename
#     )


# # ============================================================
# # START SERVER
# # ============================================================

# if __name__ == "__main__":

#     print("\n" + "=" * 70)
#     print("SERVER STARTING")
#     print("=" * 70)

#     print("\nOpen in browser:")
#     print("http://192.168.2.240:5001")

#     print("\nPress CTRL+C to stop.")

#     app.run(
#         host="0.0.0.0",
#         port=5004,
#         debug=False
#     )




###############################################################################################################################
###############################################################################################################################
###############################################################################################################################

# from flask import Flask, request, render_template_string, send_from_directory
# from ultralytics import YOLO
# from pathlib import Path
# import uuid
# import cv2
# import numpy as np


# # ============================================================
# # ZONE COUNTING — TEST UI v2
# # YOLO26L-Seg
# #
# # Features:
# #   1. Upload image
# #   2. Select Zone
# #   3. Draw rectangular zone
# #   4. Count ONLY objects inside zone
# #   5. Objects outside zone are NOT counted
# #   6. Segmentation visualization
# # ============================================================


# # ============================================================
# # PATHS
# # ============================================================

# BASE_DIR = Path("/home/aiteam/ZONE_COUNTING")

# MODEL_PATH = (
#     BASE_DIR
#     / "runs"
#     / "segment"
#     / "zone_counting_yolo26l_seg_v1_1280"
#     / "weights"
#     / "best.pt"
# )

# UPLOAD_DIR = BASE_DIR / "test_uploads"
# OUTPUT_DIR = BASE_DIR / "test_outputs"

# UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# # ============================================================
# # MODEL
# # ============================================================

# print("=" * 70)
# print("ZONE COUNTING — V2 TEST UI")
# print("=" * 70)

# print(f"Model: {MODEL_PATH}")

# if not MODEL_PATH.exists():
#     raise FileNotFoundError(
#         f"\nBEST MODEL NOT FOUND:\n{MODEL_PATH}"
#     )

# model = YOLO(str(MODEL_PATH))

# print("Model loaded successfully.")


# # ============================================================
# # FLASK
# # ============================================================

# app = Flask(__name__)


# # ============================================================
# # HTML
# # ============================================================

# HTML = """
# <!DOCTYPE html>

# <html>

# <head>

# <title>Zone Counting AI</title>

# <meta name="viewport" content="width=device-width, initial-scale=1.0">

# <style>

# body {
#     margin: 0;
#     font-family: Arial, sans-serif;
#     background: #111827;
#     color: white;
# }

# .header {
#     padding: 20px 30px;
#     background: #1f2937;
#     border-bottom: 1px solid #374151;
# }

# .header h1 {
#     margin: 0;
# }

# .header p {
#     margin: 6px 0 0 0;
#     color: #9ca3af;
# }

# .container {
#     max-width: 1250px;
#     margin: 25px auto;
#     padding: 0 20px;
# }

# .panel {
#     background: #1f2937;
#     padding: 20px;
#     border-radius: 12px;
#     margin-bottom: 20px;
# }

# .controls {
#     display: flex;
#     flex-wrap: wrap;
#     gap: 12px;
#     align-items: center;
# }

# select,
# input[type=file],
# button {
#     padding: 10px 14px;
#     border-radius: 7px;
#     border: 1px solid #4b5563;
#     background: #374151;
#     color: white;
#     font-size: 15px;
# }

# button {
#     cursor: pointer;
# }

# button:hover {
#     background: #4b5563;
# }

# .upload-btn {
#     background: #2563eb;
# }

# .zone-btn {
#     background: #059669;
# }

# .clear-btn {
#     background: #dc2626;
# }

# .info {
#     margin-top: 15px;
#     color: #d1d5db;
# }

# .stats {
#     display: flex;
#     gap: 20px;
#     flex-wrap: wrap;
#     margin-top: 15px;
# }

# .stat {
#     background: #111827;
#     padding: 15px 25px;
#     border-radius: 10px;
#     min-width: 150px;
# }

# .stat-title {
#     color: #9ca3af;
#     font-size: 14px;
# }

# .stat-value {
#     font-size: 30px;
#     font-weight: bold;
#     margin-top: 5px;
# }

# .image-container {
#     margin-top: 20px;
#     position: relative;
#     display: inline-block;
#     max-width: 100%;
# }

# .image-container img {
#     max-width: 100%;
#     display: block;
#     border-radius: 10px;
# }

# #canvas {
#     position: absolute;
#     left: 0;
#     top: 0;
#     cursor: crosshair;
# }

# .zone-status {
#     margin-top: 10px;
#     color: #34d399;
#     font-weight: bold;
# }

# </style>

# </head>


# <body>


# <div class="header">

#     <h1>Zone Counting AI</h1>

#     <p>
#         YOLO26L Segmentation • Zone Restricted Counting
#     </p>

# </div>


# <div class="container">


#     <!-- =====================================================
#          UPLOAD
#          ===================================================== -->

#     <div class="panel">

#         <h2>1. Upload Image</h2>

#         <form
#             method="POST"
#             enctype="multipart/form-data"
#         >

#             <input
#                 type="file"
#                 name="file"
#                 accept="image/*"
#                 required
#             >

#             <button
#                 class="upload-btn"
#                 type="submit"
#             >
#                 Upload & Detect
#             </button>

#         </form>

#     </div>


#     {% if image_file %}


#     <!-- =====================================================
#          ZONE CONTROLS
#          ===================================================== -->

#     <div class="panel">

#         <h2>2. Configure Counting Zone</h2>

#         <div class="controls">

#             <label>
#                 Zone:
#             </label>

#             <select id="zoneSelect">

#                 <option value="zone1">
#                     Zone 1
#                 </option>

#                 <option value="zone2">
#                     Zone 2
#                 </option>

#                 <option value="zone3">
#                     Zone 3
#                 </option>

#             </select>


#             <button
#                 class="zone-btn"
#                 onclick="startDrawing()"
#             >
#                 Draw Zone
#             </button>


#             <button
#                 class="clear-btn"
#                 onclick="clearZone()"
#             >
#                 Clear Zone
#             </button>

#         </div>


#         <div class="zone-status" id="zoneStatus">

#             No zone selected.
#             Click "Draw Zone" and drag on the image.

#         </div>


#         <div class="image-container">

#             <img
#                 id="sourceImage"
#                 src="{{ url_for('output_file', filename=image_file) }}"
#             >

#             <canvas id="canvas"></canvas>

#         </div>

#     </div>


#     <!-- =====================================================
#          RESULTS
#          ===================================================== -->

#     <div class="panel">

#         <h2>3. Counting Result</h2>

#         <div class="stats">

#             <div class="stat">

#                 <div class="stat-title">
#                     Total Detected
#                 </div>

#                 <div
#                     class="stat-value"
#                     id="totalCount"
#                 >
#                     {{ total_count }}
#                 </div>

#             </div>


#             <div class="stat">

#                 <div class="stat-title">
#                     Zone Count
#                 </div>

#                 <div
#                     class="stat-value"
#                     id="zoneCount"
#                 >
#                     0
#                 </div>

#             </div>


#             <div class="stat">

#                 <div class="stat-title">
#                     Outside Zone
#                 </div>

#                 <div
#                     class="stat-value"
#                     id="outsideCount"
#                 >
#                     {{ total_count }}
#                 </div>

#             </div>

#         </div>

#     </div>


#     {% endif %}


# </div>


# <script>


# // ============================================================
# // VARIABLES
# // ============================================================

# let canvas =
#     document.getElementById("canvas");

# let image =
#     document.getElementById("sourceImage");

# let ctx;

# let drawing = false;

# let startX = 0;
# let startY = 0;

# let zone = null;

# let currentZone =
#     document.getElementById("zoneSelect");


# // ============================================================
# // INITIALIZE CANVAS
# // ============================================================

# if (canvas && image) {

#     image.onload = function() {

#         canvas.width = image.clientWidth;
#         canvas.height = image.clientHeight;

#         ctx = canvas.getContext("2d");

#     };

# }


# // ============================================================
# // START DRAWING
# // ============================================================

# function startDrawing() {

#     if (!canvas) {
#         return;
#     }

#     zone = null;

#     ctx.clearRect(
#         0,
#         0,
#         canvas.width,
#         canvas.height
#     );

#     document.getElementById("zoneStatus").innerText =
#         "Draw your counting zone by dragging on the image.";

#     canvas.onmousedown = function(e) {

#         drawing = true;

#         const rect =
#             canvas.getBoundingClientRect();

#         startX =
#             e.clientX - rect.left;

#         startY =
#             e.clientY - rect.top;

#     };


#     canvas.onmousemove = function(e) {

#         if (!drawing) {
#             return;
#         }

#         const rect =
#             canvas.getBoundingClientRect();

#         let currentX =
#             e.clientX - rect.left;

#         let currentY =
#             e.clientY - rect.top;

#         ctx.clearRect(
#             0,
#             0,
#             canvas.width,
#             canvas.height
#         );

#         ctx.strokeStyle = "#00ff88";

#         ctx.lineWidth = 3;

#         ctx.strokeRect(
#             startX,
#             startY,
#             currentX - startX,
#             currentY - startY
#         );

#     };


#     canvas.onmouseup = function(e) {

#         if (!drawing) {
#             return;
#         }

#         drawing = false;

#         const rect =
#             canvas.getBoundingClientRect();

#         let endX =
#             e.clientX - rect.left;

#         let endY =
#             e.clientY - rect.top;


#         let x1 =
#             Math.min(startX, endX);

#         let y1 =
#             Math.min(startY, endY);

#         let x2 =
#             Math.max(startX, endX);

#         let y2 =
#             Math.max(startY, endY);


#         zone = {
#             x1: x1,
#             y1: y1,
#             x2: x2,
#             y2: y2
#         };


#         ctx.clearRect(
#             0,
#             0,
#             canvas.width,
#             canvas.height
#         );


#         ctx.strokeStyle = "#00ff88";

#         ctx.lineWidth = 4;

#         ctx.strokeRect(
#             x1,
#             y1,
#             x2 - x1,
#             y2 - y1
#         );


#         document.getElementById("zoneStatus").innerText =
#             currentZone.options[
#                 currentZone.selectedIndex
#             ].text
#             + " selected.";

#         calculateZoneCount();

#     };

# }


# // ============================================================
# // CLEAR ZONE
# // ============================================================

# function clearZone() {

#     zone = null;

#     if (ctx) {

#         ctx.clearRect(
#             0,
#             0,
#             canvas.width,
#             canvas.height
#         );

#     }

#     document.getElementById("zoneCount").innerText = "0";

#     document.getElementById("outsideCount").innerText =
#         "{{ total_count }}";

#     document.getElementById("zoneStatus").innerText =
#         "No zone selected.";

# }


# // ============================================================
# // ZONE COUNT
# // ============================================================

# function calculateZoneCount() {

#     if (!zone) {
#         return;
#     }

#     const detections =
#         {{ detections | safe }};

#     let count = 0;


#     detections.forEach(function(det) {

#         let cx = det.cx;
#         let cy = det.cy;


#         if (
#             cx >= zone.x1 &&
#             cx <= zone.x2 &&
#             cy >= zone.y1 &&
#             cy <= zone.y2
#         ) {

#             count++;

#         }

#     });


#     document.getElementById("zoneCount").innerText =
#         count;

#     document.getElementById("outsideCount").innerText =
#         {{ total_count }} - count;

# }


# </script>


# </body>

# </html>
# """


# # ============================================================
# # INFERENCE
# # ============================================================

# def run_inference(image_path, output_path):

#     image = cv2.imread(str(image_path))

#     if image is None:
#         raise ValueError("Unable to read image.")

#     original_h, original_w = image.shape[:2]


#     # --------------------------------------------------------
#     # YOLO INFERENCE
#     # --------------------------------------------------------

#     results = model.predict(
#         source=image,
#         imgsz=1280,
#         conf=0.25,
#         iou=0.50,
#         device=0,
#         verbose=False
#     )

#     result = results[0]


#     detections = []


#     # --------------------------------------------------------
#     # GET DETECTIONS
#     # --------------------------------------------------------

#     if result.boxes is not None:

#         boxes = (
#             result.boxes.xyxy
#             .detach()
#             .cpu()
#             .numpy()
#         )

#         confidences = (
#             result.boxes.conf
#             .detach()
#             .cpu()
#             .numpy()
#         )


#         for box, conf in zip(
#             boxes,
#             confidences
#         ):

#             x1, y1, x2, y2 = box

#             cx = (x1 + x2) / 2
#             cy = (y1 + y2) / 2


#             detections.append({

#                 "x1": float(x1),
#                 "y1": float(y1),

#                 "x2": float(x2),
#                 "y2": float(y2),

#                 "cx": float(cx),
#                 "cy": float(cy),

#                 "confidence": float(conf)

#             })


#     # --------------------------------------------------------
#     # DRAW MASKS
#     # --------------------------------------------------------

#     output = image.copy()


#     if result.masks is not None:

#         masks = (
#             result.masks.data
#             .detach()
#             .cpu()
#             .numpy()
#         )


#         for mask in masks:

#             mask = cv2.resize(
#                 mask,
#                 (
#                     original_w,
#                     original_h
#                 ),
#                 interpolation=cv2.INTER_NEAREST
#             )


#             binary =   mask > 0.5


#             overlay = output.copy()


#             overlay[binary] = (
#                 overlay[binary] * 0.45
#                 +
#                 np.array(
#                     [0, 255, 0]
#                 ) * 0.55
#             ).astype(np.uint8)


#             output = overlay


#             contours, _ = cv2.findContours(
#                 binary.astype(np.uint8),
#                 cv2.RETR_EXTERNAL,
#                 cv2.CHAIN_APPROX_SIMPLE
#             )


#             cv2.drawContours(
#                 output,
#                 contours,
#                 -1,
#                 (0, 255, 0),
#                 2
#             )


#     # --------------------------------------------------------
#     # TOTAL COUNT
#     # --------------------------------------------------------

#     total_count = len(detections)


#     cv2.rectangle(
#         output,
#         (15, 15),
#         (360, 75),
#         (0, 0, 0),
#         -1
#     )


#     cv2.putText(
#         output,
#         f"Detected: {total_count}",
#         (30, 55),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         1.0,
#         (255, 255, 255),
#         2,
#         cv2.LINE_AA
#     )


#     # --------------------------------------------------------
#     # SAVE
#     # --------------------------------------------------------

#     cv2.imwrite(
#         str(output_path),
#         output
#     )


#     return detections, total_count


# # ============================================================
# # ROUTE
# # ============================================================

# @app.route("/", methods=["GET", "POST"])
# def index():

#     if request.method == "POST":

#         uploaded_file =    request.files.get("file")


#         if uploaded_file is None:
#             return "No file uploaded."


#         if uploaded_file.filename == "":
#             return "No file selected."


#         uid =   uuid.uuid4().hex


#         original_name = Path(
#                 uploaded_file.filename
#             ).name


#         input_name =   f"{uid}_{original_name}"


#         output_name = f"{uid}_result.jpg"

#         input_path =UPLOAD_DIR / input_name


#         output_path =OUTPUT_DIR / output_name


#         uploaded_file.save(
#             input_path
#         )


#         try:

#             detections, total_count =run_inference(
#                     input_path,
#                     output_path
#                 )

#         except Exception as e:

#             return f"""
#             <h2>Inference Error</h2>
#             <pre>{e}</pre>
#             """


#         return render_template_string(
#             HTML,

#             image_file=output_name,

#             total_count=total_count,

#             detections=detections,

#             model_path=str(
#                 MODEL_PATH
#             )
#         )


#     return render_template_string(
#         HTML,

#         image_file=None,

#         total_count=0,

#         detections=[],

#         model_path=str(
#             MODEL_PATH
#         )
#     )


# # ============================================================
# # OUTPUT FILE
# # ============================================================

# @app.route("/outputs/<filename>")
# def output_file(filename):

#     return send_from_directory(
#         OUTPUT_DIR,
#         filename
#     )


# # ============================================================
# # START
# # ============================================================

# if __name__ == "__main__":

#     print("\n" + "=" * 70)
#     print("ZONE COUNTING UI")
#     print("=" * 70)

#     print("\nOpen:")
#     print("http://192.168.2.240:5001")

#     print("\nPress CTRL+C to stop.")

#     app.run(
#         host="0.0.0.0",
#         port=5004,
#         debug=False
#     )

###############################################################################################################################
###############################################################################################################################
###############################################################################################################################
#2nd bakcup


# #starting with -Upload Image
#      ↓
# Image appears WITHOUT detections
#      ↓
# Draw Zone
#      ↓
# Zone rectangle appears
#      ↓
# [ Detect in Zone ]
#      ↓
# YOLO inference
#      ↓
# Only masks belonging to the zone are rendered

# from flask import Flask, request, render_template_string, jsonify, send_from_directory
# from ultralytics import YOLO
# import os
# import uuid
# import cv2
# import numpy as np

# # ============================================================
# # CONFIGURATION
# # ============================================================

# BASE_DIR = os.path.expanduser("~/ZONE_COUNTING")

# MODEL_PATH = os.path.join(
#     BASE_DIR,
#     "runs",
#     "segment",
#     "zone_counting_yolo26l_seg_v1_1280",
#     "weights",
#     "best.pt"
# )

# UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
# OUTPUT_DIR = os.path.join(BASE_DIR, "zone_outputs")

# os.makedirs(UPLOAD_DIR, exist_ok=True)
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# HOST = "0.0.0.0"
# PORT = 5004

# IMAGE_SIZE = 1280
# CONF_THRESHOLD = 0.25
# IOU_THRESHOLD = 0.50

# # Minimum percentage of an object's mask that must be inside
# # the selected zone to count it.
# MASK_OVERLAP_THRESHOLD = 0.05


# # ============================================================
# # FLASK APP
# # ============================================================

# app = Flask(__name__)


# # ============================================================
# # LOAD MODEL
# # ============================================================

# print("=" * 70)
# print("ZONE COUNTING - MASK BASED ZONE UI")
# print("=" * 70)

# print(f"Model: {MODEL_PATH}")

# if not os.path.exists(MODEL_PATH):
#     raise FileNotFoundError(
#         f"\nModel not found:\n{MODEL_PATH}\n"
#         f"\nPlease verify your best.pt path."
#     )

# model = YOLO(MODEL_PATH)

# print("Model loaded successfully.")
# print("=" * 70)


# # ============================================================
# # HTML
# # ============================================================

# HTML = r"""
# <!DOCTYPE html>
# <html>
# <head>

#     <title>Zone Counting - Mask Based</title>

#     <meta name="viewport" content="width=device-width, initial-scale=1.0">

#     <style>

#         * {
#             box-sizing: border-box;
#         }

#         body {
#             margin: 0;
#             font-family: Arial, Helvetica, sans-serif;
#             background: #111827;
#             color: white;
#         }

#         .header {
#             padding: 18px 25px;
#             background: #1f2937;
#             border-bottom: 1px solid #374151;
#         }

#         .header h1 {
#             margin: 0;
#             font-size: 25px;
#         }

#         .header p {
#             margin: 6px 0 0;
#             color: #9ca3af;
#         }

#         .container {
#             display: flex;
#             min-height: calc(100vh - 85px);
#         }

#         .sidebar {
#             width: 300px;
#             padding: 22px;
#             background: #1f2937;
#             border-right: 1px solid #374151;
#         }

#         .main {
#             flex: 1;
#             padding: 25px;
#             overflow: auto;
#         }

#         .section {
#             margin-bottom: 25px;
#         }

#         .section-title {
#             font-size: 14px;
#             color: #9ca3af;
#             margin-bottom: 10px;
#             text-transform: uppercase;
#             letter-spacing: 0.5px;
#         }

#         input[type="file"],
#         select {
#             width: 100%;
#             padding: 11px;
#             border-radius: 8px;
#             border: 1px solid #4b5563;
#             background: #111827;
#             color: white;
#             margin-bottom: 12px;
#         }

#         button {
#             width: 100%;
#             padding: 12px;
#             margin-bottom: 10px;
#             border: none;
#             border-radius: 8px;
#             cursor: pointer;
#             font-weight: bold;
#             font-size: 14px;
#         }

#         .btn-primary {
#             background: #2563eb;
#             color: white;
#         }

#         .btn-primary:hover {
#             background: #1d4ed8;
#         }

#         .btn-success {
#             background: #059669;
#             color: white;
#         }

#         .btn-success:hover {
#             background: #047857;
#         }

#         .btn-danger {
#             background: #dc2626;
#             color: white;
#         }

#         .btn-danger:hover {
#             background: #b91c1c;
#         }

#         .btn-secondary {
#             background: #374151;
#             color: white;
#         }

#         .btn-secondary:hover {
#             background: #4b5563;
#         }

#         button:disabled {
#             opacity: 0.4;
#             cursor: not-allowed;
#         }

#         .info-box {
#             padding: 14px;
#             border-radius: 8px;
#             background: #111827;
#             border: 1px solid #374151;
#             margin-top: 15px;
#         }

#         .status {
#             font-size: 13px;
#             line-height: 1.5;
#             color: #d1d5db;
#         }

#         .count-box {
#             text-align: center;
#             padding: 18px;
#             border-radius: 10px;
#             background: #111827;
#             border: 1px solid #374151;
#             margin-top: 15px;
#         }

#         .count-label {
#             font-size: 13px;
#             color: #9ca3af;
#         }

#         .count {
#             font-size: 42px;
#             font-weight: bold;
#             margin-top: 5px;
#         }

#         .viewer {
#             width: 100%;
#             display: flex;
#             justify-content: center;
#             align-items: flex-start;
#         }

#         #canvasContainer {
#             position: relative;
#             display: inline-block;
#             max-width: 100%;
#             background: #000;
#             border-radius: 10px;
#             overflow: hidden;
#         }

#         #mainImage {
#             display: block;
#             max-width: 100%;
#             max-height: 78vh;
#             width: auto;
#             height: auto;
#         }

#         #zoneCanvas {
#             position: absolute;
#             top: 0;
#             left: 0;
#             cursor: crosshair;
#         }

#         .hint {
#             margin-top: 12px;
#             text-align: center;
#             color: #9ca3af;
#             font-size: 13px;
#         }

#         .badge {
#             display: inline-block;
#             padding: 5px 9px;
#             border-radius: 20px;
#             background: #374151;
#             font-size: 12px;
#             margin-top: 8px;
#         }

#         .loading {
#             display: none;
#             text-align: center;
#             padding: 15px;
#             color: #60a5fa;
#         }

#         @media (max-width: 800px) {

#             .container {
#                 flex-direction: column;
#             }

#             .sidebar {
#                 width: 100%;
#                 border-right: none;
#                 border-bottom: 1px solid #374151;
#             }

#             .main {
#                 padding: 15px;
#             }

#         }

#     </style>

# </head>

# <body>

# <div class="header">

#     <h1>🎯 Zone Counting — Mask Based</h1>

#     <p>
#         Draw a zone first. Detection will appear only inside the selected zone.
#     </p>

# </div>


# <div class="container">


#     <!-- ======================================================
#          SIDEBAR
#          ====================================================== -->

#     <div class="sidebar">

#         <div class="section">

#             <div class="section-title">
#                 1. Upload Image
#             </div>

#             <input
#                 type="file"
#                 id="imageInput"
#                 accept="image/*"
#             >

#         </div>


#         <div class="section">

#             <div class="section-title">
#                 2. Select Zone
#             </div>

#             <select id="zoneSelect">

#                 <option value="Zone 1">
#                     Zone 1
#                 </option>

#                 <option value="Zone 2">
#                     Zone 2
#                 </option>

#                 <option value="Zone 3">
#                     Zone 3
#                 </option>

#             </select>


#             <button
#                 class="btn-primary"
#                 id="drawButton"
#                 disabled
#                 onclick="startDrawing()"
#             >
#                 ✏️ Draw / Redraw Zone
#             </button>


#             <button
#                 class="btn-secondary"
#                 id="clearButton"
#                 disabled
#                 onclick="clearCurrentZone()"
#             >
#                 🗑 Clear Zone
#             </button>

#         </div>


#         <div class="section">

#             <div class="section-title">
#                 3. Detection
#             </div>

#             <button
#                 class="btn-success"
#                 id="detectButton"
#                 disabled
#                 onclick="detectInZone()"
#             >
#                 🔍 Detect in Zone
#             </button>

#         </div>


#         <div class="info-box">

#             <div class="status" id="status">

#                 Upload an image to begin.

#             </div>

#         </div>


#         <div class="count-box">

#             <div class="count-label">
#                 CURRENT ZONE COUNT
#             </div>

#             <div class="count" id="count">
#                 0
#             </div>

#             <div
#                 class="badge"
#                 id="zoneName"
#             >
#                 Zone 1
#             </div>

#         </div>

#     </div>


#     <!-- ======================================================
#          IMAGE VIEWER
#          ====================================================== -->

#     <div class="main">

#         <div class="viewer">

#             <div id="canvasContainer">

#                 <img
#                     id="mainImage"
#                     src=""
#                     style="display:none;"
#                 >

#                 <canvas
#                     id="zoneCanvas"
#                 ></canvas>

#             </div>

#         </div>


#         <div
#             class="hint"
#             id="hint"
#         >
#             Upload an image. No detection will be performed until
#             you draw a zone and click Detect in Zone.
#         </div>


#         <div
#             class="loading"
#             id="loading"
#         >
#             ⏳ Running YOLO segmentation...
#         </div>

#     </div>

# </div>


# <script>

#     // =========================================================
#     // GLOBAL STATE
#     // =========================================================

#     const imageInput = document.getElementById("imageInput");

#     const image = document.getElementById("mainImage");

#     const canvas = document.getElementById("zoneCanvas");

#     const ctx = canvas.getContext("2d");

#     const zoneSelect = document.getElementById("zoneSelect");

#     const drawButton = document.getElementById("drawButton");

#     const clearButton = document.getElementById("clearButton");

#     const detectButton = document.getElementById("detectButton");

#     const statusBox = document.getElementById("status");

#     const countBox = document.getElementById("count");

#     const zoneNameBox = document.getElementById("zoneName");

#     const hint = document.getElementById("hint");

#     const loading = document.getElementById("loading");


#     let uploadedFilename = null;

#     let isDrawing = false;

#     let startX = 0;

#     let startY = 0;

#     let currentZone = null;

#     let drawingEnabled = false;


#     // Store separate zones

#     const zones = {

#         "Zone 1": null,

#         "Zone 2": null,

#         "Zone 3": null

#     };


#     // =========================================================
#     // IMAGE UPLOAD
#     // =========================================================

#     imageInput.addEventListener("change", function() {

#         const file = this.files[0];

#         if (!file) {
#             return;
#         }

#         const formData = new FormData();

#         formData.append("image", file);


#         statusBox.innerHTML =
#             "Uploading image...";


#         fetch("/upload", {

#             method: "POST",

#             body: formData

#         })

#         .then(response => response.json())

#         .then(data => {

#             if (!data.success) {

#                 statusBox.innerHTML =
#                     "❌ " + data.error;

#                 return;
#             }


#             uploadedFilename = data.filename;


#             // Show RAW image only.

#             image.src =
#                 "/uploads/" + uploadedFilename;

#             image.style.display = "block";


#             image.onload = function() {

#                 resizeCanvas();

#                 drawSavedZone();

#             };


#             // Enable zone tools

#             drawButton.disabled = false;

#             clearButton.disabled = false;


#             // Detection remains disabled

#             detectButton.disabled = true;


#             countBox.innerText = "0";


#             statusBox.innerHTML =
#                 "Image uploaded.<br>" +
#                 "Draw a zone before detection.";


#             hint.innerText =
#                 "Draw a rectangle around the area you want to count.";

#         })

#         .catch(error => {

#             console.error(error);

#             statusBox.innerHTML =
#                 "❌ Upload failed.";

#         });

#     });


#     // =========================================================
#     // CANVAS RESIZE
#     // =========================================================

#     function resizeCanvas() {

#         canvas.width = image.clientWidth;

#         canvas.height = image.clientHeight;

#         canvas.style.width =
#             image.clientWidth + "px";

#         canvas.style.height =
#             image.clientHeight + "px";

#     }


#     window.addEventListener(
#         "resize",
#         function() {

#             if (image.style.display !== "none") {

#                 resizeCanvas();

#                 drawSavedZone();

#             }

#         }
#     );


#     // =========================================================
#     // ZONE SELECTION
#     // =========================================================

#     zoneSelect.addEventListener("change", function() {

#         const selectedZone =
#             zoneSelect.value;

#         zoneNameBox.innerText =
#             selectedZone;


#         // IMPORTANT:
#         // Whenever zone changes, show RAW image.
#         // Do not show previous detection.

#         resetDetectionView();


#         drawSavedZone();


#         if (zones[selectedZone]) {

#             statusBox.innerHTML =
#                 selectedZone +
#                 " already exists.<br>" +
#                 "Click Detect in Zone to run detection.";

#             detectButton.disabled = false;

#         } else {

#             statusBox.innerHTML =
#                 "No zone drawn for " +
#                 selectedZone +
#                 ".<br>" +
#                 "Draw a zone first.";

#             detectButton.disabled = true;

#         }

#     });


#     // =========================================================
#     // START DRAWING
#     // =========================================================

#     function startDrawing() {

#         if (!uploadedFilename) {

#             alert("Please upload an image first.");

#             return;

#         }


#         drawingEnabled = true;


#         // Reset any previous detection

#         resetDetectionView();


#         statusBox.innerHTML =
#             "✏️ Draw a rectangle on the image.";


#         hint.innerText =
#             "Click and drag to create the zone.";


#         detectButton.disabled = true;


#         clearCanvasOnly();

#     }


#     // =========================================================
#     // MOUSE EVENTS
#     // =========================================================

#     canvas.addEventListener(
#         "mousedown",
#         function(event) {

#             if (!drawingEnabled) {
#                 return;
#             }


#             const rect =
#                 canvas.getBoundingClientRect();


#             startX =
#                 event.clientX - rect.left;

#             startY =
#                 event.clientY - rect.top;


#             isDrawing = true;

#         }
#     );


#     canvas.addEventListener(
#         "mousemove",
#         function(event) {

#             if (!isDrawing) {
#                 return;
#             }


#             const rect =
#                 canvas.getBoundingClientRect();


#             const currentX =
#                 event.clientX - rect.left;

#             const currentY =
#                 event.clientY - rect.top;


#             clearCanvasOnly();


#             // Draw temporary rectangle

#             ctx.strokeStyle = "#00ff88";

#             ctx.lineWidth = 3;

#             ctx.setLineDash([8, 5]);


#             ctx.strokeRect(

#                 startX,
#                 startY,
#                 currentX - startX,
#                 currentY - startY

#             );


#             ctx.setLineDash([]);

#         }
#     );


#     canvas.addEventListener(
#         "mouseup",
#         function(event) {

#             if (!isDrawing) {
#                 return;
#             }


#             const rect =
#                 canvas.getBoundingClientRect();


#             let endX =
#                 event.clientX - rect.left;

#             let endY =
#                 event.clientY - rect.top;


#             isDrawing = false;

#             drawingEnabled = false;


#             // Normalize coordinates

#             let x1 =
#                 Math.min(startX, endX);

#             let y1 =
#                 Math.min(startY, endY);

#             let x2 =
#                 Math.max(startX, endX);

#             let y2 =
#                 Math.max(startY, endY);


#             // Ignore tiny rectangles

#             if (
#                 Math.abs(x2 - x1) < 10 ||
#                 Math.abs(y2 - y1) < 10
#             ) {

#                 clearCanvasOnly();

#                 statusBox.innerHTML =
#                     "Zone too small. Draw again.";

#                 detectButton.disabled = true;

#                 return;

#             }


#             const selectedZone =
#                 zoneSelect.value;


#             zones[selectedZone] = {

#                 x1: x1,
#                 y1: y1,
#                 x2: x2,
#                 y2: y2

#             };


#             currentZone =
#                 zones[selectedZone];


#             drawSavedZone();


#             detectButton.disabled = false;


#             statusBox.innerHTML =
#                 "✅ " +
#                 selectedZone +
#                 " created.<br>" +
#                 "Click Detect in Zone.";

#             hint.innerText =
#                 "Only masks overlapping this zone will be displayed.";

#         }
#     );


#     // =========================================================
#     // TOUCH SUPPORT
#     // =========================================================

#     canvas.addEventListener(
#         "touchstart",
#         function(event) {

#             if (!drawingEnabled) {
#                 return;
#             }

#             event.preventDefault();

#             const touch =
#                 event.touches[0];

#             const rect =
#                 canvas.getBoundingClientRect();


#             startX =
#                 touch.clientX - rect.left;

#             startY =
#                 touch.clientY - rect.top;

#             isDrawing = true;

#         },
#         { passive: false }
#     );


#     canvas.addEventListener(
#         "touchmove",
#         function(event) {

#             if (!isDrawing) {
#                 return;
#             }

#             event.preventDefault();

#             const touch =
#                 event.touches[0];

#             const rect =
#                 canvas.getBoundingClientRect();


#             const currentX =
#                 touch.clientX - rect.left;

#             const currentY =
#                 touch.clientY - rect.top;


#             clearCanvasOnly();


#             ctx.strokeStyle = "#00ff88";

#             ctx.lineWidth = 3;

#             ctx.setLineDash([8, 5]);


#             ctx.strokeRect(

#                 startX,
#                 startY,
#                 currentX - startX,
#                 currentY - startY

#             );


#             ctx.setLineDash([]);

#         },
#         { passive: false }
#     );


#     canvas.addEventListener(
#         "touchend",
#         function(event) {

#             if (!isDrawing) {
#                 return;
#             }

#             event.preventDefault();


#             const touch =
#                 event.changedTouches[0];

#             const rect =
#                 canvas.getBoundingClientRect();


#             let endX =
#                 touch.clientX - rect.left;

#             let endY =
#                 touch.clientY - rect.top;


#             isDrawing = false;

#             drawingEnabled = false;


#             let x1 =
#                 Math.min(startX, endX);

#             let y1 =
#                 Math.min(startY, endY);

#             let x2 =
#                 Math.max(startX, endX);

#             let y2 =
#                 Math.max(startY, endY);


#             if (
#                 Math.abs(x2 - x1) < 10 ||
#                 Math.abs(y2 - y1) < 10
#             ) {

#                 clearCanvasOnly();

#                 statusBox.innerHTML =
#                     "Zone too small. Draw again.";

#                 detectButton.disabled = true;

#                 return;

#             }


#             const selectedZone =
#                 zoneSelect.value;


#             zones[selectedZone] = {

#                 x1: x1,
#                 y1: y1,
#                 x2: x2,
#                 y2: y2

#             };


#             currentZone =
#                 zones[selectedZone];


#             drawSavedZone();


#             detectButton.disabled = false;


#             statusBox.innerHTML =
#                 "✅ " +
#                 selectedZone +
#                 " created.<br>" +
#                 "Click Detect in Zone.";

#         },
#         { passive: false }
#     );


#     // =========================================================
#     // DRAW SAVED ZONE
#     // =========================================================

#     function drawSavedZone() {

#         clearCanvasOnly();


#         const selectedZone =
#             zoneSelect.value;


#         const zone =
#             zones[selectedZone];


#         if (!zone) {
#             return;
#         }


#         ctx.fillStyle =
#             "rgba(0, 255, 136, 0.10)";

#         ctx.fillRect(

#             zone.x1,
#             zone.y1,
#             zone.x2 - zone.x1,
#             zone.y2 - zone.y1

#         );


#         ctx.strokeStyle =
#             "#00ff88";

#         ctx.lineWidth = 3;


#         ctx.strokeRect(

#             zone.x1,
#             zone.y1,
#             zone.x2 - zone.x1,
#             zone.y2 - zone.y1

#         );


#         ctx.fillStyle =
#             "#00ff88";

#         ctx.font =
#             "bold 16px Arial";


#         ctx.fillText(

#             selectedZone,

#             zone.x1 + 8,
#             zone.y1 + 22

#         );

#     }


#     // =========================================================
#     // CLEAR CURRENT ZONE
#     // =========================================================

#     function clearCurrentZone() {

#         const selectedZone =
#             zoneSelect.value;


#         zones[selectedZone] = null;

#         currentZone = null;


#         resetDetectionView();

#         clearCanvasOnly();


#         detectButton.disabled = true;


#         statusBox.innerHTML =
#             selectedZone +
#             " cleared.<br>" +
#             "Draw a new zone.";

#     }


#     // =========================================================
#     // CLEAR CANVAS
#     // =========================================================

#     function clearCanvasOnly() {

#         ctx.clearRect(

#             0,
#             0,
#             canvas.width,
#             canvas.height

#         );

#     }


#     // =========================================================
#     // RESET DETECTION VIEW
#     // =========================================================

#     function resetDetectionView() {

#         if (!uploadedFilename) {
#             return;
#         }


#         // Return to original RAW image

#         image.src =
#             "/uploads/" +
#             uploadedFilename;


#         countBox.innerText = "0";

#     }


#     // =========================================================
#     // DETECT IN ZONE
#     // =========================================================

#     async function detectInZone() {

#         if (!uploadedFilename) {

#             alert("Please upload an image.");

#             return;

#         }


#         const selectedZone =
#             zoneSelect.value;


#         const zone =
#             zones[selectedZone];


#         if (!zone) {

#             alert(
#                 "Please draw " +
#                 selectedZone +
#                 " first."
#             );

#             return;

#         }


#         // Disable button during inference

#         detectButton.disabled = true;


#         loading.style.display = "block";


#         statusBox.innerHTML =
#             "Running segmentation...";


#         hint.innerText =
#             "YOLO is detecting objects and filtering them by mask overlap.";


#         // =====================================================
#         // Convert DISPLAY coordinates -> ORIGINAL IMAGE pixels
#         // =====================================================

#         const displayWidth =
#             image.clientWidth;

#         const displayHeight =
#             image.clientHeight;


#         const originalWidth =
#             image.naturalWidth;

#         const originalHeight =
#             image.naturalHeight;


#         const scaleX =
#             originalWidth / displayWidth;

#         const scaleY =
#             originalHeight / displayHeight;


#         const originalZone = {

#             x1: Math.round(zone.x1 * scaleX),

#             y1: Math.round(zone.y1 * scaleY),

#             x2: Math.round(zone.x2 * scaleX),

#             y2: Math.round(zone.y2 * scaleY)

#         };


#         const payload = {

#             filename: uploadedFilename,

#             zone_name: selectedZone,

#             x1: originalZone.x1,

#             y1: originalZone.y1,

#             x2: originalZone.x2,

#             y2: originalZone.y2

#         };


#         try {

#             const response =
#                 await fetch(
#                     "/detect",
#                     {
#                         method: "POST",

#                         headers: {
#                             "Content-Type":
#                                 "application/json"
#                         },

#                         body:
#                             JSON.stringify(payload)
#                     }
#                 );


#             const data =
#                 await response.json();


#             if (!data.success) {

#                 statusBox.innerHTML =
#                     "❌ " + data.error;

#                 return;

#             }


#             // Show ONLY filtered result

#             image.src =
#                 "/outputs/" +
#                 data.output_filename;


#             countBox.innerText =
#                 data.count;


#             zoneNameBox.innerText =
#                 selectedZone;


#             if (data.count > 0) {

#                 statusBox.innerHTML =
#                     "✅ Detection complete.<br>" +
#                     "Only masks overlapping " +
#                     selectedZone +
#                     " are shown.";

#             } else {

#                 statusBox.innerHTML =
#                     "ℹ️ No detected object found inside " +
#                     selectedZone +
#                     ".";

#             }


#             hint.innerText =
#                 "Outside-zone detections are hidden.";


#             image.onload = function() {

#                 resizeCanvas();

#                 drawSavedZone();

#             };

#         }

#         catch (error) {

#             console.error(error);

#             statusBox.innerHTML =
#                 "❌ Detection request failed.";

#         }

#         finally {

#             loading.style.display =
#                 "none";

#             detectButton.disabled = false;

#         }

#     }

# </script>

# </body>
# </html>
# """


# # ============================================================
# # HOME
# # ============================================================

# @app.route("/")
# def index():

#     return render_template_string(HTML)


# # ============================================================
# # UPLOAD
# # ============================================================

# @app.route("/upload", methods=["POST"])
# def upload():

#     if "image" not in request.files:

#         return jsonify({
#             "success": False,
#             "error": "No image uploaded."
#         })


#     file = request.files["image"]


#     if file.filename == "":

#         return jsonify({
#             "success": False,
#             "error": "No file selected."
#         })


#     # Generate unique filename

#     extension = os.path.splitext(
#         file.filename
#     )[1].lower()


#     allowed_extensions = {

#         ".jpg",
#         ".jpeg",
#         ".png",
#         ".bmp",
#         ".webp"

#     }


#     if extension not in allowed_extensions:

#         return jsonify({
#             "success": False,
#             "error": "Unsupported image format."
#         })


#     unique_name = (
#         str(uuid.uuid4()) +
#         extension
#     )


#     save_path = os.path.join(
#         UPLOAD_DIR,
#         unique_name
#     )


#     file.save(save_path)


#     print(
#         f"[UPLOAD] {save_path}"
#     )


#     return jsonify({

#         "success": True,

#         "filename": unique_name

#     })


# # ============================================================
# # SERVE UPLOADED IMAGE
# # ============================================================

# @app.route("/uploads/<filename>")
# def uploaded_file(filename):

#     return send_from_directory(
#         UPLOAD_DIR,
#         filename
#     )


# # ============================================================
# # DETECTION
# # ============================================================

# @app.route("/detect", methods=["POST"])
# def detect():

#     try:

#         data = request.get_json()


#         filename = data.get(
#             "filename"
#         )

#         zone_name = data.get(
#             "zone_name",
#             "Zone 1"
#         )


#         x1 = int(data.get("x1", 0))
#         y1 = int(data.get("y1", 0))
#         x2 = int(data.get("x2", 0))
#         y2 = int(data.get("y2", 0))


#         # ----------------------------------------------------
#         # Validate image
#         # ----------------------------------------------------

#         image_path = os.path.join(
#             UPLOAD_DIR,
#             filename
#         )


#         if not os.path.exists(image_path):

#             return jsonify({

#                 "success": False,

#                 "error":
#                     "Uploaded image not found."

#             })


#         # ----------------------------------------------------
#         # Read image
#         # ----------------------------------------------------

#         frame = cv2.imread(
#             image_path
#         )


#         if frame is None:

#             return jsonify({

#                 "success": False,

#                 "error":
#                     "Unable to read image."

#             })


#         height, width = frame.shape[:2]


#         # ----------------------------------------------------
#         # Clamp zone to image boundaries
#         # ----------------------------------------------------

#         x1 = max(
#             0,
#             min(x1, width - 1)
#         )

#         y1 = max(
#             0,
#             min(y1, height - 1)
#         )

#         x2 = max(
#             0,
#             min(x2, width)
#         )

#         y2 = max(
#             0,
#             min(y2, height)
#         )


#         if x2 <= x1 or y2 <= y1:

#             return jsonify({

#                 "success": False,

#                 "error":
#                     "Invalid zone coordinates."

#             })


#         # ----------------------------------------------------
#         # YOLO SEGMENTATION
#         # ----------------------------------------------------

#         print()
#         print("=" * 70)
#         print(
#             f"[DETECTION] {zone_name}"
#         )
#         print(
#             f"[ZONE] "
#             f"({x1},{y1}) -> ({x2},{y2})"
#         )


#         results = model.predict(

#             source=frame,

#             imgsz=IMAGE_SIZE,

#             conf=CONF_THRESHOLD,

#             iou=IOU_THRESHOLD,

#             device=0,

#             verbose=False

#         )


#         result = results[0]


#         # ----------------------------------------------------
#         # Create zone mask
#         # ----------------------------------------------------

#         zone_mask = np.zeros(

#             (height, width),

#             dtype=np.uint8

#         )


#         zone_mask[
#             y1:y2,
#             x1:x2
#         ] = 1


#         # ----------------------------------------------------
#         # Output image
#         #
#         # IMPORTANT:
#         # We DO NOT use result.plot()
#         #
#         # because result.plot() would draw ALL detections.
#         # ----------------------------------------------------

#         output = frame.copy()


#         selected_count = 0


#         # ----------------------------------------------------
#         # Check segmentation masks
#         # ----------------------------------------------------

#         if result.masks is not None:

#             masks = result.masks.data


#             for mask_tensor in masks:

#                 # Convert tensor -> numpy

#                 mask = (
#                     mask_tensor
#                     .detach()
#                     .cpu()
#                     .numpy()
#                 )


#                 # Resize YOLO mask to original image size

#                 mask = cv2.resize(

#                     mask,

#                     (width, height),

#                     interpolation=cv2.INTER_NEAREST

#                 )


#                 # Binary mask

#                 binary_mask = (
#                     mask > 0.5
#                 ).astype(np.uint8)


#                 # ------------------------------------------------
#                 # Calculate mask area
#                 # ------------------------------------------------

#                 mask_area = int(
#                     binary_mask.sum()
#                 )


#                 if mask_area == 0:
#                     continue


#                 # ------------------------------------------------
#                 # Calculate overlap with zone
#                 # ------------------------------------------------

#                 intersection = np.logical_and(

#                     binary_mask == 1,

#                     zone_mask == 1

#                 )


#                 intersection_pixels = int(
#                     intersection.sum()
#                 )


#                 overlap_ratio = (
#                     intersection_pixels /
#                     float(mask_area)
#                 )


#                 print(
#                     f"[MASK] "
#                     f"area={mask_area}, "
#                     f"zone_overlap="
#                     f"{overlap_ratio:.3f}"
#                 )


#                 # ------------------------------------------------
#                 # FILTER
#                 #
#                 # Only show masks that have meaningful
#                 # overlap with selected zone.
#                 # ------------------------------------------------

#                 if (
#                     overlap_ratio
#                     < MASK_OVERLAP_THRESHOLD
#                 ):

#                     continue


#                 selected_count += 1


#                 # ------------------------------------------------
#                 # Draw ONLY selected mask
#                 # ------------------------------------------------

#                 mask_bool = (
#                     binary_mask.astype(bool)
#                 )


#                 # Create overlay

#                 overlay = output.copy()


#                 # Green mask

#                 overlay[mask_bool] = (
#                     0.35 *
#                     overlay[mask_bool] +
#                     0.65 *
#                     np.array(
#                         [0, 255, 0],
#                         dtype=np.float32
#                     )
#                 ).astype(
#                     np.uint8
#                 )


#                 output = overlay


#                 # ------------------------------------------------
#                 # Draw thin mask boundary
#                 # ------------------------------------------------

#                 contours, _ = cv2.findContours(

#                     binary_mask,

#                     cv2.RETR_EXTERNAL,

#                     cv2.CHAIN_APPROX_SIMPLE

#                 )


#                 cv2.drawContours(

#                     output,

#                     contours,

#                     -1,

#                     (0, 255, 0),

#                     1

#                 )


#         # ----------------------------------------------------
#         # Draw zone rectangle
#         # ----------------------------------------------------

#         cv2.rectangle(

#             output,

#             (x1, y1),

#             (x2, y2),

#             (255, 255, 0),

#             2

#         )


#         # Small zone label only

#         cv2.putText(

#             output,

#             zone_name,

#             (x1 + 8, max(y1 + 25, 25)),

#             cv2.FONT_HERSHEY_SIMPLEX,

#             0.8,

#             (255, 255, 0),

#             2,

#             cv2.LINE_AA

#         )


#         # ----------------------------------------------------
#         # Save output
#         # ----------------------------------------------------

#         output_filename = (

#             "result_" +

#             os.path.splitext(
#                 filename
#             )[0] +

#             ".jpg"

#         )


#         output_path = os.path.join(

#             OUTPUT_DIR,

#             output_filename

#         )


#         cv2.imwrite(

#             output_path,

#             output

#         )


#         print(
#             f"[RESULT] "
#             f"Zone Count = "
#             f"{selected_count}"
#         )

#         print(
#             f"[OUTPUT] "
#             f"{output_path}"
#         )

#         print("=" * 70)


#         return jsonify({

#             "success": True,

#             "count":
#                 selected_count,

#             "zone":
#                 zone_name,

#             "output_filename":
#                 output_filename

#         })


#     except Exception as e:

#         import traceback

#         traceback.print_exc()


#         return jsonify({

#             "success": False,

#             "error":
#                 str(e)

#         })


# # ============================================================
# # SERVE OUTPUT
# # ============================================================

# @app.route("/outputs/<filename>")
# def output_file(filename):

#     return send_from_directory(

#         OUTPUT_DIR,

#         filename

#     )


# # ============================================================
# # MAIN
# # ============================================================

# if __name__ == "__main__":

#     print()
#     print("=" * 70)
#     print("ZONE COUNTING UI STARTED")
#     print("=" * 70)

#     print(
#         f"Open in browser:"
#     )

#     print(
#         f"http://192.168.2.240:{PORT}"
#     )

#     print()

#     print(
#         "Workflow:"
#     )

#     print(
#         "1. Upload image"
#     )

#     print(
#         "2. Draw zone"
#     )

#     print(
#         "3. Click Detect in Zone"
#     )

#     print(
#         "4. Only overlapping masks are displayed"
#     )

#     print("=" * 70)


#     app.run(

#         host=HOST,

#         port=PORT,

#         debug=False,

#         threaded=True

#     )
##############################################workin properly for pics and zone
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
from flask import Flask, request, render_template_string, jsonify, send_from_directory, Response
from ultralytics import YOLO
import os
import uuid
import cv2
import numpy as np
import threading

BASE_DIR = os.path.expanduser("~/ZONE_COUNTING")
MODEL_PATH = os.path.join(BASE_DIR, "runs", "segment",
                          "zone_counting_yolo26l_seg_v1_1280",
                          "weights", "best.pt")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "zone_outputs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

HOST = "0.0.0.0"
PORT = 5004
IMG_SIZE = 1280
CONF = 0.25
IOU = 0.50
MASK_OVERLAP = 0.05

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError("Model not found: {}".format(MODEL_PATH))

print("Loading model:", MODEL_PATH)
model = YOLO(MODEL_PATH)
print("Model loaded.")

app = Flask(__name__)
video_states = {}
video_lock = threading.Lock()

HTML = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Zone Counting</title>
<style>
body{margin:0;background:#111827;color:#fff;font-family:Arial}
header{padding:18px 24px;background:#1f2937}
header h1{margin:0} header p{color:#9ca3af}
.wrap{display:flex;min-height:calc(100vh - 90px)}
.side{width:300px;padding:20px;background:#1f2937}
.main{flex:1;padding:24px;text-align:center}
select,input,button{width:100%;padding:11px;margin:7px 0;
border-radius:7px;border:1px solid #4b5563;background:#111827;color:#fff}
button{border:0;font-weight:bold;cursor:pointer}
.blue{background:#2563eb}.green{background:#059669}.red{background:#dc2626}
.gray{background:#374151}button:disabled{opacity:.4;cursor:not-allowed}
.box{background:#111827;border:1px solid #374151;border-radius:8px;
padding:14px;margin-top:12px}
#viewer{position:relative;display:inline-block;max-width:100%;background:#000}
#mainImage,#mainVideo,#stream{display:block;max-width:100%;max-height:78vh}
#mainVideo,#stream{display:none}
#canvas{position:absolute;left:0;top:0;cursor:crosshair}
.count{font-size:46px;font-weight:bold;text-align:center}
.small{font-size:13px;color:#9ca3af}
@media(max-width:800px){.wrap{display:block}.side{width:auto}}
</style>
</head>
<body>
<header>
<h1>Zone Counting - Image + Video</h1>
<p>Draw a zone first. Only masks overlapping the zone are displayed.</p>
</header>
<div class="wrap">
<div class="side">
<input id="file" type="file" accept="image/*,video/*">
<select id="zone">
<option>Zone 1</option><option>Zone 2</option><option>Zone 3</option>
</select>
<button id="draw" class="blue" disabled>Draw / Redraw Zone</button>
<button id="clear" class="gray" disabled>Clear Zone</button>
<button id="detect" class="green" disabled>Detect in Zone</button>
<button id="stop" class="red" disabled>Stop Detection</button>
<div class="box" id="status">Upload an image or video.<br><br>
No detection happens until a zone is drawn.</div>
<div class="box">
<div class="small">CURRENT ZONE COUNT</div>
<div class="count" id="count">0</div>
</div>
</div>
<div class="main">
<div id="viewer">
<img id="mainImage" alt="">
<video id="mainVideo" controls muted></video>
<img id="stream" alt="">
<canvas id="canvas"></canvas>
</div>
<p class="small" id="hint">Upload an image or video.</p>
</div>
</div>

<script>
const fileEl=document.getElementById("file");
const zoneEl=document.getElementById("zone");
const drawBtn=document.getElementById("draw");
const clearBtn=document.getElementById("clear");
const detectBtn=document.getElementById("detect");
const stopBtn=document.getElementById("stop");
const statusEl=document.getElementById("status");
const countEl=document.getElementById("count");
const image=document.getElementById("mainImage");
const video=document.getElementById("mainVideo");
const stream=document.getElementById("stream");
const canvas=document.getElementById("canvas");
const ctx=canvas.getContext("2d");

let filename=null, mediaType=null, drawing=false, enabled=false;
let sx=0, sy=0, activeVideoId=null;

const zones={"Zone 1":null,"Zone 2":null,"Zone 3":null};

function status(t){statusEl.innerHTML=t}
function clearCanvas(){ctx.clearRect(0,0,canvas.width,canvas.height)}

function resizeCanvas(){
    let w=0,h=0;
    if(mediaType==="image"){w=image.clientWidth;h=image.clientHeight}
    if(mediaType==="video"){w=video.clientWidth;h=video.clientHeight}
    if(!w||!h)return;
    canvas.width=w; canvas.height=h;
    drawZone();
}

function drawZone(){
    clearCanvas();
    const z=zones[zoneEl.value];
    if(!z)return;
    ctx.fillStyle="rgba(0,255,136,.10)";
    ctx.fillRect(z.x1,z.y1,z.x2-z.x1,z.y2-z.y1);
    ctx.strokeStyle="#00ff88"; ctx.lineWidth=3;
    ctx.strokeRect(z.x1,z.y1,z.x2-z.x1,z.y2-z.y1);
}

function showRaw(){
    stream.style.display="none"; stream.src="";
    if(mediaType==="image"){
        image.style.display="block"; video.style.display="none";
    }else if(mediaType==="video"){
        image.style.display="none"; video.style.display="block";
    }
}

fileEl.addEventListener("change",async()=>{
    const f=fileEl.files[0]; if(!f)return;
    stopVideo(false);
    const fd=new FormData(); fd.append("file",f);
    status("Uploading...");
    try{
        const r=await fetch("/upload",{method:"POST",body:fd});
        const d=await r.json();
        if(!d.success){status("Upload error: "+d.error);return}
        filename=d.filename; mediaType=d.type;
        zones["Zone 1"]=null;zones["Zone 2"]=null;zones["Zone 3"]=null;
        zoneEl.value="Zone 1"; countEl.textContent="0"; clearCanvas();
        if(mediaType==="image"){
            video.pause(); video.removeAttribute("src"); video.load();
            image.src="/uploads/"+filename; image.style.display="block";
            video.style.display="none";
            image.onload=resizeCanvas;
            status("Image uploaded.<br>Draw a zone first.");
        }else{
            image.removeAttribute("src"); image.style.display="none";
            video.src="/uploads/"+filename; video.style.display="block";
            video.load(); video.onloadedmetadata=resizeCanvas;
            status("Video uploaded.<br>Preview it, then draw a zone.");
        }
        drawBtn.disabled=false; clearBtn.disabled=false; detectBtn.disabled=true;
        stopBtn.disabled=true;
    }catch(e){status("Upload failed: "+e)}
});

drawBtn.addEventListener("click",()=>{
    if(!filename)return alert("Upload a file first.");
    stopVideo(false); showRaw(); clearCanvas(); enabled=true; detectBtn.disabled=true;
    status("Click and drag to draw the zone.");
});

canvas.addEventListener("mousedown",e=>{
    if(!enabled)return;
    const r=canvas.getBoundingClientRect();
    sx=e.clientX-r.left; sy=e.clientY-r.top; drawing=true;
});

canvas.addEventListener("mousemove",e=>{
    if(!drawing)return;
    const r=canvas.getBoundingClientRect();
    const x=e.clientX-r.left,y=e.clientY-r.top;
    clearCanvas(); ctx.strokeStyle="#00ff88";ctx.lineWidth=3;
    ctx.setLineDash([8,5]);ctx.strokeRect(sx,sy,x-sx,y-sy);ctx.setLineDash([]);
});

canvas.addEventListener("mouseup",e=>{
    if(!drawing)return;
    const r=canvas.getBoundingClientRect();
    const ex=e.clientX-r.left,ey=e.clientY-r.top;
    drawing=false;enabled=false;
    const x1=Math.min(sx,ex),y1=Math.min(sy,ey);
    const x2=Math.max(sx,ex),y2=Math.max(sy,ey);
    if(x2-x1<10||y2-y1<10){clearCanvas();status("Zone too small.");return}
    zones[zoneEl.value]={x1,y1,x2,y2};
    drawZone();detectBtn.disabled=false;
    status("Zone created.<br>Click Detect in Zone.");
});

clearBtn.addEventListener("click",()=>{
    stopVideo(false); zones[zoneEl.value]=null; clearCanvas();
    countEl.textContent="0";detectBtn.disabled=true;showRaw();
    status(zoneEl.value+" cleared.<br>Draw a new zone.");
});

zoneEl.addEventListener("change",()=>{
    stopVideo(false);showRaw();clearCanvas();countEl.textContent="0";drawZone();
    detectBtn.disabled=!zones[zoneEl.value];
    status(zones[zoneEl.value]?"Zone exists.<br>Click Detect in Zone."
                              :"Draw this zone first.");
});

window.addEventListener("resize",resizeCanvas);

function originalSize(){
    if(mediaType==="image")return {w:image.naturalWidth,h:image.naturalHeight};
    return {w:video.videoWidth,h:video.videoHeight};
}

function zonePixels(){
    const z=zones[zoneEl.value], s=originalSize();
    return {
        x1:Math.round(z.x1*s.w/canvas.clientWidth),
        y1:Math.round(z.y1*s.h/canvas.clientHeight),
        x2:Math.round(z.x2*s.w/canvas.clientWidth),
        y2:Math.round(z.y2*s.h/canvas.clientHeight)
    };
}

detectBtn.addEventListener("click",async()=>{
    if(!zones[zoneEl.value])return alert("Draw the zone first.");
    if(mediaType==="image")await detectImage(); else await startVideo();
});

async function detectImage(){
    detectBtn.disabled=true;status("Running YOLO segmentation...");
    try{
        const r=await fetch("/detect_image",{
            method:"POST",headers:{"Content-Type":"application/json"},
            body:JSON.stringify({filename,zone_name:zoneEl.value,...zonePixels()})
        });
        const d=await r.json();
        if(!d.success){status("Detection error: "+d.error);return}
        image.src="/outputs/"+d.output_filename;image.style.display="block";
        countEl.textContent=d.count;clearCanvas();
        status("Detection complete.<br>Only in-zone masks are shown.");
    }catch(e){status("Detection failed: "+e)}
    finally{detectBtn.disabled=false}
}

async function startVideo(){
    const z=zonePixels();detectBtn.disabled=true;stopBtn.disabled=false;
    status("Starting video detection...");
    try{
        const r=await fetch("/start_video",{
            method:"POST",headers:{"Content-Type":"application/json"},
            body:JSON.stringify({filename,zone_name:zoneEl.value,...z})
        });
        const d=await r.json();
        if(!d.success){status("Video error: "+d.error);detectBtn.disabled=false;stopBtn.disabled=true;return}
        activeVideoId=d.video_id;video.pause();video.style.display="none";
        image.style.display="none";clearCanvas();
        stream.src="/video_stream/"+activeVideoId;stream.style.display="block";
        status("Video detection running.<br>Only in-zone masks are shown.");
        pollCount();
    }catch(e){status("Video start failed: "+e);detectBtn.disabled=false;stopBtn.disabled=true}
}

async function pollCount(){
    if(!activeVideoId)return;
    try{
        const r=await fetch("/video_count/"+activeVideoId),d=await r.json();
        countEl.textContent=d.count;
        if(d.running)setTimeout(pollCount,300);
        else{activeVideoId=null;stopBtn.disabled=true;detectBtn.disabled=false;
             status("Video finished.")}
    }catch(e){setTimeout(pollCount,500)}
}

async function stopVideo(send=true){
    if(!activeVideoId)return;
    const id=activeVideoId;activeVideoId=null;
    if(send)try{
        await fetch("/stop_video",{method:"POST",headers:{"Content-Type":"application/json"},
        body:JSON.stringify({video_id:id})});
    }catch(e){}
    stream.src="";stream.style.display="none";stopBtn.disabled=true;
    if(mediaType==="video")video.style.display="block";
    detectBtn.disabled=!zones[zoneEl.value];
}

stopBtn.addEventListener("click",()=>stopVideo(true));
</script>
</body>
</html>
"""


def clamp_zone(x1, y1, x2, y2, width, height):
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(1, min(int(x2), width))
    y2 = max(1, min(int(y2), height))
    return x1, y1, x2, y2


def process_frame(frame, x1, y1, x2, y2):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = clamp_zone(x1, y1, x2, y2, w, h)

    zone = np.zeros((h, w), dtype=np.uint8)
    zone[y1:y2, x1:x2] = 1

    result = model.predict(
        source=frame,
        imgsz=IMG_SIZE,
        conf=CONF,
        iou=IOU,
        device=0,
        verbose=False
    )[0]

    output = frame.copy()
    count = 0

    if result.masks is not None:
        for mask_tensor in result.masks.data:
            mask = mask_tensor.detach().cpu().numpy()
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            binary = (mask > 0.5).astype(np.uint8)

            area = int(binary.sum())
            if area == 0:
                continue

            intersection = np.logical_and(binary == 1, zone == 1)
            overlap = int(intersection.sum()) / float(area)

            if overlap < MASK_OVERLAP:
                continue

            count += 1
            m = binary.astype(bool)

            # Mask only; no boxes, labels, or confidence.
            green = np.zeros_like(output)
            green[:, :] = (0, 255, 0)
            output[m] = cv2.addWeighted(
                output[m], 0.35, green[m], 0.65, 0
            )

            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(output, contours, -1, (0, 255, 0), 1)

    cv2.rectangle(output, (x1, y1), (x2, y2), (255, 255, 0), 2)
    return output, count


@app.route("/")
def home():
    return render_template_string(HTML)


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify(success=False, error="No file uploaded.")

    f = request.files["file"]
    if not f.filename:
        return jsonify(success=False, error="No file selected.")

    ext = os.path.splitext(f.filename)[1].lower()
    image_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    video_ext = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    if ext in image_ext:
        kind = "image"
    elif ext in video_ext:
        kind = "video"
    else:
        return jsonify(success=False, error="Unsupported file format.")

    name = "{}{}".format(uuid.uuid4(), ext)
    f.save(os.path.join(UPLOAD_DIR, name))
    print("[UPLOAD]", kind, name)

    return jsonify(success=True, filename=name, type=kind)


@app.route("/uploads/<path:name>")
def uploads(name):
    return send_from_directory(UPLOAD_DIR, name)


@app.route("/outputs/<path:name>")
def outputs(name):
    return send_from_directory(OUTPUT_DIR, name)


@app.route("/detect_image", methods=["POST"])
def detect_image():
    try:
        d = request.get_json(force=True)
        path = os.path.join(UPLOAD_DIR, d["filename"])
        frame = cv2.imread(path)

        if frame is None:
            return jsonify(success=False, error="Unable to read image.")

        output, count = process_frame(
            frame, d["x1"], d["y1"], d["x2"], d["y2"]
        )

        name = "zone_{}.jpg".format(uuid.uuid4())
        out = os.path.join(OUTPUT_DIR, name)
        cv2.imwrite(out, output)

        return jsonify(success=True, count=count, output_filename=name)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify(success=False, error=str(exc)), 500


@app.route("/start_video", methods=["POST"])
def start_video():
    try:
        d = request.get_json(force=True)
        path = os.path.join(UPLOAD_DIR, d["filename"])

        if not os.path.isfile(path):
            return jsonify(success=False, error="Video not found.")

        with video_lock:
            for state in video_states.values():
                state["running"] = False

            vid = str(uuid.uuid4())
            video_states[vid] = {
                "filename": d["filename"],
                "x1": int(d["x1"]),
                "y1": int(d["y1"]),
                "x2": int(d["x2"]),
                "y2": int(d["y2"]),
                "running": True,
                "latest_count": 0,
            }

        return jsonify(success=True, video_id=vid)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify(success=False, error=str(exc)), 500


def generate_video(video_id):
    with video_lock:
        state = video_states.get(video_id)

    if state is None:
        return

    path = os.path.join(UPLOAD_DIR, state["filename"])
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        with video_lock:
            video_states.pop(video_id, None)
        return

    try:
        while True:
            with video_lock:
                state = video_states.get(video_id)
                if state is None or not state["running"]:
                    break

            ok, frame = cap.read()
            if not ok:
                break

            output, count = process_frame(
                frame, state["x1"], state["y1"], state["x2"], state["y2"]
            )

            with video_lock:
                if video_id in video_states:
                    video_states[video_id]["latest_count"] = count

            ok, buf = cv2.imencode(
                ".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, 85]
            )
            if not ok:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                buf.tobytes() +
                b"\r\n"
            )
    except GeneratorExit:
        pass
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        with video_lock:
            video_states.pop(video_id, None)


@app.route("/video_stream/<video_id>")
def video_stream(video_id):
    return Response(
        generate_video(video_id),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/video_count/<video_id>")
def video_count(video_id):
    with video_lock:
        state = video_states.get(video_id)
        if state is None:
            return jsonify(success=True, count=0, running=False)
        return jsonify(
            success=True,
            count=state["latest_count"],
            running=state["running"]
        )


@app.route("/stop_video", methods=["POST"])
def stop_video():
    try:
        d = request.get_json(force=True)
        vid = d.get("video_id")
        with video_lock:
            if vid in video_states:
                video_states[vid]["running"] = False
        return jsonify(success=True)
    except Exception as exc:
        return jsonify(success=False, error=str(exc)), 500


if __name__ == "__main__":
    print("=" * 60)
    print("ZONE COUNTING UI")
    print("Image + Video")
    print("http://192.168.2.240:{}".format(PORT))
    print("=" * 60)

    app.run(
        host=HOST,
        port=PORT,
        debug=False,
        threaded=True
    )
