# import os
# import json
# import time
# from pathlib import Path

# import cv2
# import numpy as np
# from flask import Flask, request, render_template_string, jsonify, send_file
# from werkzeug.utils import secure_filename
# from ultralytics import YOLO


# # ============================================================
# # CONFIGURATION
# # ============================================================

# PROJECT_ROOT = Path(__file__).resolve().parent.parent

# RUNS_ROOT = PROJECT_ROOT / "runs"

# UPLOAD_DIR = PROJECT_ROOT / "outputs" / "c_belt_uploads"
# RESULT_DIR = PROJECT_ROOT / "outputs" / "c_belt_results"

# UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
# RESULT_DIR.mkdir(parents=True, exist_ok=True)

# HOST = "0.0.0.0"
# PORT = 50011

# IMG_SIZE = 1280
# DEFAULT_CONF = 0.35
# DEFAULT_IOU = 0.50

# TRACKER = "bytetrack.yaml"

# IMAGE_EXTENSIONS = {
#     ".jpg",
#     ".jpeg",
#     ".png",
#     ".bmp",
#     ".webp",
# }

# VIDEO_EXTENSIONS = {
#     ".mp4",
#     ".avi",
#     ".mov",
#     ".mkv",
#     ".webm",
# }


# # ============================================================
# # FLASK
# # ============================================================

# app = Flask(__name__)

# app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024


# # ============================================================
# # MODEL FINDER
# # ============================================================

# def find_best_model():
#     """
#     Automatically find the trained best.pt.

#     Priority:
#     1. YOLO_MODEL environment variable
#     2. Known C-belt path
#     3. Any best.pt inside runs/
#     """

#     env_model = os.environ.get("YOLO_MODEL", "").strip()

#     if env_model:
#         env_path = Path(env_model).expanduser()

#         if env_path.is_file():
#             return env_path.resolve()

#     preferred_paths = [
#         RUNS_ROOT
#         / "segment"
#         / "runs"
#         / "segment"
#         / "final_c_belt_sack_yolo26l_seg_v1"
#         / "weights"
#         / "best.pt",

#         RUNS_ROOT
#         / "segment"
#         / "final_c_belt_sack_yolo26l_seg_v1"
#         / "weights"
#         / "best.pt",

#         RUNS_ROOT
#         / "final_c_belt_sack_yolo26l_seg_v1"
#         / "weights"
#         / "best.pt",
#     ]

#     for path in preferred_paths:
#         if path.is_file():
#             return path.resolve()

#     candidates = []

#     if RUNS_ROOT.exists():
#         candidates = list(RUNS_ROOT.rglob("best.pt"))

#     if not candidates:
#         raise FileNotFoundError(
#             "\nNo best.pt found.\n\n"
#             f"Searched inside:\n{RUNS_ROOT}\n\n"
#             "Run:\n"
#             "find ~/NEW_SACK_COUNT/runs -type f -name best.pt -print\n"
#         )

#     c_belt_candidates = [
#         p for p in candidates
#         if "final_c_belt" in str(p).lower()
#     ]

#     if c_belt_candidates:
#         candidates = c_belt_candidates

#     candidates.sort(
#         key=lambda p: p.stat().st_mtime,
#         reverse=True
#     )

#     return candidates[0].resolve()


# # ============================================================
# # DEVICE
# # ============================================================

# def get_device():
#     try:
#         import torch

#         if torch.cuda.is_available():
#             return 0

#     except Exception:
#         pass

#     return "cpu"


# # ============================================================
# # LOAD MODEL
# # ============================================================

# MODEL_PATH = find_best_model()
# DEVICE = get_device()

# print()
# print("=" * 75)
# print("C-BELT RICE BAG COUNTING SYSTEM")
# print("=" * 75)
# print(f"Project Root : {PROJECT_ROOT}")
# print(f"Model Path   : {MODEL_PATH}")
# print(f"Model Exists : {MODEL_PATH.exists()}")
# print(f"Device       : {DEVICE}")

# try:
#     import torch

#     print(f"PyTorch      : {torch.__version__}")
#     print(f"CUDA         : {torch.cuda.is_available()}")

#     if torch.cuda.is_available():
#         print(
#             f"GPU          : "
#             f"{torch.cuda.get_device_name(0)}"
#         )

# except Exception as e:
#     print(f"Torch info error: {e}")

# try:
#     import ultralytics

#     print(
#         f"Ultralytics  : "
#         f"{ultralytics.__version__}"
#     )

# except Exception:
#     pass

# print("=" * 75)

# print("Loading YOLO model...")

# model = YOLO(str(MODEL_PATH))

# print("YOLO model loaded successfully.")
# print("=" * 75)


# # ============================================================
# # HTML UI
# # ============================================================

# HTML = r"""
# <!DOCTYPE html>
# <html>
# <head>

# <meta charset="UTF-8">

# <meta name="viewport"
#       content="width=device-width, initial-scale=1.0">

# <title>C-Belt Rice Bag Counting</title>

# <style>

# body {
#     margin: 0;
#     padding: 25px;
#     background: #111827;
#     color: white;
#     font-family: Arial, sans-serif;
# }

# .container {
#     max-width: 1250px;
#     margin: auto;
# }

# .card {
#     background: #1f2937;
#     padding: 20px;
#     border-radius: 12px;
#     margin-bottom: 20px;
# }

# h1 {
#     margin-top: 0;
# }

# button,
# input {
#     padding: 10px;
#     margin: 5px;
#     border-radius: 7px;
#     border: 1px solid #4b5563;
# }

# button {
#     background: #374151;
#     color: white;
#     cursor: pointer;
# }

# button.primary {
#     background: #2563eb;
# }

# button:disabled {
#     opacity: 0.5;
#     cursor: not-allowed;
# }

# #previewWrap {
#     position: relative;
#     display: inline-block;
#     max-width: 100%;
#     background: black;
#     margin-top: 15px;
# }

# #preview {
#     display: block;
#     max-width: 100%;
#     max-height: 70vh;
# }

# #drawCanvas {
#     position: absolute;
#     left: 0;
#     top: 0;
#     cursor: crosshair;
# }

# .info {
#     color: #d1d5db;
#     line-height: 1.6;
# }

# .status {
#     margin-top: 15px;
#     white-space: pre-wrap;
#     color: #93c5fd;
# }

# #resultImage {
#     width: 100%;
#     max-height: 75vh;
#     object-fit: contain;
#     background: black;
#     display: none;
# }

# .red {
#     color: #ff3333;
# }

# .blue {
#     color: #3399ff;
# }

# .counts {
#     font-size: 24px;
#     font-weight: bold;
#     margin-top: 15px;
# }

# </style>

# </head>

# <body>

# <div class="container">

# <h1>
# Warehouse C-Belt Rice Bag Counting
# </h1>

# <div class="card">

# <div class="info">

# <b>Workflow:</b>

# <br>

# 1. Upload image/video.

# <br>

# 2. For video, first frame will appear.

# <br>

# 3. Draw RED line for unloading.

# <br>

# 4. Draw BLUE line for loading.

# <br>

# 5. Click Process.

# <br><br>

# <b class="red">
# RED = UNLOADING
# </b>

# &nbsp;&nbsp;&nbsp;

# <b class="blue">
# BLUE = LOADING
# </b>

# </div>

# <br>

# <input
#     id="fileInput"
#     type="file"
#     accept="image/*,video/*"
# >

# <br>

# <label>
# Confidence:
# <input
#     id="confidence"
#     type="number"
#     min="0.05"
#     max="0.95"
#     step="0.05"
#     value="0.35"
# >
# </label>

# <br>

# <button onclick="clearLines()">
# Clear Lines
# </button>

# <button
#     id="processButton"
#     class="primary"
#     onclick="processMedia()"
#     disabled
# >
# Process
# </button>

# <div id="lineStatus">
# RED: not drawn | BLUE: not drawn
# </div>

# <div id="status" class="status">
# Waiting for upload...
# </div>

# <div id="previewWrap">

# <img
#     id="preview"
#     alt=""
# >

# <canvas
#     id="drawCanvas"
# ></canvas>

# </div>

# </div>


# <div class="card">

# <h2>
# Result
# </h2>

# <img
#     id="resultImage"
#     alt="Result"
# >

# <div id="resultText"></div>

# </div>

# </div>


# <script>

# let selectedFile = null;

# let mediaWidth = 0;
# let mediaHeight = 0;

# let redLine = null;
# let blueLine = null;

# let drawing = false;

# let startX = 0;
# let startY = 0;

# let currentColor = "red";

# const fileInput =
#     document.getElementById("fileInput");

# const preview =
#     document.getElementById("preview");

# const canvas =
#     document.getElementById("drawCanvas");

# const ctx =
#     canvas.getContext("2d");

# const processButton =
#     document.getElementById("processButton");

# const statusBox =
#     document.getElementById("status");

# const lineStatus =
#     document.getElementById("lineStatus");


# fileInput.addEventListener(
#     "change",
#     handleFile
# );


# async function handleFile() {

#     if (!fileInput.files.length) {
#         return;
#     }

#     selectedFile = fileInput.files[0];

#     clearLines();

#     const url =
#         URL.createObjectURL(selectedFile);

#     if (
#         selectedFile.type.startsWith("image/")
#     ) {

#         preview.style.display = "block";

#         preview.src = url;

#         preview.onload = function() {

#             mediaWidth =
#                 preview.naturalWidth;

#             mediaHeight =
#                 preview.naturalHeight;

#             resizeCanvas();

#             processButton.disabled = false;

#             statusBox.textContent =
#                 "Image loaded. Draw RED and BLUE lines.";

#         };

#     } else if (
#         selectedFile.type.startsWith("video/")
#     ) {

#         await loadVideoFirstFrame(url);

#     } else {

#         statusBox.textContent =
#             "Unsupported file.";

#     }
# }


# function loadVideoFirstFrame(url) {

#     return new Promise(function(resolve, reject) {

#         const video =
#             document.createElement("video");

#         video.muted = true;

#         video.preload = "metadata";

#         video.src = url;

#         video.addEventListener(
#             "loadedmetadata",
#             function() {

#                 mediaWidth =
#                     video.videoWidth;

#                 mediaHeight =
#                     video.videoHeight;

#                 video.currentTime = 0;

#             }
#         );

#         video.addEventListener(
#             "seeked",
#             function() {

#                 const tempCanvas =
#                     document.createElement("canvas");

#                 tempCanvas.width =
#                     mediaWidth;

#                 tempCanvas.height =
#                     mediaHeight;

#                 const tempContext =
#                     tempCanvas.getContext("2d");

#                 tempContext.drawImage(
#                     video,
#                     0,
#                     0,
#                     mediaWidth,
#                     mediaHeight
#                 );

#                 preview.src =
#                     tempCanvas.toDataURL(
#                         "image/jpeg",
#                         0.9
#                     );

#                 preview.style.display =
#                     "block";

#                 resizeCanvas();

#                 processButton.disabled =
#                     false;

#                 statusBox.textContent =
#                     "Video first frame loaded. Draw RED and BLUE lines.";

#                 resolve();

#             },
#             { once: true }
#         );

#         video.addEventListener(
#             "error",
#             reject
#         );

#         video.load();

#     });
# }


# function resizeCanvas() {

#     if (
#         !mediaWidth ||
#         !mediaHeight
#     ) {
#         return;
#     }

#     canvas.width = mediaWidth;

#     canvas.height = mediaHeight;

#     const rect =
#         preview.getBoundingClientRect();

#     canvas.style.width =
#         rect.width + "px";

#     canvas.style.height =
#         rect.height + "px";

#     redraw();
# }


# window.addEventListener(
#     "resize",
#     resizeCanvas
# );


# function getPoint(event) {

#     const rect =
#         canvas.getBoundingClientRect();

#     return {

#         x:
#             (event.clientX - rect.left)
#             *
#             (canvas.width / rect.width),

#         y:
#             (event.clientY - rect.top)
#             *
#             (canvas.height / rect.height)

#     };
# }


# canvas.addEventListener(
#     "mousedown",
#     function(event) {

#         if (!selectedFile) {
#             return;
#         }

#         if (!redLine) {

#             currentColor = "red";

#         } else if (!blueLine) {

#             currentColor = "blue";

#         } else {

#             statusBox.textContent =
#                 "Both lines already drawn. Click Clear Lines to redraw.";

#             return;
#         }

#         const point =
#             getPoint(event);

#         startX = point.x;
#         startY = point.y;

#         drawing = true;

#     }
# );


# canvas.addEventListener(
#     "mousemove",
#     function(event) {

#         if (!drawing) {
#             return;
#         }

#         const point =
#             getPoint(event);

#         redraw();

#         ctx.lineWidth = 6;

#         ctx.strokeStyle =
#             currentColor === "red"
#             ? "#ff0000"
#             : "#0066ff";

#         ctx.beginPath();

#         ctx.moveTo(
#             startX,
#             startY
#         );

#         ctx.lineTo(
#             point.x,
#             point.y
#         );

#         ctx.stroke();

#     }
# );


# canvas.addEventListener(
#     "mouseup",
#     function(event) {

#         if (!drawing) {
#             return;
#         }

#         drawing = false;

#         const point =
#             getPoint(event);

#         const length =
#             Math.hypot(
#                 point.x - startX,
#                 point.y - startY
#             );

#         if (length < 20) {

#             statusBox.textContent =
#                 "Line too short. Draw a longer line.";

#             redraw();

#             return;
#         }

#         const line = {

#             x1: startX,
#             y1: startY,

#             x2: point.x,
#             y2: point.y

#         };

#         if (currentColor === "red") {

#             redLine = line;

#         } else {

#             blueLine = line;

#         }

#         redraw();

#         updateLineStatus();

#     }
# );


# function drawLine(
#     line,
#     color,
#     label
# ) {

#     if (!line) {
#         return;
#     }

#     ctx.lineWidth = 6;

#     ctx.strokeStyle = color;

#     ctx.beginPath();

#     ctx.moveTo(
#         line.x1,
#         line.y1
#     );

#     ctx.lineTo(
#         line.x2,
#         line.y2
#     );

#     ctx.stroke();

#     ctx.fillStyle = color;

#     ctx.font =
#         "bold 28px Arial";

#     ctx.fillText(
#         label,
#         line.x1 + 10,
#         line.y1 - 10
#     );
# }


# function redraw() {

#     ctx.clearRect(
#         0,
#         0,
#         canvas.width,
#         canvas.height
#     );

#     drawLine(
#         redLine,
#         "#ff0000",
#         "UNLOADING"
#     );

#     drawLine(
#         blueLine,
#         "#0066ff",
#         "LOADING"
#     );
# }


# function clearLines() {

#     redLine = null;

#     blueLine = null;

#     redraw();

#     updateLineStatus();
# }


# function updateLineStatus() {

#     lineStatus.textContent =
#         "RED: "
#         +
#         (redLine ? "drawn" : "not drawn")
#         +
#         " | BLUE: "
#         +
#         (blueLine ? "drawn" : "not drawn");

# }


# async function processMedia() {

#     if (!selectedFile) {

#         alert(
#             "Upload image or video first."
#         );

#         return;
#     }

#     if (!redLine && !blueLine) {

#         alert(
#             "Draw at least one counting line."
#         );

#         return;
#     }

#     const form =
#         new FormData();

#     form.append(
#         "file",
#         selectedFile
#     );

#     form.append(
#         "confidence",
#         document.getElementById(
#             "confidence"
#         ).value
#     );

#     form.append(
#         "red_line",
#         redLine
#             ? JSON.stringify(redLine)
#             : ""
#     );

#     form.append(
#         "blue_line",
#         blueLine
#             ? JSON.stringify(blueLine)
#             : ""
#     );

#     processButton.disabled = true;

#     statusBox.textContent =
#         "Processing... Please wait.";

#     try {

#         const response =
#             await fetch(
#                 "/process",
#                 {
#                     method: "POST",
#                     body: form
#                 }
#             );

#         const data =
#             await response.json();

#         if (!response.ok || !data.success) {

#             throw new Error(
#                 data.error ||
#                 "Processing failed."
#             );

#         }

#         const resultImage =
#             document.getElementById(
#                 "resultImage"
#             );

#         resultImage.src =
#             data.result_url
#             +
#             "?t="
#             +
#             Date.now();

#         resultImage.style.display =
#             "block";

#         document.getElementById(
#             "resultText"
#         ).innerHTML =

#             "<div class='counts'>" +

#             "<span class='red'>" +
#             "UNLOADING: " +
#             data.unloading_count +
#             "</span>" +

#             "&nbsp;&nbsp;&nbsp;" +

#             "<span class='blue'>" +
#             "LOADING: " +
#             data.loading_count +
#             "</span>" +

#             "</div>" +

#             "<p>Detected instances: " +
#             data.detected_count +
#             "</p>" +

#             "<p>Processing time: " +
#             data.processing_seconds +
#             " seconds</p>";

#         statusBox.textContent =
#             "Processing completed.";

#     } catch (error) {

#         statusBox.textContent =
#             "ERROR: "
#             +
#             error.message;

#     }

#     processButton.disabled = false;
# }

# </script>

# </body>
# </html>
# """


# # ============================================================
# # LINE PARSER
# # ============================================================

# def parse_line(value):
#     if not value:
#         return None

#     try:
#         data = json.loads(value)
#     except Exception:
#         return None

#     required = [
#         "x1",
#         "y1",
#         "x2",
#         "y2",
#     ]

#     if not all(
#         key in data
#         for key in required
#     ):
#         return None

#     try:
#         line = {
#             key: float(data[key])
#             for key in required
#         }

#     except Exception:
#         return None

#     length = np.hypot(
#         line["x2"] - line["x1"],
#         line["y2"] - line["y1"]
#     )

#     if length < 20:
#         return None

#     return line


# # ============================================================
# # LINE SIDE
# # ============================================================

# def point_side(x, y, line):

#     return (
#         (line["x2"] - line["x1"])
#         *
#         (y - line["y1"])
#         -
#         (line["y2"] - line["y1"])
#         *
#         (x - line["x1"])
#     )


# # ============================================================
# # MASK CENTROID
# # ============================================================

# def mask_centroid(mask, box):

#     ys, xs = np.where(mask > 0)

#     if len(xs) > 0:

#         return (
#             float(np.mean(xs)),
#             float(np.mean(ys))
#         )

#     x1, y1, x2, y2 = box

#     return (
#         float((x1 + x2) / 2),
#         float((y1 + y2) / 2)
#     )


# # ============================================================
# # BBOX IOU
# # ============================================================

# def bbox_iou(box_a, box_b):

#     ax1, ay1, ax2, ay2 = box_a
#     bx1, by1, bx2, by2 = box_b

#     ix1 = max(ax1, bx1)
#     iy1 = max(ay1, by1)

#     ix2 = min(ax2, bx2)
#     iy2 = min(ay2, by2)

#     iw = max(
#         0.0,
#         ix2 - ix1
#     )

#     ih = max(
#         0.0,
#         iy2 - iy1
#     )

#     intersection = iw * ih

#     area_a = (
#         max(0.0, ax2 - ax1)
#         *
#         max(0.0, ay2 - ay1)
#     )

#     area_b = (
#         max(0.0, bx2 - bx1)
#         *
#         max(0.0, by2 - by1)
#     )

#     union = (
#         area_a
#         +
#         area_b
#         -
#         intersection
#     )

#     if union <= 0:
#         return 0.0

#     return intersection / union


# # ============================================================
# # MASK IOU
# # ============================================================

# def mask_iou(mask_a, mask_b):

#     a = mask_a.astype(bool)
#     b = mask_b.astype(bool)

#     intersection = np.logical_and(
#         a,
#         b
#     ).sum()

#     union = np.logical_or(
#         a,
#         b
#     ).sum()

#     if union == 0:
#         return 0.0

#     return float(
#         intersection / union
#     )


# # ============================================================
# # EXTRACT DETECTIONS
# # ============================================================

# def extract_detections(
#     result,
#     frame_shape
# ):

#     frame_h, frame_w = frame_shape[:2]

#     detections = []

#     if result.boxes is None:
#         return detections

#     boxes = (
#         result.boxes.xyxy
#         .cpu()
#         .numpy()
#     )

#     if result.boxes.conf is not None:

#         confs = (
#             result.boxes.conf
#             .cpu()
#             .numpy()
#         )

#     else:

#         confs = np.ones(
#             len(boxes),
#             dtype=np.float32
#         )

#     if result.boxes.id is not None:

#         ids = (
#             result.boxes.id
#             .int()
#             .cpu()
#             .tolist()
#         )

#     else:

#         ids = list(
#             range(len(boxes))
#         )

#     if result.masks is not None:

#         masks = (
#             result.masks.data
#             .cpu()
#             .numpy()
#         )

#     else:

#         masks = None

#     for i, box in enumerate(boxes):

#         x1, y1, x2, y2 = map(
#             float,
#             box
#         )

#         x1 = max(
#             0.0,
#             min(x1, frame_w - 1)
#         )

#         y1 = max(
#             0.0,
#             min(y1, frame_h - 1)
#         )

#         x2 = max(
#             0.0,
#             min(x2, frame_w - 1)
#         )

#         y2 = max(
#             0.0,
#             min(y2, frame_h - 1)
#         )

#         if x2 <= x1 or y2 <= y1:
#             continue

#         mask = np.zeros(
#             (
#                 frame_h,
#                 frame_w
#             ),
#             dtype=np.uint8
#         )

#         if (
#             masks is not None
#             and i < len(masks)
#         ):

#             raw_mask = masks[i]

#             resized = cv2.resize(
#                 raw_mask,
#                 (
#                     frame_w,
#                     frame_h
#                 ),
#                 interpolation=cv2.INTER_NEAREST
#             )

#             mask = (
#                 resized > 0.5
#             ).astype(np.uint8)

#         detections.append(
#             {
#                 "box": [
#                     x1,
#                     y1,
#                     x2,
#                     y2
#                 ],
#                 "conf": float(
#                     confs[i]
#                 ),
#                 "id": int(
#                     ids[i]
#                 ),
#                 "mask": mask
#             }
#         )

#     # Highest confidence first.
#     detections.sort(
#         key=lambda d: d["conf"],
#         reverse=True
#     )

#     # --------------------------------------------------------
#     # DUPLICATE SUPPRESSION
#     # --------------------------------------------------------

#     kept = []

#     for detection in detections:

#         duplicate = False

#         for previous in kept:

#             box_overlap = bbox_iou(
#                 detection["box"],
#                 previous["box"]
#             )

#             if box_overlap >= 0.70:

#                 if (
#                     detection["mask"].sum() > 0
#                     and previous["mask"].sum() > 0
#                 ):

#                     mask_overlap = mask_iou(
#                         detection["mask"],
#                         previous["mask"]
#                     )

#                     if mask_overlap >= 0.60:

#                         duplicate = True
#                         break

#                 else:

#                     duplicate = True
#                     break

#         if not duplicate:
#             kept.append(detection)

#     return kept


# # ============================================================
# # DRAW DETECTION
# # ============================================================

# def draw_detection(
#     frame,
#     detection
# ):

#     mask = detection["mask"]

#     if mask.sum() > 0:

#         overlay = frame.copy()

#         overlay[mask > 0] = (
#             0.45
#             *
#             overlay[mask > 0]
#             +
#             0.55
#             *
#             np.array(
#                 [0, 255, 0]
#             )
#         ).astype(np.uint8)

#         frame[:] = overlay

#         contours, _ = cv2.findContours(
#             mask,
#             cv2.RETR_EXTERNAL,
#             cv2.CHAIN_APPROX_SIMPLE
#         )

#         if contours:

#             cv2.drawContours(
#                 frame,
#                 contours,
#                 -1,
#                 (0, 255, 0),
#                 2
#             )

#     cx, cy = mask_centroid(
#         mask,
#         detection["box"]
#     )

#     cv2.circle(
#         frame,
#         (
#             int(cx),
#             int(cy)
#         ),
#         5,
#         (0, 255, 255),
#         -1
#     )

#     cv2.putText(
#         frame,
#         f"ID {detection['id']}",
#         (
#             int(cx) + 8,
#             int(cy) - 8
#         ),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.55,
#         (255, 255, 255),
#         2,
#         cv2.LINE_AA
#     )


# # ============================================================
# # DRAW COUNTING LINE
# # ============================================================

# def draw_counting_line(
#     frame,
#     line,
#     color,
#     label
# ):

#     if line is None:
#         return

#     p1 = (
#         int(line["x1"]),
#         int(line["y1"])
#     )

#     p2 = (
#         int(line["x2"]),
#         int(line["y2"])
#     )

#     cv2.line(
#         frame,
#         p1,
#         p2,
#         color,
#         4,
#         cv2.LINE_AA
#     )

#     cv2.putText(
#         frame,
#         label,
#         (
#             p1[0] + 10,
#             p1[1] - 10
#         ),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.75,
#         color,
#         2,
#         cv2.LINE_AA
#     )


# # ============================================================
# # IMAGE PROCESSING
# # ============================================================

# def process_image(
#     input_path,
#     output_path,
#     conf,
#     red_line,
#     blue_line
# ):

#     image = cv2.imread(
#         str(input_path)
#     )

#     if image is None:
#         raise RuntimeError(
#             "Could not read image."
#         )

#     results = model.predict(
#         source=image,
#         conf=conf,
#         iou=DEFAULT_IOU,
#         imgsz=IMG_SIZE,
#         device=DEVICE,
#         verbose=False
#     )

#     result = results[0]

#     detections = extract_detections(
#         result,
#         image.shape
#     )

#     output = image.copy()

#     for detection in detections:

#         draw_detection(
#             output,
#             detection
#         )

#     draw_counting_line(
#         output,
#         red_line,
#         (0, 0, 255),
#         "UNLOADING"
#     )

#     draw_counting_line(
#         output,
#         blue_line,
#         (255, 0, 0),
#         "LOADING"
#     )

#     cv2.putText(
#         output,
#         f"DETECTED: {len(detections)}",
#         (20, 40),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         1.0,
#         (255, 255, 255),
#         2,
#         cv2.LINE_AA
#     )

#     if not cv2.imwrite(
#         str(output_path),
#         output
#     ):
#         raise RuntimeError(
#             "Could not save output image."
#         )

#     return (
#         len(detections),
#         0,
#         0
#     )


# # ============================================================
# # VIDEO PROCESSING
# # ============================================================

# def process_video(
#     input_path,
#     output_path,
#     conf,
#     red_line,
#     blue_line
# ):

#     cap = cv2.VideoCapture(
#         str(input_path)
#     )

#     if not cap.isOpened():
#         raise RuntimeError(
#             "Could not open video."
#         )

#     width = int(
#         cap.get(
#             cv2.CAP_PROP_FRAME_WIDTH
#         )
#     )

#     height = int(
#         cap.get(
#             cv2.CAP_PROP_FRAME_HEIGHT
#         )
#     )

#     fps = cap.get(
#         cv2.CAP_PROP_FPS
#     )

#     if fps <= 0:
#         fps = 25.0

#     total_frames = int(
#         cap.get(
#             cv2.CAP_PROP_FRAME_COUNT
#         )
#     )

#     fourcc = cv2.VideoWriter_fourcc(
#         *"mp4v"
#     )

#     writer = cv2.VideoWriter(
#         str(output_path),
#         fourcc,
#         fps,
#         (
#             width,
#             height
#         )
#     )

#     if not writer.isOpened():

#         cap.release()

#         raise RuntimeError(
#             "Could not create output video."
#         )

#     red_previous = {}
#     blue_previous = {}

#     red_counted = set()
#     blue_counted = set()

#     max_detected = 0

#     frame_number = 0

#     try:

#         while True:

#             ok, frame = cap.read()

#             if not ok:
#                 break

#             frame_number += 1

#             results = model.track(
#                 source=frame,
#                 conf=conf,
#                 iou=DEFAULT_IOU,
#                 imgsz=IMG_SIZE,
#                 tracker=TRACKER,
#                 persist=True,
#                 device=DEVICE,
#                 verbose=False
#             )

#             result = results[0]

#             detections = extract_detections(
#                 result,
#                 frame.shape
#             )

#             max_detected = max(
#                 max_detected,
#                 len(detections)
#             )

#             tolerance = max(
#                 8.0,
#                 min(
#                     width,
#                     height
#                 ) * 0.005
#             )

#             for detection in detections:

#                 draw_detection(
#                     frame,
#                     detection
#                 )

#                 track_id = detection["id"]

#                 cx, cy = mask_centroid(
#                     detection["mask"],
#                     detection["box"]
#                 )

#                 # ------------------------------------------------
#                 # RED = UNLOADING
#                 # ------------------------------------------------

#                 if red_line is not None:

#                     side = point_side(
#                         cx,
#                         cy,
#                         red_line
#                     )

#                     if abs(side) > tolerance:

#                         current_side = (
#                             1
#                             if side > 0
#                             else -1
#                         )

#                         previous_side = (
#                             red_previous.get(
#                                 track_id
#                             )
#                         )

#                         if (
#                             previous_side is not None
#                             and previous_side != current_side
#                             and track_id not in red_counted
#                         ):

#                             red_counted.add(
#                                 track_id
#                             )

#                         red_previous[
#                             track_id
#                         ] = current_side

#                 # ------------------------------------------------
#                 # BLUE = LOADING
#                 # ------------------------------------------------

#                 if blue_line is not None:

#                     side = point_side(
#                         cx,
#                         cy,
#                         blue_line
#                     )

#                     if abs(side) > tolerance:

#                         current_side = (
#                             1
#                             if side > 0
#                             else -1
#                         )

#                         previous_side = (
#                             blue_previous.get(
#                                 track_id
#                             )
#                         )

#                         if (
#                             previous_side is not None
#                             and previous_side != current_side
#                             and track_id not in blue_counted
#                         ):

#                             blue_counted.add(
#                                 track_id
#                             )

#                         blue_previous[
#                             track_id
#                         ] = current_side

#             # ----------------------------------------------------
#             # DRAW LINES
#             # ----------------------------------------------------

#             draw_counting_line(
#                 frame,
#                 red_line,
#                 (0, 0, 255),
#                 f"UNLOADING: {len(red_counted)}"
#             )

#             draw_counting_line(
#                 frame,
#                 blue_line,
#                 (255, 0, 0),
#                 f"LOADING: {len(blue_counted)}"
#             )

#             # ----------------------------------------------------
#             # STATUS
#             # ----------------------------------------------------

#             cv2.putText(
#                 frame,
#                 f"Frame: {frame_number}/{total_frames}",
#                 (20, 35),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.7,
#                 (255, 255, 255),
#                 2,
#                 cv2.LINE_AA
#             )

#             cv2.putText(
#                 frame,
#                 f"UNLOADING: {len(red_counted)}",
#                 (20, 70),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.8,
#                 (0, 0, 255),
#                 2,
#                 cv2.LINE_AA
#             )

#             cv2.putText(
#                 frame,
#                 f"LOADING: {len(blue_counted)}",
#                 (20, 105),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.8,
#                 (255, 0, 0),
#                 2,
#                 cv2.LINE_AA
#             )

#             writer.write(frame)

#     finally:

#         cap.release()
#         writer.release()

#     return (
#         max_detected,
#         len(red_counted),
#         len(blue_counted)
#     )


# # ============================================================
# # HOME
# # ============================================================

# @app.route(
#     "/",
#     methods=["GET"]
# )
# def home():

#     return render_template_string(
#         HTML
#     )


# # ============================================================
# # HEALTH CHECK
# # ============================================================

# @app.route(
#     "/health",
#     methods=["GET"]
# )
# def health():

#     return jsonify(
#         {
#             "status": "ok",
#             "model": str(MODEL_PATH),
#             "model_exists": MODEL_PATH.exists(),
#             "device": str(DEVICE),
#             "port": PORT
#         }
#     )


# # ============================================================
# # PROCESS
# # ============================================================

# @app.route(
#     "/process",
#     methods=["POST"]
# )
# def process():

#     start_time = time.time()

#     try:

#         if "file" not in request.files:

#             return jsonify(
#                 {
#                     "success": False,
#                     "error": "No file uploaded."
#                 }
#             ), 400

#         uploaded = request.files["file"]

#         if not uploaded.filename:

#             return jsonify(
#                 {
#                     "success": False,
#                     "error": "Empty filename."
#                 }
#             ), 400

#         filename = secure_filename(
#             uploaded.filename
#         )

#         suffix = Path(
#             filename
#         ).suffix.lower()

#         if (
#             suffix not in IMAGE_EXTENSIONS
#             and suffix not in VIDEO_EXTENSIONS
#         ):

#             return jsonify(
#                 {
#                     "success": False,
#                     "error":
#                         f"Unsupported file type: {suffix}"
#                 }
#             ), 400

#         try:

#             conf = float(
#                 request.form.get(
#                     "confidence",
#                     DEFAULT_CONF
#                 )
#             )

#         except ValueError:

#             conf = DEFAULT_CONF

#         conf = max(
#             0.05,
#             min(0.95, conf)
#         )

#         red_line = parse_line(
#             request.form.get(
#                 "red_line",
#                 ""
#             )
#         )

#         blue_line = parse_line(
#             request.form.get(
#                 "blue_line",
#                 ""
#             )
#         )

#         timestamp = int(
#             time.time() * 1000
#         )

#         input_path = (
#             UPLOAD_DIR
#             /
#             f"{timestamp}_{filename}"
#         )

#         uploaded.save(
#             str(input_path)
#         )

#         if suffix in IMAGE_EXTENSIONS:

#             output_path = (
#                 RESULT_DIR
#                 /
#                 f"result_{timestamp}.jpg"
#             )

#             detected, unloading, loading = (
#                 process_image(
#                     input_path,
#                     output_path,
#                     conf,
#                     red_line,
#                     blue_line
#                 )
#             )

#         else:

#             output_path = (
#                 RESULT_DIR
#                 /
#                 f"result_{timestamp}.mp4"
#             )

#             detected, unloading, loading = (
#                 process_video(
#                     input_path,
#                     output_path,
#                     conf,
#                     red_line,
#                     blue_line
#                 )
#             )

#         elapsed = (
#             time.time()
#             -
#             start_time
#         )

#         print(
#             f"Processed: {filename} | "
#             f"Detected: {detected} | "
#             f"Unloading: {unloading} | "
#             f"Loading: {loading} | "
#             f"Time: {elapsed:.2f}s"
#         )

#         return jsonify(
#             {
#                 "success": True,
#                 "result_url":
#                     f"/result/{output_path.name}",
#                 "detected_count":
#                     detected,
#                 "unloading_count":
#                     unloading,
#                 "loading_count":
#                     loading,
#                 "processing_seconds":
#                     round(elapsed, 2)
#             }
#         )

#     except Exception as e:

#         import traceback

#         traceback.print_exc()

#         return jsonify(
#             {
#                 "success": False,
#                 "error": str(e)
#             }
#         ), 500


# # ============================================================
# # RESULT FILE
# # ============================================================

# @app.route(
#     "/result/<filename>",
#     methods=["GET"]
# )
# def result_file(filename):

#     safe_name = secure_filename(
#         filename
#     )

#     path = RESULT_DIR / safe_name

#     if not path.is_file():

#         return jsonify(
#             {
#                 "error": "Result file not found."
#             }
#         ), 404

#     return send_file(
#         path
#     )


# # ============================================================
# # MAIN
# # ============================================================

# if __name__ == "__main__":

#     print()
#     print("=" * 75)
#     print("FLASK SERVER STARTING")
#     print("=" * 75)

#     print(
#         f"Local GPU server URL:"
#         f" http://127.0.0.1:{PORT}/"
#     )

#     print(
#         f"Health check:"
#         f" http://127.0.0.1:{PORT}/health"
#     )

#     print()
#     print(
#         "Server is listening on:"
#         f" 0.0.0.0:{PORT}"
#     )

#     print("=" * 75)

#     app.run(
#         host=HOST,
#         port=PORT,
#         debug=False,
#         threaded=True
#     )

import os
import json
import time
from pathlib import Path

import cv2
import numpy as np
from flask import (
    Flask,
    request,
    render_template_string,
    jsonify,
    send_file,
)
from werkzeug.utils import secure_filename
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = PROJECT_ROOT / "runs"

UPLOAD_DIR = PROJECT_ROOT / "outputs" / "c_belt_uploads"
RESULT_DIR = PROJECT_ROOT / "outputs" / "c_belt_results"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

HOST = "0.0.0.0"
PORT = 50012

IMG_SIZE = 1280
DEFAULT_CONF = 0.35
DEFAULT_IOU = 0.50

TRACKER = "bytetrack.yaml"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
}


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024


# ============================================================
# MODEL FINDER
# ============================================================

def find_best_model():
    """
    Find trained best.pt automatically.

    Priority:
    1. YOLO_MODEL environment variable
    2. Known C-belt YOLO26L segmentation paths
    3. Any best.pt under runs/
    """

    # --------------------------------------------------------
    # 1. Environment variable
    # --------------------------------------------------------

    env_model = os.environ.get("YOLO_MODEL", "").strip()

    if env_model:
        env_path = Path(env_model).expanduser()

        if env_path.is_file():
            return env_path.resolve()

        print(
            f"WARNING: YOLO_MODEL was set but file was not found:\n"
            f"{env_path}"
        )

    # --------------------------------------------------------
    # 2. Known paths
    # --------------------------------------------------------

    preferred_paths = [
        (
            RUNS_ROOT
            / "segment"
            / "runs"
            / "segment"
            / "final_c_belt_sack_yolo26l_seg_v1"
            / "weights"
            / "best.pt"
        ),

        (
            RUNS_ROOT
            / "segment"
            / "final_c_belt_sack_yolo26l_seg_v1"
            / "weights"
            / "best.pt"
        ),

        (
            RUNS_ROOT
            / "final_c_belt_sack_yolo26l_seg_v1"
            / "weights"
            / "best.pt"
        ),
    ]

    for path in preferred_paths:
        print(f"Checking model path: {path}")

        if path.is_file():
            print(f"FOUND MODEL: {path}")
            return path.resolve()

    # --------------------------------------------------------
    # 3. Search all runs
    # --------------------------------------------------------

    candidates = []

    if RUNS_ROOT.exists():
        try:
            candidates = list(
                RUNS_ROOT.rglob("best.pt")
            )
        except Exception as exc:
            print(f"Model search error: {exc}")

    if not candidates:
        raise FileNotFoundError(
            "\n"
            "============================================================\n"
            "ERROR: best.pt NOT FOUND\n"
            "============================================================\n"
            f"Searched inside:\n{RUNS_ROOT}\n\n"
            "Run this command to verify:\n\n"
            "find ~/NEW_SACK_COUNT/runs -type f -name best.pt -print\n\n"
            "Expected model is similar to:\n"
            "~/NEW_SACK_COUNT/runs/segment/runs/segment/"
            "final_c_belt_sack_yolo26l_seg_v1/weights/best.pt\n"
            "============================================================\n"
        )

    # Prefer C-belt models
    c_belt_candidates = [
        p
        for p in candidates
        if "c_belt" in str(p).lower()
    ]

    if c_belt_candidates:
        candidates = c_belt_candidates

    candidates.sort(
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    print("\nAvailable best.pt candidates:")

    for candidate in candidates:
        print(f"  {candidate}")

    selected = candidates[0].resolve()

    print(f"\nSelected model:\n{selected}\n")

    return selected


# ============================================================
# DEVICE
# ============================================================

def get_device():
    try:
        import torch

        if torch.cuda.is_available():
            return 0

    except Exception:
        pass

    return "cpu"


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = find_best_model()
DEVICE = get_device()

print()
print("=" * 75)
print("C-BELT RICE BAG COUNTING SYSTEM")
print("=" * 75)

print(f"Project Root : {PROJECT_ROOT}")
print(f"Runs Root    : {RUNS_ROOT}")
print(f"Model Path   : {MODEL_PATH}")
print(f"Model Exists : {MODEL_PATH.exists()}")
print(f"Device       : {DEVICE}")

try:
    import torch

    print(f"PyTorch      : {torch.__version__}")
    print(f"CUDA         : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(
            "GPU          : "
            f"{torch.cuda.get_device_name(0)}"
        )

except Exception as exc:
    print(f"Torch info error: {exc}")

try:
    import ultralytics

    print(
        "Ultralytics  : "
        f"{ultralytics.__version__}"
    )

except Exception:
    pass

print("=" * 75)
print("Loading YOLO model...")

model = YOLO(str(MODEL_PATH))

print("YOLO model loaded successfully.")
print("=" * 75)


# ============================================================
# HTML UI
# ============================================================

HTML = r"""
<!DOCTYPE html>
<html>

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>C-Belt Rice Bag Counting</title>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    padding: 25px;
    background: #111827;
    color: white;
    font-family: Arial, sans-serif;
}

.container {
    max-width: 1400px;
    margin: auto;
}

.card {
    background: #1f2937;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
}

h1,
h2 {
    margin-top: 0;
}

.info {
    color: #d1d5db;
    line-height: 1.7;
}

.controls {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    align-items: center;
    margin-top: 15px;
}

button,
input {
    padding: 10px 14px;
    border-radius: 7px;
    border: 1px solid #4b5563;
    font-size: 15px;
}

button {
    background: #374151;
    color: white;
    cursor: pointer;
}

button:hover {
    background: #4b5563;
}

button.active-red {
    background: #dc2626;
}

button.active-blue {
    background: #2563eb;
}

button.primary {
    background: #059669;
    border-color: #059669;
}

button.primary:hover {
    background: #047857;
}

button:disabled {
    opacity: 0.45;
    cursor: not-allowed;
}

#fileInput {
    background: #111827;
    width: 100%;
    max-width: 600px;
}

#canvasWrapper {
    width: 100%;
    margin-top: 20px;
    background: black;
    border: 2px solid #374151;
    border-radius: 8px;
    overflow: hidden;
    text-align: center;
}

#drawCanvas {
    display: block;
    width: 100%;
    height: auto;
    max-height: 72vh;
    object-fit: contain;
    margin: auto;
    background: black;
    cursor: crosshair;
    touch-action: none;
}

.line-status {
    margin-top: 15px;
    padding: 12px;
    background: #111827;
    border-radius: 8px;
    font-weight: bold;
}

.red {
    color: #ff3333;
}

.blue {
    color: #3399ff;
}

.yellow {
    color: #facc15;
}

.status {
    margin-top: 15px;
    padding: 12px;
    background: #111827;
    border-radius: 8px;
    white-space: pre-wrap;
    color: #93c5fd;
}

.instructions {
    margin-top: 15px;
    padding: 15px;
    background: #111827;
    border-radius: 8px;
    line-height: 1.7;
}

.result-box {
    margin-top: 15px;
}

#resultImage,
#resultVideo {
    width: 100%;
    max-height: 75vh;
    object-fit: contain;
    background: black;
    border-radius: 8px;
    display: none;
}

.counts {
    font-size: 25px;
    font-weight: bold;
    margin-top: 15px;
}

.small {
    color: #9ca3af;
    font-size: 13px;
}

</style>

</head>


<body>

<div class="container">

<div class="card">

<h1>
Warehouse C-Belt Rice Bag Counting
</h1>

<div class="info">

<b>How to draw counting lines:</b>

<br>

1. Upload your image/video.

<br>

2. For video, its first frame will appear below.

<br>

3. Click <b class="red">DRAW RED LINE</b>.

<br>

4. Now click + drag directly on the image to create the RED unloading line.

<br>

5. Click <b class="blue">DRAW BLUE LINE</b>.

<br>

6. Click + drag directly on the image to create the BLUE loading line.

<br>

7. Click <b>PROCESS</b>.

</div>


<div class="controls">

<input
    id="fileInput"
    type="file"
    accept="image/*,video/*"
>

</div>


<div class="controls">

<label>
Confidence:
<input
    id="confidence"
    type="number"
    min="0.05"
    max="0.95"
    step="0.05"
    value="0.35"
>
</label>

<button
    id="redButton"
    onclick="selectRedMode()"
>
DRAW RED LINE
</button>

<button
    id="blueButton"
    onclick="selectBlueMode()"
>
DRAW BLUE LINE
</button>

<button onclick="clearRed()">
CLEAR RED
</button>

<button onclick="clearBlue()">
CLEAR BLUE
</button>

<button onclick="clearAllLines()">
CLEAR ALL
</button>

<button
    id="processButton"
    class="primary"
    onclick="processMedia()"
    disabled
>
PROCESS
</button>

</div>


<div class="instructions">

<b>Current drawing mode:</b>

<span id="drawingMode" class="yellow">
NONE
</span>

<br>

<span class="small">
Click a line button first, then click and drag on the image.
The line will be created exactly where you drag.
</span>

</div>


<div class="line-status" id="lineStatus">

<span class="red">
RED / UNLOADING:
</span>

not drawn

&nbsp;&nbsp; | &nbsp;&nbsp;

<span class="blue">
BLUE / LOADING:
</span>

not drawn

</div>


<div id="status" class="status">

Waiting for image/video upload...

</div>


<div id="canvasWrapper">

<canvas id="drawCanvas"></canvas>

</div>

</div>


<div class="card">

<h2>
Result
</h2>

<img
    id="resultImage"
    alt="Processed result"
>

<video
    id="resultVideo"
    controls
    playsinline
>
</video>

<div
    id="resultText"
    class="result-box"
>
</div>

</div>

</div>


<script>


// ============================================================
// GLOBAL STATE
// ============================================================

let selectedFile = null;

let mediaWidth = 0;
let mediaHeight = 0;

let sourceImage = null;

let redLine = null;
let blueLine = null;

let drawingMode = null;

let drawing = false;

let startX = 0;
let startY = 0;

let tempX = 0;
let tempY = 0;


// ============================================================
// DOM
// ============================================================

const fileInput =
    document.getElementById("fileInput");

const canvas =
    document.getElementById("drawCanvas");

const ctx =
    canvas.getContext("2d");

const processButton =
    document.getElementById("processButton");

const statusBox =
    document.getElementById("status");

const lineStatus =
    document.getElementById("lineStatus");

const drawingModeText =
    document.getElementById("drawingMode");

const redButton =
    document.getElementById("redButton");

const blueButton =
    document.getElementById("blueButton");

const resultImage =
    document.getElementById("resultImage");

const resultVideo =
    document.getElementById("resultVideo");


// ============================================================
// FILE UPLOAD
// ============================================================

fileInput.addEventListener(
    "change",
    handleFile
);


async function handleFile() {

    if (!fileInput.files.length) {
        return;
    }

    selectedFile = fileInput.files[0];

    redLine = null;
    blueLine = null;

    drawingMode = null;
    drawing = false;

    updateDrawingMode();
    updateLineStatus();

    const fileUrl =
        URL.createObjectURL(selectedFile);

    statusBox.textContent =
        "Loading media...";

    processButton.disabled = true;

    if (
        selectedFile.type.startsWith("image/")
    ) {

        await loadImage(fileUrl);

    } else if (
        selectedFile.type.startsWith("video/")
    ) {

        await loadVideoFirstFrame(fileUrl);

    } else {

        statusBox.textContent =
            "Unsupported file type.";

    }
}


// ============================================================
// LOAD IMAGE
// ============================================================

function loadImage(url) {

    return new Promise(
        function(resolve, reject) {

            const image =
                new Image();

            image.onload =
                function() {

                    sourceImage = image;

                    mediaWidth =
                        image.naturalWidth;

                    mediaHeight =
                        image.naturalHeight;

                    setupCanvas();

                    processButton.disabled =
                        false;

                    statusBox.textContent =
                        "Image loaded.\n\n"
                        +
                        "Click DRAW RED LINE, "
                        +
                        "then drag on the image.";

                    resolve();

                };

            image.onerror =
                function() {

                    statusBox.textContent =
                        "Could not load image.";

                    reject(
                        new Error(
                            "Image loading failed."
                        )
                    );

                };

            image.src = url;

        }
    );
}


// ============================================================
// LOAD VIDEO FIRST FRAME
// ============================================================

function loadVideoFirstFrame(url) {

    return new Promise(
        function(resolve, reject) {

            const video =
                document.createElement("video");

            video.muted = true;
            video.preload = "auto";
            video.playsInline = true;

            video.src = url;

            video.addEventListener(
                "loadedmetadata",
                function() {

                    mediaWidth =
                        video.videoWidth;

                    mediaHeight =
                        video.videoHeight;

                    video.currentTime = 0;

                },
                { once: true }
            );

            video.addEventListener(
                "seeked",
                function() {

                    const tempCanvas =
                        document.createElement(
                            "canvas"
                        );

                    tempCanvas.width =
                        mediaWidth;

                    tempCanvas.height =
                        mediaHeight;

                    const tempCtx =
                        tempCanvas.getContext(
                            "2d"
                        );

                    tempCtx.drawImage(
                        video,
                        0,
                        0,
                        mediaWidth,
                        mediaHeight
                    );

                    const image =
                        new Image();

                    image.onload =
                        function() {

                            sourceImage =
                                image;

                            setupCanvas();

                            processButton.disabled =
                                false;

                            statusBox.textContent =
                                "Video first frame loaded.\n\n"
                                +
                                "Click DRAW RED LINE, "
                                +
                                "then drag on the image.";

                            resolve();

                        };

                    image.onerror =
                        function() {

                            reject(
                                new Error(
                                    "Could not create video preview."
                                )
                            );

                        };

                    image.src =
                        tempCanvas.toDataURL(
                            "image/jpeg",
                            0.92
                        );

                },
                { once: true }
            );

            video.addEventListener(
                "error",
                function() {

                    statusBox.textContent =
                        "Could not load video.";

                    reject(
                        new Error(
                            "Video loading failed."
                        )
                    );

                },
                { once: true }
            );

            video.load();

        }
    );
}


// ============================================================
// CANVAS SETUP
// ============================================================

function setupCanvas() {

    if (
        !mediaWidth ||
        !mediaHeight ||
        !sourceImage
    ) {
        return;
    }

    canvas.width =
        mediaWidth;

    canvas.height =
        mediaHeight;

    redrawCanvas();
}


// ============================================================
// GET CANVAS POINT
// ============================================================

function getCanvasPoint(event) {

    const rect =
        canvas.getBoundingClientRect();

    const scaleX =
        canvas.width / rect.width;

    const scaleY =
        canvas.height / rect.height;

    let x =
        (event.clientX - rect.left)
        * scaleX;

    let y =
        (event.clientY - rect.top)
        * scaleY;

    x =
        Math.max(
            0,
            Math.min(
                canvas.width,
                x
            )
        );

    y =
        Math.max(
            0,
            Math.min(
                canvas.height,
                y
            )
        );

    return {
        x: x,
        y: y
    };
}


// ============================================================
// SELECT RED MODE
// ============================================================

function selectRedMode() {

    if (!sourceImage) {

        alert(
            "Upload an image/video first."
        );

        return;
    }

    drawingMode = "red";

    updateDrawingMode();

    statusBox.textContent =
        "RED LINE MODE ACTIVE.\n\n"
        +
        "Now click and drag on the image "
        +
        "where you want the UNLOADING line.";

}


// ============================================================
// SELECT BLUE MODE
// ============================================================

function selectBlueMode() {

    if (!sourceImage) {

        alert(
            "Upload an image/video first."
        );

        return;
    }

    drawingMode = "blue";

    updateDrawingMode();

    statusBox.textContent =
        "BLUE LINE MODE ACTIVE.\n\n"
        +
        "Now click and drag on the image "
        +
        "where you want the LOADING line.";

}


// ============================================================
// POINTER DOWN
// ============================================================

canvas.addEventListener(
    "pointerdown",
    function(event) {

        if (!drawingMode) {

            statusBox.textContent =
                "First click DRAW RED LINE "
                +
                "or DRAW BLUE LINE.";

            return;
        }

        event.preventDefault();

        canvas.setPointerCapture(
            event.pointerId
        );

        const point =
            getCanvasPoint(event);

        startX =
            point.x;

        startY =
            point.y;

        tempX =
            point.x;

        tempY =
            point.y;

        drawing = true;

        redrawCanvas();

        drawTemporaryLine();

    }
);


// ============================================================
// POINTER MOVE
// ============================================================

canvas.addEventListener(
    "pointermove",
    function(event) {

        if (!drawing) {
            return;
        }

        event.preventDefault();

        const point =
            getCanvasPoint(event);

        tempX =
            point.x;

        tempY =
            point.y;

        redrawCanvas();

        drawTemporaryLine();

    }
);


// ============================================================
// POINTER UP
// ============================================================

canvas.addEventListener(
    "pointerup",
    function(event) {

        if (!drawing) {
            return;
        }

        event.preventDefault();

        try {
            canvas.releasePointerCapture(
                event.pointerId
            );
        } catch (e) {
            // Ignore.
        }

        const point =
            getCanvasPoint(event);

        tempX =
            point.x;

        tempY =
            point.y;

        drawing = false;

        const length =
            Math.hypot(
                tempX - startX,
                tempY - startY
            );

        if (length < 30) {

            statusBox.textContent =
                "Line is too short.\n\n"
                +
                "Please click and drag a longer line.";

            redrawCanvas();

            return;
        }

        const newLine = {
            x1: startX,
            y1: startY,
            x2: tempX,
            y2: tempY
        };

        if (drawingMode === "red") {

            redLine =
                newLine;

            statusBox.textContent =
                "RED / UNLOADING line created.\n\n"
                +
                "You can now click DRAW BLUE LINE "
                +
                "to create the loading line.";

        } else {

            blueLine =
                newLine;

            statusBox.textContent =
                "BLUE / LOADING line created.\n\n"
                +
                "Click PROCESS when ready.";

        }

        drawingMode = null;

        updateDrawingMode();

        updateLineStatus();

        redrawCanvas();

    }
);


// ============================================================
// POINTER CANCEL
// ============================================================

canvas.addEventListener(
    "pointercancel",
    function(event) {

        drawing = false;

        try {
            canvas.releasePointerCapture(
                event.pointerId
            );
        } catch (e) {
            // Ignore.
        }

        redrawCanvas();
    }
);


// ============================================================
// DRAW TEMPORARY LINE
// ============================================================

function drawTemporaryLine() {

    ctx.lineWidth =
        Math.max(
            4,
            canvas.width / 400
        );

    ctx.strokeStyle =
        drawingMode === "red"
        ? "#ff0000"
        : "#0066ff";

    ctx.beginPath();

    ctx.moveTo(
        startX,
        startY
    );

    ctx.lineTo(
        tempX,
        tempY
    );

    ctx.stroke();

    // Start point
    ctx.fillStyle =
        "#ffffff";

    ctx.beginPath();

    ctx.arc(
        startX,
        startY,
        8,
        0,
        Math.PI * 2
    );

    ctx.fill();

}


// ============================================================
// DRAW SAVED LINE
// ============================================================

function drawSavedLine(
    line,
    color,
    label
) {

    if (!line) {
        return;
    }

    ctx.lineWidth =
        Math.max(
            5,
            canvas.width / 350
        );

    ctx.strokeStyle =
        color;

    ctx.beginPath();

    ctx.moveTo(
        line.x1,
        line.y1
    );

    ctx.lineTo(
        line.x2,
        line.y2
    );

    ctx.stroke();


    // Start point
    ctx.fillStyle =
        "#ffffff";

    ctx.beginPath();

    ctx.arc(
        line.x1,
        line.y1,
        8,
        0,
        Math.PI * 2
    );

    ctx.fill();


    // End point
    ctx.beginPath();

    ctx.arc(
        line.x2,
        line.y2,
        8,
        0,
        Math.PI * 2
    );

    ctx.fill();


    // Label
    const fontSize =
        Math.max(
            20,
            canvas.width / 55
        );

    ctx.font =
        "bold " +
        fontSize +
        "px Arial";

    ctx.fillStyle =
        color;

    ctx.fillText(
        label,
        line.x1 + 15,
        Math.max(
            30,
            line.y1 - 15
        )
    );
}


// ============================================================
// REDRAW CANVAS
// ============================================================

function redrawCanvas() {

    if (!sourceImage) {
        return;
    }

    ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
    );

    ctx.drawImage(
        sourceImage,
        0,
        0,
        mediaWidth,
        mediaHeight
    );

    drawSavedLine(
        redLine,
        "#ff0000",
        "UNLOADING"
    );

    drawSavedLine(
        blueLine,
        "#0066ff",
        "LOADING"
    );
}


// ============================================================
// CLEAR RED
// ============================================================

function clearRed() {

    redLine = null;

    redrawCanvas();

    updateLineStatus();

    statusBox.textContent =
        "RED line cleared.";

}


// ============================================================
// CLEAR BLUE
// ============================================================

function clearBlue() {

    blueLine = null;

    redrawCanvas();

    updateLineStatus();

    statusBox.textContent =
        "BLUE line cleared.";

}


// ============================================================
// CLEAR ALL
// ============================================================

function clearAllLines() {

    redLine = null;
    blueLine = null;

    drawingMode = null;
    drawing = false;

    redrawCanvas();

    updateDrawingMode();
    updateLineStatus();

    statusBox.textContent =
        "All lines cleared.";

}


// ============================================================
// UPDATE DRAWING MODE
// ============================================================

function updateDrawingMode() {

    if (drawingMode === "red") {

        drawingModeText.textContent =
            "RED / UNLOADING";

        drawingModeText.className =
            "red";

        redButton.classList.add(
            "active-red"
        );

        blueButton.classList.remove(
            "active-blue"
        );

    } else if (drawingMode === "blue") {

        drawingModeText.textContent =
            "BLUE / LOADING";

        drawingModeText.className =
            "blue";

        blueButton.classList.add(
            "active-blue"
        );

        redButton.classList.remove(
            "active-red"
        );

    } else {

        drawingModeText.textContent =
            "NONE";

        drawingModeText.className =
            "yellow";

        redButton.classList.remove(
            "active-red"
        );

        blueButton.classList.remove(
            "active-blue"
        );
    }
}


// ============================================================
// UPDATE LINE STATUS
// ============================================================

function updateLineStatus() {

    lineStatus.innerHTML =
        "<span class='red'>"
        +
        "RED / UNLOADING:"
        +
        "</span> "
        +
        (redLine ? "DRAWN" : "NOT DRAWN")
        +
        "&nbsp;&nbsp; | &nbsp;&nbsp;"
        +
        "<span class='blue'>"
        +
        "BLUE / LOADING:"
        +
        "</span> "
        +
        (blueLine ? "DRAWN" : "NOT DRAWN");

}


// ============================================================
// WINDOW RESIZE
// ============================================================

window.addEventListener(
    "resize",
    function() {

        redrawCanvas();

    }
);


// ============================================================
// PROCESS MEDIA
// ============================================================

async function processMedia() {

    if (!selectedFile) {

        alert(
            "Upload image/video first."
        );

        return;
    }

    if (!redLine && !blueLine) {

        alert(
            "Draw at least one line first."
        );

        return;
    }

    const form =
        new FormData();

    form.append(
        "file",
        selectedFile
    );

    form.append(
        "confidence",
        document.getElementById(
            "confidence"
        ).value
    );

    form.append(
        "red_line",
        redLine
            ? JSON.stringify(redLine)
            : ""
    );

    form.append(
        "blue_line",
        blueLine
            ? JSON.stringify(blueLine)
            : ""
    );

    processButton.disabled =
        true;

    statusBox.textContent =
        "Processing...\n\n"
        +
        "YOLO is running on the GPU.\n"
        +
        "Please wait.";

    resultImage.style.display =
        "none";

    resultVideo.style.display =
        "none";

    try {

        const response =
            await fetch(
                "/process",
                {
                    method: "POST",
                    body: form
                }
            );

        const data =
            await response.json();

        if (
            !response.ok ||
            !data.success
        ) {

            throw new Error(
                data.error ||
                "Processing failed."
            );
        }


        const resultUrl =
            data.result_url
            +
            "?t="
            +
            Date.now();


        if (
            data.media_type === "image"
        ) {

            resultImage.src =
                resultUrl;

            resultImage.style.display =
                "block";

        } else {

            resultVideo.src =
                resultUrl;

            resultVideo.style.display =
                "block";

            resultVideo.load();

        }


        document.getElementById(
            "resultText"
        ).innerHTML =

            "<div class='counts'>"
            +

            "<span class='red'>"
            +
            "UNLOADING: "
            +
            data.unloading_count
            +
            "</span>"

            +

            "&nbsp;&nbsp;&nbsp;"

            +

            "<span class='blue'>"
            +
            "LOADING: "
            +
            data.loading_count
            +
            "</span>"

            +

            "</div>"

            +

            "<p>"
            +
            "Maximum simultaneous detections: "
            +
            data.detected_count
            +
            "</p>"

            +

            "<p>"
            +
            "Processing time: "
            +
            data.processing_seconds
            +
            " seconds"
            +
            "</p>";


        statusBox.textContent =
            "Processing completed successfully.";

    } catch (error) {

        statusBox.textContent =
            "ERROR:\n\n"
            +
            error.message;

    }

    processButton.disabled =
        false;
}

</script>

</body>

</html>
"""


# ============================================================
# LINE PARSER
# ============================================================

def parse_line(value):
    if not value:
        return None

    try:
        data = json.loads(value)
    except Exception:
        return None

    required = [
        "x1",
        "y1",
        "x2",
        "y2",
    ]

    if not isinstance(data, dict):
        return None

    if not all(
        key in data
        for key in required
    ):
        return None

    try:
        line = {
            key: float(data[key])
            for key in required
        }
    except Exception:
        return None

    length = float(
        np.hypot(
            line["x2"] - line["x1"],
            line["y2"] - line["y1"],
        )
    )

    if length < 20:
        return None

    return line


# ============================================================
# NORMALIZED LINE SIDE
# ============================================================

def point_side(x, y, line):
    """
    Signed perpendicular distance from point to infinite line.

    Unlike raw cross-product, this is normalized by line length,
    so tolerance is expressed approximately in pixels.
    """

    dx = line["x2"] - line["x1"]
    dy = line["y2"] - line["y1"]

    length = float(
        np.hypot(dx, dy)
    )

    if length <= 0:
        return 0.0

    cross = (
        dx * (y - line["y1"])
        -
        dy * (x - line["x1"])
    )

    return float(
        cross / length
    )


# ============================================================
# MASK CENTROID
# ============================================================

def mask_centroid(mask, box):

    ys, xs = np.where(mask > 0)

    if len(xs) > 0:

        return (
            float(np.mean(xs)),
            float(np.mean(ys)),
        )

    x1, y1, x2, y2 = box

    return (
        float((x1 + x2) / 2.0),
        float((y1 + y2) / 2.0),
    )


# ============================================================
# BBOX IOU
# ============================================================

def bbox_iou(box_a, box_b):

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)

    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(
        0.0,
        ix2 - ix1,
    )

    ih = max(
        0.0,
        iy2 - iy1,
    )

    intersection = iw * ih

    area_a = (
        max(0.0, ax2 - ax1)
        *
        max(0.0, ay2 - ay1)
    )

    area_b = (
        max(0.0, bx2 - bx1)
        *
        max(0.0, by2 - by1)
    )

    union = (
        area_a
        +
        area_b
        -
        intersection
    )

    if union <= 0:
        return 0.0

    return float(
        intersection / union
    )


# ============================================================
# MASK IOU
# ============================================================

def mask_iou(mask_a, mask_b):

    a = mask_a.astype(bool)
    b = mask_b.astype(bool)

    intersection = np.logical_and(
        a,
        b,
    ).sum()

    union = np.logical_or(
        a,
        b,
    ).sum()

    if union == 0:
        return 0.0

    return float(
        intersection / union
    )


# ============================================================
# EXTRACT DETECTIONS
# ============================================================

def extract_detections(
    result,
    frame_shape,
):

    frame_h, frame_w = frame_shape[:2]

    detections = []

    if result.boxes is None:
        return detections

    boxes = (
        result.boxes.xyxy
        .cpu()
        .numpy()
    )

    if result.boxes.conf is not None:

        confs = (
            result.boxes.conf
            .cpu()
            .numpy()
        )

    else:

        confs = np.ones(
            len(boxes),
            dtype=np.float32,
        )

    if result.boxes.id is not None:

        ids = (
            result.boxes.id
            .int()
            .cpu()
            .tolist()
        )

    else:

        ids = list(
            range(len(boxes))
        )

    if result.masks is not None:

        masks = (
            result.masks.data
            .cpu()
            .numpy()
        )

    else:

        masks = None


    for i, box in enumerate(boxes):

        x1, y1, x2, y2 = map(
            float,
            box,
        )

        x1 = max(
            0.0,
            min(
                x1,
                frame_w - 1,
            ),
        )

        y1 = max(
            0.0,
            min(
                y1,
                frame_h - 1,
            ),
        )

        x2 = max(
            0.0,
            min(
                x2,
                frame_w - 1,
            ),
        )

        y2 = max(
            0.0,
            min(
                y2,
                frame_h - 1,
            ),
        )

        if x2 <= x1 or y2 <= y1:
            continue

        mask = np.zeros(
            (
                frame_h,
                frame_w,
            ),
            dtype=np.uint8,
        )


        if (
            masks is not None
            and i < len(masks)
        ):

            raw_mask = masks[i]

            resized = cv2.resize(
                raw_mask,
                (
                    frame_w,
                    frame_h,
                ),
                interpolation=cv2.INTER_NEAREST,
            )

            mask = (
                resized > 0.5
            ).astype(np.uint8)


        detections.append(
            {
                "box": [
                    x1,
                    y1,
                    x2,
                    y2,
                ],
                "conf": float(
                    confs[i]
                ),
                "id": int(
                    ids[i]
                ),
                "mask": mask,
            }
        )


    # --------------------------------------------------------
    # Highest confidence first
    # --------------------------------------------------------

    detections.sort(
        key=lambda d: d["conf"],
        reverse=True,
    )


    # --------------------------------------------------------
    # Duplicate suppression
    # --------------------------------------------------------

    kept = []

    for detection in detections:

        duplicate = False

        for previous in kept:

            box_overlap = bbox_iou(
                detection["box"],
                previous["box"],
            )

            if box_overlap >= 0.70:

                current_mask =detection["mask"]

                previous_mask =previous["mask"]

                if (
                    current_mask.sum() > 0
                    and previous_mask.sum() > 0
                ):

                    mask_overlap = mask_iou(
                        current_mask,
                        previous_mask,
                    )

                    if mask_overlap >= 0.60:

                        duplicate = True
                        break

                else:

                    duplicate = True
                    break

        if not duplicate:
            kept.append(detection)

    return kept


# ============================================================
# DRAW DETECTION
# ============================================================

def draw_detection(
    frame,
    detection,
):

    mask = detection["mask"]

    if mask.sum() > 0:

        overlay = frame.copy()

        overlay[mask > 0] = (
            0.45 * overlay[mask > 0]
            +
            0.55 * np.array(
                [0, 255, 0]
            )
        ).astype(np.uint8)

        frame[:] = overlay


        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if contours:

            cv2.drawContours(
                frame,
                contours,
                -1,
                (0, 255, 0),
                2,
            )


    cx, cy = mask_centroid(
        mask,
        detection["box"],
    )


    cv2.circle(
        frame,
        (
            int(cx),
            int(cy),
        ),
        5,
        (0, 255, 255),
        -1,
    )


    cv2.putText(
        frame,
        f"ID {detection['id']}",
        (
            int(cx) + 8,
            int(cy) - 8,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


# ============================================================
# DRAW COUNTING LINE
# ============================================================

def draw_counting_line(
    frame,
    line,
    color,
    label,
):

    if line is None:
        return

    p1 = (
        int(line["x1"]),
        int(line["y1"]),
    )

    p2 = (
        int(line["x2"]),
        int(line["y2"]),
    )

    cv2.line(
        frame,
        p1,
        p2,
        color,
        5,
        cv2.LINE_AA,
    )

    cv2.circle(
        frame,
        p1,
        7,
        (255, 255, 255),
        -1,
    )

    cv2.circle(
        frame,
        p2,
        7,
        (255, 255, 255),
        -1,
    )

    cv2.putText(
        frame,
        label,
        (
            p1[0] + 10,
            max(30, p1[1] - 10),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )


# ============================================================
# IMAGE PROCESSING
# ============================================================

def process_image(
    input_path,
    output_path,
    conf,
    red_line,
    blue_line,
):

    image = cv2.imread(
        str(input_path)
    )

    if image is None:
        raise RuntimeError(
            "Could not read image."
        )

    results = model.predict(
        source=image,
        conf=conf,
        iou=DEFAULT_IOU,
        imgsz=IMG_SIZE,
        device=DEVICE,
        verbose=False,
    )

    result = results[0]

    detections = extract_detections(
        result,
        image.shape,
    )

    output = image.copy()

    for detection in detections:

        draw_detection(
            output,
            detection,
        )


    draw_counting_line(
        output,
        red_line,
        (0, 0, 255),
        "UNLOADING",
    )

    draw_counting_line(
        output,
        blue_line,
        (255, 0, 0),
        "LOADING",
    )


    cv2.putText(
        output,
        f"DETECTED: {len(detections)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


    if not cv2.imwrite(
        str(output_path),
        output,
    ):

        raise RuntimeError(
            "Could not save output image."
        )


    return (
        len(detections),
        0,
        0,
    )


# ============================================================
# VIDEO PROCESSING
# ============================================================

def process_video(
    input_path,
    output_path,
    conf,
    red_line,
    blue_line,
):

    cap = cv2.VideoCapture(
        str(input_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open video."
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


    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )


    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )


    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (
            width,
            height,
        ),
    )


    if not writer.isOpened():

        cap.release()

        raise RuntimeError(
            "Could not create output video."
        )


    red_previous = {}
    blue_previous = {}

    red_counted = set()
    blue_counted = set()

    max_detected = 0

    frame_number = 0


    # Pixel tolerance from counting line.
    tolerance = max(
        8.0,
        min(width, height) * 0.005,
    )


    try:

        while True:

            ok, frame = cap.read()

            if not ok:
                break

            frame_number += 1


            results = model.track(
                source=frame,
                conf=conf,
                iou=DEFAULT_IOU,
                imgsz=IMG_SIZE,
                tracker=TRACKER,
                persist=True,
                device=DEVICE,
                verbose=False,
            )


            result = results[0]


            detections = extract_detections(
                result,
                frame.shape,
            )


            max_detected = max(
                max_detected,
                len(detections),
            )


            # =================================================
            # TRACK EACH BAG
            # =================================================

            for detection in detections:

                draw_detection(
                    frame,
                    detection,
                )


                track_id =detection["id"]


                cx, cy = mask_centroid(
                    detection["mask"],
                    detection["box"],
                )


                # =============================================
                # RED / UNLOADING
                # =============================================

                if red_line is not None:

                    distance = point_side(
                        cx,
                        cy,
                        red_line,
                    )


                    if abs(distance) > tolerance:

                        current_side = (
                            1
                            if distance > 0
                            else -1
                        )


                        previous_side = (
                            red_previous.get(
                                track_id
                            )
                        )


                        if (
                            previous_side is not None
                            and
                            previous_side != current_side
                            and
                            track_id not in red_counted
                        ):

                            red_counted.add(
                                track_id
                            )


                        red_previous[
                            track_id
                        ] = current_side


                # =============================================
                # BLUE / LOADING
                # =============================================

                if blue_line is not None:

                    distance = point_side(
                        cx,
                        cy,
                        blue_line,
                    )


                    if abs(distance) > tolerance:

                        current_side = (
                            1
                            if distance > 0
                            else -1
                        )


                        previous_side = (
                            blue_previous.get(
                                track_id
                            )
                        )


                        if (
                            previous_side is not None
                            and
                            previous_side != current_side
                            and
                            track_id not in blue_counted
                        ):

                            blue_counted.add(
                                track_id
                            )


                        blue_previous[
                            track_id
                        ] = current_side


            # =================================================
            # DRAW LINES
            # =================================================

            draw_counting_line(
                frame,
                red_line,
                (0, 0, 255),
                f"UNLOADING: {len(red_counted)}",
            )


            draw_counting_line(
                frame,
                blue_line,
                (255, 0, 0),
                f"LOADING: {len(blue_counted)}",
            )


            # =================================================
            # STATUS
            # =================================================

            cv2.putText(
                frame,
                f"Frame: {frame_number}/{total_frames}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )


            cv2.putText(
                frame,
                f"UNLOADING: {len(red_counted)}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )


            cv2.putText(
                frame,
                f"LOADING: {len(blue_counted)}",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )


            writer.write(frame)


    finally:

        cap.release()
        writer.release()


    return (
        max_detected,
        len(red_counted),
        len(blue_counted),
    )


# ============================================================
# HOME
# ============================================================

@app.route(
    "/",
    methods=["GET"],
)
def home():

    return render_template_string(
        HTML
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route(
    "/health",
    methods=["GET"],
)
def health():

    return jsonify(
        {
            "status": "ok",
            "model": str(MODEL_PATH),
            "model_exists": MODEL_PATH.exists(),
            "device": str(DEVICE),
            "port": PORT,
        }
    )


# ============================================================
# PROCESS
# ============================================================

@app.route(
    "/process",
    methods=["POST"],
)
def process():

    start_time = time.time()

    try:

        if "file" not in request.files:

            return jsonify(
                {
                    "success": False,
                    "error": "No file uploaded.",
                }
            ), 400


        uploaded =request.files["file"]


        if not uploaded.filename:

            return jsonify(
                {
                    "success": False,
                    "error": "Empty filename.",
                }
            ), 400


        filename = secure_filename(
            uploaded.filename
        )


        suffix = Path(
            filename
        ).suffix.lower()


        if (
            suffix not in IMAGE_EXTENSIONS
            and
            suffix not in VIDEO_EXTENSIONS
        ):

            return jsonify(
                {
                    "success": False,
                    "error":
                        f"Unsupported file type: {suffix}",
                }
            ), 400


        try:

            conf = float(
                request.form.get(
                    "confidence",
                    DEFAULT_CONF,
                )
            )

        except ValueError:

            conf = DEFAULT_CONF


        conf = max(
            0.05,
            min(0.95, conf),
        )


        red_line = parse_line(
            request.form.get(
                "red_line",
                "",
            )
        )


        blue_line = parse_line(
            request.form.get(
                "blue_line",
                "",
            )
        )


        timestamp = int(
            time.time() * 1000
        )


        input_path = (
            UPLOAD_DIR
            /
            f"{timestamp}_{filename}"
        )


        uploaded.save(
            str(input_path)
        )


        if suffix in IMAGE_EXTENSIONS:

            output_path = (
                RESULT_DIR
                /
                f"result_{timestamp}.jpg"
            )


            detected, unloading, loading = (
                process_image(
                    input_path,
                    output_path,
                    conf,
                    red_line,
                    blue_line,
                )
            )


            media_type = "image"


        else:

            output_path = (
                RESULT_DIR
                /
                f"result_{timestamp}.mp4"
            )


            detected, unloading, loading = (
                process_video(
                    input_path,
                    output_path,
                    conf,
                    red_line,
                    blue_line,
                )
            )


            media_type = "video"


        elapsed = (
            time.time()
            -
            start_time
        )


        print(
            f"Processed: {filename} | "
            f"Detected: {detected} | "
            f"Unloading: {unloading} | "
            f"Loading: {loading} | "
            f"Time: {elapsed:.2f}s"
        )


        return jsonify(
            {
                "success": True,
                "result_url":
                    f"/result/{output_path.name}",
                "media_type":
                    media_type,
                "detected_count":
                    detected,
                "unloading_count":
                    unloading,
                "loading_count":
                    loading,
                "processing_seconds":
                    round(elapsed, 2),
            }
        )


    except Exception as exc:

        import traceback

        traceback.print_exc()


        return jsonify(
            {
                "success": False,
                "error": str(exc),
            }
        ), 500


# ============================================================
# RESULT FILE
# ============================================================

@app.route(
    "/result/<filename>",
    methods=["GET"],
)
def result_file(filename):

    safe_name = secure_filename(
        filename
    )

    path = RESULT_DIR / safe_name


    if not path.is_file():

        return jsonify(
            {
                "error":
                    "Result file not found.",
            }
        ), 404


    return send_file(
        path
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 75)
    print("FLASK SERVER STARTING")
    print("=" * 75)

    print(
        f"Server URL:"
        f" http://127.0.0.1:{PORT}/"
    )

    print(
        f"Health check:"
        f" http://127.0.0.1:{PORT}/health"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "If this is a remote GPU server, "
        "127.0.0.1 means the GPU server itself."
    )

    print(
        "Use SSH port forwarding to open it "
        "in your local browser."
    )

    print("=" * 75)

    app.run(
        host=HOST,
        port=PORT,
        debug=False,
        threaded=False,
    )