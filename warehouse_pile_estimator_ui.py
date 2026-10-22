# ============================================================================
# BACKUP OF THE PREVIOUS WORKING VERSION (kept for reference / rollback).
# This was the bbox-based detection-first pile estimator, before syncing
# its detection stage with the dot + duplicate-dedup version now used in
# src/test_rice_bag_v9_ui.py. Restore by deleting everything below this
# block down to "END OF BACKUP", then un-prefix each line above it.
# ============================================================================
#
# """
# Warehouse Pile Estimator - detection-first workflow.
#
# Flow:
#     Image -> YOLO Detection (rice_bag_10_v9_best.pt) -> Corner/Edge sacks
#           -> Length / Breadth / Height sack counts (via reference lines)
#           -> Length x Breadth x Height -> Total sacks
#
# Stage 1 (detection) mirrors src/test_rice_bag_v9_ui.py exactly: same model,
# same confidence/IOU thresholds, same box/label drawing style. Stage 2 (pile
# estimation) is separate and only consumes Stage 1's output -- it never runs
# another model.
#
# Run: python warehouse_pile_estimator_ui.py
# Access: http://localhost:5006
# """
#
# from pathlib import Path
# import base64
# import statistics
#
# import cv2
# import numpy as np
# import torch
# from flask import Flask, jsonify, request, render_template_string
# from ultralytics import YOLO
#
#
# # ============================================================
# # PATHS & MODEL (same model + settings as src/test_rice_bag_v9_ui.py)
# # ============================================================
#
# ROOT = Path(__file__).resolve().parent
# MODEL_PATH = ROOT / "models" / "rice_bag_10_v9" / "rice_bag_10_v9_best.pt"
#
# if not MODEL_PATH.exists():
#     raise FileNotFoundError(
#         f"Trained model not found: {MODEL_PATH}\n"
#         "Train it first (python src/train_rice_bag_v9.py) or place the weights there."
#     )
#
# print(f"\nLoading V9 model: {MODEL_PATH}")
# model = YOLO(str(MODEL_PATH))
# print(f"Model loaded. Task: {model.task}")
#
# CONF_THRESHOLD = 0.50
# IOU_THRESHOLD = 0.45
# MAX_DETECTIONS = 500
# DEVICE = 0 if torch.cuda.is_available() else "cpu"
#
# app = Flask(__name__)
# app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
#
#
# # ============================================================
# # STAGE 1 - DETECTION (mirrors test_rice_bag_v9_ui.py /predict)
# # ============================================================
#
# def run_detection(image_bgr):
#     """Run YOLO detection and draw boxes/confidence/ids exactly like
#     src/test_rice_bag_v9_ui.py. Returns (annotated_image, boxes, avg_conf, edges)."""
#
#     results = model.predict(
#         source=image_bgr,
#         conf=CONF_THRESHOLD,
#         iou=IOU_THRESHOLD,
#         max_det=MAX_DETECTIONS,
#         device=DEVICE,
#         verbose=False,
#     )
#
#     result = results[0]
#     boxes_obj = result.boxes
#
#     annotated = image_bgr.copy()
#     boxes = []
#     confidences = []
#
#     if boxes_obj is not None and len(boxes_obj) > 0:
#         xyxy = boxes_obj.xyxy.cpu().numpy()
#         confs = boxes_obj.conf.cpu().numpy()
#
#         for i in range(len(boxes_obj)):
#             x1, y1, x2, y2 = map(float, xyxy[i])
#             conf = float(confs[i])
#             x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
#
#             confidences.append(conf)
#             box_id = i + 1
#
#             boxes.append({
#                 "id": box_id,
#                 "x1": x1, "y1": y1, "x2": x2, "y2": y2,
#                 "cx": (x1 + x2) / 2.0,
#                 "cy": (y1 + y2) / 2.0,
#                 "w": x2 - x1,
#                 "h": y2 - y1,
#                 "conf": conf,
#             })
#
#             cv2.rectangle(annotated, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
#             label = f"Bag {box_id} ({conf:.2f})"
#             cv2.putText(
#                 annotated, label,
#                 (x1i, max(y1i - 5, 20)),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
#             )
#
#     avg_conf = float(np.mean(confidences)) if confidences else 0.0
#
#     cv2.putText(
#         annotated,
#         f"Total: {len(boxes)} bags | Avg: {avg_conf:.3f}",
#         (10, 30),
#         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
#     )
#
#     # Corner / edge sacks: extreme detected boxes by center position.
#     # This is a lightweight extreme-point heuristic, not full 3D corner
#     # reconstruction -- it marks the leftmost/rightmost/topmost/bottommost
#     # detected sack so you can see where the pile's visible extent is.
#     edges = {}
#     if boxes:
#         edges = {
#             "left": min(boxes, key=lambda b: b["cx"])["id"],
#             "right": max(boxes, key=lambda b: b["cx"])["id"],
#             "top": min(boxes, key=lambda b: b["cy"])["id"],
#             "bottom": max(boxes, key=lambda b: b["cy"])["id"],
#         }
#         by_id = {b["id"]: b for b in boxes}
#         for box_id in set(edges.values()):
#             b = by_id[box_id]
#             cx, cy = int(b["cx"]), int(b["cy"])
#             cv2.circle(annotated, (cx, cy), 14, (0, 140, 255), 3)
#             cv2.putText(
#                 annotated, "EDGE",
#                 (cx + 16, cy),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2
#             )
#
#     return annotated, boxes, avg_conf, edges
#
#
# # ============================================================
# # STAGE 2 (auto hint only) - HEIGHT ROW CLUSTERING
# #
# # Only height is attempted automatically. Length/breadth need the pile's
# # front/side face split, which a single image can't reliably infer with a
# # simple heuristic -- those are handled by the interactive reference lines
# # in the UI instead, per your instruction to fall back to manual lines
# # when automatic direction detection isn't reliable.
# # ============================================================
#
# def estimate_height_rows(boxes):
#     if len(boxes) < 2:
#         return {"count": len(boxes), "confidence": 0.0}
#
#     ys = sorted(b["cy"] for b in boxes)
#     heights = [b["h"] for b in boxes if b["h"] > 0]
#     typical_h = statistics.median(heights) if heights else 20.0
#     gap_threshold = max(typical_h * 0.6, 5.0)
#
#     clusters = [[ys[0]]]
#     for y in ys[1:]:
#         if y - clusters[-1][-1] > gap_threshold:
#             clusters.append([y])
#         else:
#             clusters[-1].append(y)
#
#     row_count = len(clusters)
#     centers = [statistics.mean(c) for c in clusters]
#
#     if len(centers) >= 3:
#         gaps = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
#         mean_gap = statistics.mean(gaps)
#         stdev_gap = statistics.pstdev(gaps)
#         consistency = 1.0 - min(1.0, stdev_gap / mean_gap) if mean_gap > 0 else 0.0
#     elif len(centers) == 2:
#         consistency = 0.5
#     else:
#         consistency = 0.2
#
#     confidence = round(max(0.0, min(1.0, consistency)), 2)
#     return {"count": row_count, "confidence": confidence}
#
#
# # ============================================================
# # HTML
# # ============================================================
#
# HTML = """
# <!DOCTYPE html>
# <html>
# <head>
#     <meta charset="UTF-8">
#     <title>Warehouse Pile Estimator</title>
#
#     <style>
#         * { box-sizing: border-box; }
#
#         body {
#             margin: 0;
#             background: #08111c;
#             color: #edf6ff;
#             font-family: Arial, sans-serif;
#         }
#
#         .container {
#             max-width: 1560px;
#             margin: auto;
#             padding: 25px;
#         }
#
#         h1 { margin-bottom: 5px; }
#
#         .subtitle {
#             color: #9db1c7;
#             margin-bottom: 20px;
#         }
#
#         .flow {
#             color: #6fe3c4;
#             font-size: 13px;
#             margin-bottom: 20px;
#             letter-spacing: 0.02em;
#         }
#
#         .layout {
#             display: grid;
#             grid-template-columns: 1.5fr 0.7fr;
#             gap: 20px;
#         }
#
#         .panel {
#             background: #101d2b;
#             border: 1px solid #26384b;
#             border-radius: 16px;
#             padding: 18px;
#             margin-bottom: 18px;
#         }
#
#         .canvas-wrapper {
#             width: 100%;
#             overflow: auto;
#             background: #050b12;
#             border-radius: 12px;
#             padding: 10px;
#             text-align: center;
#         }
#
#         canvas {
#             max-width: 100%;
#             cursor: crosshair;
#             border-radius: 8px;
#         }
#
#         .buttons {
#             display: flex;
#             gap: 10px;
#             flex-wrap: wrap;
#             margin-top: 15px;
#             align-items: center;
#         }
#
#         button {
#             border: 0;
#             padding: 12px 18px;
#             border-radius: 10px;
#             cursor: pointer;
#             font-weight: bold;
#             background: #20c997;
#             color: #06120f;
#         }
#
#         button.secondary { background: #26384b; color: white; }
#         button.danger { background: #e05252; color: white; }
#
#         button:disabled {
#             opacity: 0.4;
#             cursor: not-allowed;
#         }
#
#         .instructions {
#             background: #0b1723;
#             border-radius: 12px;
#             padding: 15px;
#             line-height: 1.7;
#             margin-bottom: 18px;
#             font-size: 14px;
#         }
#
#         .locked {
#             opacity: 0.45;
#             pointer-events: none;
#         }
#
#         .measurement {
#             background: #0b1723;
#             border: 1px solid #26384b;
#             border-radius: 12px;
#             padding: 14px;
#             margin-bottom: 12px;
#         }
#
#         .measurement h3 { margin: 0 0 10px; }
#
#         .measurement.active { border-color: #20c997; }
#
#         .axis-row {
#             display: flex;
#             gap: 10px;
#             align-items: center;
#             flex-wrap: wrap;
#             margin-top: 8px;
#         }
#
#         input[type="number"] {
#             width: 90px;
#             padding: 8px;
#             border-radius: 8px;
#             border: 1px solid #34495e;
#             background: #07111b;
#             color: white;
#         }
#
#         .count-value {
#             margin-top: 8px;
#             font-weight: bold;
#             color: #6fe3c4;
#         }
#
#         .result {
#             background: #07111b;
#             border-radius: 12px;
#             padding: 15px;
#         }
#
#         .result-row {
#             display: flex;
#             justify-content: space-between;
#             padding: 9px 0;
#             border-bottom: 1px solid #1d2c3b;
#             gap: 10px;
#         }
#
#         .result-row:last-child { border-bottom: none; }
#
#         .big {
#             font-size: 22px;
#             font-weight: bold;
#             color: #20c997;
#         }
#
#         .status {
#             margin-top: 12px;
#             padding: 10px;
#             border-radius: 8px;
#             background: #172839;
#             color: #bcd0e3;
#         }
#
#         .warnings {
#             margin-top: 14px;
#             padding: 12px 14px;
#             border-radius: 10px;
#             background: rgba(224, 82, 82, 0.12);
#             border: 1px solid rgba(224, 82, 82, 0.4);
#             color: #ffb3b3;
#             font-size: 13.5px;
#             line-height: 1.6;
#         }
#
#         .swatch {
#             display: inline-block;
#             width: 10px;
#             height: 10px;
#             border-radius: 3px;
#             margin-right: 6px;
#         }
#
#         @media(max-width: 1000px) {
#             .layout { grid-template-columns: 1fr; }
#         }
#     </style>
# </head>
#
# <body>
#
# <div class="container">
#
#     <h1>Warehouse Pile Estimator</h1>
#     <div class="subtitle">YOLO sack detection first, then Length / Breadth / Height sack counts from reference lines.</div>
#     <div class="flow">Image &rarr; YOLO Detection &rarr; Corner/Edge Sacks &rarr; L / B / H Sack Count &rarr; L &times; B &times; H &rarr; Total Sacks</div>
#
#     <div class="layout">
#
#         <!-- LEFT: IMAGE + CONTROLS -->
#         <div>
#             <div class="panel">
#
#                 <div class="canvas-wrapper">
#                     <canvas id="canvas"></canvas>
#                 </div>
#
#                 <div class="buttons">
#                     <input type="file" id="imageInput" accept="image/*" style="display:none">
#
#                     <button onclick="document.getElementById('imageInput').click()">
#                         Upload Image
#                     </button>
#
#                     <button id="detectBtn" disabled onclick="detectSacks()">
#                         1. Detect Sacks
#                     </button>
#
#                     <button class="danger" onclick="resetAll()">
#                         Reset
#                     </button>
#                 </div>
#
#                 <div id="status" class="status">
#                     Upload an image to begin.
#                 </div>
#             </div>
#
#             <div id="axisSection" class="panel locked">
#
#                 <div class="instructions">
#                     <b>Step 2 &mdash; sack counts along each axis</b><br><br>
#                     Select an axis below, click 2 points on the detected image along that axis, then tune the
#                     corridor width (px) until the highlighted dots match one row/column of sacks.
#                     Total sacks = Length count &times; Breadth count &times; Height count. No manual bag count needed.
#                 </div>
#
#                 <div id="lengthBox" class="measurement">
#                     <h3><span class="swatch" style="background:#00ff88;"></span>Length</h3>
#                     <div class="axis-row">
#                         <button class="secondary axis-btn" disabled onclick="selectMeasurement('length')">Select Length</button>
#                         <label>Corridor (px): <input id="lengthCorridor" type="number" min="5" step="5" value="40" oninput="onCorridorChange('length')"></label>
#                     </div>
#                     <div id="lengthCountValue" class="count-value">--</div>
#                 </div>
#
#                 <div id="breadthBox" class="measurement">
#                     <h3><span class="swatch" style="background:#00aaff;"></span>Breadth</h3>
#                     <div class="axis-row">
#                         <button class="secondary axis-btn" disabled onclick="selectMeasurement('breadth')">Select Breadth</button>
#                         <label>Corridor (px): <input id="breadthCorridor" type="number" min="5" step="5" value="40" oninput="onCorridorChange('breadth')"></label>
#                     </div>
#                     <div id="breadthCountValue" class="count-value">--</div>
#                 </div>
#
#                 <div id="heightBox" class="measurement">
#                     <h3><span class="swatch" style="background:#ffcc00;"></span>Height</h3>
#                     <div class="axis-row">
#                         <button class="secondary axis-btn" disabled onclick="selectMeasurement('height')">Select Height</button>
#                         <label>Corridor (px): <input id="heightCorridor" type="number" min="5" step="5" value="40" oninput="onCorridorChange('height')"></label>
#                     </div>
#                     <div id="heightCountValue" class="count-value">--</div>
#                 </div>
#
#                 <div class="buttons">
#                     <button class="secondary axis-btn" disabled onclick="clearCurrentLine()">Clear Current Line</button>
#                 </div>
#             </div>
#         </div>
#
#         <!-- RIGHT: RESULTS -->
#         <div>
#             <div class="panel">
#                 <div class="result">
#                     <h2 style="margin-top:0;">Results</h2>
#
#                     <div class="result-row"><span>Detected Sacks</span><span id="resultDetected" class="big">--</span></div>
#                     <div class="result-row"><span>Avg Confidence</span><span id="resultAvgConf">--</span></div>
#                     <div class="result-row"><span>Auto Height Hint</span><span id="resultAutoHeight">--</span></div>
#
#                     <div class="result-row"><span>Sacks Along Length</span><span id="resultSacksLength">--</span></div>
#                     <div class="result-row"><span>Sacks Along Breadth</span><span id="resultSacksBreadth">--</span></div>
#                     <div class="result-row"><span>Sacks Along Height</span><span id="resultSacksHeight">--</span></div>
#
#                     <div class="result-row"><span>Total Estimated Sacks (L&times;B&times;H)</span><span id="resultTotal" class="big">--</span></div>
#                 </div>
#
#                 <div id="warnings" class="warnings" style="display:none;"></div>
#             </div>
#         </div>
#
#     </div>
# </div>
#
#
# <script>
#
# const canvas = document.getElementById("canvas");
# const ctx = canvas.getContext("2d");
#
# let image = new Image();
# let imageLoaded = false;
# let detecting = false;
#
# let detectedBoxes = [];
# let edgeBoxes = {};
# let autoHeight = null;
#
# let currentMeasurement = null;
#
# let lines = { length: [], breadth: [], height: [] };
# let corridorPx = { length: 40, breadth: 40, height: 40 };
# let axisCounts = { length: null, breadth: null, height: null };
# let matchedBoxIds = { length: [], breadth: [], height: [] };
#
# const AXIS_COLORS = { length: "#00ff88", breadth: "#00aaff", height: "#ffcc00" };
# const AXIS_LABELS = { length: "LENGTH", breadth: "BREADTH", height: "HEIGHT" };
#
#
# function setStatus(text) {
#     document.getElementById("status").innerText = text;
# }
#
# function median(arr) {
#     if (!arr.length) return 0;
#     const s = [...arr].sort((a, b) => a - b);
#     const mid = Math.floor(s.length / 2);
#     return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
# }
#
# function confidenceLabel(c) {
#     if (c >= 0.7) return "High";
#     if (c >= 0.4) return "Medium";
#     return "Low";
# }
#
# function resetResultFields() {
#     ["resultDetected", "resultAvgConf", "resultAutoHeight",
#      "resultSacksLength", "resultSacksBreadth", "resultSacksHeight", "resultTotal"]
#         .forEach(id => document.getElementById(id).innerText = "--");
#
#     ["length", "breadth", "height"].forEach(type => {
#         document.getElementById(type + "CountValue").innerText = "--";
#     });
#
#     document.getElementById("warnings").style.display = "none";
#     document.getElementById("warnings").innerHTML = "";
# }
#
#
# // ------------------------------------------------------------
# // IMAGE UPLOAD
# // ------------------------------------------------------------
#
# document.getElementById("imageInput").addEventListener("change", function (e) {
#     const file = e.target.files[0];
#     if (!file) return;
#
#     const reader = new FileReader();
#
#     reader.onload = function (event) {
#         image.onload = function () {
#             imageLoaded = true;
#
#             canvas.width = image.naturalWidth;
#             canvas.height = image.naturalHeight;
#
#             detectedBoxes = [];
#             edgeBoxes = {};
#             autoHeight = null;
#             lines = { length: [], breadth: [], height: [] };
#             axisCounts = { length: null, breadth: null, height: null };
#             matchedBoxIds = { length: [], breadth: [], height: [] };
#             currentMeasurement = null;
#
#             redraw();
#             resetResultFields();
#
#             document.querySelectorAll(".axis-btn").forEach(b => b.disabled = true);
#             document.getElementById("axisSection").classList.add("locked");
#             document.querySelectorAll(".measurement").forEach(el => el.classList.remove("active"));
#
#             document.getElementById("detectBtn").disabled = false;
#
#             setStatus('Image loaded. Click "1. Detect Sacks" to run YOLO detection first.');
#         };
#         image.src = event.target.result;
#     };
#
#     reader.readAsDataURL(file);
# });
#
#
# // ------------------------------------------------------------
# // STAGE 1 - DETECTION
# // ------------------------------------------------------------
#
# async function detectSacks() {
#
#     if (!imageLoaded) { setStatus("Upload an image first."); return; }
#
#     const file = document.getElementById("imageInput").files[0];
#     if (!file) { setStatus("Upload an image first."); return; }
#
#     if (detecting) return;
#     detecting = true;
#
#     const btn = document.getElementById("detectBtn");
#     btn.disabled = true;
#     setStatus("Running YOLO detection (rice_bag_10_v9_best.pt)...");
#
#     const form = new FormData();
#     form.append("image", file);
#
#     try {
#         const response = await fetch("/detect", { method: "POST", body: form });
#         const data = await response.json();
#
#         if (!data.success) {
#             setStatus(data.message || "Detection failed.");
#             detecting = false;
#             btn.disabled = false;
#             return;
#         }
#
#         detectedBoxes = data.boxes;
#         edgeBoxes = data.edges;
#         autoHeight = data.auto_height;
#
#         lines = { length: [], breadth: [], height: [] };
#         axisCounts = { length: null, breadth: null, height: null };
#         matchedBoxIds = { length: [], breadth: [], height: [] };
#
#         image.onload = function () {
#             canvas.width = image.naturalWidth;
#             canvas.height = image.naturalHeight;
#             redraw();
#         };
#         image.src = "data:image/jpeg;base64," + data.image_base64;
#
#         document.getElementById("resultDetected").innerText = data.detected_sacks;
#         document.getElementById("resultAvgConf").innerText = data.avg_confidence.toFixed(3);
#
#         if (data.detected_sacks > 0) {
#             document.getElementById("resultAutoHeight").innerText =
#                 autoHeight.count + " rows (" + confidenceLabel(autoHeight.confidence) +
#                 " confidence, " + Math.round(autoHeight.confidence * 100) + "%)";
#         } else {
#             document.getElementById("resultAutoHeight").innerText = "No sacks detected";
#         }
#
#         const dims = [];
#         detectedBoxes.forEach(b => { dims.push(b.w); dims.push(b.h); });
#         const med = median(dims) || 40;
#
#         ["length", "breadth", "height"].forEach(type => {
#             corridorPx[type] = Math.max(10, Math.round(med * 0.6));
#             document.getElementById(type + "Corridor").value = corridorPx[type];
#             document.getElementById(type + "CountValue").innerText = "--";
#         });
#
#         if (data.detected_sacks > 0) {
#             document.querySelectorAll(".axis-btn").forEach(b => b.disabled = false);
#             document.getElementById("axisSection").classList.remove("locked");
#
#             currentMeasurement = "length";
#             document.querySelectorAll(".measurement").forEach(el => el.classList.remove("active"));
#             document.getElementById("lengthBox").classList.add("active");
#
#             setStatus("Detected " + data.detected_sacks + " sacks. Now select Length and click 2 points across the pile.");
#         } else {
#             setStatus("No sacks detected in this image. Try a clearer photo of the pile.");
#         }
#
#         updateTotal();
#         updateWarnings();
#
#     } catch (err) {
#         setStatus("Detection error: " + err.message);
#     }
#
#     detecting = false;
#     btn.disabled = false;
# }
#
#
# // ------------------------------------------------------------
# // STAGE 2 - AXIS SELECTION & COUNTING
# // ------------------------------------------------------------
#
# function selectMeasurement(type) {
#     if (!detectedBoxes.length) {
#         setStatus('Run "1. Detect Sacks" first.');
#         return;
#     }
#
#     currentMeasurement = type;
#
#     document.querySelectorAll(".measurement").forEach(el => el.classList.remove("active"));
#     document.getElementById(type + "Box").classList.add("active");
#
#     setStatus("Now click 2 points across the pile's " + AXIS_LABELS[type].toLowerCase() + ".");
# }
#
# function onCorridorChange(type) {
#     const val = parseFloat(document.getElementById(type + "Corridor").value);
#     corridorPx[type] = isNaN(val) ? corridorPx[type] : val;
#     if (lines[type].length === 2) {
#         recomputeAxis(type);
#     }
# }
#
# function countAlongLine(p1, p2, halfWidthPx) {
#     const dx = p2.x - p1.x, dy = p2.y - p1.y;
#     const len = Math.hypot(dx, dy);
#     if (len < 1) return { count: 0, ids: [] };
#
#     const ux = dx / len, uy = dy / len;
#     const nx = -uy, ny = ux;
#     const margin = len * 0.15;
#
#     const ids = [];
#     detectedBoxes.forEach(b => {
#         const rx = b.cx - p1.x, ry = b.cy - p1.y;
#         const t = rx * ux + ry * uy;
#         const d = rx * nx + ry * ny;
#
#         if (t >= -margin && t <= len + margin && Math.abs(d) <= halfWidthPx) {
#             ids.push(b.id);
#         }
#     });
#
#     return { count: ids.length, ids: ids };
# }
#
# function recomputeAxis(type) {
#     const pts = lines[type];
#     if (pts.length !== 2) return;
#
#     const hw = corridorPx[type];
#     const { count, ids } = countAlongLine(pts[0], pts[1], hw);
#
#     axisCounts[type] = count;
#     matchedBoxIds[type] = ids;
#
#     document.getElementById(type + "CountValue").innerText = count + " sacks matched";
#
#     redraw();
#     updateTotal();
#     updateWarnings();
# }
#
# function clearCurrentLine() {
#     if (!currentMeasurement) {
#         setStatus("Select Length, Breadth or Height first.");
#         return;
#     }
#
#     lines[currentMeasurement] = [];
#     axisCounts[currentMeasurement] = null;
#     matchedBoxIds[currentMeasurement] = [];
#     document.getElementById(currentMeasurement + "CountValue").innerText = "--";
#
#     redraw();
#     updateTotal();
#     updateWarnings();
#
#     setStatus(AXIS_LABELS[currentMeasurement] + " line cleared.");
# }
#
# canvas.addEventListener("click", function (event) {
#     if (!imageLoaded) return;
#
#     if (!detectedBoxes.length) {
#         setStatus('Run "1. Detect Sacks" first before drawing reference lines.');
#         return;
#     }
#
#     if (!currentMeasurement) {
#         setStatus("Select Length, Breadth or Height first.");
#         return;
#     }
#
#     const rect = canvas.getBoundingClientRect();
#     const scaleX = canvas.width / rect.width;
#     const scaleY = canvas.height / rect.height;
#
#     const x = (event.clientX - rect.left) * scaleX;
#     const y = (event.clientY - rect.top) * scaleY;
#
#     const pts = lines[currentMeasurement];
#
#     if (pts.length >= 2) {
#         pts.length = 0;
#         axisCounts[currentMeasurement] = null;
#         matchedBoxIds[currentMeasurement] = [];
#     }
#
#     pts.push({ x: x, y: y });
#
#     if (pts.length === 2) {
#         recomputeAxis(currentMeasurement);
#         setStatus(AXIS_LABELS[currentMeasurement] + ": line complete, " +
#             axisCounts[currentMeasurement] + " sacks matched. Adjust corridor width if needed.");
#     } else {
#         redraw();
#         setStatus(AXIS_LABELS[currentMeasurement] + ": first point selected. Click second point.");
#     }
# });
#
#
# // ------------------------------------------------------------
# // TOTALS & WARNINGS
# // ------------------------------------------------------------
#
# function updateTotal() {
#     const l = axisCounts.length, b = axisCounts.breadth, h = axisCounts.height;
#
#     document.getElementById("resultSacksLength").innerText = (l === null ? "--" : l);
#     document.getElementById("resultSacksBreadth").innerText = (b === null ? "--" : b);
#     document.getElementById("resultSacksHeight").innerText = (h === null ? "--" : h);
#
#     const totalEl = document.getElementById("resultTotal");
#     if (l && b && h) {
#         totalEl.innerText = (l * b * h) + " sacks";
#     } else {
#         totalEl.innerText = "--";
#     }
# }
#
# function updateWarnings() {
#     const warnings = [];
#
#     ["length", "breadth", "height"].forEach(type => {
#         const c = axisCounts[type];
#         if (lines[type].length === 2) {
#             if (c === 0) {
#                 warnings.push(AXIS_LABELS[type] + ": no sacks matched near this line. Widen the corridor or redraw closer to sack centers.");
#             } else if (c === 1 && detectedBoxes.length > 8) {
#                 warnings.push(AXIS_LABELS[type] + ": only 1 sack matched - this count looks unreliable. Adjust the corridor width or the line position.");
#             }
#         }
#     });
#
#     if (detectedBoxes.length > 0 && detectedBoxes.length < 3) {
#         warnings.push("Very few sacks were detected overall - axis counts and the total will not be reliable.");
#     }
#
#     if (autoHeight && lines.height.length === 2 && axisCounts.height !== null) {
#         if (Math.abs(axisCounts.height - autoHeight.count) > Math.max(2, autoHeight.count * 0.4)) {
#             warnings.push("Height line count (" + axisCounts.height + ") disagrees a lot with the automatic height hint (" +
#                 autoHeight.count + "). Re-check the line placement.");
#         }
#     }
#
#     const box = document.getElementById("warnings");
#     if (warnings.length) {
#         box.style.display = "block";
#         box.innerHTML = "&#9888; " + warnings.join("<br>&#9888; ");
#     } else {
#         box.style.display = "none";
#         box.innerHTML = "";
#     }
# }
#
#
# // ------------------------------------------------------------
# // DRAWING
# // ------------------------------------------------------------
#
# function drawAxisLine(type) {
#     const pts = lines[type];
#     if (!pts.length) return;
#
#     const color = AXIS_COLORS[type];
#
#     pts.forEach((p, index) => {
#         ctx.beginPath();
#         ctx.arc(p.x, p.y, 8, 0, Math.PI * 2);
#         ctx.fillStyle = color;
#         ctx.fill();
#
#         ctx.fillStyle = "white";
#         ctx.font = "bold 16px Arial";
#         ctx.fillText(AXIS_LABELS[type] + " " + (index + 1), p.x + 10, p.y - 10);
#     });
#
#     if (pts.length === 2) {
#         const p1 = pts[0], p2 = pts[1];
#         const dx = p2.x - p1.x, dy = p2.y - p1.y;
#         const len = Math.hypot(dx, dy) || 1;
#         const ux = dx / len, uy = dy / len;
#         const nx = -uy, ny = ux;
#         const hw = corridorPx[type];
#
#         ctx.beginPath();
#         ctx.moveTo(p1.x, p1.y);
#         ctx.lineTo(p2.x, p2.y);
#         ctx.strokeStyle = color;
#         ctx.lineWidth = 3;
#         ctx.stroke();
#
#         ctx.setLineDash([6, 6]);
#         ctx.lineWidth = 1.5;
#         [hw, -hw].forEach(offset => {
#             ctx.beginPath();
#             ctx.moveTo(p1.x + nx * offset, p1.y + ny * offset);
#             ctx.lineTo(p2.x + nx * offset, p2.y + ny * offset);
#             ctx.strokeStyle = color;
#             ctx.stroke();
#         });
#         ctx.setLineDash([]);
#
#         (matchedBoxIds[type] || []).forEach(id => {
#             const b = detectedBoxes.find(bb => bb.id === id);
#             if (!b) return;
#             ctx.beginPath();
#             ctx.arc(b.cx, b.cy, 7, 0, Math.PI * 2);
#             ctx.globalAlpha = 0.85;
#             ctx.fillStyle = color;
#             ctx.fill();
#             ctx.globalAlpha = 1.0;
#         });
#
#         const midX = (p1.x + p2.x) / 2, midY = (p1.y + p2.y) / 2;
#         ctx.fillStyle = color;
#         ctx.font = "bold 20px Arial";
#         const cnt = axisCounts[type];
#         ctx.fillText(AXIS_LABELS[type] + ": " + (cnt === null ? "?" : cnt) + " sacks", midX, midY - 16);
#     }
# }
#
# function redraw() {
#     if (!imageLoaded) return;
#
#     ctx.clearRect(0, 0, canvas.width, canvas.height);
#     ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
#
#     drawAxisLine("length");
#     drawAxisLine("breadth");
#     drawAxisLine("height");
# }
#
#
# // ------------------------------------------------------------
# // RESET
# // ------------------------------------------------------------
#
# function resetAll() {
#     document.getElementById("imageInput").value = "";
#
#     imageLoaded = false;
#     detecting = false;
#     detectedBoxes = [];
#     edgeBoxes = {};
#     autoHeight = null;
#     lines = { length: [], breadth: [], height: [] };
#     corridorPx = { length: 40, breadth: 40, height: 40 };
#     axisCounts = { length: null, breadth: null, height: null };
#     matchedBoxIds = { length: [], breadth: [], height: [] };
#     currentMeasurement = null;
#
#     ctx.clearRect(0, 0, canvas.width, canvas.height);
#     canvas.width = 0;
#     canvas.height = 0;
#
#     resetResultFields();
#
#     ["length", "breadth", "height"].forEach(type => {
#         document.getElementById(type + "Corridor").value = 40;
#     });
#
#     document.querySelectorAll(".axis-btn").forEach(b => b.disabled = true);
#     document.getElementById("axisSection").classList.add("locked");
#     document.querySelectorAll(".measurement").forEach(el => el.classList.remove("active"));
#     document.getElementById("detectBtn").disabled = true;
#
#     setStatus("Upload an image to begin.");
# }
#
# </script>
#
# </body>
# </html>
# """
#
#
# @app.route("/")
# def index():
#     return render_template_string(HTML)
#
#
# @app.route("/health")
# def health():
#     return jsonify({
#         "status": "ok",
#         "service": "Warehouse Pile Estimator",
#         "device": str(DEVICE),
#     })
#
#
# @app.route("/detect", methods=["POST"])
# def detect():
#     try:
#         image_file = request.files.get("image")
#         if image_file is None or image_file.filename == "":
#             return jsonify({"success": False, "message": "Please upload an image first."})
#
#         image_bytes = image_file.read()
#         if len(image_bytes) == 0:
#             return jsonify({"success": False, "message": "Uploaded image is empty."})
#
#         image_array = np.frombuffer(image_bytes, dtype=np.uint8)
#         image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
#         if image_bgr is None:
#             return jsonify({"success": False, "message": "The uploaded file is not a valid image."})
#
#         annotated, boxes, avg_conf, edges = run_detection(image_bgr)
#         auto_height = estimate_height_rows(boxes)
#
#         is_success, buffer = cv2.imencode(".jpg", annotated)
#         if not is_success:
#             return jsonify({"success": False, "message": "Failed to create annotated image."})
#
#         return jsonify({
#             "success": True,
#             "detected_sacks": len(boxes),
#             "avg_confidence": avg_conf,
#             "boxes": boxes,
#             "edges": edges,
#             "auto_height": auto_height,
#             "image_base64": base64.b64encode(buffer.tobytes()).decode("utf-8"),
#         })
#
#     except Exception as exc:
#         return jsonify({"success": False, "message": f"Runtime error: {exc}"})
#
#
# if __name__ == "__main__":
#     print("=" * 60)
#     print("WAREHOUSE PILE ESTIMATOR - DETECTION FIRST")
#     print("=" * 60)
#     print(f"Model : {MODEL_PATH}")
#     print(f"Device: {DEVICE}")
#     print("Local URL: http://127.0.0.1:5006")
#     print("Network URL: http://0.0.0.0:5006")
#     print("=" * 60)
#
#     app.run(host="0.0.0.0", port=5006, debug=False)
#
# ============================================================================
# END OF BACKUP
# ============================================================================


# ============================================================================
# BACKUP 2 - PREVIOUS WORKING VERSION (dot + dedup detection, WITH edge
# points and a wide/generous line-corridor tolerance). This was the last
# version before: removing edge-point generation, tightening line-to-point
# selection with a small named LINE_POINT_TOLERANCE_FACTOR, and syncing in
# Stage B (center-point distance clustering) from src/test_rice_bag_v9_ui.py.
# Restore the same way as backup 1: remove the leading "# " from each line
# below.
# ============================================================================
#
# """
# Warehouse Pile Estimator - detection-first workflow (dot + dedup detection).
#
# Flow:
#     Image -> YOLO Detection (rice_bag_10_v9_best.pt, duplicate-deduped, drawn
#              as center-point dots -- synced with src/test_rice_bag_v9_ui.py)
#           -> Corner/Edge sacks
#           -> Length / Breadth / Height sack counts (via reference lines)
#           -> Length x Breadth x Height -> Total sacks
#
# Stage 1 (detection) mirrors the CURRENT src/test_rice_bag_v9_ui.py: same
# model, same confidence/IOU thresholds, same duplicate-box post-processing
# (IoU + containment dedup), same center-point dot drawing. Stage 2 (pile
# estimation) is separate and only consumes Stage 1's output -- it never runs
# another model.
#
# Run: python warehouse_pile_estimator_ui.py
# Access: http://localhost:5006
# """
#
# from pathlib import Path
# import base64
# import statistics
#
# import cv2
# import numpy as np
# import torch
# from flask import Flask, jsonify, request, render_template_string
# from ultralytics import YOLO
#
#
# # ============================================================
# # PATHS & MODEL (same model + settings as src/test_rice_bag_v9_ui.py)
# # ============================================================
#
# ROOT = Path(__file__).resolve().parent
# MODEL_PATH = ROOT / "models" / "rice_bag_10_v9" / "rice_bag_10_v9_best.pt"
#
# if not MODEL_PATH.exists():
#     raise FileNotFoundError(
#         f"Trained model not found: {MODEL_PATH}\n"
#         "Train it first (python src/train_rice_bag_v9.py) or place the weights there."
#     )
#
# print(f"\nLoading V9 model: {MODEL_PATH}")
# model = YOLO(str(MODEL_PATH))
# print(f"Model loaded. Task: {model.task}")
#
# CONF_THRESHOLD = 0.50
# IOU_THRESHOLD = 0.45
# MAX_DETECTIONS = 500
# DEVICE = 0 if torch.cuda.is_available() else "cpu"
#
# # Duplicate-box post-processing settings (same as src/test_rice_bag_v9_ui.py)
# DEDUP_IOU_THRESHOLD = 0.50
# DEDUP_CONTAINMENT_THRESHOLD = 0.75
# DEDUP_MODE = "merge"
#
# DOT_RADIUS = 6  # px, tune if dots look too small/large on your images
#
# app = Flask(__name__)
# app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
#
#
# # ============================================================
# # DUPLICATE-BOX POST-PROCESSING (identical to src/test_rice_bag_v9_ui.py)
# # ============================================================
#
# def _box_area(box):
#     return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
#
#
# def _iou_and_containment(box_a, box_b):
#     ix1, iy1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
#     ix2, iy2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
#     inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
#     if inter <= 0:
#         return 0.0, 0.0
#     area_a, area_b = _box_area(box_a), _box_area(box_b)
#     union = area_a + area_b - inter
#     iou = inter / union if union > 0 else 0.0
#     containment = inter / max(1e-6, min(area_a, area_b))
#     return iou, containment
#
#
# def deduplicate_boxes(xyxy, confs,
#                        iou_thresh=DEDUP_IOU_THRESHOLD,
#                        containment_thresh=DEDUP_CONTAINMENT_THRESHOLD,
#                        mode=DEDUP_MODE):
#     n = len(xyxy)
#     if n == 0:
#         return np.zeros((0, 4)), np.zeros((0,))
#
#     parent = list(range(n))
#
#     def find(i):
#         while parent[i] != i:
#             parent[i] = parent[parent[i]]
#             i = parent[i]
#         return i
#
#     def union(i, j):
#         ri, rj = find(i), find(j)
#         if ri != rj:
#             parent[rj] = ri
#
#     for i in range(n):
#         for j in range(i + 1, n):
#             iou, containment = _iou_and_containment(xyxy[i], xyxy[j])
#             if iou >= iou_thresh or containment >= containment_thresh:
#                 union(i, j)
#
#     clusters = {}
#     for i in range(n):
#         clusters.setdefault(find(i), []).append(i)
#
#     merged_boxes = []
#     merged_confs = []
#
#     for members in clusters.values():
#         if mode == "suppress" or len(members) == 1:
#             best = max(members, key=lambda idx: confs[idx])
#             merged_boxes.append(xyxy[best])
#             merged_confs.append(confs[best])
#             continue
#
#         member_confs = np.array([confs[m] for m in members], dtype=np.float64)
#         weights = member_confs / member_confs.sum()
#         coords = np.array([xyxy[m] for m in members], dtype=np.float64)
#         fused = (coords * weights[:, None]).sum(axis=0)
#
#         merged_boxes.append(fused)
#         merged_confs.append(float(member_confs.max()))
#
#     return np.array(merged_boxes), np.array(merged_confs)
#
#
# # ============================================================
# # STAGE 1 - DETECTION (mirrors CURRENT test_rice_bag_v9_ui.py /predict:
# # dedup post-processing + center-point dot drawing)
# # ============================================================
#
# def run_detection(image_bgr):
#     """Run YOLO detection, dedup duplicate boxes, and draw each surviving
#     sack as a center-point dot exactly like src/test_rice_bag_v9_ui.py.
#     Returns (annotated_image, boxes, avg_conf, edges, raw_count, duplicates_removed)."""
#
#     results = model.predict(
#         source=image_bgr,
#         conf=CONF_THRESHOLD,
#         iou=IOU_THRESHOLD,
#         max_det=MAX_DETECTIONS,
#         device=DEVICE,
#         verbose=False,
#     )
#
#     result = results[0]
#     boxes_obj = result.boxes
#
#     annotated = image_bgr.copy()
#     boxes = []
#     confidences = []
#     raw_count = 0
#
#     if boxes_obj is not None and len(boxes_obj) > 0:
#         xyxy = boxes_obj.xyxy.cpu().numpy()
#         confs = boxes_obj.conf.cpu().numpy()
#         raw_count = len(boxes_obj)
#
#         # duplicate-box post-processing (same as src/test_rice_bag_v9_ui.py)
#         xyxy, confs = deduplicate_boxes(xyxy, confs)
#
#         for i in range(len(xyxy)):
#             x1, y1, x2, y2 = map(float, xyxy[i])
#             conf = float(confs[i])
#             cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
#
#             confidences.append(conf)
#             box_id = i + 1
#
#             boxes.append({
#                 "id": box_id,
#                 "x1": x1, "y1": y1, "x2": x2, "y2": y2,
#                 "cx": cx, "cy": cy,
#                 "w": x2 - x1,
#                 "h": y2 - y1,
#                 "conf": conf,
#             })
#
#             cxi, cyi = int(cx), int(cy)
#
#             # --- center-point dot, same style as src/test_rice_bag_v9_ui.py ---
#             cv2.circle(annotated, (cxi, cyi), DOT_RADIUS, (0, 255, 0), -1)
#             cv2.circle(annotated, (cxi, cyi), DOT_RADIUS + 2, (0, 0, 0), 1)
#
#             label = f"{box_id} ({conf:.2f})"
#             cv2.putText(
#                 annotated, label,
#                 (cxi + DOT_RADIUS + 3, cyi - DOT_RADIUS),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1
#             )
#
#     avg_conf = float(np.mean(confidences)) if confidences else 0.0
#     duplicates_removed = raw_count - len(boxes)
#
#     cv2.putText(
#         annotated,
#         f"Total: {len(boxes)} bags | Avg: {avg_conf:.3f} | Dupes removed: {duplicates_removed}",
#         (10, 30),
#         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
#     )
#
#     # Corner / edge sacks: extreme detected dots by center position.
#     edges = {}
#     if boxes:
#         edges = {
#             "left": min(boxes, key=lambda b: b["cx"])["id"],
#             "right": max(boxes, key=lambda b: b["cx"])["id"],
#             "top": min(boxes, key=lambda b: b["cy"])["id"],
#             "bottom": max(boxes, key=lambda b: b["cy"])["id"],
#         }
#         by_id = {b["id"]: b for b in boxes}
#         for box_id in set(edges.values()):
#             b = by_id[box_id]
#             cx, cy = int(b["cx"]), int(b["cy"])
#             cv2.circle(annotated, (cx, cy), 14, (0, 140, 255), 3)
#             cv2.putText(
#                 annotated, "EDGE",
#                 (cx + 16, cy),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2
#             )
#
#     return annotated, boxes, avg_conf, edges, raw_count, duplicates_removed
#
#
# # ============================================================
# # STAGE 2 (auto hint only) - HEIGHT ROW CLUSTERING (unchanged)
# # ============================================================
#
# def estimate_height_rows(boxes):
#     if len(boxes) < 2:
#         return {"count": len(boxes), "confidence": 0.0}
#
#     ys = sorted(b["cy"] for b in boxes)
#     heights = [b["h"] for b in boxes if b["h"] > 0]
#     typical_h = statistics.median(heights) if heights else 20.0
#     gap_threshold = max(typical_h * 0.6, 5.0)
#
#     clusters = [[ys[0]]]
#     for y in ys[1:]:
#         if y - clusters[-1][-1] > gap_threshold:
#             clusters.append([y])
#         else:
#             clusters[-1].append(y)
#
#     row_count = len(clusters)
#     centers = [statistics.mean(c) for c in clusters]
#
#     if len(centers) >= 3:
#         gaps = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
#         mean_gap = statistics.mean(gaps)
#         stdev_gap = statistics.pstdev(gaps)
#         consistency = 1.0 - min(1.0, stdev_gap / mean_gap) if mean_gap > 0 else 0.0
#     elif len(centers) == 2:
#         consistency = 0.5
#     else:
#         consistency = 0.2
#
#     confidence = round(max(0.0, min(1.0, consistency)), 2)
#     return {"count": row_count, "confidence": confidence}
#
#
# # ============================================================
# # HTML
# # ============================================================
#
# HTML = """
# <!DOCTYPE html>
# <html>
# <head>
#     <meta charset="UTF-8">
#     <title>Warehouse Pile Estimator</title>
#
#     <style>
#         * { box-sizing: border-box; }
#
#         body {
#             margin: 0;
#             background: #08111c;
#             color: #edf6ff;
#             font-family: Arial, sans-serif;
#         }
#
#         .container {
#             max-width: 1560px;
#             margin: auto;
#             padding: 25px;
#         }
#
#         h1 { margin-bottom: 5px; }
#
#         .subtitle {
#             color: #9db1c7;
#             margin-bottom: 20px;
#         }
#
#         .flow {
#             color: #6fe3c4;
#             font-size: 13px;
#             margin-bottom: 20px;
#             letter-spacing: 0.02em;
#         }
#
#         .layout {
#             display: grid;
#             grid-template-columns: 1.5fr 0.7fr;
#             gap: 20px;
#         }
#
#         .panel {
#             background: #101d2b;
#             border: 1px solid #26384b;
#             border-radius: 16px;
#             padding: 18px;
#             margin-bottom: 18px;
#         }
#
#         .canvas-wrapper {
#             width: 100%;
#             overflow: auto;
#             background: #050b12;
#             border-radius: 12px;
#             padding: 10px;
#             text-align: center;
#         }
#
#         canvas {
#             max-width: 100%;
#             cursor: crosshair;
#             border-radius: 8px;
#         }
#
#         .buttons {
#             display: flex;
#             gap: 10px;
#             flex-wrap: wrap;
#             margin-top: 15px;
#             align-items: center;
#         }
#
#         button {
#             border: 0;
#             padding: 12px 18px;
#             border-radius: 10px;
#             cursor: pointer;
#             font-weight: bold;
#             background: #20c997;
#             color: #06120f;
#         }
#
#         button.secondary { background: #26384b; color: white; }
#         button.danger { background: #e05252; color: white; }
#
#         button:disabled {
#             opacity: 0.4;
#             cursor: not-allowed;
#         }
#
#         .instructions {
#             background: #0b1723;
#             border-radius: 12px;
#             padding: 15px;
#             line-height: 1.7;
#             margin-bottom: 18px;
#             font-size: 14px;
#         }
#
#         .locked {
#             opacity: 0.45;
#             pointer-events: none;
#         }
#
#         .measurement {
#             background: #0b1723;
#             border: 1px solid #26384b;
#             border-radius: 12px;
#             padding: 14px;
#             margin-bottom: 12px;
#         }
#
#         .measurement h3 { margin: 0 0 10px; }
#
#         .measurement.active { border-color: #20c997; }
#
#         .axis-row {
#             display: flex;
#             gap: 10px;
#             align-items: center;
#             flex-wrap: wrap;
#             margin-top: 8px;
#         }
#
#         input[type="number"] {
#             width: 90px;
#             padding: 8px;
#             border-radius: 8px;
#             border: 1px solid #34495e;
#             background: #07111b;
#             color: white;
#         }
#
#         .count-value {
#             margin-top: 8px;
#             font-weight: bold;
#             color: #6fe3c4;
#         }
#
#         .result {
#             background: #07111b;
#             border-radius: 12px;
#             padding: 15px;
#         }
#
#         .result-row {
#             display: flex;
#             justify-content: space-between;
#             padding: 9px 0;
#             border-bottom: 1px solid #1d2c3b;
#             gap: 10px;
#         }
#
#         .result-row:last-child { border-bottom: none; }
#
#         .big {
#             font-size: 22px;
#             font-weight: bold;
#             color: #20c997;
#         }
#
#         .status {
#             margin-top: 12px;
#             padding: 10px;
#             border-radius: 8px;
#             background: #172839;
#             color: #bcd0e3;
#         }
#
#         .warnings {
#             margin-top: 14px;
#             padding: 12px 14px;
#             border-radius: 10px;
#             background: rgba(224, 82, 82, 0.12);
#             border: 1px solid rgba(224, 82, 82, 0.4);
#             color: #ffb3b3;
#             font-size: 13.5px;
#             line-height: 1.6;
#         }
#
#         .swatch {
#             display: inline-block;
#             width: 10px;
#             height: 10px;
#             border-radius: 3px;
#             margin-right: 6px;
#         }
#
#         @media(max-width: 1000px) {
#             .layout { grid-template-columns: 1fr; }
#         }
#     </style>
# </head>
#
# <body>
#
# <div class="container">
#
#     <h1>Warehouse Pile Estimator</h1>
#     <div class="subtitle">YOLO sack detection first (dot + duplicate-dedup), then Length / Breadth / Height sack counts from reference lines.</div>
#     <div class="flow">Image &rarr; YOLO Detection (dots, deduped) &rarr; Corner/Edge Sacks &rarr; L / B / H Sack Count &rarr; L &times; B &times; H &rarr; Total Sacks</div>
#
#     <div class="layout">
#
#         <!-- LEFT: IMAGE + CONTROLS -->
#         <div>
#             <div class="panel">
#
#                 <div class="canvas-wrapper">
#                     <canvas id="canvas"></canvas>
#                 </div>
#
#                 <div class="buttons">
#                     <input type="file" id="imageInput" accept="image/*" style="display:none">
#
#                     <button onclick="document.getElementById('imageInput').click()">
#                         Upload Image
#                     </button>
#
#                     <button id="detectBtn" disabled onclick="detectSacks()">
#                         1. Detect Sacks
#                     </button>
#
#                     <button class="danger" onclick="resetAll()">
#                         Reset
#                     </button>
#                 </div>
#
#                 <div id="status" class="status">
#                     Upload an image to begin.
#                 </div>
#             </div>
#
#             <div id="axisSection" class="panel locked">
#
#                 <div class="instructions">
#                     <b>Step 2 &mdash; sack counts along each axis</b><br><br>
#                     Select an axis below, click 2 points on the detected image along that axis, then tune the
#                     corridor width (px) until the highlighted dots match one row/column of sacks.
#                     Total sacks = Length count &times; Breadth count &times; Height count. No manual bag count needed.
#                 </div>
#
#                 <div id="lengthBox" class="measurement">
#                     <h3><span class="swatch" style="background:#00ff88;"></span>Length</h3>
#                     <div class="axis-row">
#                         <button class="secondary axis-btn" disabled onclick="selectMeasurement('length')">Select Length</button>
#                         <label>Corridor (px): <input id="lengthCorridor" type="number" min="5" step="5" value="40" oninput="onCorridorChange('length')"></label>
#                     </div>
#                     <div id="lengthCountValue" class="count-value">--</div>
#                 </div>
#
#                 <div id="breadthBox" class="measurement">
#                     <h3><span class="swatch" style="background:#00aaff;"></span>Breadth</h3>
#                     <div class="axis-row">
#                         <button class="secondary axis-btn" disabled onclick="selectMeasurement('breadth')">Select Breadth</button>
#                         <label>Corridor (px): <input id="breadthCorridor" type="number" min="5" step="5" value="40" oninput="onCorridorChange('breadth')"></label>
#                     </div>
#                     <div id="breadthCountValue" class="count-value">--</div>
#                 </div>
#
#                 <div id="heightBox" class="measurement">
#                     <h3><span class="swatch" style="background:#ffcc00;"></span>Height</h3>
#                     <div class="axis-row">
#                         <button class="secondary axis-btn" disabled onclick="selectMeasurement('height')">Select Height</button>
#                         <label>Corridor (px): <input id="heightCorridor" type="number" min="5" step="5" value="40" oninput="onCorridorChange('height')"></label>
#                     </div>
#                     <div id="heightCountValue" class="count-value">--</div>
#                 </div>
#
#                 <div class="buttons">
#                     <button class="secondary axis-btn" disabled onclick="clearCurrentLine()">Clear Current Line</button>
#                 </div>
#             </div>
#         </div>
#
#         <!-- RIGHT: RESULTS -->
#         <div>
#             <div class="panel">
#                 <div class="result">
#                     <h2 style="margin-top:0;">Results</h2>
#
#                     <div class="result-row"><span>Detected Sacks</span><span id="resultDetected" class="big">--</span></div>
#                     <div class="result-row"><span>Avg Confidence</span><span id="resultAvgConf">--</span></div>
#                     <div class="result-row"><span>Duplicates Removed</span><span id="resultDuplicates">--</span></div>
#                     <div class="result-row"><span>Auto Height Hint</span><span id="resultAutoHeight">--</span></div>
#
#                     <div class="result-row"><span>Sacks Along Length</span><span id="resultSacksLength">--</span></div>
#                     <div class="result-row"><span>Sacks Along Breadth</span><span id="resultSacksBreadth">--</span></div>
#                     <div class="result-row"><span>Sacks Along Height</span><span id="resultSacksHeight">--</span></div>
#
#                     <div class="result-row"><span>Total Estimated Sacks (L&times;B&times;H)</span><span id="resultTotal" class="big">--</span></div>
#                 </div>
#
#                 <div id="warnings" class="warnings" style="display:none;"></div>
#             </div>
#         </div>
#
#     </div>
# </div>
#
#
# <script>
#
# const canvas = document.getElementById("canvas");
# const ctx = canvas.getContext("2d");
#
# let image = new Image();
# let imageLoaded = false;
# let detecting = false;
#
# let detectedBoxes = [];
# let edgeBoxes = {};
# let autoHeight = null;
#
# let currentMeasurement = null;
#
# let lines = { length: [], breadth: [], height: [] };
# let corridorPx = { length: 40, breadth: 40, height: 40 };
# let axisCounts = { length: null, breadth: null, height: null };
# let matchedBoxIds = { length: [], breadth: [], height: [] };
#
# const AXIS_COLORS = { length: "#00ff88", breadth: "#00aaff", height: "#ffcc00" };
# const AXIS_LABELS = { length: "LENGTH", breadth: "BREADTH", height: "HEIGHT" };
#
#
# function setStatus(text) {
#     document.getElementById("status").innerText = text;
# }
#
# function median(arr) {
#     if (!arr.length) return 0;
#     const s = [...arr].sort((a, b) => a - b);
#     const mid = Math.floor(s.length / 2);
#     return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
# }
#
# function confidenceLabel(c) {
#     if (c >= 0.7) return "High";
#     if (c >= 0.4) return "Medium";
#     return "Low";
# }
#
# function resetResultFields() {
#     ["resultDetected", "resultAvgConf", "resultDuplicates", "resultAutoHeight",
#      "resultSacksLength", "resultSacksBreadth", "resultSacksHeight", "resultTotal"]
#         .forEach(id => document.getElementById(id).innerText = "--");
#
#     ["length", "breadth", "height"].forEach(type => {
#         document.getElementById(type + "CountValue").innerText = "--";
#     });
#
#     document.getElementById("warnings").style.display = "none";
#     document.getElementById("warnings").innerHTML = "";
# }
#
#
# // ------------------------------------------------------------
# // IMAGE UPLOAD
# // ------------------------------------------------------------
#
# document.getElementById("imageInput").addEventListener("change", function (e) {
#     const file = e.target.files[0];
#     if (!file) return;
#
#     const reader = new FileReader();
#
#     reader.onload = function (event) {
#         image.onload = function () {
#             imageLoaded = true;
#
#             canvas.width = image.naturalWidth;
#             canvas.height = image.naturalHeight;
#
#             detectedBoxes = [];
#             edgeBoxes = {};
#             autoHeight = null;
#             lines = { length: [], breadth: [], height: [] };
#             axisCounts = { length: null, breadth: null, height: null };
#             matchedBoxIds = { length: [], breadth: [], height: [] };
#             currentMeasurement = null;
#
#             redraw();
#             resetResultFields();
#
#             document.querySelectorAll(".axis-btn").forEach(b => b.disabled = true);
#             document.getElementById("axisSection").classList.add("locked");
#             document.querySelectorAll(".measurement").forEach(el => el.classList.remove("active"));
#
#             document.getElementById("detectBtn").disabled = false;
#
#             setStatus('Image loaded. Click "1. Detect Sacks" to run YOLO detection first.');
#         };
#         image.src = event.target.result;
#     };
#
#     reader.readAsDataURL(file);
# });
#
#
# // ------------------------------------------------------------
# // STAGE 1 - DETECTION
# // ------------------------------------------------------------
#
# async function detectSacks() {
#
#     if (!imageLoaded) { setStatus("Upload an image first."); return; }
#
#     const file = document.getElementById("imageInput").files[0];
#     if (!file) { setStatus("Upload an image first."); return; }
#
#     if (detecting) return;
#     detecting = true;
#
#     const btn = document.getElementById("detectBtn");
#     btn.disabled = true;
#     setStatus("Running YOLO detection (rice_bag_10_v9_best.pt)...");
#
#     const form = new FormData();
#     form.append("image", file);
#
#     try {
#         const response = await fetch("/detect", { method: "POST", body: form });
#         const data = await response.json();
#
#         if (!data.success) {
#             setStatus(data.message || "Detection failed.");
#             detecting = false;
#             btn.disabled = false;
#             return;
#         }
#
#         detectedBoxes = data.boxes;
#         edgeBoxes = data.edges;
#         autoHeight = data.auto_height;
#
#         lines = { length: [], breadth: [], height: [] };
#         axisCounts = { length: null, breadth: null, height: null };
#         matchedBoxIds = { length: [], breadth: [], height: [] };
#
#         image.onload = function () {
#             canvas.width = image.naturalWidth;
#             canvas.height = image.naturalHeight;
#             redraw();
#         };
#         image.src = "data:image/jpeg;base64," + data.image_base64;
#
#         document.getElementById("resultDetected").innerText = data.detected_sacks;
#         document.getElementById("resultAvgConf").innerText = data.avg_confidence.toFixed(3);
#         document.getElementById("resultDuplicates").innerText = data.duplicates_removed;
#
#         if (data.detected_sacks > 0) {
#             document.getElementById("resultAutoHeight").innerText =
#                 autoHeight.count + " rows (" + confidenceLabel(autoHeight.confidence) +
#                 " confidence, " + Math.round(autoHeight.confidence * 100) + "%)";
#         } else {
#             document.getElementById("resultAutoHeight").innerText = "No sacks detected";
#         }
#
#         const dims = [];
#         detectedBoxes.forEach(b => { dims.push(b.w); dims.push(b.h); });
#         const med = median(dims) || 40;
#
#         ["length", "breadth", "height"].forEach(type => {
#             corridorPx[type] = Math.max(10, Math.round(med * 0.6));
#             document.getElementById(type + "Corridor").value = corridorPx[type];
#             document.getElementById(type + "CountValue").innerText = "--";
#         });
#
#         if (data.detected_sacks > 0) {
#             document.querySelectorAll(".axis-btn").forEach(b => b.disabled = false);
#             document.getElementById("axisSection").classList.remove("locked");
#
#             currentMeasurement = "length";
#             document.querySelectorAll(".measurement").forEach(el => el.classList.remove("active"));
#             document.getElementById("lengthBox").classList.add("active");
#
#             setStatus("Detected " + data.detected_sacks + " sacks (" + data.duplicates_removed +
#                 " duplicates removed). Now select Length and click 2 points across the pile.");
#         } else {
#             setStatus("No sacks detected in this image. Try a clearer photo of the pile.");
#         }
#
#         updateTotal();
#         updateWarnings();
#
#     } catch (err) {
#         setStatus("Detection error: " + err.message);
#     }
#
#     detecting = false;
#     btn.disabled = false;
# }
#
#
# // ------------------------------------------------------------
# // STAGE 2 - AXIS SELECTION & COUNTING
# // ------------------------------------------------------------
#
# function selectMeasurement(type) {
#     if (!detectedBoxes.length) {
#         setStatus('Run "1. Detect Sacks" first.');
#         return;
#     }
#
#     currentMeasurement = type;
#
#     document.querySelectorAll(".measurement").forEach(el => el.classList.remove("active"));
#     document.getElementById(type + "Box").classList.add("active");
#
#     setStatus("Now click 2 points across the pile's " + AXIS_LABELS[type].toLowerCase() + ".");
# }
#
# function onCorridorChange(type) {
#     const val = parseFloat(document.getElementById(type + "Corridor").value);
#     corridorPx[type] = isNaN(val) ? corridorPx[type] : val;
#     if (lines[type].length === 2) {
#         recomputeAxis(type);
#     }
# }
#
# function countAlongLine(p1, p2, halfWidthPx) {
#     const dx = p2.x - p1.x, dy = p2.y - p1.y;
#     const len = Math.hypot(dx, dy);
#     if (len < 1) return { count: 0, ids: [] };
#
#     const ux = dx / len, uy = dy / len;
#     const nx = -uy, ny = ux;
#     const margin = len * 0.15;
#
#     const ids = [];
#     detectedBoxes.forEach(b => {
#         const rx = b.cx - p1.x, ry = b.cy - p1.y;
#         const t = rx * ux + ry * uy;
#         const d = rx * nx + ry * ny;
#
#         if (t >= -margin && t <= len + margin && Math.abs(d) <= halfWidthPx) {
#             ids.push(b.id);
#         }
#     });
#
#     return { count: ids.length, ids: ids };
# }
#
# function recomputeAxis(type) {
#     const pts = lines[type];
#     if (pts.length !== 2) return;
#
#     const hw = corridorPx[type];
#     const { count, ids } = countAlongLine(pts[0], pts[1], hw);
#
#     axisCounts[type] = count;
#     matchedBoxIds[type] = ids;
#
#     document.getElementById(type + "CountValue").innerText = count + " sacks matched";
#
#     redraw();
#     updateTotal();
#     updateWarnings();
# }
#
# function clearCurrentLine() {
#     if (!currentMeasurement) {
#         setStatus("Select Length, Breadth or Height first.");
#         return;
#     }
#
#     lines[currentMeasurement] = [];
#     axisCounts[currentMeasurement] = null;
#     matchedBoxIds[currentMeasurement] = [];
#     document.getElementById(currentMeasurement + "CountValue").innerText = "--";
#
#     redraw();
#     updateTotal();
#     updateWarnings();
#
#     setStatus(AXIS_LABELS[currentMeasurement] + " line cleared.");
# }
#
# canvas.addEventListener("click", function (event) {
#     if (!imageLoaded) return;
#
#     if (!detectedBoxes.length) {
#         setStatus('Run "1. Detect Sacks" first before drawing reference lines.');
#         return;
#     }
#
#     if (!currentMeasurement) {
#         setStatus("Select Length, Breadth or Height first.");
#         return;
#     }
#
#     const rect = canvas.getBoundingClientRect();
#     const scaleX = canvas.width / rect.width;
#     const scaleY = canvas.height / rect.height;
#
#     const x = (event.clientX - rect.left) * scaleX;
#     const y = (event.clientY - rect.top) * scaleY;
#
#     const pts = lines[currentMeasurement];
#
#     if (pts.length >= 2) {
#         pts.length = 0;
#         axisCounts[currentMeasurement] = null;
#         matchedBoxIds[currentMeasurement] = [];
#     }
#
#     pts.push({ x: x, y: y });
#
#     if (pts.length === 2) {
#         recomputeAxis(currentMeasurement);
#         setStatus(AXIS_LABELS[currentMeasurement] + ": line complete, " +
#             axisCounts[currentMeasurement] + " sacks matched. Adjust corridor width if needed.");
#     } else {
#         redraw();
#         setStatus(AXIS_LABELS[currentMeasurement] + ": first point selected. Click second point.");
#     }
# });
#
#
# // ------------------------------------------------------------
# // TOTALS & WARNINGS
# // ------------------------------------------------------------
#
# function updateTotal() {
#     const l = axisCounts.length, b = axisCounts.breadth, h = axisCounts.height;
#
#     document.getElementById("resultSacksLength").innerText = (l === null ? "--" : l);
#     document.getElementById("resultSacksBreadth").innerText = (b === null ? "--" : b);
#     document.getElementById("resultSacksHeight").innerText = (h === null ? "--" : h);
#
#     const totalEl = document.getElementById("resultTotal");
#     if (l && b && h) {
#         totalEl.innerText = (l * b * h) + " sacks";
#     } else {
#         totalEl.innerText = "--";
#     }
# }
#
# function updateWarnings() {
#     const warnings = [];
#
#     ["length", "breadth", "height"].forEach(type => {
#         const c = axisCounts[type];
#         if (lines[type].length === 2) {
#             if (c === 0) {
#                 warnings.push(AXIS_LABELS[type] + ": no sacks matched near this line. Widen the corridor or redraw closer to sack centers.");
#             } else if (c === 1 && detectedBoxes.length > 8) {
#                 warnings.push(AXIS_LABELS[type] + ": only 1 sack matched - this count looks unreliable. Adjust the corridor width or the line position.");
#             }
#         }
#     });
#
#     if (detectedBoxes.length > 0 && detectedBoxes.length < 3) {
#         warnings.push("Very few sacks were detected overall - axis counts and the total will not be reliable.");
#     }
#
#     if (autoHeight && lines.height.length === 2 && axisCounts.height !== null) {
#         if (Math.abs(axisCounts.height - autoHeight.count) > Math.max(2, autoHeight.count * 0.4)) {
#             warnings.push("Height line count (" + axisCounts.height + ") disagrees a lot with the automatic height hint (" +
#                 autoHeight.count + "). Re-check the line placement.");
#         }
#     }
#
#     const box = document.getElementById("warnings");
#     if (warnings.length) {
#         box.style.display = "block";
#         box.innerHTML = "&#9888; " + warnings.join("<br>&#9888; ");
#     } else {
#         box.style.display = "none";
#         box.innerHTML = "";
#     }
# }
#
#
# // ------------------------------------------------------------
# // DRAWING
# // ------------------------------------------------------------
#
# function drawAxisLine(type) {
#     const pts = lines[type];
#     if (!pts.length) return;
#
#     const color = AXIS_COLORS[type];
#
#     pts.forEach((p, index) => {
#         ctx.beginPath();
#         ctx.arc(p.x, p.y, 8, 0, Math.PI * 2);
#         ctx.fillStyle = color;
#         ctx.fill();
#
#         ctx.fillStyle = "white";
#         ctx.font = "bold 16px Arial";
#         ctx.fillText(AXIS_LABELS[type] + " " + (index + 1), p.x + 10, p.y - 10);
#     });
#
#     if (pts.length === 2) {
#         const p1 = pts[0], p2 = pts[1];
#         const dx = p2.x - p1.x, dy = p2.y - p1.y;
#         const len = Math.hypot(dx, dy) || 1;
#         const ux = dx / len, uy = dy / len;
#         const nx = -uy, ny = ux;
#         const hw = corridorPx[type];
#
#         ctx.beginPath();
#         ctx.moveTo(p1.x, p1.y);
#         ctx.lineTo(p2.x, p2.y);
#         ctx.strokeStyle = color;
#         ctx.lineWidth = 3;
#         ctx.stroke();
#
#         ctx.setLineDash([6, 6]);
#         ctx.lineWidth = 1.5;
#         [hw, -hw].forEach(offset => {
#             ctx.beginPath();
#             ctx.moveTo(p1.x + nx * offset, p1.y + ny * offset);
#             ctx.lineTo(p2.x + nx * offset, p2.y + ny * offset);
#             ctx.strokeStyle = color;
#             ctx.stroke();
#         });
#         ctx.setLineDash([]);
#
#         (matchedBoxIds[type] || []).forEach(id => {
#             const b = detectedBoxes.find(bb => bb.id === id);
#             if (!b) return;
#             ctx.beginPath();
#             ctx.arc(b.cx, b.cy, 7, 0, Math.PI * 2);
#             ctx.globalAlpha = 0.85;
#             ctx.fillStyle = color;
#             ctx.fill();
#             ctx.globalAlpha = 1.0;
#         });
#
#         const midX = (p1.x + p2.x) / 2, midY = (p1.y + p2.y) / 2;
#         ctx.fillStyle = color;
#         ctx.font = "bold 20px Arial";
#         const cnt = axisCounts[type];
#         ctx.fillText(AXIS_LABELS[type] + ": " + (cnt === null ? "?" : cnt) + " sacks", midX, midY - 16);
#     }
# }
#
# function redraw() {
#     if (!imageLoaded) return;
#
#     ctx.clearRect(0, 0, canvas.width, canvas.height);
#     ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
#
#     drawAxisLine("length");
#     drawAxisLine("breadth");
#     drawAxisLine("height");
# }
#
#
# // ------------------------------------------------------------
# // RESET
# // ------------------------------------------------------------
#
# function resetAll() {
#     document.getElementById("imageInput").value = "";
#
#     imageLoaded = false;
#     detecting = false;
#     detectedBoxes = [];
#     edgeBoxes = {};
#     autoHeight = null;
#     lines = { length: [], breadth: [], height: [] };
#     corridorPx = { length: 40, breadth: 40, height: 40 };
#     axisCounts = { length: null, breadth: null, height: null };
#     matchedBoxIds = { length: [], breadth: [], height: [] };
#     currentMeasurement = null;
#
#     ctx.clearRect(0, 0, canvas.width, canvas.height);
#     canvas.width = 0;
#     canvas.height = 0;
#
#     resetResultFields();
#
#     ["length", "breadth", "height"].forEach(type => {
#         document.getElementById(type + "Corridor").value = 40;
#     });
#
#     document.querySelectorAll(".axis-btn").forEach(b => b.disabled = true);
#     document.getElementById("axisSection").classList.add("locked");
#     document.querySelectorAll(".measurement").forEach(el => el.classList.remove("active"));
#     document.getElementById("detectBtn").disabled = true;
#
#     setStatus("Upload an image to begin.");
# }
#
# </script>
#
# </body>
# </html>
# """
#
#
# @app.route("/")
# def index():
#     return render_template_string(HTML)
#
#
# @app.route("/health")
# def health():
#     return jsonify({
#         "status": "ok",
#         "service": "Warehouse Pile Estimator",
#         "device": str(DEVICE),
#     })
#
#
# @app.route("/detect", methods=["POST"])
# def detect():
#     try:
#         image_file = request.files.get("image")
#         if image_file is None or image_file.filename == "":
#             return jsonify({"success": False, "message": "Please upload an image first."})
#
#         image_bytes = image_file.read()
#         if len(image_bytes) == 0:
#             return jsonify({"success": False, "message": "Uploaded image is empty."})
#
#         image_array = np.frombuffer(image_bytes, dtype=np.uint8)
#         image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
#         if image_bgr is None:
#             return jsonify({"success": False, "message": "The uploaded file is not a valid image."})
#
#         annotated, boxes, avg_conf, edges, raw_count, duplicates_removed = run_detection(image_bgr)
#         auto_height = estimate_height_rows(boxes)
#
#         is_success, buffer = cv2.imencode(".jpg", annotated)
#         if not is_success:
#             return jsonify({"success": False, "message": "Failed to create annotated image."})
#
#         return jsonify({
#             "success": True,
#             "detected_sacks": len(boxes),
#             "raw_detections": raw_count,
#             "duplicates_removed": duplicates_removed,
#             "avg_confidence": avg_conf,
#             "boxes": boxes,
#             "edges": edges,
#             "auto_height": auto_height,
#             "image_base64": base64.b64encode(buffer.tobytes()).decode("utf-8"),
#         })
#
#     except Exception as exc:
#         return jsonify({"success": False, "message": f"Runtime error: {exc}"})
#
#
# if __name__ == "__main__":
#     print("=" * 60)
#     print("WAREHOUSE PILE ESTIMATOR - DETECTION FIRST (dots + dedup)")
#     print("=" * 60)
#     print(f"Model : {MODEL_PATH}")
#     print(f"Device: {DEVICE}")
#     print("Local URL: http://127.0.0.1:5006")
#     print("Network URL: http://0.0.0.0:5006")
#     print("=" * 60)
#
#     app.run(host="0.0.0.0", port=5006, debug=False)
#
# ============================================================================
# END OF BACKUP 2
# ============================================================================


# ============================================================
# NEW UPDATED VERSION
# ============================================================
"""
Warehouse Pile Estimator - detection-first workflow (dot + dedup detection,
no edge points, tight named line tolerance).

Flow:
    Image -> YOLO Detection (rice_bag_10_v9_best.pt, Stage A box-dedup +
             Stage B center-point clustering -- mirrors the CURRENT
             src/test_rice_bag_v9_ui.py exactly, that file is not modified)
          -> Length / Breadth / Height sack counts (via reference lines,
             using a small, clearly-named, size-adaptive tolerance)
          -> Length x Breadth x Height -> Total sacks

Changes from the previous version:
  1. Edge/corner-point generation removed entirely (see note above
     run_detection) -- only real sack-center dots are produced now.
  2. Line-to-point selection tightened: LINE_POINT_TOLERANCE_FACTOR
     replaces the old, too-generous default so a drawn line only picks up
     dots genuinely close to it, not ones above/below/beside it.
  3. Detection/dot logic (Stage A dedup + Stage B center clustering) is
     copied from the CURRENT src/test_rice_bag_v9_ui.py -- that file was
     read for reference only and was NOT modified in any way.

Run: python warehouse_pile_estimator_ui.py
Access: http://localhost:5006
"""

from pathlib import Path
import base64
import statistics

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request, render_template_string
from ultralytics import YOLO


# ============================================================
# PATHS & MODEL (same model + settings as src/test_rice_bag_v9_ui.py)
# ============================================================

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "rice_bag_10_v9" / "rice_bag_10_v9_best.pt"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}\n"
        "Train it first (python src/train_rice_bag_v9.py) or place the weights there."
    )

print(f"\nLoading V9 model: {MODEL_PATH}")
model = YOLO(str(MODEL_PATH))
print(f"Model loaded. Task: {model.task}")

CONF_THRESHOLD = 0.50
IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 500
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# Stage A: box-level dedup (copied from the CURRENT src/test_rice_bag_v9_ui.py)
DEDUP_IOU_THRESHOLD = 0.50
DEDUP_CONTAINMENT_THRESHOLD = 0.75
DEDUP_MODE = "merge"

# Stage B: center-point distance clustering (copied from the CURRENT
# src/test_rice_bag_v9_ui.py -- same value it currently uses, 0.35)
CENTER_DIST_FACTOR = 0.35

DOT_RADIUS = 2  # px, copied from the CURRENT src/test_rice_bag_v9_ui.py

# Stage 2 (this file only): how close a detected dot must be to a drawn
# L/B/H reference line, as a FRACTION of the median detected sack size in
# that image (not a fixed pixel value, so it scales with photo distance).
# Kept small on purpose so a line only picks up the row/column it was
# actually drawn through, not neighbouring rows above/below/beside it.
# Tune this if lines pick up too many (raise it slightly) or too few
# (lower it slightly) dots.
LINE_POINT_TOLERANCE_FACTOR = 0.25

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


# ============================================================
# STAGE A - BOX-LEVEL DEDUP (identical to CURRENT src/test_rice_bag_v9_ui.py)
# ============================================================

def _box_area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _iou_and_containment(box_a, box_b):
    ix1, iy1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    ix2, iy2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0, 0.0
    area_a, area_b = _box_area(box_a), _box_area(box_b)
    union = area_a + area_b - inter
    iou = inter / union if union > 0 else 0.0
    containment = inter / max(1e-6, min(area_a, area_b))
    return iou, containment


def deduplicate_boxes(xyxy, confs,
                       iou_thresh=DEDUP_IOU_THRESHOLD,
                       containment_thresh=DEDUP_CONTAINMENT_THRESHOLD,
                       mode=DEDUP_MODE):
    n = len(xyxy)
    if n == 0:
        return np.zeros((0, 4)), np.zeros((0,))

    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        for j in range(i + 1, n):
            iou, containment = _iou_and_containment(xyxy[i], xyxy[j])
            if iou >= iou_thresh or containment >= containment_thresh:
                union(i, j)

    clusters = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    merged_boxes = []
    merged_confs = []

    for members in clusters.values():
        if mode == "suppress" or len(members) == 1:
            best = max(members, key=lambda idx: confs[idx])
            merged_boxes.append(xyxy[best])
            merged_confs.append(confs[best])
            continue

        member_confs = np.array([confs[m] for m in members], dtype=np.float64)
        weights = member_confs / member_confs.sum()
        coords = np.array([xyxy[m] for m in members], dtype=np.float64)
        fused = (coords * weights[:, None]).sum(axis=0)

        merged_boxes.append(fused)
        merged_confs.append(float(member_confs.max()))

    return np.array(merged_boxes), np.array(merged_confs)


# ============================================================
# STAGE B - CENTER-POINT DISTANCE CLUSTERING
# (identical to CURRENT src/test_rice_bag_v9_ui.py)
# ============================================================

def cluster_center_points(boxes, distance_factor=CENTER_DIST_FACTOR):
    n = len(boxes)
    if n <= 1:
        return boxes

    widths = [b["w"] for b in boxes if b["w"] > 0]
    heights = [b["h"] for b in boxes if b["h"] > 0]
    median_w = statistics.median(widths) if widths else 20.0
    median_h = statistics.median(heights) if heights else 20.0
    radius = distance_factor * (median_w + median_h) / 2.0

    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        for j in range(i + 1, n):
            dx = boxes[i]["cx"] - boxes[j]["cx"]
            dy = boxes[i]["cy"] - boxes[j]["cy"]
            if (dx * dx + dy * dy) ** 0.5 <= radius:
                union(i, j)

    clusters = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    fused_boxes = []

    for members in clusters.values():
        if len(members) == 1:
            fused_boxes.append(boxes[members[0]])
            continue

        member_confs = np.array([boxes[m]["conf"] for m in members], dtype=np.float64)
        weights = member_confs / member_confs.sum()

        cx = sum(boxes[m]["cx"] * w for m, w in zip(members, weights))
        cy = sum(boxes[m]["cy"] * w for m, w in zip(members, weights))
        x1 = sum(boxes[m]["x1"] * w for m, w in zip(members, weights))
        y1 = sum(boxes[m]["y1"] * w for m, w in zip(members, weights))
        x2 = sum(boxes[m]["x2"] * w for m, w in zip(members, weights))
        y2 = sum(boxes[m]["y2"] * w for m, w in zip(members, weights))

        fused_boxes.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "cx": cx, "cy": cy,
            "w": x2 - x1, "h": y2 - y1,
            "conf": float(member_confs.max()),
        })

    return fused_boxes


# ============================================================
# STAGE 1 - DETECTION (Stage A + Stage B, mirrors CURRENT
# src/test_rice_bag_v9_ui.py exactly)
#
# EDGE POINTS REMOVED: the previous version marked the extreme
# leftmost/rightmost/topmost/bottommost dot as a "corner/edge sack" with
# an orange ring. Per your feedback those were low quality and not
# useful, so that entire block (the `edges` dict, the by-id lookup, and
# the cv2.circle/putText "EDGE" drawing) has been deleted. run_detection
# now only produces and draws real sack-center dots.
# ============================================================

def run_detection(image_bgr):
    """Run YOLO detection, Stage A box dedup, Stage B center clustering,
    and draw each surviving sack as a center-point dot -- exactly like
    the CURRENT src/test_rice_bag_v9_ui.py. No edge/corner markers.
    Returns (annotated_image, boxes, avg_conf, raw_count, duplicates_removed)."""

    results = model.predict(
        source=image_bgr,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        max_det=MAX_DETECTIONS,
        device=DEVICE,
        verbose=False,
    )

    result = results[0]
    boxes_obj = result.boxes

    annotated = image_bgr.copy()
    boxes = []
    confidences = []
    raw_count = 0

    if boxes_obj is not None and len(boxes_obj) > 0:
        xyxy = boxes_obj.xyxy.cpu().numpy()
        confs = boxes_obj.conf.cpu().numpy()
        raw_count = len(boxes_obj)

        # Stage A: box-level dedup (IoU + containment)
        xyxy, confs = deduplicate_boxes(xyxy, confs)

        stage_a_boxes = []
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = map(float, xyxy[i])
            conf = float(confs[i])
            stage_a_boxes.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "cx": (x1 + x2) / 2.0, "cy": (y1 + y2) / 2.0,
                "w": x2 - x1, "h": y2 - y1,
                "conf": conf,
            })

        # Stage B: center-point distance clustering
        final_boxes = cluster_center_points(stage_a_boxes)

        for i, b in enumerate(final_boxes):
            cx, cy = b["cx"], b["cy"]
            conf = b["conf"]
            box_id = i + 1

            confidences.append(conf)

            boxes.append({
                "id": box_id,
                "x1": b["x1"], "y1": b["y1"], "x2": b["x2"], "y2": b["y2"],
                "cx": cx, "cy": cy,
                "w": b["w"], "h": b["h"],
                "conf": conf,
            })

            cxi, cyi = int(cx), int(cy)

            # center-point dot, same style as src/test_rice_bag_v9_ui.py
            cv2.circle(annotated, (cxi, cyi), DOT_RADIUS, (0, 255, 0), -1)
            cv2.circle(annotated, (cxi, cyi), DOT_RADIUS + 2, (0, 0, 0), 1)

            label = f"{box_id} ({conf:.2f})"
            cv2.putText(
                annotated, label,
                (cxi + DOT_RADIUS + 3, cyi - DOT_RADIUS),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1
            )

    avg_conf = float(np.mean(confidences)) if confidences else 0.0
    duplicates_removed = raw_count - len(boxes)

    cv2.putText(
        annotated,
        f"Total: {len(boxes)} bags | Avg: {avg_conf:.3f} | Dupes removed: {duplicates_removed}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
    )

    return annotated, boxes, avg_conf, raw_count, duplicates_removed


# ============================================================
# STAGE 2 (auto hint only) - HEIGHT ROW CLUSTERING (unchanged)
# ============================================================

def estimate_height_rows(boxes):
    if len(boxes) < 2:
        return {"count": len(boxes), "confidence": 0.0}

    ys = sorted(b["cy"] for b in boxes)
    heights = [b["h"] for b in boxes if b["h"] > 0]
    typical_h = statistics.median(heights) if heights else 20.0
    gap_threshold = max(typical_h * 0.6, 5.0)

    clusters = [[ys[0]]]
    for y in ys[1:]:
        if y - clusters[-1][-1] > gap_threshold:
            clusters.append([y])
        else:
            clusters[-1].append(y)

    row_count = len(clusters)
    centers = [statistics.mean(c) for c in clusters]

    if len(centers) >= 3:
        gaps = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
        mean_gap = statistics.mean(gaps)
        stdev_gap = statistics.pstdev(gaps)
        consistency = 1.0 - min(1.0, stdev_gap / mean_gap) if mean_gap > 0 else 0.0
    elif len(centers) == 2:
        consistency = 0.5
    else:
        consistency = 0.2

    confidence = round(max(0.0, min(1.0, consistency)), 2)
    return {"count": row_count, "confidence": confidence}


# ============================================================
# HTML
# ============================================================

HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Warehouse Pile Estimator</title>

    <style>
        * { box-sizing: border-box; }

        body {
            margin: 0;
            background: #08111c;
            color: #edf6ff;
            font-family: Arial, sans-serif;
        }

        .container {
            max-width: 1560px;
            margin: auto;
            padding: 25px;
        }

        h1 { margin-bottom: 5px; }

        .subtitle {
            color: #9db1c7;
            margin-bottom: 20px;
        }

        .flow {
            color: #6fe3c4;
            font-size: 13px;
            margin-bottom: 20px;
            letter-spacing: 0.02em;
        }

        .layout {
            display: grid;
            grid-template-columns: 1.5fr 0.7fr;
            gap: 20px;
        }

        .panel {
            background: #101d2b;
            border: 1px solid #26384b;
            border-radius: 16px;
            padding: 18px;
            margin-bottom: 18px;
        }

        .canvas-wrapper {
            width: 100%;
            overflow: auto;
            background: #050b12;
            border-radius: 12px;
            padding: 10px;
            text-align: center;
        }

        canvas {
            max-width: 100%;
            cursor: crosshair;
            border-radius: 8px;
        }

        .buttons {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-top: 15px;
            align-items: center;
        }

        button {
            border: 0;
            padding: 12px 18px;
            border-radius: 10px;
            cursor: pointer;
            font-weight: bold;
            background: #20c997;
            color: #06120f;
        }

        button.secondary { background: #26384b; color: white; }
        button.danger { background: #e05252; color: white; }

        button:disabled {
            opacity: 0.4;
            cursor: not-allowed;
        }

        .instructions {
            background: #0b1723;
            border-radius: 12px;
            padding: 15px;
            line-height: 1.7;
            margin-bottom: 18px;
            font-size: 14px;
        }

        .locked {
            opacity: 0.45;
            pointer-events: none;
        }

        .measurement {
            background: #0b1723;
            border: 1px solid #26384b;
            border-radius: 12px;
            padding: 14px;
            margin-bottom: 12px;
        }

        .measurement h3 { margin: 0 0 10px; }

        .measurement.active { border-color: #20c997; }

        .axis-row {
            display: flex;
            gap: 10px;
            align-items: center;
            flex-wrap: wrap;
            margin-top: 8px;
        }

        input[type="number"] {
            width: 90px;
            padding: 8px;
            border-radius: 8px;
            border: 1px solid #34495e;
            background: #07111b;
            color: white;
        }

        .count-value {
            margin-top: 8px;
            font-weight: bold;
            color: #6fe3c4;
        }

        .result {
            background: #07111b;
            border-radius: 12px;
            padding: 15px;
        }

        .result-row {
            display: flex;
            justify-content: space-between;
            padding: 9px 0;
            border-bottom: 1px solid #1d2c3b;
            gap: 10px;
        }

        .result-row:last-child { border-bottom: none; }

        .big {
            font-size: 22px;
            font-weight: bold;
            color: #20c997;
        }

        .status {
            margin-top: 12px;
            padding: 10px;
            border-radius: 8px;
            background: #172839;
            color: #bcd0e3;
        }

        .warnings {
            margin-top: 14px;
            padding: 12px 14px;
            border-radius: 10px;
            background: rgba(224, 82, 82, 0.12);
            border: 1px solid rgba(224, 82, 82, 0.4);
            color: #ffb3b3;
            font-size: 13.5px;
            line-height: 1.6;
        }

        .swatch {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 3px;
            margin-right: 6px;
        }

        @media(max-width: 1000px) {
            .layout { grid-template-columns: 1fr; }
        }
    </style>
</head>

<body>

<div class="container">

    <h1>Warehouse Pile Estimator</h1>
    <div class="subtitle">YOLO sack detection first (dot + duplicate-dedup, no edge points), then Length / Breadth / Height sack counts from reference lines.</div>
    <div class="flow">Image &rarr; YOLO Detection (dots, deduped) &rarr; L / B / H Sack Count &rarr; L &times; B &times; H &rarr; Total Sacks</div>

    <div class="layout">

        <!-- LEFT: IMAGE + CONTROLS -->
        <div>
            <div class="panel">

                <div class="canvas-wrapper">
                    <canvas id="canvas"></canvas>
                </div>

                <div class="buttons">
                    <input type="file" id="imageInput" accept="image/*" style="display:none">

                    <button onclick="document.getElementById('imageInput').click()">
                        Upload Image
                    </button>

                    <button id="detectBtn" disabled onclick="detectSacks()">
                        1. Detect Sacks
                    </button>

                    <button class="danger" onclick="resetAll()">
                        Reset
                    </button>
                </div>

                <div id="status" class="status">
                    Upload an image to begin.
                </div>
            </div>

            <div id="axisSection" class="panel locked">

                <div class="instructions">
                    <b>Step 2 &mdash; sack counts along each axis</b><br><br>
                    Select an axis below, click 2 points on the detected image along that axis, then tune the
                    line tolerance (px) until only the dots actually on that row/column are selected &mdash;
                    not ones above, below, or beside the line.
                    Total sacks = Length count &times; Breadth count &times; Height count. No manual bag count needed.
                </div>

                <div id="lengthBox" class="measurement">
                    <h3><span class="swatch" style="background:#00ff88;"></span>Length</h3>
                    <div class="axis-row">
                        <button class="secondary axis-btn" disabled onclick="selectMeasurement('length')">Select Length</button>
                        <label>Line tolerance (px): <input id="lengthTolerance" type="number" min="2" step="1" value="15" oninput="onToleranceChange('length')"></label>
                    </div>
                    <div id="lengthCountValue" class="count-value">--</div>
                </div>

                <div id="breadthBox" class="measurement">
                    <h3><span class="swatch" style="background:#00aaff;"></span>Breadth</h3>
                    <div class="axis-row">
                        <button class="secondary axis-btn" disabled onclick="selectMeasurement('breadth')">Select Breadth</button>
                        <label>Line tolerance (px): <input id="breadthTolerance" type="number" min="2" step="1" value="15" oninput="onToleranceChange('breadth')"></label>
                    </div>
                    <div id="breadthCountValue" class="count-value">--</div>
                </div>

                <div id="heightBox" class="measurement">
                    <h3><span class="swatch" style="background:#ffcc00;"></span>Height</h3>
                    <div class="axis-row">
                        <button class="secondary axis-btn" disabled onclick="selectMeasurement('height')">Select Height</button>
                        <label>Line tolerance (px): <input id="heightTolerance" type="number" min="2" step="1" value="15" oninput="onToleranceChange('height')"></label>
                    </div>
                    <div id="heightCountValue" class="count-value">--</div>
                </div>

                <div class="buttons">
                    <button class="secondary axis-btn" disabled onclick="clearCurrentLine()">Clear Current Line</button>
                </div>
            </div>
        </div>

        <!-- RIGHT: RESULTS -->
        <div>
            <div class="panel">
                <div class="result">
                    <h2 style="margin-top:0;">Results</h2>

                    <div class="result-row"><span>Detected Sacks</span><span id="resultDetected" class="big">--</span></div>
                    <div class="result-row"><span>Avg Confidence</span><span id="resultAvgConf">--</span></div>
                    <div class="result-row"><span>Duplicates Removed</span><span id="resultDuplicates">--</span></div>
                    <div class="result-row"><span>Auto Height Hint</span><span id="resultAutoHeight">--</span></div>

                    <div class="result-row"><span>Sacks Along Length</span><span id="resultSacksLength">--</span></div>
                    <div class="result-row"><span>Sacks Along Breadth</span><span id="resultSacksBreadth">--</span></div>
                    <div class="result-row"><span>Sacks Along Height</span><span id="resultSacksHeight">--</span></div>

                    <div class="result-row"><span>Total Estimated Sacks (L&times;B&times;H)</span><span id="resultTotal" class="big">--</span></div>
                </div>

                <div id="warnings" class="warnings" style="display:none;"></div>
            </div>
        </div>

    </div>
</div>


<script>

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

// Injected from the Python-side LINE_POINT_TOLERANCE_FACTOR constant, so
// there is one source of truth for the default line tolerance.
const LINE_POINT_TOLERANCE_FACTOR = {{ line_tolerance_factor }};

let image = new Image();
let imageLoaded = false;
let detecting = false;

let detectedBoxes = [];
let autoHeight = null;

let currentMeasurement = null;

let lines = { length: [], breadth: [], height: [] };
let linePointTolerancePx = { length: 15, breadth: 15, height: 15 };
let axisCounts = { length: null, breadth: null, height: null };
let matchedBoxIds = { length: [], breadth: [], height: [] };

const AXIS_COLORS = { length: "#00ff88", breadth: "#00aaff", height: "#ffcc00" };
const AXIS_LABELS = { length: "LENGTH", breadth: "BREADTH", height: "HEIGHT" };


function setStatus(text) {
    document.getElementById("status").innerText = text;
}

function median(arr) {
    if (!arr.length) return 0;
    const s = [...arr].sort((a, b) => a - b);
    const mid = Math.floor(s.length / 2);
    return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

function confidenceLabel(c) {
    if (c >= 0.7) return "High";
    if (c >= 0.4) return "Medium";
    return "Low";
}

function resetResultFields() {
    ["resultDetected", "resultAvgConf", "resultDuplicates", "resultAutoHeight",
     "resultSacksLength", "resultSacksBreadth", "resultSacksHeight", "resultTotal"]
        .forEach(id => document.getElementById(id).innerText = "--");

    ["length", "breadth", "height"].forEach(type => {
        document.getElementById(type + "CountValue").innerText = "--";
    });

    document.getElementById("warnings").style.display = "none";
    document.getElementById("warnings").innerHTML = "";
}


// ------------------------------------------------------------
// IMAGE UPLOAD
// ------------------------------------------------------------

document.getElementById("imageInput").addEventListener("change", function (e) {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();

    reader.onload = function (event) {
        image.onload = function () {
            imageLoaded = true;

            canvas.width = image.naturalWidth;
            canvas.height = image.naturalHeight;

            detectedBoxes = [];
            autoHeight = null;
            lines = { length: [], breadth: [], height: [] };
            axisCounts = { length: null, breadth: null, height: null };
            matchedBoxIds = { length: [], breadth: [], height: [] };
            currentMeasurement = null;

            redraw();
            resetResultFields();

            document.querySelectorAll(".axis-btn").forEach(b => b.disabled = true);
            document.getElementById("axisSection").classList.add("locked");
            document.querySelectorAll(".measurement").forEach(el => el.classList.remove("active"));

            document.getElementById("detectBtn").disabled = false;

            setStatus('Image loaded. Click "1. Detect Sacks" to run YOLO detection first.');
        };
        image.src = event.target.result;
    };

    reader.readAsDataURL(file);
});


// ------------------------------------------------------------
// STAGE 1 - DETECTION
// ------------------------------------------------------------

async function detectSacks() {

    if (!imageLoaded) { setStatus("Upload an image first."); return; }

    const file = document.getElementById("imageInput").files[0];
    if (!file) { setStatus("Upload an image first."); return; }

    if (detecting) return;
    detecting = true;

    const btn = document.getElementById("detectBtn");
    btn.disabled = true;
    setStatus("Running YOLO detection (rice_bag_10_v9_best.pt)...");

    const form = new FormData();
    form.append("image", file);

    try {
        const response = await fetch("/detect", { method: "POST", body: form });
        const data = await response.json();

        if (!data.success) {
            setStatus(data.message || "Detection failed.");
            detecting = false;
            btn.disabled = false;
            return;
        }

        detectedBoxes = data.boxes;
        autoHeight = data.auto_height;

        lines = { length: [], breadth: [], height: [] };
        axisCounts = { length: null, breadth: null, height: null };
        matchedBoxIds = { length: [], breadth: [], height: [] };

        image.onload = function () {
            canvas.width = image.naturalWidth;
            canvas.height = image.naturalHeight;
            redraw();
        };
        image.src = "data:image/jpeg;base64," + data.image_base64;

        document.getElementById("resultDetected").innerText = data.detected_sacks;
        document.getElementById("resultAvgConf").innerText = data.avg_confidence.toFixed(3);
        document.getElementById("resultDuplicates").innerText = data.duplicates_removed;

        if (data.detected_sacks > 0) {
            document.getElementById("resultAutoHeight").innerText =
                autoHeight.count + " rows (" + confidenceLabel(autoHeight.confidence) +
                " confidence, " + Math.round(autoHeight.confidence * 100) + "%)";
        } else {
            document.getElementById("resultAutoHeight").innerText = "No sacks detected";
        }

        // Default line tolerance: a SMALL fraction (LINE_POINT_TOLERANCE_FACTOR)
        // of the median detected sack size, so a drawn line only picks up the
        // row/column it actually crosses -- not neighbours above/below/beside.
        const dims = [];
        detectedBoxes.forEach(b => { dims.push(b.w); dims.push(b.h); });
        const med = median(dims) || 30;

        ["length", "breadth", "height"].forEach(type => {
            linePointTolerancePx[type] = Math.max(3, Math.round(med * LINE_POINT_TOLERANCE_FACTOR));
            document.getElementById(type + "Tolerance").value = linePointTolerancePx[type];
            document.getElementById(type + "CountValue").innerText = "--";
        });

        if (data.detected_sacks > 0) {
            document.querySelectorAll(".axis-btn").forEach(b => b.disabled = false);
            document.getElementById("axisSection").classList.remove("locked");

            currentMeasurement = "length";
            document.querySelectorAll(".measurement").forEach(el => el.classList.remove("active"));
            document.getElementById("lengthBox").classList.add("active");

            setStatus("Detected " + data.detected_sacks + " sacks (" + data.duplicates_removed +
                " duplicates removed). Now select Length and click 2 points across the pile.");
        } else {
            setStatus("No sacks detected in this image. Try a clearer photo of the pile.");
        }

        updateTotal();
        updateWarnings();

    } catch (err) {
        setStatus("Detection error: " + err.message);
    }

    detecting = false;
    btn.disabled = false;
}


// ------------------------------------------------------------
// STAGE 2 - AXIS SELECTION & COUNTING
// ------------------------------------------------------------

function selectMeasurement(type) {
    if (!detectedBoxes.length) {
        setStatus('Run "1. Detect Sacks" first.');
        return;
    }

    currentMeasurement = type;

    document.querySelectorAll(".measurement").forEach(el => el.classList.remove("active"));
    document.getElementById(type + "Box").classList.add("active");

    setStatus("Now click 2 points across the pile's " + AXIS_LABELS[type].toLowerCase() + ".");
}

function onToleranceChange(type) {
    const val = parseFloat(document.getElementById(type + "Tolerance").value);
    linePointTolerancePx[type] = isNaN(val) ? linePointTolerancePx[type] : val;
    if (lines[type].length === 2) {
        recomputeAxis(type);
    }
}

// Only points genuinely close to the drawn line are selected:
//   - perpendicular distance (d) must be within the small, named
//     linePointTolerancePx[type] -- this is what keeps dots that are
//     above/below/beside the line from being picked up.
//   - the point's projection onto the line (t) must fall within the
//     segment itself (plus a small margin so the two clicked endpoints
//     don't have to land exactly on a dot).
function countAlongLine(p1, p2, tolerancePx) {
    const dx = p2.x - p1.x, dy = p2.y - p1.y;
    const len = Math.hypot(dx, dy);
    if (len < 1) return { count: 0, ids: [] };

    const ux = dx / len, uy = dy / len;
    const nx = -uy, ny = ux;
    const margin = len * 0.08;

    const ids = [];
    detectedBoxes.forEach(b => {
        const rx = b.cx - p1.x, ry = b.cy - p1.y;
        const t = rx * ux + ry * uy;
        const d = rx * nx + ry * ny;

        if (t >= -margin && t <= len + margin && Math.abs(d) <= tolerancePx) {
            ids.push(b.id);
        }
    });

    return { count: ids.length, ids: ids };
}

function recomputeAxis(type) {
    const pts = lines[type];
    if (pts.length !== 2) return;

    const tol = linePointTolerancePx[type];
    const { count, ids } = countAlongLine(pts[0], pts[1], tol);

    axisCounts[type] = count;
    matchedBoxIds[type] = ids;

    document.getElementById(type + "CountValue").innerText = count + " sacks matched";

    redraw();
    updateTotal();
    updateWarnings();
}

function clearCurrentLine() {
    if (!currentMeasurement) {
        setStatus("Select Length, Breadth or Height first.");
        return;
    }

    lines[currentMeasurement] = [];
    axisCounts[currentMeasurement] = null;
    matchedBoxIds[currentMeasurement] = [];
    document.getElementById(currentMeasurement + "CountValue").innerText = "--";

    redraw();
    updateTotal();
    updateWarnings();

    setStatus(AXIS_LABELS[currentMeasurement] + " line cleared.");
}

canvas.addEventListener("click", function (event) {
    if (!imageLoaded) return;

    if (!detectedBoxes.length) {
        setStatus('Run "1. Detect Sacks" first before drawing reference lines.');
        return;
    }

    if (!currentMeasurement) {
        setStatus("Select Length, Breadth or Height first.");
        return;
    }

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const x = (event.clientX - rect.left) * scaleX;
    const y = (event.clientY - rect.top) * scaleY;

    const pts = lines[currentMeasurement];

    if (pts.length >= 2) {
        pts.length = 0;
        axisCounts[currentMeasurement] = null;
        matchedBoxIds[currentMeasurement] = [];
    }

    pts.push({ x: x, y: y });

    if (pts.length === 2) {
        recomputeAxis(currentMeasurement);
        setStatus(AXIS_LABELS[currentMeasurement] + ": line complete, " +
            axisCounts[currentMeasurement] + " sacks matched. Adjust line tolerance if needed.");
    } else {
        redraw();
        setStatus(AXIS_LABELS[currentMeasurement] + ": first point selected. Click second point.");
    }
});


// ------------------------------------------------------------
// TOTALS & WARNINGS
// ------------------------------------------------------------

function updateTotal() {
    const l = axisCounts.length, b = axisCounts.breadth, h = axisCounts.height;

    document.getElementById("resultSacksLength").innerText = (l === null ? "--" : l);
    document.getElementById("resultSacksBreadth").innerText = (b === null ? "--" : b);
    document.getElementById("resultSacksHeight").innerText = (h === null ? "--" : h);

    const totalEl = document.getElementById("resultTotal");
    if (l && b && h) {
        totalEl.innerText = (l * b * h) + " sacks";
    } else {
        totalEl.innerText = "--";
    }
}

function updateWarnings() {
    const warnings = [];

    ["length", "breadth", "height"].forEach(type => {
        const c = axisCounts[type];
        if (lines[type].length === 2) {
            if (c === 0) {
                warnings.push(AXIS_LABELS[type] + ": no sacks matched near this line. Increase the line tolerance slightly or redraw closer to sack centers.");
            } else if (c === 1 && detectedBoxes.length > 8) {
                warnings.push(AXIS_LABELS[type] + ": only 1 sack matched - this count looks unreliable. Adjust the line tolerance or the line position.");
            }
        }
    });

    if (detectedBoxes.length > 0 && detectedBoxes.length < 3) {
        warnings.push("Very few sacks were detected overall - axis counts and the total will not be reliable.");
    }

    if (autoHeight && lines.height.length === 2 && axisCounts.height !== null) {
        if (Math.abs(axisCounts.height - autoHeight.count) > Math.max(2, autoHeight.count * 0.4)) {
            warnings.push("Height line count (" + axisCounts.height + ") disagrees a lot with the automatic height hint (" +
                autoHeight.count + "). Re-check the line placement.");
        }
    }

    const box = document.getElementById("warnings");
    if (warnings.length) {
        box.style.display = "block";
        box.innerHTML = "&#9888; " + warnings.join("<br>&#9888; ");
    } else {
        box.style.display = "none";
        box.innerHTML = "";
    }
}


// ------------------------------------------------------------
// DRAWING
// ------------------------------------------------------------

function drawAxisLine(type) {
    const pts = lines[type];
    if (!pts.length) return;

    const color = AXIS_COLORS[type];

    pts.forEach((p, index) => {
        ctx.beginPath();
        ctx.arc(p.x, p.y, 8, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();

        ctx.fillStyle = "white";
        ctx.font = "bold 16px Arial";
        ctx.fillText(AXIS_LABELS[type] + " " + (index + 1), p.x + 10, p.y - 10);
    });

    if (pts.length === 2) {
        const p1 = pts[0], p2 = pts[1];
        const dx = p2.x - p1.x, dy = p2.y - p1.y;
        const len = Math.hypot(dx, dy) || 1;
        const ux = dx / len, uy = dy / len;
        const nx = -uy, ny = ux;
        const tol = linePointTolerancePx[type];

        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.stroke();

        ctx.setLineDash([6, 6]);
        ctx.lineWidth = 1.5;
        [tol, -tol].forEach(offset => {
            ctx.beginPath();
            ctx.moveTo(p1.x + nx * offset, p1.y + ny * offset);
            ctx.lineTo(p2.x + nx * offset, p2.y + ny * offset);
            ctx.strokeStyle = color;
            ctx.stroke();
        });
        ctx.setLineDash([]);

        (matchedBoxIds[type] || []).forEach(id => {
            const b = detectedBoxes.find(bb => bb.id === id);
            if (!b) return;
            ctx.beginPath();
            ctx.arc(b.cx, b.cy, 7, 0, Math.PI * 2);
            ctx.globalAlpha = 0.85;
            ctx.fillStyle = color;
            ctx.fill();
            ctx.globalAlpha = 1.0;
        });

        const midX = (p1.x + p2.x) / 2, midY = (p1.y + p2.y) / 2;
        ctx.fillStyle = color;
        ctx.font = "bold 20px Arial";
        const cnt = axisCounts[type];
        ctx.fillText(AXIS_LABELS[type] + ": " + (cnt === null ? "?" : cnt) + " sacks", midX, midY - 16);
    }
}

function redraw() {
    if (!imageLoaded) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

    drawAxisLine("length");
    drawAxisLine("breadth");
    drawAxisLine("height");
}


// ------------------------------------------------------------
// RESET
// ------------------------------------------------------------

function resetAll() {
    document.getElementById("imageInput").value = "";

    imageLoaded = false;
    detecting = false;
    detectedBoxes = [];
    autoHeight = null;
    lines = { length: [], breadth: [], height: [] };
    linePointTolerancePx = { length: 15, breadth: 15, height: 15 };
    axisCounts = { length: null, breadth: null, height: null };
    matchedBoxIds = { length: [], breadth: [], height: [] };
    currentMeasurement = null;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    canvas.width = 0;
    canvas.height = 0;

    resetResultFields();

    ["length", "breadth", "height"].forEach(type => {
        document.getElementById(type + "Tolerance").value = 15;
    });

    document.querySelectorAll(".axis-btn").forEach(b => b.disabled = true);
    document.getElementById("axisSection").classList.add("locked");
    document.querySelectorAll(".measurement").forEach(el => el.classList.remove("active"));
    document.getElementById("detectBtn").disabled = true;

    setStatus("Upload an image to begin.");
}

</script>

</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML, line_tolerance_factor=LINE_POINT_TOLERANCE_FACTOR)


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "Warehouse Pile Estimator",
        "device": str(DEVICE),
    })


@app.route("/detect", methods=["POST"])
def detect():
    try:
        image_file = request.files.get("image")
        if image_file is None or image_file.filename == "":
            return jsonify({"success": False, "message": "Please upload an image first."})

        image_bytes = image_file.read()
        if len(image_bytes) == 0:
            return jsonify({"success": False, "message": "Uploaded image is empty."})

        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image_bgr is None:
            return jsonify({"success": False, "message": "The uploaded file is not a valid image."})

        annotated, boxes, avg_conf, raw_count, duplicates_removed = run_detection(image_bgr)
        auto_height = estimate_height_rows(boxes)

        is_success, buffer = cv2.imencode(".jpg", annotated)
        if not is_success:
            return jsonify({"success": False, "message": "Failed to create annotated image."})

        return jsonify({
            "success": True,
            "detected_sacks": len(boxes),
            "raw_detections": raw_count,
            "duplicates_removed": duplicates_removed,
            "avg_confidence": avg_conf,
            "boxes": boxes,
            "auto_height": auto_height,
            "image_base64": base64.b64encode(buffer.tobytes()).decode("utf-8"),
        })

    except Exception as exc:
        return jsonify({"success": False, "message": f"Runtime error: {exc}"})


if __name__ == "__main__":
    print("=" * 60)
    print("WAREHOUSE PILE ESTIMATOR - DETECTION FIRST (dots, no edges)")
    print("=" * 60)
    print(f"Model : {MODEL_PATH}")
    print(f"Device: {DEVICE}")
    print(f"Line point tolerance factor: {LINE_POINT_TOLERANCE_FACTOR}")
    print("Local URL: http://127.0.0.1:5006")
    print("Network URL: http://0.0.0.0:5006")
    print("=" * 60)

    app.run(host="0.0.0.0", port=5006, debug=False)

