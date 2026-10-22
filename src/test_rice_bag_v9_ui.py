# ============================================================================
# BACKUP OF THE ORIGINAL WORKING VERSION (kept for reference / rollback).
# To restore it: delete everything from here down to "END OF BACKUP",
# then remove the leading "# " from each of the lines above that marker.
# ============================================================================
#
# """
# RICE BAG V9 - FLASK WEB UI
#
# Web interface for rice bag detection using trained V9 model.
#
# Features:
# - Image upload
# - Real-time detection
# - Annotated image output
# - Bag count and statistics
# - Confidence display
#
# Run: python src/test_rice_bag_v9_ui.py
# Access: http://localhost:5000
# """
#
# from pathlib import Path
# from flask import Flask, request, render_template_string, send_from_directory
# from ultralytics import YOLO
# import uuid
# import cv2
# import numpy as np
# import torch
#
#
# # ============================================================
# # PATHS
# # ============================================================
#
# ROOT = Path(__file__).resolve().parents[1]
# MODEL_PATH = ROOT / "models" / "rice_bag_10_v9" / "rice_bag_10_v9_best.pt"
# UPLOAD_DIR = ROOT / "runs" / "rice_bag_10_v9" / "ui_uploads"
# UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
#
#
# # ============================================================
# # FLASK APP
# # ============================================================
#
# app = Flask(__name__)
#
# # Load model
# print(f"\nLoading V9 model...")
# if not MODEL_PATH.exists():
#     print(f"✗ Model not found: {MODEL_PATH}")
#     print("Please run training first: python src/train_rice_bag_v9.py")
#     exit(1)
#
# model = YOLO(str(MODEL_PATH))
# print(f"✓ Model loaded: {MODEL_PATH}")
# print(f"  Task: {model.task}")
#
#
# # ============================================================
# # INFERENCE SETTINGS
# # ============================================================
#
# CONF_THRESHOLD = 0.50
# IOU_THRESHOLD = 0.45
# MAX_DETECTIONS = 500
# DEVICE = 0 if torch.cuda.is_available() else "cpu"
#
#
# # ============================================================
# # HTML TEMPLATE
# # ============================================================
#
# HTML_TEMPLATE = """
# <!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="UTF-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#     <title>Rice Bag V9 - Detection</title>
#     <style>
#         * {
#             margin: 0;
#             padding: 0;
#             box-sizing: border-box;
#         }
#
#         body {
#             font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
#             background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#             min-height: 100vh;
#             display: flex;
#             align-items: center;
#             justify-content: center;
#             padding: 20px;
#         }
#
#         .container {
#             width: 100%;
#             max-width: 900px;
#             background: white;
#             border-radius: 16px;
#             box-shadow: 0 20px 60px rgba(0,0,0,0.3);
#             overflow: hidden;
#         }
#
#         .header {
#             background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#             color: white;
#             padding: 40px;
#             text-align: center;
#         }
#
#         .header h1 {
#             font-size: 32px;
#             margin-bottom: 8px;
#         }
#
#         .header p {
#             opacity: 0.9;
#             font-size: 16px;
#         }
#
#         .content {
#             padding: 40px;
#         }
#
#         .upload-section {
#             margin-bottom: 30px;
#         }
#
#         .upload-box {
#             border: 3px dashed #667eea;
#             border-radius: 12px;
#             padding: 40px;
#             text-align: center;
#             background: #f8f9ff;
#             transition: all 0.3s;
#             cursor: pointer;
#         }
#
#         .upload-box:hover {
#             border-color: #764ba2;
#             background: #f0f2ff;
#         }
#
#         .upload-box input {
#             display: none;
#         }
#
#         .upload-icon {
#             font-size: 48px;
#             margin-bottom: 12px;
#         }
#
#         .upload-text {
#             font-size: 18px;
#             color: #333;
#             margin-bottom: 8px;
#         }
#
#         .upload-subtext {
#             font-size: 14px;
#             color: #666;
#         }
#
#         .button {
#             background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#             color: white;
#             border: none;
#             padding: 12px 32px;
#             border-radius: 8px;
#             font-size: 16px;
#             font-weight: 500;
#             cursor: pointer;
#             transition: transform 0.2s;
#         }
#
#         .button:hover {
#             transform: translateY(-2px);
#         }
#
#         .button:active {
#             transform: translateY(0);
#         }
#
#         .button:disabled {
#             opacity: 0.5;
#             cursor: not-allowed;
#             transform: none;
#         }
#
#         .results {
#             display: none;
#         }
#
#         .results.show {
#             display: block;
#         }
#
#         .result-image {
#             width: 100%;
#             border-radius: 12px;
#             margin-bottom: 24px;
#             box-shadow: 0 4px 12px rgba(0,0,0,0.1);
#         }
#
#         .stats {
#             display: grid;
#             grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
#             gap: 16px;
#             margin-bottom: 24px;
#         }
#
#         .stat-card {
#             background: #f8f9ff;
#             border-left: 4px solid #667eea;
#             padding: 20px;
#             border-radius: 8px;
#             text-align: center;
#         }
#
#         .stat-value {
#             font-size: 28px;
#             font-weight: bold;
#             color: #667eea;
#             margin-bottom: 4px;
#         }
#
#         .stat-label {
#             font-size: 14px;
#             color: #666;
#             text-transform: uppercase;
#             letter-spacing: 0.5px;
#         }
#
#         .detections-list {
#             background: #f8f9ff;
#             border-radius: 8px;
#             padding: 20px;
#             max-height: 400px;
#             overflow-y: auto;
#         }
#
#         .detection-item {
#             padding: 8px 0;
#             font-size: 14px;
#             color: #333;
#             border-bottom: 1px solid #e2e8f0;
#         }
#
#         .detection-item:last-child {
#             border-bottom: none;
#         }
#
#         .detection-confidence {
#             float: right;
#             background: #667eea;
#             color: white;
#             padding: 2px 8px;
#             border-radius: 4px;
#             font-size: 12px;
#         }
#
#         .loading {
#             display: none;
#             text-align: center;
#             padding: 20px;
#         }
#
#         .loading.show {
#             display: block;
#         }
#
#         .spinner {
#             border: 4px solid #f3f3f3;
#             border-top: 4px solid #667eea;
#             border-radius: 50%;
#             width: 40px;
#             height: 40px;
#             animation: spin 1s linear infinite;
#             margin: 0 auto 12px;
#         }
#
#         @keyframes spin {
#             0% { transform: rotate(0deg); }
#             100% { transform: rotate(360deg); }
#         }
#
#         .reset-btn {
#             background: #6366f1;
#             margin-top: 20px;
#             width: 100%;
#         }
#     </style>
# </head>
# <body>
#     <div class="container">
#         <div class="header">
#             <h1>🔍 Rice Bag Detection V9</h1>
#             <p>YOLO11m Real-time Detection</p>
#         </div>
#
#         <div class="content">
#             <div class="upload-section">
#                 <div class="upload-box" onclick="document.getElementById('fileInput').click()">
#                     <div class="upload-icon">📸</div>
#                     <div class="upload-text">Click to upload an image</div>
#                     <div class="upload-subtext">or drag and drop</div>
#                     <input type="file" id="fileInput" accept="image/*">
#                 </div>
#             </div>
#
#             <div class="loading" id="loading">
#                 <div class="spinner"></div>
#                 <div>Processing image...</div>
#             </div>
#
#             <div class="results" id="results">
#                 <img id="resultImage" class="result-image" src="" alt="Detection Result">
#
#                 <div class="stats">
#                     <div class="stat-card">
#                         <div class="stat-value" id="totalBags">0</div>
#                         <div class="stat-label">Total Bags</div>
#                     </div>
#                     <div class="stat-card">
#                         <div class="stat-value" id="avgConfidence">0.0</div>
#                         <div class="stat-label">Avg Confidence</div>
#                     </div>
#                     <div class="stat-card">
#                         <div class="stat-value" id="totalWeight">0</div>
#                         <div class="stat-label">Est. Weight (kg)</div>
#                     </div>
#                 </div>
#
#                 <div class="detections-list" id="detectionsList"></div>
#
#                 <button class="button reset-btn" onclick="resetUI()">Detect Another Image</button>
#             </div>
#         </div>
#     </div>
#
#     <script>
#         const fileInput = document.getElementById('fileInput');
#         const uploadBox = document.querySelector('.upload-box');
#         const loading = document.getElementById('loading');
#         const results = document.getElementById('results');
#
#         // File input change
#         fileInput.addEventListener('change', handleFile);
#
#         // Drag and drop
#         uploadBox.addEventListener('dragover', (e) => {
#             e.preventDefault();
#             uploadBox.style.borderColor = '#764ba2';
#         });
#
#         uploadBox.addEventListener('dragleave', () => {
#             uploadBox.style.borderColor = '#667eea';
#         });
#
#         uploadBox.addEventListener('drop', (e) => {
#             e.preventDefault();
#             uploadBox.style.borderColor = '#667eea';
#             if (e.dataTransfer.files.length > 0) {
#                 fileInput.files = e.dataTransfer.files;
#                 handleFile();
#             }
#         });
#
#         function handleFile() {
#             if (fileInput.files.length === 0) return;
#
#             const file = fileInput.files[0];
#             if (!file.type.startsWith('image/')) {
#                 alert('Please select an image file');
#                 return;
#             }
#
#             const formData = new FormData();
#             formData.append('file', file);
#
#             loading.classList.add('show');
#             results.classList.remove('show');
#
#             fetch('/predict', {
#                 method: 'POST',
#                 body: formData
#             })
#             .then(response => response.json())
#             .then(data => {
#                 loading.classList.remove('show');
#
#                 if (data.success) {
#                     displayResults(data);
#                 } else {
#                     alert('Error: ' + data.message);
#                 }
#             })
#             .catch(error => {
#                 loading.classList.remove('show');
#                 alert('Error: ' + error);
#             });
#         }
#
#         function displayResults(data) {
#             document.getElementById('resultImage').src = data.image_url;
#             document.getElementById('totalBags').textContent = data.total_detections;
#             document.getElementById('avgConfidence').textContent = data.avg_confidence.toFixed(3);
#             document.getElementById('totalWeight').textContent = (data.total_detections * 50).toLocaleString();
#
#             let html = '';
#             data.detections.forEach((det, i) => {
#                 html += `<div class="detection-item">Bag ${i+1}: <span class="detection-confidence">${det.toFixed(3)}</span></div>`;
#             });
#             document.getElementById('detectionsList').innerHTML = html;
#
#             results.classList.add('show');
#         }
#
#         function resetUI() {
#             fileInput.value = '';
#             loading.classList.remove('show');
#             results.classList.remove('show');
#         }
#     </script>
# </body>
# </html>
# """
#
#
# # ============================================================
# # ROUTES
# # ============================================================
#
# @app.route("/", methods=["GET"])
# def index():
#     return render_template_string(HTML_TEMPLATE)
#
#
# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         # Get uploaded file
#         if "file" not in request.files:
#             return {"success": False, "message": "No file uploaded"}
#
#         file = request.files["file"]
#         if file.filename == "":
#             return {"success": False, "message": "No file selected"}
#
#         # Save uploaded file
#         filename = f"{uuid.uuid4()}.jpg"
#         filepath = UPLOAD_DIR / filename
#         file.save(filepath)
#
#         # Load image
#         image = cv2.imread(str(filepath))
#         if image is None:
#             return {"success": False, "message": "Failed to load image"}
#
#         # Run inference
#         results = model.predict(
#             source=str(filepath),
#             conf=CONF_THRESHOLD,
#             iou=IOU_THRESHOLD,
#             max_det=MAX_DETECTIONS,
#             device=DEVICE,
#             verbose=False,
#         )
#
#         result = results[0]
#         boxes = result.boxes
#
#         # Draw annotations
#         annotated = image.copy()
#         confidences = []
#         box_count = 0
#
#         if boxes is not None and len(boxes) > 0:
#             xyxy = boxes.xyxy.cpu().numpy()
#             confs = boxes.conf.cpu().numpy()
#             clss = boxes.cls.cpu().numpy()
#
#             for i in range(len(boxes)):
#                 x1, y1, x2, y2 = map(float, xyxy[i])
#                 conf = float(confs[i])
#                 cls = int(clss[i])
#                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#
#                 confidences.append(conf)
#                 box_count += 1
#
#                 # Draw box
#                 cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
#
#                 # Draw label
#                 label = f"Bag {i+1} ({conf:.2f})"
#                 cv2.putText(
#                     annotated, label,
#                     (x1, max(y1 - 5, 20)),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
#                 )
#
#         # Add stats
#         avg_conf = float(np.mean(confidences)) if confidences else 0.0
#         cv2.putText(
#             annotated,
#             f"Total: {box_count} bags | Avg: {avg_conf:.3f}",
#             (10, 30),
#             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
#         )
#
#         # Save result
#         output_filename = f"{uuid.uuid4()}_result.jpg"
#         output_path = UPLOAD_DIR / output_filename
#         cv2.imwrite(str(output_path), annotated)
#
#         # Return response
#         avg_conf = float(np.mean(confidences)) if confidences else 0.0
#
#         return {
#             "success": True,
#             "image_url": f"/uploads/{output_filename}",
#             "total_detections": box_count,
#             "avg_confidence": avg_conf,
#             "detections": confidences,
#         }
#
#     except Exception as e:
#         print(f"Error: {e}")
#         return {"success": False, "message": str(e)}
#
#
# @app.route("/uploads/<filename>")
# def uploaded_file(filename):
#     return send_from_directory(UPLOAD_DIR, filename)
#
#
# # ============================================================
# # MAIN
# # ============================================================
#
# if __name__ == "__main__":
#     print("\n" + "=" * 70)
#     print("RICE BAG V9 - FLASK WEB UI")
#     print("=" * 70)
#     print(f"\nModel: {MODEL_PATH}")
#     print(f"Uploads: {UPLOAD_DIR}")
#     print(f"\n✓ Starting Flask server...")
#     print(f"✓ Listen on http://0.0.0.0:5000")
#     print(f"\nOpen browser: http://localhost:5000")
#     print("=" * 70 + "\n")
#
#     app.run(host="0.0.0.0", port=5000, debug=False)
#
# ============================================================================
# END OF BACKUP
# ============================================================================


# ============================================================================
# BACKUP 2 - PREVIOUS WORKING VERSION (bounding-box + duplicate-dedup).
# This was the last working version before switching box drawing to a
# center-point/dot marker. Restore it the same way as backup 1: delete
# everything from here down to the next "END OF BACKUP 2" marker is not
# needed -- just remove the leading "# " from each line below.
# ============================================================================
#
# """
# RICE BAG V9 - FLASK WEB UI (duplicate-aware detection)
#
# Same model, same Flask routes and UI as the backed-up version above, plus a
# post-processing pass that fixes bags getting 2+ overlapping boxes -- without
# just retuning the confidence threshold.
#
# Run: python src/test_rice_bag_v9_ui.py
# Access: http://localhost:5000
# """
#
# from pathlib import Path
# from flask import Flask, request, render_template_string, send_from_directory
# from ultralytics import YOLO
# import uuid
# import cv2
# import numpy as np
# import torch
#
#
# # ============================================================
# # PATHS
# # ============================================================
#
# ROOT = Path(__file__).resolve().parents[1]
# MODEL_PATH = ROOT / "models" / "rice_bag_10_v9" / "rice_bag_10_v9_best.pt"
# UPLOAD_DIR = ROOT / "runs" / "rice_bag_10_v9" / "ui_uploads"
# UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
#
#
# # ============================================================
# # FLASK APP
# # ============================================================
#
# app = Flask(__name__)
#
# print(f"\nLoading V9 model...")
# if not MODEL_PATH.exists():
#     print(f"Model not found: {MODEL_PATH}")
#     print("Please run training first: python src/train_rice_bag_v9.py")
#     exit(1)
#
# model = YOLO(str(MODEL_PATH))
# print(f"Model loaded: {MODEL_PATH}")
# print(f"  Task: {model.task}")
#
#
# # ============================================================
# # INFERENCE SETTINGS
# # ============================================================
#
# CONF_THRESHOLD = 0.50
# IOU_THRESHOLD = 0.45       # Ultralytics' own first-pass NMS (unchanged)
# MAX_DETECTIONS = 500
# DEVICE = 0 if torch.cuda.is_available() else "cpu"
#
# # --- Duplicate-box post-processing settings ---
# # Standard IoU-NMS above only merges boxes whose IoU is high. It misses a
# # common failure mode on these bags: one box tightly wraps a sack and a
# # second, differently-SIZED box mostly overlaps it (a fold line or shadow
# # on the repetitive bag texture triggering a second detection) but has a
# # LOW IoU purely because of the size mismatch, so it survives normal NMS.
# # DEDUP_CONTAINMENT_THRESHOLD adds a check for that: intersection area
# # relative to the SMALLER box's area, which stays high regardless of the
# # size difference between the two duplicate boxes.
# DEDUP_IOU_THRESHOLD = 0.50          # classic same-footprint duplicate
# DEDUP_CONTAINMENT_THRESHOLD = 0.75  # one box mostly swallowed by another
# DEDUP_MODE = "merge"                # "merge" = confidence-weighted fusion, "suppress" = keep best / drop rest
#
#
# # ============================================================
# # DUPLICATE-BOX POST-PROCESSING
# # ============================================================
#
# def _box_area(box):
#     return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
#
#
# def _iou_and_containment(box_a, box_b):
#     """Return (IoU, containment) for two [x1, y1, x2, y2] boxes.
#
#     Two boxes on the SAME sack where one is bigger than the other have low
#     IoU but high containment. Two boxes on genuinely different, merely
#     touching sacks have both low IoU and low containment, since real
#     neighbours only share a thin sliver of overlap at most.
#     """
#     ix1, iy1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
#     ix2, iy2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
#     inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
#     if inter <= 0:
#         return 0.0, 0.0
#
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
#     """Collapse duplicate boxes on the same physical sack into one box.
#
#     1. Union-find clustering: any pair of boxes whose IoU or containment
#        exceeds its threshold joins the same cluster, so a chain of 3+
#        overlapping duplicates on one sack all end up together.
#     2. Each cluster becomes exactly one output box:
#          - mode="merge": confidence-weighted average of the cluster's
#            coordinates (usually tighter and better centered than any
#            single input box), confidence = the cluster's max confidence.
#          - mode="suppress": classic NMS behaviour -- keep the
#            highest-confidence box in the cluster, drop the rest.
#
#     Distinct, merely-adjacent sacks are left alone: real neighbours share
#     only a thin edge, so both their IoU and containment stay well below
#     the thresholds above.
#     """
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
# # HTML TEMPLATE (unchanged, plus a "Duplicates Removed" stat card)
# # ============================================================
#
# HTML_TEMPLATE = """
# <!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="UTF-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#     <title>Rice Bag V9 - Detection</title>
#     <style>
#         * {
#             margin: 0;
#             padding: 0;
#             box-sizing: border-box;
#         }
#
#         body {
#             font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
#             background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#             min-height: 100vh;
#             display: flex;
#             align-items: center;
#             justify-content: center;
#             padding: 20px;
#         }
#
#         .container {
#             width: 100%;
#             max-width: 900px;
#             background: white;
#             border-radius: 16px;
#             box-shadow: 0 20px 60px rgba(0,0,0,0.3);
#             overflow: hidden;
#         }
#
#         .header {
#             background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#             color: white;
#             padding: 40px;
#             text-align: center;
#         }
#
#         .header h1 {
#             font-size: 32px;
#             margin-bottom: 8px;
#         }
#
#         .header p {
#             opacity: 0.9;
#             font-size: 16px;
#         }
#
#         .content {
#             padding: 40px;
#         }
#
#         .upload-section {
#             margin-bottom: 30px;
#         }
#
#         .upload-box {
#             border: 3px dashed #667eea;
#             border-radius: 12px;
#             padding: 40px;
#             text-align: center;
#             background: #f8f9ff;
#             transition: all 0.3s;
#             cursor: pointer;
#         }
#
#         .upload-box:hover {
#             border-color: #764ba2;
#             background: #f0f2ff;
#         }
#
#         .upload-box input {
#             display: none;
#         }
#
#         .upload-icon {
#             font-size: 48px;
#             margin-bottom: 12px;
#         }
#
#         .upload-text {
#             font-size: 18px;
#             color: #333;
#             margin-bottom: 8px;
#         }
#
#         .upload-subtext {
#             font-size: 14px;
#             color: #666;
#         }
#
#         .button {
#             background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#             color: white;
#             border: none;
#             padding: 12px 32px;
#             border-radius: 8px;
#             font-size: 16px;
#             font-weight: 500;
#             cursor: pointer;
#             transition: transform 0.2s;
#         }
#
#         .button:hover {
#             transform: translateY(-2px);
#         }
#
#         .button:active {
#             transform: translateY(0);
#         }
#
#         .button:disabled {
#             opacity: 0.5;
#             cursor: not-allowed;
#             transform: none;
#         }
#
#         .results {
#             display: none;
#         }
#
#         .results.show {
#             display: block;
#         }
#
#         .result-image {
#             width: 100%;
#             border-radius: 12px;
#             margin-bottom: 24px;
#             box-shadow: 0 4px 12px rgba(0,0,0,0.1);
#         }
#
#         .stats {
#             display: grid;
#             grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
#             gap: 16px;
#             margin-bottom: 24px;
#         }
#
#         .stat-card {
#             background: #f8f9ff;
#             border-left: 4px solid #667eea;
#             padding: 20px;
#             border-radius: 8px;
#             text-align: center;
#         }
#
#         .stat-value {
#             font-size: 28px;
#             font-weight: bold;
#             color: #667eea;
#             margin-bottom: 4px;
#         }
#
#         .stat-label {
#             font-size: 14px;
#             color: #666;
#             text-transform: uppercase;
#             letter-spacing: 0.5px;
#         }
#
#         .detections-list {
#             background: #f8f9ff;
#             border-radius: 8px;
#             padding: 20px;
#             max-height: 400px;
#             overflow-y: auto;
#         }
#
#         .detection-item {
#             padding: 8px 0;
#             font-size: 14px;
#             color: #333;
#             border-bottom: 1px solid #e2e8f0;
#         }
#
#         .detection-item:last-child {
#             border-bottom: none;
#         }
#
#         .detection-confidence {
#             float: right;
#             background: #667eea;
#             color: white;
#             padding: 2px 8px;
#             border-radius: 4px;
#             font-size: 12px;
#         }
#
#         .loading {
#             display: none;
#             text-align: center;
#             padding: 20px;
#         }
#
#         .loading.show {
#             display: block;
#         }
#
#         .spinner {
#             border: 4px solid #f3f3f3;
#             border-top: 4px solid #667eea;
#             border-radius: 50%;
#             width: 40px;
#             height: 40px;
#             animation: spin 1s linear infinite;
#             margin: 0 auto 12px;
#         }
#
#         @keyframes spin {
#             0% { transform: rotate(0deg); }
#             100% { transform: rotate(360deg); }
#         }
#
#         .reset-btn {
#             background: #6366f1;
#             margin-top: 20px;
#             width: 100%;
#         }
#     </style>
# </head>
# <body>
#     <div class="container">
#         <div class="header">
#             <h1>🔍 Rice Bag Detection V9</h1>
#             <p>YOLO11m Real-time Detection (duplicate-aware)</p>
#         </div>
#
#         <div class="content">
#             <div class="upload-section">
#                 <div class="upload-box" onclick="document.getElementById('fileInput').click()">
#                     <div class="upload-icon">📸</div>
#                     <div class="upload-text">Click to upload an image</div>
#                     <div class="upload-subtext">or drag and drop</div>
#                     <input type="file" id="fileInput" accept="image/*">
#                 </div>
#             </div>
#
#             <div class="loading" id="loading">
#                 <div class="spinner"></div>
#                 <div>Processing image...</div>
#             </div>
#
#             <div class="results" id="results">
#                 <img id="resultImage" class="result-image" src="" alt="Detection Result">
#
#                 <div class="stats">
#                     <div class="stat-card">
#                         <div class="stat-value" id="totalBags">0</div>
#                         <div class="stat-label">Total Bags</div>
#                     </div>
#                     <div class="stat-card">
#                         <div class="stat-value" id="avgConfidence">0.0</div>
#                         <div class="stat-label">Avg Confidence</div>
#                     </div>
#                     <div class="stat-card">
#                         <div class="stat-value" id="duplicatesRemoved">0</div>
#                         <div class="stat-label">Duplicates Removed</div>
#                     </div>
#                     <div class="stat-card">
#                         <div class="stat-value" id="totalWeight">0</div>
#                         <div class="stat-label">Est. Weight (kg)</div>
#                     </div>
#                 </div>
#
#                 <div class="detections-list" id="detectionsList"></div>
#
#                 <button class="button reset-btn" onclick="resetUI()">Detect Another Image</button>
#             </div>
#         </div>
#     </div>
#
#     <script>
#         const fileInput = document.getElementById('fileInput');
#         const uploadBox = document.querySelector('.upload-box');
#         const loading = document.getElementById('loading');
#         const results = document.getElementById('results');
#
#         // File input change
#         fileInput.addEventListener('change', handleFile);
#
#         // Drag and drop
#         uploadBox.addEventListener('dragover', (e) => {
#             e.preventDefault();
#             uploadBox.style.borderColor = '#764ba2';
#         });
#
#         uploadBox.addEventListener('dragleave', () => {
#             uploadBox.style.borderColor = '#667eea';
#         });
#
#         uploadBox.addEventListener('drop', (e) => {
#             e.preventDefault();
#             uploadBox.style.borderColor = '#667eea';
#             if (e.dataTransfer.files.length > 0) {
#                 fileInput.files = e.dataTransfer.files;
#                 handleFile();
#             }
#         });
#
#         function handleFile() {
#             if (fileInput.files.length === 0) return;
#
#             const file = fileInput.files[0];
#             if (!file.type.startsWith('image/')) {
#                 alert('Please select an image file');
#                 return;
#             }
#
#             const formData = new FormData();
#             formData.append('file', file);
#
#             loading.classList.add('show');
#             results.classList.remove('show');
#
#             fetch('/predict', {
#                 method: 'POST',
#                 body: formData
#             })
#             .then(response => response.json())
#             .then(data => {
#                 loading.classList.remove('show');
#
#                 if (data.success) {
#                     displayResults(data);
#                 } else {
#                     alert('Error: ' + data.message);
#                 }
#             })
#             .catch(error => {
#                 loading.classList.remove('show');
#                 alert('Error: ' + error);
#             });
#         }
#
#         function displayResults(data) {
#             document.getElementById('resultImage').src = data.image_url;
#             document.getElementById('totalBags').textContent = data.total_detections;
#             document.getElementById('avgConfidence').textContent = data.avg_confidence.toFixed(3);
#             document.getElementById('duplicatesRemoved').textContent = data.duplicates_removed;
#             document.getElementById('totalWeight').textContent = (data.total_detections * 50).toLocaleString();
#
#             let html = '';
#             data.detections.forEach((det, i) => {
#                 html += `<div class="detection-item">Bag ${i+1}: <span class="detection-confidence">${det.toFixed(3)}</span></div>`;
#             });
#             document.getElementById('detectionsList').innerHTML = html;
#
#             results.classList.add('show');
#         }
#
#         function resetUI() {
#             fileInput.value = '';
#             loading.classList.remove('show');
#             results.classList.remove('show');
#         }
#     </script>
# </body>
# </html>
# """
#
#
# # ============================================================
# # ROUTES
# # ============================================================
#
# @app.route("/", methods=["GET"])
# def index():
#     return render_template_string(HTML_TEMPLATE)
#
#
# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         # Get uploaded file
#         if "file" not in request.files:
#             return {"success": False, "message": "No file uploaded"}
#
#         file = request.files["file"]
#         if file.filename == "":
#             return {"success": False, "message": "No file selected"}
#
#         # Save uploaded file
#         filename = f"{uuid.uuid4()}.jpg"
#         filepath = UPLOAD_DIR / filename
#         file.save(filepath)
#
#         # Load image
#         image = cv2.imread(str(filepath))
#         if image is None:
#             return {"success": False, "message": "Failed to load image"}
#
#         # Run inference
#         results = model.predict(
#             source=str(filepath),
#             conf=CONF_THRESHOLD,
#             iou=IOU_THRESHOLD,
#             max_det=MAX_DETECTIONS,
#             device=DEVICE,
#             verbose=False,
#         )
#
#         result = results[0]
#         boxes = result.boxes
#
#         # Draw annotations
#         annotated = image.copy()
#         confidences = []
#         box_count = 0
#         raw_count = 0
#
#         if boxes is not None and len(boxes) > 0:
#             xyxy = boxes.xyxy.cpu().numpy()
#             confs = boxes.conf.cpu().numpy()
#             raw_count = len(boxes)
#
#             # --- duplicate-box post-processing (the actual fix) ---
#             xyxy, confs = deduplicate_boxes(xyxy, confs)
#
#             for i in range(len(xyxy)):
#                 x1, y1, x2, y2 = map(float, xyxy[i])
#                 conf = float(confs[i])
#                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#
#                 confidences.append(conf)
#                 box_count += 1
#
#                 # Draw box
#                 cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
#
#                 # Draw label
#                 label = f"Bag {box_count} ({conf:.2f})"
#                 cv2.putText(
#                     annotated, label,
#                     (x1, max(y1 - 5, 20)),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
#                 )
#
#         # Add stats
#         avg_conf = float(np.mean(confidences)) if confidences else 0.0
#         duplicates_removed = raw_count - box_count
#         cv2.putText(
#             annotated,
#             f"Total: {box_count} bags | Avg: {avg_conf:.3f} | Dupes removed: {duplicates_removed}",
#             (10, 30),
#             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
#         )
#
#         # Save result
#         output_filename = f"{uuid.uuid4()}_result.jpg"
#         output_path = UPLOAD_DIR / output_filename
#         cv2.imwrite(str(output_path), annotated)
#
#         # Return response
#         return {
#             "success": True,
#             "image_url": f"/uploads/{output_filename}",
#             "total_detections": box_count,
#             "raw_detections": raw_count,
#             "duplicates_removed": duplicates_removed,
#             "avg_confidence": avg_conf,
#             "detections": confidences,
#         }
#
#     except Exception as e:
#         print(f"Error: {e}")
#         return {"success": False, "message": str(e)}
#
#
# @app.route("/uploads/<filename>")
# def uploaded_file(filename):
#     return send_from_directory(UPLOAD_DIR, filename)
#
#
# # ============================================================
# # MAIN
# # ============================================================
#
# if __name__ == "__main__":
#     print("\n" + "=" * 70)
#     print("RICE BAG V9 - FLASK WEB UI (duplicate-aware)")
#     print("=" * 70)
#     print(f"\nModel: {MODEL_PATH}")
#     print(f"Uploads: {UPLOAD_DIR}")
#     print(f"\nStarting Flask server...")
#     print(f"Listen on http://0.0.0.0:5000")
#     print(f"\nOpen browser: http://localhost:5000")
#     print("=" * 70 + "\n")
#
#     app.run(host="0.0.0.0", port=5000, debug=False)
#
# ============================================================================
# END OF BACKUP 2
# ============================================================================


# ============================================================================
# BACKUP 3 - PREVIOUS WORKING VERSION (dot detection, box-level dedup only).
# This was the last version before adding a second-stage center-point
# distance clustering pass to catch same-bag duplicate dots that box-level
# IoU/containment dedup alone didn't catch. Restore the same way as the
# earlier backups: just remove the leading "# " from each line below.
# ============================================================================
#
# """
# RICE BAG V9 - FLASK WEB UI (center-point / dot detection)
#
# Same model + same duplicate-dedup pipeline as backup 2, but each detected
# sack is drawn as a single dot at its box center instead of a rectangle.
#
# Run: python src/test_rice_bag_v9_ui.py
# Access: http://localhost:5000
# """
#
# from pathlib import Path
# from flask import Flask, request, render_template_string, send_from_directory
# from ultralytics import YOLO
# import uuid
# import cv2
# import numpy as np
# import torch
#
#
# ROOT = Path(__file__).resolve().parents[1]
# MODEL_PATH = ROOT / "models" / "rice_bag_10_v9" / "rice_bag_10_v9_best.pt"
# UPLOAD_DIR = ROOT / "runs" / "rice_bag_10_v9" / "ui_uploads"
# UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
#
# app = Flask(__name__)
#
# print(f"\nLoading V9 model...")
# if not MODEL_PATH.exists():
#     print(f"Model not found: {MODEL_PATH}")
#     print("Please run training first: python src/train_rice_bag_v9.py")
#     exit(1)
#
# model = YOLO(str(MODEL_PATH))
# print(f"Model loaded: {MODEL_PATH}")
# print(f"  Task: {model.task}")
#
# CONF_THRESHOLD = 0.50
# IOU_THRESHOLD = 0.45
# MAX_DETECTIONS = 500
# DEVICE = 0 if torch.cuda.is_available() else "cpu"
#
# DEDUP_IOU_THRESHOLD = 0.50
# DEDUP_CONTAINMENT_THRESHOLD = 0.75
# DEDUP_MODE = "merge"
#
# DOT_RADIUS = 6  # px, tune if dots look too small/large on your images
#
#
# # ============================================================
# # DUPLICATE-BOX POST-PROCESSING (same as backup 2)
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
# # HTML TEMPLATE (unchanged from backup 2)
# # ============================================================
#
# HTML_TEMPLATE = """
# <!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="UTF-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#     <title>Rice Bag V9 - Detection</title>
#     <style>
#         * {
#             margin: 0;
#             padding: 0;
#             box-sizing: border-box;
#         }
#
#         body {
#             font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
#             background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#             min-height: 100vh;
#             display: flex;
#             align-items: center;
#             justify-content: center;
#             padding: 20px;
#         }
#
#         .container {
#             width: 100%;
#             max-width: 900px;
#             background: white;
#             border-radius: 16px;
#             box-shadow: 0 20px 60px rgba(0,0,0,0.3);
#             overflow: hidden;
#         }
#
#         .header {
#             background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#             color: white;
#             padding: 40px;
#             text-align: center;
#         }
#
#         .header h1 {
#             font-size: 32px;
#             margin-bottom: 8px;
#         }
#
#         .header p {
#             opacity: 0.9;
#             font-size: 16px;
#         }
#
#         .content {
#             padding: 40px;
#         }
#
#         .upload-section {
#             margin-bottom: 30px;
#         }
#
#         .upload-box {
#             border: 3px dashed #667eea;
#             border-radius: 12px;
#             padding: 40px;
#             text-align: center;
#             background: #f8f9ff;
#             transition: all 0.3s;
#             cursor: pointer;
#         }
#
#         .upload-box:hover {
#             border-color: #764ba2;
#             background: #f0f2ff;
#         }
#
#         .upload-box input {
#             display: none;
#         }
#
#         .upload-icon {
#             font-size: 48px;
#             margin-bottom: 12px;
#         }
#
#         .upload-text {
#             font-size: 18px;
#             color: #333;
#             margin-bottom: 8px;
#         }
#
#         .upload-subtext {
#             font-size: 14px;
#             color: #666;
#         }
#
#         .button {
#             background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#             color: white;
#             border: none;
#             padding: 12px 32px;
#             border-radius: 8px;
#             font-size: 16px;
#             font-weight: 500;
#             cursor: pointer;
#             transition: transform 0.2s;
#         }
#
#         .button:hover {
#             transform: translateY(-2px);
#         }
#
#         .button:active {
#             transform: translateY(0);
#         }
#
#         .button:disabled {
#             opacity: 0.5;
#             cursor: not-allowed;
#             transform: none;
#         }
#
#         .results {
#             display: none;
#         }
#
#         .results.show {
#             display: block;
#         }
#
#         .result-image {
#             width: 100%;
#             border-radius: 12px;
#             margin-bottom: 24px;
#             box-shadow: 0 4px 12px rgba(0,0,0,0.1);
#         }
#
#         .stats {
#             display: grid;
#             grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
#             gap: 16px;
#             margin-bottom: 24px;
#         }
#
#         .stat-card {
#             background: #f8f9ff;
#             border-left: 4px solid #667eea;
#             padding: 20px;
#             border-radius: 8px;
#             text-align: center;
#         }
#
#         .stat-value {
#             font-size: 28px;
#             font-weight: bold;
#             color: #667eea;
#             margin-bottom: 4px;
#         }
#
#         .stat-label {
#             font-size: 14px;
#             color: #666;
#             text-transform: uppercase;
#             letter-spacing: 0.5px;
#         }
#
#         .detections-list {
#             background: #f8f9ff;
#             border-radius: 8px;
#             padding: 20px;
#             max-height: 400px;
#             overflow-y: auto;
#         }
#
#         .detection-item {
#             padding: 8px 0;
#             font-size: 14px;
#             color: #333;
#             border-bottom: 1px solid #e2e8f0;
#         }
#
#         .detection-item:last-child {
#             border-bottom: none;
#         }
#
#         .detection-confidence {
#             float: right;
#             background: #667eea;
#             color: white;
#             padding: 2px 8px;
#             border-radius: 4px;
#             font-size: 12px;
#         }
#
#         .loading {
#             display: none;
#             text-align: center;
#             padding: 20px;
#         }
#
#         .loading.show {
#             display: block;
#         }
#
#         .spinner {
#             border: 4px solid #f3f3f3;
#             border-top: 4px solid #667eea;
#             border-radius: 50%;
#             width: 40px;
#             height: 40px;
#             animation: spin 1s linear infinite;
#             margin: 0 auto 12px;
#         }
#
#         @keyframes spin {
#             0% { transform: rotate(0deg); }
#             100% { transform: rotate(360deg); }
#         }
#
#         .reset-btn {
#             background: #6366f1;
#             margin-top: 20px;
#             width: 100%;
#         }
#     </style>
# </head>
# <body>
#     <div class="container">
#         <div class="header">
#             <h1>🔍 Rice Bag Detection V9</h1>
#             <p>YOLO11m Real-time Detection (center-point dots)</p>
#         </div>
#
#         <div class="content">
#             <div class="upload-section">
#                 <div class="upload-box" onclick="document.getElementById('fileInput').click()">
#                     <div class="upload-icon">📸</div>
#                     <div class="upload-text">Click to upload an image</div>
#                     <div class="upload-subtext">or drag and drop</div>
#                     <input type="file" id="fileInput" accept="image/*">
#                 </div>
#             </div>
#
#             <div class="loading" id="loading">
#                 <div class="spinner"></div>
#                 <div>Processing image...</div>
#             </div>
#
#             <div class="results" id="results">
#                 <img id="resultImage" class="result-image" src="" alt="Detection Result">
#
#                 <div class="stats">
#                     <div class="stat-card">
#                         <div class="stat-value" id="totalBags">0</div>
#                         <div class="stat-label">Total Bags</div>
#                     </div>
#                     <div class="stat-card">
#                         <div class="stat-value" id="avgConfidence">0.0</div>
#                         <div class="stat-label">Avg Confidence</div>
#                     </div>
#                     <div class="stat-card">
#                         <div class="stat-value" id="duplicatesRemoved">0</div>
#                         <div class="stat-label">Duplicates Removed</div>
#                     </div>
#                     <div class="stat-card">
#                         <div class="stat-value" id="totalWeight">0</div>
#                         <div class="stat-label">Est. Weight (kg)</div>
#                     </div>
#                 </div>
#
#                 <div class="detections-list" id="detectionsList"></div>
#
#                 <button class="button reset-btn" onclick="resetUI()">Detect Another Image</button>
#             </div>
#         </div>
#     </div>
#
#     <script>
#         const fileInput = document.getElementById('fileInput');
#         const uploadBox = document.querySelector('.upload-box');
#         const loading = document.getElementById('loading');
#         const results = document.getElementById('results');
#
#         fileInput.addEventListener('change', handleFile);
#
#         uploadBox.addEventListener('dragover', (e) => {
#             e.preventDefault();
#             uploadBox.style.borderColor = '#764ba2';
#         });
#
#         uploadBox.addEventListener('dragleave', () => {
#             uploadBox.style.borderColor = '#667eea';
#         });
#
#         uploadBox.addEventListener('drop', (e) => {
#             e.preventDefault();
#             uploadBox.style.borderColor = '#667eea';
#             if (e.dataTransfer.files.length > 0) {
#                 fileInput.files = e.dataTransfer.files;
#                 handleFile();
#             }
#         });
#
#         function handleFile() {
#             if (fileInput.files.length === 0) return;
#
#             const file = fileInput.files[0];
#             if (!file.type.startsWith('image/')) {
#                 alert('Please select an image file');
#                 return;
#             }
#
#             const formData = new FormData();
#             formData.append('file', file);
#
#             loading.classList.add('show');
#             results.classList.remove('show');
#
#             fetch('/predict', {
#                 method: 'POST',
#                 body: formData
#             })
#             .then(response => response.json())
#             .then(data => {
#                 loading.classList.remove('show');
#
#                 if (data.success) {
#                     displayResults(data);
#                 } else {
#                     alert('Error: ' + data.message);
#                 }
#             })
#             .catch(error => {
#                 loading.classList.remove('show');
#                 alert('Error: ' + error);
#             });
#         }
#
#         function displayResults(data) {
#             document.getElementById('resultImage').src = data.image_url;
#             document.getElementById('totalBags').textContent = data.total_detections;
#             document.getElementById('avgConfidence').textContent = data.avg_confidence.toFixed(3);
#             document.getElementById('duplicatesRemoved').textContent = data.duplicates_removed;
#             document.getElementById('totalWeight').textContent = (data.total_detections * 50).toLocaleString();
#
#             let html = '';
#             data.detections.forEach((det, i) => {
#                 html += `<div class="detection-item">Bag ${i+1}: <span class="detection-confidence">${det.toFixed(3)}</span></div>`;
#             });
#             document.getElementById('detectionsList').innerHTML = html;
#
#             results.classList.add('show');
#         }
#
#         function resetUI() {
#             fileInput.value = '';
#             loading.classList.remove('show');
#             results.classList.remove('show');
#         }
#     </script>
# </body>
# </html>
# """
#
#
# # ============================================================
# # ROUTES
# # ============================================================
#
# @app.route("/", methods=["GET"])
# def index():
#     return render_template_string(HTML_TEMPLATE)
#
#
# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         if "file" not in request.files:
#             return {"success": False, "message": "No file uploaded"}
#
#         file = request.files["file"]
#         if file.filename == "":
#             return {"success": False, "message": "No file selected"}
#
#         filename = f"{uuid.uuid4()}.jpg"
#         filepath = UPLOAD_DIR / filename
#         file.save(filepath)
#
#         image = cv2.imread(str(filepath))
#         if image is None:
#             return {"success": False, "message": "Failed to load image"}
#
#         results = model.predict(
#             source=str(filepath),
#             conf=CONF_THRESHOLD,
#             iou=IOU_THRESHOLD,
#             max_det=MAX_DETECTIONS,
#             device=DEVICE,
#             verbose=False,
#         )
#
#         result = results[0]
#         boxes = result.boxes
#
#         annotated = image.copy()
#         confidences = []
#         box_count = 0
#         raw_count = 0
#
#         if boxes is not None and len(boxes) > 0:
#             xyxy = boxes.xyxy.cpu().numpy()
#             confs = boxes.conf.cpu().numpy()
#             raw_count = len(boxes)
#
#             # duplicate-box post-processing, same as backup 2
#             xyxy, confs = deduplicate_boxes(xyxy, confs)
#
#             for i in range(len(xyxy)):
#                 x1, y1, x2, y2 = map(float, xyxy[i])
#                 conf = float(confs[i])
#                 cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
#
#                 confidences.append(conf)
#                 box_count += 1
#
#                 # --- bbox replaced with a single center-point dot ---
#                 cv2.circle(annotated, (cx, cy), DOT_RADIUS, (0, 255, 0), -1)
#                 cv2.circle(annotated, (cx, cy), DOT_RADIUS + 2, (0, 0, 0), 1)
#
#                 label = f"{box_count} ({conf:.2f})"
#                 cv2.putText(
#                     annotated, label,
#                     (cx + DOT_RADIUS + 3, cy - DOT_RADIUS),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1
#                 )
#
#         avg_conf = float(np.mean(confidences)) if confidences else 0.0
#         duplicates_removed = raw_count - box_count
#         cv2.putText(
#             annotated,
#             f"Total: {box_count} bags | Avg: {avg_conf:.3f} | Dupes removed: {duplicates_removed}",
#             (10, 30),
#             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
#         )
#
#         output_filename = f"{uuid.uuid4()}_result.jpg"
#         output_path = UPLOAD_DIR / output_filename
#         cv2.imwrite(str(output_path), annotated)
#
#         return {
#             "success": True,
#             "image_url": f"/uploads/{output_filename}",
#             "total_detections": box_count,
#             "raw_detections": raw_count,
#             "duplicates_removed": duplicates_removed,
#             "avg_confidence": avg_conf,
#             "detections": confidences,
#         }
#
#     except Exception as e:
#         print(f"Error: {e}")
#         return {"success": False, "message": str(e)}
#
#
# @app.route("/uploads/<filename>")
# def uploaded_file(filename):
#     return send_from_directory(UPLOAD_DIR, filename)
#
#
# if __name__ == "__main__":
#     print("\n" + "=" * 70)
#     print("RICE BAG V9 - FLASK WEB UI (center-point dots)")
#     print("=" * 70)
#     print(f"\nModel: {MODEL_PATH}")
#     print(f"Uploads: {UPLOAD_DIR}")
#     print(f"\nStarting Flask server...")
#     print(f"Listen on http://0.0.0.0:5000")
#     print(f"\nOpen browser: http://localhost:5000")
#     print("=" * 70 + "\n")
#
#     app.run(host="0.0.0.0", port=5000, debug=False)
#
# ============================================================================
# END OF BACKUP 3
# ============================================================================


# ============================================================
# VERSION 3 / NEW FINE-TUNED DOT DETECTION
# ============================================================
"""
RICE BAG V9 - FLASK WEB UI (fine-tuned dot detection, v3)

Same model, same Flask routes/UI as backup 3, plus a second post-processing
stage that catches same-bag duplicate dots the box-level dedup (backup 3)
alone doesn't catch. See the module-level comment block below the imports
for why this was needed and how it works.

Run: python src/test_rice_bag_v9_ui.py
Access: http://localhost:5000
"""

from pathlib import Path
from flask import Flask, request, render_template_string, send_from_directory
from ultralytics import YOLO
import uuid
import cv2
import numpy as np
import torch
import statistics


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "rice_bag_10_v9" / "rice_bag_10_v9_best.pt"
UPLOAD_DIR = ROOT / "runs" / "rice_bag_10_v9" / "ui_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

print(f"\nLoading V9 model...")
if not MODEL_PATH.exists():
    print(f"Model not found: {MODEL_PATH}")
    print("Please run training first: python src/train_rice_bag_v9.py")
    exit(1)

model = YOLO(str(MODEL_PATH))
print(f"Model loaded: {MODEL_PATH}")
print(f"  Task: {model.task}")

CONF_THRESHOLD = 0.50
IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 500
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# --- Stage A: box-level dedup (same as backup 3, unchanged) ---
DEDUP_IOU_THRESHOLD = 0.50
DEDUP_CONTAINMENT_THRESHOLD = 0.75
DEDUP_MODE = "merge"

# --- Stage B: center-point distance clustering (NEW in v3) ---
# See explanation below. CENTER_DIST_FACTOR is a fraction of the median
# detected bag size: two dots closer together than this fraction are
# treated as the same physical bag. Genuinely adjacent, distinct bags in
# a packed pile sit roughly one full bag-width apart, so keeping this
# well under 1.0 (default 0.55) merges true duplicates without merging
# real neighbours.
CENTER_DIST_FACTOR = 0.35

DOT_RADIUS = 2  # px, tune if dots look too small/large on your images


# ============================================================
# WHY DUPLICATE DOTS WERE STILL HAPPENING
# ============================================================
# Backup 3's dedup only looks at BOX overlap: two boxes merge if their IoU
# is high, or if one box's area is mostly contained inside the other's.
# That catches the common case (two boxes drawn over nearly the same
# footprint). It does NOT catch a second, very common case on these bags:
# the repetitive fold/stitch/shadow pattern on a rice sack sometimes
# triggers two CONFIDENT, TIGHT detections on two different sub-regions of
# the SAME physical bag (e.g. top-left corner and bottom-right corner of
# one sack) -- their boxes barely overlap, so both IoU and containment
# stay low and they sail through Stage A as "different objects", even
# though their center points sit right next to each other on one bag.
#
# The fix (Stage B below) stops thinking in terms of boxes at all for
# this second pass and instead clusters the DOT CENTERS directly: if two
# centers are closer than CENTER_DIST_FACTOR times the typical bag size,
# they're almost certainly the same physical bag, regardless of how their
# boxes happened to overlap.


# ============================================================
# STAGE A - BOX-LEVEL DEDUP (identical to backup 3)
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
# STAGE B - CENTER-POINT DISTANCE CLUSTERING (NEW in v3)
# ============================================================

def cluster_center_points(boxes, distance_factor=CENTER_DIST_FACTOR):
    """Second dedup pass, run AFTER Stage A, operating purely on center
    points instead of box overlap.

    Adaptive radius: computed per-image from the MEDIAN width/height of
    Stage A's surviving boxes, so it automatically scales with however
    large or small the bags appear in a given photo (close-up vs. wide
    warehouse shot) instead of using one fixed pixel distance for every
    image.

    Union-find clustering (same pattern as Stage A) so a chain of 3+
    nearby duplicate centers on one bag all collapse together, not just
    pairs. Each cluster becomes one confidence-weighted dot.
    """
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
# HTML TEMPLATE (unchanged from backup 3)
# ============================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rice Bag V9 - Detection</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }

        .container {
            width: 100%;
            max-width: 900px;
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }

        .header h1 {
            font-size: 32px;
            margin-bottom: 8px;
        }

        .header p {
            opacity: 0.9;
            font-size: 16px;
        }

        .content {
            padding: 40px;
        }

        .upload-section {
            margin-bottom: 30px;
        }

        .upload-box {
            border: 3px dashed #667eea;
            border-radius: 12px;
            padding: 40px;
            text-align: center;
            background: #f8f9ff;
            transition: all 0.3s;
            cursor: pointer;
        }

        .upload-box:hover {
            border-color: #764ba2;
            background: #f0f2ff;
        }

        .upload-box input {
            display: none;
        }

        .upload-icon {
            font-size: 48px;
            margin-bottom: 12px;
        }

        .upload-text {
            font-size: 18px;
            color: #333;
            margin-bottom: 8px;
        }

        .upload-subtext {
            font-size: 14px;
            color: #666;
        }

        .button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 32px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 500;
            cursor: pointer;
            transition: transform 0.2s;
        }

        .button:hover {
            transform: translateY(-2px);
        }

        .button:active {
            transform: translateY(0);
        }

        .button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }

        .results {
            display: none;
        }

        .results.show {
            display: block;
        }

        .result-image {
            width: 100%;
            border-radius: 12px;
            margin-bottom: 24px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }

        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }

        .stat-card {
            background: #f8f9ff;
            border-left: 4px solid #667eea;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }

        .stat-value {
            font-size: 28px;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 4px;
        }

        .stat-label {
            font-size: 14px;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .detections-list {
            background: #f8f9ff;
            border-radius: 8px;
            padding: 20px;
            max-height: 400px;
            overflow-y: auto;
        }

        .detection-item {
            padding: 8px 0;
            font-size: 14px;
            color: #333;
            border-bottom: 1px solid #e2e8f0;
        }

        .detection-item:last-child {
            border-bottom: none;
        }

        .detection-confidence {
            float: right;
            background: #667eea;
            color: white;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 12px;
        }

        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }

        .loading.show {
            display: block;
        }

        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 12px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .reset-btn {
            background: #6366f1;
            margin-top: 20px;
            width: 100%;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Rice Bag Detection V9</h1>
            <p>YOLO11m Real-time Detection (fine-tuned dots)</p>
        </div>

        <div class="content">
            <div class="upload-section">
                <div class="upload-box" onclick="document.getElementById('fileInput').click()">
                    <div class="upload-icon">📸</div>
                    <div class="upload-text">Click to upload an image</div>
                    <div class="upload-subtext">or drag and drop</div>
                    <input type="file" id="fileInput" accept="image/*">
                </div>
            </div>

            <div class="loading" id="loading">
                <div class="spinner"></div>
                <div>Processing image...</div>
            </div>

            <div class="results" id="results">
                <img id="resultImage" class="result-image" src="" alt="Detection Result">

                <div class="stats">
                    <div class="stat-card">
                        <div class="stat-value" id="totalBags">0</div>
                        <div class="stat-label">Total Bags</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="avgConfidence">0.0</div>
                        <div class="stat-label">Avg Confidence</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="duplicatesRemoved">0</div>
                        <div class="stat-label">Duplicates Removed</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="totalWeight">0</div>
                        <div class="stat-label">Est. Weight (kg)</div>
                    </div>
                </div>

                <div class="detections-list" id="detectionsList"></div>

                <button class="button reset-btn" onclick="resetUI()">Detect Another Image</button>
            </div>
        </div>
    </div>

    <script>
        const fileInput = document.getElementById('fileInput');
        const uploadBox = document.querySelector('.upload-box');
        const loading = document.getElementById('loading');
        const results = document.getElementById('results');

        fileInput.addEventListener('change', handleFile);

        uploadBox.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadBox.style.borderColor = '#764ba2';
        });

        uploadBox.addEventListener('dragleave', () => {
            uploadBox.style.borderColor = '#667eea';
        });

        uploadBox.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadBox.style.borderColor = '#667eea';
            if (e.dataTransfer.files.length > 0) {
                fileInput.files = e.dataTransfer.files;
                handleFile();
            }
        });

        function handleFile() {
            if (fileInput.files.length === 0) return;

            const file = fileInput.files[0];
            if (!file.type.startsWith('image/')) {
                alert('Please select an image file');
                return;
            }

            const formData = new FormData();
            formData.append('file', file);

            loading.classList.add('show');
            results.classList.remove('show');

            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                loading.classList.remove('show');

                if (data.success) {
                    displayResults(data);
                } else {
                    alert('Error: ' + data.message);
                }
            })
            .catch(error => {
                loading.classList.remove('show');
                alert('Error: ' + error);
            });
        }

        function displayResults(data) {
            document.getElementById('resultImage').src = data.image_url;
            document.getElementById('totalBags').textContent = data.total_detections;
            document.getElementById('avgConfidence').textContent = data.avg_confidence.toFixed(3);
            document.getElementById('duplicatesRemoved').textContent = data.duplicates_removed;
            document.getElementById('totalWeight').textContent = (data.total_detections * 50).toLocaleString();

            let html = '';
            data.detections.forEach((det, i) => {
                html += `<div class="detection-item">Bag ${i+1}: <span class="detection-confidence">${det.toFixed(3)}</span></div>`;
            });
            document.getElementById('detectionsList').innerHTML = html;

            results.classList.add('show');
        }

        function resetUI() {
            fileInput.value = '';
            loading.classList.remove('show');
            results.classList.remove('show');
        }
    </script>
</body>
</html>
"""


# ============================================================
# ROUTES
# ============================================================

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return {"success": False, "message": "No file uploaded"}

        file = request.files["file"]
        if file.filename == "":
            return {"success": False, "message": "No file selected"}

        filename = f"{uuid.uuid4()}.jpg"
        filepath = UPLOAD_DIR / filename
        file.save(filepath)

        image = cv2.imread(str(filepath))
        if image is None:
            return {"success": False, "message": "Failed to load image"}

        results = model.predict(
            source=str(filepath),
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            max_det=MAX_DETECTIONS,
            device=DEVICE,
            verbose=False,
        )

        result = results[0]
        boxes = result.boxes

        annotated = image.copy()
        confidences = []
        box_count = 0
        raw_count = 0

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            raw_count = len(boxes)

            # --- Stage A: box-level dedup (IoU + containment), same as backup 3 ---
            xyxy, confs = deduplicate_boxes(xyxy, confs)

            # Build box dicts for Stage B
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

            # --- Stage B: center-point distance clustering (NEW in v3) ---
            final_boxes = cluster_center_points(stage_a_boxes)

            for b in final_boxes:
                cx, cy = int(b["cx"]), int(b["cy"])
                conf = b["conf"]

                confidences.append(conf)
                box_count += 1

                cv2.circle(annotated, (cx, cy), DOT_RADIUS, (0, 255, 0), -1)
                cv2.circle(annotated, (cx, cy), DOT_RADIUS + 2, (0, 0, 0), 1)

                label = f"{box_count} ({conf:.2f})"
                cv2.putText(
                    annotated, label,
                    (cx + DOT_RADIUS + 3, cy - DOT_RADIUS),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1
                )

        avg_conf = float(np.mean(confidences)) if confidences else 0.0
        duplicates_removed = raw_count - box_count
        cv2.putText(
            annotated,
            f"Total: {box_count} bags | Avg: {avg_conf:.3f} | Dupes removed: {duplicates_removed}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )

        output_filename = f"{uuid.uuid4()}_result.jpg"
        output_path = UPLOAD_DIR / output_filename
        cv2.imwrite(str(output_path), annotated)

        return {
            "success": True,
            "image_url": f"/uploads/{output_filename}",
            "total_detections": box_count,
            "raw_detections": raw_count,
            "duplicates_removed": duplicates_removed,
            "avg_confidence": avg_conf,
            "detections": confidences,
        }

    except Exception as e:
        print(f"Error: {e}")
        return {"success": False, "message": str(e)}


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("RICE BAG V9 - FLASK WEB UI (v3 fine-tuned dots)")
    print("=" * 70)
    print(f"\nModel: {MODEL_PATH}")
    print(f"Uploads: {UPLOAD_DIR}")
    print(f"\nStarting Flask server...")
    print(f"Listen on http://0.0.0.0:5000")
    print(f"\nOpen browser: http://localhost:5000")
    print("=" * 70 + "\n")

    app.run(host="0.0.0.0", port=5008, debug=False)

