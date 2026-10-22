"""
RICE BAG LOADING / UNLOADING VIDEO COUNTER (mask-based, tracker-driven)

New, fully independent Flask tool. Does NOT modify any other file in this
project -- it only loads the existing trained V11 segmentation model.

============================================================
PIPELINE
============================================================

    Video --> YOLO Segmentation (per frame) --> Individual instance MASKS
          --> Multi-Object Tracking (ByteTrack / BoT-SORT, ultralytics'
              built-in `model.track()`) --> stable Track ID per physical bag
          --> Mask-vs-Line geometry (signed side, % mask area per side)
          --> Direction check --> Temporal state machine (debounce +
              cooldown) --> Count

============================================================
WHY BYTETRACK (default) OVER BOT-SORT -- read before changing TRACKER_TYPE
============================================================

Both are supported out of the box via ultralytics' `model.track()` --
this file does not reimplement either; reinventing a tracker instead of
using a proven, actively-maintained one would be a worse engineering
choice, not a more "robust" one.

  BoT-SORT = ByteTrack's two-stage (high-conf then low-conf) association
  + a Kalman filter refinement + two OPTIONAL extras: camera-motion
  compensation (CMC) and an appearance/ReID embedding for re-matching
  lost tracks by what they look like.

  Neither optional extra earns its extra compute here:
    - CMC corrects for a MOVING camera. A warehouse doorway/dock camera
      is bolted down and static -- there is no camera motion to correct.
    - ReID re-matches a lost track using visual appearance. Every bag in
      a rice-bag pile looks nearly identical (same sack, same print,
      same size) -- an appearance embedding has very little discriminative
      signal here and mostly adds GPU cost without adding accuracy. For a
      genuinely distinctive cargo (mixed boxes/pallets/people) BoT-SORT's
      ReID would be the better default; for visually-uniform sacks it
      is not.

  What actually matters for this footage is ByteTrack's core idea: don't
  throw away low-confidence detections. A bag that's briefly half-hidden
  behind a worker or another bag often drops below the normal confidence
  cutoff for a few frames -- ByteTrack still tries to associate those
  low-confidence boxes with existing tracks (instead of discarding them
  outright, which is what naive "high-confidence-only" trackers do), so
  a track survives partial occlusion instead of dying and reappearing as
  a new ID. That is exactly the failure mode this application cares most
  about (an ID switch right at the counting line = a duplicate or missed
  count), and it's what ByteTrack was built for.

  TRACKER_TYPE below is a plain config switch -- set it to "botsort.yaml"
  if your footage has camera shake/pan, or your cargo is visually
  distinctive enough for ReID to help.

============================================================
WHY MASK-AREA-PERCENTAGE CROSSING (not centroid/point crossing)
============================================================

A centroid is a single point -- it crosses a line at one exact instant,
which is fragile: mask/box jitter of just a few pixels right at that
instant can flip the "which side" answer back and forth, and a partially
occluded bag's centroid can jump when the visible mask shape changes.

Instead, every frame, for every tracked bag near a line, this file
computes what FRACTION of that bag's actual mask area sits on each side
of the line (a vectorized boolean test over the mask array, not a
per-pixel Python loop). A side only "counts" as the bag's current side
once that fraction clears SIDE_CONFIRM_FRACTION (default 70%) -- e.g.
"80% mask on Side A" is a confirmed Side A, but "52% on Side A" is
treated as still-ambiguous (genuinely straddling the line) and ignored.
That confirmed side then has to PERSIST for MIN_CONFIRM_FRAMES consecutive
frames before it's trusted as the track's settled side at all -- this is
the debounce that absorbs single-frame mask jitter. Only a fully
persisted Side A -> persisted Side B (or the reverse) transition, in the
direction configured for that line, is a valid crossing.

============================================================
DUPLICATE-COUNT PREVENTION
============================================================

  - Track ID + per-(track, line) state machine: a crossing only counts
    once the confirmed side actually flips (not on every frame the bag
    happens to be mostly past the line).
  - COOLDOWN_FRAMES after a counted crossing: absorbs a mask flickering
    back and forth right at the line from re-triggering the same physical
    crossing multiple times.
  - Spatial/temporal duplicate guard (recent_counts log): if the SAME
    line+direction was just counted within DEDUP_TIME_WINDOW_FRAMES and
    DEDUP_DIST_PX of a new crossing, the new one is suppressed. This is
    the safety net for the one failure mode a state machine alone can't
    catch -- the tracker itself assigning a brand-new ID to a bag that
    was just counted (an ID switch right at the line).
  - LINE_PROXIMITY_MARGIN_FACTOR: bags nowhere near a line skip the
    per-frame mask/line percentage math entirely (cheap AABB pre-filter)
    -- performance, not correctness, but keeps the real per-pixel work
    limited to bags that could plausibly be crossing.

Note on the single-image test endpoint (/api/test_image): it does NOT
need the box/mask duplicate-dedup that the other test UIs in this project
use, because that dedup exists to fix ONE image having two overlapping
detections of the same physical bag. In the video pipeline the tracker's
persistent ID already IS that de-duplication -- one physical bag keeps
one ID across frames by construction, so there is nothing left to dedup.
The test-image endpoint still borrows that same dedup logic (copied in,
this file changes nothing elsewhere) purely so a single still frame looks
clean while you're verifying the model/masks, not because the video
counter needs it.

============================================================
KNOWN PROTOTYPE-SCOPE LIMITATIONS (documented on purpose)
============================================================

  - Jobs run as an in-process background thread with in-memory state
    (JOBS dict). Fine for one operator running this locally; a real
    multi-user deployment should move this to a proper job queue
    (Celery/RQ) and persistent job storage.
  - Output video is written with OpenCV's VideoWriter. This build's
    OpenCV may or may not have an H.264 ("avc1") encoder available (a
    common licensing-driven omission in default OpenCV builds); this
    file tries "avc1" first and falls back to "mp4v" automatically. The
    "mp4v" fallback still writes a fully valid, downloadable .mp4, but
    some browsers won't play it inline (VLC and most desktop players
    will). The UI always offers a direct download link either way.

Run: python src/video_bag_counter_ui.py
Access: http://localhost:5008
"""

from pathlib import Path
import base64
import json
import statistics
import threading
import time
import uuid

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request, render_template_string, send_file, Response
from ultralytics import YOLO


# ============================================================
# CONFIGURATION -- everything tunable lives here
# ============================================================

# Set an explicit path to force a specific checkpoint, e.g.:
#   MODEL_PATH = r"runs/rice_bag_v11_segmentation/weights/best.pt"
# Leave as None to auto-discover the same trained V11 segmentation model
# the other UIs in this project use.
MODEL_PATH = None

CLASS_NAMES = {0: "Rice-bag-10", 1: "other", 2: "pile"}
BAG_CLASS_ID = 0  # only this class is tracked/counted -- "other"/"pile" are excluded entirely from the video pipeline

CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.50
MAX_DETECTIONS = 300
INFERENCE_IMG_SIZE = 640     # lower = much faster on CPU; raise back toward 960-1280 only if you have a CUDA GPU and need finer mask detail
RETINA_MASKS = False         # True recomputes full-res masks per instance (noticeably slower); False is plenty precise for the 70%-area-per-side test this file does

# CPU inference is the dominant cost here, not video decode/writing -- the
# single biggest speed lever is not running the model on every frame.
# FRAME_STRIDE=2 means "run detection+tracking on every 2nd raw video
# frame"; frames in between are written through with just the lines/HUD
# redrawn (no new detections that frame), so the OUTPUT video stays full
# length/fps -- only the actual model.track() calls are skipped. Raise it
# (3-5) for more speed if your bags move slowly frame-to-frame relative to
# the camera; keep it at 1 (no skipping) if a GPU is available.
FRAME_STRIDE = 2

# JPEG quality for the live in-progress preview stream (/api/live/<job_id>)
# -- this is a low-stakes visual "watch it happen" feed, not the saved
# output video, so a lower quality here costs nothing but bandwidth.
LIVE_PREVIEW_JPEG_QUALITY = 70

DEVICE = 0 if torch.cuda.is_available() else "cpu"

# "bytetrack.yaml" or "botsort.yaml" -- see the module docstring above for
# why ByteTrack is the default for visually-uniform rice bags on a static
# camera. Ultralytics ships both; no separate tracker package to install.
TRACKER_TYPE = "bytetrack.yaml"

# --- mask-vs-line crossing state machine (see module docstring) ---
SIDE_CONFIRM_FRACTION = 0.70
MIN_CONFIRM_FRAMES = 3
COOLDOWN_FRAMES = 20
DEDUP_TIME_WINDOW_FRAMES = 45
DEDUP_DIST_PX = 80
LINE_PROXIMITY_MARGIN_FACTOR = 3.0

# --- output video codec attempts, in order ---
OUTPUT_FOURCC_CANDIDATES = ["avc1", "mp4v"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"
VIDEO_COUNTER_DIR = RUNS_DIR / "video_bag_counter"
UPLOAD_DIR = VIDEO_COUNTER_DIR / "uploads"
OUTPUT_DIR = VIDEO_COUNTER_DIR / "outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ROLE_COLORS_BGR = {
    "unloading": (0, 0, 255),   # red
    "loading": (255, 130, 0),   # blue
}
ROLE_LABELS = {
    "unloading": "UNLOADING",
    "loading": "LOADING",
}


# ============================================================
# MODEL AUTO DISCOVERY + LOAD
# ============================================================

def find_model():
    if MODEL_PATH:
        p = Path(MODEL_PATH)
        if not p.exists():
            raise FileNotFoundError(f"Configured MODEL_PATH does not exist: {p}")
        return p

    best_models = [
        p for p in RUNS_DIR.rglob("best.pt")
        if "rice_bag_v11_segmentation" in str(p)
    ]
    if best_models:
        best_models.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return best_models[0]

    last_models = [
        p for p in RUNS_DIR.rglob("last.pt")
        if "rice_bag_v11_segmentation" in str(p)
    ]
    if last_models:
        last_models.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return last_models[0]

    raise FileNotFoundError(
        "No segmentation checkpoint found under runs/, and MODEL_PATH is not set."
    )


MODEL_PATH_RESOLVED = find_model()
print(f"\nLoading segmentation model: {MODEL_PATH_RESOLVED}")
model = YOLO(str(MODEL_PATH_RESOLVED))
print(f"Model loaded. Task: {model.task}")

if model.task != "segment":
    raise RuntimeError(
        f"Loaded model's task is '{model.task}', not 'segment' -- "
        "this tool requires an instance-segmentation model."
    )

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # videos are much larger than single images

VIDEOS = {}  # video_id -> {"path": Path, "width":, "height":, "fps":, "frame_count":}
JOBS = {}    # job_id -> {"status": "running"|"done"|"error", "progress": float, "counts": {...}, "message": str, "output_path": str|None}


# ============================================================
# SINGLE-IMAGE TEST PATH -- Stage A/B mask dedup copied in from
# test_rice_bag_v11_segmentation_ui.py (see module docstring for why
# this is only needed here, not in the video pipeline).
# ============================================================

DEDUP_MASK_IOU_THRESHOLD = 0.45
DEDUP_MASK_CONTAINMENT_THRESHOLD = 0.70
CENTER_DIST_FACTOR = 0.45


def _mask_iou_and_containment(mask_a, mask_b):
    inter = np.logical_and(mask_a, mask_b).sum()
    if inter == 0:
        return 0.0, 0.0
    area_a = mask_a.sum()
    area_b = mask_b.sum()
    union = area_a + area_b - inter
    iou = inter / union if union > 0 else 0.0
    containment = inter / max(1, min(area_a, area_b))
    return float(iou), float(containment)


def deduplicate_instances(low_res_masks, boxes_xyxy, confs, cls_ids,
                           iou_thresh=DEDUP_MASK_IOU_THRESHOLD,
                           containment_thresh=DEDUP_MASK_CONTAINMENT_THRESHOLD):
    n = len(low_res_masks)
    if n == 0:
        return []

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
        bi = boxes_xyxy[i]
        for j in range(i + 1, n):
            if cls_ids[i] != cls_ids[j]:
                continue
            bj = boxes_xyxy[j]
            if bi[2] <= bj[0] or bj[2] <= bi[0] or bi[3] <= bj[1] or bj[3] <= bi[1]:
                continue
            iou, containment = _mask_iou_and_containment(low_res_masks[i], low_res_masks[j])
            if iou >= iou_thresh or containment >= containment_thresh:
                union(i, j)

    clusters = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    keep = [max(members, key=lambda idx: confs[idx]) for members in clusters.values()]
    return sorted(keep)


def cluster_center_points(centers, sizes, cls_ids, confs, factor=CENTER_DIST_FACTOR):
    n = len(centers)
    if n == 0:
        return []

    typical_size = statistics.median(sizes) if sizes else 0.0
    threshold = typical_size * factor

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
            if cls_ids[i] != cls_ids[j]:
                continue
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            if (dx * dx + dy * dy) ** 0.5 <= threshold:
                union(i, j)

    clusters = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    keep_local = [max(members, key=lambda idx: confs[idx]) for members in clusters.values()]
    return sorted(keep_local)


def annotate_test_image(image_bgr):
    """Runs plain (non-tracked) segmentation + Stage A/B dedup on one
    image, for verifying the model/masks before trusting the video
    pipeline. Returns (annotated_bgr, per_class_counts, detections)."""

    h, w = image_bgr.shape[:2]

    results = model.predict(
        source=image_bgr, imgsz=INFERENCE_IMG_SIZE, conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD, max_det=MAX_DETECTIONS, device=DEVICE,
        retina_masks=RETINA_MASKS, verbose=False,
    )
    result = results[0]
    output = image_bgr.copy()

    if result.boxes is not None:
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    else:
        classes = np.array([], dtype=int)
        confidences = np.array([])
        boxes_xyxy = np.zeros((0, 4))

    masks = result.masks.data.cpu().numpy() if result.masks is not None else None

    if len(classes) > 0 and masks is not None:
        low_res_masks_bool = masks > 0.5
        stage_a = deduplicate_instances(low_res_masks_bool, boxes_xyxy, confidences, classes)

        mh, mw = low_res_masks_bool.shape[1:]
        centers, sizes = [], []
        for idx in stage_a:
            ys, xs = np.where(low_res_masks_bool[idx])
            if len(xs) == 0:
                bx = boxes_xyxy[idx]
                cx_full, cy_full = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
            else:
                cx_full = float(np.mean(xs)) * (w / mw)
                cy_full = float(np.mean(ys)) * (h / mh)
            centers.append((cx_full, cy_full))
            bx = boxes_xyxy[idx]
            box_area = max(1.0, (bx[2] - bx[0]) * (bx[3] - bx[1]))
            sizes.append(float(np.sqrt(box_area)))

        stage_b_local = cluster_center_points(
            centers, sizes,
            [classes[idx] for idx in stage_a], [confidences[idx] for idx in stage_a],
        )
        final_indices = sorted(stage_a[k] for k in stage_b_local)
    else:
        final_indices = list(range(len(classes)))

    overlay_accum = np.zeros_like(output)
    draw_items = []
    per_class_counts = {"Rice-bag-10": 0, "other": 0, "pile": 0}
    detections = []
    bag_number = 0

    for i in final_indices:
        cls_id = int(classes[i])
        conf = float(confidences[i])
        per_class_counts[CLASS_NAMES.get(cls_id, "other")] += 1

        if masks is None or i >= len(masks):
            continue

        mask_full = cv2.resize(masks[i], (w, h), interpolation=cv2.INTER_NEAREST)
        binary_mask = (mask_full > 0.5).astype(np.uint8)
        mask_bool = binary_mask.astype(bool)

        if cls_id == 0:
            overlay_accum[:, :, 1][mask_bool] = 255
        elif cls_id == 1:
            overlay_accum[:, :, 2][mask_bool] = 255
        else:
            overlay_accum[:, :, 0][mask_bool] = 255

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        M = cv2.moments(binary_mask)
        if M["m00"] != 0:
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        else:
            ys, xs = np.where(binary_mask > 0)
            if len(xs) == 0:
                continue
            cx, cy = int(np.mean(xs)), int(np.mean(ys))

        if cls_id == 0:
            bag_number += 1
            label = f"Bag {bag_number} {conf:.2f}"
            detections.append({"id": bag_number, "class_name": "Rice-bag-10", "confidence": f"{conf*100:.1f}%"})
        else:
            label = f"{CLASS_NAMES.get(cls_id, 'Unknown')} {conf:.2f}"
            detections.append({"id": i + 1, "class_name": CLASS_NAMES.get(cls_id, "Unknown"), "confidence": f"{conf*100:.1f}%"})

        draw_items.append((contours, cx, cy, label))

    output = cv2.addWeighted(output, 0.72, overlay_accum, 0.28, 0)
    for contours, cx, cy, label in draw_items:
        cv2.drawContours(output, contours, -1, (255, 255, 255), 2)
        cv2.circle(output, (cx, cy), 6, (255, 255, 255), -1)
        cv2.circle(output, (cx, cy), 9, (0, 0, 0), 2)
        cv2.putText(output, label, (cx + 10, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    return output, per_class_counts, detections


# ============================================================
# LINE / MASK GEOMETRY
# ============================================================

def compute_side_grid(shape_hw, p1, p2):
    """Boolean grid, True where a pixel is on the "positive" side of the
    directed line p1->p2 (2D cross-product sign). Computed ONCE per line
    per video (the line never moves), then reused every frame -- the
    per-frame cost is just a boolean AND + sum against this cached grid.
    """
    h, w = shape_hw
    ys, xs = np.mgrid[0:h, 0:w]
    x1, y1 = p1
    x2, y2 = p2
    cross = (x2 - x1) * (ys - y1) - (y2 - y1) * (xs - x1)
    return cross > 0


def near_line(box_xyxy, line, margin_factor):
    """Cheap AABB pre-filter: skip the real mask/line percentage math for
    bags whose box isn't even within a few bag-widths of this line."""
    x1, y1, x2, y2 = box_xyxy
    bag_w = max(1.0, x2 - x1)
    margin = bag_w * margin_factor
    lx1, ly1 = line["p1"]
    lx2, ly2 = line["p2"]
    line_min_x, line_max_x = min(lx1, lx2) - margin, max(lx1, lx2) + margin
    line_min_y, line_max_y = min(ly1, ly2) - margin, max(ly1, ly2) + margin
    return not (x2 < line_min_x or x1 > line_max_x or y2 < line_min_y or y1 > line_max_y)


def draw_lines_and_hud(frame, lines_config, counts):
    for line in lines_config:
        p1 = (int(line["p1"][0]), int(line["p1"][1]))
        p2 = (int(line["p2"][0]), int(line["p2"][1]))
        color = line["color"]

        cv2.line(frame, p1, p2, color, 3)
        cv2.circle(frame, p1, 6, color, -1)
        cv2.circle(frame, p2, 6, color, -1)

        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        length = max(1.0, (dx ** 2 + dy ** 2) ** 0.5)
        nx, ny = -dy / length, dx / length  # points toward the cross>0 ("positive") side
        if line["direction"] == "pos_to_neg":
            nx, ny = -nx, -ny
        tip = (int(mx + nx * 40), int(my + ny * 40))
        cv2.arrowedLine(frame, (int(mx), int(my)), tip, color, 3, tipLength=0.4)

        cv2.putText(frame, line["label"], (p1[0] + 8, p1[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    hud = [
        (f"UNLOADING: {counts.get('unloading', 0)}", ROLE_COLORS_BGR["unloading"]),
        (f"LOADING: {counts.get('loading', 0)}", ROLE_COLORS_BGR["loading"]),
        (f"TOTAL: {counts.get('unloading', 0) + counts.get('loading', 0)}", (255, 255, 255)),
    ]
    y = 34
    for text, color in hud:
        cv2.putText(frame, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(frame, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
        y += 34


def open_video_writer(path, fourcc_candidates, fps, size):
    for tag in fourcc_candidates:
        fourcc = cv2.VideoWriter_fourcc(*tag)
        writer = cv2.VideoWriter(str(path), fourcc, fps, size)
        if writer.isOpened():
            return writer, tag
        writer.release()
    return None, None


# ============================================================
# VIDEO PROCESSING (runs in a background thread)
# ============================================================

def process_video(job_id, video_path, lines_config):
    """Manually reads frames via cv2.VideoCapture (instead of handing the
    whole video path to model.track(stream=True)) so this function can:
      1. skip full inference on most frames (FRAME_STRIDE) while still
         writing every raw frame to the output video, and
      2. publish the latest annotated frame into job["latest_frame_jpeg"]
         every frame, for the live MJPEG preview at /api/live/<job_id>.

    Calling model.track(source=<single frame>, persist=True, ...)
    repeatedly on the SAME model object is ultralytics' own supported
    pattern for custom per-frame streaming loops -- tracker state carries
    over between calls exactly like it would with stream=True on a video
    path, as long as persist=True is passed every time.
    """
    job = JOBS[job_id]

    try:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if frame_w == 0 or frame_h == 0:
            cap.release()
            raise RuntimeError("Could not read video dimensions -- is the file a valid video?")

        output_path = OUTPUT_DIR / f"{job_id}.mp4"
        writer, used_codec = open_video_writer(output_path, OUTPUT_FOURCC_CANDIDATES, fps, (frame_w, frame_h))
        if writer is None:
            cap.release()
            raise RuntimeError("Could not open a video writer with any available codec.")
        job["codec"] = used_codec

        counts = {"unloading": 0, "loading": 0}
        side_grids = None
        track_states = {}
        recent_counts = []

        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            run_inference = (frame_idx % FRAME_STRIDE == 0)

            if run_inference:
                results = model.track(
                    source=frame,
                    tracker=TRACKER_TYPE,
                    persist=True,
                    conf=CONF_THRESHOLD,
                    iou=IOU_THRESHOLD,
                    max_det=MAX_DETECTIONS,
                    imgsz=INFERENCE_IMG_SIZE,
                    classes=[BAG_CLASS_ID],
                    device=DEVICE,
                    retina_masks=RETINA_MASKS,
                    verbose=False,
                )
                result = results[0]
                boxes = result.boxes
                masks_obj = result.masks

                if boxes is not None and boxes.id is not None and masks_obj is not None and len(boxes) > 0:
                    track_ids = boxes.id.cpu().numpy().astype(int)
                    confs = boxes.conf.cpu().numpy()
                    xyxy = boxes.xyxy.cpu().numpy()
                    masks_low = masks_obj.data.cpu().numpy()  # (n, mh, mw) -- cheap working resolution

                    if side_grids is None:
                        mh, mw = masks_low.shape[1:]
                        sx, sy = mw / frame_w, mh / frame_h
                        side_grids = []
                        for line in lines_config:
                            p1m = (line["p1"][0] * sx, line["p1"][1] * sy)
                            p2m = (line["p2"][0] * sx, line["p2"][1] * sy)
                            side_grids.append(compute_side_grid((mh, mw), p1m, p2m))

                    overlay_accum = np.zeros_like(frame)
                    draw_labels = []

                    for i in range(len(track_ids)):
                        tid = int(track_ids[i])
                        conf = float(confs[i])
                        box = xyxy[i]
                        mask_bool = masks_low[i] > 0.5
                        mask_area = int(mask_bool.sum())
                        if mask_area == 0:
                            continue

                        mask_full = cv2.resize(masks_low[i], (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
                        overlay_accum[:, :, 1][mask_full > 0.5] = 255

                        cx = float((box[0] + box[2]) / 2)
                        cy = float((box[1] + box[3]) / 2)

                        for line_idx, line in enumerate(lines_config):
                            if not near_line(box, line, LINE_PROXIMITY_MARGIN_FACTOR):
                                continue

                            frac_pos = float(np.count_nonzero(mask_bool & side_grids[line_idx])) / mask_area
                            if frac_pos >= SIDE_CONFIRM_FRACTION:
                                instant_side = "pos"
                            elif frac_pos <= (1 - SIDE_CONFIRM_FRACTION):
                                instant_side = "neg"
                            else:
                                instant_side = None

                            key = (tid, line_idx)
                            st = track_states.setdefault(key, {
                                "confirmed_side": None, "pending_side": None,
                                "pending_count": 0, "cooldown": 0,
                            })

                            if st["cooldown"] > 0:
                                st["cooldown"] -= 1

                            if instant_side is None:
                                st["pending_side"] = None
                                st["pending_count"] = 0
                                continue

                            if instant_side == st["pending_side"]:
                                st["pending_count"] += 1
                            else:
                                st["pending_side"] = instant_side
                                st["pending_count"] = 1

                            if st["pending_count"] < MIN_CONFIRM_FRAMES:
                                continue

                            if st["confirmed_side"] is None:
                                st["confirmed_side"] = instant_side
                                continue

                            if instant_side != st["confirmed_side"]:
                                if st["cooldown"] == 0:
                                    transition = f"{st['confirmed_side']}_to_{instant_side}"
                                    if transition == line["direction"]:
                                        is_dup = any(
                                            rc["line_idx"] == line_idx
                                            and rc["direction"] == transition
                                            and (frame_idx - rc["frame"]) <= DEDUP_TIME_WINDOW_FRAMES
                                            and ((rc["x"] - cx) ** 2 + (rc["y"] - cy) ** 2) ** 0.5 <= DEDUP_DIST_PX
                                            for rc in recent_counts
                                        )
                                        if not is_dup:
                                            counts[line["role"]] += 1
                                            recent_counts.append({
                                                "line_idx": line_idx, "direction": transition,
                                                "x": cx, "y": cy, "frame": frame_idx,
                                            })
                                            draw_labels.append((cx, cy, "+1", line["color"]))
                                    st["cooldown"] = COOLDOWN_FRAMES
                                st["confirmed_side"] = instant_side

                        draw_labels.append((cx, cy, f"ID {tid} {conf:.2f}", (255, 255, 255)))

                    recent_counts = [rc for rc in recent_counts if (frame_idx - rc["frame"]) <= DEDUP_TIME_WINDOW_FRAMES]

                    frame = cv2.addWeighted(frame, 0.78, overlay_accum, 0.22, 0)
                    for cx, cy, text, color in draw_labels:
                        cv2.putText(frame, text, (int(cx) + 6, int(cy) - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

            draw_lines_and_hud(frame, lines_config, counts)
            writer.write(frame)

            # live preview -- updated every frame (cheap JPEG encode) so
            # the /api/live stream feels smooth even though full detection
            # only actually runs every FRAME_STRIDE frames
            ok_enc, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, LIVE_PREVIEW_JPEG_QUALITY])
            if ok_enc:
                job["latest_frame_jpeg"] = buf.tobytes()

            frame_idx += 1
            if total_frames > 0:
                job["progress"] = round(min(99.0, frame_idx / total_frames * 100), 1)
            job["counts"] = dict(counts)

        cap.release()
        writer.release()
        job["status"] = "done"
        job["progress"] = 100.0
        job["output_path"] = str(output_path)
        job["counts"] = dict(counts)

    except Exception as exc:
        job["status"] = "error"
        job["message"] = f"{type(exc).__name__}: {exc}"


# ============================================================
# HTML
# ============================================================

HTML = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Rice Bag AI Vision | Loading/Unloading Counter</title>

<style>

* { box-sizing: border-box; }

body {
    margin: 0;
    background: #0b1120;
    color: #f8fafc;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

button, input, select { font-family: inherit; }

button {
    border: 0;
    padding: 11px 16px;
    border-radius: 10px;
    cursor: pointer;
    font-weight: 700;
    font-size: 13px;
    background: #2563eb;
    color: white;
}

button.secondary { background: #1e293b; color: #cbd5e1; }
button.danger { background: #7f1d1d; color: #fecaca; }
button:disabled { opacity: 0.4; cursor: not-allowed; }

.header {
    background: #111827;
    border-bottom: 1px solid #1f2937;
    padding: 18px 30px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 20px;
}

.brand { display: flex; align-items: center; gap: 14px; }

.logo {
    width: 46px; height: 46px; border-radius: 12px; background: #2563eb;
    display: flex; align-items: center; justify-content: center; font-size: 23px;
}

.brand h1 { margin: 0; font-size: 21px; }
.brand p { margin: 3px 0 0; color: #94a3b8; font-size: 13px; }

.tabs { display: flex; gap: 8px; }
.tab-btn { background: #1e293b; color: #94a3b8; }
.tab-btn.active { background: #2563eb; color: white; }

.container { max-width: 1500px; margin: auto; padding: 25px 30px 40px; }

.panel {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 16px;
    overflow: hidden;
    margin-bottom: 20px;
}

.panel-title { padding: 16px 18px; border-bottom: 1px solid #1f2937; font-size: 15px; font-weight: 700; }
.panel-body { padding: 18px; }

.step { display: none; }
.step.active { display: block; }

.drop-zone {
    border: 2px dashed #334155;
    border-radius: 14px;
    padding: 35px 20px;
    text-align: center;
    cursor: pointer;
    transition: 0.2s;
}
.drop-zone:hover { border-color: #60a5fa; background: #0f172a; }
.drop-zone.dragover { border-color: #60a5fa; background: #172554; }
.upload-icon { font-size: 36px; margin-bottom: 8px; }
.file-name { color: #60a5fa; font-size: 13px; margin-top: 10px; }
input[type=file] { display: none; }

.instructions {
    background: #0b1220;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 14px;
    line-height: 1.6;
    margin-bottom: 16px;
    font-size: 13px;
    color: #cbd5e1;
}

.canvas-wrapper {
    background: #080d18;
    border-radius: 10px;
    padding: 10px;
    text-align: center;
    overflow: auto;
}
canvas { max-width: 100%; cursor: crosshair; border-radius: 8px; }

.line-cards { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 16px; }

.line-card {
    background: #0b1220;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 14px;
}
.line-card.ready { border-color: #22c55e; }
.line-card h3 { margin: 0 0 8px; font-size: 14px; display: flex; align-items: center; gap: 8px; }
.swatch { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }
.line-status { font-size: 12px; color: #94a3b8; margin-bottom: 8px; }

select {
    width: 100%;
    padding: 8px;
    border-radius: 8px;
    border: 1px solid #334155;
    background: #07111b;
    color: #f8fafc;
    margin-top: 6px;
}

.action-row { display: flex; gap: 10px; margin-top: 18px; flex-wrap: wrap; }

.progress-wrap { background: #0b1220; border-radius: 10px; height: 14px; overflow: hidden; margin: 14px 0; }
.progress-bar { height: 100%; background: #2563eb; width: 0%; transition: width 0.3s; }

.kpis { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-bottom: 20px; }
.kpi { background: #111827; border: 1px solid #1f2937; border-radius: 14px; padding: 18px; text-align: center; }
.kpi-top { color: #94a3b8; font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; }
.kpi-number { font-size: 30px; font-weight: 800; margin-top: 8px; }

video { width: 100%; border-radius: 10px; background: #000; }

.status-banner {
    margin-top: 14px; padding: 12px 14px; border-radius: 10px;
    background: #0b1220; border: 1px solid #1e293b; color: #cbd5e1; font-size: 13px;
}

.image-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.image-container { padding: 15px; background: #080d18; text-align: center; border-radius: 10px; }
.image-container img { width: 100%; border-radius: 8px; }

.table-wrapper { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { text-align: left; color: #64748b; font-weight: 600; padding: 10px 14px; border-bottom: 1px solid #1f2937; }
td { padding: 10px 14px; border-bottom: 1px solid #1e293b; }

.error {
    background: #450a0a; border: 1px solid #7f1d1d; color: #fecaca;
    border-radius: 12px; padding: 15px; margin-bottom: 20px; white-space: pre-wrap;
}

.footer { text-align: center; color: #475569; font-size: 12px; padding-top: 25px; }

@media(max-width: 900px) {
    .line-cards, .image-grid, .kpis { grid-template-columns: 1fr; }
}

</style>
</head>

<body>

<div class="header">
    <div class="brand">
        <div class="logo">🚚</div>
        <div>
            <h1>Rice Bag AI Vision</h1>
            <p>Loading / Unloading Video Counter &middot; V11 Segmentation + Tracking</p>
        </div>
    </div>
    <div class="tabs">
        <button class="tab-btn active" id="tabVideoBtn" onclick="switchTab('video')">Video Counter</button>
        <button class="tab-btn" id="tabImageBtn" onclick="switchTab('image')">Test Single Image</button>
    </div>
</div>

<div class="container">

{% if error %}
<div class="error">{{ error }}</div>
{% endif %}

<!-- ============================================================
     VIDEO COUNTER TAB
     ============================================================ -->
<div id="tabVideo">

<!-- STEP 1: UPLOAD -->
<div class="step active" id="stepUpload">
    <div class="panel">
        <div class="panel-title">Step 1 &middot; Upload Video</div>
        <div class="panel-body">
            <div class="drop-zone" id="videoDropZone">
                <div class="upload-icon">🎬</div>
                <h3>Upload Warehouse Video</h3>
                <p style="color:#94a3b8;">Drag & drop a video here or click to browse</p>
                <strong>Select Video</strong>
                <div class="file-name" id="videoFileName"></div>
            </div>
            <input type="file" id="videoInput" accept="video/*">
            <div class="action-row">
                <button id="uploadBtn" disabled onclick="uploadVideo()">Upload & Continue</button>
            </div>
            <div id="uploadStatus" class="status-banner">Choose a video to begin.</div>
        </div>
    </div>
</div>

<!-- STEP 2: DRAW LINES -->
<div class="step" id="stepDraw">
    <div class="panel">
        <div class="panel-title">Step 2 &middot; Draw the Unloading (Red) and Loading (Blue) lines</div>
        <div class="panel-body">
            <div class="instructions" id="drawInstructions">
                Click <b>2 points</b> on the frame below to draw the <b style="color:#ff4d4d;">RED Unloading line</b> first.
                A bag only counts once its mask has clearly moved from one persisted side of a line to the other, in the
                direction you choose &mdash; not just for touching the line.
            </div>

            <div class="canvas-wrapper">
                <canvas id="drawCanvas"></canvas>
            </div>

            <div class="line-cards">

                <div class="line-card" id="unloadingCard">
                    <h3><span class="swatch" style="background:#ff4d4d;"></span>Unloading Line (Red)</h3>
                    <div class="line-status" id="unloadingStatus">Not drawn yet.</div>
                    <select id="unloadingDirection" disabled></select>
                    <div class="action-row">
                        <button class="secondary" onclick="redoLine('unloading')">Redo This Line</button>
                    </div>
                </div>

                <div class="line-card" id="loadingCard">
                    <h3><span class="swatch" style="background:#3b82f6;"></span>Loading Line (Blue)</h3>
                    <div class="line-status" id="loadingStatus">Draw the Unloading line first.</div>
                    <select id="loadingDirection" disabled></select>
                    <div class="action-row">
                        <button class="secondary" onclick="redoLine('loading')">Redo This Line</button>
                    </div>
                </div>

            </div>

            <div class="action-row">
                <button class="secondary" onclick="goToStep('stepUpload')">Back</button>
                <button id="startBtn" disabled onclick="startProcessing()">🚀 Start Processing</button>
            </div>
        </div>
    </div>
</div>

<!-- STEP 3: PROCESSING -->
<div class="step" id="stepProcessing">
    <div class="panel">
        <div class="panel-title">Step 3 &middot; Processing (live preview)</div>
        <div class="panel-body">
            <div class="canvas-wrapper" style="margin-bottom:14px;">
                <img id="liveStream" style="max-width:100%;border-radius:10px;display:block;margin:auto;">
            </div>

            <div class="progress-wrap"><div class="progress-bar" id="progressBar"></div></div>
            <div id="processingStatus" class="status-banner">
                Starting... CPU inference is the slow part here, not the video itself -- the live view above shows
                exactly what's been processed so far, it isn't stuck even if the bar moves slowly.
            </div>

            <div class="kpis" style="margin-top:18px;">
                <div class="kpi">
                    <div class="kpi-top">🔴 Unloading</div>
                    <div class="kpi-number" id="liveUnloading">0</div>
                </div>
                <div class="kpi">
                    <div class="kpi-top">🔵 Loading</div>
                    <div class="kpi-number" id="liveLoading">0</div>
                </div>
                <div class="kpi">
                    <div class="kpi-top">Total</div>
                    <div class="kpi-number" id="liveTotal">0</div>
                </div>
            </div>
        </div>
    </div>
</div>

<!-- STEP 4: RESULT -->
<div class="step" id="stepResult">
    <div class="kpis">
        <div class="kpi">
            <div class="kpi-top">🔴 Unloading</div>
            <div class="kpi-number" id="finalUnloading">0</div>
        </div>
        <div class="kpi">
            <div class="kpi-top">🔵 Loading</div>
            <div class="kpi-number" id="finalLoading">0</div>
        </div>
        <div class="kpi">
            <div class="kpi-top">Total</div>
            <div class="kpi-number" id="finalTotal">0</div>
        </div>
    </div>

    <div class="panel">
        <div class="panel-title">Annotated Result Video</div>
        <div class="panel-body">
            <video id="resultVideo" controls></video>
            <div class="action-row">
                <a id="downloadLink" href="#" download>
                    <button class="secondary">⬇ Download Video</button>
                </a>
                <button onclick="resetVideoFlow()">Process Another Video</button>
            </div>
            <div class="status-banner">
                If the preview above doesn't play, your OpenCV build's codec still produced a valid, downloadable .mp4 &mdash;
                open it with VLC or any desktop player.
            </div>
        </div>
    </div>
</div>

</div>


<!-- ============================================================
     SINGLE IMAGE TEST TAB
     ============================================================ -->
<div id="tabImage" style="display:none;">

    <div class="panel">
        <div class="panel-title">Verify Segmentation on a Single Image</div>
        <div class="panel-body">
            <div class="drop-zone" id="imageDropZone">
                <div class="upload-icon">📷</div>
                <h3>Upload a Warehouse Photo</h3>
                <p style="color:#94a3b8;">Drag & drop an image here or click to browse</p>
                <strong>Select Image</strong>
                <div class="file-name" id="imageFileName"></div>
            </div>
            <input type="file" id="imageInput" accept="image/*">
            <div class="action-row">
                <button id="testImageBtn" disabled onclick="runImageTest()">Run Segmentation Test</button>
            </div>
            <div id="imageStatus" class="status-banner">Choose an image to begin.</div>
        </div>
    </div>

    <div class="kpis" id="imageKpis" style="display:none;">
        <div class="kpi"><div class="kpi-top">Rice Bags</div><div class="kpi-number" id="imgBagCount">0</div></div>
        <div class="kpi"><div class="kpi-top">Other</div><div class="kpi-number" id="imgOtherCount">0</div></div>
        <div class="kpi"><div class="kpi-top">Pile</div><div class="kpi-number" id="imgPileCount">0</div></div>
    </div>

    <div class="panel" id="imageResultPanel" style="display:none;">
        <div class="panel-title">Result</div>
        <div class="panel-body">
            <div class="image-grid">
                <div class="image-container"><img id="imgOriginal"></div>
                <div class="image-container"><img id="imgResult"></div>
            </div>

            <div class="table-wrapper" style="margin-top:18px;">
                <table>
                    <thead><tr><th>ID</th><th>Class</th><th>Confidence</th></tr></thead>
                    <tbody id="imgDetTable"></tbody>
                </table>
            </div>
        </div>
    </div>

</div>


<div class="footer">Rice Bag AI Vision &middot; Loading/Unloading Video Counter</div>

</div>


<script>

// ==========================================================
// TABS
// ==========================================================

function switchTab(tab) {
    document.getElementById("tabVideo").style.display = tab === "video" ? "block" : "none";
    document.getElementById("tabImage").style.display = tab === "image" ? "block" : "none";
    document.getElementById("tabVideoBtn").classList.toggle("active", tab === "video");
    document.getElementById("tabImageBtn").classList.toggle("active", tab === "image");
}

function goToStep(stepId) {
    document.querySelectorAll(".step").forEach(el => el.classList.remove("active"));
    document.getElementById(stepId).classList.add("active");
}


// ==========================================================
// VIDEO UPLOAD (Step 1)
// ==========================================================

const videoDropZone = document.getElementById("videoDropZone");
const videoInput = document.getElementById("videoInput");
const videoFileNameEl = document.getElementById("videoFileName");
const uploadBtn = document.getElementById("uploadBtn");

videoDropZone.addEventListener("click", () => videoInput.click());
videoDropZone.addEventListener("dragover", e => { e.preventDefault(); videoDropZone.classList.add("dragover"); });
videoDropZone.addEventListener("dragleave", () => videoDropZone.classList.remove("dragover"));
videoDropZone.addEventListener("drop", e => {
    e.preventDefault();
    videoDropZone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
        videoInput.files = e.dataTransfer.files;
        onVideoFileChosen();
    }
});
videoInput.addEventListener("change", onVideoFileChosen);

function onVideoFileChosen() {
    const file = videoInput.files[0];
    if (!file) return;
    videoFileNameEl.textContent = "Selected: " + file.name;
    uploadBtn.disabled = false;
}

let videoId = null;
let frameWidth = 0, frameHeight = 0;

async function uploadVideo() {
    const file = videoInput.files[0];
    if (!file) return;

    uploadBtn.disabled = true;
    document.getElementById("uploadStatus").innerText = "Uploading & extracting first frame...";

    const form = new FormData();
    form.append("video", file);

    try {
        const resp = await fetch("/api/upload", { method: "POST", body: form });
        const data = await resp.json();

        if (!data.success) {
            document.getElementById("uploadStatus").innerText = data.message || "Upload failed.";
            uploadBtn.disabled = false;
            return;
        }

        videoId = data.video_id;
        frameWidth = data.width;
        frameHeight = data.height;

        loadFrameIntoCanvas(data.first_frame_base64);
        resetLineDrawingState();
        goToStep("stepDraw");

    } catch (err) {
        document.getElementById("uploadStatus").innerText = "Upload error: " + err.message;
    }

    uploadBtn.disabled = false;
}


// ==========================================================
// LINE DRAWING (Step 2)
// ==========================================================

const drawCanvas = document.getElementById("drawCanvas");
const drawCtx = drawCanvas.getContext("2d");
let frameImage = new Image();

const ROLE_COLORS = { unloading: "#ff4d4d", loading: "#3b82f6" };
const ROLE_ORDER = ["unloading", "loading"];

let currentRole = "unloading";
let pendingPoints = [];
let linesData = { unloading: null, loading: null };

function loadFrameIntoCanvas(base64) {
    frameImage = new Image();
    frameImage.onload = function () {
        drawCanvas.width = frameImage.naturalWidth;
        drawCanvas.height = frameImage.naturalHeight;
        redrawCanvas();
    };
    frameImage.src = "data:image/jpeg;base64," + base64;
}

function resetLineDrawingState() {
    currentRole = "unloading";
    pendingPoints = [];
    linesData = { unloading: null, loading: null };

    document.getElementById("unloadingStatus").innerText = "Click 2 points on the frame.";
    document.getElementById("loadingStatus").innerText = "Draw the Unloading line first.";
    document.getElementById("unloadingDirection").disabled = true;
    document.getElementById("unloadingDirection").innerHTML = "";
    document.getElementById("loadingDirection").disabled = true;
    document.getElementById("loadingDirection").innerHTML = "";
    document.getElementById("unloadingCard").classList.remove("ready");
    document.getElementById("loadingCard").classList.remove("ready");
    document.getElementById("startBtn").disabled = true;
    document.getElementById("drawInstructions").innerHTML =
        'Click <b>2 points</b> on the frame below to draw the <b style="color:#ff4d4d;">RED Unloading line</b> first. ' +
        "A bag only counts once its mask has clearly moved from one persisted side of a line to the other, in the " +
        "direction you choose &mdash; not just for touching the line.";
}

function redoLine(role) {
    linesData[role] = null;
    if (role === "unloading") {
        linesData.loading = null; // redrawing the first line invalidates the second too (order matters for the flow)
        document.getElementById("loadingStatus").innerText = "Draw the Unloading line first.";
        document.getElementById("loadingDirection").disabled = true;
        document.getElementById("loadingDirection").innerHTML = "";
        document.getElementById("loadingCard").classList.remove("ready");
    }
    document.getElementById(role + "Status").innerText = "Click 2 points on the frame.";
    document.getElementById(role + "Direction").disabled = true;
    document.getElementById(role + "Direction").innerHTML = "";
    document.getElementById(role + "Card").classList.remove("ready");
    currentRole = role;
    pendingPoints = [];
    document.getElementById("startBtn").disabled = true;
    redrawCanvas();
}

drawCanvas.addEventListener("click", function (event) {
    if (!frameImage.src) return;
    if (linesData[currentRole]) return; // this line already complete -- use "Redo This Line" to change it

    const rect = drawCanvas.getBoundingClientRect();
    const scaleX = drawCanvas.width / rect.width;
    const scaleY = drawCanvas.height / rect.height;
    const x = (event.clientX - rect.left) * scaleX;
    const y = (event.clientY - rect.top) * scaleY;

    pendingPoints.push({ x: x, y: y });
    redrawCanvas();

    if (pendingPoints.length === 2) {
        finishLine(currentRole, pendingPoints[0], pendingPoints[1]);
        pendingPoints = [];
    }
});

function crossSign(p1, p2, testPoint) {
    const val = (p2.x - p1.x) * (testPoint.y - p1.y) - (p2.y - p1.y) * (testPoint.x - p1.x);
    return val > 0 ? "pos" : "neg";
}

function buildDirectionOptions(p1, p2) {
    const dx = Math.abs(p2.x - p1.x);
    const dy = Math.abs(p2.y - p1.y);
    const midY = (p1.y + p2.y) / 2;
    const midX = (p1.x + p2.x) / 2;

    let labelForPos, labelForNeg;

    if (dx >= dy) {
        const rightSide = crossSign(p1, p2, { x: drawCanvas.width, y: midY });
        labelForPos = rightSide === "pos" ? "Left → Right" : "Right → Left";
        labelForNeg = rightSide === "pos" ? "Right → Left" : "Left → Right";
    } else {
        const bottomSide = crossSign(p1, p2, { x: midX, y: drawCanvas.height });
        labelForPos = bottomSide === "pos" ? "Top → Bottom" : "Bottom → Top";
        labelForNeg = bottomSide === "pos" ? "Bottom → Top" : "Top → Bottom";
    }

    return [
        { value: "neg_to_pos", label: labelForPos },
        { value: "pos_to_neg", label: labelForNeg },
    ];
}

function finishLine(role, p1, p2) {
    const options = buildDirectionOptions(p1, p2);
    linesData[role] = { p1: p1, p2: p2, direction: options[0].value };

    const select = document.getElementById(role + "Direction");
    select.innerHTML = "";
    options.forEach(opt => {
        const el = document.createElement("option");
        el.value = opt.value;
        el.textContent = "Counts as: " + opt.label;
        select.appendChild(el);
    });
    select.disabled = false;
    select.onchange = () => { linesData[role].direction = select.value; redrawCanvas(); };

    document.getElementById(role + "Status").innerText = "Line drawn. Choose the counted direction.";
    document.getElementById(role + "Card").classList.add("ready");

    if (role === "unloading") {
        currentRole = "loading";
        document.getElementById("loadingStatus").innerText = "Click 2 points on the frame.";
        document.getElementById("drawInstructions").innerHTML =
            'Now click <b>2 points</b> to draw the <b style="color:#3b82f6;">BLUE Loading line</b>.';
    } else {
        document.getElementById("drawInstructions").innerHTML =
            "Both lines are drawn. Confirm each direction, then start processing.";
    }

    if (linesData.unloading && linesData.loading) {
        document.getElementById("startBtn").disabled = false;
    }

    redrawCanvas();
}

function drawArrowPreview(p1, p2, direction, color) {
    const mx = (p1.x + p2.x) / 2, my = (p1.y + p2.y) / 2;
    const dx = p2.x - p1.x, dy = p2.y - p1.y;
    const len = Math.max(1, Math.hypot(dx, dy));
    let nx = -dy / len, ny = dx / len;
    if (direction === "pos_to_neg") { nx = -nx; ny = -ny; }

    drawCtx.strokeStyle = color;
    drawCtx.lineWidth = 3;
    drawCtx.beginPath();
    drawCtx.moveTo(mx, my);
    drawCtx.lineTo(mx + nx * 45, my + ny * 45);
    drawCtx.stroke();

    drawCtx.fillStyle = color;
    drawCtx.beginPath();
    drawCtx.arc(mx + nx * 45, my + ny * 45, 6, 0, Math.PI * 2);
    drawCtx.fill();
}

function redrawCanvas() {
    if (!frameImage.src) return;
    drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
    drawCtx.drawImage(frameImage, 0, 0, drawCanvas.width, drawCanvas.height);

    ROLE_ORDER.forEach(role => {
        const color = ROLE_COLORS[role];
        const line = linesData[role];

        if (line) {
            drawCtx.strokeStyle = color;
            drawCtx.lineWidth = 4;
            drawCtx.beginPath();
            drawCtx.moveTo(line.p1.x, line.p1.y);
            drawCtx.lineTo(line.p2.x, line.p2.y);
            drawCtx.stroke();

            [line.p1, line.p2].forEach(p => {
                drawCtx.fillStyle = color;
                drawCtx.beginPath();
                drawCtx.arc(p.x, p.y, 7, 0, Math.PI * 2);
                drawCtx.fill();
            });

            drawArrowPreview(line.p1, line.p2, line.direction, color);
        } else if (role === currentRole && pendingPoints.length === 1) {
            drawCtx.fillStyle = color;
            drawCtx.beginPath();
            drawCtx.arc(pendingPoints[0].x, pendingPoints[0].y, 7, 0, Math.PI * 2);
            drawCtx.fill();
        }
    });
}


// ==========================================================
// PROCESSING (Step 3) + POLLING
// ==========================================================

let jobId = null;
let pollTimer = null;

async function startProcessing() {
    if (!linesData.unloading || !linesData.loading) return;

    goToStep("stepProcessing");
    document.getElementById("progressBar").style.width = "0%";
    document.getElementById("processingStatus").innerText = "Submitting job...";

    const payload = {
        video_id: videoId,
        lines: [
            { role: "unloading", p1: [linesData.unloading.p1.x, linesData.unloading.p1.y],
              p2: [linesData.unloading.p2.x, linesData.unloading.p2.y], direction: linesData.unloading.direction },
            { role: "loading", p1: [linesData.loading.p1.x, linesData.loading.p1.y],
              p2: [linesData.loading.p2.x, linesData.loading.p2.y], direction: linesData.loading.direction },
        ],
    };

    try {
        const resp = await fetch("/api/start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        const data = await resp.json();

        if (!data.success) {
            document.getElementById("processingStatus").innerText = data.message || "Failed to start.";
            return;
        }

        jobId = data.job_id;
        document.getElementById("liveStream").src = "/api/live/" + jobId;
        pollTimer = setInterval(pollStatus, 1000);

    } catch (err) {
        document.getElementById("processingStatus").innerText = "Start error: " + err.message;
    }
}

function stopLiveStream() {
    // Browsers keep an MJPEG multipart/x-mixed-replace request open
    // indefinitely -- clearing src is what actually closes the connection.
    document.getElementById("liveStream").src = "";
}

async function pollStatus() {
    if (!jobId) return;

    try {
        const resp = await fetch("/api/status/" + jobId);
        const data = await resp.json();

        if (data.status === "error") {
            clearInterval(pollTimer);
            stopLiveStream();
            document.getElementById("processingStatus").innerText = "Error: " + (data.message || "unknown error");
            return;
        }

        document.getElementById("progressBar").style.width = (data.progress || 0) + "%";
        document.getElementById("processingStatus").innerText =
            "Processing... " + (data.progress || 0) + "%";

        const counts = data.counts || { unloading: 0, loading: 0 };
        document.getElementById("liveUnloading").innerText = counts.unloading;
        document.getElementById("liveLoading").innerText = counts.loading;
        document.getElementById("liveTotal").innerText = counts.unloading + counts.loading;

        if (data.status === "done") {
            clearInterval(pollTimer);
            stopLiveStream();
            showResult(counts);
        }

    } catch (err) {
        clearInterval(pollTimer);
        document.getElementById("processingStatus").innerText = "Status polling error: " + err.message;
    }
}

function showResult(counts) {
    document.getElementById("finalUnloading").innerText = counts.unloading;
    document.getElementById("finalLoading").innerText = counts.loading;
    document.getElementById("finalTotal").innerText = counts.unloading + counts.loading;

    const videoUrl = "/api/video/" + jobId;
    document.getElementById("resultVideo").src = videoUrl;
    document.getElementById("downloadLink").href = videoUrl;

    goToStep("stepResult");
}

function resetVideoFlow() {
    stopLiveStream();
    videoId = null;
    jobId = null;
    videoInput.value = "";
    videoFileNameEl.textContent = "";
    uploadBtn.disabled = true;
    document.getElementById("uploadStatus").innerText = "Choose a video to begin.";
    goToStep("stepUpload");
}


// ==========================================================
// SINGLE IMAGE TEST TAB
// ==========================================================

const imageDropZone = document.getElementById("imageDropZone");
const imageInputEl = document.getElementById("imageInput");
const imageFileNameEl = document.getElementById("imageFileName");
const testImageBtn = document.getElementById("testImageBtn");

imageDropZone.addEventListener("click", () => imageInputEl.click());
imageDropZone.addEventListener("dragover", e => { e.preventDefault(); imageDropZone.classList.add("dragover"); });
imageDropZone.addEventListener("dragleave", () => imageDropZone.classList.remove("dragover"));
imageDropZone.addEventListener("drop", e => {
    e.preventDefault();
    imageDropZone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
        imageInputEl.files = e.dataTransfer.files;
        onImageFileChosen();
    }
});
imageInputEl.addEventListener("change", onImageFileChosen);

function onImageFileChosen() {
    const file = imageInputEl.files[0];
    if (!file) return;
    imageFileNameEl.textContent = "Selected: " + file.name;
    testImageBtn.disabled = false;
}

async function runImageTest() {
    const file = imageInputEl.files[0];
    if (!file) return;

    testImageBtn.disabled = true;
    document.getElementById("imageStatus").innerText = "Running segmentation...";

    const form = new FormData();
    form.append("image", file);

    try {
        const resp = await fetch("/api/test_image", { method: "POST", body: form });
        const data = await resp.json();

        if (!data.success) {
            document.getElementById("imageStatus").innerText = data.message || "Failed.";
            testImageBtn.disabled = false;
            return;
        }

        document.getElementById("imgOriginal").src = "data:image/jpeg;base64," + data.original_base64;
        document.getElementById("imgResult").src = "data:image/jpeg;base64," + data.result_base64;

        document.getElementById("imgBagCount").innerText = data.counts["Rice-bag-10"];
        document.getElementById("imgOtherCount").innerText = data.counts["other"];
        document.getElementById("imgPileCount").innerText = data.counts["pile"];
        document.getElementById("imageKpis").style.display = "grid";

        const tbody = document.getElementById("imgDetTable");
        tbody.innerHTML = "";
        data.detections.forEach(d => {
            const tr = document.createElement("tr");
            tr.innerHTML = "<td>" + d.id + "</td><td>" + d.class_name + "</td><td>" + d.confidence + "</td>";
            tbody.appendChild(tr);
        });

        document.getElementById("imageResultPanel").style.display = "block";
        document.getElementById("imageStatus").innerText = "Done.";

    } catch (err) {
        document.getElementById("imageStatus").innerText = "Error: " + err.message;
    }

    testImageBtn.disabled = false;
}

</script>

</body>
</html>
"""


# ============================================================
# ROUTES
# ============================================================

def encode_jpeg(image_bgr):
    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("Could not encode image.")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


@app.route("/")
def index():
    return render_template_string(HTML, error=None)


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "device": str(DEVICE),
        "model": str(MODEL_PATH_RESOLVED),
        "tracker": TRACKER_TYPE,
    })


@app.route("/api/upload", methods=["POST"])
def api_upload():
    try:
        video_file = request.files.get("video")
        if video_file is None or video_file.filename == "":
            return jsonify({"success": False, "message": "Please choose a video file."})

        video_id = uuid.uuid4().hex
        suffix = Path(video_file.filename).suffix or ".mp4"
        video_path = UPLOAD_DIR / f"{video_id}{suffix}"
        video_file.save(video_path)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            video_path.unlink(missing_ok=True)
            return jsonify({"success": False, "message": "Could not open the uploaded file as a video."})

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        ok, first_frame = cap.read()
        cap.release()

        if not ok or first_frame is None:
            video_path.unlink(missing_ok=True)
            return jsonify({"success": False, "message": "Could not read a frame from this video."})

        VIDEOS[video_id] = {
            "path": video_path, "width": width, "height": height,
            "fps": fps, "frame_count": frame_count,
        }

        return jsonify({
            "success": True,
            "video_id": video_id,
            "width": width, "height": height, "fps": fps, "frame_count": frame_count,
            "first_frame_base64": encode_jpeg(first_frame),
        })

    except Exception as exc:
        return jsonify({"success": False, "message": f"Runtime error: {exc}"})


def _validate_line_payload(line):
    role = line.get("role")
    if role not in ("unloading", "loading"):
        return None, f"Invalid line role: {role!r}"
    p1, p2 = line.get("p1"), line.get("p2")
    if not (isinstance(p1, list) and len(p1) == 2 and isinstance(p2, list) and len(p2) == 2):
        return None, f"Invalid points for {role} line."
    direction = line.get("direction")
    if direction not in ("neg_to_pos", "pos_to_neg"):
        return None, f"Invalid direction for {role} line."

    return {
        "role": role,
        "p1": (float(p1[0]), float(p1[1])),
        "p2": (float(p2[0]), float(p2[1])),
        "direction": direction,
        "color": ROLE_COLORS_BGR[role],
        "label": f"{ROLE_LABELS[role]} ({'RED' if role == 'unloading' else 'BLUE'})",
    }, None


@app.route("/api/start", methods=["POST"])
def api_start():
    try:
        payload = request.get_json(force=True)
        video_id = payload.get("video_id")
        raw_lines = payload.get("lines")

        if video_id not in VIDEOS:
            return jsonify({"success": False, "message": "Unknown video_id -- please re-upload the video."})

        if not isinstance(raw_lines, list) or len(raw_lines) != 2:
            return jsonify({"success": False, "message": "Exactly 2 lines (unloading + loading) are required."})

        lines_config = []
        for raw in raw_lines:
            line, err = _validate_line_payload(raw)
            if err:
                return jsonify({"success": False, "message": err})
            lines_config.append(line)

        roles = {line["role"] for line in lines_config}
        if roles != {"unloading", "loading"}:
            return jsonify({"success": False, "message": "Both an unloading line and a loading line are required."})

        job_id = uuid.uuid4().hex
        JOBS[job_id] = {
            "status": "running", "progress": 0.0,
            "counts": {"unloading": 0, "loading": 0},
            "message": "", "output_path": None,
        }

        video_path = VIDEOS[video_id]["path"]
        thread = threading.Thread(
            target=process_video, args=(job_id, video_path, lines_config), daemon=True
        )
        thread.start()

        return jsonify({"success": True, "job_id": job_id})

    except Exception as exc:
        return jsonify({"success": False, "message": f"Runtime error: {exc}"})


@app.route("/api/status/<job_id>")
def api_status(job_id):
    job = JOBS.get(job_id)
    if job is None:
        return jsonify({"status": "error", "message": "Unknown job_id."}), 404
    return jsonify({
        "status": job["status"],
        "progress": job["progress"],
        "counts": job["counts"],
        "message": job.get("message", ""),
    })


@app.route("/api/live/<job_id>")
def api_live(job_id):
    """MJPEG live preview of the job actually processing -- lets the UI
    show masks/lines/counts updating in real time during Step 3, instead
    of a bare progress bar with no visual feedback. Standard
    multipart/x-mixed-replace trick: a plain <img> tag repaints itself as
    each new JPEG part arrives, no JS polling or websockets needed."""

    def generate():
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        while True:
            job = JOBS.get(job_id)
            if job is None:
                break
            frame_bytes = job.get("latest_frame_jpeg")
            if frame_bytes:
                yield boundary + frame_bytes + b"\r\n"
            if job["status"] in ("done", "error"):
                break
            time.sleep(0.15)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/video/<job_id>")
def api_video(job_id):
    job = JOBS.get(job_id)
    if job is None or job["status"] != "done" or not job.get("output_path"):
        return jsonify({"success": False, "message": "Video not ready."}), 404
    return send_file(job["output_path"], mimetype="video/mp4")


@app.route("/api/test_image", methods=["POST"])
def api_test_image():
    try:
        image_file = request.files.get("image")
        if image_file is None or image_file.filename == "":
            return jsonify({"success": False, "message": "Please select an image."})

        file_bytes = np.frombuffer(image_file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"success": False, "message": "Could not read the uploaded image."})

        annotated, counts, detections = annotate_test_image(image)

        return jsonify({
            "success": True,
            "original_base64": encode_jpeg(image),
            "result_base64": encode_jpeg(annotated),
            "counts": counts,
            "detections": detections,
        })

    except Exception as exc:
        return jsonify({"success": False, "message": f"Runtime error: {exc}"})


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("RICE BAG LOADING / UNLOADING VIDEO COUNTER")
    print("=" * 70)
    print(f"Model   : {MODEL_PATH_RESOLVED}")
    print(f"Device  : {DEVICE}")
    print(f"Tracker : {TRACKER_TYPE}")
    print("Local URL: http://127.0.0.1:5008")
    print("Network URL: http://0.0.0.0:5008")
    print("=" * 70)

    app.run(host="0.0.0.0", port=5008, debug=False, threaded=True)
