from pathlib import Path
import uuid

import cv2
import numpy as np
import torch
from flask import Flask, render_template_string, request, send_file
from ultralytics import YOLO


# ============================================================
# C-BELT SACK - SEGMENTATION + TRACKING + LINE COUNTING UI
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ACTUAL training location:
# runs/segment/runs/segment/final_c_belt_sack_yolo26l_seg_v1/
RUN_ROOT = (
    PROJECT_ROOT
    / "runs"
    / "segment"
    / "runs"
    / "segment"
    / "final_c_belt_sack_yolo26l_seg_v1"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "final_c_belt_line_count_ui"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PORT = 50011
IMG_SIZE = 1280

DEFAULT_CONF = 0.35
DEFAULT_IOU = 0.50

# ByteTrack
TRACKER = "bytetrack.yaml"

# Extra duplicate-mask suppression.
# This is deliberately conservative so nearby real sacks are not
# accidentally merged.
DUP_BOX_IOU = 0.70
DUP_MASK_IOU = 0.60

# Small dead-band around the counting line to reduce jitter.
LINE_TOLERANCE = 10


# ============================================================
# FIND BEST.PT
# ============================================================

def find_best_model():
    exact = RUN_ROOT / "weights" / "best.pt"

    if exact.exists():
        return exact

    candidates = list(RUN_ROOT.rglob("best.pt")) if RUN_ROOT.exists() else []

    if not candidates:
        candidates = list((PROJECT_ROOT / "runs").rglob("best.pt"))

    if not candidates:
        raise FileNotFoundError(
            "best.pt not found.\n\n"
            f"Expected training run:\n{RUN_ROOT}\n\n"
            "Run:\n"
            f"find '{PROJECT_ROOT}/runs' -name 'best.pt' -type f"
        )

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    print("\nBEST.PT CANDIDATES:")
    for p in candidates:
        print(f"  {p}")

    return candidates[0]


# ============================================================
# DEVICE + MODEL
# ============================================================

if torch.cuda.is_available():
    DEVICE = 0
    DEVICE_NAME = torch.cuda.get_device_name(0)
else:
    DEVICE = "cpu"
    DEVICE_NAME = "CPU"

MODEL_PATH = find_best_model()

print("\n" + "=" * 75)
print("C-BELT SACK SEGMENTATION + LINE COUNTING")
print("=" * 75)
print(f"Model : {MODEL_PATH}")
print(f"Device: {DEVICE_NAME}")
print(f"Tracker: {TRACKER}")
print("=" * 75)

model = YOLO(str(MODEL_PATH))


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


# ============================================================
# HTML
# ============================================================

HTML = r"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>C-Belt Sack Counter</title>

    <style>
        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            min-height: 100vh;
            font-family: Inter, Arial, sans-serif;
            background: #0b1220;
            color: #e8eef7;
        }

        .header {
            background: linear-gradient(135deg, #111c33, #142d46);
            border-bottom: 1px solid #24435f;
            padding: 24px 30px;
        }

        .header h1 {
            margin: 0 0 7px 0;
            font-size: 28px;
        }

        .header p {
            margin: 0;
            color: #9fb4c9;
        }

        .container {
            width: min(1250px, 94%);
            margin: 25px auto 50px auto;
        }

        .card {
            background: #111b2e;
            border: 1px solid #203653;
            border-radius: 14px;
            padding: 22px;
            margin-bottom: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,.18);
        }

        .model-info {
            color: #9fb4c9;
            font-size: 13px;
            line-height: 1.7;
            word-break: break-all;
        }

        .model-info strong {
            color: #dce9f5;
        }

        .controls {
            display: grid;
            grid-template-columns: 2fr 1fr 1fr 1fr;
            gap: 16px;
            align-items: end;
        }

        .field label {
            display: block;
            margin-bottom: 7px;
            color: #b9cbe0;
            font-size: 13px;
            font-weight: 600;
        }

        input, select {
            width: 100%;
            background: #0b1425;
            color: #e8eef7;
            border: 1px solid #2a4663;
            border-radius: 8px;
            padding: 11px;
            outline: none;
        }

        input:focus, select:focus {
            border-color: #20c7b8;
        }

        .btn {
            width: 100%;
            padding: 12px 18px;
            border: 0;
            border-radius: 8px;
            background: #19b8a8;
            color: #061513;
            font-weight: 800;
            cursor: pointer;
        }

        .btn:hover {
            background: #24cfc0;
        }

        .stats {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 14px;
            margin-top: 18px;
        }

        .stat {
            background: #0c1729;
            border: 1px solid #203b56;
            border-radius: 10px;
            padding: 17px;
        }

        .stat .label {
            color: #8fa6bc;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: .6px;
        }

        .stat .value {
            margin-top: 5px;
            font-size: 28px;
            font-weight: 800;
            color: #edf7ff;
        }

        .video-wrap {
            background: #050a12;
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid #203653;
        }

        video, img {
            display: block;
            width: 100%;
            max-height: 720px;
            object-fit: contain;
            background: #050a12;
        }

        .note {
            color: #8fa6bc;
            font-size: 13px;
            margin-top: 13px;
            line-height: 1.6;
        }

        .error {
            background: #351722;
            border: 1px solid #733044;
            color: #ffb6c5;
            padding: 15px;
            border-radius: 10px;
        }

        .success {
            background: #102c2a;
            border: 1px solid #1d615a;
            color: #9de9df;
            padding: 15px;
            border-radius: 10px;
        }

        .legend {
            display: flex;
            gap: 18px;
            flex-wrap: wrap;
            margin-top: 14px;
            color: #a8bacd;
            font-size: 13px;
        }

        .dot {
            width: 10px;
            height: 10px;
            display: inline-block;
            border-radius: 50%;
            margin-right: 5px;
        }

        @media (max-width: 900px) {
            .controls {
                grid-template-columns: 1fr 1fr;
            }

            .stats {
                grid-template-columns: 1fr;
            }
        }

        @media (max-width: 600px) {
            .controls {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>

<body>

<div class="header">
    <div class="container" style="margin-top:0;margin-bottom:0;">
        <h1>📦 C-Belt Sack Counter</h1>
        <p>YOLO26L-Seg + ByteTrack + Mask-based Line Crossing</p>
    </div>
</div>

<div class="container">

    <div class="card">
        <div class="model-info">
            <strong>Model:</strong> {{ model_path }}<br>
            <strong>Device:</strong> {{ device }}<br>
            <strong>Tracker:</strong> ByteTrack
        </div>
    </div>

    <div class="card">
        <form method="POST" enctype="multipart/form-data">

            <div class="controls">

                <div class="field">
                    <label>Upload Image / Video</label>
                    <input
                        type="file"
                        name="media"
                        accept="image/*,video/*"
                        required
                    >
                </div>

                <div class="field">
                    <label>Confidence</label>
                    <input
                        type="number"
                        name="conf"
                        min="0.05"
                        max="0.95"
                        step="0.05"
                        value="{{ conf }}"
                    >
                </div>

                <div class="field">
                    <label>Line Orientation</label>
                    <select name="orientation">
                        <option value="horizontal"
                            {% if orientation == "horizontal" %}selected{% endif %}>
                            Horizontal
                        </option>
                        <option value="vertical"
                            {% if orientation == "vertical" %}selected{% endif %}>
                            Vertical
                        </option>
                    </select>
                </div>

                <div class="field">
                    <label>Line Position %</label>
                    <input
                        type="number"
                        name="line_position"
                        min="10"
                        max="90"
                        step="1"
                        value="{{ line_position }}"
                    >
                </div>

            </div>

            <div style="margin-top:16px;">
                <button class="btn" type="submit">
                    ▶ Process & Count
                </button>
            </div>

        </form>

        <div class="note">
            The line position is relative to the video frame.
            A sack is counted once when its tracked mask centroid
            crosses the line. ByteTrack IDs prevent the same tracked
            sack from being counted repeatedly.
        </div>
    </div>


    {% if error %}
        <div class="card">
            <div class="error">
                <strong>Error:</strong><br><br>
                {{ error }}
            </div>
        </div>
    {% endif %}


    {% if result_type == "image" %}
        <div class="card">

            <div class="stats">
                <div class="stat">
                    <div class="label">Detected Sacks</div>
                    <div class="value">{{ count }}</div>
                </div>

                <div class="stat">
                    <div class="label">Line Crossings</div>
                    <div class="value">{{ count }}</div>
                </div>

                <div class="stat">
                    <div class="label">Mode</div>
                    <div class="value" style="font-size:20px;">
                        Segmentation
                    </div>
                </div>
            </div>

            <div style="margin-top:20px;" class="video-wrap">
                <img src="{{ result_url }}">
            </div>

        </div>
    {% endif %}


    {% if result_type == "video" %}
        <div class="card">

            <div class="success">
                <strong>Processing completed successfully.</strong>
            </div>

            <div class="stats">
                <div class="stat">
                    <div class="label">Total Count</div>
                    <div class="value">{{ count }}</div>
                </div>

                <div class="stat">
                    <div class="label">Line</div>
                    <div class="value" style="font-size:20px;">
                        {{ orientation }}
                    </div>
                </div>

                <div class="stat">
                    <div class="label">Position</div>
                    <div class="value">
                        {{ line_position }}%
                    </div>
                </div>
            </div>

            <div style="margin-top:20px;" class="video-wrap">
                <video controls>
                    <source src="{{ result_url }}" type="video/mp4">
                    Your browser does not support video playback.
                </video>
            </div>

            <div class="legend">
                <span>
                    <span class="dot" style="background:#ff4757;"></span>
                    Counting line
                </span>

                <span>
                    Masks = detected sack instances
                </span>

                <span>
                    Same Track ID = counted only once
                </span>
            </div>

            <div class="note">
                The output video contains segmentation masks, the
                counting line, and the running count. Bounding boxes,
                labels and confidence text are intentionally hidden.
            </div>

        </div>
    {% endif %}

</div>

</body>
</html>
"""


# ============================================================
# GEOMETRY HELPERS
# ============================================================

def bbox_iou(box_a, box_b):
    """IoU between two xyxy boxes."""

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    intersection = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def mask_iou(mask_a, mask_b):
    """IoU between two binary masks."""

    a = mask_a.astype(bool)
    b = mask_b.astype(bool)

    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()

    if union == 0:
        return 0.0

    return float(intersection / union)


def mask_centroid(mask, box):
    """
    Calculate centroid from the segmentation mask.
    Falls back to bbox center if the mask is empty.
    """

    ys, xs = np.where(mask)

    if len(xs) == 0:
        x1, y1, x2, y2 = box
        return (
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0
        )

    return (
        float(xs.mean()),
        float(ys.mean())
    )


# ============================================================
# DUPLICATE MASK SUPPRESSION
# ============================================================

def suppress_duplicate_masks(result):
    """
    Conservative duplicate suppression.

    A detection is considered a duplicate only when BOTH:
      1. bounding boxes overlap strongly
      2. segmentation masks overlap strongly

    The lower-confidence detection is removed from visualization
    and counting for this frame.
    """

    if result.masks is None:
        return list(range(len(result.boxes)))

    if result.boxes is None or len(result.boxes) == 0:
        return []

    masks = result.masks.data.detach().cpu().numpy()

    boxes = result.boxes.xyxy.detach().cpu().numpy()

    confs = result.boxes.conf.detach().cpu().numpy()

    n = len(masks)

    keep = [True] * n

    # Highest confidence gets priority.
    order = np.argsort(-confs)

    for oi in range(n):

        i = int(order[oi])

        if not keep[i]:
            continue

        for oj in range(oi + 1, n):

            j = int(order[oj])

            if not keep[j]:
                continue

            biou = bbox_iou(
                boxes[i],
                boxes[j]
            )

            if biou < DUP_BOX_IOU:
                continue

            miou = mask_iou(
                masks[i],
                masks[j]
            )

            if miou >= DUP_MASK_IOU:
                # i has higher/equal confidence
                keep[j] = False

    return [
        i for i in range(n)
        if keep[i]
    ]


# ============================================================
# COUNTING-LINE HELPERS
# ============================================================

def point_side(x, y, orientation, line_position, width, height):
    """
    Returns:
      -1 = one side
       0 = inside dead-band
       1 = opposite side
    """

    if orientation == "horizontal":

        line_y = height * line_position

        distance = y - line_y

    else:

        line_x = width * line_position

        distance = x - line_x

    if abs(distance) <= LINE_TOLERANCE:
        return 0

    return 1 if distance > 0 else -1


def draw_counting_line(
    frame,
    orientation,
    line_position
):

    h, w = frame.shape[:2]

    if orientation == "horizontal":

        y = int(h * line_position)

        cv2.line(
            frame,
            (0, y),
            (w, y),
            (71, 77, 255),
            4
        )

        # Small end markers
        cv2.circle(
            frame,
            (20, y),
            8,
            (71, 77, 255),
            -1
        )

        cv2.circle(
            frame,
            (w - 20, y),
            8,
            (71, 77, 255),
            -1
        )

    else:

        x = int(w * line_position)

        cv2.line(
            frame,
            (x, 0),
            (x, h),
            (71, 77, 255),
            4
        )

        cv2.circle(
            frame,
            (x, 20),
            8,
            (71, 77, 255),
            -1
        )

        cv2.circle(
            frame,
            (x, h - 20),
            8,
            (71, 77, 255),
            -1
        )


def draw_counter(frame, count):

    cv2.rectangle(
        frame,
        (20, 20),
        (260, 82),
        (10, 18, 32),
        -1
    )

    cv2.putText(
        frame,
        f"SACK COUNT: {count}",
        (38, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


# ============================================================
# PROCESS IMAGE
# ============================================================

def process_image(
    input_path,
    output_path,
    conf,
    orientation,
    line_position
):

    # For a single image there is no temporal track crossing.
    # We still show the configured line and clean masks.
    results = model.predict(
        source=str(input_path),
        conf=conf,
        iou=DEFAULT_IOU,
        imgsz=IMG_SIZE,
        device=DEVICE,
        verbose=False
    )

    result = results[0]

    keep_indices = suppress_duplicate_masks(result)

    if result.masks is not None:
        count = len(keep_indices)
    else:
        count = 0

    plotted = result.plot(
        boxes=False,
        labels=False,
        conf=False,
        masks=True
    )

    h, w = plotted.shape[:2]

    draw_counting_line(
        plotted,
        orientation,
        line_position
    )

    draw_counter(
        plotted,
        count
    )

    cv2.imwrite(
        str(output_path),
        plotted
    )

    return count


# ============================================================
# PROCESS VIDEO
# ============================================================

def process_video(
    input_path,
    output_path,
    conf,
    iou,
    orientation,
    line_position
):

    cap = cv2.VideoCapture(
        str(input_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open uploaded video."
        )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
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
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            "Could not create output video."
        )

    # --------------------------------------------------------
    # Reset tracking state for every new video.
    # --------------------------------------------------------

    model.predictor = None

    # --------------------------------------------------------
    # Tracking state
    # --------------------------------------------------------

    previous_side = {}
    counted_ids = set()

    # Track-ID aliases created by duplicate suppression.
    # If two tracker IDs are found to be the same overlapping
    # object, both resolve to one canonical ID.
    parent = {}

    def find_root(x):
        parent.setdefault(x, x)

        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]

        return x

    def union(a, b):
        ra = find_root(a)
        rb = find_root(b)

        if ra != rb:
            # Stable canonical ID:
            # keep the older/smaller tracker ID.
            root = min(ra, rb)
            other = max(ra, rb)
            parent[other] = root

    total_frames = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        total_frames += 1

        # ----------------------------------------------------
        # YOLO26L-Seg + ByteTrack
        # ----------------------------------------------------

        results = model.track(
            source=frame,
            conf=conf,
            iou=iou,
            imgsz=IMG_SIZE,
            device=DEVICE,
            tracker=TRACKER,
            persist=True,
            verbose=False
        )

        result = results[0]

        # ----------------------------------------------------
        # No detections
        # ----------------------------------------------------

        if (
            result.boxes is None
            or len(result.boxes) == 0
        ):

            draw_counting_line(
                frame,
                orientation,
                line_position
            )

            draw_counter(
                frame,
                len(counted_ids)
            )

            writer.write(frame)

            continue

        # ----------------------------------------------------
        # Track IDs
        # ----------------------------------------------------

        if result.boxes.id is None:

            draw_counting_line(
                frame,
                orientation,
                line_position
            )

            draw_counter(
                frame,
                len(counted_ids)
            )

            writer.write(frame)

            continue

        track_ids = (
            result.boxes.id
            .detach()
            .cpu()
            .numpy()
            .astype(int)
        )

        boxes = (
            result.boxes.xyxy
            .detach()
            .cpu()
            .numpy()
        )

        confs = (
            result.boxes.conf
            .detach()
            .cpu()
            .numpy()
        )

        # ----------------------------------------------------
        # Segmentation masks
        # ----------------------------------------------------

        if result.masks is not None:

            masks = (
                result.masks.data
                .detach()
                .cpu()
                .numpy()
            )

            # Ensure masks are frame-size.
            if (
                masks.shape[1] != height
                or masks.shape[2] != width
            ):

                resized_masks = []

                for mask in masks:

                    resized = cv2.resize(
                        mask.astype(np.uint8),
                        (width, height),
                        interpolation=cv2.INTER_NEAREST
                    )

                    resized_masks.append(
                        resized > 0
                    )

                masks = np.array(
                    resized_masks,
                    dtype=bool
                )

            else:

                masks = masks > 0

        else:

            masks = None

        # ----------------------------------------------------
        # Additional overlap suppression
        #
        # This operates on the current tracked detections.
        # For count purposes, duplicate tracker IDs are merged
        # when their masks strongly overlap.
        # ----------------------------------------------------

        keep_indices = []

        if masks is not None:

            n = min(
                len(masks),
                len(track_ids)
            )

            keep_flags = [True] * n

            order = np.argsort(-confs[:n])

            for oi in range(n):

                i = int(order[oi])

                if not keep_flags[i]:
                    continue

                for oj in range(oi + 1, n):

                    j = int(order[oj])

                    if not keep_flags[j]:
                        continue

                    biou = bbox_iou(
                        boxes[i],
                        boxes[j]
                    )

                    if biou < DUP_BOX_IOU:
                        continue

                    miou = mask_iou(
                        masks[i],
                        masks[j]
                    )

                    if miou >= DUP_MASK_IOU:

                        # Merge tracker IDs.
                        union(
                            int(track_ids[i]),
                            int(track_ids[j])
                        )

                        # Hide lower-confidence duplicate.
                        keep_flags[j] = False

            keep_indices = [
                i for i in range(n)
                if keep_flags[i]
            ]

        else:

            keep_indices = list(
                range(len(track_ids))
            )

        # ----------------------------------------------------
        # Build clean visualization
        # ----------------------------------------------------

        # Start with original frame.
        annotated = frame.copy()

        # ----------------------------------------------------
        # Draw masks manually so suppressed duplicates are
        # not displayed.
        # ----------------------------------------------------

        if masks is not None:

            overlay = annotated.copy()

            for idx in keep_indices:

                mask = masks[idx]

                # Use deterministic per-track color.
                tid = int(track_ids[idx])

                rng = np.random.default_rng(
                    tid * 9973 + 17
                )

                color = tuple(
                    int(x)
                    for x in rng.integers(
                        60,
                        230,
                        size=3
                    )
                )

                overlay[mask] = (
                    0.55 * overlay[mask]
                    + 0.45 * np.array(
                        color,
                        dtype=np.uint8
                    )
                ).astype(np.uint8)

            annotated = cv2.addWeighted(
                overlay,
                0.70,
                annotated,
                0.30,
                0
            )

        # ----------------------------------------------------
        # Line crossing
        # ----------------------------------------------------

        for idx in keep_indices:

            tid = int(
                track_ids[idx]
            )

            canonical_id = find_root(tid)

            if masks is not None:

                cx, cy = mask_centroid(
                    masks[idx],
                    boxes[idx]
                )

            else:

                x1, y1, x2, y2 = boxes[idx]

                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

            current_side = point_side(
                cx,
                cy,
                orientation,
                line_position,
                width,
                height
            )

            old_side = previous_side.get(
                canonical_id
            )

            # Ignore positions inside the dead-band.
            if current_side != 0:

                if (
                    old_side is not None
                    and old_side != 0
                    and old_side != current_side
                    and canonical_id not in counted_ids
                ):

                    counted_ids.add(
                        canonical_id
                    )

                previous_side[
                    canonical_id
                ] = current_side

        # ----------------------------------------------------
        # Counting line + counter
        # ----------------------------------------------------

        draw_counting_line(
            annotated,
            orientation,
            line_position
        )

        draw_counter(
            annotated,
            len(counted_ids)
        )

        writer.write(
            annotated
        )

        if total_frames % 50 == 0:

            print(
                f"Frames: {total_frames} | "
                f"Count: {len(counted_ids)}"
            )

    cap.release()
    writer.release()

    print("\n" + "-" * 70)
    print("VIDEO PROCESSING COMPLETED")
    print(f"Frames : {total_frames}")
    print(f"Count  : {len(counted_ids)}")
    print(f"Output : {output_path}")
    print("-" * 70)

    return len(counted_ids)


# ============================================================
# FLASK ROUTE
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

    orientation = "horizontal"
    line_position_percent = 50

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

            iou = DEFAULT_IOU

            orientation = request.form.get(
                "orientation",
                "horizontal"
            )

            line_position_percent = int(
                request.form.get(
                    "line_position",
                    "50"
                )
            )

            if not 0.01 <= conf <= 0.99:

                raise ValueError(
                    "Confidence must be between "
                    "0.01 and 0.99."
                )

            if orientation not in {
                "horizontal",
                "vertical"
            }:

                raise ValueError(
                    "Invalid line orientation."
                )

            if not 10 <= line_position_percent <= 90:

                raise ValueError(
                    "Line position must be between "
                    "10 and 90 percent."
                )

            line_position = (
                line_position_percent / 100.0
            )

            extension = Path(
                file.filename
            ).suffix.lower()

            unique_id = uuid.uuid4().hex[:10]

            input_path = (
                OUTPUT_DIR
                / f"input_{unique_id}{extension}"
            )

            file.save(
                input_path
            )

            image_extensions = {
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".webp"
            }

            video_extensions = {
                ".mp4",
                ".avi",
                ".mov",
                ".mkv",
                ".webm"
            }

            # ------------------------------------------------
            # IMAGE
            # ------------------------------------------------

            if extension in image_extensions:

                output_path = (
                    OUTPUT_DIR
                    / f"result_{unique_id}.jpg"
                )

                count = process_image(
                    input_path,
                    output_path,
                    conf,
                    orientation,
                    line_position
                )

                result_type = "image"

                result_url = (
                    f"/result/"
                    f"{output_path.name}"
                )

            # ------------------------------------------------
            # VIDEO
            # ------------------------------------------------

            elif extension in video_extensions:

                output_path = (
                    OUTPUT_DIR
                    / f"result_{unique_id}.mp4"
                )

                count = process_video(
                    input_path,
                    output_path,
                    conf,
                    iou,
                    orientation,
                    line_position
                )

                result_type = "video"

                result_url = (
                    f"/result/"
                    f"{output_path.name}"
                )

            else:

                raise ValueError(
                    f"Unsupported file type: {extension}"
                )

        except Exception as e:

            error = str(e)

    return render_template_string(
        HTML,
        model_path=str(MODEL_PATH),
        device=DEVICE_NAME,
        result_type=result_type,
        result_url=result_url,
        count=count,
        error=error,
        conf=conf,
        iou=iou,
        orientation=orientation,
        line_position=line_position_percent
    )


# ============================================================
# SERVE RESULTS
# ============================================================

@app.route(
    "/result/<filename>"
)
def result_file(filename):

    path = OUTPUT_DIR / filename

    if not path.exists():
        return "File not found", 404

    return send_file(path)


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 75)
    print("C-BELT SACK COUNTER UI")
    print("=" * 75)
    print(f"Model : {MODEL_PATH}")
    print(f"Device: {DEVICE_NAME}")
    print("Tracker: ByteTrack")
    print("URL   : http://localhost:50011")
    print("=" * 75)

    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=False,
        threaded=True
    )
