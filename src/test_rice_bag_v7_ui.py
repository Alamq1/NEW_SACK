from flask import Flask, request, render_template_string
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import uuid
import io
import base64


# ============================================================
# V7 UNSEEN IMAGE TESTING
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT
    / "outputs"
    / "rice_bag_10_v7_detection"
    / "weights"
    / "best.pt"
)

UPLOAD_DIR = ROOT / "data" / "test_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CHECK MODEL
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"\nV7 model not found:\n{MODEL_PATH}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("RICE BAG 10 - V7 UNSEEN IMAGE TESTING")
print("=" * 70)
print("Model:", MODEL_PATH)

model = YOLO(str(MODEL_PATH))

print("Classes:", model.names)


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

<title>Rice Bag V7 Detection</title>

<style>

body {
    margin: 0;
    font-family: Arial, sans-serif;
    background: #f4f6f8;
}

.container {
    width: 90%;
    max-width: 1200px;
    margin: 40px auto;
}

.card {
    background: white;
    padding: 30px;
    border-radius: 16px;
    box-shadow: 0 5px 25px rgba(0,0,0,0.08);
}

h1 {
    margin-top: 0;
}

.upload {
    padding: 25px;
    border: 2px dashed #aaa;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 25px;
}

button {
    margin-top: 15px;
    padding: 12px 25px;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    font-size: 16px;
}

.result {
    margin-top: 30px;
}

.result img {
    max-width: 100%;
    border-radius: 12px;
}

.stats {
    display: flex;
    gap: 20px;
    margin-bottom: 20px;
}

.stat {
    background: #f1f3f5;
    padding: 20px;
    border-radius: 12px;
    flex: 1;
    text-align: center;
}

.number {
    font-size: 32px;
    font-weight: bold;
}

</style>

</head>

<body>

<div class="container">

<div class="card">

<h1>🌾 Rice Bag V7 Detection</h1>

<p>Upload an unseen image to test the trained model.</p>

<div class="upload">

<form method="POST" enctype="multipart/form-data">

<input
    type="file"
    name="image"
    accept="image/*"
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

<div class="stat">

<div>Total Bags</div>

<div class="number">
{{ count }}
</div>

</div>

<div class="stat">

<div>Model</div>

<div class="number" style="font-size:20px;">
V7
</div>

</div>

</div>

<h2>Detection Result</h2>

<img src="data:image/jpeg;base64,{{ result }}">

</div>

{% endif %}

</div>

</div>

</body>
</html>
"""


# ============================================================
# DETECTION
# ============================================================

def detect_image(image_bytes):

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    # --------------------------------------------------------
    # YOLO INFERENCE
    # --------------------------------------------------------

    results = model.predict(
        source=image,
        imgsz=1280,
        conf=0.25,
        iou=0.50,
        max_det=1000,
        device=0,
        verbose=False
    )

    result = results[0]

    # --------------------------------------------------------
    # DRAW ONLY CENTER POINTS
    # --------------------------------------------------------

    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            22
        )
    except:
        font = ImageFont.load_default()

    count = 0

    if result.boxes is not None:

        boxes = result.boxes

        for box in boxes:

            xyxy = box.xyxy[0].tolist()

            x1, y1, x2, y2 = xyxy

            # Center
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            count += 1

            # ------------------------------------------------
            # POINT
            # ------------------------------------------------

            radius = 7

            draw.ellipse(
                [
                    cx - radius,
                    cy - radius,
                    cx + radius,
                    cy + radius
                ],
                fill="red"
            )

            # ------------------------------------------------
            # BAG NUMBER
            # ------------------------------------------------

            label = str(count)

            draw.text(
                (cx + 10, cy - 12),
                label,
                fill="white",
                font=font,
                stroke_width=3,
                stroke_fill="black"
            )

    # --------------------------------------------------------
    # ENCODE IMAGE
    # --------------------------------------------------------

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=95
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return encoded, count


# ============================================================
# ROUTE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    count = 0

    if request.method == "POST":

        file = request.files.get("image")

        if file and file.filename:

            image_bytes = file.read()

            result, count = detect_image(
                image_bytes
            )

            print(
                f"Detected bags: {count}"
            )

    return render_template_string(
        HTML,
        result=result,
        count=count
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("FLASK SERVER STARTING")
    print("=" * 70)

    print("\nOpen in browser:")
    print("http://127.0.0.1:5000")

    print("\nUpload an UNSEEN image and test V7.")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )