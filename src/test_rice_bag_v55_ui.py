from pathlib import Path
from flask import Flask, request, render_template_string
from ultralytics import YOLO
import cv2
import uuid


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT
    / "outputs"
    / "rice_bag_10_v5_detection"
    / "weights"
    / "best.pt"
)

UPLOAD_DIR = ROOT / "outputs" / "v5_test_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.50

IMAGE_SIZE = 1280

# Change this according to actual bag weight
BAG_WEIGHT_KG = 50


# ============================================================
# MODEL
# ============================================================

print("=" * 70)
print("RICE BAG 10 - V5 UNSEEN IMAGE TESTING")
print("=" * 70)

print(f"Model: {MODEL_PATH}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"\nV5 detection model not found:\n{MODEL_PATH}\n\n"
        "Train the V5 detection model first."
    )

model = YOLO(str(MODEL_PATH))

print("✓ Model loaded")


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

<title>Rice Bag V5 Testing</title>

<style>

body {
    margin: 0;
    padding: 30px;
    background: #111;
    color: white;
    font-family: Arial, sans-serif;
}

.container {
    max-width: 1200px;
    margin: auto;
    text-align: center;
}

h1 {
    margin-bottom: 5px;
}

.subtitle {
    color: #aaa;
    margin-bottom: 30px;
}

.upload {
    background: #1e1e1e;
    padding: 30px;
    border-radius: 15px;
    border: 1px solid #333;
}

input {
    margin: 15px;
}

button {
    padding: 12px 25px;
    border: none;
    border-radius: 8px;
    background: #00c853;
    color: white;
    font-size: 16px;
    cursor: pointer;
}

button:hover {
    opacity: 0.85;
}

.stats {
    display: flex;
    justify-content: center;
    gap: 20px;
    margin: 30px 0;
    flex-wrap: wrap;
}

.card {
    background: #1e1e1e;
    border-radius: 12px;
    padding: 20px 35px;
    min-width: 170px;
}

.number {
    font-size: 32px;
    font-weight: bold;
}

.label {
    color: #aaa;
    margin-top: 5px;
}

.result img {
    max-width: 100%;
    border-radius: 12px;
    border: 1px solid #444;
}

.error {
    color: #ff5252;
    margin-top: 20px;
}

</style>

</head>


<body>

<div class="container">

<h1>🌾 Rice Bag Detection — V5</h1>

<div class="subtitle">
Unseen Image Testing
</div>


<div class="upload">

<form
method="POST"
action="/predict"
enctype="multipart/form-data"
>

<input
type="file"
name="image"
accept="image/*"
required
>

<br>

<button type="submit">
Upload & Detect
</button>

</form>

</div>


{% if error %}

<div class="error">
{{ error }}
</div>

{% endif %}


{% if result %}

<div class="stats">

<div class="card">

<div class="number">
{{ total_bags }}
</div>

<div class="label">
Total Bags
</div>

</div>


<div class="card">

<div class="number">
{{ total_weight }} KG
</div>

<div class="label">
Estimated Weight
</div>

</div>


<div class="card">

<div class="number">
{{ processing_time }} sec
</div>

<div class="label">
Inference Time
</div>

</div>

</div>


<div class="result">

<h2>Detection Result</h2>

<img
src="/result/{{ result }}"
>

</div>

{% endif %}

</div>

</body>

</html>
"""


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template_string(
        HTML,
        result=None,
        error=None
    )


# ============================================================
# PREDICT
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    if "image" not in request.files:

        return render_template_string(
            HTML,
            result=None,
            error="No image uploaded."
        )


    uploaded = request.files["image"]


    if uploaded.filename == "":

        return render_template_string(
            HTML,
            result=None,
            error="Please select an image."
        )


    # --------------------------------------------------------
    # Unique filenames
    # --------------------------------------------------------

    unique_id = uuid.uuid4().hex

    input_path = (
        UPLOAD_DIR
        / f"{unique_id}_input.jpg"
    )

    output_path = (
        UPLOAD_DIR
        / f"{unique_id}_result.jpg"
    )


    uploaded.save(str(input_path))


    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    image = cv2.imread(
        str(input_path)
    )


    if image is None:

        return render_template_string(
            HTML,
            result=None,
            error="Could not read uploaded image."
        )


    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    import time

    start = time.perf_counter()


    results = model.predict(

        source=image,

        imgsz=IMAGE_SIZE,

        conf=CONF_THRESHOLD,

        iou=IOU_THRESHOLD,

        max_det=1000,

        device=0,

        verbose=False

    )


    processing_time = (
        time.perf_counter() - start
    )


    result = results[0]


    # --------------------------------------------------------
    # DETECTIONS
    # --------------------------------------------------------

    detections = []


    if result.boxes is not None:

        boxes = (
            result.boxes.xyxy
            .cpu()
            .numpy()
        )

        confidences = (
            result.boxes.conf
            .cpu()
            .numpy()
        )


        for box, confidence in zip(
            boxes,
            confidences
        ):

            x1, y1, x2, y2 = box

            detections.append(
                (
                    x1,
                    y1,
                    x2,
                    y2,
                    float(confidence)
                )
            )


    # --------------------------------------------------------
    # SORT DETECTIONS
    # --------------------------------------------------------

    detections.sort(
        key=lambda d: (
            d[1],
            d[0]
        )
    )


    # --------------------------------------------------------
    # DRAW BOX + BAG NUMBER
    # --------------------------------------------------------

    output = image.copy()


    for index, detection in enumerate(
        detections,
        start=1
    ):

        x1, y1, x2, y2, confidence = detection

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)


        # Bounding box

        cv2.rectangle(

            output,

            (x1, y1),

            (x2, y2),

            (0, 255, 0),

            2

        )


        # Bag number only
        # Confidence is NOT shown

        cv2.putText(

            output,

            f"Bag {index}",

            (
                x1,
                max(25, y1 - 8)
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (0, 255, 0),

            2,

            cv2.LINE_AA

        )


    # --------------------------------------------------------
    # TOTAL
    # --------------------------------------------------------

    total_bags = len(
        detections
    )

    total_weight = (
        total_bags
        * BAG_WEIGHT_KG
    )


    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    cv2.rectangle(

        output,

        (10, 10),

        (430, 105),

        (0, 0, 0),

        -1

    )


    cv2.putText(

        output,

        f"TOTAL BAGS: {total_bags}",

        (25, 50),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.9,

        (255, 255, 255),

        2,

        cv2.LINE_AA

    )


    cv2.putText(

        output,

        f"TOTAL WEIGHT: {total_weight} KG",

        (25, 88),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (255, 255, 255),

        2,

        cv2.LINE_AA

    )


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    cv2.imwrite(
        str(output_path),
        output
    )


    print(
        f"\nDetected: {total_bags} bags"
    )

    print(
        f"Weight: {total_weight} KG"
    )

    print(
        f"Inference: {processing_time:.3f} sec"
    )


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    return render_template_string(

        HTML,

        result=output_path.name,

        total_bags=total_bags,

        total_weight=total_weight,

        processing_time=f"{processing_time:.3f}",

        error=None

    )


# ============================================================
# RESULT FILE
# ============================================================

@app.route(
    "/result/<filename>"
)
def result(filename):

    from flask import send_file

    path = UPLOAD_DIR / filename

    if not path.exists():

        return "Result not found", 404

    return send_file(str(path))


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("FLASK SERVER")
    print("=" * 70)

    print("\nLocal:")
    print("http://127.0.0.1:5000")

    print("\nNetwork:")
    print("http://192.168.2.240:5000")

    print("\nPress CTRL+C to stop.")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )