from pathlib import Path
import random
import cv2
import html

ROOT = Path("data/Final_C_belt_yolo26")
OUT = Path("outputs/phase1_visual_audit")
OUT.mkdir(parents=True, exist_ok=True)

SAMPLES = 24
random.seed(42)

images = []
for split in ["train", "valid", "test"]:
    img_dir = ROOT / split / "images"
    for p in img_dir.iterdir():
        if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp"}:
            images.append((split, p))

samples = random.sample(images, min(SAMPLES, len(images)))

cards = []

for i, (split, img_path) in enumerate(samples, 1):
    label_path = ROOT / split / "labels" / f"{img_path.stem}.txt"

    img = cv2.imread(str(img_path))
    if img is None:
        continue

    h, w = img.shape[:2]
    count = 0

    if label_path.exists() and label_path.read_text().strip():
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 7:
                continue

            try:
                coords = list(map(float, parts[1:]))
                pts = []

                for j in range(0, len(coords), 2):
                    x = int(coords[j] * w)
                    y = int(coords[j+1] * h)
                    pts.append([x, y])

                if len(pts) >= 3:
                    polygon = cv2.UMat(
                        cv2.convexHull(
                            cv2.UMat(
                                __import__("numpy").array(pts, dtype="int32")
                            )
                        )
                    )
                    cv2.polylines(
                        img,
                        [polygon.get()],
                        True,
                        (0, 255, 0),
                        2
                    )
                    count += 1

            except Exception:
                continue

    # Add image info
    cv2.putText(
        img,
        f"{split.upper()} | polygons: {count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2
    )

    out_name = f"{i:02d}_{split}_{img_path.stem}.jpg"
    out_path = OUT / out_name
    cv2.imwrite(str(out_path), img)

    cards.append(
        f"""
        <div class="card">
            <div><b>{i:02d} | {split.upper()}</b> — {count} polygons</div>
            <img src="{html.escape(out_name)}">
        </div>
        """
    )

html_page = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Phase 1 Polygon Audit</title>
<style>
body {{
    font-family: Arial;
    margin: 20px;
}}
.grid {{
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 20px;
}}
.card {{
    border: 1px solid #ccc;
    padding: 10px;
}}
img {{
    width: 100%;
    margin-top: 8px;
}}
</style>
</head>
<body>
<h1>Phase 1 — Polygon Annotation Audit</h1>
<p>24 random samples | Green = annotated polygon</p>
<div class="grid">
{''.join(cards)}
</div>
</body>
</html>
"""

(OUT / "index.html").write_text(html_page)

print("=" * 60)
print("PHASE 1 VISUAL AUDIT GENERATED")
print("=" * 60)
print(f"Samples : {len(cards)}")
print(f"Output  : {OUT}")
print(f"Viewer  : {OUT / 'index.html'}")
