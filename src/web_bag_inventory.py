"""Local browser UI for manual SAM2.1 multi-point bag segmentation."""

import argparse
import base64
import json
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import SAM


ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATH = ROOT / "data" / "images" / "test.jpg"
MODEL_PATH = ROOT / "sam2.1_l.pt"
OUTPUT_DIR = ROOT / "outputs" / "predictions"

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Grain Bag Inventory | SAM2.1</title>
<style>
:root{--ink:#17211b;--muted:#66736a;--line:#d7ded8;--paper:#f4f6f1;--accent:#d97842;--green:#217a58;--red:#bd3f45}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:Georgia,'Times New Roman',serif}
header{padding:25px 5vw 18px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;align-items:end;gap:20px;background:#fafbf8}
h1{font-size:clamp(25px,4vw,45px);font-weight:500;letter-spacing:0;margin:0}.eyebrow{font:700 11px system-ui,sans-serif;letter-spacing:2px;text-transform:uppercase;color:var(--accent);margin-bottom:8px}.status{font:13px system-ui,sans-serif;color:var(--muted)}
main{max-width:1440px;margin:28px auto;padding:0 5vw;display:grid;grid-template-columns:minmax(0,1fr) 310px;gap:28px}.stage{background:#18221d;border:1px solid #26352c;box-shadow:0 14px 35px #26352c1c;min-height:400px;display:grid;place-items:center;overflow:auto}.canvas-wrap{position:relative;line-height:0;max-width:100%}canvas{display:block;max-width:100%;height:auto;cursor:crosshair}.panel{background:#fff;border:1px solid var(--line);padding:22px;align-self:start}.panel h2{font-size:21px;font-weight:500;margin:0 0 18px}.help{font:13px/1.45 system-ui,sans-serif;color:var(--muted);margin:0 0 19px}.legend{display:grid;gap:9px;margin:16px 0 24px;font:13px system-ui,sans-serif}.dot{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:7px}.positive{background:#28c978}.negative{background:#e4535e}.row{display:flex;align-items:center;justify-content:space-between;gap:12px;margin:12px 0;font:13px system-ui,sans-serif}.row input{width:90px;padding:8px;border:1px solid var(--line);font:inherit}button{width:100%;padding:12px 14px;border:0;background:var(--green);color:white;font:700 13px system-ui,sans-serif;cursor:pointer;margin-top:8px}button.secondary{background:#e9eee9;color:var(--ink)}button:disabled{opacity:.45;cursor:wait}.counts{border-top:1px solid var(--line);margin-top:20px;padding-top:16px;font:13px/1.8 system-ui,sans-serif}.counts strong{font-size:21px;color:var(--green)}.results{grid-column:1/-1;background:#fff;border:1px solid var(--line);padding:18px 22px;display:none}.results h2{margin:0 0 8px;font-size:21px;font-weight:500}.results p{font:14px system-ui,sans-serif;margin:5px 0;color:var(--muted)}.results a{color:var(--green)}
@media(max-width:900px){header{align-items:start;flex-direction:column}main{grid-template-columns:1fr;margin-top:18px}.panel{order:-1}.stage{min-height:250px}}
</style></head>
<body>
<header><div><div class="eyebrow">Warehouse grain bag POC</div><h1>Manual SAM2.1 inventory</h1></div><div id="status" class="status">Loading model...</div></header>
<main><section class="stage"><div class="canvas-wrap"><canvas id="canvas"></canvas></div></section>
<aside class="panel"><h2>Prompt the image</h2><p class="help">Click once inside each visible bag. Use right-click on nearby bags or background that should be excluded. Then generate masks.</p>
<div class="legend"><div><span class="dot positive"></span>Left click: positive bag point</div><div><span class="dot negative"></span>Right click: negative point</div></div>
<div class="row"><span>Positive points</span><strong id="positiveCount">0</strong></div><div class="row"><span>Negative points</span><strong id="negativeCount">0</strong></div>
<label class="row"><span>Weight per bag (kg)</span><input id="weight" type="number" min="0.1" step="0.1" value="50"></label>
<button id="generate">Generate masks and count</button><button id="clear" class="secondary">Clear all points</button>
<label class="row"><span>Find matching visible bags</span><input id="match" type="checkbox" checked style="width:18px"></label>
<div class="counts"><div>Expected workflow: 10-15 positive points</div><div>Matching mode includes all usable SAM2 candidates.</div></div></aside>
<section id="results" class="results"><h2>Latest result</h2><p id="resultText"></p><p id="resultFiles"></p></section></main>
<script>
const canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d'),image=new Image();let positives=[],negatives=[],resultImage=null;
image.src='/image';image.onload=()=>{canvas.width=image.naturalWidth;canvas.height=image.naturalHeight;draw()};
function draw(){ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(resultImage||image,0,0);positives.forEach((p,i)=>{ctx.beginPath();ctx.arc(p[0],p[1],11,0,Math.PI*2);ctx.fillStyle='#28c978';ctx.fill();ctx.strokeStyle='white';ctx.lineWidth=3;ctx.stroke();ctx.fillStyle='#123';ctx.font='bold 16px system-ui';ctx.fillText('+'+(i+1),p[0]+14,p[1]-10)});negatives.forEach(p=>{ctx.beginPath();ctx.moveTo(p[0]-12,p[1]-12);ctx.lineTo(p[0]+12,p[1]+12);ctx.moveTo(p[0]+12,p[1]-12);ctx.lineTo(p[0]-12,p[1]+12);ctx.strokeStyle='#e4535e';ctx.lineWidth=4;ctx.stroke()})}
function point(e){const r=canvas.getBoundingClientRect();return [Math.round((e.clientX-r.left)*canvas.width/r.width),Math.round((e.clientY-r.top)*canvas.height/r.height)]}
canvas.addEventListener('click',e=>{positives.push(point(e));update()});canvas.addEventListener('contextmenu',e=>{e.preventDefault();negatives.push(point(e));update()});
function update(){document.getElementById('positiveCount').textContent=positives.length;document.getElementById('negativeCount').textContent=negatives.length;draw()}
document.getElementById('clear').onclick=()=>{positives=[];negatives=[];resultImage=null;document.getElementById('results').style.display='none';update()};
document.getElementById('generate').onclick=async()=>{if(!positives.length){alert('Add at least one positive point inside a bag.');return}const b=document.getElementById('generate');b.disabled=true;b.textContent='Running SAM2.1...';try{const r=await fetch('/segment',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({positive_points:positives,negative_points:negatives,weight_kg:Number(document.getElementById('weight').value),match_similar:document.getElementById('match').checked})});const d=await r.json();if(!r.ok)throw Error(d.error);resultImage=new Image();resultImage.onload=draw;resultImage.src='data:image/jpeg;base64,'+d.image_base64;document.getElementById('resultText').textContent=`Visible masks: ${d.count} | Weight per bag: ${d.weight_kg} kg | Visible estimated weight: ${d.total_weight_kg} kg`;document.getElementById('resultFiles').innerHTML=`Saved visualization: <a href="${d.visualization_url}" target="_blank">${d.visualization_name}</a> | Masks: <a href="${d.masks_url}" target="_blank">${d.masks_name}</a>`;document.getElementById('results').style.display='block';document.getElementById('status').textContent='Ready';}catch(e){alert(e.message)}finally{b.disabled=false;b.textContent='Generate masks and count'}};
</script></body></html>"""


def image_data_uri(path: Path) -> bytes:
    return path.read_bytes()


class InventoryService:
    def __init__(self) -> None:
        self.image = cv2.imread(str(IMAGE_PATH))
        if self.image is None:
            raise FileNotFoundError(IMAGE_PATH)
        if not MODEL_PATH.exists():
            raise FileNotFoundError(MODEL_PATH)
        self.device = "0" if torch.cuda.is_available() else "cpu"
        print(f"Loading SAM2.1: {MODEL_PATH} on device {self.device}")
        self.model = SAM(str(MODEL_PATH))
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    @staticmethod
    def iou(first: np.ndarray, second: np.ndarray) -> float:
        intersection = np.logical_and(first, second).sum()
        union = np.logical_or(first, second).sum()
        return float(intersection / max(union, 1))

    @staticmethod
    def contains_negative(mask: np.ndarray, negative: list[list[int]]) -> bool:
        if not negative:
            return False
        exclusion = np.zeros_like(mask, dtype=np.uint8)
        for point in negative:
            if 0 <= point[0] < mask.shape[1] and 0 <= point[1] < mask.shape[0]:
                cv2.circle(exclusion, tuple(point), 45, 1, -1)
        return bool(np.logical_and(mask, exclusion).any())

    def segment(self, positive: list[list[int]], negative: list[list[int]], weight: float, match_similar: bool) -> dict:
        if not positive:
            raise ValueError("At least one positive point is required")
        if weight <= 0:
            raise ValueError("Weight must be greater than zero")
        prompt_points = [[point, *negative] for point in positive]
        prompt_labels = [[1, *([0] * len(negative))] for _ in positive]
        with self.lock:
            results = self.model.predict(source=self.image, points=prompt_points, labels=prompt_labels,
                                         device=self.device, save=False, verbose=False)
        result = results[0]
        if result.masks is None:
            raise RuntimeError("SAM2.1 returned no masks")
        height, width = self.image.shape[:2]
        masks: list[np.ndarray] = []
        for tensor in result.masks.data:
            mask = cv2.resize(tensor.cpu().numpy().astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
            if (mask.sum() >= 100 and not self.contains_negative(mask, negative)
                    and not any(self.iou(mask, old) > 0.75 for old in masks)):
                masks.append(mask)
        if match_similar:
            print("Generating all usable automatic SAM2 masks...")
            automatic_results = self.model.predict(source=self.image, device=self.device, save=False, verbose=False)
            automatic = automatic_results[0]
            if automatic.masks is not None:
                for tensor in automatic.masks.data:
                    candidate = cv2.resize(tensor.cpu().numpy().astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
                    if candidate.sum() < 100:
                        continue
                    if self.contains_negative(candidate, negative):
                        continue
                    if any(self.iou(candidate, old) > 0.60 for old in masks):
                        continue
                    masks.append(candidate)
        if not masks:
            raise RuntimeError("No usable masks returned. Try points closer to bag centers.")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output = self.image.copy()
        stack = np.zeros((len(masks), height, width), dtype=np.uint8)
        for index, mask in enumerate(masks, start=1):
            stack[index - 1] = mask
            color = np.array(((47 * index) % 255, (97 * index) % 255, (173 * index) % 255))
            output[mask == 1] = (0.55 * output[mask == 1] + 0.45 * color).astype(np.uint8)
            ys, xs = np.where(mask == 1)
            cv2.putText(output, f"Bag {index}", (int(xs.min()), max(int(ys.min()) - 8, 22)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        visualization = OUTPUT_DIR / f"web_inventory_{timestamp}.jpg"
        masks_file = OUTPUT_DIR / f"web_masks_{timestamp}.npz"
        cv2.imwrite(str(visualization), output)
        np.savez_compressed(str(masks_file), masks=stack, positive_points=np.array(positive), negative_points=np.array(negative))
        ok, encoded = cv2.imencode(".jpg", output)
        if not ok:
            raise RuntimeError("Could not encode result image")
        return {"count": len(masks), "weight_kg": weight, "total_weight_kg": len(masks) * weight,
                "image_base64": base64.b64encode(encoded).decode("ascii"),
                "visualization_url": "/output/" + visualization.name, "visualization_name": visualization.name,
                "masks_url": "/output/" + masks_file.name, "masks_name": masks_file.name}


SERVICE = None


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/":
            self.respond(200, "text/html; charset=utf-8", HTML.encode())
        elif self.path == "/image":
            self.respond(200, "image/jpeg", image_data_uri(IMAGE_PATH))
        elif self.path.startswith("/output/"):
            path = OUTPUT_DIR / Path(self.path.removeprefix("/output/")).name
            if path.exists() and path.parent == OUTPUT_DIR:
                content_type = "image/jpeg" if path.suffix == ".jpg" else "application/octet-stream"
                self.respond(200, content_type, path.read_bytes())
            else:
                self.respond(404, "text/plain", b"Not found")
        else:
            self.respond(404, "text/plain", b"Not found")

    def do_POST(self) -> None:
        if self.path != "/segment":
            self.respond(404, "application/json", b'{"error":"Not found"}')
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length))
            result = SERVICE.segment(data["positive_points"], data.get("negative_points", []), float(data.get("weight_kg", 50)), bool(data.get("match_similar", True)))
            self.respond(200, "application/json", json.dumps(result).encode())
        except Exception as error:
            self.respond(400, "application/json", json.dumps({"error": str(error)}).encode())

    def respond(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        print(format % args)


def main() -> None:
    global SERVICE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    SERVICE = InventoryService()
    port = args.port
    while True:
        try:
            server = ThreadingHTTPServer((args.host, port), Handler)
            break
        except OSError as error:
            if error.errno != 98:
                raise
            print(f"Port {port} is already in use; trying port {port + 1}")
            port += 1
    print(f"Open http://{args.host}:{port} in your browser")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
