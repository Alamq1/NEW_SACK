# Context: `src/test_ui_v11_segmentation.py`

Handoff notes on what this file is, where it came from, and what was built into it. (Next steps are being covered separately.)

## What it is

A Flask tool (`http://localhost:5007`) that estimates warehouse pile dimensions from a single photo:

**Upload photo → V11 segmentation detects individual rice bags → user draws 2-point reference lines for Length / Breadth / Height → Total Sacks = L × B × H**

It's a **new, standalone file** — it does not modify the two files it borrows ideas from (`test_rice_bag_v11_segmentation_ui.py`, `warehouse_pile_estimator_ui.py`).

## Where it came from

| Piece | Borrowed from | What |
|---|---|---|
| Detection engine | `test_rice_bag_v11_segmentation_ui.py` | The trained V11 instance-segmentation model + its 2-stage duplicate-dedup logic |
| L/B/H interaction | `warehouse_pile_estimator_ui.py` | Click-2-points-on-a-line tool that counts bags along an axis, multiplies L×B×H |
| Visual design | `test_rice_bag_v11_segmentation_ui.py` | Dark-slate dashboard look (header/brand, dropzone upload, KPI cards, panel/details-grid) |

## Why segmentation instead of boxes for L/B/H

A box is a rectangle around a bag; a mask is the bag's actual pixel shape. Using masks:
- **Centroid** = the mask's true center (moment-based), accurate even for angled/partly-hidden bags — not just the midpoint of a box.
- **Size/bounding box** = tight, hugs the real bag shape — a rotated sack doesn't inflate the size estimate the way a box around it would.

## Development history (3 passes, in order)

**Pass 0 — built from scratch**
Core pipeline: V11 model → Stage A/B dedup (below) → mask-derived bag list (centroid, tight bbox, polygon) → click-2-point L/B/H tool → auto height-row hint.

**Pass 1 — UI restyle**
Swapped the old teal-on-navy canvas look for the dark dashboard system (header, dropzone, KPI cards, panels) to match the other segmentation UI.

**Pass 2 — two functional fixes**
1. **Confidence/IoU sliders restored.** An earlier revision had turned these into fixed read-only numbers; they're editable sliders again, sent with every `/detect` call.
2. **Line-crossing logic changed.** Originally a bag counted on an axis if its center point fell within some pixel *tolerance* of the drawn line. Now it counts only if the drawn line **geometrically intersects the bag's actual mask polygon** — literally "does the line pass through this bag's shape," same as a human counting by eye. The old tolerance-px control was removed because it no longer means anything under this method.

## Core algorithms

**Stage A — mask-overlap dedup** (`deduplicate_instances`)
Union-find merge of same-class detections whose *actual pixel masks* overlap heavily (IoU ≥ 0.45 or one mostly contained in the other ≥ 0.70). A cheap bounding-box pre-check skips pairs that can't possibly touch before doing the real per-pixel comparison. Keeps the highest-confidence detection per merged cluster. Fixes the case where the model fires two overlapping detections on one physical bag.

**Stage B — centroid-distance dedup** (`cluster_center_points`)
Runs on Stage A survivors. Merges same-class detections whose centroids are closer than 0.45× the typical bag size. Catches the rarer case where two detections on one bag *don't* overlap much (e.g. a bag split by a fold/shadow) but are obviously the same physical object.

**Auto height-row hint** (`estimate_height_rows`)
Sorts bag Y-centers, clusters them into rows using a gap threshold derived from median bag height, and reports a row count + a confidence score (based on how evenly spaced the row gaps are). This is a background hint only — it does **not** feed into the L×B×H total, which comes entirely from the user's drawn lines.

**Line-touches-mask test** (JS: `lineTouchesMask`, `segmentsIntersect`, `pointInPolygon`)
Each bag's mask outline is simplified server-side (`cv2.approxPolyDP`, capped at 40 points) and sent to the browser as a polygon. When the user finishes a 2-point line, the browser checks every bag: does the line segment cross any edge of that bag's polygon, or does either line endpoint land inside it? Only then does the bag count on that axis. The clicked segment is extended ~8% past each end so a slightly-short click at the pile's edge doesn't miss the last bag.

## Request flow

- `GET /` — renders the dashboard shell (confidence/IoU defaults injected).
- `POST /detect` — image + confidence + iou → runs `run_detection()` (predict → Stage A → Stage B → draw) → returns annotated image (base64), the deduped bag list (each with `id, cx, cy, x1/y1/x2/y2, conf, polygon`), and the auto height-row hint.
- All L/B/H line drawing, matching, and the running total are computed **entirely client-side** in JS using the `boxes` array from that one `/detect` response — no server round-trip per line.

## Current state

Fully working and tested end-to-end (both the detection endpoint and the mask-touch line logic were verified with real images before/after each change). This is the file's state as of the last change made.
