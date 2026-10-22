"""Interactive point-prompted segmentation for one warehouse image."""

import argparse
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import SAM


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = PROJECT_ROOT / "data" / "images" / "test.jpg"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "predictions"
DEFAULT_MODEL = "sam2.1_t.pt"


class PointPromptTool:
    def __init__(self, image: np.ndarray, model: SAM, output_dir: Path, device: str, bag_weight_kg: float):
        self.image = image
        self.model = model
        self.output_dir = output_dir
        self.device = device
        self.bag_weight_kg = bag_weight_kg
        self.points: list[tuple[int, int]] = []
        self.labels: list[int] = []
        self.overlay = image.copy()
        self.reference_mask: np.ndarray | None = None
        self.last_mask: np.ndarray | None = None
        self.window_name = "SAM grain bag segmentation"

    def _image_point(self, display_x: int, display_y: int) -> tuple[int, int]:
        window_x, window_y, window_width, window_height = cv2.getWindowImageRect(
            self.window_name
        )
        del window_x, window_y
        height, width = self.image.shape[:2]
        scale_x = width / max(window_width, 1)
        scale_y = height / max(window_height, 1)
        return (
            min(max(round(display_x * scale_x), 0), width - 1),
            min(max(round(display_y * scale_y), 0), height - 1),
        )

    def _redraw_points(self) -> None:
        self.overlay = self.image.copy()
        for (x, y), label in zip(self.points, self.labels):
            color = (0, 220, 0) if label == 1 else (0, 0, 255)
            cv2.circle(self.overlay, (x, y), 8, color, -1)
            cv2.circle(self.overlay, (x, y), 11, (255, 255, 255), 2)

    def add_point(self, display_x: int, display_y: int, label: int) -> None:
        point = self._image_point(display_x, display_y)
        self.points.append(point)
        self.labels.append(label)
        point_type = "positive" if label == 1 else "negative"
        print(f"Added {point_type} point: x={point[0]}, y={point[1]}")
        self._redraw_points()

    def clear_points(self) -> None:
        self.points.clear()
        self.labels.clear()
        self.overlay = self.image.copy()
        print("Cleared points. Start a new segmentation attempt.")

    def segment_and_save(self) -> None:
        if not self.points:
            print("Add at least one point before pressing Space.")
            return

        print(f"Running SAM with {len(self.points)} point(s): {self.points}")
        results = self.model.predict(
            source=self.image,
            points=self.points,
            labels=self.labels,
            device=self.device,
            save=False,
            verbose=False,
        )
        result = results[0]
        if result.masks is None or len(result.masks.data) == 0:
            print("SAM returned no mask for these prompts.")
            return

        mask = result.masks.data[0].cpu().numpy().astype(np.uint8)
        mask = cv2.resize(
            mask,
            (self.image.shape[1], self.image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        overlay = self.image.copy()
        overlay[mask == 1] = (0.55 * overlay[mask == 1] + 0.45 * np.array([0, 200, 255])).astype(
            np.uint8
        )
        self.overlay = overlay
        self.last_mask = mask
        self.reference_mask = mask

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        mask_path = self.output_dir / f"mask_{timestamp}.png"
        overlay_path = self.output_dir / f"segmentation_{timestamp}.jpg"
        cv2.imwrite(str(mask_path), mask * 255)
        cv2.imwrite(str(overlay_path), overlay)

        print(f"Mask shape: {mask.shape}, foreground pixels: {int(mask.sum())}")
        print(f"Saved mask: {mask_path}")
        print(f"Saved visualization: {overlay_path}")

    def _mask_features(self, mask: np.ndarray) -> tuple[int, float, float, np.ndarray]:
        area = int(mask.sum())
        ys, xs = np.where(mask > 0)
        if area == 0:
            return 0, 0.0, 0.0, np.zeros(3, dtype=np.float32)
        width = float(xs.max() - xs.min() + 1)
        height = float(ys.max() - ys.min() + 1)
        mean_color = self.image[mask == 1].mean(axis=0).astype(np.float32)
        return area, width, height, mean_color

    def find_similar_bags(self, bag_weight_kg: float) -> None:
        if self.reference_mask is None:
            print("First add a positive point and press Space to create a reference mask.")
            return

        print("Generating SAM2 candidate masks for the complete image...")
        results = self.model.predict(
            source=self.image,
            device=self.device,
            save=False,
            verbose=False,
        )
        result = results[0]
        if result.masks is None or len(result.masks.data) == 0:
            print("SAM2 did not return candidate masks.")
            return

        reference_area, reference_width, reference_height, reference_color = self._mask_features(
            self.reference_mask
        )
        reference_ratio = reference_width / max(reference_height, 1.0)
        candidates: list[tuple[np.ndarray, int, float]] = []
        for candidate_tensor in result.masks.data:
            candidate = candidate_tensor.cpu().numpy().astype(np.uint8)
            candidate = cv2.resize(
                candidate,
                (self.image.shape[1], self.image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            area, width, height, color = self._mask_features(candidate)
            if area == 0 or area < reference_area * 0.35 or area > reference_area * 2.5:
                continue
            ratio = width / max(height, 1.0)
            if abs(ratio - reference_ratio) > max(reference_ratio * 0.8, 0.7):
                continue
            color_distance = float(np.linalg.norm(color - reference_color))
            if color_distance > 90.0:
                continue
            center_hit = candidate[self.points[0][1], self.points[0][0]] if self.points else 0
            if center_hit:
                continue
            candidates.append((candidate, area, color_distance))

        masks = [self.reference_mask]
        for candidate, _, _ in sorted(candidates, key=lambda item: item[2]):
            if any(np.logical_and(candidate, existing).sum() / max(candidate.sum(), 1) > 0.6 for existing in masks):
                continue
            masks.append(candidate)

        output = self.image.copy()
        for index, mask in enumerate(masks, start=1):
            color = ((37 * index) % 255, (89 + 67 * index) % 255, (180 + 41 * index) % 255)
            output[mask == 1] = (0.55 * output[mask == 1] + 0.45 * np.array(color)).astype(np.uint8)
            ys, xs = np.where(mask > 0)
            if len(xs):
                cv2.putText(output, f"Bag {index}", (int(xs.min()), max(int(ys.min()) - 5, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        self.overlay = output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        count = len(masks)
        output_path = self.output_dir / f"inventory_{timestamp}.jpg"
        cv2.imwrite(str(output_path), output)
        print("\n===== WAREHOUSE INVENTORY RESULT =====")
        print(f"Visible Bags: {count}")
        print("Estimated Hidden Bags: not calculated from pixels")
        print(f"Estimated Total Bags: {count} visible only")
        print(f"Weight Per Bag: {bag_weight_kg:g} kg")
        print(f"Estimated Total Weight: {count * bag_weight_kg:g} kg")
        print("Confidence: heuristic candidate matching; validate manually")
        print(f"Saved inventory visualization: {output_path}")
        print("======================================\n")

    def mouse_callback(self, event: int, x: int, y: int, flags: int, _param: object) -> None:
        del flags
        if event == cv2.EVENT_LBUTTONDOWN:
            self.add_point(x, y, 1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.add_point(x, y, 0)

    def run(self) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        print("Controls: left click=positive, right click=negative, Space=segment/save")
        print("          a=find similar visible bags/count, c=clear, q/Esc=quit")
        while True:
            cv2.imshow(self.window_name, self.overlay)
            key = cv2.waitKey(30) & 0xFF
            if key == ord(" "):
                self.segment_and_save()
            elif key == ord("a"):
                self.find_similar_bags(self.bag_weight_kg)
            elif key == ord("c"):
                self.clear_points()
            elif key in (ord("q"), 27):
                break
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="SAM/SAM2 checkpoint, e.g. sam2.1_t.pt or sam2.1_b.pt",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default=None, help="cuda device such as 0, or cpu")
    parser.add_argument("--bag-weight-kg", type=float, default=50.0)
    parser.add_argument("--find-similar", action="store_true", help="Also run visible-bag candidate matching")
    parser.add_argument(
        "--points",
        nargs="+",
        metavar="X,Y,LABEL",
        help="Headless test mode, e.g. --points 500,300,1 700,300,0",
    )
    return parser.parse_args()


def parse_points(raw_points: list[str]) -> tuple[list[tuple[int, int]], list[int]]:
    points: list[tuple[int, int]] = []
    labels: list[int] = []
    for raw_point in raw_points:
        x_text, y_text, label_text = raw_point.split(",")
        label = int(label_text)
        if label not in (0, 1):
            raise ValueError(f"Point label must be 0 or 1: {raw_point}")
        points.append((int(x_text), int(y_text)))
        labels.append(label)
    return points, labels


def main() -> None:
    args = parse_args()
    image_path = args.image if args.image.is_absolute() else PROJECT_ROOT / args.image
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    device = args.device or ("0" if __import__("torch").cuda.is_available() else "cpu")
    print(f"Image: {image_path} ({image.shape[1]}x{image.shape[0]})")
    print(f"Loading SAM model: {args.model} on device {device}")
    model = SAM(args.model)

    if args.points:
        points, labels = parse_points(args.points)
        tool = PointPromptTool(image, model, output_dir, device, args.bag_weight_kg)
        tool.points = points
        tool.labels = labels
        tool.segment_and_save()
        if args.find_similar:
            tool.find_similar_bags(args.bag_weight_kg)
        return

    if not os.environ.get("DISPLAY"):
        raise RuntimeError(
            "Interactive mode needs a desktop display. Use --points X,Y,LABEL for a headless test."
        )
    PointPromptTool(image, model, output_dir, device, args.bag_weight_kg).run()


if __name__ == "__main__":
    main()