"""Manual multi-point SAM2.1 bag segmentation and visible inventory POC."""

import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import SAM


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = PROJECT_ROOT / "data" / "images" / "test.jpg"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "predictions"
DEFAULT_MODEL = PROJECT_ROOT / "sam2.1_t.pt"


class MultiPointInventory:
    def __init__(self, image: np.ndarray, model: SAM, output_dir: Path, device: str, weight_kg: float):
        self.image = image
        self.model = model
        self.output_dir = output_dir
        self.device = device
        self.weight_kg = weight_kg
        self.positive_points: list[tuple[int, int]] = []
        self.negative_points: list[tuple[int, int]] = []
        self.preview = image.copy()
        self.window_name = "SAM2.1 manual grain bag inventory"

    def to_image_point(self, window_x: int, window_y: int) -> tuple[int, int]:
        _, _, window_width, window_height = cv2.getWindowImageRect(self.window_name)
        image_height, image_width = self.image.shape[:2]
        scale_x = image_width / max(window_width, 1)
        scale_y = image_height / max(window_height, 1)
        return (
            min(max(round(window_x * scale_x), 0), image_width - 1),
            min(max(round(window_y * scale_y), 0), image_height - 1),
        )

    def redraw(self) -> None:
        self.preview = self.image.copy()
        for index, (x, y) in enumerate(self.positive_points, start=1):
            cv2.circle(self.preview, (x, y), 9, (0, 220, 0), -1)
            cv2.circle(self.preview, (x, y), 12, (255, 255, 255), 2)
            cv2.putText(self.preview, f"+{index}", (x + 12, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 0), 2, cv2.LINE_AA)
        for x, y in self.negative_points:
            cv2.drawMarker(self.preview, (x, y), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 24, 3)

    def mouse_callback(self, event: int, x: int, y: int, flags: int, _param: object) -> None:
        del flags
        point = self.to_image_point(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.positive_points.append(point)
            print(f"Positive point {len(self.positive_points)}: x={point[0]}, y={point[1]}")
            self.redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.negative_points.append(point)
            print(f"Negative point {len(self.negative_points)}: x={point[0]}, y={point[1]}")
            self.redraw()

    def clear_points(self) -> None:
        self.positive_points.clear()
        self.negative_points.clear()
        self.preview = self.image.copy()
        print("Cleared all points.")

    @staticmethod
    def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
        intersection = np.logical_and(first, second).sum()
        union = np.logical_or(first, second).sum()
        return float(intersection / max(union, 1))

    def segment_selected_bags(self) -> None:
        if not self.positive_points:
            print("Add positive points inside 10-15 bags before pressing Space.")
            return

        print(
            f"Running SAM2.1 with {len(self.positive_points)} positive and "
            f"{len(self.negative_points)} negative point(s)..."
        )
        prompt_points = []
        prompt_labels = []
        for positive_point in self.positive_points:
            prompt_points.append([positive_point, *self.negative_points])
            prompt_labels.append([1, *([0] * len(self.negative_points))])

        results = self.model.predict(
            source=self.image,
            points=prompt_points,
            labels=prompt_labels,
            device=self.device,
            save=False,
            verbose=False,
        )
        result = results[0]
        if result.masks is None or len(result.masks.data) == 0:
            print("SAM2.1 returned no masks. Try points closer to bag centers.")
            return

        image_height, image_width = self.image.shape[:2]
        masks: list[np.ndarray] = []
        for mask_tensor in result.masks.data:
            mask = mask_tensor.cpu().numpy().astype(np.uint8)
            mask = cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
            if mask.sum() < 100:
                continue
            if any(self.mask_iou(mask, existing) > 0.75 for existing in masks):
                continue
            masks.append(mask)

        if not masks:
            print("All masks were empty or duplicates. Try different points.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output = self.image.copy()
        mask_stack = np.zeros((len(masks), image_height, image_width), dtype=np.uint8)
        for index, mask in enumerate(masks, start=1):
            mask_stack[index - 1] = mask
            color = np.array(((47 * index) % 255, (97 * index) % 255, (173 * index) % 255))
            output[mask == 1] = (0.55 * output[mask == 1] + 0.45 * color).astype(np.uint8)
            ys, xs = np.where(mask == 1)
            if len(xs):
                cv2.putText(output, f"Bag {index}", (int(xs.min()), max(int(ys.min()) - 8, 22)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        visualization_path = self.output_dir / f"manual_inventory_{timestamp}.jpg"
        masks_path = self.output_dir / f"manual_masks_{timestamp}.npz"
        binary_path = self.output_dir / f"manual_masks_{timestamp}.png"
        cv2.imwrite(str(visualization_path), output)
        cv2.imwrite(str(binary_path), (np.any(mask_stack, axis=0).astype(np.uint8) * 255))
        np.savez_compressed(str(masks_path), masks=mask_stack, positive_points=np.array(self.positive_points),
                            negative_points=np.array(self.negative_points))
        self.preview = output

        visible_count = len(masks)
        print("\n========== MANUAL INVENTORY RESULT ==========")
        print(f"Positive points received: {len(self.positive_points)}")
        print(f"Negative points received: {len(self.negative_points)}")
        print(f"Visible bag masks accepted: {visible_count}")
        print("Estimated hidden bags: not calculated")
        print(f"Weight per bag: {self.weight_kg:g} kg")
        print(f"Visible estimated weight: {visible_count * self.weight_kg:g} kg")
        print("Note: count equals accepted manually prompted visible masks.")
        print(f"Saved visualization: {visualization_path}")
        print(f"Saved combined binary mask: {binary_path}")
        print(f"Saved individual masks and points: {masks_path}")
        print("=============================================\n")

    def run(self) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        print("Left click inside each bag; right click unwanted/background regions.")
        print("Press Space to segment/count, c to clear, q or Esc to quit.")
        while True:
            cv2.imshow(self.window_name, self.preview)
            key = cv2.waitKey(30) & 0xFF
            if key == ord(" "):
                self.segment_selected_bags()
            elif key == ord("c"):
                self.clear_points()
            elif key in (ord("q"), 27):
                break
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default=None, help="CUDA device such as 0, or cpu")
    parser.add_argument("--bag-weight-kg", type=float, default=50.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = args.image if args.image.is_absolute() else PROJECT_ROOT / args.image
    model_path = args.model if args.model.is_absolute() else PROJECT_ROOT / args.model
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"SAM2.1 checkpoint not found: {model_path}")
    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    print(f"Image: {image_path} ({image.shape[1]}x{image.shape[0]})")
    print(f"Loading SAM2.1 model: {model_path} on device {device}")
    model = SAM(str(model_path))
    MultiPointInventory(image, model, output_dir, device, args.bag_weight_kg).run()


if __name__ == "__main__":
    main()
