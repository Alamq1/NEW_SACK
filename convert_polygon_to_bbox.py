"""
Convert polygon labels to detection bounding boxes.

Polygons are in format: class_id x1 y1 x2 y2 x3 y3 ...
Convert to: class_id center_x center_y width height
"""

from pathlib import Path
import sys


def convert_polygon_to_bbox(polygon_line):
    """
    Convert a polygon annotation line to YOLO detection format.
    
    Input: class_id x1 y1 x2 y2 x3 y3 ... (normalized coordinates)
    Output: class_id center_x center_y width height
    """
    parts = polygon_line.strip().split()
    
    if len(parts) < 7:  # At least class_id + 3 points (x1 y1 x2 y2 x3 y3)
        print(f"WARNING: Skipping invalid line with {len(parts)} parts: {polygon_line[:50]}")
        return None
    
    class_id = parts[0]
    coords = list(map(float, parts[1:]))
    
    # Extract x and y coordinates
    x_coords = coords[0::2]  # Every other element starting from 0
    y_coords = coords[1::2]  # Every other element starting from 1
    
    if not x_coords or not y_coords:
        print(f"WARNING: Could not extract coordinates from: {polygon_line[:50]}")
        return None
    
    # Get bounding box
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)
    
    # Convert to center and width/height
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    
    # Clamp values to [0, 1]
    center_x = max(0, min(1, center_x))
    center_y = max(0, min(1, center_y))
    width = max(0, min(1, width))
    height = max(0, min(1, height))
    
    return f"{class_id} {center_x} {center_y} {width} {height}\n"


def convert_label_file(label_path):
    """Convert a single label file from polygon to bbox format."""
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        converted_lines = []
        for line in lines:
            if line.strip():  # Skip empty lines
                bbox_line = convert_polygon_to_bbox(line)
                if bbox_line:
                    converted_lines.append(bbox_line)
        
        # Write back
        with open(label_path, 'w') as f:
            f.writelines(converted_lines)
        
        return len(converted_lines)
    
    except Exception as e:
        print(f"ERROR processing {label_path}: {e}")
        return 0


def main():
    project_root = Path(__file__).resolve().parent
    
    # Process all dataset splits
    datasets = [
        project_root / "data" / "Rice Bag 10",
        project_root / "data" / "Rice Bag dataset",
    ]
    
    splits = ["train", "valid", "test"]
    
    total_files = 0
    total_lines = 0
    
    for dataset_path in datasets:
        if not dataset_path.exists():
            print(f"Skipping non-existent dataset: {dataset_path}")
            continue
        
        print(f"\n{'='*70}")
        print(f"Processing: {dataset_path.name}")
        print(f"{'='*70}")
        
        for split in splits:
            labels_dir = dataset_path / split / "labels"
            
            if not labels_dir.exists():
                print(f"  ⊘ {split}/labels not found")
                continue
            
            label_files = sorted(labels_dir.glob("*.txt"))
            
            if not label_files:
                print(f"  ⊘ No label files in {split}/labels")
                continue
            
            print(f"  → {split} ({len(label_files)} files):", end=" ", flush=True)
            
            lines_converted = 0
            for label_file in label_files:
                count = convert_label_file(label_file)
                lines_converted += count
            
            print(f"✓ {lines_converted} annotations")
            total_files += len(label_files)
            total_lines += lines_converted
    
    print(f"\n{'='*70}")
    print(f"Conversion Complete!")
    print(f"  • Files processed: {total_files}")
    print(f"  • Annotations converted: {total_lines}")
    print(f"  • Format: polygon (x1 y1 x2 y2...) → bbox (cx cy w h)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
