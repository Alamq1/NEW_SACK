"""
IMPROVED Polygon to BBox Conversion (V2 - SAFE VERSION)

Previous version had a bug that deleted files when conversion wasn't complete.
This version:
✓ Reads entire file first
✓ Preserves lines that are already in correct format  
✓ Only writes back when processing is complete
✓ Includes safety checks
✓ Detailed logging
"""

from pathlib import Path
from typing import Optional, Tuple


def is_valid_bbox_format(line: str) -> bool:
    """Check if a line is already in YOLO detection bbox format: class cx cy w h"""
    parts = line.strip().split()
    if len(parts) != 5:
        return False
    
    try:
        class_id = int(parts[0])  # Must be integer
        cx, cy, w, h = map(float, parts[1:])
        # Check ranges are reasonable (normalized coordinates)
        return 0 <= cx <= 1 and 0 <= cy <= 1 and 0 <= w <= 1 and 0 <= h <= 1
    except (ValueError, IndexError):
        return False


def is_polygon_format(line: str) -> bool:
    """Check if line is polygon format: class x1 y1 x2 y2 ... xn yn"""
    parts = line.strip().split()
    
    # Must have: class_id + at least 3 coordinate pairs (6 values total) = 7 minimum
    if len(parts) < 7:
        return False
    
    # Must have odd number of parts (class + even number of coordinates)
    if len(parts) % 2 == 0:
        return False
    
    try:
        int(parts[0])  # class_id must be integer
        coords = list(map(float, parts[1:]))
        # Check all coordinates are in valid range
        return all(0 <= c <= 1 for c in coords)
    except (ValueError, IndexError):
        return False


def convert_polygon_to_bbox(polygon_line: str) -> Optional[str]:
    """
    Convert a polygon annotation line to YOLO detection format.
    
    Input: class_id x1 y1 x2 y2 x3 y3 ... (normalized coordinates)
    Output: class_id center_x center_y width height
    
    Returns:
        Converted line if successful, None otherwise
    """
    if not is_polygon_format(polygon_line):
        return None
    
    parts = polygon_line.strip().split()
    class_id = parts[0]
    coords = list(map(float, parts[1:]))
    
    # Extract x and y coordinates
    x_coords = coords[0::2]  # Every other element starting from 0
    y_coords = coords[1::2]  # Every other element starting from 1
    
    if not x_coords or not y_coords:
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


def process_label_file(label_path: Path) -> Tuple[int, int, int]:
    """
    Process a single label file. SAFE VERSION that reads all before writing.
    
    Returns: (total_lines, converted_lines, preserved_lines)
    """
    try:
        # STEP 1: READ ALL LINES FIRST
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            return (0, 0, 0)
        
        # STEP 2: PROCESS LINES
        processed_lines = []
        converted_count = 0
        preserved_count = 0
        
        for line in lines:
            if not line.strip():  # Skip empty lines
                continue
            
            # Try to convert polygon to bbox
            converted = convert_polygon_to_bbox(line)
            if converted:
                processed_lines.append(converted)
                converted_count += 1
            elif is_valid_bbox_format(line):
                # Already in correct format - preserve it
                processed_lines.append(line)
                preserved_count += 1
            else:
                # Invalid format - skip with warning
                print(f"    ⚠️  Skipping invalid line: {line[:50]}...")
        
        # STEP 3: WRITE BACK (only if we have something to write)
        if processed_lines:
            with open(label_path, 'w') as f:
                f.writelines(processed_lines)
        else:
            # Don't delete the file if we couldn't process it
            print(f"    ❌ ERROR: No valid lines in {label_path.name}")
            return (len(lines), 0, 0)
        
        return (len(lines), converted_count, preserved_count)
    
    except Exception as e:
        print(f"    ❌ ERROR: {e}")
        return (0, 0, 0)


def main():
    """Main processing function."""
    project_root = Path(__file__).resolve().parent
    
    # Datasets to process
    datasets = [
        project_root / "data" / "Rice Bag 10",
        project_root / "data" / "Rice Bag dataset",
    ]
    
    splits = ["train", "valid", "test"]
    
    total_stats = {"files": 0, "lines": 0, "converted": 0, "preserved": 0}
    
    print(f"""
╔════════════════════════════════════════════════════════════════════╗
║  POLYGON → BBOX CONVERSION (V2 - SAFE VERSION)                     ║
╚════════════════════════════════════════════════════════════════════╝
""")
    
    for dataset_path in datasets:
        if not dataset_path.exists():
            print(f"⊘ Dataset not found: {dataset_path}")
            continue
        
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_path.name}")
        print(f"{'='*70}\n")
        
        for split in splits:
            labels_dir = dataset_path / split / "labels"
            
            if not labels_dir.exists():
                print(f"  ⊘ {split}/labels/ not found")
                continue
            
            label_files = sorted(labels_dir.glob("*.txt"))
            
            if not label_files:
                print(f"  ⊘ No label files in {split}/labels/")
                continue
            
            print(f"  📁 {split:8s} ({len(label_files):2d} files):", end=" ", flush=True)
            
            split_converted = 0
            split_preserved = 0
            split_total = 0
            
            for label_file in label_files:
                total, converted, preserved = process_label_file(label_file)
                split_total += total
                split_converted += converted
                split_preserved += preserved
            
            print(f"✓ {split_converted} converted + {split_preserved} preserved = {split_total} total")
            
            total_stats["files"] += len(label_files)
            total_stats["lines"] += split_total
            total_stats["converted"] += split_converted
            total_stats["preserved"] += split_preserved
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Files processed:      {total_stats['files']}")
    print(f"  Total annotations:    {total_stats['lines']}")
    print(f"  Converted (polygon):  {total_stats['converted']}")
    print(f"  Preserved (already):  {total_stats['preserved']}")
    print(f"  Format:               polygon (x1 y1 x2 y2...) → bbox (cx cy w h)")
    print(f"{'='*70}\n")
    
    print("✅ Conversion complete!\n")
    print("Next steps:")
    print("  1. Run: bash verify_labels.sh")
    print("  2. Check all files show: '✅ Format: DETECTION'")
    print("  3. Run: python src/training/train_rice_bags_v4.py\n")


if __name__ == "__main__":
    main()
