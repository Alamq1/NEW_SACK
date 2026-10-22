"""
RICE BAG V9 - DATASET VERIFICATION

Verifies that the V9 converted dataset is correct YOLO BBox format.

Data format must be:
    class_id center_x center_y width height

Example:
    0 0.500000 0.500000 0.200000 0.300000
"""

from pathlib import Path
import sys


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "data" / "Rice Bag 10_v9_converted"


# ============================================================
# SETTINGS
# ============================================================

SPLITS = ["train", "valid", "test"]


# ============================================================
# REPORT CLASS
# ============================================================

class DatasetReport:
    def __init__(self):
        self.splits = {}
        self.issues = []
    
    def add_split(self, split, images_count, labels_count, bbox_count, empty_count, malformed_count):
        self.splits[split] = {
            "images": images_count,
            "labels": labels_count,
            "bboxes": bbox_count,
            "empty": empty_count,
            "malformed": malformed_count,
        }
    
    def add_issue(self, issue):
        self.issues.append(issue)
    
    def print_report(self):
        print("\n" + "=" * 70)
        print("RICE BAG V9 - DATASET VERIFICATION REPORT")
        print("=" * 70)
        
        print(f"\nDataset: {DATASET_ROOT}")
        
        # Check data.yaml
        yaml_path = DATASET_ROOT / "data.yaml"
        if yaml_path.exists():
            print(f"✓ data.yaml found")
        else:
            print(f"✗ data.yaml NOT found")
            self.add_issue("data.yaml missing")
        
        # Split reports
        for split in SPLITS:
            if split in self.splits:
                data = self.splits[split]
                print(f"\n{'-' * 70}")
                print(f"{split.upper()}")
                print(f"{'-' * 70}")
                print(f"  Images    : {data['images']}")
                print(f"  Labels    : {data['labels']}")
                print(f"  BBoxes    : {data['bboxes']}")
                print(f"  Empty     : {data['empty']}")
                print(f"  Malformed : {data['malformed']}")
        
        # Issues
        if self.issues:
            print(f"\n{'-' * 70}")
            print("ISSUES DETECTED")
            print(f"{'-' * 70}")
            for issue in self.issues:
                print(f"✗ {issue}")
        else:
            print(f"\n{'-' * 70}")
            print("✓ NO ISSUES DETECTED")
            print(f"{'-' * 70}")
        
        print("\n" + "=" * 70)
        
        # Final verdict
        if self.issues:
            print("VERDICT: DATASET HAS ISSUES - DO NOT TRAIN")
            print("=" * 70)
            return False
        else:
            print("VERDICT: DATASET OK - READY TO TRAIN")
            print("=" * 70)
            return True


# ============================================================
# VERIFY SPLIT
# ============================================================

def verify_split(split):
    """Verify a single train/valid/test split."""
    
    images_dir = DATASET_ROOT / split / "images"
    labels_dir = DATASET_ROOT / split / "labels"
    
    report = {
        "images_count": 0,
        "labels_count": 0,
        "bbox_count": 0,
        "empty_count": 0,
        "malformed_count": 0,
    }
    
    issues = []
    
    # Check directories exist
    if not images_dir.exists():
        issues.append(f"{split}/images does not exist")
        return report, issues
    
    if not labels_dir.exists():
        issues.append(f"{split}/labels does not exist")
        return report, issues
    
    # Count images
    image_files = set()
    for img_path in images_dir.glob("*"):
        if img_path.is_file():
            image_files.add(img_path.stem)
            report["images_count"] += 1
    
    # Verify labels
    label_files = set()
    for label_path in labels_dir.glob("*.txt"):
        if label_path.is_file():
            label_files.add(label_path.stem)
            report["labels_count"] += 1
            
            # Check label format
            with open(label_path, "r") as f:
                lines = f.readlines()
            
            if len(lines) == 0:
                report["empty_count"] += 1
                continue
            
            for line_num, line in enumerate(lines, start=1):
                parts = line.strip().split()
                
                if not parts:
                    continue
                
                if len(parts) != 5:
                    malformed_line = f"{label_path.name}:{line_num} has {len(parts)} values (expected 5)"
                    issues.append(malformed_line)
                    report["malformed_count"] += 1
                    continue
                
                try:
                    class_id = int(parts[0])
                    center_x = float(parts[1])
                    center_y = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Check value ranges
                    if not (0 <= center_x <= 1.0):
                        issues.append(f"{label_path.name}:{line_num} center_x out of range: {center_x}")
                    if not (0 <= center_y <= 1.0):
                        issues.append(f"{label_path.name}:{line_num} center_y out of range: {center_y}")
                    if not (0 <= width <= 1.0):
                        issues.append(f"{label_path.name}:{line_num} width out of range: {width}")
                    if not (0 <= height <= 1.0):
                        issues.append(f"{label_path.name}:{line_num} height out of range: {height}")
                    
                    report["bbox_count"] += 1
                
                except ValueError as e:
                    issues.append(f"{label_path.name}:{line_num} non-numeric value: {e}")
                    report["malformed_count"] += 1
    
    # Check for mismatches
    images_without_labels = image_files - label_files
    labels_without_images = label_files - image_files
    
    if images_without_labels:
        for img_name in sorted(images_without_labels)[:5]:  # Show first 5
            issues.append(f"{split}: image '{img_name}' has no label")
        if len(images_without_labels) > 5:
            issues.append(f"{split}: ... and {len(images_without_labels) - 5} more images without labels")
    
    if labels_without_images:
        for label_name in sorted(labels_without_images)[:5]:
            issues.append(f"{split}: label '{label_name}' has no image")
        if len(labels_without_images) > 5:
            issues.append(f"{split}: ... and {len(labels_without_images) - 5} more labels without images")
    
    return report, issues


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("RICE BAG V9 - DATASET VERIFICATION")
    print("=" * 70)
    
    print(f"\nDataset: {DATASET_ROOT}")
    
    if not DATASET_ROOT.exists():
        print(f"\n✗ DATASET NOT FOUND: {DATASET_ROOT}")
        sys.exit(1)
    
    print("✓ Dataset directory found")
    
    # Verify each split
    report = DatasetReport()
    all_issues = []
    
    for split in SPLITS:
        print(f"\nVerifying {split}...")
        split_report, split_issues = verify_split(split)
        report.add_split(
            split,
            split_report["images_count"],
            split_report["labels_count"],
            split_report["bbox_count"],
            split_report["empty_count"],
            split_report["malformed_count"],
        )
        all_issues.extend(split_issues)
    
    # Add all issues to report
    for issue in all_issues:
        report.add_issue(issue)
    
    # Print final report
    is_valid = report.print_report()
    
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
