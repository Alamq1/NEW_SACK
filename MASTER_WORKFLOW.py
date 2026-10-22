#!/usr/bin/env python3
"""
MASTER WORKFLOW - Complete End-to-End Execution

This script runs the complete workflow:
1. ✓ Verify dataset labels exist
2. ✓ Convert polygon labels to bbox format (safe)
3. ✓ Train model with corrected labels
4. ✓ Generate summary and next steps

Run: python MASTER_WORKFLOW.py
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime


def print_header(title):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def print_step(step_num, title, description=""):
    """Print step header"""
    print(f"\n{'─'*80}")
    print(f"STEP {step_num}: {title}")
    print(f"{'─'*80}")
    if description:
        print(f"{description}\n")


def run_command(cmd, description):
    """Run a shell command and handle errors"""
    print(f"▶ Running: {description}")
    print(f"  Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\n❌ ERROR in {description}")
        print(f"Exit code: {result.returncode}")
        return False
    
    print(f"\n✅ Completed: {description}")
    return True


def verify_labels():
    """Step 1: Verify labels exist and check format"""
    print_step(1, "VERIFY LABELS", "Check if label files exist and have content")
    
    project_root = Path(__file__).resolve().parent
    label_dirs = [
        project_root / "data/Rice Bag 10/train/labels",
        project_root / "data/Rice Bag 10/valid/labels",
        project_root / "data/Rice Bag dataset/train/labels",
        project_root / "data/Rice Bag dataset/valid/labels",
        project_root / "data/Rice Bag dataset/test/labels",
    ]
    
    total_files = 0
    total_size = 0
    empty_files = 0
    
    for label_dir in label_dirs:
        if not label_dir.exists():
            print(f"  ⚠️  Missing: {label_dir.name}/")
            continue
        
        label_files = list(label_dir.glob("*.txt"))
        if not label_files:
            continue
        
        dir_size = sum(f.stat().st_size for f in label_files)
        empty_count = sum(1 for f in label_files if f.stat().st_size == 0)
        
        total_files += len(label_files)
        total_size += dir_size
        empty_files += empty_count
        
        status = "✅" if empty_count == 0 else "⚠️"
        print(f"  {status} {label_dir.parent.name:20s}: {len(label_files):3d} files, "
              f"{dir_size/1024:.1f}KB ({empty_count} empty)")
    
    print(f"\n  Total: {total_files} files, {total_size/1024:.1f}KB")
    
    if empty_files > 0:
        print(f"\n  ❌ ERROR: {empty_files} empty label files found!")
        print(f"  These need to be restored from RoboFlow first.\n")
        print(f"  Quick fix:")
        print(f"    1. Download from: https://universe.roboflow.com/ovais-alam-qureshi/rice-bag-10")
        print(f"    2. Extract YOLO format to: data/Rice Bag 10/")
        print(f"    3. Run this script again\n")
        return False
    
    if total_size == 0:
        print(f"\n  ❌ ERROR: No labels found!")
        return False
    
    print(f"\n  ✅ Labels verified - ready for conversion")
    return True


def convert_labels():
    """Step 2: Convert polygon labels to bbox format"""
    print_step(2, "CONVERT LABELS", "Convert polygon format → bbox format (safe)")
    
    project_root = Path(__file__).resolve().parent
    conversion_script = project_root / "convert_polygon_to_bbox_v2.py"
    
    if not conversion_script.exists():
        print(f"  ❌ ERROR: Conversion script not found: {conversion_script}")
        return False
    
    cmd = f"cd {project_root} && python convert_polygon_to_bbox_v2.py"
    return run_command(cmd, "Label conversion (polygon → bbox)")


def train_model():
    """Step 3: Train the model"""
    print_step(3, "TRAIN MODEL", "Train YOLO11m with corrected labels")
    
    project_root = Path(__file__).resolve().parent
    training_script = project_root / "src/training/train_rice_bags_v2.py"
    
    if not training_script.exists():
        print(f"  ❌ ERROR: Training script not found: {training_script}")
        return False
    
    cmd = f"cd {project_root} && python {training_script}"
    return run_command(cmd, "Model training (v2)")


def main():
    """Main workflow execution"""
    print_header("🚀 RICE BAG DETECTION - MASTER WORKFLOW")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    project_root = Path(__file__).resolve().parent
    
    # Step 1: Verify labels
    print("\n")
    if not verify_labels():
        print_header("❌ WORKFLOW STOPPED - LABEL ISSUE")
        print("Please restore labels from RoboFlow before running this script.")
        print("\nRecovery steps:")
        print("  1. python recover_dataset.py  (Shows download options)")
        print("  2. Download from RoboFlow")
        print("  3. Extract to data/ folder")
        print("  4. Run this script again\n")
        sys.exit(1)
    
    # Step 2: Convert labels
    print("\n")
    if not convert_labels():
        print_header("❌ WORKFLOW STOPPED - CONVERSION ERROR")
        print("Label conversion failed. Check the error messages above.")
        sys.exit(1)
    
    # Step 3: Train model
    print("\n")
    if not train_model():
        print_header("❌ WORKFLOW STOPPED - TRAINING ERROR")
        print("Model training failed. Check the error messages above.")
        sys.exit(1)
    
    # Success
    print_header("✅ WORKFLOW COMPLETED SUCCESSFULLY")
    
    print("""
╔════════════════════════════════════════════════════════════════════╗
║                   TRAINING COMPLETE!                              ║
╚════════════════════════════════════════════════════════════════════╝

What's been done:
  ✅ Labels verified
  ✅ Polygon labels converted to bbox format
  ✅ Model trained with corrected labels
  ✅ Best weights saved

Next Steps - TEST YOUR MODEL:

1. Run the web testing UI:
   
   python src/test_rice_bag_v2_ui.py
   
   Then open in browser:
   http://127.0.0.1:5000

2. Upload an image and check predictions
   (You should see boxes aligned with rice bags!)

3. Check training results:
   outputs/rice_bag_detection_v2/results.csv
   
   Look for:
   ✓ Epoch 1: Precision > 0 (not 0!)
   ✓ Epoch 1: Recall > 0 (not 0!)
   ✓ Epoch 1: mAP50 > 0 (not 0!)
   
   If these are > 0, your model is LEARNING! 🎉

4. Compare with v2 (before fix):
   - Before: Precision=0, Recall=0, mAP50=0 ❌
   - After: Precision>0, Recall>0, mAP50>0 ✅

═══════════════════════════════════════════════════════════════════════

TROUBLESHOOTING:

If metrics are still 0 after this:
  • Check: outputs/rice_bag_detection_v2/results.csv
  • Look for first epoch with non-zero metrics
  • Read: README_SOLUTION.md (detailed troubleshooting)

If predictions look wrong:
  • Check: outputs/rice_bag_detection_v2/val_batch0_pred.jpg
  • Compare with: outputs/rice_bag_detection_v2/val_batch0_labels.jpg
  • Read: COMPLETE_PROBLEM_ANALYSIS.md (detailed analysis)

═══════════════════════════════════════════════════════════════════════

FILES CREATED/USED:

Input Files:
  ✓ data/Rice Bag 10/train/images + labels
  ✓ data/Rice Bag 10/valid/images + labels
  ✓ data/Rice Bag 10/data.yaml
  ✓ convert_polygon_to_bbox_v2.py (safe conversion)
  ✓ src/training/train_rice_bags_v2.py (training)

Output Files:
  ✓ outputs/rice_bag_detection_v2/weights/best.pt (best model)
  ✓ outputs/rice_bag_detection_v2/results.csv (metrics)
  ✓ outputs/rice_bag_detection_v2/val_batch0_labels.jpg (ground truth)
  ✓ outputs/rice_bag_detection_v2/val_batch0_pred.jpg (predictions)

═══════════════════════════════════════════════════════════════════════

Ready to test? Run:

   python src/test_rice_bag_v2_ui.py

═══════════════════════════════════════════════════════════════════════
""")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == "__main__":
    main()
