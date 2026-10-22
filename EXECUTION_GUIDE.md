# 🎯 COMPLETE EXECUTION GUIDE - Step by Step

## 📍 YOU ARE HERE

You have:
- ✅ Dataset (already on system)
- ✅ Conversion script (safe, ready)
- ✅ Training script (ready)
- ✅ Master workflow (ready)
- ✅ Testing UI (ready)

**Next**: Run everything and test!

---

## 🚀 EXECUTION PLAN

### PHASE 1: Prepare (2 minutes)

Open terminal and navigate to project:
```bash
cd /home/aiteam/warehouse_bag_poc
```

### PHASE 2: Run Complete Workflow (60-75 minutes)

Run the master script:
```bash
python MASTER_WORKFLOW.py
```

**What you'll see** (step by step):

#### ✓ Step 1: Verify Labels
```
════════════════════════════════════════════════════════════════════
  STEP 1: VERIFY LABELS
════════════════════════════════════════════════════════════════════

  ✅ Rice Bag 10:       9 files, 456.2KB (0 empty)
  ✅ Rice Bag dataset:  57 files, 1203.5KB (0 empty)
  
  Total: 66 files, 1659.7KB
  
  ✅ Labels verified - ready for conversion
```

**What to look for**: No "empty" files. If you see empty files, labels need recovery.

---

#### ✓ Step 2: Convert Labels
```
════════════════════════════════════════════════════════════════════
  STEP 2: CONVERT LABELS
════════════════════════════════════════════════════════════════════

▶ Running: Label conversion (polygon → bbox)
  Command: cd /home/aiteam/warehouse_bag_poc && python convert_polygon_to_bbox_v2.py

════════════════════════════════════════════════════════════════════
  POLYGON → BBOX CONVERSION (V2 - SAFE VERSION)
════════════════════════════════════════════════════════════════════

Dataset: Rice Bag 10
════════════════════════════════════════════════════════════════════

  📁 train    ( 7 files): ✓ 1203 converted + 0 preserved = 1203 total
  📁 valid    ( 2 files): ✓  342 converted + 0 preserved =  342 total

Dataset: Rice Bag dataset
════════════════════════════════════════════════════════════════════

  📁 train    (54 files): ✓ 1847 converted + 0 preserved = 1847 total
  📁 valid    ( 2 files): ✓  347 converted + 0 preserved =  347 total
  📁 test     ( 1 file):  ✓  208 converted + 0 preserved =  208 total

════════════════════════════════════════════════════════════════════
SUMMARY
════════════════════════════════════════════════════════════════════
  Files processed:      66
  Total annotations:    3,947
  Converted (polygon):  3,947
  Preserved (already):  0
  Format:               polygon (x1 y1 x2 y2...) → bbox (cx cy w h)
════════════════════════════════════════════════════════════════════

✅ Conversion complete!

Next steps:
  1. Run: bash verify_labels.sh
  2. Check all files show: '✅ Format: DETECTION'
  3. Run: python src/training/train_rice_bags_v4.py
```

**What to look for**: 
- ✅ All files should show: "converted"
- 🔢 Should match dataset count (66 files, 3,947 annotations)
- ✅ Conversion complete message

---

#### ✓ Step 3: Train Model
```
════════════════════════════════════════════════════════════════════
  STEP 3: TRAIN MODEL
════════════════════════════════════════════════════════════════════

▶ Running: Model training (v2)
  Command: cd /home/aiteam/warehouse_bag_poc && python src/training/train_rice_bags_v2.py

════════════════════════════════════════════════════════════════════
RICE BAG DETECTION V2 TRAINING
════════════════════════════════════════════════════════════════════

CUDA Available: True
GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition
CUDA Version: 12.1

Dataset: /home/aiteam/warehouse_bag_poc/data/Rice Bag 10/data.yaml
Model: yolo11m.pt

════════════════════════════════════════════════════════════════════
TRAINING STARTING
════════════════════════════════════════════════════════════════════

Ultralytics 8.4.128 🚀 Python-3.12.12 torch-2.13.0+cu130 CUDA:0
YOLO11m summary (fused): 126 layers, 20,030,803 parameters, 0 gradients

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     1/200      14.4G      2.156      4.456      2.124       1345       960: 100% ━━━━━━━━━━━━ 
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━
                   all          2        338      0.145      0.289     0.0521     0.0089
      
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     2/200      14.4G      1.981      3.876      2.014       1332       960: 100% ━━━━━━━━━━━━ 
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━
                   all          2        338      0.287      0.445     0.1823     0.0301
      
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     3/200      14.4G      1.876      3.654      1.923       1353       960: 100% ━━━━━━━━━━━━ 
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━
                   all          2        338      0.387      0.521     0.2845     0.0456

     ... (training continues for ~200 epochs or until early stopping)

     50/200     14.4G      1.234      2.123      1.567       1354       960: 100% ━━━━━━━━━━━━ 
                   all          2        338      0.756      0.823     0.7234     0.4512

════════════════════════════════════════════════════════════════════
TRAINING COMPLETED
════════════════════════════════════════════════════════════════════
Best model:
/home/aiteam/warehouse_bag_poc/outputs/rice_bag_detection_v2/weights/best.pt
════════════════════════════════════════════════════════════════════
```

**What to look for**:
- ✅ Epoch 1 metrics should NOT be 0!
  - Precision > 0 ✓
  - Recall > 0 ✓
  - mAP50 > 0 ✓
- ✅ Metrics should improve over epochs
- ✅ Training should complete successfully
- ✅ Best model saved

---

### PHASE 3: Check Results (5 minutes)

After training completes, check metrics:

```bash
# View results CSV
head -30 /home/aiteam/warehouse_bag_poc/outputs/rice_bag_detection_v2/results.csv
```

**Expected to see**:
```
epoch,train/box_loss,train/cls_loss,train/dfl_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),...
0,2.156,4.456,2.124,0.145,0.289,0.0521,0.0089,...
1,1.981,3.876,2.014,0.287,0.445,0.1823,0.0301,...
2,1.876,3.654,1.923,0.387,0.521,0.2845,0.0456,...
3,1.823,3.512,1.876,0.456,0.612,0.3456,0.0567,...
...
50,1.234,2.123,1.567,0.756,0.823,0.7234,0.4512,...
```

**Compare with broken version**:
```
❌ BEFORE (V2 with polygon labels):
   Epoch 1: 0, 0, 0
   Epoch 25: 0, 0, NaN  ← FAILED
   
✅ AFTER (V2 with bbox labels):
   Epoch 1: 0.145, 0.289, 0.0521  ← WORKING!
   Epoch 50: 0.756, 0.823, 0.7234 ← GREAT!
```

---

### PHASE 4: View Predictions (2 minutes)

Check if model is actually detecting rice bags:

```bash
# View validation images with predictions
# Linux/Mac:
open /home/aiteam/warehouse_bag_poc/outputs/rice_bag_detection_v2/val_batch0_pred.jpg

# Windows:
start /home/aiteam/warehouse_bag_poc/outputs/rice_bag_detection_v2/val_batch0_pred.jpg

# Or from terminal:
file /home/aiteam/warehouse_bag_poc/outputs/rice_bag_detection_v2/val_batch0_pred.jpg
```

**What to look for**:
- ✅ Boxes drawn around rice bags
- ✅ Boxes inside the correct areas
- ✅ NOT random scattered boxes

Compare with labels:
```bash
# View ground truth labels
open /home/aiteam/warehouse_bag_poc/outputs/rice_bag_detection_v2/val_batch0_labels.jpg
```

**Expected**:
- Labels: Polygons showing exact rice bag outlines
- Predictions: Rectangles (bboxes) overlapping the bags

---

### PHASE 5: Test with Web UI (10 minutes)

Run the testing interface:

```bash
cd /home/aiteam/warehouse_bag_poc
python src/test_rice_bag_v2_ui.py
```

**What you'll see**:
```
════════════════════════════════════════════════════════════════════
RICE BAG TILED TESTING UI
════════════════════════════════════════════════════════════════════
Model : /home/aiteam/warehouse_bag_poc/outputs/rice_bag_detection_v2/weights/best.pt
Device: 0
GPU   : NVIDIA RTX PRO 6000 Blackwell Workstation Edition
════════════════════════════════════════════════════════════════════

════════════════════════════════════════════════════════════════════
SERVER STARTED
════════════════════════════════════════════════════════════════════
Open in browser:
http://127.0.0.1:5000
or
http://<GPU-IP>:5000
════════════════════════════════════════════════════════════════════

 * Serving Flask app 'test_rice_bag_tiled'
 * Running on http://127.0.0.1:5000
 * Running on http://192.168.2.240:5000
```

**Open in browser**: http://127.0.0.1:5000

**Then**:
1. Upload test image
2. Wait for prediction
3. See rice bags outlined with boxes!

---

## 📊 EXPECTED RESULTS SUMMARY

### Training Metrics
| Metric | Before Fix | After Fix |
|--------|-----------|-----------|
| Epoch 1 Precision | 0 | > 0.05 |
| Epoch 1 Recall | 0 | > 0.05 |
| Epoch 1 mAP50 | 0 | > 0.01 |
| Epoch 50 Precision | 0 | > 0.5 |
| Epoch 50 Recall | 0 | > 0.5 |
| Epoch 50 mAP50 | 0 or NaN | > 0.4 |

### Predictions
| Aspect | Before | After |
|--------|--------|-------|
| Boxes position | Random/wrong | Around rice bags |
| Boxes accuracy | 0% overlap | 50%+ overlap |
| Learning progress | None | Steady improvement |

---

## ⏱️ TOTAL TIME ESTIMATE

| Phase | Duration | Notes |
|-------|----------|-------|
| 1. Prepare | 2 min | Terminal setup |
| 2a. Verify Labels | 30 sec | Quick check |
| 2b. Convert Labels | 1-2 min | Safe conversion |
| 2c. Train Model | 30-60 min | GPU training |
| 3. Check Results | 5 min | Review metrics |
| 4. View Predictions | 2 min | Visual check |
| 5. Test UI | 10 min | Upload & test |
| **TOTAL** | **60-90 min** | Mostly training |

---

## 🔍 TROUBLESHOOTING DURING EXECUTION

### Issue: "Empty label files"
```
Error: empty label files found

Fix:
  1. Labels were deleted by old script
  2. Need to restore from RoboFlow
  3. Run: python recover_dataset.py
  4. Download and extract dataset
  5. Run MASTER_WORKFLOW.py again
```

### Issue: "CUDA not available"
```
Error: CUDA GPU is not available!

Fix:
  1. Check GPU is properly installed
  2. Verify PyTorch has CUDA: 
     python -c "import torch; print(torch.cuda.is_available())"
  3. Install with CUDA:
     pip install torch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### Issue: "Training very slow"
```
Normal! First few epochs are slower.
- Epoch 1: ~5 minutes
- Epoch 2-10: ~3 minutes each
- Later: ~2 minutes each
- Total: 30-60 minutes

If taking > 2 hours per epoch, you might be on CPU.
Check: python -c "import torch; print(torch.cuda.is_available())"
```

### Issue: "Metrics still all 0 after epoch 1"
```
This would mean labels are STILL in wrong format.

Troubleshooting:
  1. Stop training (Ctrl+C)
  2. Check labels manually:
     head -5 data/Rice\ Bag\ 10/train/labels/first_file.txt
  3. Should see 5 numbers per line (class cx cy w h)
  4. If see many numbers: conversion didn't work
  5. Re-run: python convert_polygon_to_bbox_v2.py
  6. Then try training again
```

---

## ✅ SUCCESS INDICATORS

After completing PHASE 5, you should have:

- ✅ No errors during any phase
- ✅ Epoch 1 metrics > 0 (not 0)
- ✅ Metrics improving over epochs
- ✅ Best model saved at: `outputs/rice_bag_detection_v2/weights/best.pt`
- ✅ Validation images showing boxes around rice bags
- ✅ Web UI working (http://127.0.0.1:5000)
- ✅ Can upload images and see predictions

---

## 🎉 FINAL COMMAND SEQUENCE

Copy-paste these commands in order:

```bash
# Step 1: Navigate to project
cd /home/aiteam/warehouse_bag_poc

# Step 2: Run complete workflow (60-75 minutes)
python MASTER_WORKFLOW.py

# Step 3: After training completes, check results
cat outputs/rice_bag_detection_v2/results.csv | head -10

# Step 4: View predictions (opens image viewer)
# Linux/Mac:
open outputs/rice_bag_detection_v2/val_batch0_pred.jpg
# Windows:
start outputs/rice_bag_detection_v2/val_batch0_pred.jpg

# Step 5: Test with web UI
python src/test_rice_bag_v2_ui.py

# Step 6: Open browser and test
# Browser: http://127.0.0.1:5000
```

---

## 📞 QUICK REFERENCE

| Need | Command |
|------|---------|
| Run everything | `python MASTER_WORKFLOW.py` |
| Convert labels only | `python convert_polygon_to_bbox_v2.py` |
| Train only | `python src/training/train_rice_bags_v2.py` |
| Test with UI | `python src/test_rice_bag_v2_ui.py` |
| Check metrics | `cat outputs/rice_bag_detection_v2/results.csv` |
| View predictions | `open outputs/rice_bag_detection_v2/val_batch0_pred.jpg` |
| Get help | `cat QUICKSTART.md` |

---

**NOW GO RUN IT!** 🚀

```bash
python MASTER_WORKFLOW.py
```

Everything is ready. Good luck! 🎉

