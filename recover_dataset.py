"""
Recover Rice Bag Dataset from RoboFlow

This script downloads the original datasets from RoboFlow
to restore the deleted label files.
"""

import os
import subprocess
from pathlib import Path

# Dataset information from data.yaml files
DATASETS = [
    {
        "name": "Rice Bag 10",
        "workspace": "ovais-alam-qureshi",
        "project": "rice-bag-10",
        "version": 1,
        "download_url": "https://universe.roboflow.com/ovais-alam-qureshi/rice-bag-10/dataset/1",
        "output_dir": "data/Rice Bag 10",
    },
    {
        "name": "Rice Bag Detection",
        "workspace": "ovais-alam-qureshi",
        "project": "rice-bag-detection-qp7uw",
        "version": 2,
        "download_url": "https://universe.roboflow.com/ovais-alam-qureshi/rice-bag-detection-qp7uw/dataset/2",
        "output_dir": "data/Rice Bag dataset",
    },
]

ROOT = Path(__file__).resolve().parent

print("""
╔════════════════════════════════════════════════════════════════════╗
║           RICE BAG DATASET - RECOVERY FROM ROBOFLOW                ║
╚════════════════════════════════════════════════════════════════════╝

Dataset Information:
""")

for ds in DATASETS:
    print(f"""
  Dataset: {ds['name']}
  Project: {ds['workspace']}/{ds['project']} (v{ds['version']})
  URL: {ds['download_url']}
  Output: {ds['output_dir']}
""")

print("\n⚠️  NOTE: This requires RoboFlow API key!")
print("""
To recover the datasets:

Option 1: Download from RoboFlow Web UI
  1. Visit: https://universe.roboflow.com/ovais-alam-qureshi
  2. Select the dataset (Rice Bag 10 or Rice Bag Detection)
  3. Click "Download Dataset"
  4. Select YOLO format
  5. Extract to data/ folder

Option 2: Use RoboFlow Python API
  pip install roboflow
  python -c "
from roboflow import Roboflow
rf = Roboflow(api_key='YOUR_API_KEY')
project = rf.workspace('ovais-alam-qureshi').project('rice-bag-10')
dataset = project.versions(1).download('yolo', location='data/Rice Bag 10')
  "

Option 3: Manually Download
  wget https://... (check RoboFlow download link)

After downloading, run verification:
  bash verify_labels.sh
""")

print("\n╔════════════════════════════════════════════════════════════════════╗")
print("║  IMPORTANT: Labels were accidentally deleted - need recovery!     ║")
print("╚════════════════════════════════════════════════════════════════════╝\n")
