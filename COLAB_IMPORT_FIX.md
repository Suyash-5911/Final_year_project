# 🔧 Colab Import Error - Quick Fix

## The Error You're Seeing
```
ModuleNotFoundError: No module named 'models.classifier'
```

## ✅ Solution: Replace the Import Cell

### Current Code (BROKEN):
```python
# Add src to path
sys.path.insert(0, 'src')

from models.classifier import TumourClassifier
from explainability.grad_cam import ExplanationGenerator
```

### Fixed Code (USE THIS):
```python
# Setup paths correctly
import sys
import os

# Change to project directory if needed
if os.path.exists('/content/Final_year_project'):
    os.chdir('/content/Final_year_project')

# Add both project root and src to path
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

# Import with full paths
from src.models.classifier import TumourClassifier
from src.explainability.grad_cam import ExplanationGenerator
from src.explainability.visualization import (
    create_comparison_visualization,
    create_grid_visualization,
    overlay_heatmap
)
from src.evaluation.localization import LocalizationEvaluator

print("✓ Imports successful")
print(f"Working directory: {os.getcwd()}")
```

---

## 📋 Complete Colab Notebook Structure

### Cell 1: Clone Repository
```python
# Clone repository
!git clone https://github.com/Suyash-5911/Final_year_project.git
%cd Final_year_project
```

### Cell 2: Check GPU
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
```

### Cell 3: Install Dependencies
```python
!pip install -q opencv-python scikit-image matplotlib seaborn tqdm
```

### Cell 4: Verify Structure (Optional but Recommended)
```python
# Verify all files are present
import os

required_files = [
    'src/models/classifier.py',
    'src/explainability/grad_cam.py',
    'src/explainability/visualization.py',
    'src/evaluation/localization.py',
    'models/best_model/data.pkl'
]

print("Checking project structure...")
for file in required_files:
    exists = os.path.exists(file)
    print(f"{'✓' if exists else '✗'} {file}")
```

### Cell 5: Import Modules (THE FIX)
```python
# Setup imports
import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import json

# Fix paths
if os.path.exists('/content/Final_year_project'):
    os.chdir('/content/Final_year_project')
    
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

# Import project modules with full paths
from src.models.classifier import TumourClassifier
from src.explainability.grad_cam import ExplanationGenerator
from src.explainability.visualization import (
    create_comparison_visualization,
    create_grid_visualization,
    overlay_heatmap
)
from src.evaluation.localization import LocalizationEvaluator

print("✓ All imports successful!")
print(f"Working directory: {os.getcwd()}")
```

### Cell 6: Load Model
```python
# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load model
model = TumourClassifier(backbone='resnet50', num_classes=3)
model_path = 'models/best_model'

checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()

print("✓ Model loaded successfully")
print(f"  Backbone: ResNet-50")
print(f"  Classes: {model.class_names}")
```

---

## 🎯 Alternative: Use Setup Script

If you keep having issues, add this cell first:

```python
# Run setup script
%run colab_setup.py

# Then import normally
from src.models.classifier import TumourClassifier
from src.explainability.grad_cam import ExplanationGenerator
from src.explainability.visualization import create_comparison_visualization
from src.evaluation.localization import LocalizationEvaluator
```

---

## 🔍 Debugging Commands

If imports still fail, run these to diagnose:

### Check Current Directory
```python
import os
print("Current directory:", os.getcwd())
print("\nDirectory contents:")
!ls -la
```

### Check Python Path
```python
import sys
print("Python path:")
for p in sys.path:
    print(f"  {p}")
```

### Check if Module Files Exist
```python
import os
files_to_check = [
    'src/models/classifier.py',
    'src/models/__init__.py',
    'src/explainability/grad_cam.py',
    'src/explainability/__init__.py'
]

for f in files_to_check:
    print(f"{'✓' if os.path.exists(f) else '✗'} {f}")
```

---

## 💡 Why This Happens

The issue occurs because:
1. Colab starts in `/content/` directory
2. Your project is in `/content/Final_year_project/`
3. The `src` folder needs to be in Python's path
4. Imports need to reference `src.models` not just `models`

The fix ensures:
- ✅ Working directory is set to project root
- ✅ Both project root AND src are in Python path
- ✅ Imports use full module paths (`src.models.classifier`)

---

## ✅ Quick Test

After fixing imports, test with:

```python
# Test imports worked
print("Testing imports...")
print(f"TumourClassifier: {TumourClassifier}")
print(f"ExplanationGenerator: {ExplanationGenerator}")
print(f"LocalizationEvaluator: {LocalizationEvaluator}")
print("\n✓ All imports working!")
```

---

## 📝 Summary

**Replace this:**
```python
sys.path.insert(0, 'src')
from models.classifier import TumourClassifier
```

**With this:**
```python
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
from src.models.classifier import TumourClassifier
```

**That's it!** The rest of the notebook should work fine after this fix.

---

## 🚀 Ready to Continue

After fixing the imports:
1. ✅ Initialize Grad-CAM
2. ✅ Load test images
3. ✅ Generate explanations
4. ✅ Download results

All other cells should work without modification!
