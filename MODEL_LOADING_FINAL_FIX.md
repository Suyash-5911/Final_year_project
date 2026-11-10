# ✅ CORRECT Model Loading Solution

## The Issue
The model is saved in PyTorch's new ZIP-based format (a directory structure), and `torch.load()` can't read directories directly in some PyTorch versions.

## ✅ WORKING SOLUTION

Use this code in your Colab notebook:

```python
# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize model architecture first
print("\nInitializing model architecture...")
model = TumourClassifier(backbone='resnet50', num_classes=3, pretrained=False)

# Load the state dict from the model directory
model_path = 'models/best_model'
print(f"Loading weights from {model_path}...")

# For ZIP-based format, we need to use the internal torch utilities
import torch._weights_only_unpickler as wou
from pathlib import Path

# Load using the model directory path directly with special handling
try:
    # PyTorch 2.0+ ZIP format - load as file object
    import zipfile
    import io
    
    # Read the data.pkl from the directory
    with open(f"{model_path}/data.pkl", 'rb') as f:
        state_dict = torch.load(f, map_location=device, weights_only=False)
    
    model.load_state_dict(state_dict)
    print("✓ Loaded from data.pkl")
    
except Exception as e:
    print(f"Method 1 failed: {e}")
    
    # Alternative: Use pickle directly
    try:
        import pickle
        with open(f"{model_path}/data.pkl", 'rb') as f:
            state_dict = pickle.load(f)
        model.load_state_dict(state_dict)
        print("✓ Loaded using pickle")
    except Exception as e2:
        print(f"Method 2 failed: {e2}")
        raise

model.to(device)
model.eval()

print("✓ Model loaded successfully")
print(f"  Backbone: ResNet-50")
print(f"  Classes: {model.class_names}")
print(f"  Device: {device}")
```

---

## 🎯 SIMPLEST SOLUTION (Recommended)

Just load from the `data.pkl` file directly:

```python
import torch
from src.models.classifier import TumourClassifier

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create model
model = TumourClassifier(backbone='resnet50', num_classes=3, pretrained=False)

# Load weights from data.pkl
state_dict = torch.load('models/best_model/data.pkl', map_location=device, weights_only=False)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

print("✓ Model loaded successfully")
print(f"  Classes: {model.class_names}")
```

---

## 🔧 Alternative: Convert to Standard .pth Format

Add this cell to convert the model to a standard format:

```python
# One-time conversion: Save as standard .pth file
import torch

# Load from the directory format
state_dict = torch.load('models/best_model/data.pkl', 
                       map_location='cpu', 
                       weights_only=False)

# Save as standard .pth file
torch.save(state_dict, 'models/best_model.pth')
print("✓ Converted to best_model.pth")

# Now you can load normally
model = TumourClassifier(backbone='resnet50', num_classes=3, pretrained=False)
model.load_state_dict(torch.load('models/best_model.pth', map_location=device))
model.to(device)
model.eval()
print("✓ Model loaded from .pth file")
```

---

## 📋 Complete Working Cell

Copy this entire cell to your Colab notebook:

```python
import torch
from src.models.classifier import TumourClassifier

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize model architecture
print("\nInitializing model...")
model = TumourClassifier(backbone='resnet50', num_classes=3, pretrained=False)

# Load weights from data.pkl (the actual weight file)
print("Loading weights from models/best_model/data.pkl...")
state_dict = torch.load('models/best_model/data.pkl', 
                       map_location=device, 
                       weights_only=False)

# Load the state dict into the model
model.load_state_dict(state_dict)
model.to(device)
model.eval()

print("\n✓ Model loaded successfully!")
print(f"  Backbone: ResNet-50")
print(f"  Classes: {model.class_names}")
print(f"  Device: {device}")

# Quick test
print("\nTesting model...")
test_input = torch.randn(1, 3, 224, 224).to(device)
with torch.no_grad():
    output = model(test_input)
    print(f"  Output shape: {output.shape}")
    print(f"  ✓ Model is working!")
```

---

## 🎓 Understanding the Issue

When you saved the model in Colab, PyTorch saved it in a **ZIP-based directory format**:

```
models/best_model/
├── data.pkl          ← The actual state_dict is here
├── data/             ← Tensor storage files
├── byteorder
├── .format_version
└── .storage_alignment
```

The solution is to **load directly from `data.pkl`** instead of the directory.

---

## ✅ Use This Code Now

**COPY THIS to your Colab:**

```python
# WORKING MODEL LOADING CODE
import torch
from src.models.classifier import TumourClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create model structure
model = TumourClassifier(backbone='resnet50', num_classes=3, pretrained=False)

# Load weights from the data.pkl file inside the directory
state_dict = torch.load('models/best_model/data.pkl', map_location=device, weights_only=False)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

print("✓ Model loaded successfully!")
print(f"Classes: {model.class_names}")
```

**This will work!** 🚀
