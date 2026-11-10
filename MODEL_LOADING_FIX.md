# 🔧 Model Loading Error Fix

## The Error
```
IsADirectoryError: [Errno 21] Is a directory: 'models/best_model'
```

## ✅ Solution

The model is saved in PyTorch's directory format (not a single `.pth` file). Use this updated loading code:

### Replace This (BROKEN):
```python
model_path = 'models/best_model'
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint)
```

### With This (FIXED):
```python
model_path = 'models/best_model'

# Load model weights from directory format
try:
    # Try loading as a full model directory
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint)
except IsADirectoryError:
    # For newer PyTorch directory format, load the entire model
    model = torch.load(model_path, map_location=device, weights_only=False)

model.to(device)
model.eval()

print("✓ Model loaded successfully")
print(f"  Backbone: ResNet-50")
print(f"  Classes: {model.class_names}")
```

---

## 🎯 Complete Fixed Model Loading Cell

```python
# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize model architecture
model = TumourClassifier(backbone='resnet50', num_classes=3, pretrained=False)
model_path = 'models/best_model'

print(f"\nLoading model from {model_path}...")

# Load the saved model
try:
    # Method 1: Try as state dict
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict):
        # If it's a checkpoint dictionary
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        # If it's a full model
        model = checkpoint
        
except IsADirectoryError:
    # For PyTorch directory format
    import glob
    # Check if there's a consolidated file
    pth_files = glob.glob(f"{model_path}/*.pth")
    if pth_files:
        checkpoint = torch.load(pth_files[0], map_location=device, weights_only=False)
        model.load_state_dict(checkpoint)
    else:
        # Load using the directory as a whole
        model = torch.load(model_path, map_location=device, weights_only=False)

# Move to device and set to eval mode
model.to(device)
model.eval()

print("✓ Model loaded successfully")
print(f"  Backbone: ResNet-50")
print(f"  Classes: {model.class_names}")
print(f"  Device: {device}")
```

---

## 🔍 Alternative: Check Model Structure First

Add this cell before loading to understand the structure:

```python
# Check model directory structure
import os
model_path = 'models/best_model'

print(f"Model directory: {model_path}")
print(f"Is directory: {os.path.isdir(model_path)}")
print(f"Is file: {os.path.isfile(model_path)}")

print("\nContents:")
if os.path.isdir(model_path):
    for item in os.listdir(model_path):
        item_path = os.path.join(model_path, item)
        size = os.path.getsize(item_path) if os.path.isfile(item_path) else 'DIR'
        print(f"  {item} - {size}")
```

---

## 💡 Understanding the Issue

Your model was saved from Colab using:
```python
torch.save(model.state_dict(), 'best_model')
```

This created a **directory** with the model weights split across multiple files (PyTorch's new format for large models).

**Structure:**
```
models/best_model/
├── data.pkl          # Model metadata
├── byteorder         # Byte order info
└── data/             # Weight tensors (743 files)
    ├── 0, 1, 2, ...
```

---

## 🚀 Recommended Solution (Simplest)

Use this single-line fix:

```python
# Load model (handles directory format automatically)
model = torch.load('models/best_model', map_location=device, weights_only=False)
model.to(device)
model.eval()

print("✓ Model loaded successfully")
```

This works because when you save a model's state_dict, PyTorch sometimes creates a directory structure, and `torch.load()` can read it directly.

---

## ⚠️ Warning About `pretrained=True`

You're also seeing this warning:
```
UserWarning: The parameter 'pretrained' is deprecated
```

**Fix:** Change model initialization to:

```python
# OLD (deprecated)
model = TumourClassifier(backbone='resnet50', num_classes=3, pretrained=True)

# NEW (recommended)
model = TumourClassifier(backbone='resnet50', num_classes=3, pretrained=False)
```

Use `pretrained=False` because you're loading your own trained weights, not ImageNet weights.

---

## ✅ Final Complete Cell

```python
# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load trained model directly
model_path = 'models/best_model'
print(f"Loading model from {model_path}...")

# Load the full model (handles directory format)
model = torch.load(model_path, map_location=device, weights_only=False)
model.to(device)
model.eval()

print("✓ Model loaded successfully")
try:
    print(f"  Classes: {model.class_names}")
except:
    print(f"  Model type: {type(model)}")
    
print(f"  Device: {device}")
```

---

## 🧪 Test After Loading

Verify the model works:

```python
# Quick test
print("\nTesting model...")
test_input = torch.randn(1, 3, 224, 224).to(device)
with torch.no_grad():
    output = model(test_input)
    print(f"Output shape: {output.shape}")
    print(f"Number of classes: {output.shape[1]}")
    
print("✓ Model is working!")
```

---

**Use the recommended solution above and your model should load successfully!** 🚀
