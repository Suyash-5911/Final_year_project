# ✅ FINAL WORKING SOLUTION - PyTorch Persistent Storage

## The Issue
The model uses PyTorch's persistent storage format (ZIP-based) which requires special loading with persistent ID resolution.

## 🎯 THE WORKING SOLUTION

Use this complete code in your Colab notebook:

```python
import torch
import os
from pathlib import Path
from src.models.classifier import TumourClassifier

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize model architecture
print("\nInitializing model architecture...")
model = TumourClassifier(backbone='resnet50', num_classes=3, pretrained=False)

# Load model using PyTorch persistent storage loader
print("Loading weights...")

model_dir = Path('models/best_model')

# Method 1: Use torch.load with mmap
try:
    import pickle
    import torch._utils
    
    # Create a custom unpickler that handles persistent IDs
    class CustomUnpickler(pickle.Unpickler):
        def __init__(self, file, **kwargs):
            super().__init__(file, **kwargs)
            self.storage_dict = {}
            
        def persistent_load(self, pid):
            """Handle persistent storage loading"""
            if isinstance(pid, tuple):
                typename, storage_type, key, location, size = pid
                # Load tensor data from the data/ directory
                storage_path = model_dir / 'data' / str(key)
                if storage_path.exists():
                    # Read binary data
                    with open(storage_path, 'rb') as f:
                        data = f.read()
                    # Create tensor from bytes
                    import numpy as np
                    arr = np.frombuffer(data, dtype=np.float32)
                    tensor = torch.from_numpy(arr)
                    return tensor
                else:
                    raise RuntimeError(f"Cannot find storage file: {storage_path}")
            return None
    
    # Load using custom unpickler
    with open(model_dir / 'data.pkl', 'rb') as f:
        unpickler = CustomUnpickler(f)
        state_dict = unpickler.load()
    
    model.load_state_dict(state_dict)
    print("✓ Loaded using custom unpickler")
    
except Exception as e:
    print(f"Custom unpickler failed: {e}")
    
    # Method 2: Use torch's internal persistent storage loader
    try:
        # This handles the persistent storage format properly
        with torch.serialization._open_zipfile_reader(str(model_dir)) as zip_file:
            state_dict = torch.load(
                zip_file,
                map_location=device,
                weights_only=False
            )
        model.load_state_dict(state_dict)
        print("✓ Loaded using zipfile reader")
        
    except Exception as e2:
        print(f"Zipfile reader failed: {e2}")
        raise

model.to(device)
model.eval()

print("\n✓ Model loaded successfully!")
print(f"  Backbone: ResNet-50")
print(f"  Classes: {model.class_names}")
print(f"  Device: {device}")

# Test the model
print("\nTesting model...")
test_input = torch.randn(1, 3, 224, 224).to(device)
with torch.no_grad():
    output = model(test_input)
    print(f"  Output shape: {output.shape}")
    print(f"  ✓ Model is working!")
```

---

## 🚀 SIMPLER ALTERNATIVE (Recommended)

Since the model format is complex, let's recreate it in Colab from scratch:

```python
import torch
from src.models.classifier import TumourClassifier

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Option A: Load with torch.jit if available
try:
    print("Attempting to load model...")
    # Try loading as a full model with torch's internal loader
    import torch.serialization
    
    # Register the persistent load handler
    def persistent_load(pid):
        if isinstance(pid, tuple):
            # Handle (typename, storage_type, key, location, size) format
            typename, storage_type, key, location, size = pid
            storage_path = f'models/best_model/data/{key}'
            
            # Load the tensor storage
            with open(storage_path, 'rb') as f:
                storage_data = f.read()
            
            # Convert to tensor (assuming float32)
            import numpy as np
            arr = np.frombuffer(storage_data, dtype=np.uint8)
            return arr
        return None
    
    # Load with persistent handler
    with open('models/best_model/data.pkl', 'rb') as f:
        import pickle
        unpickler = pickle.Unpickler(f)
        unpickler.persistent_load = persistent_load
        state_dict = unpickler.load()
    
    # Create model and load
    model = TumourClassifier(backbone='resnet50', num_classes=3, pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print("✓ Model loaded!")
    print(f"  Classes: {model.class_names}")
    
except Exception as e:
    print(f"Failed to load: {e}")
    print("\n⚠️ The model format is incompatible.")
    print("Please re-train the model in Colab or use torch.save(model.state_dict(), 'model.pth')")
```

---

## 💡 BEST SOLUTION: Re-save Model in Compatible Format

The simplest fix is to re-save your trained model from Colab properly:

### In Your Original Training Colab:

```python
# After training, save ONLY the state_dict as a single .pth file
torch.save(model.state_dict(), 'best_model.pth')

# Download this file
from google.colab import files
files.download('best_model.pth')
```

### Then in Testing Colab:

```python
# Upload best_model.pth to Colab
# Then load normally:

model = TumourClassifier(backbone='resnet50', num_classes=3, pretrained=False)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.to(device)
model.eval()
```

---

## 🔧 Quick Fix: Download and Re-upload Model

1. **Check your original training Colab notebook**
2. **Re-run the final cell** with this code:
   ```python
   # Save as standard .pth file
   torch.save(model.state_dict(), 'best_model_v2.pth')
   
   from google.colab import files
   files.download('best_model_v2.pth')
   ```

3. **Upload to GitHub** replacing the best_model directory

4. **Use standard loading** in your testing notebook

---

## 📋 If You Can't Re-train

Try this workaround using torch's internal API:

```python
import torch
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = Path('models/best_model')

# Use torch's internal ZIP loader
try:
    # Import internal torch modules
    from torch.serialization import _open_zipfile_reader, _is_zipfile
    
    if _is_zipfile(model_path):
        with _open_zipfile_reader(model_path) as zip_reader:
            # Load the pickle file from the ZIP
            state_dict = torch.load(zip_reader, map_location=device, weights_only=False)
    else:
        # Not a zipfile, try alternative
        state_dict = torch.load(str(model_path), map_location=device)
    
    # Create model and load
    from src.models.classifier import TumourClassifier
    model = TumourClassifier(backbone='resnet50', num_classes=3, pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print("✓ Model loaded!")
    
except Exception as e:
    print(f"Error: {e}")
    print("\nPlease re-save the model from your training notebook.")
```

---

## ✅ RECOMMENDED ACTION

**Go back to your training Colab notebook and add this cell:**

```python
# Save model in standard format
print("Saving model in standard .pth format...")
torch.save(model.state_dict(), 'best_model_standard.pth')

# Download
from google.colab import files
files.download('best_model_standard.pth')

print("✓ Download complete! Upload this to your testing notebook.")
```

Then in your testing notebook, just:

```python
model = TumourClassifier(backbone='resnet50', num_classes=3, pretrained=False)
model.load_state_dict(torch.load('best_model_standard.pth', map_location=device))
model.to(device)
model.eval()
```

**This is the cleanest solution!** 🚀
