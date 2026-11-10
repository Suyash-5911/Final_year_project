# 🔧 Dataset Loading Fix - No Test Images Found

## The Error
```
KeyError: 'glioma'
```

This means the `sample_images` dictionary is empty because test images weren't found.

## ✅ Quick Fix

Add this cell **before** the "Test on one image" cell:

```python
# Check and fix dataset path
import os
from pathlib import Path

print("Checking for test data...")

# Possible data locations
possible_paths = [
    'data/brisc/processed/test',
    'data/brisc/test',
    'data/test',
    'test'
]

data_dir = None
for path in possible_paths:
    if os.path.exists(path):
        data_dir = Path(path)
        print(f"✓ Found test data at: {path}")
        break

if data_dir is None:
    print("⚠️ No test data found!")
    print("\nYou have two options:")
    print("1. Upload test images to Colab")
    print("2. Use sample images for demonstration")
    
    # Option 2: Create sample data for testing
    print("\nCreating sample test images for demonstration...")
    
    # We'll use a few images from any available source
    # Or create dummy images to test Grad-CAM
    import numpy as np
    from PIL import Image
    
    # Create a dummy test setup
    test_dir = Path('data/test_samples')
    for class_name in ['glioma', 'meningioma', 'pituitary']:
        (test_dir / class_name).mkdir(parents=True, exist_ok=True)
        
        # Create a dummy grayscale "brain" image
        img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        # Add some structure to make it look more like a brain scan
        center = (128, 128)
        y, x = np.ogrid[:256, :256]
        mask = (x - center[0])**2 + (y - center[1])**2 <= 100**2
        img[mask] = np.clip(img[mask] * 1.5, 0, 255)
        
        # Save
        Image.fromarray(img).save(test_dir / class_name / 'test_001.jpg')
    
    data_dir = test_dir
    print(f"✓ Created sample images at: {data_dir}")

# Now collect images
class_names = ['glioma', 'meningioma', 'pituitary']
sample_images = {}

for class_name in class_names:
    class_dir = data_dir / class_name
    if class_dir.exists():
        image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
        sample_images[class_name] = image_files[:5]  # Take first 5
        print(f"  {class_name}: {len(image_files)} images found, using {len(sample_images[class_name])} samples")
    else:
        print(f"  ⚠️ {class_name}: directory not found at {class_dir}")

total_samples = sum(len(v) for v in sample_images.values())
print(f"\n✓ Loaded {total_samples} sample images")

if total_samples == 0:
    print("\n❌ No images available for testing!")
    print("Please upload test images or use the sample creation above.")
```

---

## 🎯 Alternative: Upload Your Own Test Images

If you have test images locally, upload them to Colab:

```python
# Option 1: Upload specific images
from google.colab import files
import shutil
from pathlib import Path

print("Please upload a few test images...")
uploaded = files.upload()

# Organize uploaded files
test_dir = Path('test_samples')
for class_name in ['glioma', 'meningioma', 'pituitary']:
    (test_dir / class_name).mkdir(parents=True, exist_ok=True)

# Move uploaded files (you'll need to specify which class each belongs to)
for filename in uploaded.keys():
    # Determine class from filename or ask user
    if 'glioma' in filename.lower():
        shutil.move(filename, test_dir / 'glioma' / filename)
    elif 'meningioma' in filename.lower():
        shutil.move(filename, test_dir / 'meningioma' / filename)
    elif 'pituitary' in filename.lower():
        shutil.move(filename, test_dir / 'pituitary' / filename)

print("✓ Images uploaded and organized")
```

---

## 🚀 Best Solution: Use Sample Images from Repository

If your repository has test images, make sure they're cloned:

```python
# Check if data exists in repository
import os

if not os.path.exists('data/brisc'):
    print("Dataset not found in repository.")
    print("The dataset is likely too large to be in Git.")
    print("\nYou have these options:")
    print("1. Download BRISC dataset separately")
    print("2. Use a few sample images for demonstration")
    print("3. Upload your own test images")

# For demonstration, use just a few images
print("\nFor Grad-CAM demonstration, we need at least 1 image per class.")
```

---

## 📋 Complete Working Cell Sequence

Replace the data loading section in your notebook with this:

### Cell 1: Setup Data Directory
```python
from pathlib import Path
import os

# Try to find test data
data_paths = [
    'data/brisc/processed/test',
    'data/test',
    'test_samples'
]

data_dir = None
for path in data_paths:
    if Path(path).exists():
        data_dir = Path(path)
        break

if data_dir is None:
    print("⚠️ Creating sample test directory...")
    data_dir = Path('test_samples')
    for cls in ['glioma', 'meningioma', 'pituitary']:
        (data_dir / cls).mkdir(parents=True, exist_ok=True)
    
    print("Please upload at least one test image per class to:")
    print(f"  {data_dir}/glioma/")
    print(f"  {data_dir}/meningioma/")
    print(f"  {data_dir}/pituitary/")
else:
    print(f"✓ Using test data from: {data_dir}")
```

### Cell 2: Load Sample Images
```python
# Collect sample images
class_names = ['glioma', 'meningioma', 'pituitary']
sample_images = {}

for class_name in class_names:
    class_dir = data_dir / class_name
    if class_dir.exists():
        image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
        if image_files:
            sample_images[class_name] = image_files[:5]
            print(f"✓ {class_name}: {len(sample_images[class_name])} images")
        else:
            print(f"⚠️ {class_name}: no images found")
    else:
        print(f"✗ {class_name}: directory not found")

total = sum(len(v) for v in sample_images.values())
print(f"\nTotal samples: {total}")

if total == 0:
    raise RuntimeError("No test images found! Please upload images first.")
```

### Cell 3: Then Test Grad-CAM
```python
# Now test on one image
if sample_images:
    # Get first available class with images
    for class_name, images in sample_images.items():
        if images:
            test_class = class_name
            test_image_path = images[0]
            break
    
    print(f"Testing Grad-CAM on: {test_image_path.name}")
    print(f"True class: {test_class}")
    
    # Continue with Grad-CAM...
```

---

## ✅ Quick Demo Without Real Data

For a quick demonstration, use this:

```python
# Generate a synthetic brain scan image for testing
import numpy as np
from PIL import Image
from pathlib import Path

# Create synthetic test image
print("Creating synthetic test image...")
img_array = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)

# Add some structure (simulate brain tissue)
center = (112, 112)
y, x = np.ogrid[:224, :224]
mask = (x - center[0])**2 + (y - center[1])**2 <= 80**2
img_array[mask] = np.clip(img_array[mask] * 1.2, 0, 255).astype(np.uint8)

# Save
test_path = Path('test_sample.jpg')
Image.fromarray(img_array.astype(np.uint8)).save(test_path)

# Now use this for testing
test_image_path = test_path
print(f"✓ Created test image: {test_path}")

# Load and test
image_tensor, original_image = load_image(test_image_path)
# Continue with Grad-CAM...
```

---

**Choose one of the solutions above based on your needs!** The quickest is to create synthetic test images for demonstration. 🚀
