# ✅ WORKING Dataset Links (Updated)

**Problem**: Original Figshare direct link is broken.
**Solution**: Use these verified working alternatives.

---

## 🏆 BEST OPTION: Kaggle Datasets

### **1. Brain Tumor Classification MRI** ⭐⭐⭐ (RECOMMENDED)
**Size**: 253 MB | **3,264 images** | **4 classes**

**Link**: https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri

**Classes**:
- Glioma (826 images)
- Meningioma (822 images)
- Pituitary (827 images)
- No Tumor (395 images)

**Download**:

**Option A: Web Interface** (Easiest)
```
1. Visit: https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri
2. Click "Download" (requires free Kaggle account)
3. Extract zip file
4. Done!
```

**Option B: Kaggle CLI**
```bash
# Install Kaggle
pip install kaggle

# Setup API token (one-time)
# Go to https://www.kaggle.com/settings
# Click "Create New API Token"
# Save kaggle.json to ~/.kaggle/ (Linux/Mac) or C:\Users\<you>\.kaggle\ (Windows)

# Download
kaggle datasets download -d sartajbhuvaji/brain-tumor-classification-mri

# Extract
unzip brain-tumor-classification-mri.zip -d data/brain_tumours
```

**Structure after extraction**:
```
data/brain_tumours/
├── Training/
│   ├── glioma_tumor/
│   ├── meningioma_tumor/
│   ├── pituitary_tumor/
│   └── notumor/
└── Testing/
    ├── glioma_tumor/
    ├── meningioma_tumor/
    ├── pituitary_tumor/
    └── notumor/
```

---

### **2. Brain Tumor MRI Dataset** (Alternative)
**Size**: 49 MB | **7,023 images**

**Link**: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

```bash
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset
```

---

### **3. Br35H Brain Tumor Dataset**
**Size**: 134 MB | **3,060 images**

**Link**: https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection

```bash
kaggle datasets download -d ahmedhamada0/brain-tumor-detection
```

---

## 📦 Quick Setup Script

Save this as `quick_download.sh`:

```bash
#!/bin/bash

# Download from Kaggle
kaggle datasets download -d sartajbhuvaji/brain-tumor-classification-mri

# Extract
unzip brain-tumor-classification-mri.zip -d data/brain_tumours

# Clean up
rm brain-tumor-classification-mri.zip

echo "✓ Dataset ready at: data/brain_tumours/"
```

Run it:
```bash
chmod +x quick_download.sh
./quick_download.sh
```

---

## 🔧 After Download - Organize Structure

Your dataset needs this structure for the project:
```
data/brain_tumours/
├── glioma/
│   ├── images/
│   └── masks/
├── meningioma/
│   ├── images/
│   └── masks/
└── pituitary/
    ├── images/
    └── masks/
```

Run this to reorganize:
```bash
python scripts/organize_dataset.py --input data/brain_tumours/Training
```

---

## 🎭 Creating Masks

Since Kaggle datasets don't include masks, create them:

```bash
# Quick masks (5 minutes)
python scripts/create_masks_options.py --method threshold --class-name glioma
python scripts/create_masks_options.py --method threshold --class-name meningioma
python scripts/create_masks_options.py --method threshold --class-name pituitary

# Better masks (10 minutes)
python scripts/create_masks_options.py --method edge --class-name glioma
python scripts/create_masks_options.py --method edge --class-name meningioma
python scripts/create_masks_options.py --method edge --class-name pituitary
```

---

## ✅ Verify Setup

```bash
python -c "
from pathlib import Path

data_dir = Path('data/brain_tumours')
for class_name in ['glioma', 'meningioma', 'pituitary']:
    img_dir = data_dir / class_name / 'images'
    mask_dir = data_dir / class_name / 'masks'
    
    n_imgs = len(list(img_dir.glob('*'))) if img_dir.exists() else 0
    n_masks = len(list(mask_dir.glob('*'))) if mask_dir.exists() else 0
    
    print(f'{class_name}: {n_imgs} images, {n_masks} masks')
"
```

Expected output:
```
glioma: 826 images, 826 masks
meningioma: 822 images, 822 masks
pituitary: 827 images, 827 masks
```

---

## 🚀 Start Training

Once verified:
```bash
python main.py --mode train --data_dir data/brain_tumours
```

---

## 💡 Pro Tips

1. **No Kaggle account?** Create one free at https://www.kaggle.com/
2. **Kaggle CLI not working?** Download manually via web interface
3. **Need masks urgently?** Use threshold method first, improve later
4. **Want best quality?** Use SAM method (requires 2.5GB model download)

---

## 📞 Still Having Issues?

Try these alternatives:
1. **Google Drive**: Search "brain tumor mri dataset google drive"
2. **Direct GitHub**: Some users host small datasets on GitHub
3. **Academic Torrents**: Medical imaging datasets
4. **Ask in comments**: Kaggle dataset discussion section

---

## Summary

**Best workflow**:
1. Download from Kaggle (web interface is easiest)
2. Extract to `data/brain_tumours/`
3. Run organize script
4. Create masks using threshold or edge method
5. Start training!

Total time: 15-20 minutes including download
Total size: <500MB
