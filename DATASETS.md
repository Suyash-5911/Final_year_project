# Brain Tumour Datasets Guide

## 📊 Required Data

For this project, you need:
1. **MRI brain tumour images** (3 classes: glioma, meningioma, pituitary)
2. **Segmentation masks** (tumour region annotations for localization evaluation)

---

## 🏆 Recommended Datasets

### Option 1: Figshare Dataset (Quick Start) ⭐
**Link**: https://figshare.com/articles/dataset/brain_tumor_dataset/1512427

**Download**:
```bash
# Manual download from link above
# Or use wget:
wget https://figshare.com/ndownloader/files/3381289 -O brain_tumor_dataset.zip
unzip brain_tumor_dataset.zip -d data/brain_tumours
```

**Specs**:
- 3,064 images total
- Classes: Glioma (1426), Meningioma (708), Pituitary (930)
- Format: JPG images
- Size: ~150MB

**Note**: No masks included. Options:
1. Use dummy masks for development (see below)
2. Download BraTS for real masks
3. Annotate manually using LabelMe

---

### Option 2: BraTS Dataset (With Real Masks) ⭐⭐⭐
**Link**: http://braintumorsegmentation.org/
**Registration**: https://www.synapse.org/#!Synapse:syn25829067

**Download**:
```bash
# After registration and approval:
# Download via Synapse interface or:
synapse get -r syn25829067
```

**Specs**:
- 2,000+ multi-modal MRI scans
- **Includes expert segmentation masks**
- Modalities: T1, T2, FLAIR, T1ce
- Format: NIfTI (.nii.gz)
- Size: ~50GB

**Preprocessing needed**:
```python
# Convert NIfTI to PNG and extract 2D slices
import nibabel as nib
from PIL import Image

# Load 3D volume
img = nib.load('BraTS_001_t1ce.nii.gz')
data = img.get_fdata()

# Extract middle slice
slice_2d = data[:, :, data.shape[2]//2]
Image.fromarray(slice_2d).save('slice.png')
```

---

### Option 3: Kaggle Brain MRI Dataset
**Link**: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

**Download**:
```bash
# Install Kaggle CLI:
pip install kaggle

# Setup API key (from Kaggle account settings)
# Download:
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset
unzip brain-tumor-mri-dataset.zip -d data/brain_tumours
```

**Specs**:
- 7,023 images
- 4 classes (includes "No Tumor")
- Pre-split train/test
- No masks

---

### Option 4: LGG Segmentation Dataset (Glioma Only)
**Link**: https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation

**Download**:
```bash
kaggle datasets download -d mateuszbuda/lgg-mri-segmentation
unzip lgg-mri-segmentation.zip -d data/brain_tumours
```

**Specs**:
- 3,929 images
- **Includes masks** (Lower Grade Glioma only)
- Format: TIFF
- Good for glioma-specific analysis

---

## 🛠️ Quick Setup

### Method 1: Automated Download
```bash
# Download Figshare dataset
python scripts/download_dataset.py --source figshare --output data/brain_tumours

# Create dummy masks for testing
python scripts/download_dataset.py --create-dummy-masks
```

### Method 2: Manual Download

1. **Download dataset** from any link above
2. **Extract** to `data/brain_tumours/`
3. **Organize structure**:
```
data/brain_tumours/
├── glioma/
│   ├── images/
│   │   ├── img001.jpg
│   │   └── ...
│   └── masks/
│       ├── img001.jpg  # Binary mask
│       └── ...
├── meningioma/
│   ├── images/
│   └── masks/
└── pituitary/
    ├── images/
    └── masks/
```

---

## 🎭 Creating Masks

### Option A: Dummy Masks (Testing Only)
```python
python scripts/download_dataset.py --create-dummy-masks
```

### Option B: Grad-CAM Pseudo-Masks
```python
# Train model first without masks
python main.py --mode train

# Generate pseudo-masks using Grad-CAM
python scripts/generate_pseudo_masks.py
```

### Option C: Manual Annotation
Use LabelMe: https://github.com/wkentaro/labelme
```bash
pip install labelme
labelme data/brain_tumours/glioma/images
```

### Option D: Use BraTS Real Masks (Recommended)
Download BraTS dataset with expert annotations.

---

## 📏 Expected Dataset Size

| Dataset | Images | Classes | Masks | Size |
|---------|--------|---------|-------|------|
| Figshare | 3,064 | 3 | No | 150MB |
| BraTS | 2,000+ | Multi | Yes | 50GB |
| Kaggle MRI | 7,023 | 4 | No | 100MB |
| LGG Seg | 3,929 | 1 | Yes | 2GB |

---

## ✅ Verify Dataset

After setup, verify structure:
```bash
python -c "
from src.data import DatasetLoader
loader = DatasetLoader('data/brain_tumours')
loader.load_slices_and_masks()
stats = loader.get_dataset_statistics()
print(stats)
"
```

Expected output:
```
{
  'total_samples': 3064,
  'train_samples': 2144,
  'val_samples': 460,
  'test_samples': 460,
  'class_distribution': {...}
}
```

---

## 🔗 Additional Resources

- **Brain Tumor Dataset Papers**: https://www.nature.com/articles/sdata2017117
- **BraTS Challenge**: https://www.med.upenn.edu/cbica/brats2021/
- **Medical Image Datasets**: https://www.kaggle.com/datasets?search=brain+tumor
- **Preprocessing Tools**: https://nipy.org/nibabel/

---

## 💡 My Recommendation

**For Development/Testing**: Use Figshare + dummy masks
**For Final Project**: Use BraTS with real expert masks
**For Quick Results**: Use LGG Segmentation (glioma only)

---

## 📞 Need Help?

If you have issues:
1. Check dataset format matches expected structure
2. Verify file paths in `src/config.py`
3. Run dataset verification script
4. Check QUICKSTART.md for troubleshooting
