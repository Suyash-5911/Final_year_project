# ✅ BRISC Dataset Successfully Downloaded!

**Date**: November 9, 2026  
**Source**: Figshare (https://doi.org/10.6084/m9.figshare.30533120)  
**Location**: `data/brisc/brisc2025/`

---

## 📊 Dataset Overview

**BRISC** (Brain Tumor Image Segmentation and Classification) is a comprehensive dataset featuring:

- **6,000** T1-weighted contrast-enhanced MRI images
- **Expert-annotated segmentation masks** by certified radiologists
- **4 classes**: Glioma, Meningioma, Pituitary, Non-tumorous
- **3 anatomical planes**: Axial, Coronal, Sagittal

---

## 📁 Dataset Structure

```
data/brisc/brisc2025/
├── classification_task/
│   ├── train/                   (5,000 images)
│   │   ├── glioma/             (1,147 images)
│   │   ├── meningioma/         (1,329 images)
│   │   ├── pituitary/          (1,457 images)
│   │   └── no_tumor/           (1,067 images)
│   └── test/                    (1,000 images)
│       ├── glioma/             (254 images)
│       ├── meningioma/         (306 images)
│       ├── pituitary/          (300 images)
│       └── no_tumor/           (140 images)
│
└── segmentation_task/
    ├── train/
    │   ├── images/              (3,933 images)
    │   └── masks/               (3,933 masks)
    └── test/
        ├── images/              (860 images)
        └── masks/               (860 masks)
```

---

## 🎯 Perfect for Your Project!

This dataset is **EXACTLY** what you need for your "Evidence-Aware Brain Tumour Classification System":

### ✅ Classification Task
- **3 tumor types**: Glioma, Meningioma, Pituitary ✓
- **Balanced distribution**: Well-distributed across classes ✓
- **Train/Test split**: Pre-split and ready to use ✓

### ✅ Segmentation Masks
- **Expert-annotated**: Certified radiologist verification ✓
- **4,793 total masks**: 3,933 train + 860 test ✓
- **Binary masks**: Tumor vs non-tumor regions ✓

### ✅ Evaluation Capabilities
- **Grad-CAM localization**: Compare heatmaps with ground truth masks ✓
- **IoU/Dice metrics**: Calculate localization accuracy ✓
- **Pointing Game**: Evaluate explanation quality ✓
- **Energy Ratio**: Measure focus on tumor regions ✓
- **Reliability scoring**: Complete evidence-aware assessment ✓

---

## 📏 Dataset Statistics

### Classification Task
| Class       | Training | Testing | Total  |
|-------------|----------|---------|--------|
| Glioma      | 1,147    | 254     | 1,401  |
| Meningioma  | 1,329    | 306     | 1,635  |
| Pituitary   | 1,457    | 300     | 1,757  |
| No Tumor    | 1,067    | 140     | 1,207  |
| **Total**   | **5,000**| **1,000**| **6,000** |

### Segmentation Task
| Split    | Images | Masks  |
|----------|--------|--------|
| Training | 3,933  | 3,933  |
| Testing  | 860    | 860    |
| **Total**| **4,793** | **4,793** |

**Note**: Segmentation task only includes tumor classes (excludes no_tumor)

---

## 🏆 Key Features

### 1. **Expert Annotation**
- Certified radiologists reviewed all annotations
- Mean Dice coefficient: 0.924 between initial and expert-verified masks
- Only 4.8% of images required correction
- Full consensus achieved on all annotations

### 2. **Multi-Plane Coverage**
- **Axial**: ~1,937 images (front-to-back view)
- **Coronal**: ~1,976 images (top-to-bottom view)
- **Sagittal**: ~2,087 images (side-to-side view)

### 3. **High Quality**
- T1-weighted contrast-enhanced sequences
- Consistent spatial dimensions
- No duplicates or corrupted files
- Patient-wise split to prevent data leakage

---

## 🚀 Next Steps

### 1. Update Your Configuration
```python
# In src/config.py
DATASET_PATH = 'data/brisc/brisc2025'
DATASET_TYPE = 'brisc'

# Dataset paths
CLASSIFICATION_PATH = 'data/brisc/brisc2025/classification_task'
SEGMENTATION_PATH = 'data/brisc/brisc2025/segmentation_task'
```

### 2. Create Dataset Loader
```python
from pathlib import Path
from PIL import Image
import numpy as np

class BRISCDataset:
    def __init__(self, root_dir, task='classification', split='train'):
        """
        Args:
            root_dir: Path to brisc2025/
            task: 'classification' or 'segmentation'
            split: 'train' or 'test'
        """
        self.root = Path(root_dir)
        self.task = task
        self.split = split
        
        if task == 'classification':
            self.data_dir = self.root / 'classification_task' / split
            self.classes = ['glioma', 'meningioma', 'pituitary', 'no_tumor']
            self.load_classification_data()
        else:
            self.data_dir = self.root / 'segmentation_task' / split
            self.load_segmentation_data()
    
    def load_classification_data(self):
        """Load classification task images and labels"""
        self.samples = []
        for class_idx, class_name in enumerate(self.classes):
            class_dir = self.data_dir / class_name
            for img_path in class_dir.glob('*.jpg'):
                self.samples.append((img_path, class_idx, class_name))
    
    def load_segmentation_data(self):
        """Load segmentation task images and masks"""
        self.samples = []
        img_dir = self.data_dir / 'images'
        mask_dir = self.data_dir / 'masks'
        
        for img_path in sorted(img_dir.glob('*.jpg')):
            # Mask has same basename but .png extension
            mask_name = img_path.stem + '.png'
            mask_path = mask_dir / mask_name
            
            if mask_path.exists():
                self.samples.append((img_path, mask_path))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if self.task == 'classification':
            img_path, label, class_name = self.samples[idx]
            image = Image.open(img_path).convert('RGB')
            return image, label, class_name
        else:
            img_path, mask_path = self.samples[idx]
            image = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')  # Grayscale
            return image, mask, img_path.stem

# Usage example
dataset = BRISCDataset('data/brisc/brisc2025', task='segmentation', split='train')
print(f"Loaded {len(dataset)} samples")

# Get first sample
image, mask, filename = dataset[0]
print(f"Image size: {image.size}")
print(f"Mask size: {mask.size}")
print(f"Filename: {filename}")
```

### 3. Test Data Loading
```bash
python -c "
from pathlib import Path
from PIL import Image

# Test classification
class_img = Path('data/brisc/brisc2025/classification_task/train/glioma').glob('*.jpg').__next__()
img = Image.open(class_img)
print(f'Classification image: {img.size} {img.mode}')

# Test segmentation
seg_img = Path('data/brisc/brisc2025/segmentation_task/train/images').glob('*.jpg').__next__()
seg_mask = Path('data/brisc/brisc2025/segmentation_task/train/masks').glob('*.png').__next__()
img = Image.open(seg_img)
mask = Image.open(seg_mask)
print(f'Segmentation image: {img.size} {img.mode}')
print(f'Segmentation mask: {mask.size} {mask.mode}')
"
```

### 4. Start Training
```bash
# Train CNN classifier
python main.py --mode train --dataset brisc --epochs 50

# Evaluate with masks
python main.py --mode eval --dataset brisc --use-masks

# Generate Grad-CAM with localization metrics
python demo_single_prediction.py --image path/to/image.jpg --mask path/to/mask.png
```

---

## 📚 Citation

When using this dataset, please cite:

```bibtex
@article{fateh2025brisc,
  title={BRISC: Annotated Dataset for Brain Tumor Segmentation and Classification},
  author={Fateh, Amirreza and Rezvani, Yasin and Moayedi, Sara and Rezvani, Sadjad 
          and Fateh, Fatemeh and Fateh, Mansoor and Abolghasemi, Vahid},
  journal={arXiv preprint arXiv:2506.14318v4},
  year={2025}
}
```

---

## 📄 Dataset Files

Additional files downloaded:
- `manifest.csv`: Metadata for all images
- `manifest.json`: JSON format metadata
- `README.md`: Dataset documentation

---

## ✅ Verification Checklist

- [x] Dataset downloaded (260 MB)
- [x] Files extracted (15,605 files)
- [x] Classification task: 6,000 images ✓
- [x] Segmentation task: 4,793 image-mask pairs ✓
- [x] Expert-annotated masks included ✓
- [x] Train/test splits ready ✓
- [x] 3 tumor classes available ✓
- [x] Ready for Grad-CAM evaluation ✓

---

## 🎉 Success!

You now have a **high-quality, expertly-annotated brain tumor dataset** with:

1. ✅ **Real ground truth masks** for localization evaluation
2. ✅ **3 tumor types** matching your project requirements
3. ✅ **Balanced distribution** across classes
4. ✅ **Multi-plane coverage** for robust training
5. ✅ **Ready-to-use splits** for immediate development

**You're all set to implement your Evidence-Aware Brain Tumour Classification System!** 🚀

---

## 📞 Support

- **Dataset Homepage**: https://www.kaggle.com/datasets/briscdataset/brisc2025
- **Paper**: https://arxiv.org/abs/2506.14318v4
- **Issues**: Report on dataset homepage

---

**Last Updated**: November 9, 2026, 12:09 PM IST
