# BraTS 2021 Dataset Download Guide

## Current Status
✅ Successfully authenticated with Synapse API  
⚠️ The main Synapse page (syn27046444) appears to be a documentation/challenge page rather than the actual dataset repository

## Alternative Download Methods

### Option 1: Official BraTS Challenge Portal (Recommended)
The BraTS 2021 dataset is typically distributed through the official challenge portal:

1. **Visit**: https://www.synapse.org/#!Synapse:syn27046444/wiki/616571
2. **Register** for the BraTS 2021 Challenge
3. **Navigate to** the "Files" tab or "Data" section
4. Look for folders named:
   - `BraTS2021_Training_Data`
   - `BraTS2021_Validation_Data`
   - `BraTS2021_Testing_Data`

### Option 2: Direct Synapse File IDs
If you can find the specific Synapse File IDs for the training/validation data archives, you can download them directly using:

```python
import synapseclient
syn = synapseclient.Synapse()
syn.login(authToken='YOUR_TOKEN')

# Example (replace with actual file IDs)
# syn.get('synXXXXXXX', downloadLocation='data/brats/brats_2021/')
```

### Option 3: Use synapseclient sync command
Try syncing the entire folder recursively:

```bash
synapse get -r syn27046444 --downloadLocation data/brats/brats_2021/
```

### Option 4: Kaggle or Academic Mirrors
BraTS datasets are sometimes mirrored on:
- **Kaggle**: Search for "BraTS 2021" datasets
- **Medical Segmentation Decathlon**: http://medicaldecathlon.com/
- **Academic institution mirrors**: Check your university's data access

## Alternative: Use BraTS 2020 (Smaller, Easier Access)

BraTS 2020 is often easier to download and still excellent for your project:

```python
# BraTS 2020 Synapse ID: syn25829067
# Try this if BraTS 2021 is inaccessible
```

## For Your Project - Alternative Datasets

Since your project focuses on **brain tumour classification with Grad-CAM**, consider these alternatives:

### 1. **Brain Tumor MRI Dataset (Kaggle)**
- **Size**: ~250 MB (much smaller!)
- **Classes**: Glioma, Meningioma, Pituitary, No Tumor
- **Format**: JPG images ready to use
- **Link**: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
- **Includes**: ~7,000 images with bounding boxes/masks available

### 2. **Br35H Brain Tumor Detection**
- **Size**: ~400 MB
- **Format**: PNG/JPG
- **Link**: https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection

### 3. **TCIA Brain Tumor Collections**
- The Cancer Imaging Archive has multiple brain tumor datasets
- **Link**: https://www.cancerimagingarchive.net/

## Recommended Action for Your Timeline

Given your project timeline (currently Nov 2026, dataset prep deadline Dec 2026), I recommend:

1. **Start with Kaggle Brain Tumor MRI Dataset** (quick download, ready to use)
2. **Continue pursuing BraTS 2021** in parallel for ground truth masks
3. **Use Kaggle dataset** to:
   - Develop your CNN classifier
   - Implement Grad-CAM
   - Test your pipeline end-to-end
4. **Integrate BraTS** once downloaded for:
   - Better ground truth evaluation
   - Expert-annotated masks
   - Publication-quality results

## Download Kaggle Dataset (Quick Start)

```bash
# Install Kaggle API
pip install kaggle

# Set up Kaggle credentials (get from kaggle.com/account)
mkdir -p ~/.kaggle
# Place your kaggle.json in ~/.kaggle/

# Download Brain Tumor MRI Dataset
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset
unzip brain-tumor-mri-dataset.zip -d data/brain_tumor_mri/
```

## Next Steps

1. Check if BraTS 2021 requires additional challenge registration
2. Consider starting with Kaggle dataset for immediate development
3. Continue BraTS download process in background
4. Update dataset loader to work with whichever dataset is available first

---
**Note**: The Synapse token you provided is valid and working - the issue is finding the correct Synapse ID for the actual data files rather than the challenge documentation page.
