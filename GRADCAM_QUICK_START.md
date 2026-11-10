# Grad-CAM Quick Start Guide 🚀

## 1-Minute Setup

### Upload to Google Colab

**Required Files:**
```
GradCAM_Testing_Colab.ipynb    ← Main notebook
models/best_model/             ← Trained weights
src/                           ← Source code
data/brisc/processed/test/     ← Test images (optional: subset)
```

### Run in 3 Steps

1. **Open in Colab**: Upload `GradCAM_Testing_Colab.ipynb`

2. **Select GPU Runtime**: 
   - Runtime → Change runtime type → GPU → Save

3. **Run All Cells**: 
   - Runtime → Run all (or Ctrl+F9)

### Expected Runtime
- Single image test: ~30 seconds
- Full test set (15 samples): ~5-10 minutes
- Download: ~1 minute

---

## What You'll Get

### Visualizations
- Individual Grad-CAM overlays for each image
- 6-panel comparison (image, heatmap, overlay, probabilities, mask, metrics)
- Grid summary of all samples

### Metrics
- Classification accuracy per class
- Localization scores (IoU, Dice, Pointing Game, Energy Ratio)
- JSON file with all results

### Downloads
- `gradcam_results.zip` containing all visualizations and results

---

## Quick Modifications

### Test on Fewer Images
In cell with `sample_images`:
```python
sample_images[class_name] = image_files[:2]  # Change 5 to 2
```

### Change Colormap
In Grad-CAM initialization:
```python
explainer = ExplanationGenerator(
    model=model,
    colormap='hot',  # Change from 'jet' to 'hot'
    device=device
)
```

### Adjust Overlay Transparency
In visualization calls:
```python
overlay = overlay_heatmap(original_image, cam, alpha=0.6)  # Change 0.4 to 0.6
```

---

## Troubleshooting One-Liners

**"No module named 'src'"**
```python
import sys; sys.path.insert(0, 'src')
```

**"CUDA out of memory"**
```python
# Reduce sample size or restart runtime
```

**"No masks found"**
```
This is expected - notebook will work without masks
```

---

## Expected Results

### Classification
- Accuracy: ~99% (based on training results)
- High confidence: >95% for most predictions

### Localization (if masks available)
- IoU: 0.5-0.8 (good to excellent)
- Dice: 0.6-0.85
- Pointing Game: >80% hits

### Visual Quality
- Heatmaps should highlight tumor regions
- Overlays should show clear activation on lesions
- Minimal activation in healthy tissue

---

## After Running

1. **Review** visualizations manually
2. **Check** metrics in JSON file
3. **Download** results zip
4. **Document** findings
5. **Move to** Increment 3 (Stability Evaluation)

---

## Need More Details?

See `GRADCAM_IMPLEMENTATION_COMPLETE.md` for:
- Detailed technical documentation
- Full API reference
- Troubleshooting guide
- Next steps

---

**Ready?** Upload to Colab and click "Run All"! ▶️
