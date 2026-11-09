"""
Diagnostic script to analyze the mask generation problem
"""
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

print("=" * 80)
print("DIAGNOSTIC ANALYSIS: MASK GENERATION PROBLEM")
print("=" * 80)

def analyze_mask_quality(tumor_type, num_samples=5):
    """Analyze multiple masks to identify the problem"""
    
    print(f"\n{'=' * 80}")
    print(f"ANALYZING {tumor_type.upper()} MASKS")
    print(f"{'=' * 80}")
    
    img_dir = Path(f'data/brain_tumours/{tumor_type}/images')
    mask_dir = Path(f'data/brain_tumours/{tumor_type}/masks')
    
    images = sorted(list(img_dir.glob('*.jpg')))[:num_samples]
    
    for idx, img_path in enumerate(images, 1):
        mask_path = mask_dir / img_path.name
        
        if not mask_path.exists():
            print(f"⚠ Mask not found: {mask_path.name}")
            continue
        
        # Load image and mask
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Analyze mask
        white_pixels = (mask > 127).sum()
        total_pixels = mask.size
        white_pct = 100 * white_pixels / total_pixels
        
        # Analyze image intensity
        img_mean = image.mean()
        img_std = image.std()
        img_max = image.max()
        img_min = image.min()
        
        # Count contours in mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = len(contours)
        
        if contours:
            largest_contour_area = max([cv2.contourArea(c) for c in contours])
            largest_contour_pct = 100 * largest_contour_area / total_pixels
        else:
            largest_contour_pct = 0
        
        print(f"\n📄 Sample {idx}: {img_path.name}")
        print(f"   Image stats:")
        print(f"      Mean intensity: {img_mean:.1f}")
        print(f"      Std dev: {img_std:.1f}")
        print(f"      Range: {img_min} - {img_max}")
        print(f"   Mask stats:")
        print(f"      White pixels: {white_pct:.1f}%")
        print(f"      Number of contours: {num_contours}")
        print(f"      Largest contour: {largest_contour_pct:.1f}%")
        print(f"      Unique values: {len(np.unique(mask))}")
        
        # DIAGNOSIS
        if white_pct > 80:
            print(f"   🚨 PROBLEM: Mask covers {white_pct:.1f}% - ENTIRE BRAIN DETECTED!")
            print(f"      This is brain extraction, NOT tumor segmentation")
        elif white_pct > 60:
            print(f"   ⚠ WARNING: Mask covers {white_pct:.1f}% - likely too large")
        elif white_pct < 5:
            print(f"   ⚠ WARNING: Mask covers only {white_pct:.1f}% - might be too small")
        else:
            print(f"   ✓ Mask coverage looks reasonable: {white_pct:.1f}%")

# Analyze all tumor types
for tumor_type in ['glioma', 'meningioma', 'pituitary']:
    analyze_mask_quality(tumor_type, num_samples=5)

print("\n" + "=" * 80)
print("ROOT CAUSE ANALYSIS")
print("=" * 80)

print("""
Based on the diagnostic analysis, the problem is likely:

ISSUE: Otsu's thresholding is detecting the ENTIRE BRAIN, not just the tumor

WHY THIS HAPPENS:
1. Brain MRI images have bright brain tissue (white/gray matter)
2. Background around brain is dark (black)
3. Otsu's thresholding separates "brain vs background", NOT "tumor vs brain"
4. Result: The mask highlights the ENTIRE brain region

WHAT THE MASKS ACTUALLY SHOW:
✗ NOT tumor segmentation masks
✓ Brain extraction masks (skull stripping)

THE FUNDAMENTAL PROBLEM:
- Your dataset has classification labels (glioma, meningioma, etc.)
- Your dataset does NOT have pixel-level tumor annotations
- Thresholding cannot distinguish tumor from healthy brain tissue
- Both tumor and brain appear bright on MRI scans

SOLUTION OPTIONS:
1. Accept that ground truth masks don't exist for this dataset
2. Use a different dataset that includes segmentation masks (BraTS, LGG)
3. Use Grad-CAM predictions as pseudo-labels (but can't evaluate against them)
4. Train a U-Net using a segmentation dataset, then apply to your data
5. Manually annotate tumor boundaries (time-consuming)
""")

# Create visual diagnostic
print("\n" + "=" * 80)
print("CREATING VISUAL DIAGNOSTIC")
print("=" * 80)

for tumor_type in ['glioma']:  # Just show one example
    img_dir = Path(f'data/brain_tumours/{tumor_type}/images')
    mask_dir = Path(f'data/brain_tumours/{tumor_type}/masks')
    
    images = sorted(list(img_dir.glob('*.jpg')))[:3]
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    for idx, img_path in enumerate(images):
        mask_path = mask_dir / img_path.name
        
        # Load
        image = cv2.imread(str(img_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Original image
        axes[idx, 0].imshow(image_rgb)
        axes[idx, 0].set_title(f'Sample {idx+1}: Original MRI')
        axes[idx, 0].axis('off')
        
        # Generated mask
        axes[idx, 1].imshow(mask, cmap='gray')
        white_pct = 100 * (mask > 127).sum() / mask.size
        axes[idx, 1].set_title(f'Generated Mask ({white_pct:.1f}% white)')
        axes[idx, 1].axis('off')
        
        # Overlay
        overlay = image_rgb.copy()
        red_mask = np.zeros_like(image_rgb)
        red_mask[mask > 127] = [255, 0, 0]
        overlay = cv2.addWeighted(overlay, 0.7, red_mask, 0.3, 0)
        axes[idx, 2].imshow(overlay)
        axes[idx, 2].set_title('Overlay (Red = Mask)')
        axes[idx, 2].axis('off')
    
    plt.suptitle('DIAGNOSTIC: Mask Generation Problem\\n' +
                 'Notice: Red overlay covers ENTIRE BRAIN, not just tumor',
                 fontsize=14, fontweight='bold', color='red')
    plt.tight_layout()
    plt.savefig('diagnostic_mask_problem.png', dpi=150, bbox_inches='tight')
    print("✓ Saved diagnostic_mask_problem.png")
    plt.close()

print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)
print("""
YOUR DATASET (Kaggle Brain Tumor Classification MRI):
   ✓ Has classification labels (glioma, meningioma, pituitary, no tumor)
   ✗ Does NOT have pixel-level tumor segmentation masks

WHAT TO DO:
1. Use Grad-CAM for visualization ONLY (not as ground truth)
2. Focus on classification accuracy metrics
3. If localization evaluation is required:
   - Download BraTS or LGG dataset (has real masks)
   - Or remove localization metrics from your project scope
   
DO NOT:
   ✗ Use these thresholded masks as "ground truth"
   ✗ Calculate IoU/Dice scores against these masks
   ✗ Claim these represent actual tumor boundaries
""")

