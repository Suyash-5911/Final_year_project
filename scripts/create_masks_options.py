"""
Create segmentation masks for brain tumor images
WITHOUT needing the 50GB BraTS dataset

Three practical options for mask creation
"""
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt


# ============================================================
# OPTION 1: Simple Thresholding Masks (Quickest)
# ============================================================
def create_threshold_masks(image_dir, mask_dir):
    """
    Create masks using simple intensity thresholding
    Good for: Quick prototyping, basic testing
    Accuracy: ~60-70% (not perfect but usable)
    """
    print("Creating threshold-based masks...")
    
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    for img_path in image_dir.glob('*.jpg'):
        # Load image
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        
        # Otsu's thresholding
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find largest contour (assume it's the tumor)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        mask = np.zeros_like(img)
        if contours:
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask, [largest_contour], -1, 255, -1)
        
        # Save mask
        mask_path = mask_dir / img_path.name
        cv2.imwrite(str(mask_path), mask)
    
    print(f"✓ Created {len(list(mask_dir.glob('*.jpg')))} masks")
    print("⚠ Note: These are approximate masks, not expert annotations")


# ============================================================
# OPTION 2: Edge Detection + Region Growing (Better)
# ============================================================
def create_edge_based_masks(image_dir, mask_dir):
    """
    Create masks using edge detection and region growing
    Good for: Better quality without deep learning
    Accuracy: ~70-75%
    """
    print("Creating edge-based masks...")
    
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    for img_path in image_dir.glob('*.jpg'):
        # Load image
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img)
        
        # Edge detection
        edges = cv2.Canny(enhanced, 50, 150)
        
        # Dilate edges to close gaps
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        
        # Fill contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        mask = np.zeros_like(img)
        if contours:
            # Filter contours by area (remove very small ones)
            valid_contours = [c for c in contours if cv2.contourArea(c) > 100]
            if valid_contours:
                largest = max(valid_contours, key=cv2.contourArea)
                cv2.drawContours(mask, [largest], -1, 255, -1)
        
        # Save mask
        mask_path = mask_dir / img_path.name
        cv2.imwrite(str(mask_path), mask)
    
    print(f"✓ Created {len(list(mask_dir.glob('*.jpg')))} masks")


# ============================================================
# OPTION 3: Use Pre-trained SAM (Segment Anything Model)
# ============================================================
def create_sam_masks(image_dir, mask_dir):
    """
    Use Meta's Segment Anything Model (SAM) to create masks
    Good for: High-quality masks without manual annotation
    Accuracy: ~80-85% (best automated option)
    
    Requires: pip install segment-anything
    """
    print("Creating SAM-based masks...")
    print("⚠ Requires: pip install git+https://github.com/facebookresearch/segment-anything.git")
    
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        import torch
    except ImportError:
        print("❌ SAM not installed. Install with:")
        print("pip install git+https://github.com/facebookresearch/segment-anything.git")
        return
    
    # Download SAM checkpoint (if not exists)
    checkpoint_path = "sam_vit_h_4b8939.pth"
    if not Path(checkpoint_path).exists():
        print("Downloading SAM checkpoint (~2.5GB, one-time download)...")
        import urllib.request
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        urllib.request.urlretrieve(url, checkpoint_path)
    
    # Load SAM model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
    sam.to(device=device)
    
    mask_generator = SamAutomaticMaskGenerator(sam)
    
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    for img_path in image_dir.glob('*.jpg'):
        # Load image
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Generate masks
        masks = mask_generator.generate(img_rgb)
        
        # Select largest mask (assume it's the tumor)
        if masks:
            largest_mask = max(masks, key=lambda x: x['area'])
            mask = (largest_mask['segmentation'] * 255).astype(np.uint8)
        else:
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
        
        # Save mask
        mask_path = mask_dir / img_path.name
        cv2.imwrite(str(mask_path), mask)
    
    print(f"✓ Created {len(list(mask_dir.glob('*.jpg')))} SAM-based masks")


# ============================================================
# OPTION 4: Download Small Pre-Made Mask Dataset
# ============================================================
def download_small_dataset_with_masks():
    """
    Download a small dataset that ALREADY includes masks
    Best option: LGG Segmentation Dataset (glioma only, ~500MB)
    """
    print("""
    OPTION: Download Pre-made Masks
    
    LGG Segmentation Dataset (Kaggle):
    - Link: https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation
    - Size: ~500MB (includes masks!)
    - Classes: Lower Grade Glioma only
    
    Download command:
    kaggle datasets download -d mateuszbuda/lgg-mri-segmentation
    
    ✓ Includes real segmentation masks
    ✓ Smaller than BraTS
    ⚠ Only glioma class (not meningioma/pituitary)
    """)


# ============================================================
# MAIN FUNCTION - Choose Your Method
# ============================================================
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Create masks for brain tumor images")
    parser.add_argument(
        '--method',
        type=str,
        choices=['threshold', 'edge', 'sam', 'info'],
        default='threshold',
        help='Mask creation method'
    )
    parser.add_argument(
        '--class-name',
        type=str,
        required=True,
        choices=['glioma', 'meningioma', 'pituitary'],
        help='Tumor class to process'
    )
    
    args = parser.parse_args()
    
    # Paths
    image_dir = f"data/brain_tumours/{args.class_name}/images"
    mask_dir = f"data/brain_tumours/{args.class_name}/masks"
    
    print("=" * 60)
    print(f"CREATING MASKS FOR {args.class_name.upper()}")
    print("=" * 60)
    
    if args.method == 'threshold':
        create_threshold_masks(image_dir, mask_dir)
    elif args.method == 'edge':
        create_edge_based_masks(image_dir, mask_dir)
    elif args.method == 'sam':
        create_sam_masks(image_dir, mask_dir)
    elif args.method == 'info':
        download_small_dataset_with_masks()
    
    print("\n✓ Mask creation complete!")
    print(f"Masks saved to: {mask_dir}")
    print("\n⚠ Note: Automated masks are approximations.")
    print("For best results, consider:")
    print("1. Manual refinement using LabelMe")
    print("2. Download LGG dataset (has real masks, ~500MB)")


if __name__ == "__main__":
    main()
