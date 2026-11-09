"""
Download and prepare brain tumour dataset
Supports Figshare and Kaggle datasets
"""
import os
import zipfile
import urllib.request
from pathlib import Path
import shutil


def download_figshare_dataset(output_dir="data/brain_tumours"):
    """
    Download Brain Tumor Dataset from Kaggle (Most Reliable)
    """
    print("=" * 60)
    print("DOWNLOADING BRAIN TUMOR DATASET FROM KAGGLE")
    print("=" * 60)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n⚠ MANUAL DOWNLOAD REQUIRED:")
    print("=" * 60)
    print("\n1. Visit: https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri")
    print("2. Click 'Download' button (requires Kaggle account)")
    print("3. Extract the downloaded zip file")
    print("4. Move contents to:", output_path.absolute())
    
    print("\n--- OR USE KAGGLE CLI ---")
    print("\nRun these commands:")
    print("  pip install kaggle")
    print("  kaggle datasets download -d sartajbhuvaji/brain-tumor-classification-mri")
    print(f"  unzip brain-tumor-classification-mri.zip -d {output_dir}")
    
    print("\n--- KAGGLE API SETUP ---")
    print("\n1. Go to: https://www.kaggle.com/settings")
    print("2. Scroll to 'API' section")
    print("3. Click 'Create New API Token'")
    print("4. Save kaggle.json to:")
    print("   Windows: C:\\Users\\<username>\\.kaggle\\kaggle.json")
    print("   Linux/Mac: ~/.kaggle/kaggle.json")
    
    print("\n✓ Follow instructions above to download dataset")
    print(f"Target location: {output_path.absolute()}")


def organize_dataset_structure(source_dir, output_dir="data/brain_tumours"):
    """
    Organize dataset into required structure:
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
    """
    print("\n" + "=" * 60)
    print("ORGANIZING DATASET STRUCTURE")
    print("=" * 60)
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Create structure
    classes = ['glioma', 'meningioma', 'pituitary']
    
    for cls in classes:
        images_dir = output_path / cls / 'images'
        masks_dir = output_path / cls / 'masks'
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created {cls} directories")
    
    # Move images (you'll need to adapt this based on actual structure)
    print("\n⚠ Note: Mask directories are empty.")
    print("Options:")
    print("1. Download BraTS dataset for real masks")
    print("2. Use Grad-CAM to generate pseudo-masks")
    print("3. Manually annotate using tools like LabelMe")
    
    print(f"\n✓ Dataset organized in: {output_path.absolute()}")


def create_dummy_masks(dataset_dir="data/brain_tumours"):
    """
    Create dummy masks for testing
    (Replace with real masks for production)
    """
    print("\n" + "=" * 60)
    print("CREATING DUMMY MASKS FOR TESTING")
    print("=" * 60)
    
    from PIL import Image
    import numpy as np
    
    dataset_path = Path(dataset_dir)
    
    for class_dir in dataset_path.iterdir():
        if not class_dir.is_dir():
            continue
        
        images_dir = class_dir / 'images'
        masks_dir = class_dir / 'masks'
        
        if not images_dir.exists():
            continue
        
        print(f"\nProcessing {class_dir.name}...")
        
        for img_file in images_dir.glob('*.jpg'):
            # Create corresponding dummy mask
            mask_file = masks_dir / img_file.name
            
            if mask_file.exists():
                continue
            
            # Load image to get size
            img = Image.open(img_file)
            width, height = img.size
            
            # Create centered circular mask (dummy)
            mask = np.zeros((height, width), dtype=np.uint8)
            center_x, center_y = width // 2, height // 2
            radius = min(width, height) // 4
            
            y, x = np.ogrid[:height, :width]
            circle_mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            mask[circle_mask] = 255
            
            # Save mask
            mask_img = Image.fromarray(mask, mode='L')
            mask_img.save(mask_file)
        
        print(f"✓ Created {len(list(masks_dir.glob('*.jpg')))} dummy masks")
    
    print("\n⚠ WARNING: These are DUMMY masks for testing only!")
    print("For accurate localization evaluation, use real expert annotations.")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Download and prepare brain tumour dataset")
    parser.add_argument(
        '--source',
        type=str,
        choices=['figshare', 'kaggle', 'organize'],
        default='figshare',
        help='Dataset source'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/brain_tumours',
        help='Output directory'
    )
    parser.add_argument(
        '--create-dummy-masks',
        action='store_true',
        help='Create dummy masks for testing'
    )
    
    args = parser.parse_args()
    
    if args.source == 'figshare':
        download_figshare_dataset(args.output)
    elif args.source == 'organize':
        organize_dataset_structure('.', args.output)
    
    if args.create_dummy_masks:
        create_dummy_masks(args.output)
    
    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 60)
    print(f"\nDataset location: {Path(args.output).absolute()}")
    print("\nNext steps:")
    print("1. Verify dataset structure")
    print("2. Run: python main.py --mode train")


if __name__ == "__main__":
    main()
