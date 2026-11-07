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
    Download Brain Tumor Dataset from Figshare
    """
    print("=" * 60)
    print("DOWNLOADING FIGSHARE BRAIN TUMOR DATASET")
    print("=" * 60)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Figshare download URLs
    urls = {
        "training": "https://figshare.com/ndownloader/files/3381289",
        # Add more if available
    }
    
    for name, url in urls.items():
        print(f"\nDownloading {name} dataset...")
        zip_path = output_path / f"{name}.zip"
        
        try:
            urllib.request.urlretrieve(url, zip_path)
            print(f"✓ Downloaded to {zip_path}")
            
            # Extract
            print(f"Extracting {name}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_path)
            print(f"✓ Extracted to {output_path}")
            
            # Clean up zip
            os.remove(zip_path)
            print(f"✓ Cleaned up {zip_path}")
            
        except Exception as e:
            print(f"✗ Error downloading {name}: {e}")
            print(f"Please download manually from: {url}")
    
    print("\n✓ Dataset download complete!")
    print(f"Location: {output_path.absolute()}")


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
