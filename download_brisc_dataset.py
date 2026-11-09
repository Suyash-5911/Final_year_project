#!/usr/bin/env python3
"""
Download BRISC Dataset - Brain Tumor Segmentation and Classification
===================================================================

BRISC Dataset Specifications:
- 6,000 T1-weighted contrast-enhanced MRI images
- 4 classes: Glioma, Meningioma, Pituitary, Non-tumorous
- Expert-annotated segmentation masks by certified radiologists
- 3 anatomical planes: Axial, Coronal, Sagittal
- Size: ~2-3 GB

Dataset Sources:
1. Kaggle: https://www.kaggle.com/datasets/briscdataset/brisc2025/
2. Figshare: https://doi.org/10.6084/m9.figshare.30533120
3. Zenodo: https://doi.org/10.5281/zenodo.17524350

Citation:
Fateh, A., Rezvani, Y., Moayedi, S., et al. (2025). 
BRISC: Annotated Dataset for Brain Tumor Segmentation and Classification.
arXiv:2506.14318v4
"""

import requests
import os
import zipfile
from pathlib import Path
from tqdm import tqdm
import argparse


def download_file(url, output_path, chunk_size=8192):
    """
    Download a file with progress bar
    
    Args:
        url: Download URL
        output_path: Output file path
        chunk_size: Download chunk size
    """
    response = requests.get(url, stream=True, allow_redirects=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=output_path.name) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    print(f"\n✓ Downloaded to: {output_path}")
    return output_path


def extract_zip(zip_path, extract_to):
    """
    Extract zip file with progress
    
    Args:
        zip_path: Path to zip file
        extract_to: Extraction directory
    """
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)
    
    print(f"\nExtracting {zip_path.name}...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = zip_ref.namelist()
        with tqdm(total=len(members), desc="Extracting") as pbar:
            for member in members:
                zip_ref.extract(member, extract_to)
                pbar.update(1)
    
    print(f"✓ Extracted to: {extract_to}")


def download_brisc_figshare(output_dir='data/brisc'):
    """
    Download BRISC dataset from Figshare
    
    Args:
        output_dir: Output directory
    """
    print("=" * 80)
    print("DOWNLOADING BRISC DATASET FROM FIGSHARE")
    print("=" * 80)
    
    # Figshare direct download URL
    # Note: This is the DOI link - Figshare will redirect to actual download
    figshare_url = "https://figshare.com/ndownloader/files/43904155"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    zip_file = output_dir / "brisc_dataset.zip"
    
    print("\n📊 BRISC Dataset Information:")
    print("  - Images: 6,000 T1-weighted MRI scans")
    print("  - Classes: Glioma, Meningioma, Pituitary, Non-tumorous")
    print("  - Masks: Expert-annotated segmentation masks")
    print("  - Planes: Axial, Coronal, Sagittal")
    print("  - Size: ~2-3 GB")
    print("\n")
    
    # Download
    try:
        print(f"Downloading from Figshare...")
        download_file(figshare_url, zip_file)
        
        # Extract
        extract_zip(zip_file, output_dir)
        
        # Clean up zip file (optional)
        print("\n🗑️  Cleaning up...")
        zip_file.unlink()
        print("✓ Removed zip file")
        
        # Verify structure
        print("\n📁 Dataset Structure:")
        for item in sorted(output_dir.rglob("*"))[:20]:
            if item.is_file():
                print(f"  {item.relative_to(output_dir)}")
        
        print(f"\n✅ DOWNLOAD COMPLETE!")
        print(f"✓ Dataset saved to: {output_dir.absolute()}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading from Figshare: {e}")
        print("\n📝 Alternative Download Options:")
        print("1. Kaggle:")
        print("   kaggle datasets download -d briscdataset/brisc2025")
        print("2. Zenodo:")
        print("   https://doi.org/10.5281/zenodo.17524350")
        print("3. Manual Download:")
        print("   Visit: https://figshare.com/articles/dataset/30533120")
        return False


def download_from_kaggle(output_dir='data/brisc'):
    """
    Download BRISC dataset from Kaggle (requires Kaggle API)
    
    Args:
        output_dir: Output directory
    """
    try:
        import kaggle
        
        print("=" * 80)
        print("DOWNLOADING BRISC DATASET FROM KAGGLE")
        print("=" * 80)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Download using Kaggle API
        kaggle.api.dataset_download_files(
            'briscdataset/brisc2025',
            path=output_dir,
            unzip=True
        )
        
        print(f"\n✅ DOWNLOAD COMPLETE!")
        print(f"✓ Dataset saved to: {output_dir.absolute()}")
        return True
        
    except ImportError:
        print("❌ Kaggle API not installed")
        print("\nInstall with: pip install kaggle")
        print("Setup: https://www.kaggle.com/docs/api")
        return False
    except Exception as e:
        print(f"❌ Error downloading from Kaggle: {e}")
        return False


def show_dataset_info():
    """Display information about the BRISC dataset"""
    
    print("""
╔════════════════════════════════════════════════════════════════════╗
║                  BRISC DATASET INFORMATION                         ║
╠════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  📊 Dataset: BRISC 2025                                            ║
║  🏥 Purpose: Brain Tumor Segmentation & Classification             ║
║                                                                    ║
║  📈 Statistics:                                                    ║
║     • Total Images: 6,000                                          ║
║     • Training: 5,000                                              ║
║     • Testing: 1,000                                               ║
║                                                                    ║
║  🎯 Classes (with segmentation masks):                             ║
║     • Glioma: 1,401 images                                         ║
║     • Meningioma: 1,635 images                                     ║
║     • Pituitary: 1,757 images                                      ║
║     • Non-tumorous: 1,207 images                                   ║
║                                                                    ║
║  📐 Image Planes:                                                  ║
║     • Axial: ~1,937 images                                         ║
║     • Coronal: ~1,976 images                                       ║
║     • Sagittal: ~2,087 images                                      ║
║                                                                    ║
║  🏆 Key Features:                                                  ║
║     ✅ Expert-annotated segmentation masks                         ║
║     ✅ Certified radiologist verification                          ║
║     ✅ T1-weighted contrast-enhanced MRI                           ║
║     ✅ Balanced class distribution                                 ║
║     ✅ Multi-plane coverage                                        ║
║                                                                    ║
║  💾 Dataset Size: ~2-3 GB                                          ║
║                                                                    ║
╠════════════════════════════════════════════════════════════════════╣
║  📚 Citation:                                                      ║
║                                                                    ║
║  Fateh, A., Rezvani, Y., Moayedi, S., et al. (2025).              ║
║  BRISC: Annotated Dataset for Brain Tumor Segmentation and        ║
║  Classification. arXiv:2506.14318v4                               ║
║                                                                    ║
╠════════════════════════════════════════════════════════════════════╣
║  🔗 Download Links:                                                ║
║                                                                    ║
║  1. Kaggle:                                                        ║
║     https://www.kaggle.com/datasets/briscdataset/brisc2025/       ║
║                                                                    ║
║  2. Figshare:                                                      ║
║     https://doi.org/10.6084/m9.figshare.30533120                  ║
║                                                                    ║
║  3. Zenodo:                                                        ║
║     https://doi.org/10.5281/zenodo.17524350                       ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝

🎯 PERFECT FOR YOUR PROJECT:
────────────────────────────────────────────────────────────────────

✅ Matches your requirements:
   • 3 tumor classes (Glioma, Meningioma, Pituitary) ✅
   • Expert-annotated segmentation masks ✅
   • Suitable for Grad-CAM localization evaluation ✅
   • Can calculate IoU, Dice with ground truth ✅
   • Ready for reliability assessment ✅

✅ Dataset structure:
   data/brisc/
   ├── classification/
   │   ├── train/
   │   │   ├── glioma/
   │   │   ├── meningioma/
   │   │   ├── pituitary/
   │   │   └── non_tumorous/
   │   └── test/ (same structure)
   └── segmentation/
       ├── train/
       │   ├── images/
       │   └── masks/
       └── test/
           ├── images/
           └── masks/
    """)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Download BRISC Brain Tumor Dataset with Segmentation Masks"
    )
    parser.add_argument(
        '--source',
        choices=['figshare', 'kaggle', 'info'],
        default='figshare',
        help='Download source (default: figshare)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/brisc',
        help='Output directory (default: data/brisc)'
    )
    
    args = parser.parse_args()
    
    if args.source == 'info':
        show_dataset_info()
        return
    
    # Show info first
    show_dataset_info()
    
    print("\n" + "=" * 80)
    input("Press Enter to start download (Ctrl+C to cancel)...")
    print("=" * 80 + "\n")
    
    # Download based on source
    if args.source == 'figshare':
        success = download_brisc_figshare(args.output)
    elif args.source == 'kaggle':
        success = download_from_kaggle(args.output)
    
    if success:
        print("\n" + "=" * 80)
        print("NEXT STEPS")
        print("=" * 80)
        print("""
1. Verify dataset structure:
   ls -R data/brisc/

2. Update your config:
   # In src/config.py or similar
   DATASET_PATH = 'data/brisc'

3. Test data loading:
   python -c "from src.data import DatasetLoader; \\
              loader = DatasetLoader('data/brisc'); \\
              print(loader.get_dataset_statistics())"

4. Start training:
   python main.py --mode train --dataset brisc

5. Evaluate with masks:
   python main.py --mode eval --use-masks
        """)


if __name__ == "__main__":
    main()
