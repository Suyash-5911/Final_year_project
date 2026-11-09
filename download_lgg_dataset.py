"""
Download LGG Segmentation Dataset
==================================

This dataset includes REAL pixel-level tumor segmentation masks
for Low-Grade Glioma tumors.

Dataset: LGG Segmentation Dataset (Kaggle)
Size: ~500 MB
Classes: Glioma only
Masks: Expert-annotated tumor boundaries (REAL ground truth)

Link: https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation
"""
import subprocess
import zipfile
from pathlib import Path
import shutil
import os


def check_kaggle_setup():
    """Check if Kaggle CLI is installed and configured"""
    print("=" * 70)
    print("CHECKING KAGGLE CLI SETUP")
    print("=" * 70)
    
    try:
        result = subprocess.run(['kaggle', '--version'], 
                              capture_output=True, text=True)
        print(f"✓ Kaggle CLI installed: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        print("✗ Kaggle CLI not found")
        print("\nInstall with: pip install kaggle")
        return False


def setup_kaggle_api():
    """Guide user through Kaggle API setup"""
    print("\n" + "=" * 70)
    print("KAGGLE API SETUP INSTRUCTIONS")
    print("=" * 70)
    print("""
1. Go to: https://www.kaggle.com/settings
2. Scroll to 'API' section
3. Click 'Create New API Token'
4. This downloads 'kaggle.json'
5. Place it in:
   Windows: C:\\Users\\<username>\\.kaggle\\kaggle.json
   Linux/Mac: ~/.kaggle/kaggle.json

6. Run this script again after setup
    """)


def download_lgg_dataset(output_dir: str = "data/lgg_segmentation"):
    """Download LGG Segmentation Dataset from Kaggle"""
    
    print("\n" + "=" * 70)
    print("DOWNLOADING LGG SEGMENTATION DATASET")
    print("=" * 70)
    print("\nDataset: LGG MRI Segmentation")
    print("Size: ~500 MB")
    print("Contains: REAL tumor segmentation masks")
    print()
    
    if not check_kaggle_setup():
        setup_kaggle_api()
        return False
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download dataset
        print("\nDownloading dataset...")
        cmd = [
            'kaggle', 'datasets', 'download',
            '-d', 'mateuszbuda/lgg-mri-segmentation',
            '-p', str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✓ Download complete")
        
        # Unzip
        print("\nExtracting files...")
        zip_file = output_path / "lgg-mri-segmentation.zip"
        
        if zip_file.exists():
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(output_path)
            print("✓ Extraction complete")
            
            # Remove zip file
            zip_file.unlink()
            print("✓ Cleaned up zip file")
        
        # Verify dataset structure
        verify_lgg_dataset(output_path)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Error downloading dataset: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def verify_lgg_dataset(dataset_dir: Path):
    """Verify downloaded dataset structure"""
    print("\n" + "=" * 70)
    print("VERIFYING DATASET STRUCTURE")
    print("=" * 70)
    
    # Count files
    image_files = list(dataset_dir.rglob("*_*.tif"))  # LGG uses .tif format
    mask_files = list(dataset_dir.rglob("*_mask.tif"))
    
    image_files = [f for f in image_files if "_mask" not in f.name]
    
    print(f"\n✓ Found {len(image_files)} MRI images")
    print(f"✓ Found {len(mask_files)} tumor masks")
    
    if len(image_files) > 0 and len(mask_files) > 0:
        print("\n✅ Dataset structure verified!")
        print(f"\nDataset location: {dataset_dir.absolute()}")
        
        # Show sample
        print("\nSample files:")
        for i, img_file in enumerate(image_files[:3]):
            mask_file = str(img_file).replace(".tif", "_mask.tif")
            has_mask = Path(mask_file).exists()
            print(f"  {i+1}. {img_file.name} → Mask: {'✓' if has_mask else '✗'}")
        
        return True
    else:
        print("\n⚠ Warning: Dataset structure unexpected")
        return False


def compare_datasets():
    """Compare Kaggle Classification dataset vs LGG Segmentation dataset"""
    print("\n" + "=" * 70)
    print("DATASET COMPARISON")
    print("=" * 70)
    
    comparison = """
╔════════════════════════════════════════════════════════════════════════╗
║                    DATASET COMPARISON                                  ║
╠════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  CURRENT: Kaggle Brain Tumor Classification MRI                        ║
║  ────────────────────────────────────────────────────────────────      ║
║  ✓ Size: ~3,000 images                                                 ║
║  ✓ Classes: Glioma, Meningioma, Pituitary, No Tumor                   ║
║  ✓ Task: Classification                                                ║
║  ✗ Masks: None (classification labels only)                            ║
║                                                                        ║
║  USE FOR:                                                              ║
║    • Tumor type classification                                         ║
║    • Grad-CAM visualization (qualitative)                              ║
║    • Architecture comparison                                           ║
║                                                                        ║
╠════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  NEW: LGG MRI Segmentation Dataset                                     ║
║  ────────────────────────────────────────────────────────────────      ║
║  ✓ Size: ~3,900 images                                                 ║
║  ✓ Classes: Low-Grade Glioma only                                      ║
║  ✓ Task: Segmentation                                                  ║
║  ✓ Masks: Expert-annotated tumor boundaries                            ║
║                                                                        ║
║  USE FOR:                                                              ║
║    • Tumor segmentation (glioma)                                       ║
║    • Grad-CAM validation with ground truth                             ║
║    • IoU/Dice score calculation                                        ║
║    • Pointing game, energy ratio metrics                               ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝

RECOMMENDED APPROACH:
───────────────────────────────────────────────────────────────────────

1. Classification Task (All Tumor Types)
   Dataset: Kaggle Classification (current)
   Metrics: Accuracy, Precision, Recall, F1-score
   
2. Localization Task (Glioma Only)
   Dataset: LGG Segmentation (download with this script)
   Metrics: IoU, Dice, Pointing Game, Energy Ratio
   
3. Visualization
   Both datasets: Grad-CAM heatmaps, saliency maps
   
This hybrid approach gives you:
  ✓ Multi-class classification
  ✓ Quantitative localization evaluation
  ✓ Honest about dataset capabilities
  ✓ Complete project scope
    """
    
    print(comparison)


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download LGG Segmentation Dataset with real tumor masks"
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/lgg_segmentation',
        help='Output directory for dataset'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Show comparison between datasets'
    )
    
    args = parser.parse_args()
    
    if args.compare:
        compare_datasets()
    else:
        success = download_lgg_dataset(args.output)
        
        if success:
            print("\n" + "=" * 70)
            print("✅ SUCCESS!")
            print("=" * 70)
            print(f"""
Dataset downloaded to: {Path(args.output).absolute()}

This dataset includes REAL tumor segmentation masks!

Next steps:
1. Integrate LGG dataset into your pipeline
2. Use for glioma localization evaluation
3. Calculate IoU/Dice scores with ground truth
4. Keep classification dataset for multi-class task

Run with --compare flag to see dataset comparison.
            """)
        else:
            print("\n" + "=" * 70)
            print("⚠ DOWNLOAD FAILED")
            print("=" * 70)
            print("Please check Kaggle CLI setup and try again.")


if __name__ == "__main__":
    main()
