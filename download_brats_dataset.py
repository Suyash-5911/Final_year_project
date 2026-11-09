"""
Download BraTS (Brain Tumor Segmentation) Dataset
==================================================

BraTS is the gold standard for brain tumor segmentation with:
- Expert-annotated 3D tumor masks
- Multiple MRI modalities (T1, T1ce, T2, FLAIR)
- Multiple tumor regions (whole tumor, tumor core, enhancing tumor)
- Used in international challenges

Dataset Info:
- Size: ~50-100 GB (depending on version)
- Format: NIfTI (.nii.gz) - 3D medical imaging
- Classes: Glioma (multiple grades)
- Masks: Expert radiologist annotations

Note: BraTS requires registration and agreement to terms of use
"""

import subprocess
import os
from pathlib import Path
import requests
import zipfile
import tarfile
import gdown  # For Google Drive downloads


def show_brats_info():
    """Display information about BraTS dataset versions"""
    
    print("=" * 80)
    print("BraTS DATASET OPTIONS")
    print("=" * 80)
    
    print("""
╔════════════════════════════════════════════════════════════════════════╗
║                     BraTS Dataset Versions                             ║
╠════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  BraTS 2021 (Recommended)                                              ║
║  ────────────────────────────────────────────────────────────────      ║
║  Size: ~90 GB                                                          ║
║  Training: 1,251 cases                                                 ║
║  Validation: 219 cases                                                 ║
║  Format: NIfTI (.nii.gz)                                               ║
║  Masks: 3 tumor regions (whole, core, enhancing)                      ║
║  Link: https://www.synapse.org/#!Synapse:syn27046444                   ║
║                                                                        ║
║  BraTS 2020                                                            ║
║  ────────────────────────────────────────────────────────────────      ║
║  Size: ~85 GB                                                          ║
║  Training: 369 cases                                                   ║
║  Validation: 125 cases                                                 ║
║  Format: NIfTI (.nii.gz)                                               ║
║  Link: https://www.med.upenn.edu/cbica/brats2020/data.html             ║
║                                                                        ║
║  BraTS 2018 (Smaller, easier to start)                                 ║
║  ────────────────────────────────────────────────────────────────      ║
║  Size: ~45 GB                                                          ║
║  Training: 285 cases                                                   ║
║  Validation: 66 cases                                                  ║
║  Format: NIfTI (.nii.gz)                                               ║
║  Link: https://www.med.upenn.edu/sbia/brats2018/data.html              ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝

WHAT'S INCLUDED:
────────────────────────────────────────────────────────────────────────

Each case contains:
  1. T1-weighted MRI (pre-contrast)
  2. T1-weighted MRI with contrast enhancement (T1ce)
  3. T2-weighted MRI
  4. FLAIR (Fluid Attenuated Inversion Recovery)
  5. Segmentation mask with labels:
     - Label 0: Background
     - Label 1: Necrotic tumor core (NCR)
     - Label 2: Peritumoral edema (ED)
     - Label 4: Enhancing tumor (ET)

TUMOR REGIONS:
  • Whole Tumor (WT): All labels (1 + 2 + 4)
  • Tumor Core (TC): Labels 1 + 4
  • Enhancing Tumor (ET): Label 4 only
    """)


def download_brats_2021_instructions():
    """Provide step-by-step instructions for BraTS 2021 download"""
    
    print("\n" + "=" * 80)
    print("HOW TO DOWNLOAD BraTS 2021 (RECOMMENDED)")
    print("=" * 80)
    
    print("""
STEP 1: Register on Synapse
────────────────────────────────────────────────────────────────────────

1. Go to: https://www.synapse.org/
2. Click "Register" (top right)
3. Create account with email verification
4. Complete profile information

STEP 2: Accept Terms of Use
────────────────────────────────────────────────────────────────────────

1. Visit: https://www.synapse.org/#!Synapse:syn27046444
2. Click "Files" tab
3. You'll be asked to accept terms of use
4. Read and accept the data use agreement

STEP 3: Get Your API Key
────────────────────────────────────────────────────────────────────────

1. Go to: https://www.synapse.org/#!Profile:v/settings
2. Scroll to "Personal Access Tokens"
3. Click "Manage Personal Access Tokens"
4. Click "Create New Token"
5. Give it a name (e.g., "BraTS Download")
6. Select "Download" permission
7. Copy the token (you'll need this)

STEP 4: Install Synapse Python Client
────────────────────────────────────────────────────────────────────────

pip install synapseclient

STEP 5: Download Using Python
────────────────────────────────────────────────────────────────────────

Run this script with your API token:

    python download_brats_dataset.py --version 2021 --token YOUR_TOKEN_HERE

Or use interactive mode:

    python download_brats_dataset.py --interactive

ALTERNATIVE: Manual Download
────────────────────────────────────────────────────────────────────────

If automated download fails:
1. Visit: https://www.synapse.org/#!Synapse:syn27046444
2. Navigate to "Files" tab
3. Download files manually:
   - BraTS2021_Training_Data.tar (training set)
   - BraTS2021_Validation_Data.tar (validation set)
4. Extract to data/brats_2021/
    """)


def download_brats_automated(version='2021', output_dir='data/brats', api_token=None):
    """
    Automated download of BraTS dataset using Synapse client
    
    Args:
        version: BraTS version ('2021', '2020', '2018')
        output_dir: Output directory
        api_token: Synapse API token
    """
    
    print("\n" + "=" * 80)
    print(f"DOWNLOADING BraTS {version} DATASET")
    print("=" * 80)
    
    # Check if synapseclient is installed
    try:
        import synapseclient
        from synapseclient import Synapse
    except ImportError:
        print("✗ synapseclient not installed")
        print("\nInstall with: pip install synapseclient")
        return False
    
    if not api_token:
        print("✗ API token required")
        print("\nGet your token from: https://www.synapse.org/#!Profile:v/settings")
        print("Then run: python download_brats_dataset.py --version 2021 --token YOUR_TOKEN")
        return False
    
    # Synapse IDs for different BraTS versions
    synapse_ids = {
        '2021': 'syn27046444',  # BraTS 2021
        '2020': 'syn25829067',  # BraTS 2020
        '2018': 'syn15089892',  # BraTS 2018 (approximate)
    }
    
    if version not in synapse_ids:
        print(f"✗ Unknown version: {version}")
        print(f"Available versions: {', '.join(synapse_ids.keys())}")
        return False
    
    # Create output directory
    output_path = Path(output_dir) / f"brats_{version}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n✓ Output directory: {output_path.absolute()}")
    print(f"✓ Synapse ID: {synapse_ids[version]}")
    print("\nConnecting to Synapse...")
    
    try:
        # Login to Synapse
        syn = Synapse()
        syn.login(authToken=api_token)
        print("✓ Logged in to Synapse")
        
        # Get entity
        print(f"\nFetching BraTS {version} data...")
        entity = syn.get(synapse_ids[version], downloadLocation=str(output_path))
        
        print("\n✓ Download complete!")
        print(f"✓ Files saved to: {output_path.absolute()}")
        
        # Extract if tar/zip
        extract_archives(output_path)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error downloading: {e}")
        print("\nTry manual download from:")
        print(f"https://www.synapse.org/#!Synapse:{synapse_ids[version]}")
        return False


def extract_archives(directory: Path):
    """Extract tar/zip archives in directory"""
    
    print("\n" + "=" * 80)
    print("EXTRACTING ARCHIVES")
    print("=" * 80)
    
    # Find archives
    tar_files = list(directory.glob("*.tar")) + list(directory.glob("*.tar.gz"))
    zip_files = list(directory.glob("*.zip"))
    
    if not tar_files and not zip_files:
        print("✓ No archives to extract")
        return
    
    # Extract tar files
    for tar_file in tar_files:
        print(f"\nExtracting {tar_file.name}...")
        try:
            with tarfile.open(tar_file, 'r:*') as tar:
                tar.extractall(directory)
            print(f"✓ Extracted {tar_file.name}")
            
            # Optionally remove tar file to save space
            # tar_file.unlink()
            
        except Exception as e:
            print(f"✗ Error extracting {tar_file.name}: {e}")
    
    # Extract zip files
    for zip_file in zip_files:
        print(f"\nExtracting {zip_file.name}...")
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(directory)
            print(f"✓ Extracted {zip_file.name}")
            
            # Optionally remove zip file
            # zip_file.unlink()
            
        except Exception as e:
            print(f"✗ Error extracting {zip_file.name}: {e}")
    
    print("\n✓ Extraction complete")


def verify_brats_dataset(dataset_dir: Path):
    """Verify BraTS dataset structure"""
    
    print("\n" + "=" * 80)
    print("VERIFYING DATASET")
    print("=" * 80)
    
    if not dataset_dir.exists():
        print(f"✗ Directory not found: {dataset_dir}")
        return False
    
    # Look for NIfTI files
    nii_files = list(dataset_dir.rglob("*.nii")) + list(dataset_dir.rglob("*.nii.gz"))
    
    # Count different types
    t1_files = [f for f in nii_files if '_t1.' in f.name.lower()]
    t1ce_files = [f for f in nii_files if '_t1ce.' in f.name.lower()]
    t2_files = [f for f in nii_files if '_t2.' in f.name.lower()]
    flair_files = [f for f in nii_files if '_flair.' in f.name.lower()]
    seg_files = [f for f in nii_files if '_seg.' in f.name.lower()]
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total NIfTI files: {len(nii_files)}")
    print(f"   T1 images: {len(t1_files)}")
    print(f"   T1ce images: {len(t1ce_files)}")
    print(f"   T2 images: {len(t2_files)}")
    print(f"   FLAIR images: {len(flair_files)}")
    print(f"   Segmentation masks: {len(seg_files)}")
    
    if seg_files:
        print(f"\n✅ Dataset verified successfully!")
        print(f"✓ Found {len(seg_files)} cases with segmentation masks")
        return True
    else:
        print("\n⚠ Warning: No segmentation masks found")
        print("Dataset may still be downloading or extracting")
        return False


def show_next_steps():
    """Show what to do after download"""
    
    print("\n" + "=" * 80)
    print("NEXT STEPS AFTER DOWNLOAD")
    print("=" * 80)
    
    print("""
1. CONVERT NIfTI TO PNG/JPG (For Your Project)
   ────────────────────────────────────────────────────────────────
   
   BraTS uses 3D NIfTI format, but your project likely uses 2D images.
   Run the conversion script:
   
   python convert_brats_to_2d.py
   
   This will:
   - Extract 2D slices from 3D volumes
   - Convert to PNG/JPG format
   - Create matching 2D segmentation masks
   - Organize for your classification pipeline

2. INTEGRATE WITH YOUR PROJECT
   ────────────────────────────────────────────────────────────────
   
   Update your data paths:
   - Training images: data/brats_2d/images/
   - Training masks: data/brats_2d/masks/
   
3. TRAIN YOUR MODEL
   ────────────────────────────────────────────────────────────────
   
   Now you can:
   - Train CNN for classification with REAL ground truth
   - Calculate IoU/Dice scores with expert annotations
   - Validate Grad-CAM against true tumor locations
   - Perform complete localization evaluation

4. IMPORTANT: DATA FORMAT
   ────────────────────────────────────────────────────────────────
   
   BraTS masks have multiple labels:
   - 0: Background
   - 1: Necrotic tumor core
   - 2: Peritumoral edema  
   - 4: Enhancing tumor
   
   For binary classification (tumor vs non-tumor):
   - Combine all labels: (mask > 0) = tumor
   
   For multi-region:
   - Keep original labels for detailed analysis
    """)


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download BraTS Brain Tumor Segmentation Dataset"
    )
    parser.add_argument(
        '--version',
        type=str,
        choices=['2021', '2020', '2018'],
        default='2021',
        help='BraTS version to download'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/brats',
        help='Output directory'
    )
    parser.add_argument(
        '--token',
        type=str,
        help='Synapse API token'
    )
    parser.add_argument(
        '--info',
        action='store_true',
        help='Show dataset information only'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Interactive mode with prompts'
    )
    parser.add_argument(
        '--verify',
        type=str,
        help='Verify existing dataset directory'
    )
    
    args = parser.parse_args()
    
    # Show info
    if args.info:
        show_brats_info()
        download_brats_2021_instructions()
        return
    
    # Verify existing dataset
    if args.verify:
        verify_brats_dataset(Path(args.verify))
        show_next_steps()
        return
    
    # Interactive mode
    if args.interactive:
        show_brats_info()
        download_brats_2021_instructions()
        
        print("\n" + "=" * 80)
        input("Press Enter when you're ready to download (Ctrl+C to cancel)...")
        
        token = input("\nEnter your Synapse API token: ").strip()
        if not token:
            print("✗ Token required")
            return
        
        version = input(f"Enter BraTS version (2021/2020/2018) [default: 2021]: ").strip() or '2021'
        output_dir = input(f"Enter output directory [default: data/brats]: ").strip() or 'data/brats'
        
        success = download_brats_automated(version, output_dir, token)
        
        if success:
            show_next_steps()
    
    # Non-interactive mode
    elif args.token:
        success = download_brats_automated(args.version, args.output, args.token)
        if success:
            show_next_steps()
    
    else:
        # Show instructions
        print("\n⚠ No token provided. Showing instructions...\n")
        show_brats_info()
        download_brats_2021_instructions()
        print("\n" + "=" * 80)
        print("Run with --interactive for guided download")
        print("Or run with --token YOUR_TOKEN for automated download")
        print("=" * 80)


if __name__ == "__main__":
    main()
