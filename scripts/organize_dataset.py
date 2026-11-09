"""
Organize downloaded dataset into required structure
"""
import shutil
from pathlib import Path


def organize_kaggle_dataset(input_dir, output_dir="data/brain_tumours"):
    """
    Reorganize Kaggle dataset from:
      Training/glioma_tumor/*.jpg
    To:
      data/brain_tumours/glioma/images/*.jpg
    """
    print("=" * 60)
    print("ORGANIZING DATASET")
    print("=" * 60)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Mapping of original names to target names
    class_mapping = {
        'glioma_tumor': 'glioma',
        'meningioma_tumor': 'meningioma',
        'pituitary_tumor': 'pituitary',
        'notumor': 'notumor'  # optional
    }
    
    for orig_name, target_name in class_mapping.items():
        orig_dir = input_path / orig_name
        
        if not orig_dir.exists():
            print(f"⚠ Skipping {orig_name} (not found)")
            continue
        
        # Create target directories
        target_img_dir = output_path / target_name / 'images'
        target_mask_dir = output_path / target_name / 'masks'
        target_img_dir.mkdir(parents=True, exist_ok=True)
        target_mask_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy images
        image_files = list(orig_dir.glob('*.jpg')) + list(orig_dir.glob('*.png'))
        
        for img_file in image_files:
            target_file = target_img_dir / img_file.name
            if not target_file.exists():
                shutil.copy2(img_file, target_file)
        
        print(f"✓ {target_name}: {len(image_files)} images organized")
    
    print(f"\n✓ Dataset organized in: {output_path.absolute()}")
    print("\nNext steps:")
    print("1. Create masks: python scripts/create_masks_options.py --method threshold --class-name glioma")
    print("2. Train model: python main.py --mode train")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input directory (e.g., data/brain_tumours/Training)')
    parser.add_argument('--output', default='data/brain_tumours', help='Output directory')
    
    args = parser.parse_args()
    
    organize_kaggle_dataset(args.input, args.output)
