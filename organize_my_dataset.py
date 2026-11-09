"""
Quick script to organize your brain_tumor dataset
"""
import shutil
from pathlib import Path

print("=" * 70)
print("ORGANIZING YOUR DATASET")
print("=" * 70)

# Source and destination
source_train = Path("brain_tumor/Training")
source_test = Path("brain_tumor/Testing")
dest = Path("data/brain_tumours")

# Class mapping
class_map = {
    'glioma_tumor': 'glioma',
    'meningioma_tumor': 'meningioma',
    'pituitary_tumor': 'pituitary'
}

# Create organized structure
for orig_name, new_name in class_map.items():
    print(f"\nProcessing {new_name}...")
    
    # Create directories
    img_dir = dest / new_name / 'images'
    mask_dir = dest / new_name / 'masks'
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy training images
    train_src = source_train / orig_name
    if train_src.exists():
        for img_file in train_src.glob('*.jpg'):
            target = img_dir / img_file.name
            if not target.exists():
                shutil.copy2(img_file, target)
        
        train_count = len(list(img_dir.glob('*.jpg')))
        print(f"  ✓ Copied {train_count} training images")
    
    # Copy testing images too
    test_src = source_test / orig_name
    if test_src.exists():
        test_copied = 0
        for img_file in test_src.glob('*.jpg'):
            target = img_dir / f"test_{img_file.name}"
            if not target.exists():
                shutil.copy2(img_file, target)
                test_copied += 1
        
        print(f"  ✓ Copied {test_copied} testing images")
    
    total = len(list(img_dir.glob('*.jpg')))
    print(f"  ✓ Total {new_name} images: {total}")

print("\n" + "=" * 70)
print("✓ DATASET ORGANIZED SUCCESSFULLY!")
print("=" * 70)
print(f"\nLocation: {dest.absolute()}")
print("\nStructure:")
for class_name in ['glioma', 'meningioma', 'pituitary']:
    img_dir = dest / class_name / 'images'
    n_images = len(list(img_dir.glob('*.jpg')))
    print(f"  {class_name}/images/: {n_images} files")
    print(f"  {class_name}/masks/:  (empty - will create next)")

print("\n" + "=" * 70)
print("NEXT STEP: Create Masks")
print("=" * 70)
print("\nRun these commands to create masks:")
print("\npython scripts/create_masks_options.py --method threshold --class-name glioma")
print("python scripts/create_masks_options.py --method threshold --class-name meningioma")
print("python scripts/create_masks_options.py --method threshold --class-name pituitary")

print("\n✓ Done! Dataset is ready for training.")
