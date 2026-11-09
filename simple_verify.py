"""Simple dataset verification without dependencies"""
from pathlib import Path

print("=" * 60)
print("DATASET VERIFICATION")
print("=" * 60)

data_dir = Path('data/brain_tumours')
classes = ['glioma', 'meningioma', 'pituitary']

print("\n✓ Checking dataset structure...")

total_images = 0
total_masks = 0

for cls in classes:
    img_dir = data_dir / cls / 'images'
    mask_dir = data_dir / cls / 'masks'
    
    n_images = len(list(img_dir.glob('*.jpg')))
    n_masks = len(list(mask_dir.glob('*.jpg')))
    
    total_images += n_images
    total_masks += n_masks
    
    status = "✓" if n_images == n_masks else "⚠"
    print(f"\n{status} {cls.upper()}:")
    print(f"  Images: {n_images}")
    print(f"  Masks:  {n_masks}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Total images: {total_images}")
print(f"Total masks:  {total_masks}")

if total_images == total_masks and total_images > 0:
    print("\n✓✓✓ DATASET IS READY! ✓✓✓")
    print("\nYou can now:")
    print("1. Install PyTorch: pip install torch torchvision")
    print("2. Install other dependencies: pip install -r requirements.txt")
    print("3. Start training: python main.py --mode train")
else:
    print("\n⚠ Issue detected. Check your dataset structure.")
