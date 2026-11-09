"""Quick dataset verification"""
from src.data import DatasetLoader

print("=" * 60)
print("VERIFYING DATASET")
print("=" * 60)

loader = DatasetLoader('data/brain_tumours')
loader.load_slices_and_masks()
stats = loader.get_dataset_statistics()

print("\nDataset Statistics:")
print(f"  Total samples: {stats['total_samples']}")
print(f"  Train samples: {stats['train_samples']}")
print(f"  Val samples: {stats['val_samples']}")
print(f"  Test samples: {stats['test_samples']}")

print("\nClass Distribution (Train):")
for cls, count in stats['class_distribution']['train'].items():
    class_names = {0: 'Glioma', 1: 'Meningioma', 2: 'Pituitary'}
    print(f"  {class_names[cls]}: {count} samples")

print("\n" + "=" * 60)
print("✓ DATASET READY FOR TRAINING!")
print("=" * 60)
print("\nNext step: python main.py --mode train")
