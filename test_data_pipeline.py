"""
Test script for BRISC dataset data pipeline.

Verifies:
- Dataset loading for classification and segmentation tasks
- Data preprocessing and augmentation
- DataLoader creation
- Visualization of samples
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data import BRISCDataset, DatasetLoader, Preprocessor, TrainTransform, ValTestTransform


def test_dataset_info():
    """Test dataset information display."""
    print("\n" + "="*70)
    print("TEST 1: Dataset Information")
    print("="*70)
    
    loader = DatasetLoader(
        root_dir='data/brisc/brisc2025',
        batch_size=32,
        include_no_tumor=False,  # 3-class problem
    )
    
    loader.print_dataset_info()


def test_classification_loading():
    """Test classification task data loading."""
    print("\n" + "="*70)
    print("TEST 2: Classification Task Loading")
    print("="*70)
    
    # Create preprocessor and transforms
    preprocessor = Preprocessor(image_size=224)
    train_transform = TrainTransform(preprocessor)
    val_transform = ValTestTransform(preprocessor)
    
    # Load training data
    train_dataset = BRISCDataset(
        root_dir='data/brisc/brisc2025',
        task='classification',
        split='train',
        transform=train_transform,
        include_no_tumor=False,
    )
    
    # Load test data
    test_dataset = BRISCDataset(
        root_dir='data/brisc/brisc2025',
        task='classification',
        split='test',
        transform=val_transform,
        include_no_tumor=False,
    )
    
    print(f"\n✓ Train dataset: {len(train_dataset)} samples")
    print(f"✓ Test dataset: {len(test_dataset)} samples")
    print(f"✓ Number of classes: {train_dataset.num_classes}")
    print(f"✓ Classes: {train_dataset.classes}")
    
    # Get class distribution
    print("\nClass distribution (train):")
    distribution = train_dataset.get_class_distribution()
    for class_name, count in distribution.items():
        print(f"  {class_name:12s}: {count:4d}")
    
    # Test single sample
    image, label = train_dataset[0]
    print(f"\n✓ Sample shape: {image.shape}")
    print(f"✓ Sample label: {label} ({train_dataset.classes[label]})")
    print(f"✓ Sample dtype: {image.dtype}")
    print(f"✓ Sample range: [{image.min():.3f}, {image.max():.3f}]")
    
    return train_dataset, test_dataset


def test_segmentation_loading():
    """Test segmentation task data loading."""
    print("\n" + "="*70)
    print("TEST 3: Segmentation Task Loading")
    print("="*70)
    
    # Create preprocessor and transforms
    preprocessor = Preprocessor(image_size=224)
    train_transform = TrainTransform(preprocessor)
    val_transform = ValTestTransform(preprocessor)
    
    # Load training data
    train_dataset = BRISCDataset(
        root_dir='data/brisc/brisc2025',
        task='segmentation',
        split='train',
        transform=train_transform,
    )
    
    # Load test data
    test_dataset = BRISCDataset(
        root_dir='data/brisc/brisc2025',
        task='segmentation',
        split='test',
        transform=val_transform,
    )
    
    print(f"\n✓ Train dataset: {len(train_dataset)} samples")
    print(f"✓ Test dataset: {len(test_dataset)} samples")
    
    # Test single sample
    image, mask = train_dataset[0]
    print(f"\n✓ Image shape: {image.shape}")
    print(f"✓ Mask shape: {mask.shape}")
    print(f"✓ Image dtype: {image.dtype}")
    print(f"✓ Mask dtype: {mask.dtype}")
    print(f"✓ Mask values: unique={torch.unique(mask).tolist()}")
    
    return train_dataset, test_dataset


def test_dataloader():
    """Test DataLoader creation and batching."""
    print("\n" + "="*70)
    print("TEST 4: DataLoader Creation")
    print("="*70)
    
    # Create DatasetLoader
    loader = DatasetLoader(
        root_dir='data/brisc/brisc2025',
        batch_size=16,
        num_workers=2,
        include_no_tumor=False,
    )
    
    # Create preprocessor and transforms
    preprocessor = Preprocessor(image_size=224)
    train_transform = TrainTransform(preprocessor)
    
    # Get DataLoader
    train_loader = loader.get_classification_loader(
        split='train',
        transform=train_transform,
        shuffle=True,
    )
    
    print(f"\n✓ Created DataLoader with {len(train_loader)} batches")
    
    # Test one batch
    images, labels = next(iter(train_loader))
    print(f"✓ Batch images shape: {images.shape}")
    print(f"✓ Batch labels shape: {labels.shape}")
    print(f"✓ Batch labels: {labels.tolist()}")
    
    return train_loader


def visualize_classification_samples(dataset, num_samples=6):
    """Visualize classification samples."""
    print("\n" + "="*70)
    print("TEST 5: Visualize Classification Samples")
    print("="*70)
    
    # Create preprocessor for denormalization
    preprocessor = Preprocessor(image_size=224)
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    for idx in range(num_samples):
        image, label = dataset[idx]
        
        # Denormalize and convert to displayable format
        denorm_image = preprocessor.denormalize(image)
        denorm_image = denorm_image.permute(1, 2, 0).numpy()
        
        # Plot
        axes[idx].imshow(denorm_image)
        axes[idx].set_title(f"Class: {dataset.classes[label]}")
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/classification_samples.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved visualization to: results/classification_samples.png")
    plt.close()


def visualize_segmentation_samples(dataset, num_samples=6):
    """Visualize segmentation samples with masks."""
    print("\n" + "="*70)
    print("TEST 6: Visualize Segmentation Samples")
    print("="*70)
    
    # Create preprocessor for denormalization
    preprocessor = Preprocessor(image_size=224)
    
    fig, axes = plt.subplots(3, num_samples, figsize=(15, 9))
    
    for idx in range(num_samples):
        image, mask = dataset[idx]
        
        # Denormalize image
        denorm_image = preprocessor.denormalize(image)
        denorm_image = denorm_image.permute(1, 2, 0).numpy()
        
        # Convert mask to displayable format
        mask_np = mask.squeeze().numpy()
        
        # Create overlay
        overlay = denorm_image.copy()
        overlay[mask_np > 0.5] = [1.0, 0.0, 0.0]  # Red overlay
        blended = 0.6 * denorm_image + 0.4 * overlay
        
        # Plot image
        axes[0, idx].imshow(denorm_image)
        axes[0, idx].set_title(f"Image {idx+1}")
        axes[0, idx].axis('off')
        
        # Plot mask
        axes[1, idx].imshow(mask_np, cmap='gray')
        axes[1, idx].set_title(f"Mask {idx+1}")
        axes[1, idx].axis('off')
        
        # Plot overlay
        axes[2, idx].imshow(np.clip(blended, 0, 1))
        axes[2, idx].set_title(f"Overlay {idx+1}")
        axes[2, idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/segmentation_samples.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved visualization to: results/segmentation_samples.png")
    plt.close()


def visualize_augmentation(dataset, sample_idx=0, num_augmentations=6):
    """Visualize augmentation effects."""
    print("\n" + "="*70)
    print("TEST 7: Visualize Augmentation Effects")
    print("="*70)
    
    # Get original PIL image (before transforms)
    from PIL import Image
    sample = dataset.samples[sample_idx]
    original_image = Image.open(sample['image_path']).convert('RGB')
    
    # Create preprocessor and transform
    preprocessor = Preprocessor(image_size=224)
    train_transform = TrainTransform(preprocessor)
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    for idx in range(num_augmentations):
        # Apply augmentation
        augmented = train_transform(original_image)
        
        # Denormalize
        denorm = preprocessor.denormalize(augmented)
        denorm = denorm.permute(1, 2, 0).numpy()
        
        # Plot
        axes[idx].imshow(denorm)
        axes[idx].set_title(f"Augmentation {idx+1}")
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/augmentation_samples.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved visualization to: results/augmentation_samples.png")
    plt.close()


def test_sample_weights():
    """Test sample weighting for class balancing."""
    print("\n" + "="*70)
    print("TEST 8: Sample Weights for Class Balancing")
    print("="*70)
    
    dataset = BRISCDataset(
        root_dir='data/brisc/brisc2025',
        task='classification',
        split='train',
        include_no_tumor=False,
    )
    
    # Get sample weights
    weights = dataset.get_sample_weights()
    
    print(f"\n✓ Sample weights computed: {len(weights)} samples")
    print(f"✓ Weight range: [{weights.min():.6f}, {weights.max():.6f}]")
    print(f"✓ Mean weight: {weights.mean():.6f}")
    
    # Show weights per class
    print("\nAverage weight per class:")
    for class_idx, class_name in enumerate(dataset.classes):
        class_mask = torch.tensor([s['label'] == class_idx for s in dataset.samples])
        avg_weight = weights[class_mask].mean()
        print(f"  {class_name:12s}: {avg_weight:.6f}")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("BRISC DATASET PIPELINE TESTING")
    print("="*70)
    
    # Create results directory
    Path('results').mkdir(exist_ok=True)
    
    try:
        # Test 1: Dataset information
        test_dataset_info()
        
        # Test 2: Classification loading
        train_class, test_class = test_classification_loading()
        
        # Test 3: Segmentation loading
        train_seg, test_seg = test_segmentation_loading()
        
        # Test 4: DataLoader creation
        train_loader = test_dataloader()
        
        # Test 5: Visualize classification samples
        visualize_classification_samples(train_class)
        
        # Test 6: Visualize segmentation samples
        visualize_segmentation_samples(train_seg)
        
        # Test 7: Visualize augmentation
        visualize_augmentation(train_class)
        
        # Test 8: Sample weights
        test_sample_weights()
        
        # Final summary
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        print("\nData pipeline is ready for training!")
        print("\nGenerated visualizations:")
        print("  - results/classification_samples.png")
        print("  - results/segmentation_samples.png")
        print("  - results/augmentation_samples.png")
        print("\n" + "="*70)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
