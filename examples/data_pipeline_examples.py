"""
Quick Reference: Data Pipeline Usage Examples
"""

# =============================================================================
# EXAMPLE 1: Basic Training Setup
# =============================================================================

from src.data import DatasetLoader, Preprocessor, TrainTransform, ValTestTransform

# Create loader
loader = DatasetLoader(
    root_dir='data/brisc/brisc2025',
    batch_size=32,
    num_workers=4,
    include_no_tumor=False,  # 3-class problem: glioma, meningioma, pituitary
)

# Create transforms
preprocessor = Preprocessor(image_size=224)
train_transform = TrainTransform(preprocessor)
val_transform = ValTestTransform(preprocessor)

# Get DataLoaders
train_loader = loader.get_classification_loader('train', train_transform)
test_loader = loader.get_classification_loader('test', val_transform)

# Training loop
for epoch in range(50):
    for images, labels in train_loader:
        # images: (32, 3, 224, 224)
        # labels: (32,)
        # Train your model here
        pass


# =============================================================================
# EXAMPLE 2: Custom Augmentation Levels
# =============================================================================

from src.data import get_train_transform, get_val_test_transform

# Light augmentation (for small datasets)
light_transform = get_train_transform(image_size=224, augmentation_level='light')

# Medium augmentation (default, balanced)
medium_transform = get_train_transform(image_size=224, augmentation_level='medium')

# Heavy augmentation (for large datasets or to prevent overfitting)
heavy_transform = get_train_transform(image_size=224, augmentation_level='heavy')

# Validation/test (no augmentation)
val_transform = get_val_test_transform(image_size=224)


# =============================================================================
# EXAMPLE 3: Balanced Training with Sample Weights
# =============================================================================

from src.data import BRISCDataset, TrainTransform, Preprocessor
from torch.utils.data import DataLoader, WeightedRandomSampler

# Create dataset
preprocessor = Preprocessor(image_size=224)
train_transform = TrainTransform(preprocessor)

dataset = BRISCDataset(
    root_dir='data/brisc/brisc2025',
    task='classification',
    split='train',
    transform=train_transform,
    include_no_tumor=False,
)

# Get sample weights for balanced sampling
sample_weights = dataset.get_sample_weights()

# Create weighted sampler
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(dataset),
    replacement=True,
)

# Create DataLoader with sampler
train_loader = DataLoader(
    dataset,
    batch_size=32,
    sampler=sampler,  # Don't use shuffle with sampler
    num_workers=4,
    pin_memory=True,
)


# =============================================================================
# EXAMPLE 4: Segmentation Task (for Grad-CAM evaluation)
# =============================================================================

from src.data import DatasetLoader, Preprocessor, ValTestTransform

loader = DatasetLoader(
    root_dir='data/brisc/brisc2025',
    batch_size=16,
)

preprocessor = Preprocessor(image_size=224)
val_transform = ValTestTransform(preprocessor)

# Get segmentation DataLoader
seg_loader = loader.get_segmentation_loader(
    split='test',
    transform=val_transform,
    return_path=True,  # Get image paths for analysis
)

# Evaluate Grad-CAM localization
for images, masks, paths in seg_loader:
    # images: (16, 3, 224, 224)
    # masks: (16, 1, 224, 224) - binary masks
    # paths: list of image file paths
    
    # Generate Grad-CAM and compare with ground truth masks
    # Calculate IoU, Dice, Pointing Game, Energy Ratio
    pass


# =============================================================================
# EXAMPLE 5: Mixed Precision Training with MixUp
# =============================================================================

import torch
from src.data import DatasetLoader, TrainTransform, Preprocessor, MixUp

# Setup
loader = DatasetLoader('data/brisc/brisc2025', batch_size=32)
preprocessor = Preprocessor(image_size=224)
train_transform = TrainTransform(preprocessor)
train_loader = loader.get_classification_loader('train', train_transform)

# MixUp augmentation
mixup = MixUp(alpha=0.2)

# Training loop with MixUp
for images, labels in train_loader:
    # Apply MixUp
    mixed_images, labels_a, labels_b, lam = mixup(images, labels)
    
    # Forward pass
    outputs = model(mixed_images)
    
    # MixUp loss
    loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
    
    # Backward pass
    loss.backward()
    optimizer.step()


# =============================================================================
# EXAMPLE 6: Test-Time Augmentation (TTA)
# =============================================================================

from src.data import TestTimeAugmentation, Preprocessor
import torch.nn.functional as F

preprocessor = Preprocessor(image_size=224)
tta = TestTimeAugmentation(
    preprocessor=preprocessor,
    num_augmentations=5,
    hflip=True,
    rotation_angles=[0, -5, 5, -10, 10],
)

# Single image prediction with TTA
from PIL import Image

image = Image.open('path/to/image.jpg').convert('RGB')

# Generate augmented versions
augmented_tensors = tta(image)  # List of 5 tensors

# Stack and batch predict
batch = torch.stack(augmented_tensors)  # (5, 3, 224, 224)
predictions = model(batch)  # (5, num_classes)

# Average predictions
final_prediction = F.softmax(predictions, dim=1).mean(dim=0)
predicted_class = final_prediction.argmax()


# =============================================================================
# EXAMPLE 7: Visualization and Denormalization
# =============================================================================

import matplotlib.pyplot as plt
from src.data import Preprocessor

preprocessor = Preprocessor(image_size=224)

# Get a batch
images, labels = next(iter(train_loader))

# Visualize first 6 images
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

for idx in range(6):
    # Denormalize
    denorm_image = preprocessor.denormalize(images[idx])
    
    # Convert to numpy (H, W, C)
    img_np = denorm_image.permute(1, 2, 0).cpu().numpy()
    
    # Plot
    axes[idx].imshow(img_np)
    axes[idx].set_title(f"Label: {labels[idx].item()}")
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('visualizations.png')
plt.close()


# =============================================================================
# EXAMPLE 8: Dataset Statistics
# =============================================================================

from src.data import DatasetLoader, BRISCDataset

# Print comprehensive dataset info
loader = DatasetLoader('data/brisc/brisc2025')
loader.print_dataset_info()

# Get class distribution
train_dataset = BRISCDataset(
    root_dir='data/brisc/brisc2025',
    task='classification',
    split='train',
    include_no_tumor=False,
)

distribution = train_dataset.get_class_distribution()
print("\nClass Distribution:")
for class_name, count in distribution.items():
    percentage = (count / len(train_dataset)) * 100
    print(f"  {class_name:12s}: {count:4d} ({percentage:5.2f}%)")

# Calculate custom normalization statistics
from src.data import calculate_dataset_statistics
from torch.utils.data import DataLoader

# Create loader without normalization
temp_loader = DataLoader(train_dataset, batch_size=32, num_workers=4)
mean, std = calculate_dataset_statistics(temp_loader, num_batches=50)
print(f"\nDataset Statistics:")
print(f"  Mean: {mean}")
print(f"  Std:  {std}")


# =============================================================================
# EXAMPLE 9: Single Image Inference Pipeline
# =============================================================================

from PIL import Image
import torch
from src.data import Preprocessor

# Load model and preprocessor
model = load_your_trained_model()
model.eval()
preprocessor = Preprocessor(image_size=224)

# Load and preprocess image
image_path = 'path/to/brain_scan.jpg'
image = Image.open(image_path).convert('RGB')
image_tensor = preprocessor.preprocess_image(image)
image_batch = image_tensor.unsqueeze(0)  # Add batch dimension

# Inference
with torch.no_grad():
    outputs = model(image_batch)
    probabilities = torch.softmax(outputs, dim=1)
    predicted_class = probabilities.argmax(dim=1).item()
    confidence = probabilities[0, predicted_class].item()

print(f"Predicted Class: {predicted_class}")
print(f"Confidence: {confidence:.2%}")

# Class names
class_names = ['glioma', 'meningioma', 'pituitary']
print(f"Diagnosis: {class_names[predicted_class]}")


# =============================================================================
# EXAMPLE 10: Complete Training Script Template
# =============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.data import DatasetLoader, Preprocessor, TrainTransform, ValTestTransform

# Hyperparameters
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224
NUM_CLASSES = 3

# Data pipeline
loader = DatasetLoader(
    root_dir='data/brisc/brisc2025',
    batch_size=BATCH_SIZE,
    num_workers=4,
    include_no_tumor=False,
)

preprocessor = Preprocessor(image_size=IMAGE_SIZE)
train_transform = TrainTransform(preprocessor)
val_transform = ValTestTransform(preprocessor)

train_loader = loader.get_classification_loader('train', train_transform)
test_loader = loader.get_classification_loader('test', val_transform)

# Model, loss, optimizer
model = YourCNNModel(num_classes=NUM_CLASSES).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

# Training loop
best_acc = 0.0
for epoch in range(EPOCHS):
    # Train
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for images, labels in train_loader:
        images, labels = images.cuda(), labels.cuda()
        
        # Forward
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Metrics
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_correct += predicted.eq(labels).sum().item()
        train_total += labels.size(0)
    
    train_acc = 100.0 * train_correct / train_total
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_correct += predicted.eq(labels).sum().item()
            val_total += labels.size(0)
    
    val_acc = 100.0 * val_correct / val_total
    
    # Learning rate scheduling
    scheduler.step()
    
    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'models/best_model.pth')
    
    print(f"Epoch {epoch+1}/{EPOCHS}: "
          f"Train Loss={train_loss/len(train_loader):.4f}, Train Acc={train_acc:.2f}%, "
          f"Val Loss={val_loss/len(test_loader):.4f}, Val Acc={val_acc:.2f}%")

print(f"\nBest Validation Accuracy: {best_acc:.2f}%")
