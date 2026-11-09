"""
Dataset loader for BRISC Brain Tumor Dataset.

Supports both classification and segmentation tasks with train/test splits.
"""

from pathlib import Path
from typing import Optional, Tuple, Union, Literal
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class BRISCDataset(Dataset):
    """
    BRISC Brain Tumor Dataset loader.
    
    Supports:
    - Classification task: 3 tumor types (glioma, meningioma, pituitary) or 4 classes with no_tumor
    - Segmentation task: images with expert-annotated masks
    - Train/test splits
    
    Args:
        root_dir: Path to brisc2025/ directory
        task: Either 'classification' or 'segmentation'
        split: Either 'train' or 'test'
        transform: Optional transform to apply to images
        include_no_tumor: Whether to include no_tumor class (classification only)
        return_path: Whether to return image path along with data
    """
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        task: Literal['classification', 'segmentation'] = 'classification',
        split: Literal['train', 'test'] = 'train',
        transform: Optional[callable] = None,
        include_no_tumor: bool = False,
        return_path: bool = False,
    ):
        self.root = Path(root_dir)
        self.task = task
        self.split = split
        self.transform = transform
        self.include_no_tumor = include_no_tumor
        self.return_path = return_path
        
        # Define class mappings (3 tumor types by default)
        if include_no_tumor:
            self.classes = ['glioma', 'meningioma', 'pituitary', 'no_tumor']
            self.num_classes = 4
        else:
            self.classes = ['glioma', 'meningioma', 'pituitary']
            self.num_classes = 3
        
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # Load dataset samples
        if task == 'classification':
            self._load_classification_data()
        elif task == 'segmentation':
            self._load_segmentation_data()
        else:
            raise ValueError(f"Invalid task: {task}. Must be 'classification' or 'segmentation'")
        
        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in {self.data_dir}")
    
    def _load_classification_data(self):
        """Load classification task images and labels."""
        self.data_dir = self.root / 'classification_task' / self.split
        self.samples = []
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Classification directory not found: {self.data_dir}")
        
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            
            if not class_dir.exists():
                print(f"Warning: Class directory not found: {class_dir}")
                continue
            
            class_idx = self.class_to_idx[class_name]
            
            # Load all .jpg images from class directory
            for img_path in sorted(class_dir.glob('*.jpg')):
                self.samples.append({
                    'image_path': img_path,
                    'label': class_idx,
                    'class_name': class_name,
                })
        
        print(f"Loaded {len(self.samples)} classification samples from {self.split} split")
    
    def _load_segmentation_data(self):
        """Load segmentation task images and masks."""
        self.data_dir = self.root / 'segmentation_task' / self.split
        self.samples = []
        
        img_dir = self.data_dir / 'images'
        mask_dir = self.data_dir / 'masks'
        
        if not img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
        if not mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {mask_dir}")
        
        # Load image-mask pairs
        for img_path in sorted(img_dir.glob('*.jpg')):
            # Corresponding mask has same basename but .png extension
            mask_name = img_path.stem + '.png'
            mask_path = mask_dir / mask_name
            
            if mask_path.exists():
                self.samples.append({
                    'image_path': img_path,
                    'mask_path': mask_path,
                    'filename': img_path.stem,
                })
            else:
                print(f"Warning: Mask not found for {img_path.name}")
        
        print(f"Loaded {len(self.samples)} segmentation samples from {self.split} split")
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Union[Tuple, dict]:
        """
        Get a sample from the dataset.
        
        Returns:
            Classification task: (image, label) or (image, label, path)
            Segmentation task: (image, mask) or (image, mask, path)
        """
        sample = self.samples[idx]
        
        if self.task == 'classification':
            # Load image
            image = Image.open(sample['image_path']).convert('RGB')
            label = sample['label']
            
            # Apply transforms
            if self.transform is not None:
                image = self.transform(image)
            
            if self.return_path:
                return image, label, str(sample['image_path'])
            else:
                return image, label
        
        else:  # segmentation
            # Load image and mask
            image = Image.open(sample['image_path']).convert('RGB')
            mask = Image.open(sample['mask_path']).convert('L')  # Grayscale
            
            # Apply transforms (should handle both image and mask)
            if self.transform is not None:
                image, mask = self.transform(image, mask)
            
            if self.return_path:
                return image, mask, str(sample['image_path'])
            else:
                return image, mask
    
    def get_class_distribution(self) -> dict:
        """Get distribution of samples per class (classification only)."""
        if self.task != 'classification':
            raise ValueError("Class distribution only available for classification task")
        
        distribution = {cls_name: 0 for cls_name in self.classes}
        
        for sample in self.samples:
            class_name = sample['class_name']
            distribution[class_name] += 1
        
        return distribution
    
    def get_sample_weights(self) -> torch.Tensor:
        """
        Calculate sample weights for balanced training (classification only).
        
        Returns:
            Tensor of weights for each sample (inverse class frequency)
        """
        if self.task != 'classification':
            raise ValueError("Sample weights only available for classification task")
        
        # Count samples per class
        class_counts = torch.zeros(self.num_classes)
        for sample in self.samples:
            class_counts[sample['label']] += 1
        
        # Calculate weights (inverse frequency)
        class_weights = 1.0 / class_counts
        
        # Assign weight to each sample
        sample_weights = torch.zeros(len(self.samples))
        for idx, sample in enumerate(self.samples):
            sample_weights[idx] = class_weights[sample['label']]
        
        return sample_weights


class DatasetLoader:
    """
    High-level dataset loader with DataLoader creation.
    
    Provides convenient methods to create PyTorch DataLoaders for training
    and evaluation with appropriate configurations.
    
    Args:
        root_dir: Path to brisc2025/ directory
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        include_no_tumor: Whether to include no_tumor class
    """
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        include_no_tumor: bool = False,
    ):
        self.root_dir = Path(root_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.include_no_tumor = include_no_tumor
        
        # Verify dataset exists
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")
    
    def get_classification_loader(
        self,
        split: Literal['train', 'test'],
        transform: Optional[callable] = None,
        shuffle: Optional[bool] = None,
        return_path: bool = False,
    ) -> DataLoader:
        """
        Create DataLoader for classification task.
        
        Args:
            split: 'train' or 'test'
            transform: Transform to apply to images
            shuffle: Whether to shuffle data (defaults to True for train, False for test)
            return_path: Whether to return image paths
        
        Returns:
            PyTorch DataLoader
        """
        if shuffle is None:
            shuffle = (split == 'train')
        
        dataset = BRISCDataset(
            root_dir=self.root_dir,
            task='classification',
            split=split,
            transform=transform,
            include_no_tumor=self.include_no_tumor,
            return_path=return_path,
        )
        
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=(split == 'train'),  # Drop last incomplete batch in training
        )
        
        return loader
    
    def get_segmentation_loader(
        self,
        split: Literal['train', 'test'],
        transform: Optional[callable] = None,
        shuffle: Optional[bool] = None,
        return_path: bool = False,
    ) -> DataLoader:
        """
        Create DataLoader for segmentation task.
        
        Args:
            split: 'train' or 'test'
            transform: Transform to apply to images and masks
            shuffle: Whether to shuffle data (defaults to True for train, False for test)
            return_path: Whether to return image paths
        
        Returns:
            PyTorch DataLoader
        """
        if shuffle is None:
            shuffle = (split == 'train')
        
        dataset = BRISCDataset(
            root_dir=self.root_dir,
            task='segmentation',
            split=split,
            transform=transform,
            return_path=return_path,
        )
        
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,  # Keep all samples for segmentation
        )
        
        return loader
    
    def get_data_loader(
        self,
        task: Literal['classification', 'segmentation'],
        split: Literal['train', 'test'],
        transform: Optional[callable] = None,
        shuffle: Optional[bool] = None,
        return_path: bool = False,
    ) -> DataLoader:
        """
        Generic method to create DataLoader for any task.
        
        Args:
            task: 'classification' or 'segmentation'
            split: 'train' or 'test'
            transform: Transform to apply
            shuffle: Whether to shuffle data
            return_path: Whether to return image paths
        
        Returns:
            PyTorch DataLoader
        """
        if task == 'classification':
            return self.get_classification_loader(split, transform, shuffle, return_path)
        elif task == 'segmentation':
            return self.get_segmentation_loader(split, transform, shuffle, return_path)
        else:
            raise ValueError(f"Invalid task: {task}")
    
    def print_dataset_info(self):
        """Print statistics about the dataset."""
        print("=" * 70)
        print("BRISC DATASET INFORMATION")
        print("=" * 70)
        
        # Classification statistics
        print("\n📊 CLASSIFICATION TASK:")
        for split in ['train', 'test']:
            try:
                dataset = BRISCDataset(
                    self.root_dir,
                    task='classification',
                    split=split,
                    include_no_tumor=self.include_no_tumor,
                )
                distribution = dataset.get_class_distribution()
                
                print(f"\n  {split.upper()} Split: {len(dataset)} images")
                for class_name, count in distribution.items():
                    percentage = (count / len(dataset)) * 100
                    print(f"    - {class_name:12s}: {count:4d} ({percentage:5.2f}%)")
            except Exception as e:
                print(f"  Error loading {split} split: {e}")
        
        # Segmentation statistics
        print("\n🎯 SEGMENTATION TASK:")
        for split in ['train', 'test']:
            try:
                dataset = BRISCDataset(
                    self.root_dir,
                    task='segmentation',
                    split=split,
                )
                print(f"  {split.upper()} Split: {len(dataset)} image-mask pairs")
            except Exception as e:
                print(f"  Error loading {split} split: {e}")
        
        print("\n" + "=" * 70)
