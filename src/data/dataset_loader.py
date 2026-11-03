"""
Dataset Loader for Brain Tumour MRI Images
Handles patient-wise data splitting and mask alignment
"""
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd


class BrainTumourDataset(Dataset):
    """Brain Tumour MRI Dataset with Masks"""
    
    def __init__(
        self, 
        image_paths: List[Path], 
        mask_paths: List[Path],
        labels: List[int],
        transform=None
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.labels = labels
        self.transform = transform
        
        assert len(image_paths) == len(mask_paths) == len(labels), \
            "Image, mask, and label counts must match"
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        
        # Load mask
        mask = Image.open(self.mask_paths[idx]).convert('L')
        
        # Apply transforms
        if self.transform:
            image, mask = self.transform(image, mask)
        
        return {
            'image': image,
            'mask': mask,
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
            'image_path': str(self.image_paths[idx]),
            'mask_path': str(self.mask_paths[idx])
        }


class DatasetLoader:
    """
    Manages dataset loading with patient-wise stratified splitting
    """
    
    def __init__(
        self, 
        dataset_dir: str,
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15,
        random_seed: int = 42
    ):
        self.dataset_dir = Path(dataset_dir)
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.random_seed = random_seed
        
        assert abs(train_split + val_split + test_split - 1.0) < 1e-6, \
            "Splits must sum to 1.0"
        
        self.image_mask_pairs = []
        self.patient_splits = {'train': [], 'val': [], 'test': []}
    
    def load_slices_and_masks(self) -> List[Tuple[Path, Path, int, str]]:
        """
        Load all image-mask pairs from dataset directory
        Returns: List of (image_path, mask_path, label, patient_id)
        """
        pairs = []
        
        # Expected structure: dataset_dir/class_name/images/ and dataset_dir/class_name/masks/
        for class_idx, class_name in enumerate(['glioma', 'meningioma', 'pituitary']):
            class_dir = self.dataset_dir / class_name
            if not class_dir.exists():
                continue
            
            images_dir = class_dir / 'images'
            masks_dir = class_dir / 'masks'
            
            if images_dir.exists() and masks_dir.exists():
                image_files = sorted(images_dir.glob('*.png')) + \
                             sorted(images_dir.glob('*.jpg')) + \
                             sorted(images_dir.glob('*.jpeg'))
                
                for img_path in image_files:
                    # Corresponding mask with same filename
                    mask_path = masks_dir / img_path.name
                    
                    if mask_path.exists():
                        # Extract patient ID from filename (assuming format: patientXXX_slice_YY.png)
                        patient_id = img_path.stem.split('_')[0]
                        pairs.append((img_path, mask_path, class_idx, patient_id))
        
        self.image_mask_pairs = pairs
        return pairs
    
    def get_patient_stratified_split(self) -> Dict[str, List[Tuple]]:
        """
        Perform patient-wise stratified split
        Ensures no patient appears in multiple splits
        """
        np.random.seed(self.random_seed)
        
        # Group by patient and class
        from collections import defaultdict
        patients_by_class = defaultdict(list)
        
        for img, mask, label, patient_id in self.image_mask_pairs:
            patients_by_class[label].append(patient_id)
        
        # Get unique patients per class
        for label in patients_by_class:
            patients_by_class[label] = list(set(patients_by_class[label]))
        
        # Split patients per class
        splits = {'train': [], 'val': [], 'test': []}
        
        for label, patients in patients_by_class.items():
            np.random.shuffle(patients)
            n = len(patients)
            
            train_end = int(n * self.train_split)
            val_end = train_end + int(n * self.val_split)
            
            splits['train'].extend(patients[:train_end])
            splits['val'].extend(patients[train_end:val_end])
            splits['test'].extend(patients[val_end:])
        
        self.patient_splits = splits
        
        # Now assign image-mask pairs to splits based on patient ID
        split_data = {'train': [], 'val': [], 'test': []}
        
        for img, mask, label, patient_id in self.image_mask_pairs:
            for split_name, patient_list in splits.items():
                if patient_id in patient_list:
                    split_data[split_name].append((img, mask, label, patient_id))
                    break
        
        return split_data
    
    def verify_mask_alignment(self, image_path: Path, mask_path: Path) -> bool:
        """Verify that image and mask have same dimensions"""
        try:
            img = Image.open(image_path)
            mask = Image.open(mask_path)
            return img.size == mask.size
        except Exception as e:
            print(f"Error verifying alignment: {e}")
            return False
    
    def get_data_loader(
        self, 
        split: str,
        batch_size: int = 32,
        transform=None,
        shuffle: bool = True,
        num_workers: int = 4
    ) -> DataLoader:
        """
        Get DataLoader for specified split
        """
        split_data = self.get_patient_stratified_split()
        
        if split not in split_data:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
        
        data = split_data[split]
        
        if len(data) == 0:
            raise ValueError(f"No data found for split: {split}")
        
        image_paths = [d[0] for d in data]
        mask_paths = [d[1] for d in data]
        labels = [d[2] for d in data]
        
        dataset = BrainTumourDataset(
            image_paths=image_paths,
            mask_paths=mask_paths,
            labels=labels,
            transform=transform
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return dataloader
    
    def get_dataset_statistics(self) -> Dict:
        """Get statistics about the dataset"""
        split_data = self.get_patient_stratified_split()
        
        stats = {
            'total_samples': len(self.image_mask_pairs),
            'train_samples': len(split_data['train']),
            'val_samples': len(split_data['val']),
            'test_samples': len(split_data['test']),
            'class_distribution': {}
        }
        
        # Count samples per class
        from collections import Counter
        for split_name, data in split_data.items():
            labels = [d[2] for d in data]
            class_counts = Counter(labels)
            stats['class_distribution'][split_name] = dict(class_counts)
        
        return stats
