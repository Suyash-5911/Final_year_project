"""
Image Preprocessing and Augmentation
Handles normalization, resizing, and synchronized augmentation
"""
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from typing import Tuple, Optional
import random


class Preprocessor:
    """
    Handles preprocessing of MRI images and masks
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        intensity_range: Tuple[float, float] = (0, 1)
    ):
        self.target_size = target_size
        self.norm_mean = normalize_mean
        self.norm_std = normalize_std
        self.intensity_range = intensity_range
    
    def resize(self, img: Image.Image, mask: Optional[Image.Image] = None):
        """Resize image and mask to target size"""
        img_resized = TF.resize(img, self.target_size, interpolation=Image.BILINEAR)
        
        if mask is not None:
            mask_resized = TF.resize(mask, self.target_size, interpolation=Image.NEAREST)
            return img_resized, mask_resized
        
        return img_resized
    
    def normalize_intensity(self, img: torch.Tensor) -> torch.Tensor:
        """Normalize image intensities"""
        # Convert to tensor if not already
        if isinstance(img, Image.Image):
            img = TF.to_tensor(img)
        
        # Normalize using ImageNet statistics
        img = TF.normalize(img, mean=self.norm_mean, std=self.norm_std)
        
        return img
    
    def synchronized_augment(
        self, 
        img: Image.Image, 
        mask: Image.Image,
        apply_aug: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply synchronized augmentation to image and mask
        Ensures geometric transforms are applied identically to both
        """
        if apply_aug and random.random() > 0.5:
            # Random horizontal flip
            if random.random() > 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            
            # Random rotation (-15 to +15 degrees)
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                img = TF.rotate(img, angle, interpolation=Image.BILINEAR)
                mask = TF.rotate(mask, angle, interpolation=Image.NEAREST)
            
            # Random affine transformation
            if random.random() > 0.5:
                angle = random.uniform(-10, 10)
                translate = (random.randint(-10, 10), random.randint(-10, 10))
                scale = random.uniform(0.9, 1.1)
                shear = random.uniform(-5, 5)
                
                img = TF.affine(img, angle=angle, translate=translate, 
                               scale=scale, shear=shear, interpolation=Image.BILINEAR)
                mask = TF.affine(mask, angle=angle, translate=translate,
                                scale=scale, shear=shear, interpolation=Image.NEAREST)
        
        # Resize both
        img, mask = self.resize(img, mask)
        
        # Convert to tensor
        img_tensor = TF.to_tensor(img)
        mask_tensor = TF.to_tensor(mask)
        
        # Normalize image only (not mask)
        img_tensor = TF.normalize(img_tensor, mean=self.norm_mean, std=self.norm_std)
        
        return img_tensor, mask_tensor
    
    def to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """Convert numpy array to tensor"""
        if isinstance(img, np.ndarray):
            # If grayscale, add channel dimension
            if img.ndim == 2:
                img = np.expand_dims(img, axis=0)
            # If HWC, convert to CHW
            elif img.ndim == 3 and img.shape[2] in [1, 3]:
                img = np.transpose(img, (2, 0, 1))
            
            return torch.from_numpy(img).float()
        
        return img
    
    def denormalize(self, img_tensor: torch.Tensor) -> np.ndarray:
        """
        Denormalize image tensor for visualization
        """
        # Clone to avoid modifying original
        img = img_tensor.clone()
        
        # Denormalize
        for c in range(3):
            img[c] = img[c] * self.norm_std[c] + self.norm_mean[c]
        
        # Clip to [0, 1]
        img = torch.clamp(img, 0, 1)
        
        # Convert to numpy
        img_np = img.permute(1, 2, 0).cpu().numpy()
        
        return img_np


class TrainTransform:
    """Training data transforms with augmentation"""
    
    def __init__(self, preprocessor: Preprocessor):
        self.preprocessor = preprocessor
    
    def __call__(self, img: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.preprocessor.synchronized_augment(img, mask, apply_aug=True)


class ValTestTransform:
    """Validation/Test transforms without augmentation"""
    
    def __init__(self, preprocessor: Preprocessor):
        self.preprocessor = preprocessor
    
    def __call__(self, img: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.preprocessor.synchronized_augment(img, mask, apply_aug=False)
