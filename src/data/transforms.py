"""
Data augmentation and transformation pipelines.

Provides separate transforms for training (with augmentation) and 
validation/testing (without augmentation).
"""

from typing import Union, Tuple, Optional
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image
import random
import numpy as np

from .preprocessor import Preprocessor


class TrainTransform:
    """
    Training data augmentation pipeline.
    
    Applies random augmentations for robust model training:
    - Random horizontal flip
    - Random rotation
    - Random brightness/contrast adjustment
    - Color jittering
    - Optional Gaussian blur
    
    Then applies standard preprocessing (resize, normalize).
    
    Args:
        preprocessor: Preprocessor instance for final preprocessing
        hflip_prob: Probability of horizontal flip
        rotation_degrees: Range of random rotation in degrees
        color_jitter: Whether to apply color jittering
        brightness: Brightness jitter factor
        contrast: Contrast jitter factor
        saturation: Saturation jitter factor
        blur_prob: Probability of applying Gaussian blur
        blur_kernel_size: Kernel size for Gaussian blur
    """
    
    def __init__(
        self,
        preprocessor: Preprocessor,
        hflip_prob: float = 0.5,
        rotation_degrees: int = 15,
        color_jitter: bool = True,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.1,
        blur_prob: float = 0.1,
        blur_kernel_size: int = 5,
    ):
        self.preprocessor = preprocessor
        self.hflip_prob = hflip_prob
        self.rotation_degrees = rotation_degrees
        self.blur_prob = blur_prob
        self.blur_kernel_size = blur_kernel_size
        
        # Color jitter augmentation
        if color_jitter:
            self.color_jitter = transforms.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
            )
        else:
            self.color_jitter = None
        
        # Gaussian blur
        if blur_prob > 0:
            self.gaussian_blur = transforms.GaussianBlur(
                kernel_size=blur_kernel_size,
                sigma=(0.1, 2.0),
            )
        else:
            self.gaussian_blur = None
    
    def __call__(self, image: Image.Image, mask: Optional[Image.Image] = None):
        """
        Apply augmentation and preprocessing.
        
        Args:
            image: PIL Image (RGB)
            mask: Optional PIL Image (grayscale) for segmentation task
        
        Returns:
            If mask is None: preprocessed image tensor
            If mask is provided: (preprocessed image tensor, preprocessed mask tensor)
        """
        # Random horizontal flip
        if random.random() < self.hflip_prob:
            image = TF.hflip(image)
            if mask is not None:
                mask = TF.hflip(mask)
        
        # Random rotation
        if self.rotation_degrees > 0:
            angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
            image = TF.rotate(image, angle, interpolation=Image.BILINEAR)
            if mask is not None:
                mask = TF.rotate(mask, angle, interpolation=Image.NEAREST)
        
        # Color jitter (only on image, not mask)
        if self.color_jitter is not None:
            image = self.color_jitter(image)
        
        # Gaussian blur (only on image, not mask)
        if self.gaussian_blur is not None and random.random() < self.blur_prob:
            image = self.gaussian_blur(image)
        
        # Apply preprocessing
        image_tensor = self.preprocessor.preprocess_image(image)
        
        if mask is not None:
            mask_tensor = self.preprocessor.preprocess_mask(mask)
            return image_tensor, mask_tensor
        else:
            return image_tensor


class ValTestTransform:
    """
    Validation/Test data transformation pipeline.
    
    Applies only preprocessing without augmentation:
    - Resize to target size
    - Convert to tensor
    - Normalize
    
    Args:
        preprocessor: Preprocessor instance for preprocessing
    """
    
    def __init__(self, preprocessor: Preprocessor):
        self.preprocessor = preprocessor
    
    def __call__(self, image: Image.Image, mask: Optional[Image.Image] = None):
        """
        Apply preprocessing without augmentation.
        
        Args:
            image: PIL Image (RGB)
            mask: Optional PIL Image (grayscale) for segmentation task
        
        Returns:
            If mask is None: preprocessed image tensor
            If mask is provided: (preprocessed image tensor, preprocessed mask tensor)
        """
        # Apply preprocessing
        image_tensor = self.preprocessor.preprocess_image(image)
        
        if mask is not None:
            mask_tensor = self.preprocessor.preprocess_mask(mask)
            return image_tensor, mask_tensor
        else:
            return image_tensor


class TestTimeAugmentation:
    """
    Test-time augmentation (TTA) for improved inference.
    
    Generates multiple augmented versions of input, runs inference on each,
    and aggregates predictions for more robust results.
    
    Args:
        preprocessor: Preprocessor instance
        num_augmentations: Number of augmented versions (including original)
        hflip: Whether to include horizontal flip
        rotation_angles: List of rotation angles to apply
    """
    
    def __init__(
        self,
        preprocessor: Preprocessor,
        num_augmentations: int = 5,
        hflip: bool = True,
        rotation_angles: list = None,
    ):
        self.preprocessor = preprocessor
        self.num_augmentations = num_augmentations
        self.hflip = hflip
        self.rotation_angles = rotation_angles or [0, -5, 5, -10, 10]
    
    def __call__(self, image: Image.Image) -> list:
        """
        Generate augmented versions of image.
        
        Args:
            image: PIL Image (RGB)
        
        Returns:
            List of augmented image tensors
        """
        augmented = []
        
        # Original image
        augmented.append(self.preprocessor.preprocess_image(image))
        
        # Horizontal flip
        if self.hflip and len(augmented) < self.num_augmentations:
            flipped = TF.hflip(image)
            augmented.append(self.preprocessor.preprocess_image(flipped))
        
        # Rotations
        for angle in self.rotation_angles:
            if len(augmented) >= self.num_augmentations:
                break
            rotated = TF.rotate(image, angle, interpolation=Image.BILINEAR)
            augmented.append(self.preprocessor.preprocess_image(rotated))
        
        return augmented[:self.num_augmentations]


class MixUp:
    """
    MixUp data augmentation.
    
    Creates synthetic training examples by mixing pairs of images and labels.
    Reference: https://arxiv.org/abs/1710.09412
    
    Args:
        alpha: MixUp interpolation strength (higher = more mixing)
    """
    
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
    
    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Apply MixUp to a batch.
        
        Args:
            images: Batch of images (B, C, H, W)
            labels: Batch of labels (B,)
        
        Returns:
            Mixed images, original labels, mixed labels, lambda value
        """
        batch_size = images.size(0)
        
        # Sample mixing coefficient
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        # Random permutation of batch
        index = torch.randperm(batch_size, device=images.device)
        
        # Mix images
        mixed_images = lam * images + (1 - lam) * images[index]
        
        # Return both label sets for mixed loss computation
        labels_a = labels
        labels_b = labels[index]
        
        return mixed_images, labels_a, labels_b, lam


class CutMix:
    """
    CutMix data augmentation.
    
    Cuts and pastes patches between training images and mixes labels accordingly.
    Reference: https://arxiv.org/abs/1905.04899
    
    Args:
        alpha: CutMix interpolation strength
    """
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
    
    def _rand_bbox(
        self,
        size: Tuple[int, int, int, int],
        lam: float,
    ) -> Tuple[int, int, int, int]:
        """Generate random bounding box."""
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # Uniform sampling
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2
    
    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply CutMix to a batch.
        
        Args:
            images: Batch of images (B, C, H, W)
            labels: Batch of labels (B,)
        
        Returns:
            Mixed images, original labels, mixed labels, lambda value
        """
        batch_size = images.size(0)
        
        # Sample mixing coefficient
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        # Random permutation
        index = torch.randperm(batch_size, device=images.device)
        
        # Generate bounding box
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(images.size(), lam)
        
        # Apply CutMix
        images[:, :, bbx1:bbx2, bby1:bby2] = images[index, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda based on actual box area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
        
        labels_a = labels
        labels_b = labels[index]
        
        return images, labels_a, labels_b, lam


def get_train_transform(
    image_size: int = 224,
    augmentation_level: str = 'medium',
) -> TrainTransform:
    """
    Create training transform with predefined augmentation levels.
    
    Args:
        image_size: Target image size
        augmentation_level: 'light', 'medium', or 'heavy'
    
    Returns:
        TrainTransform instance
    """
    preprocessor = Preprocessor(image_size=image_size)
    
    if augmentation_level == 'light':
        return TrainTransform(
            preprocessor,
            hflip_prob=0.3,
            rotation_degrees=10,
            color_jitter=False,
            blur_prob=0.0,
        )
    elif augmentation_level == 'medium':
        return TrainTransform(
            preprocessor,
            hflip_prob=0.5,
            rotation_degrees=15,
            color_jitter=True,
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
            blur_prob=0.1,
        )
    elif augmentation_level == 'heavy':
        return TrainTransform(
            preprocessor,
            hflip_prob=0.7,
            rotation_degrees=20,
            color_jitter=True,
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            blur_prob=0.2,
        )
    else:
        raise ValueError(f"Invalid augmentation level: {augmentation_level}")


def get_val_test_transform(image_size: int = 224) -> ValTestTransform:
    """
    Create validation/test transform.
    
    Args:
        image_size: Target image size
    
    Returns:
        ValTestTransform instance
    """
    preprocessor = Preprocessor(image_size=image_size)
    return ValTestTransform(preprocessor)
