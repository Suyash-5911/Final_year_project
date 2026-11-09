"""
Image preprocessing utilities for brain tumor MRI images.

Provides resizing, normalization, and conversion utilities compatible
with PyTorch and PIL images.
"""

from typing import Union, Tuple
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image
import numpy as np


class Preprocessor:
    """
    Image preprocessing pipeline for brain tumor MRI images.
    
    Handles:
    - Resizing to target dimensions
    - Normalization using ImageNet statistics (for transfer learning)
    - Tensor conversion
    - Mask preprocessing
    
    Args:
        image_size: Target size for images (height, width) or single int for square
        mean: Mean values for normalization (RGB channels)
        std: Standard deviation for normalization (RGB channels)
        use_imagenet_stats: Whether to use ImageNet statistics (recommended for pretrained models)
    """
    
    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]] = 224,
        mean: Tuple[float, float, float] = None,
        std: Tuple[float, float, float] = None,
        use_imagenet_stats: bool = True,
    ):
        # Handle image size
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = image_size
        
        # Normalization statistics
        if use_imagenet_stats:
            # ImageNet statistics (recommended for transfer learning)
            self.mean = mean or (0.485, 0.456, 0.406)
            self.std = std or (0.229, 0.224, 0.225)
        else:
            # Custom statistics (if provided)
            self.mean = mean or (0.5, 0.5, 0.5)
            self.std = std or (0.5, 0.5, 0.5)
        
        # Create base transforms
        self.resize = transforms.Resize(self.image_size, interpolation=Image.BILINEAR)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess a single PIL image for model input.
        
        Args:
            image: PIL Image in RGB format
        
        Returns:
            Preprocessed tensor of shape (3, H, W)
        """
        # Resize
        image = self.resize(image)
        
        # Convert to tensor [0, 1]
        tensor = self.to_tensor(image)
        
        # Normalize
        tensor = self.normalize(tensor)
        
        return tensor
    
    def preprocess_mask(self, mask: Image.Image, binarize: bool = True) -> torch.Tensor:
        """
        Preprocess a segmentation mask.
        
        Args:
            mask: PIL Image in grayscale (L mode)
            binarize: Whether to binarize mask (threshold at 0.5)
        
        Returns:
            Preprocessed mask tensor of shape (1, H, W)
        """
        # Resize mask (use nearest neighbor to preserve labels)
        mask = TF.resize(mask, self.image_size, interpolation=Image.NEAREST)
        
        # Convert to tensor
        mask_tensor = self.to_tensor(mask)
        
        # Binarize if requested
        if binarize:
            mask_tensor = (mask_tensor > 0.5).float()
        
        return mask_tensor
    
    def preprocess_batch(self, images: list) -> torch.Tensor:
        """
        Preprocess a batch of PIL images.
        
        Args:
            images: List of PIL Images
        
        Returns:
            Batched tensor of shape (B, 3, H, W)
        """
        tensors = [self.preprocess_image(img) for img in images]
        return torch.stack(tensors)
    
    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Denormalize a tensor back to [0, 1] range.
        
        Useful for visualization after normalization.
        
        Args:
            tensor: Normalized tensor of shape (3, H, W) or (B, 3, H, W)
        
        Returns:
            Denormalized tensor in [0, 1] range
        """
        mean = torch.tensor(self.mean).view(-1, 1, 1)
        std = torch.tensor(self.std).view(-1, 1, 1)
        
        # Handle batch dimension
        if tensor.dim() == 4:
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        
        # Denormalize: x_original = x_normalized * std + mean
        denormalized = tensor * std + mean
        
        # Clamp to [0, 1]
        denormalized = torch.clamp(denormalized, 0, 1)
        
        return denormalized
    
    def tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """
        Convert a tensor to PIL Image.
        
        Args:
            tensor: Tensor of shape (3, H, W), normalized or denormalized
        
        Returns:
            PIL Image in RGB mode
        """
        # Denormalize if needed
        if tensor.min() < 0 or tensor.max() > 1:
            tensor = self.denormalize(tensor)
        
        # Convert to numpy
        array = tensor.cpu().numpy()
        array = np.transpose(array, (1, 2, 0))  # (H, W, 3)
        array = (array * 255).astype(np.uint8)
        
        return Image.fromarray(array, mode='RGB')
    
    def mask_to_pil(self, mask_tensor: torch.Tensor) -> Image.Image:
        """
        Convert a mask tensor to PIL Image.
        
        Args:
            mask_tensor: Tensor of shape (1, H, W) or (H, W)
        
        Returns:
            PIL Image in grayscale (L mode)
        """
        # Remove channel dimension if present
        if mask_tensor.dim() == 3:
            mask_tensor = mask_tensor.squeeze(0)
        
        # Convert to numpy
        array = mask_tensor.cpu().numpy()
        array = (array * 255).astype(np.uint8)
        
        return Image.fromarray(array, mode='L')
    
    def get_transform_params(self) -> dict:
        """
        Get preprocessing parameters.
        
        Useful for saving/loading configurations.
        
        Returns:
            Dictionary of parameters
        """
        return {
            'image_size': self.image_size,
            'mean': self.mean,
            'std': self.std,
        }


class InverseNormalize:
    """
    Inverse normalization transform.
    
    Converts normalized tensors back to [0, 1] range for visualization.
    
    Args:
        mean: Mean values used in normalization
        std: Standard deviation values used in normalization
    """
    
    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply inverse normalization."""
        # Move mean and std to same device as tensor
        mean = self.mean.to(tensor.device)
        std = self.std.to(tensor.device)
        
        # Handle batch dimension
        if tensor.dim() == 4:
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        
        # Denormalize
        denormalized = tensor * std + mean
        
        # Clamp to [0, 1]
        denormalized = torch.clamp(denormalized, 0, 1)
        
        return denormalized


def calculate_dataset_statistics(dataloader, num_batches: int = None) -> Tuple[Tuple, Tuple]:
    """
    Calculate mean and std of a dataset.
    
    Useful for computing custom normalization statistics instead of using ImageNet.
    
    Args:
        dataloader: PyTorch DataLoader
        num_batches: Number of batches to use (None = all batches)
    
    Returns:
        Tuple of (mean, std) for each channel
    """
    mean = 0.0
    std = 0.0
    total_images = 0
    
    for batch_idx, batch in enumerate(dataloader):
        if num_batches is not None and batch_idx >= num_batches:
            break
        
        # Extract images (handle both classification and segmentation)
        if isinstance(batch, (list, tuple)):
            images = batch[0]
        else:
            images = batch
        
        # Reshape to (B*C, H*W)
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        
        # Update running mean and std
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples
    
    mean /= total_images
    std /= total_images
    
    return tuple(mean.tolist()), tuple(std.tolist())


def resize_with_padding(
    image: Image.Image,
    target_size: Tuple[int, int],
    fill_color: Tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """
    Resize image while maintaining aspect ratio with padding.
    
    Alternative to center cropping - preserves entire image.
    
    Args:
        image: PIL Image
        target_size: Target (height, width)
        fill_color: RGB color for padding
    
    Returns:
        Resized and padded image
    """
    # Calculate scaling factor
    width, height = image.size
    target_height, target_width = target_size
    
    scale = min(target_width / width, target_height / height)
    
    # Resize maintaining aspect ratio
    new_width = int(width * scale)
    new_height = int(height * scale)
    image = image.resize((new_width, new_height), Image.BILINEAR)
    
    # Create new image with padding
    padded_image = Image.new('RGB', (target_width, target_height), fill_color)
    
    # Paste resized image in center
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    padded_image.paste(image, (x_offset, y_offset))
    
    return padded_image
