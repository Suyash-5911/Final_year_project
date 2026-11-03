"""
Stability Evaluation through Perturbation Analysis
Measures explanation robustness to input perturbations
"""
import numpy as np
import torch
import cv2
from typing import List, Dict, Tuple
from skimage.metrics import structural_similarity as ssim
from scipy.stats import spearmanr


class Perturbation:
    """
    Generates perturbed versions of input images
    """
    
    def __init__(
        self,
        noise_stds: List[float] = [0.01, 0.03, 0.05],
        blur_radii: List[float] = [1.0, 1.5, 2.0],
        max_shift_px: int = 3,
        brightness_range: Tuple[float, float] = (0.8, 1.2)
    ):
        self.noise_stds = noise_stds
        self.blur_radii = blur_radii
        self.max_shift_px = max_shift_px
        self.brightness_range = brightness_range
    
    def add_gaussian_noise(
        self, 
        img: torch.Tensor, 
        std: float
    ) -> torch.Tensor:
        """
        Add Gaussian noise to image
        
        Args:
            img: Image tensor [C, H, W]
            std: Standard deviation of noise
        
        Returns:
            noisy_img: Perturbed image
        """
        noise = torch.randn_like(img) * std
        noisy_img = img + noise
        return torch.clamp(noisy_img, 0, 1)
    
    def apply_gaussian_blur(
        self,
        img: torch.Tensor,
        radius: float
    ) -> torch.Tensor:
        """
        Apply Gaussian blur
        
        Args:
            img: Image tensor [C, H, W]
            radius: Blur kernel radius
        
        Returns:
            blurred_img: Blurred image
        """
        # Convert to numpy
        img_np = img.cpu().numpy()
        
        # Apply blur channel-wise
        blurred = np.zeros_like(img_np)
        kernel_size = int(2 * radius) * 2 + 1
        
        for c in range(img_np.shape[0]):
            blurred[c] = cv2.GaussianBlur(
                img_np[c], 
                (kernel_size, kernel_size), 
                radius
            )
        
        return torch.from_numpy(blurred).to(img.device)
    
    def alter_brightness_contrast(
        self,
        img: torch.Tensor,
        brightness_factor: float = 1.0,
        contrast_factor: float = 1.0
    ) -> torch.Tensor:
        """
        Alter brightness and contrast
        
        Args:
            img: Image tensor [C, H, W]
            brightness_factor: Multiplier for brightness
            contrast_factor: Multiplier for contrast
        
        Returns:
            adjusted_img: Adjusted image
        """
        # Adjust brightness
        img_adjusted = img * brightness_factor
        
        # Adjust contrast (around mean)
        mean = img_adjusted.mean()
        img_adjusted = (img_adjusted - mean) * contrast_factor + mean
        
        return torch.clamp(img_adjusted, 0, 1)
    
    def spatial_shift(
        self,
        img: torch.Tensor,
        shift_x: int,
        shift_y: int
    ) -> torch.Tensor:
        """
        Apply spatial shift (translation)
        
        Args:
            img: Image tensor [C, H, W]
            shift_x: Horizontal shift in pixels
            shift_y: Vertical shift in pixels
        
        Returns:
            shifted_img: Shifted image
        """
        # Use affine transformation
        _, h, w = img.shape
        
        # Create transformation matrix
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        
        # Apply to each channel
        img_np = img.cpu().numpy()
        shifted = np.zeros_like(img_np)
        
        for c in range(img_np.shape[0]):
            shifted[c] = cv2.warpAffine(img_np[c], M, (w, h))
        
        return torch.from_numpy(shifted).to(img.device)
    
    def generate_perturbation_suite(
        self,
        img: torch.Tensor,
        num_perturbations: int = 5
    ) -> List[torch.Tensor]:
        """
        Generate a suite of perturbed images
        
        Args:
            img: Original image tensor [C, H, W]
            num_perturbations: Number of perturbations to generate
        
        Returns:
            perturbed_images: List of perturbed image tensors
        """
        perturbed = []
        
        # Add noise perturbations
        for std in self.noise_stds[:num_perturbations]:
            perturbed.append(self.add_gaussian_noise(img, std))
        
        # Add blur perturbations
        if len(perturbed) < num_perturbations:
            for radius in self.blur_radii[:num_perturbations - len(perturbed)]:
                perturbed.append(self.apply_gaussian_blur(img, radius))
        
        # Add brightness perturbations
        if len(perturbed) < num_perturbations:
            for i in range(num_perturbations - len(perturbed)):
                factor = np.random.uniform(*self.brightness_range)
                perturbed.append(self.alter_brightness_contrast(img, factor, 1.0))
        
        return perturbed[:num_perturbations]


class StabilityEvaluator:
    """
    Evaluates explanation stability under input perturbations
    """
    
    def __init__(
        self,
        num_perturbations: int = 5,
        ssim_window_size: int = 7
    ):
        self.num_perturbations = num_perturbations
        self.ssim_window_size = ssim_window_size
        self.orig_heatmap = None
        self.perturbed_heatmaps = []
    
    def compute_pairwise_ssim(
        self,
        h1: np.ndarray,
        h2: np.ndarray
    ) -> float:
        """
        Compute SSIM between two heatmaps
        
        Args:
            h1, h2: Heatmaps [H, W] in range [0, 1]
        
        Returns:
            ssim_score: SSIM value in [-1, 1], typically [0, 1]
        """
        # Ensure same size
        if h1.shape != h2.shape:
            h2 = cv2.resize(h2, (h1.shape[1], h1.shape[0]))
        
        # Compute SSIM
        ssim_score = ssim(
            h1, h2,
            data_range=1.0,
            win_size=self.ssim_window_size
        )
        
        return float(ssim_score)
    
    def compute_spearman_rank(
        self,
        h1: np.ndarray,
        h2: np.ndarray
    ) -> float:
        """
        Compute Spearman rank correlation between heatmaps
        
        Args:
            h1, h2: Heatmaps [H, W]
        
        Returns:
            rho: Spearman correlation coefficient in [-1, 1]
        """
        # Flatten heatmaps
        h1_flat = h1.flatten()
        h2_flat = h2.flatten()
        
        # Ensure same size
        if len(h1_flat) != len(h2_flat):
            min_len = min(len(h1_flat), len(h2_flat))
            h1_flat = h1_flat[:min_len]
            h2_flat = h2_flat[:min_len]
        
        # Compute Spearman correlation
        rho, _ = spearmanr(h1_flat, h2_flat)
        
        return float(rho)
    
    def evaluate_stability(
        self,
        original_heatmap: np.ndarray,
        perturbed_heatmaps: List[np.ndarray]
    ) -> Dict[str, float]:
        """
        Evaluate stability of explanations
        
        Args:
            original_heatmap: Original Grad-CAM [H, W]
            perturbed_heatmaps: List of perturbed Grad-CAMs
        
        Returns:
            stability_metrics: Dictionary of stability scores
        """
        self.orig_heatmap = original_heatmap
        self.perturbed_heatmaps = perturbed_heatmaps
        
        ssim_scores = []
        spearman_scores = []
        
        # Compute metrics for each perturbed heatmap
        for h_perturbed in perturbed_heatmaps:
            ssim_score = self.compute_pairwise_ssim(original_heatmap, h_perturbed)
            spearman_score = self.compute_spearman_rank(original_heatmap, h_perturbed)
            
            ssim_scores.append(ssim_score)
            spearman_scores.append(spearman_score)
        
        # Aggregate
        metrics = {
            'mean_ssim': np.mean(ssim_scores),
            'std_ssim': np.std(ssim_scores),
            'min_ssim': np.min(ssim_scores),
            'mean_spearman': np.mean(spearman_scores),
            'std_spearman': np.std(spearman_scores),
            'min_spearman': np.min(spearman_scores),
        }
        
        return metrics
    
    def aggregate_stability_metrics(self, metrics: Dict[str, float]) -> float:
        """
        Compute composite stability score
        """
        # Weight mean SSIM and mean Spearman equally
        stability_score = 0.5 * metrics['mean_ssim'] + 0.5 * metrics['mean_spearman']
        
        return stability_score
