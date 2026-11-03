"""
Localization Evaluation Metrics
Measures spatial agreement between Grad-CAM heatmaps and expert masks
"""
import numpy as np
import cv2
from typing import Dict, Tuple
from skimage.filters import threshold_otsu


class LocalizationEvaluator:
    """
    Evaluates localization quality of explanation heatmaps
    """
    
    def __init__(
        self,
        binarize_threshold: float = 0.5,
        binarize_strategy: str = "otsu"
    ):
        self.binarize_threshold = binarize_threshold
        self.binarize_strategy = binarize_strategy
        self.score_cache = {}
    
    def binarize_heatmap(self, cam: np.ndarray) -> np.ndarray:
        """
        Binarize heatmap using threshold or Otsu's method
        
        Args:
            cam: Heatmap [H, W] in range [0, 1]
        
        Returns:
            binary_map: Binary mask [H, W]
        """
        if self.binarize_strategy == "otsu":
            # Otsu's automatic thresholding
            threshold = threshold_otsu(cam)
            binary_map = (cam >= threshold).astype(np.uint8)
        elif self.binarize_strategy == "fixed":
            # Fixed threshold
            binary_map = (cam >= self.binarize_threshold).astype(np.uint8)
        elif self.binarize_strategy == "adaptive":
            # Adaptive threshold
            cam_uint8 = (cam * 255).astype(np.uint8)
            binary_map = cv2.adaptiveThreshold(
                cam_uint8, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
        else:
            raise ValueError(f"Unknown binarization strategy: {self.binarize_strategy}")
        
        return binary_map
    
    def compute_iou(self, cam_bin: np.ndarray, mask: np.ndarray) -> float:
        """
        Compute Intersection over Union (IoU)
        
        Args:
            cam_bin: Binarized heatmap [H, W]
            mask: Ground truth mask [H, W]
        
        Returns:
            iou: IoU score in [0, 1]
        """
        # Ensure binary
        cam_bin = (cam_bin > 0).astype(np.uint8)
        mask = (mask > 0).astype(np.uint8)
        
        # Compute intersection and union
        intersection = np.logical_and(cam_bin, mask).sum()
        union = np.logical_or(cam_bin, mask).sum()
        
        if union == 0:
            return 0.0
        
        iou = intersection / union
        return float(iou)
    
    def compute_dice(self, cam_bin: np.ndarray, mask: np.ndarray) -> float:
        """
        Compute Dice Coefficient (F1 Score)
        
        Args:
            cam_bin: Binarized heatmap [H, W]
            mask: Ground truth mask [H, W]
        
        Returns:
            dice: Dice coefficient in [0, 1]
        """
        # Ensure binary
        cam_bin = (cam_bin > 0).astype(np.uint8)
        mask = (mask > 0).astype(np.uint8)
        
        # Compute intersection
        intersection = np.logical_and(cam_bin, mask).sum()
        
        # Compute dice
        cam_sum = cam_bin.sum()
        mask_sum = mask.sum()
        
        if (cam_sum + mask_sum) == 0:
            return 0.0
        
        dice = 2 * intersection / (cam_sum + mask_sum)
        return float(dice)
    
    def pointing_game(self, cam: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """
        Pointing Game metric: Does the maximum activation point fall inside the mask?
        
        Args:
            cam: Heatmap [H, W] in range [0, 1]
            mask: Ground truth mask [H, W]
        
        Returns:
            dict with 'hit' (1.0 or 0.0) and 'max_point' coordinates
        """
        # Find maximum activation point
        max_idx = np.unravel_index(np.argmax(cam), cam.shape)
        
        # Check if it falls inside mask
        hit = 1.0 if mask[max_idx] > 0 else 0.0
        
        return {
            'hit': hit,
            'max_point': max_idx,
            'max_value': cam[max_idx]
        }
    
    def energy_ratio(self, cam: np.ndarray, mask: np.ndarray) -> float:
        """
        Energy Ratio: Proportion of total heatmap energy inside the mask
        
        Args:
            cam: Heatmap [H, W] in range [0, 1]
            mask: Ground truth mask [H, W]
        
        Returns:
            ratio: Energy ratio in [0, 1]
        """
        # Ensure mask is binary
        mask_bin = (mask > 0).astype(np.float32)
        
        # Total energy
        total_energy = cam.sum()
        
        if total_energy == 0:
            return 0.0
        
        # Energy inside mask
        energy_inside = (cam * mask_bin).sum()
        
        ratio = energy_inside / total_energy
        return float(ratio)
    
    def evaluate_all(
        self, 
        cam: np.ndarray, 
        mask: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute all localization metrics
        
        Args:
            cam: Grad-CAM heatmap [H, W] normalized to [0, 1]
            mask: Ground truth mask [H, W], binary or grayscale
        
        Returns:
            metrics: Dictionary of all localization scores
        """
        # Ensure mask is binary
        mask_bin = (mask > 0).astype(np.uint8)
        
        # Resize cam to match mask if needed
        if cam.shape != mask.shape:
            cam = cv2.resize(cam, (mask.shape[1], mask.shape[0]))
        
        # Binarize heatmap
        cam_bin = self.binarize_heatmap(cam)
        
        # Compute all metrics
        metrics = {
            'iou': self.compute_iou(cam_bin, mask_bin),
            'dice': self.compute_dice(cam_bin, mask_bin),
            'energy_ratio': self.energy_ratio(cam, mask_bin),
        }
        
        # Pointing game
        pg_result = self.pointing_game(cam, mask_bin)
        metrics['pointing_game'] = pg_result['hit']
        metrics['pg_max_value'] = pg_result['max_value']
        
        # Store in cache
        self.score_cache = metrics
        
        return metrics
    
    def get_localization_score(self, metrics: Dict[str, float]) -> float:
        """
        Compute composite localization score
        Weighted average of IoU, Dice, Pointing Game, and Energy Ratio
        """
        weights = {
            'iou': 0.3,
            'dice': 0.3,
            'pointing_game': 0.2,
            'energy_ratio': 0.2
        }
        
        score = sum(metrics.get(k, 0) * v for k, v in weights.items())
        return score
