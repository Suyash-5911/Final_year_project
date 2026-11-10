"""
Visualization utilities for Grad-CAM explanations
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from typing import Optional, List, Tuple
import torch


def create_comparison_visualization(
    original_image: np.ndarray,
    cam: np.ndarray,
    prediction: str,
    confidence: float,
    probabilities: torch.Tensor,
    class_names: List[str],
    mask: Optional[np.ndarray] = None,
    metrics: Optional[dict] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create comprehensive comparison visualization
    
    Args:
        original_image: Original RGB image [H, W, 3]
        cam: Grad-CAM heatmap [H, W] in [0, 1]
        prediction: Predicted class name
        confidence: Prediction confidence
        probabilities: Class probabilities tensor
        class_names: List of class names
        mask: Optional ground truth mask [H, W]
        metrics: Optional localization metrics dict
        save_path: Optional path to save figure
    
    Returns:
        fig: Matplotlib figure
    """
    # Determine layout based on mask availability
    if mask is not None:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
    
    # 1. Original Image
    axes[0].imshow(original_image)
    axes[0].set_title(f'Original Image\nPrediction: {prediction} ({confidence:.2%})', 
                      fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # 2. Heatmap only
    cam_resized = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
    im = axes[1].imshow(cam_resized, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # 3. Overlay
    overlay = overlay_heatmap(original_image, cam, alpha=0.4)
    axes[2].imshow(overlay)
    axes[2].set_title('Grad-CAM Overlay', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    # 4. Class Probabilities
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    probs = probabilities.cpu().numpy() * 100
    bars = axes[3].barh(class_names, probs, color=colors[:len(class_names)])
    axes[3].set_xlabel('Probability (%)', fontsize=11)
    axes[3].set_title('Class Probabilities', fontsize=12, fontweight='bold')
    axes[3].set_xlim(0, 100)
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        axes[3].text(width + 2, bar.get_y() + bar.get_height()/2,
                    f'{width:.1f}%', va='center', fontsize=10)
    
    # 5. Ground truth mask (if available)
    if mask is not None:
        axes[4].imshow(original_image)
        mask_overlay = np.zeros_like(original_image)
        mask_overlay[:, :, 1] = (mask > 0).astype(np.uint8) * 200  # Green channel
        axes[4].imshow(mask_overlay, alpha=0.4)
        axes[4].set_title('Ground Truth Mask', fontsize=12, fontweight='bold')
        axes[4].axis('off')
        
        # 6. Metrics visualization
        if metrics is not None:
            axes[5].axis('off')
            metric_text = "Localization Metrics:\n\n"
            metric_text += f"IoU: {metrics.get('iou', 0):.3f}\n"
            metric_text += f"Dice: {metrics.get('dice', 0):.3f}\n"
            metric_text += f"Pointing Game: {'✓' if metrics.get('pointing_game', 0) > 0 else '✗'}\n"
            metric_text += f"Energy Ratio: {metrics.get('energy_ratio', 0):.3f}\n"
            
            if 'iou' in metrics:
                avg_score = (metrics['iou'] + metrics['dice'] + 
                           metrics['pointing_game'] + metrics['energy_ratio']) / 4
                metric_text += f"\nAverage: {avg_score:.3f}"
            
            axes[5].text(0.1, 0.5, metric_text, fontsize=14,
                        verticalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            axes[5].set_title('Evaluation Metrics', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    return fig


def overlay_heatmap(
    image: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlay heatmap on image
    
    Args:
        image: RGB image [H, W, 3], range [0, 255] or [0, 1]
        cam: Heatmap [H, W], range [0, 1]
        alpha: Blending weight for heatmap
        colormap: OpenCV colormap constant
    
    Returns:
        overlay: Blended image [H, W, 3]
    """
    # Ensure image is uint8
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    
    # Resize CAM to match image
    h, w = image.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    
    # Convert to uint8
    cam_uint8 = (cam_resized * 255).astype(np.uint8)
    
    # Apply colormap
    heatmap = cv2.applyColorMap(cam_uint8, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Blend
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    
    return overlay


def create_grid_visualization(
    images: List[np.ndarray],
    cams: List[np.ndarray],
    predictions: List[str],
    confidences: List[float],
    titles: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    ncols: int = 4
) -> plt.Figure:
    """
    Create grid visualization of multiple samples
    
    Args:
        images: List of RGB images
        cams: List of Grad-CAM heatmaps
        predictions: List of prediction strings
        confidences: List of confidence scores
        titles: Optional list of titles
        save_path: Optional save path
        ncols: Number of columns in grid
    
    Returns:
        fig: Matplotlib figure
    """
    n_samples = len(images)
    nrows = (n_samples + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    axes = np.array(axes).flatten()
    
    for idx in range(n_samples):
        overlay = overlay_heatmap(images[idx], cams[idx])
        axes[idx].imshow(overlay)
        
        title = titles[idx] if titles else f"Sample {idx+1}"
        title += f"\n{predictions[idx]} ({confidences[idx]:.1%})"
        
        axes[idx].set_title(title, fontsize=10, fontweight='bold')
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(n_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def visualize_cam_progression(
    image: np.ndarray,
    cams: List[np.ndarray],
    labels: List[str],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize CAM progression across different thresholds or classes
    
    Args:
        image: Original RGB image
        cams: List of CAM heatmaps
        labels: Labels for each CAM
        save_path: Optional save path
    
    Returns:
        fig: Matplotlib figure
    """
    n_cams = len(cams)
    fig, axes = plt.subplots(1, n_cams + 1, figsize=(4*(n_cams+1), 4))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # CAMs
    for idx, (cam, label) in enumerate(zip(cams, labels), 1):
        overlay = overlay_heatmap(image, cam)
        axes[idx].imshow(overlay)
        axes[idx].set_title(label, fontsize=12, fontweight='bold')
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def save_cam_with_contours(
    image: np.ndarray,
    cam: np.ndarray,
    threshold: float = 0.5,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Save CAM with contour boundaries
    
    Args:
        image: Original RGB image
        cam: Grad-CAM heatmap [H, W]
        threshold: Threshold for contour detection
        save_path: Optional save path
    
    Returns:
        result: Image with contours
    """
    # Ensure proper format
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # Resize CAM
    h, w = image.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    
    # Threshold and find contours
    cam_binary = (cam_resized > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(cam_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on image
    result = image.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
    
    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    
    return result


def plot_localization_metrics_comparison(
    metrics_list: List[dict],
    labels: List[str],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare localization metrics across multiple samples or methods
    
    Args:
        metrics_list: List of metric dictionaries
        labels: Labels for each metric set
        save_path: Optional save path
    
    Returns:
        fig: Matplotlib figure
    """
    metric_names = ['iou', 'dice', 'pointing_game', 'energy_ratio']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(metric_names))
    width = 0.8 / len(metrics_list)
    
    for idx, (metrics, label) in enumerate(zip(metrics_list, labels)):
        values = [metrics.get(m, 0) for m in metric_names]
        offset = (idx - len(metrics_list)/2) * width + width/2
        ax.bar(x + offset, values, width, label=label)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Localization Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['IoU', 'Dice', 'Pointing\nGame', 'Energy\nRatio'])
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
