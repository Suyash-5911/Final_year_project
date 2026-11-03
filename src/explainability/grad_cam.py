"""
Grad-CAM (Gradient-weighted Class Activation Mapping)
Visual explanation generation for CNN predictions
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional
from PIL import Image


class ExplanationGenerator:
    """
    Generates Grad-CAM explanations for model predictions
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: Optional[torch.nn.Module] = None,
        colormap: str = "jet",
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model
        self.device = device
        self.colormap = colormap
        
        # Auto-detect target layer if not provided
        if target_layer is None:
            self.target_layer = model.get_target_layer()
        else:
            self.target_layer = target_layer
        
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate_cam(
        self, 
        input_tensor: torch.Tensor, 
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_tensor: Input image tensor [1, C, H, W]
            target_class: Target class index (if None, use predicted class)
        
        Returns:
            cam: Grad-CAM heatmap [H, W] normalized to [0, 1]
        """
        self.model.eval()
        input_tensor = input_tensor.to(self.device)
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2), keepdim=True)  # [C, 1, 1]
        
        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=0)  # [H, W]
        
        # ReLU to keep only positive influences
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam.cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam
    
    def extract_gradients(self):
        """Return stored gradients"""
        return self.gradients
    
    def overlay_on_image(
        self, 
        img: np.ndarray, 
        cam: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Overlay Grad-CAM heatmap on original image
        
        Args:
            img: Original image [H, W, 3] in RGB format, range [0, 255]
            cam: Grad-CAM heatmap [H, W] in range [0, 1]
            alpha: Blending factor
        
        Returns:
            overlaid: Blended image [H, W, 3]
        """
        # Resize CAM to match image size
        h, w = img.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        
        # Apply colormap
        heatmap = self.normalize_heatmap(cam_resized)
        if self.colormap == "jet":
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        elif self.colormap == "hot":
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
        else:
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Convert BGR to RGB
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Ensure image is in correct format
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        
        # Blend
        overlaid = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
        
        return overlaid
    
    def normalize_heatmap(self, cam: np.ndarray) -> np.ndarray:
        """
        Normalize heatmap to uint8 range [0, 255]
        """
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        cam = (cam * 255).astype(np.uint8)
        return cam
    
    def save_heatmap(self, cam: np.ndarray, save_path: str):
        """Save heatmap as image"""
        heatmap = self.normalize_heatmap(cam)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        cv2.imwrite(save_path, heatmap)


class GuidedBackprop:
    """
    Guided Backpropagation for visualization (alternative method)
    """
    
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.gradients = None
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to modify ReLU gradients"""
        def relu_backward_hook(module, grad_input, grad_output):
            # Only pass positive gradients
            return (F.relu(grad_input[0]),)
        
        for module in self.model.modules():
            if isinstance(module, torch.nn.ReLU):
                module.register_full_backward_hook(relu_backward_hook)
    
    def generate_gradients(
        self, 
        input_tensor: torch.Tensor, 
        target_class: int
    ) -> np.ndarray:
        """Generate guided backpropagation gradients"""
        input_tensor.requires_grad = True
        
        output = self.model(input_tensor)
        self.model.zero_grad()
        
        class_score = output[0, target_class]
        class_score.backward()
        
        gradients = input_tensor.grad.data.cpu().numpy()[0]
        gradients = np.transpose(gradients, (1, 2, 0))
        
        return gradients
