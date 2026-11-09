"""
Brain Tumor Classifier using ResNet backbone.

Supports multiple ResNet architectures (ResNet18, ResNet34, ResNet50, ResNet101, ResNet152)
with pretrained ImageNet weights for transfer learning.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Literal, Tuple, Optional
import torch.nn.functional as F


class TumourClassifier(nn.Module):
    """
    CNN classifier for brain tumor classification.
    
    Uses ResNet backbone pretrained on ImageNet with custom classification head.
    Designed for Grad-CAM explainability with feature extraction support.
    
    Args:
        backbone: ResNet architecture ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
        num_classes: Number of tumor classes (default: 3 for glioma, meningioma, pituitary)
        pretrained: Whether to use ImageNet pretrained weights
        freeze_backbone: Whether to freeze backbone layers initially
        dropout: Dropout rate for classifier head (0.0 to disable)
    """
    
    def __init__(
        self,
        backbone: Literal['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'] = 'resnet50',
        num_classes: int = 3,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.5,
    ):
        super(TumourClassifier, self).__init__()
        
        self.backbone_name = backbone
        self.num_classes = num_classes
        self.pretrained = pretrained
        
        # Load pretrained ResNet
        if backbone == 'resnet18':
            self.backbone = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
            self.feature_dim = 512
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
            self.feature_dim = 512
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
            self.feature_dim = 2048
        elif backbone == 'resnet101':
            self.backbone = models.resnet101(weights='IMAGENET1K_V2' if pretrained else None)
            self.feature_dim = 2048
        elif backbone == 'resnet152':
            self.backbone = models.resnet152(weights='IMAGENET1K_V2' if pretrained else None)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Invalid backbone: {backbone}")
        
        # Remove the original fully connected layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Freeze backbone if requested
        if freeze_backbone:
            self.freeze_backbone_layers()
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(512, num_classes),
        )
        
        # Initialize classifier weights
        self._initialize_weights()
        
        # Store intermediate features for Grad-CAM
        self.features = None
        self.gradients = None
        
        # Register hooks for Grad-CAM
        self._register_hooks()
    
    def _initialize_weights(self):
        """Initialize classifier head weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _register_hooks(self):
        """Register forward and backward hooks for Grad-CAM."""
        def forward_hook(module, input, output):
            self.features = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Hook on the last convolutional layer (before adaptive pooling)
        # For ResNet, this is the last layer in the backbone
        target_layer = self.backbone[-1]  # This gets the last Sequential block
        if hasattr(target_layer, 'register_forward_hook'):
            target_layer.register_forward_hook(forward_hook)
            target_layer.register_full_backward_hook(backward_hook)
    
    def freeze_backbone_layers(self):
        """Freeze all backbone layers."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print(f"Froze {self.backbone_name} backbone layers")
    
    def unfreeze_backbone_layers(self):
        """Unfreeze all backbone layers for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print(f"Unfroze {self.backbone_name} backbone layers")
    
    def unfreeze_last_n_layers(self, n: int):
        """
        Unfreeze the last n layers of the backbone for progressive fine-tuning.
        
        Args:
            n: Number of layers to unfreeze from the end
        """
        backbone_layers = list(self.backbone.children())
        layers_to_unfreeze = backbone_layers[-n:]
        
        for layer in layers_to_unfreeze:
            for param in layer.parameters():
                param.requires_grad = True
        
        print(f"Unfroze last {n} layers of {self.backbone_name}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
        
        Returns:
            Logits of shape (B, num_classes)
        """
        # Extract features
        features = self.backbone(x)
        
        # Classify
        logits = self.classifier(features)
        
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature representations before classification.
        
        Useful for visualization, clustering, or downstream tasks.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
        
        Returns:
            Features of shape (B, feature_dim)
        """
        features = self.backbone(x)
        features = features.flatten(1)
        return features
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict class probabilities and labels.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
        
        Returns:
            probabilities: Class probabilities (B, num_classes)
            predictions: Predicted class indices (B,)
        """
        logits = self.forward(x)
        probabilities = F.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)
        return probabilities, predictions
    
    def get_cam_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get features and gradients for Grad-CAM computation.
        
        Must be called after backward pass.
        
        Returns:
            features: Feature maps from target layer
            gradients: Gradients of target layer
        """
        if self.features is None or self.gradients is None:
            raise RuntimeError(
                "Features and gradients not available. "
                "Make sure to call forward() and backward() first."
            )
        return self.features, self.gradients
    
    def get_model_info(self) -> dict:
        """
        Get model information.
        
        Returns:
            Dictionary with model configuration
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'backbone': self.backbone_name,
            'num_classes': self.num_classes,
            'pretrained': self.pretrained,
            'feature_dim': self.feature_dim,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': total_params - trainable_params,
        }
    
    def print_model_info(self):
        """Print model information in a readable format."""
        info = self.get_model_info()
        
        print("=" * 70)
        print("MODEL INFORMATION")
        print("=" * 70)
        print(f"Architecture:         {info['backbone']}")
        print(f"Number of Classes:    {info['num_classes']}")
        print(f"Pretrained:           {info['pretrained']}")
        print(f"Feature Dimension:    {info['feature_dim']}")
        print(f"-" * 70)
        print(f"Total Parameters:     {info['total_parameters']:,}")
        print(f"Trainable Parameters: {info['trainable_parameters']:,}")
        print(f"Frozen Parameters:    {info['frozen_parameters']:,}")
        print("=" * 70)


class EnsembleClassifier(nn.Module):
    """
    Ensemble of multiple TumourClassifier models.
    
    Combines predictions from multiple models for improved accuracy and robustness.
    
    Args:
        models: List of trained TumourClassifier instances
        weights: Optional weights for each model (defaults to equal weighting)
        aggregation: How to combine predictions ('mean', 'max', or 'weighted')
    """
    
    def __init__(
        self,
        models: list,
        weights: Optional[torch.Tensor] = None,
        aggregation: Literal['mean', 'max', 'weighted'] = 'mean',
    ):
        super(EnsembleClassifier, self).__init__()
        
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        self.aggregation = aggregation
        
        # Validate that all models have same num_classes
        num_classes_list = [model.num_classes for model in models]
        if len(set(num_classes_list)) > 1:
            raise ValueError("All models must have the same number of classes")
        self.num_classes = num_classes_list[0]
        
        # Set weights
        if weights is None:
            self.weights = torch.ones(self.num_models) / self.num_models
        else:
            if len(weights) != self.num_models:
                raise ValueError(f"Expected {self.num_models} weights, got {len(weights)}")
            self.weights = weights / weights.sum()  # Normalize
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ensemble.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
        
        Returns:
            Aggregated logits of shape (B, num_classes)
        """
        # Get predictions from all models
        logits_list = [model(x) for model in self.models]
        
        # Stack logits
        logits_stack = torch.stack(logits_list)  # (num_models, B, num_classes)
        
        # Aggregate
        if self.aggregation == 'mean':
            logits = logits_stack.mean(dim=0)
        elif self.aggregation == 'max':
            logits = logits_stack.max(dim=0)[0]
        elif self.aggregation == 'weighted':
            weights = self.weights.view(-1, 1, 1).to(x.device)
            logits = (logits_stack * weights).sum(dim=0)
        else:
            raise ValueError(f"Invalid aggregation method: {self.aggregation}")
        
        return logits
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict class probabilities and labels.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
        
        Returns:
            probabilities: Class probabilities (B, num_classes)
            predictions: Predicted class indices (B,)
        """
        logits = self.forward(x)
        probabilities = F.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)
        return probabilities, predictions


def create_model(
    backbone: str = 'resnet50',
    num_classes: int = 3,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout: float = 0.5,
) -> TumourClassifier:
    """
    Factory function to create a TumourClassifier.
    
    Args:
        backbone: ResNet architecture
        num_classes: Number of classes
        pretrained: Use pretrained weights
        freeze_backbone: Freeze backbone layers
        dropout: Dropout rate
    
    Returns:
        TumourClassifier instance
    """
    model = TumourClassifier(
        backbone=backbone,
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        dropout=dropout,
    )
    
    model.print_model_info()
    
    return model


def load_model(checkpoint_path: str, device: str = 'cuda') -> TumourClassifier:
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        Loaded TumourClassifier instance
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model configuration
    config = checkpoint.get('config', {})
    
    # Create model
    model = TumourClassifier(
        backbone=config.get('backbone', 'resnet50'),
        num_classes=config.get('num_classes', 3),
        pretrained=False,  # We're loading weights
        dropout=config.get('dropout', 0.5),
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Best Val Acc: {checkpoint.get('best_val_acc', 'N/A'):.2f}%")
    
    return model
