"""
Brain Tumour Classification Model
CNN-based classifier with ResNet backbone
"""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Optional


class TumourClassifier(nn.Module):
    """
    CNN-based tumour classifier with configurable backbone
    """
    
    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 3,
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        super(TumourClassifier, self).__init__()
        
        self.backbone_name = backbone
        self.num_classes = num_classes
        self.class_names = ["glioma", "meningioma", "pituitary"]
        
        # Load backbone
        if backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_features, num_classes)
        elif backbone == "resnet34":
            self.backbone = models.resnet34(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_features, num_classes)
        elif backbone == "densenet121":
            self.backbone = models.densenet121(pretrained=pretrained)
            num_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Linear(num_features, num_classes)
        elif backbone == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Linear(num_features, num_classes)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Freeze backbone layers if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Unfreeze final layer
            if backbone in ["resnet50", "resnet34"]:
                for param in self.backbone.fc.parameters():
                    param.requires_grad = True
            elif backbone == "densenet121":
                for param in self.backbone.classifier.parameters():
                    param.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.backbone(x)
    
    def predict(self, x: torch.Tensor) -> Tuple[str, float, torch.Tensor]:
        """
        Make prediction with class name and confidence
        Returns: (class_name, confidence, all_probabilities)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted_class = torch.max(probabilities, dim=1)
            
            class_name = self.class_names[predicted_class.item()]
            confidence_value = confidence.item()
            
        return class_name, confidence_value, probabilities[0]
    
    def get_target_layer(self):
        """Get target layer for Grad-CAM"""
        if self.backbone_name in ["resnet50", "resnet34"]:
            return self.backbone.layer4[-1]
        elif self.backbone_name == "densenet121":
            return self.backbone.features[-1]
        elif self.backbone_name == "efficientnet_b0":
            return self.backbone.features[-1]
        else:
            raise ValueError(f"Target layer not defined for backbone: {self.backbone_name}")
    
    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss"""
        criterion = nn.CrossEntropyLoss()
        return criterion(logits, targets)
    
    def get_feature_maps(self, x: torch.Tensor, layer_name: str = "layer4"):
        """
        Extract feature maps from specified layer
        Used for Grad-CAM computation
        """
        features = {}
        
        def hook_fn(module, input, output):
            features['output'] = output
        
        # Register hook
        if self.backbone_name in ["resnet50", "resnet34"]:
            if layer_name == "layer4":
                handle = self.backbone.layer4.register_forward_hook(hook_fn)
            elif layer_name == "layer3":
                handle = self.backbone.layer3.register_forward_hook(hook_fn)
            else:
                raise ValueError(f"Unknown layer: {layer_name}")
        
        # Forward pass
        _ = self.forward(x)
        
        # Remove hook
        handle.remove()
        
        return features['output']


class ModelTrainer:
    """
    Training manager for TumourClassifier
    """
    
    def __init__(
        self,
        model: TumourClassifier,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 0.001,
        optimizer_type: str = 'adam'
    ):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        
        # Setup optimizer
        if optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=learning_rate
            )
        elif optimizer_type == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=1e-4
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
        
        # Setup learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1,
            patience=5,
            verbose=True
        )
        
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self, dataloader) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in dataloader:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, dataloader) -> Tuple[float, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, path: str, epoch: int, val_loss: float):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['val_loss']
